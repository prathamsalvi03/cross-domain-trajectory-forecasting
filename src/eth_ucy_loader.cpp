#include "eth_ucy_loader.hpp"
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <stdexcept>

namespace traj {

namespace fs = std::filesystem;

std::vector<PedRow> read_ped_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Could not open file: " + path);
    }
    std::vector<PedRow> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;  // skip blank lines
        std::istringstream iss(line);
        PedRow row;
        if (iss >> row.frame >> row.ped_id >> row.x >> row.y) {
            rows.push_back(row);
        }
    }
    return rows;
}

std::pair<torch::Tensor, torch::Tensor> build_scene_tensor(
    const std::vector<PedRow>& raw, int64_t n_pad) {
    // Unique, sorted frames and ped_ids (mirrors np.unique's sorted-ascending
    // behaviour), each mapped to a dense [0, count) slot index.
    std::map<double, int64_t> frame_to_t;
    std::map<double, int64_t> ped_to_slot;
    for (const auto& r : raw) {
        frame_to_t[r.frame] = 0;
        ped_to_slot[r.ped_id] = 0;
    }
    int64_t t_idx = 0;
    for (auto& kv : frame_to_t) kv.second = t_idx++;
    int64_t p_idx = 0;
    for (auto& kv : ped_to_slot) kv.second = p_idx++;

    int64_t T = static_cast<int64_t>(frame_to_t.size());
    int64_t N = static_cast<int64_t>(ped_to_slot.size());

    auto positions = torch::zeros({T, N, 2}, torch::kFloat32);
    auto mask = torch::zeros({T, N}, torch::kFloat32);

    for (const auto& r : raw) {
        int64_t t = frame_to_t.at(r.frame);
        int64_t n = ped_to_slot.at(r.ped_id);
        positions.index_put_({t, n, 0}, static_cast<float>(r.x));
        positions.index_put_({t, n, 1}, static_cast<float>(r.y));
        mask.index_put_({t, n}, 1.0f);
    }

    if (N > n_pad) {
        // Keep the n_pad pedestrians with the most total valid timesteps.
        auto counts = mask.sum(/*dim=*/0);  // (N,)
        auto sorted = torch::argsort(counts, /*dim=*/0, /*descending=*/true);
        auto keep = sorted.index({torch::indexing::Slice(torch::indexing::None, n_pad)});
        positions = positions.index_select(1, keep);
        mask = mask.index_select(1, keep);
    } else if (N < n_pad) {
        auto pad_pos = torch::zeros({T, n_pad - N, 2}, torch::kFloat32);
        auto pad_mask = torch::zeros({T, n_pad - N}, torch::kFloat32);
        positions = torch::cat({positions, pad_pos}, /*dim=*/1);
        mask = torch::cat({mask, pad_mask}, /*dim=*/1);
    }

    return {positions, mask};
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> sliding_windows(
    const torch::Tensor& positions, const torch::Tensor& mask,
    int64_t window_len, int64_t stride, int64_t min_agents) {
    int64_t T = positions.size(0);
    std::vector<std::pair<torch::Tensor, torch::Tensor>> windows;
    for (int64_t start = 0; start + window_len <= T; start += stride) {
        auto pos_w = positions.index({torch::indexing::Slice(start, start + window_len)});
        auto mask_w = mask.index({torch::indexing::Slice(start, start + window_len)});
        auto n_active = (mask_w.sum(/*dim=*/0) > 0).sum().item<int64_t>();
        if (n_active >= min_agents) {
            windows.emplace_back(pos_w.clone(), mask_w.clone());
        }
    }
    return windows;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> load_eth_ucy_windows(
    const std::string& root_dir, const std::string& scene, const std::string& split,
    int64_t window_len, int64_t stride, int64_t n_pad, int64_t min_agents) {
    fs::path dir = fs::path(root_dir) / scene / split;
    if (!fs::exists(dir)) {
        throw std::runtime_error("Directory does not exist: " + dir.string());
    }

    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".txt") files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No .txt files found in: " + dir.string());
    }

    std::vector<std::pair<torch::Tensor, torch::Tensor>> all_windows;
    for (const auto& fpath : files) {
        auto raw = read_ped_file(fpath);
        if (raw.empty()) continue;
        auto [positions, mask] = build_scene_tensor(raw, n_pad);
        auto windows = sliding_windows(positions, mask, window_len, stride, min_agents);
        all_windows.insert(all_windows.end(), windows.begin(), windows.end());
    }
    return all_windows;
}

std::pair<torch::Tensor, torch::Tensor> make_pedestrian_batch(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& windows,
    int64_t batch_size, torch::Generator& gen) {
    if (windows.empty()) {
        throw std::runtime_error("make_pedestrian_batch: window list is empty");
    }
    std::vector<torch::Tensor> pos_list, mask_list;
    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t idx = torch::randint(0, static_cast<int64_t>(windows.size()), {1}, gen).item<int64_t>();
        pos_list.push_back(windows[idx].first);
        mask_list.push_back(windows[idx].second);
    }
    return {torch::stack(pos_list, 0), torch::stack(mask_list, 0)};
}

}  // namespace traj