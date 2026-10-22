#include "metrica_loader.hpp"
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace traj {

namespace {

// Robust comma-split that preserves trailing empty fields (a naive
// getline-based split silently drops a trailing empty field, which matters
// here since every Metrica header/data row ends in a trailing comma).
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        size_t comma = line.find(',', start);
        if (comma == std::string::npos) {
            fields.push_back(line.substr(start));
            break;
        }
        fields.push_back(line.substr(start, comma - start));
        start = comma + 1;
    }
    return fields;
}

double parse_or_nan(const std::string& s) {
    if (s.empty()) return std::numeric_limits<double>::quiet_NaN();
    try {
        return std::stod(s);
    } catch (...) {
        return std::numeric_limits<double>::quiet_NaN();
    }
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> read_team_tracking(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Could not open file: " + path);

    std::string line;
    // Skip the 2 metadata header rows, then the 3rd row is the column-name
    // header (we don't need the names, just to skip it).
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(f, line)) throw std::runtime_error("File too short: " + path);
    }

    std::vector<std::vector<double>> rows;  // each row: full coordinate fields (post meta-cols)
    while (std::getline(f, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        auto fields = split_csv_line(line);
        if (fields.size() < 3) continue;
        std::vector<double> coords;
        coords.reserve(fields.size() - 3);
        for (size_t i = 3; i < fields.size(); ++i) coords.push_back(parse_or_nan(fields[i]));
        rows.push_back(std::move(coords));
    }
    if (rows.empty()) throw std::runtime_error("No data rows parsed from: " + path);

    int64_t n_cols = static_cast<int64_t>(rows[0].size());
    TORCH_CHECK(n_cols % 2 == 0, "expected an even number of x,y columns in ", path, ", got ", n_cols);
    int64_t n_entities = n_cols / 2;
    int64_t T = static_cast<int64_t>(rows.size());

    auto positions = torch::zeros({T, n_entities, 2}, torch::kFloat32);
    auto mask = torch::zeros({T, n_entities}, torch::kFloat32);

    for (int64_t t = 0; t < T; ++t) {
        const auto& row = rows[t];
        for (int64_t e = 0; e < n_entities; ++e) {
            double x = row[2 * e], y = row[2 * e + 1];
            bool valid = !std::isnan(x) && !std::isnan(y);
            mask.index_put_({t, e}, valid ? 1.0f : 0.0f);
            if (valid) {
                positions.index_put_({t, e, 0}, static_cast<float>(x));
                positions.index_put_({t, e, 1}, static_cast<float>(y));
            }
        }
    }
    return {positions, mask};
}

std::pair<torch::Tensor, torch::Tensor> build_match_scene(
    const std::string& home_path, const std::string& away_path,
    double field_x, double field_y, int64_t n_pad) {
    auto [home_pos, home_mask] = read_team_tracking(home_path);
    auto [away_pos, away_mask] = read_team_tracking(away_path);

    int64_t T = std::min(home_pos.size(0), away_pos.size(0));
    home_pos = home_pos.index({torch::indexing::Slice(0, T)});
    home_mask = home_mask.index({torch::indexing::Slice(0, T)});
    away_pos = away_pos.index({torch::indexing::Slice(0, T)});
    away_mask = away_mask.index({torch::indexing::Slice(0, T)});

    int64_t home_n = home_pos.size(1), away_n = away_pos.size(1);
    auto home_players = home_pos.index({torch::indexing::Slice(), torch::indexing::Slice(0, home_n - 1)});
    auto home_ball = home_pos.index({torch::indexing::Slice(), torch::indexing::Slice(home_n - 1, home_n)});
    auto home_players_mask = home_mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, home_n - 1)});
    auto home_ball_mask = home_mask.index({torch::indexing::Slice(), torch::indexing::Slice(home_n - 1, home_n)});
    auto away_players = away_pos.index({torch::indexing::Slice(), torch::indexing::Slice(0, away_n - 1)});
    auto away_players_mask = away_mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, away_n - 1)});

    auto positions = torch::cat({home_players, away_players, home_ball}, /*dim=*/1);
    auto mask = torch::cat({home_players_mask, away_players_mask, home_ball_mask}, /*dim=*/1);

    auto scale = torch::tensor({field_x, field_y}, positions.options());
    positions = positions * scale;
    positions = positions * mask.unsqueeze(-1);  // re-zero anything invalid after scaling

    int64_t n_entities = positions.size(1);
    if (n_pad > 0 && n_pad > n_entities) {
        auto pad_pos = torch::zeros({T, n_pad - n_entities, 2}, torch::kFloat32);
        auto pad_mask = torch::zeros({T, n_pad - n_entities}, torch::kFloat32);
        positions = torch::cat({positions, pad_pos}, 1);
        mask = torch::cat({mask, pad_mask}, 1);
    } else if (n_pad > 0 && n_pad < n_entities) {
        throw std::runtime_error("n_pad smaller than natural agent count; increase n_pad instead of truncating real players.");
    }
    return {positions, mask};
}

std::pair<torch::Tensor, torch::Tensor> resample(
    const torch::Tensor& positions, const torch::Tensor& mask, int64_t factor) {
    auto idx = torch::arange(0, positions.size(0), factor, torch::kInt64);
    return {positions.index_select(0, idx), mask.index_select(0, idx)};
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> sliding_windows_metrica(
    const torch::Tensor& positions, const torch::Tensor& mask,
    int64_t window_len, int64_t stride, int64_t min_agents) {
    int64_t T = positions.size(0);
    std::vector<std::pair<torch::Tensor, torch::Tensor>> windows;
    for (int64_t start = 0; start + window_len <= T; start += stride) {
        auto pos_w = positions.index({torch::indexing::Slice(start, start + window_len)});
        auto mask_w = mask.index({torch::indexing::Slice(start, start + window_len)});
        auto n_active = (mask_w.sum(/*dim=*/0) > 0).sum().item<int64_t>();
        if (n_active >= min_agents) windows.emplace_back(pos_w.clone(), mask_w.clone());
    }
    return windows;
}

std::vector<std::pair<torch::Tensor, torch::Tensor>> load_metrica_windows(
    const std::string& home_path, const std::string& away_path,
    int64_t window_len, int64_t stride, int64_t resample_factor,
    int64_t min_agents, int64_t n_pad) {
    auto [positions, mask] = build_match_scene(home_path, away_path, 105.0, 68.0, n_pad);
    auto [pos_r, mask_r] = resample(positions, mask, resample_factor);
    return sliding_windows_metrica(pos_r, mask_r, window_len, stride, min_agents);
}

std::pair<torch::Tensor, torch::Tensor> make_sports_batch(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& windows,
    int64_t batch_size, torch::Generator& gen) {
    if (windows.empty()) throw std::runtime_error("make_sports_batch: window list is empty");
    std::vector<torch::Tensor> pos_list, mask_list;
    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t idx = torch::randint(0, static_cast<int64_t>(windows.size()), {1}, gen).item<int64_t>();
        pos_list.push_back(windows[idx].first);
        mask_list.push_back(windows[idx].second);
    }
    return {torch::stack(pos_list, 0), torch::stack(mask_list, 0)};
}

}  // namespace traj