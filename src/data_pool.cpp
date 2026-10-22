#include "data_pool.hpp"
#include <stdexcept>

namespace traj {

RealDataPool::RealDataPool(const std::string& eth_ucy_root,
                            const std::vector<std::pair<std::string, std::string>>& metrica_games,
                            int64_t window_len_, int64_t n_pad_,
                            int64_t pedestrian_stride, int64_t sports_stride,
                            int64_t sports_resample,
                            const std::vector<std::string>& pedestrian_scenes,
                            const std::string& split)
    : n_pad(n_pad_), window_len(window_len_) {
    for (const auto& scene : pedestrian_scenes) {
        auto w = load_eth_ucy_windows(eth_ucy_root, scene, split, window_len, pedestrian_stride, n_pad);
        ped_windows.insert(ped_windows.end(), w.begin(), w.end());
    }
    if (ped_windows.empty()) {
        throw std::runtime_error("No ETH/UCY windows loaded - check eth_ucy_root path.");
    }

    for (const auto& [home_path, away_path] : metrica_games) {
        auto w = load_metrica_windows(home_path, away_path, window_len, sports_stride,
                                        sports_resample, /*min_agents=*/5, n_pad);
        sports_windows.insert(sports_windows.end(), w.begin(), w.end());
    }
    if (sports_windows.empty()) {
        throw std::runtime_error("No Metrica windows loaded - check metrica_games paths.");
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RealDataPool::sample_batch(
    int64_t batch_size, double domain_mix, torch::Generator& gen) {
    int64_t n_ped = static_cast<int64_t>(std::llround(batch_size * domain_mix));
    n_ped = std::max<int64_t>(0, std::min(batch_size, n_ped));
    int64_t n_sports = batch_size - n_ped;

    std::vector<torch::Tensor> pos_parts, mask_parts, domain_parts;

    if (n_ped > 0) {
        auto [pos, mask] = make_pedestrian_batch(ped_windows, n_ped, gen);
        pos_parts.push_back(pos);
        mask_parts.push_back(mask);
        domain_parts.push_back(torch::ones({n_ped}, torch::kInt64));
    }
    if (n_sports > 0) {
        auto [pos, mask] = make_sports_batch(sports_windows, n_sports, gen);
        pos_parts.push_back(pos);
        mask_parts.push_back(mask);
        domain_parts.push_back(torch::zeros({n_sports}, torch::kInt64));
    }

    auto positions = torch::cat(pos_parts, /*dim=*/0);
    auto mask = torch::cat(mask_parts, /*dim=*/0);
    auto domain = torch::cat(domain_parts, /*dim=*/0);

    auto order = torch::randperm(batch_size, gen, torch::kInt64);
    return {positions.index_select(0, order), mask.index_select(0, order), domain.index_select(0, order)};
}

}  // namespace traj

