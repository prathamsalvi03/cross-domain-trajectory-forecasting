#pragma once
#include <torch/torch.h>
#include "eth_ucy_loader.hpp"
#include "metrica_loader.hpp"
#include <string>
#include <vector>
#include <tuple>

namespace traj {

// Pre-loads all windows from both real datasets once, then samples cheap
// mixed-domain batches. Mirrors data/real_data.py's RealDataPool.
struct RealDataPool {
    RealDataPool(const std::string& eth_ucy_root,
                 const std::vector<std::pair<std::string, std::string>>& metrica_games,
                 int64_t window_len = 20, int64_t n_pad = 30,
                 int64_t pedestrian_stride = 5, int64_t sports_stride = 10,
                 int64_t sports_resample = 10,
                 const std::vector<std::string>& pedestrian_scenes =
                     {"eth", "hotel", "univ", "zara1", "zara2"},
                 const std::string& split = "train");

    // domain_mix: fraction of the batch drawn from the pedestrian domain.
    // Returns {positions (B,T,n_pad,2), mask (B,T,n_pad), domain (B,) int64:
    // 0=sports, 1=pedestrian}.
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_batch(
        int64_t batch_size, double domain_mix, torch::Generator& gen);

    std::vector<std::pair<torch::Tensor, torch::Tensor>> ped_windows;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> sports_windows;
    int64_t n_pad;
    int64_t window_len;
};

}  // namespace traj