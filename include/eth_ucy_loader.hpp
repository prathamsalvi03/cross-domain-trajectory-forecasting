#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <utility>

namespace traj {

// One (frame, ped_id, x, y) row from a raw ETH/UCY file.
struct PedRow {
    double frame;
    double ped_id;
    double x;
    double y;
};

// Reads a whitespace-delimited file of "<frame> <ped_id> <x> <y>" rows
// (the standard Social-GAN preprocessed ETH/UCY format).
std::vector<PedRow> read_ped_file(const std::string& path);

// Builds a dense (T, n_pad, 2) positions tensor + (T, n_pad) mask from raw
// rows. If more than n_pad pedestrians appear, keeps the n_pad with the most
// valid timesteps (busiest/most-observed agents).
std::pair<torch::Tensor, torch::Tensor> build_scene_tensor(
    const std::vector<PedRow>& raw, int64_t n_pad = 20);

// Slices a full (T, n_pad, 2) scene into fixed-length windows, dropping
// windows with fewer than min_agents agents having any valid timestep.
std::vector<std::pair<torch::Tensor, torch::Tensor>> sliding_windows(
    const torch::Tensor& positions, const torch::Tensor& mask,
    int64_t window_len, int64_t stride = 1, int64_t min_agents = 1);

// Loads every "<root_dir>/<scene>/<split>/*.txt" file and pools their
// sliding windows together. Mirrors data/eth_ucy_loader.py's load_eth_ucy_windows.
std::vector<std::pair<torch::Tensor, torch::Tensor>> load_eth_ucy_windows(
    const std::string& root_dir, const std::string& scene, const std::string& split = "train",
    int64_t window_len = 20, int64_t stride = 1, int64_t n_pad = 20, int64_t min_agents = 2);

// Samples a batch from a pre-loaded list of windows.
std::pair<torch::Tensor, torch::Tensor> make_pedestrian_batch(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& windows,
    int64_t batch_size, torch::Generator& gen);

}  // namespace traj