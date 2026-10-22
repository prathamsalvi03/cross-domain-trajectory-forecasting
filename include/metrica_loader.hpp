#pragma once
#include <torch/torch.h>
#include <string>
#include <vector>
#include <utility>

namespace traj {

// Reads one Metrica raw tracking CSV (2 metadata header rows, then a header
// row, then data rows of Period,Frame,Time[s],<entity_1>_x,<entity_1>_y,...
// with the LAST entity always being the ball).
// Returns positions (T, n_entities, 2) with NaN->0 substitution, and
// mask (T, n_entities) = 1 where both x and y were present (not NaN).
std::pair<torch::Tensor, torch::Tensor> read_team_tracking(const std::string& path);

// Combines home players + away players + one shared ball (from the home
// file) into a single scene, converting normalized [0,1] coordinates to
// meters. Agent order: [home_players..., away_players..., ball].
// If n_pad is given and larger than the natural agent count, pads with
// zero/mask=0 slots (e.g. to match another domain's tensor shape).
std::pair<torch::Tensor, torch::Tensor> build_match_scene(
    const std::string& home_path, const std::string& away_path,
    double field_x = 105.0, double field_y = 68.0, int64_t n_pad = -1);

// Takes every `factor`-th frame (e.g. factor=10 at 25Hz -> 2.5Hz, matching
// the ETH/UCY sampling cadence).
std::pair<torch::Tensor, torch::Tensor> resample(
    const torch::Tensor& positions, const torch::Tensor& mask, int64_t factor);

std::vector<std::pair<torch::Tensor, torch::Tensor>> sliding_windows_metrica(
    const torch::Tensor& positions, const torch::Tensor& mask,
    int64_t window_len, int64_t stride = 1, int64_t min_agents = 1);

// End-to-end: load, resample, and window one match.
std::vector<std::pair<torch::Tensor, torch::Tensor>> load_metrica_windows(
    const std::string& home_path, const std::string& away_path,
    int64_t window_len = 20, int64_t stride = 10, int64_t resample_factor = 10,
    int64_t min_agents = 5, int64_t n_pad = -1);

std::pair<torch::Tensor, torch::Tensor> make_sports_batch(
    const std::vector<std::pair<torch::Tensor, torch::Tensor>>& windows,
    int64_t batch_size, torch::Generator& gen);

}  // namespace traj
