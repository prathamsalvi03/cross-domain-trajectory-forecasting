#include "metrica_loader.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : "/home/claude/sample-data/data/Sample_Game_1";
    std::string home = root + "/Sample_Game_1_RawTrackingData_Home_Team.csv";
    std::string away = root + "/Sample_Game_1_RawTrackingData_Away_Team.csv";

    auto [positions, mask] = traj::build_match_scene(home, away);
    std::cout << "full match positions: " << positions.sizes() << "  mask: " << mask.sizes() << "\n";

    auto agents_frame1000 = mask.index({1000}).sum();
    std::cout << "agents per frame (frame 1000): " << agents_frame1000.item<float>()
              << " / " << mask.size(1) << "\n";

    auto home_p0_x = positions.index({torch::indexing::Slice(), 0, 0});
    std::cout << "home player 0 x range (m): " << home_p0_x.min().item<float>()
              << " " << home_p0_x.max().item<float>() << "\n";

    auto ball_x = positions.index({torch::indexing::Slice(), -1, 0});
    std::cout << "ball x range (m): " << ball_x.min().item<float>()
              << " " << ball_x.max().item<float>() << "\n";

    auto windows = traj::load_metrica_windows(home, away, /*window_len=*/20, /*stride=*/20,
                                                /*resample_factor=*/10);
    std::cout << "windows: " << windows.size() << " of shape " << windows[0].first.sizes() << "\n";

    torch::Generator gen = at::detail::createCPUGenerator();
    gen.set_current_seed(0);
    auto [pos_batch, mask_batch] = traj::make_sports_batch(windows, /*batch_size=*/4, gen);
    std::cout << "batch positions: " << pos_batch.sizes() << "  batch mask: " << mask_batch.sizes() << "\n";

    auto valid_agents = mask_batch.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).sum(-1);
    std::cout << "valid agents per scene: " << valid_agents << "\n";

    auto [positions_padded, mask_padded] = traj::build_match_scene(home, away, 105.0, 68.0, /*n_pad=*/30);
    std::cout << "padded positions: " << positions_padded.sizes() << "\n";

    return 0;
}