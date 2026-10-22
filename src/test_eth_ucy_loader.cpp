#include "eth_ucy_loader.hpp"
#include <iostream>

int main(int argc, char** argv) {
    std::string root = (argc > 1) ? argv[1] : "/home/claude/GPGraph/dataset";

    auto windows = traj::load_eth_ucy_windows(root, "zara1", "train", /*window_len=*/20, /*stride=*/5);
    std::cout << "zara1/train: " << windows.size() << " windows of shape "
              << windows[0].first.sizes() << "\n";

    torch::Generator gen = at::detail::createCPUGenerator();
    gen.set_current_seed(0);
    auto [pos_batch, mask_batch] = traj::make_pedestrian_batch(windows, /*batch_size=*/4, gen);
    std::cout << "batch positions: " << pos_batch.sizes() << "  batch mask: " << mask_batch.sizes() << "\n";

    auto valid_agents = mask_batch.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}).sum(-1);
    std::cout << "valid agents per scene: " << valid_agents << "\n";

    for (const std::string& scene : {"eth", "hotel", "univ", "zara1", "zara2"}) {
        for (const std::string& split : {"train", "val", "test"}) {
            auto w = traj::load_eth_ucy_windows(root, scene, split, 20, 10);
            std::cout << scene << "/" << split << ": " << w.size() << " windows\n";
        }
    }
    return 0;
}