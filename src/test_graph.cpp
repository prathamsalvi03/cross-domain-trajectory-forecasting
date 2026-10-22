#include "graph.hpp"
#include <iostream>

int main() {
    torch::manual_seed(1);
    int64_t B = 2, T = 20, N = 8;

    auto positions = torch::randn({B, T, N, 2});
    auto mask = torch::zeros({B, T, N});
    mask.index_put_({0, torch::indexing::Slice(), torch::indexing::Slice(0, 6)}, 1.0);
    mask.index_put_({1, torch::indexing::Slice(), torch::indexing::Slice(0, 8)}, 1.0);
    positions = positions * mask.unsqueeze(-1);

    auto node_feats = traj::build_node_features(positions, mask);
    auto [edge_feats, edge_mask] = traj::build_edge_features(positions, mask);
    auto adj = traj::knn_adjacency(edge_feats, edge_mask, /*k=*/3);

    std::cout << "node_feats sizes: " << node_feats.sizes() << "\n";
    std::cout << "edge_feats sizes: " << edge_feats.sizes() << "\n";
    std::cout << "edge_mask nonzero frac: " << edge_mask.mean().item<float>() << "\n";

    auto deg = adj.index({0, 5}).sum(-1);
    std::cout << "knn adjacency degree (scene 0, t=5): " << deg << "\n";

    auto dist = edge_feats.index({0, 5, torch::indexing::Slice(), torch::indexing::Slice(), 2});
    auto closing = edge_feats.index({0, 5, torch::indexing::Slice(), torch::indexing::Slice(), 3});
    auto rel_dx = edge_feats.index({0, 5, torch::indexing::Slice(), torch::indexing::Slice(), 0});

    bool dist_sym = torch::allclose(dist, dist.transpose(0, 1), 1e-4, 1e-4);
    bool closing_sym = torch::allclose(closing, closing.transpose(0, 1), 1e-4, 1e-4);
    bool rel_antisym = torch::allclose(rel_dx, -rel_dx.transpose(0, 1), 1e-4, 1e-4);

    std::cout << "distance symmetric: " << (dist_sym ? "True" : "False") << "\n";
    std::cout << "closing speed symmetric: " << (closing_sym ? "True" : "False") << "\n";
    std::cout << "relative displacement antisymmetric: " << (rel_antisym ? "True" : "False") << "\n";

    return (dist_sym && closing_sym && rel_antisym) ? 0 : 1;
}