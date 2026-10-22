#include "graph.hpp"
#include <cmath>

namespace traj {

using torch::indexing::Slice;
using torch::indexing::None;

std::pair<torch::Tensor, torch::Tensor> compute_kinematics(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt) {
    auto T = positions.size(1);
    auto vel = torch::zeros_like(positions);

    if (T > 1) {
        auto diff = (positions.index({Slice(), Slice(1, None)}) -
                     positions.index({Slice(), Slice(None, -1)})) / dt;
        vel.index_put_({Slice(), Slice(1, None)}, diff);
        vel.index_put_({Slice(), 0}, vel.index({Slice(), 1}));
    }
    vel = vel * mask.unsqueeze(-1);

    auto heading = torch::atan2(vel.index({"...", 1}), vel.index({"...", 0})).unsqueeze(-1);
    return {vel, heading};
}

torch::Tensor build_node_features(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt) {
    auto [vel, heading] = compute_kinematics(positions, mask, dt);
    auto node_feats = torch::cat({positions, vel, heading}, /*dim=*/-1);
    node_feats = node_feats * mask.unsqueeze(-1);
    return node_feats;
}

std::pair<torch::Tensor, torch::Tensor> build_edge_features(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt) {
    auto sizes = positions.sizes();
    int64_t N = sizes[2];
    auto [vel, heading] = compute_kinematics(positions, mask, dt);
    (void)heading;

    auto pos_i = positions.unsqueeze(3);  // (B,T,N,1,2)
    auto pos_j = positions.unsqueeze(2);  // (B,T,1,N,2)
    auto rel = pos_j - pos_i;             // (B,T,N,N,2) vector from i to j
    auto dist = rel.norm(2, /*dim=*/-1, /*keepdim=*/true);  // (B,T,N,N,1)

    auto vel_i = vel.unsqueeze(3);
    auto vel_j = vel.unsqueeze(2);
    auto rel_vel = vel_j - vel_i;
    auto unit = rel / (dist + 1e-6);
    auto closing_speed = (rel_vel * unit).sum(/*dim=*/-1, /*keepdim=*/true);

    auto edge_feats = torch::cat({rel, dist, closing_speed}, /*dim=*/-1).to(torch::kFloat32);

    auto valid_i = mask.unsqueeze(3);  // (B,T,N,1)
    auto valid_j = mask.unsqueeze(2);  // (B,T,1,N)
    auto edge_mask = valid_i * valid_j;  // (B,T,N,N)
    auto eye = 1.0 - torch::eye(N, positions.options()).view({1, 1, N, N});
    edge_mask = (edge_mask * eye).to(torch::kFloat32);

    edge_feats = edge_feats * edge_mask.unsqueeze(-1);
    return {edge_feats, edge_mask};
}

torch::Tensor knn_adjacency(
    const torch::Tensor& edge_feats, const torch::Tensor& edge_mask, int64_t k) {
    auto dist = edge_feats.index({"...", 2}).clone();  // (B,T,N,N)
    dist = torch::where(edge_mask == 0, torch::full_like(dist, std::numeric_limits<float>::infinity()), dist);

    int64_t N = dist.size(-1);
    int64_t k_eff = (N > 1) ? std::min(k, N - 1) : 0;
    if (k_eff <= 0) {
        return torch::zeros_like(edge_mask);
    }

    auto sorted_idx = std::get<1>(torch::sort(dist, /*dim=*/-1));
    auto nearest_idx = sorted_idx.index({"...", Slice(None, k_eff)});  // (B,T,N,k_eff)

    auto adj = torch::zeros_like(edge_mask);
    auto ones = torch::ones_like(nearest_idx, edge_mask.options());
    adj = adj.scatter(/*dim=*/-1, nearest_idx, ones);

    return adj * edge_mask;
}

}  // namespace traj