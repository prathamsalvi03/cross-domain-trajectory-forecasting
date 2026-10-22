#pragma once
#include <torch/torch.h>
#include <tuple>

namespace traj {

// Mirrors data/graph.py's compute_kinematics: forward-difference velocity,
// heading from atan2(vy, vx). positions: (B,T,N,2), mask: (B,T,N).
std::pair<torch::Tensor, torch::Tensor> compute_kinematics(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt = 1.0);

// Mirrors build_node_features: returns (B,T,N,5) = [x,y,vx,vy,heading]
torch::Tensor build_node_features(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt = 1.0);

// Mirrors build_edge_features: returns
//   edge_feats (B,T,N,N,4) = [rel_dx, rel_dy, distance, closing_speed]
//   edge_mask  (B,T,N,N)   1 where both endpoints valid and i != j
std::pair<torch::Tensor, torch::Tensor> build_edge_features(
    const torch::Tensor& positions, const torch::Tensor& mask, double dt = 1.0);

// Mirrors knn_adjacency: sparsify to k nearest neighbours per agent by distance
// (edge_feats channel index 2), ANDed with edge_mask.
torch::Tensor knn_adjacency(
    const torch::Tensor& edge_feats, const torch::Tensor& edge_mask, int64_t k);

}  // namespace traj