#pragma once
#include <torch/torch.h>
#include <tuple>

namespace traj {

// Note: gt_traj/gt_mask below are AGENT-MAJOR (B,N,T_pred,...), matching
// pred_traj's (B,N,K,T_pred,2) layout out of MultiModalHead. If you slice
// your scene tensor as (B,T,N,...) elsewhere (the convention used by
// graph.hpp/encoder.hpp), transpose dims 1,2 before calling this.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ewta_regression_loss(
    torch::Tensor pred_traj, torch::Tensor gt_traj, torch::Tensor gt_mask, int64_t k);

int64_t ewta_k_schedule(int64_t epoch, int64_t total_epochs, int64_t k_max, int64_t k_min = 1);

torch::Tensor mode_classification_loss(
    torch::Tensor mode_logits, torch::Tensor best_mode_idx, torch::Tensor agent_has_future);

torch::Tensor diversity_regularizer(torch::Tensor pred_traj, torch::Tensor gt_mask);

torch::Tensor domain_adversarial_loss(
    torch::Tensor domain_logits, torch::Tensor domain_labels, torch::Tensor agent_mask);

}  // namespace traj