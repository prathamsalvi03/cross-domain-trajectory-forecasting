#include "losses.hpp"
#include <algorithm>
#include <cmath>

namespace traj {

using torch::indexing::Slice;
using torch::indexing::None;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ewta_regression_loss(
    torch::Tensor pred_traj, torch::Tensor gt_traj, torch::Tensor gt_mask, int64_t k) {
    int64_t B = pred_traj.size(0), N = pred_traj.size(1), K = pred_traj.size(2), T = pred_traj.size(3);

    auto gt_exp = gt_traj.unsqueeze(2).expand({B, N, K, T, 2});
    auto mask_exp = gt_mask.unsqueeze(2).expand({B, N, K, T});

    auto per_step_err = (pred_traj - gt_exp).pow(2).sum(/*dim=*/-1);  // (B,N,K,T)
    per_step_err = per_step_err * mask_exp;
    auto valid_steps = mask_exp.sum(/*dim=*/-1).clamp_min(1);          // (B,N,K)
    auto per_mode_error = per_step_err.sum(/*dim=*/-1) / valid_steps;  // (B,N,K)

    int64_t k_eff = std::max<int64_t>(1, std::min<int64_t>(k, K));
    auto topk = torch::topk(per_mode_error, k_eff, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
    auto topk_vals = std::get<0>(topk);
    auto topk_idx = std::get<1>(topk);

    auto agent_has_future = (gt_mask.sum(/*dim=*/-1) > 0).to(torch::kFloat32);  // (B,N)
    auto loss_per_agent = topk_vals.mean(/*dim=*/-1) * agent_has_future;
    auto denom = agent_has_future.sum().clamp_min(1);
    auto loss = loss_per_agent.sum() / denom;

    return {loss, per_mode_error, topk_idx};
}

int64_t ewta_k_schedule(int64_t epoch, int64_t total_epochs, int64_t k_max, int64_t k_min) {
    if (total_epochs <= 1) return k_min;
    double frac = std::min(std::max(static_cast<double>(epoch) / (total_epochs - 1), 0.0), 1.0);
    int64_t k = static_cast<int64_t>(std::llround(k_max - frac * (k_max - k_min)));
    return std::max(k_min, std::min(k_max, k));
}

torch::Tensor mode_classification_loss(
    torch::Tensor mode_logits, torch::Tensor best_mode_idx, torch::Tensor agent_has_future) {
    int64_t B = mode_logits.size(0), N = mode_logits.size(1), K = mode_logits.size(2);

    auto logits_flat = mode_logits.reshape({B * N, K});
    auto target_flat = best_mode_idx.reshape({B * N}).to(torch::kInt64);
    auto mask_flat = agent_has_future.reshape({B * N});

    auto ce = torch::nn::functional::cross_entropy(
        logits_flat, target_flat,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
    ce = ce * mask_flat;
    return ce.sum() / mask_flat.sum().clamp_min(1);
}

torch::Tensor diversity_regularizer(torch::Tensor pred_traj, torch::Tensor gt_mask) {
    int64_t K = pred_traj.size(2);

    auto endpoints = pred_traj.index({Slice(), Slice(), Slice(), -1, Slice()});  // (B,N,K,2)
    auto diff = endpoints.unsqueeze(3) - endpoints.unsqueeze(2);                  // (B,N,K,K,2)
    auto pairwise_dist = diff.norm(2, /*dim=*/-1);                                // (B,N,K,K)
    auto eye = torch::eye(K, pred_traj.options()).view({1, 1, K, K});
    pairwise_dist = pairwise_dist + eye * 1e6;
    auto min_dist = std::get<0>(pairwise_dist.min(/*dim=*/-1));                   // (B,N,K)

    auto agent_valid = (gt_mask.sum(/*dim=*/-1) > 0).to(torch::kFloat32).unsqueeze(-1);  // (B,N,1)
    auto penalty = torch::clamp_min(1.0 - min_dist, 0.0) * agent_valid;
    return penalty.sum() / agent_valid.sum().clamp_min(1);
}

torch::Tensor domain_adversarial_loss(
    torch::Tensor domain_logits, torch::Tensor domain_labels, torch::Tensor agent_mask) {
    int64_t B = domain_logits.size(0), N = domain_logits.size(1), C = domain_logits.size(2);

    auto labels_exp = domain_labels.view({B, 1}).expand({B, N}).reshape({-1}).to(torch::kInt64);
    auto logits_flat = domain_logits.reshape({B * N, C});
    auto mask_flat = agent_mask.reshape({-1});

    auto ce = torch::nn::functional::cross_entropy(
        logits_flat, labels_exp,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone));
    ce = ce * mask_flat;
    return ce.sum() / mask_flat.sum().clamp_min(1);
}

}  // namespace traj