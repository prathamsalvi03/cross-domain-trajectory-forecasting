#include "heads.hpp"

namespace traj {

MultiModalHeadImpl::MultiModalHeadImpl(int64_t embed_dim, int64_t pred_len, int64_t n_modes,
                                          int64_t hidden_dim, double dropout)
    : n_modes_(n_modes), pred_len_(pred_len) {
    anchors = register_parameter("anchors", torch::randn({n_modes, embed_dim}) * 0.02);

    decoder = register_module("decoder", torch::nn::Sequential(
        torch::nn::Linear(embed_dim * 2, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Dropout(dropout),
        torch::nn::Linear(hidden_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim, pred_len * 2)
    ));

    mode_scorer = register_module("mode_scorer", torch::nn::Sequential(
        torch::nn::Linear(embed_dim * 2, hidden_dim / 2),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden_dim / 2, 1)
    ));
}

std::pair<torch::Tensor, torch::Tensor> MultiModalHeadImpl::forward(torch::Tensor h) {
    int64_t B = h.size(0), N = h.size(1), D = h.size(2);
    int64_t K = n_modes_;

    auto h_exp = h.unsqueeze(2).expand({B, N, K, D});                       // (B,N,K,D)
    auto anchors_exp = anchors.view({1, 1, K, D}).expand({B, N, K, D});
    auto joint = torch::cat({h_exp, anchors_exp}, /*dim=*/-1);              // (B,N,K,2D)

    auto traj = decoder->forward(joint).view({B, N, K, pred_len_, 2});
    auto mode_logits = mode_scorer->forward(joint).squeeze(-1);             // (B,N,K)
    return {traj, mode_logits};
}

}  // namespace traj