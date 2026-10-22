#pragma once
#include <torch/torch.h>

namespace traj {

struct MultiModalHeadImpl : torch::nn::Module {
    MultiModalHeadImpl(int64_t embed_dim, int64_t pred_len, int64_t n_modes = 6,
                         int64_t hidden_dim = 128, double dropout = 0.1);

    // h: (B,N,embed_dim) -> (trajectories (B,N,K,pred_len,2), mode_logits (B,N,K))
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor h);

    int64_t n_modes_, pred_len_;
    torch::Tensor anchors;
    torch::nn::Sequential decoder{nullptr}, mode_scorer{nullptr};
};
TORCH_MODULE(MultiModalHead);

}  // namespace traj