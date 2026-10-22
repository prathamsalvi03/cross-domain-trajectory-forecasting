#pragma once
#include <torch/torch.h>

namespace traj {

// Ganin & Lempitsky gradient reversal, as a custom autograd::Function.
torch::Tensor gradient_reversal(torch::Tensor x, double lambda_ = 1.0);

struct DomainClassifierImpl : torch::nn::Module {
    DomainClassifierImpl(int64_t in_dim, int64_t hidden_dim = 64, int64_t n_domains = 2,
                           double dropout = 0.1);
    torch::Tensor forward(torch::Tensor h, double lambda_ = 1.0);

    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(DomainClassifier);

double lambda_schedule(int64_t epoch, int64_t total_epochs, double gamma = 10.0);

}  // namespace traj