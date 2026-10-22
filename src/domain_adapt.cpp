#include "domain_adapt.hpp"
#include <cmath>

namespace traj {

namespace {

class GradientReversalFunction : public torch::autograd::Function<GradientReversalFunction> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, torch::Tensor x, double lambda_) {
        ctx->saved_data["lambda"] = lambda_;
        return x.view_as(x);
    }
    static torch::autograd::tensor_list backward(torch::autograd::AutogradContext* ctx,
                                                    torch::autograd::tensor_list grad_outputs) {
        double lambda_ = ctx->saved_data["lambda"].toDouble();
        return {-lambda_ * grad_outputs[0], torch::Tensor()};
    }
};

}  // namespace

torch::Tensor gradient_reversal(torch::Tensor x, double lambda_) {
    return GradientReversalFunction::apply(x, lambda_);
}

DomainClassifierImpl::DomainClassifierImpl(int64_t in_dim, int64_t hidden_dim, int64_t n_domains,
                                             double dropout) {
    net = register_module("net", torch::nn::Sequential(
        torch::nn::Linear(in_dim, hidden_dim),
        torch::nn::ReLU(),
        torch::nn::Dropout(dropout),
        torch::nn::Linear(hidden_dim, n_domains)
    ));
}

torch::Tensor DomainClassifierImpl::forward(torch::Tensor h, double lambda_) {
    auto reversed_h = gradient_reversal(h, lambda_);
    return net->forward(reversed_h);
}

double lambda_schedule(int64_t epoch, int64_t total_epochs, double gamma) {
    double p = std::min(std::max(static_cast<double>(epoch) / std::max<int64_t>(total_epochs, 1), 0.0), 1.0);
    return 2.0 / (1.0 + std::exp(-gamma * p)) - 1.0;
}

}  // namespace traj