#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch version check via CMake TORCH_VERSION build info.\n";
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA available - using GPU.\n";
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not available - using CPU.\n";
    }

    torch::Tensor x = torch::rand({3, 3}, device);
    torch::Tensor y = torch::mm(x, x);
    std::cout << "x =\n" << x << "\n";
    std::cout << "x @ x =\n" << y << "\n";

    torch::Tensor w = torch::randn({3, 3}, torch::requires_grad(true));
    torch::Tensor loss = (w * w).sum();
    loss.backward();
    std::cout << "grad of sum(w^2) w.r.t. w (should be 2w):\n" << w.grad() << "\n";

    return 0;
}
