// Runs a trained checkpoint on pre-built observation windows (e.g. nuScenes,
// produced by ns_prepare.py) and writes the K predicted futures per agent.
//
//   ./predict_windows <windows.txt> <checkpoint.pt|random> <out_predictions.txt>
//
// windows.txt : header "W N T", then W*T*N lines "x y valid" (window-major,
//               then time, then agent), positions in metres.
// output      : header "W N K P", then one line per agent that is present at
//               the last observed step:
//               "w n prob_0..prob_{K-1} dx dy (K*P pairs)"
//               dx,dy = displacement from that agent's last observed position.
// Pass "random" instead of a checkpoint to smoke-test with untrained weights.
#include "graph.hpp"
#include "encoder.hpp"
#include "domain_adapt.hpp"
#include "heads.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

namespace traj {

// Must match train.cpp's TrajectoryModelImpl exactly for torch::load to work.
struct TrajectoryModelImpl : torch::nn::Module {
    TrajectoryModelImpl(int64_t node_dim, int64_t edge_dim, int64_t embed_dim,
                        int64_t n_modes, int64_t pred_len) {
        encoder = register_module("encoder", SpatioTemporalEncoder(node_dim, edge_dim, embed_dim));
        domain_clf = register_module("domain_clf", DomainClassifier(embed_dim));
        head = register_module("head", MultiModalHead(embed_dim, pred_len, n_modes));
    }
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
        torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_mask,
        torch::Tensor agent_mask, double lambda_grl) {
        auto h = encoder->forward(node_feats, edge_feats, edge_mask, agent_mask);
        auto domain_logits = domain_clf->forward(h, lambda_grl);
        auto [traj, mode_logits] = head->forward(h);
        return {traj, mode_logits, domain_logits};
    }
    SpatioTemporalEncoder encoder{nullptr};
    DomainClassifier domain_clf{nullptr};
    MultiModalHead head{nullptr};
};
TORCH_MODULE(TrajectoryModel);

}  // namespace traj

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <windows.txt> <checkpoint.pt|random> <out.txt>\n";
        return 1;
    }
    const int64_t n_modes = 6, pred_len = 10, k_neighbors = 4, batch = 32;

    std::ifstream in(argv[1]);
    if (!in) { std::cerr << "cannot open " << argv[1] << "\n"; return 1; }
    int64_t W, N, T;
    in >> W >> N >> T;
    if (!in || W <= 0) { std::cerr << "no windows in " << argv[1] << "\n"; return 1; }
    auto data = torch::empty({W, T, N, 3}, torch::kFloat32);
    auto acc = data.accessor<float, 4>();
    for (int64_t w = 0; w < W; ++w)
        for (int64_t t = 0; t < T; ++t)
            for (int64_t n = 0; n < N; ++n)
                in >> acc[w][t][n][0] >> acc[w][t][n][1] >> acc[w][t][n][2];
    if (!in) { std::cerr << "windows file ended early or is malformed\n"; return 1; }
    auto positions = data.index({"...", torch::indexing::Slice(0, 2)}).contiguous();
    auto mask = data.index({"...", 2}).contiguous();
    std::cout << "loaded " << W << " windows, " << N << " agent slots, " << T << " observed steps\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    traj::TrajectoryModel model(/*node_dim=*/5, /*edge_dim=*/4, /*embed_dim=*/64, n_modes, pred_len);
    if (std::string(argv[2]) != "random") {
        torch::load(model, argv[2]);
        std::cout << "loaded checkpoint " << argv[2] << "\n";
    } else {
        std::cout << "WARNING: using untrained random weights (smoke test only)\n";
    }
    model->to(device);
    model->eval();
    torch::NoGradGuard no_grad;

    std::ofstream out(argv[3]);
    out << W << " " << N << " " << n_modes << " " << pred_len << "\n";
    out << std::setprecision(5);
    int64_t n_written = 0;

    for (int64_t s = 0; s < W; s += batch) {
        int64_t e = std::min(W, s + batch);
        auto pos = positions.index({torch::indexing::Slice(s, e)});
        auto msk = mask.index({torch::indexing::Slice(s, e)});
        auto node_feats = traj::build_node_features(pos, msk).to(device);
        auto [edge_feats, edge_mask_full] = traj::build_edge_features(pos, msk);
        auto adj = traj::knn_adjacency(edge_feats, edge_mask_full, k_neighbors).to(device);
        auto [pred, mode_logits, dom] = model->forward(
            node_feats, edge_feats.to(device), adj, msk.to(device), /*lambda_grl=*/0.0);
        auto probs = torch::softmax(mode_logits, -1).cpu().contiguous();
        pred = pred.cpu().contiguous();
        auto pa = pred.accessor<float, 5>();     // (b, N, K, P, 2)
        auto pr = probs.accessor<float, 3>();    // (b, N, K)
        auto ma = msk.accessor<float, 3>();      // (b, T, N)
        for (int64_t b = 0; b < e - s; ++b) {
            for (int64_t n = 0; n < N; ++n) {
                if (ma[b][T - 1][n] < 0.5f) continue;
                out << (s + b) << " " << n;
                for (int64_t k = 0; k < n_modes; ++k) out << " " << pr[b][n][k];
                for (int64_t k = 0; k < n_modes; ++k)
                    for (int64_t p = 0; p < pred_len; ++p)
                        out << " " << pa[b][n][k][p][0] << " " << pa[b][n][k][p][1];
                out << "\n";
                ++n_written;
            }
        }
    }
    std::cout << "wrote predictions for " << n_written << " agents to " << argv[3] << "\n";
    return 0;
}
