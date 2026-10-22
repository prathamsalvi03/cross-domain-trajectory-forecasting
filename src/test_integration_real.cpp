#include "eth_ucy_loader.hpp"
#include "graph.hpp"
#include "encoder.hpp"
#include "domain_adapt.hpp"
#include "heads.hpp"
#include "losses.hpp"
#include <iostream>

int main(int argc, char** argv) {
    torch::manual_seed(0);
    std::string root = (argc > 1) ? argv[1] : "/home/claude/GPGraph/dataset";

    int64_t obs_len = 10, pred_len = 10, n_pad = 20, n_modes = 6;
    int64_t window_len = obs_len + pred_len;

    auto windows = traj::load_eth_ucy_windows(root, "zara1", "train", window_len, /*stride=*/5, n_pad);
    std::cout << "loaded " << windows.size() << " real ETH/UCY windows\n";

    torch::Generator gen = at::detail::createCPUGenerator();
    gen.set_current_seed(0);
    int64_t B = 8;
    auto [positions, mask] = traj::make_pedestrian_batch(windows, B, gen);
    auto domain = torch::ones({B}, torch::kInt64);

    auto obs_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)});
    auto obs_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)});
    auto fut_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(obs_len, window_len)});
    auto fut_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(obs_len, window_len)});

    auto last_obs = obs_pos.index({torch::indexing::Slice(), -1}).unsqueeze(1);
    auto fut_disp = (fut_pos - last_obs) * fut_mask.unsqueeze(-1);
    // losses.hpp/heads.hpp work in agent-major (B,N,T,...) for the future
    // ground truth (matching pred_traj's (B,N,K,T,2) layout out of the head),
    // but slicing the scene tensor keeps it time-major (B,T,N,...) like
    // everywhere else -- transpose here, once, at the boundary.
    fut_disp = fut_disp.transpose(1, 2);      // (B,N,T_pred,2)
    fut_mask = fut_mask.transpose(1, 2);      // (B,N,T_pred)

    auto node_feats = traj::build_node_features(obs_pos, obs_mask);
    auto [edge_feats, edge_mask_full] = traj::build_edge_features(obs_pos, obs_mask);
    auto adj = traj::knn_adjacency(edge_feats, edge_mask_full, /*k=*/4);

    std::cout << "node_feats: " << node_feats.sizes() << "\n";
    std::cout << "no NaNs in node_feats: " << (!torch::isnan(node_feats).any().item<bool>()) << "\n";

    traj::SpatioTemporalEncoder encoder(/*node_dim=*/5, /*edge_dim=*/4, /*gat_dim=*/32);
    traj::DomainClassifier domain_clf(/*in_dim=*/32);
    traj::MultiModalHead head(/*embed_dim=*/32, pred_len, n_modes);

    auto h = encoder->forward(node_feats, edge_feats, adj, obs_mask);
    std::cout << "h (agent embeddings): " << h.sizes() << "\n";

    auto domain_logits = domain_clf->forward(h, /*lambda_=*/0.5);
    auto [pred_traj, mode_logits] = head->forward(h);
    std::cout << "pred_traj: " << pred_traj.sizes() << "  mode_logits: " << mode_logits.sizes() << "\n";

    auto [reg_loss, per_mode_err, topk_idx] = traj::ewta_regression_loss(pred_traj, fut_disp, fut_mask, /*k=*/3);
    auto best_mode_idx = topk_idx.index({"...", 0});
    auto agent_has_future = (fut_mask.sum(-1) > 0).to(torch::kFloat32);

    auto mode_loss = traj::mode_classification_loss(mode_logits, best_mode_idx, agent_has_future);
    auto div_loss = traj::diversity_regularizer(pred_traj, fut_mask);

    auto agent_mask_any = (obs_mask.sum(1) > 0).to(torch::kFloat32);
    auto dom_loss = traj::domain_adversarial_loss(domain_logits, domain, agent_mask_any);

    auto total_loss = reg_loss + 0.5 * mode_loss + 0.1 * div_loss + 0.3 * dom_loss;

    std::cout << "reg_loss=" << reg_loss.item<float>()
              << " mode_loss=" << mode_loss.item<float>()
              << " div_loss=" << div_loss.item<float>()
              << " dom_loss=" << dom_loss.item<float>()
              << " total=" << total_loss.item<float>() << "\n";

    total_loss.backward();

    bool any_grad = false;
    for (auto& p : encoder->parameters()) {
        if (p.grad().defined() && p.grad().abs().sum().item<float>() > 0) { any_grad = true; break; }
    }
    std::cout << "encoder received nonzero gradients on REAL data: " << (any_grad ? "True" : "False") << "\n";

    return any_grad ? 0 : 1;
}