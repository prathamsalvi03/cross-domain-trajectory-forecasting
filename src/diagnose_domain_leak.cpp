// Diagnostic: proves (or disproves) whether the domain classifier's accuracy
// is actually explained by something as trivial as agent count, rather than
// genuine learned motion-pattern differences. If a 1D decision-stump on
// agent count alone matches the learned classifier's accuracy, that's strong
// evidence the domain gap is structural (soccer scenes have ~22-23 agents,
// ETH/UCY scenes have far fewer) rather than fixable via better features.
//
// Loads a checkpoint saved by train.cpp's torch::save(model, path), so the
// TrajectoryModelImpl definition below must match train.cpp's exactly.
#include "data_pool.hpp"
#include "graph.hpp"
#include "encoder.hpp"
#include "domain_adapt.hpp"
#include "heads.hpp"
#include "losses.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

namespace traj {

// Must match train.cpp's TrajectoryModelImpl exactly for torch::load to work.
struct TrajectoryModelImpl : torch::nn::Module {
    TrajectoryModelImpl(int64_t node_dim, int64_t edge_dim, int64_t embed_dim,
                          int64_t n_modes, int64_t pred_len) {
        encoder = register_module("encoder",
            SpatioTemporalEncoder(node_dim, edge_dim, embed_dim));
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
        std::cerr << "Usage: " << argv[0]
                  << " <eth_ucy_root> <metrica_root> <checkpoint_path> [n_batches=20]\n";
        return 1;
    }
    std::string eth_ucy_root = argv[1];
    std::string metrica_root = argv[2];
    std::string checkpoint_path = argv[3];
    int64_t n_batches = (argc > 4) ? std::stoll(argv[4]) : 20;

    int64_t obs_len = 10, pred_len = 10, n_pad = 30, n_modes = 6, batch_size = 16;
    int64_t window_len = obs_len + pred_len;

    std::vector<std::pair<std::string, std::string>> eval_games = {
        {metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
         metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"},
    };
    traj::RealDataPool eval_pool(eth_ucy_root, eval_games, window_len, n_pad,
                                    5, 10, 10, {"eth", "hotel", "univ", "zara1", "zara2"}, "val");
    std::cout << "val pedestrian windows: " << eval_pool.ped_windows.size()
              << "   val sports windows: " << eval_pool.sports_windows.size() << "\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    traj::TrajectoryModel model(/*node_dim=*/5, /*edge_dim=*/4, /*embed_dim=*/64, n_modes, pred_len);
    torch::load(model, checkpoint_path);
    model->to(device);
    model->eval();
    torch::NoGradGuard no_grad;

    torch::Generator gen = at::detail::createCPUGenerator();
    gen.set_current_seed(123);  // different seed than training, for an independent eval set

    std::vector<int64_t> agent_counts;
    std::vector<int64_t> true_domains;
    std::vector<int64_t> predicted_domains;

    for (int64_t b = 0; b < n_batches; ++b) {
        auto [positions, mask, domain] = eval_pool.sample_batch(batch_size, /*domain_mix=*/0.5, gen);

        auto obs_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)});
        auto obs_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)});

        auto node_feats = traj::build_node_features(obs_pos, obs_mask).to(device);
        auto [edge_feats, edge_mask_full] = traj::build_edge_features(obs_pos, obs_mask);
        auto adj = traj::knn_adjacency(edge_feats, edge_mask_full, /*k=*/4).to(device);
        edge_feats = edge_feats.to(device);
        auto obs_mask_dev = obs_mask.to(device);

        auto [pred_traj, mode_logits, domain_logits] =
            model->forward(node_feats, edge_feats, adj, obs_mask_dev, /*lambda_grl=*/0.0);

        auto pred_domain_per_agent = std::get<1>(domain_logits.max(/*dim=*/-1));  // (B,N)
        auto agent_count_per_scene = obs_mask.index({torch::indexing::Slice(), -1})
                                              .sum(/*dim=*/-1).to(torch::kInt64);  // (B,)

        auto pred_cpu = pred_domain_per_agent.cpu();
        auto count_cpu = agent_count_per_scene.cpu();
        auto obs_mask_cpu = obs_mask.cpu();
        int64_t B = positions.size(0), N = positions.size(2);

        for (int64_t s = 0; s < B; ++s) {
            int64_t count = count_cpu[s].item<int64_t>();
            int64_t true_dom = domain[s].item<int64_t>();
            int64_t votes_sports = 0, votes_ped = 0;
            for (int64_t n = 0; n < N; ++n) {
                if (obs_mask_cpu[s][obs_len - 1][n].item<float>() < 0.5) continue;
                int64_t p = pred_cpu[s][n].item<int64_t>();
                if (p == 0) votes_sports++; else votes_ped++;
            }
            int64_t pred_dom = (votes_sports >= votes_ped) ? 0 : 1;

            agent_counts.push_back(count);
            true_domains.push_back(true_dom);
            predicted_domains.push_back(pred_dom);
        }
    }

    int64_t n = static_cast<int64_t>(agent_counts.size());
    int64_t correct = 0;
    for (int64_t i = 0; i < n; ++i) if (predicted_domains[i] == true_domains[i]) correct++;
    double learned_acc = static_cast<double>(correct) / n;

    std::vector<int64_t> sorted_counts = agent_counts;
    std::sort(sorted_counts.begin(), sorted_counts.end());
    sorted_counts.erase(std::unique(sorted_counts.begin(), sorted_counts.end()), sorted_counts.end());

    double best_baseline_acc = 0.0;
    double best_threshold = 0.0;
    bool best_high_is_sports = true;

    auto eval_threshold = [&](double thresh, bool high_is_sports) {
        int64_t c = 0;
        for (int64_t i = 0; i < n; ++i) {
            bool predicts_sports = high_is_sports ? (agent_counts[i] >= thresh) : (agent_counts[i] < thresh);
            int64_t pred = predicts_sports ? 0 : 1;
            if (pred == true_domains[i]) c++;
        }
        return static_cast<double>(c) / n;
    };

    for (size_t idx = 0; idx + 1 < sorted_counts.size(); ++idx) {
        double thresh = (sorted_counts[idx] + sorted_counts[idx + 1]) / 2.0;
        for (bool high_is_sports : {true, false}) {
            double acc = eval_threshold(thresh, high_is_sports);
            if (acc > best_baseline_acc) {
                best_baseline_acc = acc;
                best_threshold = thresh;
                best_high_is_sports = high_is_sports;
            }
        }
    }

    std::cout << "\n=== Domain-leak diagnostic (" << n << " scenes) ===\n";
    std::cout << "Learned domain classifier accuracy:        " << learned_acc << "\n";
    std::cout << "Agent-count-only baseline accuracy:        " << best_baseline_acc
              << "  (threshold=" << best_threshold
              << ", " << (best_high_is_sports ? "count>=T -> sports" : "count<T -> sports") << ")\n";
    std::cout << "Gap (learned - baseline):                  " << (learned_acc - best_baseline_acc) << "\n";
    if (learned_acc - best_baseline_acc < 0.05) {
        std::cout << "-> The learned classifier barely beats a trivial agent-count rule.\n"
                  << "   This supports the hypothesis that domain leakage is structural\n"
                  << "   (driven by agent count/density), not something gradient reversal\n"
                  << "   on motion features alone can fix.\n";
    } else {
        std::cout << "-> The learned classifier meaningfully outperforms agent count alone,\n"
                  << "   so the domain gap is NOT purely explained by agent count --\n"
                  << "   something else in the learned features still reveals domain.\n";
    }

    return 0;
}
