// Exports a predictions.json (consumed by make_video.py) from a trained
// checkpoint, using real held-out ETH/UCY pedestrian windows and real
// Metrica soccer windows -- not training data, not fake test data.
//
//   ./export_showcase <eth_ucy_root> <metrica_root> <checkpoint.pt> <out.json> [n_candidates]
//
// Samples n_candidates windows per domain, picks the one with the most
// fully-observed-and-future agents (best for a visual), runs the model,
// and writes obs/gt_future/pred_modes per agent.
#include "graph.hpp"
#include "encoder.hpp"
#include "domain_adapt.hpp"
#include "heads.hpp"
#include "data_pool.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
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

static std::string fnum(double v) {
    std::ostringstream o;
    o << std::setprecision(5) << v;
    return o.str();
}

// Samples a batch from one domain, picks the window with the most agents
// valid across the whole 20-step span, runs the model, and appends a JSON
// scene object to `out`. Returns true if a scene was written.
static bool export_one_domain(traj::RealDataPool& pool, double domain_mix, const char* domain_name,
                              traj::TrajectoryModel& model, torch::Device device,
                              int64_t obs_len, int64_t pred_len, int64_t k_neighbors,
                              int64_t n_candidates, torch::Generator& gen, std::ostream& out,
                              bool& first_scene) {
    auto sampled = pool.sample_batch(n_candidates, domain_mix, gen);
    auto positions = std::get<0>(sampled);   // (B,T,N,2)
    auto mask = std::get<1>(sampled);        // (B,T,N)

    auto min_pair = mask.min(1);
    auto full_valid = std::get<0>(min_pair).ge(0.5).contiguous();  // (B,N) bool
    auto counts = full_valid.sum(1);                               // (B,)
    int64_t best_b = counts.argmax().item<int64_t>();
    if (counts[best_b].item<int64_t>() == 0) {
        std::cerr << "no fully-valid agents found in " << n_candidates << " " << domain_name
                  << " windows; try a larger n_candidates\n";
        return false;
    }

    auto obs_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)}).contiguous();
    auto obs_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, obs_len)}).contiguous();
    auto fut_pos = positions.index({torch::indexing::Slice(),
                                    torch::indexing::Slice(obs_len, obs_len + pred_len)}).contiguous();
    auto last_obs = obs_pos.index({torch::indexing::Slice(), obs_len - 1}).contiguous();  // (B,N,2)

    auto node_feats = traj::build_node_features(obs_pos, obs_mask).to(device);
    auto edge_pair = traj::build_edge_features(obs_pos, obs_mask);
    auto edge_feats = std::get<0>(edge_pair);
    auto edge_mask_full = std::get<1>(edge_pair);
    auto adj = traj::knn_adjacency(edge_feats, edge_mask_full, k_neighbors).to(device);

    torch::NoGradGuard no_grad;
    auto fwd = model->forward(node_feats, edge_feats.to(device), adj, obs_mask.to(device), /*lambda_grl=*/0.0);
    auto pred = std::get<0>(fwd).cpu().contiguous();          // (B,N,K,P,2) displacement from last obs
    auto probs = torch::softmax(std::get<1>(fwd), -1).cpu().contiguous();  // (B,N,K)

    auto pa = pred.accessor<float, 5>();
    auto pr = probs.accessor<float, 3>();
    auto fv = full_valid.accessor<bool, 2>();
    auto op = obs_pos.accessor<float, 4>();
    auto fp = fut_pos.accessor<float, 4>();
    auto lo = last_obs.accessor<float, 3>();

    int64_t N = positions.size(2), K = probs.size(2);
    bool any_agent = false;
    std::ostringstream scene;
    scene << "{\"domain\":\"" << domain_name << "\",\"agents\":[";
    bool first_agent = true;
    for (int64_t n = 0; n < N; ++n) {
        if (!fv[best_b][n]) continue;
        any_agent = true;
        if (!first_agent) scene << ",";
        first_agent = false;
        scene << "{\"obs\":[";
        for (int64_t t = 0; t < obs_len; ++t) {
            if (t) scene << ",";
            scene << "[" << fnum(op[best_b][t][n][0]) << "," << fnum(op[best_b][t][n][1]) << "]";
        }
        scene << "],\"gt_future\":[";
        for (int64_t t = 0; t < pred_len; ++t) {
            if (t) scene << ",";
            scene << "[" << fnum(fp[best_b][t][n][0]) << "," << fnum(fp[best_b][t][n][1]) << "]";
        }
        scene << "],\"pred_modes\":[";
        for (int64_t k = 0; k < K; ++k) {
            if (k) scene << ",";
            scene << "{\"prob\":" << fnum(pr[best_b][n][k]) << ",\"traj\":[";
            for (int64_t t = 0; t < pred_len; ++t) {
                if (t) scene << ",";
                double x = lo[best_b][n][0] + pa[best_b][n][k][t][0];
                double y = lo[best_b][n][1] + pa[best_b][n][k][t][1];
                scene << "[" << fnum(x) << "," << fnum(y) << "]";
            }
            scene << "]}";
        }
        scene << "]}";
    }
    scene << "]}";
    if (!any_agent) return false;
    if (!first_scene) out << ",";
    out << scene.str();
    first_scene = false;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <eth_ucy_root> <metrica_root> <checkpoint.pt> <out.json> [n_candidates]\n";
        return 1;
    }
    std::string eth_ucy_root = argv[1], metrica_root = argv[2], ckpt_path = argv[3], out_path = argv[4];
    int64_t n_candidates = (argc > 5) ? std::stoll(argv[5]) : 24;

    const int64_t obs_len = 10, pred_len = 10, window_len = 20, n_pad = 30, k_neighbors = 4;
    const int64_t n_modes = 6;

    // Same val-split sports game as train.cpp's eval_pool: not truly held-out
    // for sports (see the comment in train.cpp), but genuinely held-out for
    // pedestrians via the "val" split.
    std::vector<std::pair<std::string, std::string>> games = {
        {metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
         metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"},
    };
    traj::RealDataPool pool(eth_ucy_root, games, window_len, n_pad,
                            /*pedestrian_stride=*/5, /*sports_stride=*/10, /*sports_resample=*/10,
                            {"eth", "hotel", "univ", "zara1", "zara2"}, /*split=*/"val");
    std::cout << "pedestrian windows: " << pool.ped_windows.size()
              << "   sports windows: " << pool.sports_windows.size() << "\n";

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    traj::TrajectoryModel model(/*node_dim=*/5, /*edge_dim=*/4, /*embed_dim=*/64, n_modes, pred_len);
    torch::load(model, ckpt_path);
    model->to(device);
    model->eval();
    std::cout << "loaded checkpoint " << ckpt_path << " on " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    auto gen = at::make_generator<at::CPUGeneratorImpl>(42);

    std::ofstream out(out_path);
    out << "{\"obs_len\":" << obs_len << ",\"pred_len\":" << pred_len << ",\"scenes\":[";
    bool first_scene = true;
    bool ok1 = export_one_domain(pool, /*domain_mix=*/1.0, "pedestrian", model, device,
                                 obs_len, pred_len, k_neighbors, n_candidates, gen, out, first_scene);
    bool ok2 = export_one_domain(pool, /*domain_mix=*/0.0, "sports", model, device,
                                 obs_len, pred_len, k_neighbors, n_candidates, gen, out, first_scene);
    out << "]}";
    out.close();
    std::cout << "pedestrian scene written: " << ok1 << "   sports scene written: " << ok2 << "\n";
    std::cout << "wrote " << out_path << "\n";
    return 0;
}
