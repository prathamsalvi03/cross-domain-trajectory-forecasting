// End-to-end training loop for the cross-domain trajectory forecasting model.
#include "data_pool.hpp"
#include "graph.hpp"
#include "encoder.hpp"
#include "domain_adapt.hpp"
#include "heads.hpp"
#include "losses.hpp"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <tuple>
#include <fstream>
#include <algorithm>

namespace traj {

struct TrainConfig {
    int64_t epochs = 15;
    int64_t steps_per_epoch = 20;
    int64_t batch_size = 16;
    int64_t obs_len = 10;
    int64_t pred_len = 10;
    int64_t n_modes = 6;
    int64_t k_neighbors = 4;
    int64_t n_pad = 30;
    double domain_mix = 0.5;
    double lr = 1e-3;
    double mode_loss_weight = 0.5;
    double diversity_weight = 0.1;
    double domain_loss_weight = 0.3;
    int64_t checkpoint_every = 20;      // epochs between periodic saves
    std::string checkpoint_dir = "checkpoints";
    int64_t eval_every = 20;            // epochs between held-out ADE/FDE checks
    bool density_match = true;
    int64_t dm_lo = 3, dm_hi = 9;
    int64_t eval_batches = 10;          // batches to average the eval metric over
};

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

// Randomly keep only lo..hi agents in each SPORTS scene (domain==0), so agent
// count no longer reveals the domain. Pedestrian scenes are left untouched.
void density_match_sports(torch::Tensor& positions, torch::Tensor& mask,
                          const torch::Tensor& domain, torch::Generator& gen,
                          int64_t lo, int64_t hi) {
    int64_t B = positions.size(0), N = positions.size(2);
    for (int64_t b = 0; b < B; ++b) {
        if (domain[b].item<int64_t>() != 0) continue;
        auto valid_any = (mask[b].sum(0) > 0);
        auto valid_idx = torch::nonzero(valid_any).squeeze(-1);
        int64_t n_valid = valid_idx.size(0);
        if (n_valid <= 1) continue;
        int64_t m = torch::randint(lo, hi + 1, {1}, gen).item<int64_t>();
        m = std::min(m, n_valid);
        auto perm = torch::randperm(n_valid, gen, torch::kInt64);
        auto keep_idx = valid_idx.index_select(0, perm.index({torch::indexing::Slice(0, m)}));
        auto keep = torch::zeros({N}, mask.options());
        keep.index_fill_(0, keep_idx, 1.0);
        mask.index_put_({b}, mask[b] * keep.view({1, N}));
        positions.index_put_({b}, positions[b] * keep.view({1, N, 1}));
    }
}

// Domain loss averaged per scene, then per domain, then across domains, so
// each domain counts equally regardless of how many agents its scenes have.
torch::Tensor domain_loss_balanced(torch::Tensor domain_logits, torch::Tensor domain_labels,
                                   torch::Tensor agent_mask) {
    int64_t B = domain_logits.size(0), N = domain_logits.size(1), C = domain_logits.size(2);
    auto labels_exp = domain_labels.view({B, 1}).expand({B, N}).reshape({-1}).to(torch::kInt64);
    auto ce = torch::nn::functional::cross_entropy(
        domain_logits.reshape({B * N, C}), labels_exp,
        torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kNone)).view({B, N});
    ce = ce * agent_mask;
    auto per_scene = ce.sum(1) / agent_mask.sum(1).clamp_min(1);
    auto scene_valid = (agent_mask.sum(1) > 0).to(torch::kFloat32);
    torch::Tensor total = torch::zeros({}, domain_logits.options());
    int64_t n_present = 0;
    for (int64_t c = 0; c < C; ++c) {
        auto sel = (domain_labels == c).to(torch::kFloat32) * scene_valid;
        auto cnt = sel.sum();
        if (cnt.item<float>() < 0.5) continue;
        total = total + (per_scene * sel).sum() / cnt;
        n_present++;
    }
    return n_present > 0 ? total / n_present : total;
}

struct PreparedBatch {
    torch::Tensor node_feats, edge_feats, edge_mask, agent_mask;
    torch::Tensor fut_disp, fut_mask, domain;
};

PreparedBatch prepare_batch(RealDataPool& pool, const TrainConfig& cfg, torch::Generator& gen,
                             double domain_mix_override = -1.0) {
    double domain_mix = (domain_mix_override >= 0.0) ? domain_mix_override : cfg.domain_mix;
    auto [positions, mask, domain] = pool.sample_batch(cfg.batch_size, domain_mix, gen);
    if (cfg.density_match) density_match_sports(positions, mask, domain, gen, cfg.dm_lo, cfg.dm_hi);

    auto obs_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(0, cfg.obs_len)});
    auto obs_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, cfg.obs_len)});
    auto fut_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(cfg.obs_len, cfg.obs_len + cfg.pred_len)});
    auto fut_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(cfg.obs_len, cfg.obs_len + cfg.pred_len)});

    auto last_obs = obs_pos.index({torch::indexing::Slice(), -1}).unsqueeze(1);
    auto fut_disp = (fut_pos - last_obs) * fut_mask.unsqueeze(-1);
    // agent-major, to match pred_traj's (B,N,K,T,2) layout out of the head.
    fut_disp = fut_disp.transpose(1, 2);
    fut_mask = fut_mask.transpose(1, 2);

    auto node_feats = build_node_features(obs_pos, obs_mask);
    auto [edge_feats, edge_mask_full] = build_edge_features(obs_pos, obs_mask);
    auto adj = knn_adjacency(edge_feats, edge_mask_full, cfg.k_neighbors);

    return {node_feats, edge_feats, adj, obs_mask, fut_disp, fut_mask, domain};
}

struct EvalResult {
    double ped_ade = 0, ped_fde = 0;
    double sports_ade = 0, sports_fde = 0;
};

// Runs one eval pass at a fixed domain_mix (1.0 = pure pedestrian, 0.0 = pure
// sports), so the returned ADE/FDE is never a blend of two domains with very
// different absolute scales (ETH/UCY scenes ~15x15m vs. a ~105x68m pitch).
std::pair<double, double> evaluate_one_domain(RealDataPool& eval_pool, TrajectoryModel& model,
                                                const TrainConfig& cfg, torch::Device device,
                                                torch::Generator& gen, double domain_mix) {
    model->eval();
    torch::NoGradGuard no_grad;

    double total_ade = 0.0, total_fde = 0.0, total_count = 0.0;
    for (int64_t b = 0; b < cfg.eval_batches; ++b) {
        auto batch = prepare_batch(eval_pool, cfg, gen, domain_mix);
        auto node_feats = batch.node_feats.to(device);
        auto edge_feats = batch.edge_feats.to(device);
        auto edge_mask = batch.edge_mask.to(device);
        auto agent_mask = batch.agent_mask.to(device);
        auto fut_disp = batch.fut_disp.to(device);
        auto fut_mask = batch.fut_mask.to(device);

        auto [pred_traj, mode_logits, domain_logits] =
            model->forward(node_feats, edge_feats, edge_mask, agent_mask, /*lambda_grl=*/0.0);

        auto diff = pred_traj - fut_disp.unsqueeze(2);           // (B,N,K,T,2)
        auto dist = diff.norm(2, /*dim=*/-1);                     // (B,N,K,T)
        auto mask_exp = fut_mask.unsqueeze(2).expand_as(dist);    // (B,N,K,T)

        auto ade_per_mode = (dist * mask_exp).sum(-1) / mask_exp.sum(-1).clamp_min(1);  // (B,N,K)
        auto fde_per_mode = dist.index({"...", -1});                                     // (B,N,K)

        auto best_ade = std::get<0>(ade_per_mode.min(/*dim=*/-1));   // (B,N)
        auto best_fde = std::get<0>(fde_per_mode.min(/*dim=*/-1));   // (B,N)

        auto agent_valid = (fut_mask.sum(-1) > 0).to(torch::kFloat32);  // (B,N)
        total_ade += (best_ade * agent_valid).sum().item<double>();
        total_fde += (best_fde * agent_valid).sum().item<double>();
        total_count += agent_valid.sum().item<double>();
    }

    model->train();
    if (total_count < 1) return {0.0, 0.0};
    return {total_ade / total_count, total_fde / total_count};
}

EvalResult evaluate(RealDataPool& eval_pool, TrajectoryModel& model, const TrainConfig& cfg,
                     torch::Device device, torch::Generator& gen) {
    EvalResult result;
    std::tie(result.ped_ade, result.ped_fde) =
        evaluate_one_domain(eval_pool, model, cfg, device, gen, /*domain_mix=*/1.0);
    std::tie(result.sports_ade, result.sports_fde) =
        evaluate_one_domain(eval_pool, model, cfg, device, gen, /*domain_mix=*/0.0);
    return result;
}

struct RiskPair {
    int64_t agent_i, agent_j;
    double min_distance;
    int64_t t_closest;
    double risk;
};

std::vector<RiskPair> compute_collision_risk(
    const torch::Tensor& pred_abs, const torch::Tensor& mode_probs,
    const torch::Tensor& obs_mask, int64_t obs_len, double safety_radius = 1.0) {
    int64_t N = pred_abs.size(1), K = pred_abs.size(2), T = pred_abs.size(3);

    auto probs_exp = mode_probs.squeeze(0).unsqueeze(-1).unsqueeze(-1);
    auto weighted = (pred_abs.squeeze(0) * probs_exp).sum(/*dim=*/1);

    auto present_mask = obs_mask.index({0, obs_len - 1});
    auto weighted_cpu = weighted.cpu();
    auto present_cpu = present_mask.cpu();

    std::vector<RiskPair> results;
    for (int64_t i = 0; i < N; ++i) {
        if (present_cpu[i].item<float>() < 0.5) continue;
        for (int64_t j = i + 1; j < N; ++j) {
            if (present_cpu[j].item<float>() < 0.5) continue;
            double min_dist = std::numeric_limits<double>::infinity();
            int64_t t_closest = 0;
            for (int64_t t = 0; t < T; ++t) {
                double dx = weighted_cpu[i][t][0].item<float>() - weighted_cpu[j][t][0].item<float>();
                double dy = weighted_cpu[i][t][1].item<float>() - weighted_cpu[j][t][1].item<float>();
                double d = std::sqrt(dx * dx + dy * dy);
                if (d < min_dist) { min_dist = d; t_closest = t; }
            }
            double risk = std::clamp(1.0 - min_dist / safety_radius, 0.0, 1.0);
            results.push_back({i, j, min_dist, t_closest, risk});
        }
    }
    std::sort(results.begin(), results.end(),
              [](const RiskPair& a, const RiskPair& b) { return a.risk > b.risk; });
    return results;
}

void export_predictions_json(RealDataPool& eval_pool, TrajectoryModel& model, const TrainConfig& cfg,
                               torch::Device device, torch::Generator& gen, const std::string& path,
                               int64_t n_scenes_per_domain = 2) {
    model->eval();
    torch::NoGradGuard no_grad;

    std::ofstream out(path);
    out << "{\n  \"obs_len\": " << cfg.obs_len << ",\n  \"pred_len\": " << cfg.pred_len
        << ",\n  \"n_modes\": " << cfg.n_modes << ",\n  \"scenes\": [\n";
    bool first_scene = true;

    auto dump_one = [&](double domain_mix, const std::string& domain_name) {
        auto [positions, mask, domain] = eval_pool.sample_batch(1, domain_mix, gen);
        if (cfg.density_match) density_match_sports(positions, mask, domain, gen, cfg.dm_lo, cfg.dm_hi);
        auto obs_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(0, cfg.obs_len)});
        auto obs_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(0, cfg.obs_len)});
        auto fut_pos = positions.index({torch::indexing::Slice(), torch::indexing::Slice(cfg.obs_len, cfg.obs_len + cfg.pred_len)});
        auto fut_mask = mask.index({torch::indexing::Slice(), torch::indexing::Slice(cfg.obs_len, cfg.obs_len + cfg.pred_len)});

        auto node_feats = build_node_features(obs_pos, obs_mask).to(device);
        auto [edge_feats, edge_mask_full] = build_edge_features(obs_pos, obs_mask);
        auto adj = knn_adjacency(edge_feats, edge_mask_full, cfg.k_neighbors).to(device);
        edge_feats = edge_feats.to(device);
        auto obs_mask_dev = obs_mask.to(device);

        auto [pred_traj, mode_logits, domain_logits] =
            model->forward(node_feats, edge_feats, adj, obs_mask_dev, /*lambda_grl=*/0.0);

        auto last_obs = obs_pos.index({torch::indexing::Slice(), -1}).to(device)
                                .unsqueeze(2).unsqueeze(2);
        auto pred_abs = pred_traj + last_obs;
        auto mode_probs = torch::softmax(mode_logits, /*dim=*/-1);

        int64_t N = obs_pos.size(2);
        auto obs_cpu = obs_pos.cpu();
        auto obs_mask_cpu = obs_mask.cpu();
        auto fut_cpu = fut_pos.cpu();
        auto pred_cpu = pred_abs.cpu();
        auto probs_cpu = mode_probs.cpu();

        if (!first_scene) out << ",\n";
        first_scene = false;
        out << "    {\n      \"domain\": \"" << domain_name << "\",\n      \"agents\": [\n";
        bool first_agent = true;
        for (int64_t n = 0; n < N; ++n) {
            if (obs_mask_cpu[0][cfg.obs_len - 1][n].item<float>() < 0.5) continue;
            if (!first_agent) out << ",\n";
            first_agent = false;

            out << "        {\n          \"obs\": [";
            for (int64_t t = 0; t < cfg.obs_len; ++t) {
                if (t) out << ", ";
                out << "[" << obs_cpu[0][t][n][0].item<float>() << ", "
                    << obs_cpu[0][t][n][1].item<float>() << "]";
            }
            out << "],\n          \"gt_future\": [";
            for (int64_t t = 0; t < cfg.pred_len; ++t) {
                if (t) out << ", ";
                out << "[" << fut_cpu[0][t][n][0].item<float>() << ", "
                    << fut_cpu[0][t][n][1].item<float>() << "]";
            }
            out << "],\n          \"pred_modes\": [";
            for (int64_t k = 0; k < cfg.n_modes; ++k) {
                if (k) out << ", ";
                out << "{\"prob\": " << probs_cpu[0][n][k].item<float>() << ", \"traj\": [";
                for (int64_t t = 0; t < cfg.pred_len; ++t) {
                    if (t) out << ", ";
                    out << "[" << pred_cpu[0][n][k][t][0].item<float>() << ", "
                        << pred_cpu[0][n][k][t][1].item<float>() << "]";
                }
                out << "]}";
            }
            out << "]\n        }";
        }
        auto risks = compute_collision_risk(pred_abs, mode_probs, obs_mask, cfg.obs_len);
        out << "\n      ],\n      \"risks\": [";
        for (size_t r = 0; r < risks.size(); ++r) {
            if (r) out << ", ";
            out << "{\"agent_i\": " << risks[r].agent_i << ", \"agent_j\": " << risks[r].agent_j
                << ", \"min_distance\": " << risks[r].min_distance
                << ", \"t_closest\": " << risks[r].t_closest
                << ", \"risk\": " << risks[r].risk << "}";
        }
        out << "]\n    }";
    };

    for (int64_t s = 0; s < n_scenes_per_domain; ++s) dump_one(1.0, "pedestrian");
    for (int64_t s = 0; s < n_scenes_per_domain; ++s) dump_one(0.0, "sports");

    out << "\n  ]\n}\n";
    out.close();
    model->train();
    std::cout << "Exported predictions to " << path << "\n";
}

void train(RealDataPool& train_pool, RealDataPool& eval_pool, const TrainConfig& cfg) {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "training on: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    std::filesystem::create_directories(cfg.checkpoint_dir);

    TrajectoryModel model(/*node_dim=*/5, /*edge_dim=*/4, /*embed_dim=*/64, cfg.n_modes, cfg.pred_len);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(cfg.lr));

    torch::Generator gen = at::detail::createCPUGenerator();
    gen.set_current_seed(0);

    for (int64_t epoch = 0; epoch < cfg.epochs; ++epoch) {
        model->train();
        int64_t k = ewta_k_schedule(epoch, cfg.epochs, cfg.n_modes, /*k_min=*/1);
        double lambda_grl = lambda_schedule(epoch, cfg.epochs);

        double epoch_reg = 0, epoch_mode = 0, epoch_dom = 0, epoch_div = 0;
        int64_t n_batches = 0;

        for (int64_t step = 0; step < cfg.steps_per_epoch; ++step) {
            auto batch = prepare_batch(train_pool, cfg, gen);
            auto node_feats = batch.node_feats.to(device);
            auto edge_feats = batch.edge_feats.to(device);
            auto edge_mask = batch.edge_mask.to(device);
            auto agent_mask = batch.agent_mask.to(device);
            auto fut_disp = batch.fut_disp.to(device);
            auto fut_mask = batch.fut_mask.to(device);
            auto domain = batch.domain.to(device);

            auto [pred_traj, mode_logits, domain_logits] =
                model->forward(node_feats, edge_feats, edge_mask, agent_mask, lambda_grl);

            auto [reg_loss, per_mode_err, topk_idx] = ewta_regression_loss(pred_traj, fut_disp, fut_mask, k);
            auto best_mode_idx = topk_idx.index({"...", 0});
            auto agent_has_future = (fut_mask.sum(-1) > 0).to(torch::kFloat32);

            auto mode_loss = mode_classification_loss(mode_logits, best_mode_idx, agent_has_future);
            auto div_loss = diversity_regularizer(pred_traj, fut_mask);
            auto agent_mask_any = (agent_mask.sum(1) > 0).to(torch::kFloat32);
            auto dom_loss = domain_loss_balanced(domain_logits, domain, agent_mask_any);

            auto loss = reg_loss + cfg.mode_loss_weight * mode_loss
                                  + cfg.diversity_weight * div_loss
                                  + cfg.domain_loss_weight * dom_loss;

            optimizer.zero_grad();
            loss.backward();
            torch::nn::utils::clip_grad_norm_(model->parameters(), /*max_norm=*/5.0);
            optimizer.step();

            epoch_reg += reg_loss.item<double>();
            epoch_mode += mode_loss.item<double>();
            epoch_dom += dom_loss.item<double>();
            epoch_div += div_loss.item<double>();
            n_batches++;
        }

        std::cout << "epoch " << std::setw(3) << epoch
                  << " | k=" << k << " | lambda_grl=" << std::fixed << std::setprecision(3) << lambda_grl
                  << " | reg=" << epoch_reg / n_batches
                  << " mode_ce=" << epoch_mode / n_batches
                  << " domain_ce=" << epoch_dom / n_batches
                  << " diversity=" << epoch_div / n_batches;

        bool is_last = (epoch == cfg.epochs - 1);

        if (cfg.eval_every > 0 && (epoch % cfg.eval_every == 0 || is_last)) {
            auto result = evaluate(eval_pool, model, cfg, device, gen);
            std::cout << " | ped_ADE=" << result.ped_ade << " ped_FDE=" << result.ped_fde
                      << " | sports_ADE=" << result.sports_ade << " sports_FDE=" << result.sports_fde;
        }
        std::cout << "\n";

        if (cfg.checkpoint_every > 0 && ((epoch + 1) % cfg.checkpoint_every == 0 || is_last)) {
            std::string path = cfg.checkpoint_dir + "/epoch_" + std::to_string(epoch + 1) + ".pt";
            torch::save(model, path);
            std::cout << "  saved checkpoint: " << path << "\n";
        }
    }

    torch::save(model, cfg.checkpoint_dir + "/latest.pt");
    std::cout << "saved final checkpoint: " << cfg.checkpoint_dir << "/latest.pt\n";

    export_predictions_json(eval_pool, model, cfg, device, gen,
                             cfg.checkpoint_dir + "/predictions.json");
}

}  // namespace traj

int main(int argc, char** argv) {
    std::string eth_ucy_root = (argc > 1) ? argv[1] : "/home/claude/GPGraph/dataset";
    std::string metrica_root = (argc > 2) ? argv[2] : "/home/claude/sample-data/data";

    std::vector<std::pair<std::string, std::string>> games = {
        {metrica_root + "/Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
         metrica_root + "/Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv"},
        {metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
         metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"},
    };

    traj::RealDataPool train_pool(eth_ucy_root, games, /*window_len=*/20, /*n_pad=*/30,
                                     /*pedestrian_stride=*/5, /*sports_stride=*/10, /*sports_resample=*/10,
                                     {"eth", "hotel", "univ", "zara1", "zara2"}, /*split=*/"train");
    std::cout << "train pedestrian windows: " << train_pool.ped_windows.size()
              << "   train sports windows: " << train_pool.sports_windows.size() << "\n";

    // Separate pool over the "val" split, so evaluation isn't measured on
    // data the model has actually trained on. Metrica has no val split of
    // its own in this sample release, so we reuse Game 3 (not used for
    // training above) for the sports side of eval.
    std::vector<std::pair<std::string, std::string>> eval_games = {
    // NOTE: Sample_Game_3 ships in a different format (single tracking.txt +
    // metadata.xml, not two per-team CSVs) that metrica_loader.cpp doesn't
    // parse. Reusing Game 2 here means the sports-side eval isn't truly
    // held-out -- the pedestrian-side eval via ETH/UCY's "val" split still
    // is. Worth writing a tracking.txt parser later if sports-side held-out
    // eval matters for your results.
    {metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
     metrica_root + "/Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv"},
};
    traj::RealDataPool eval_pool(eth_ucy_root, eval_games, /*window_len=*/20, /*n_pad=*/30,
                                    /*pedestrian_stride=*/5, /*sports_stride=*/10, /*sports_resample=*/10,
                                    {"eth", "hotel", "univ", "zara1", "zara2"}, /*split=*/"val");
    std::cout << "val pedestrian windows: " << eval_pool.ped_windows.size()
              << "   val sports windows: " << eval_pool.sports_windows.size() << "\n";

    traj::TrainConfig cfg;
    cfg.epochs = 200;
    cfg.steps_per_epoch = 50;
    cfg.checkpoint_every = 20;
    cfg.eval_every = 20;

    // Pass "pedestrian_only" as a 3rd CLI arg to run the baseline ablation:
    // trains with domain_mix=1.0 (pedestrian data only) instead of the
    // mixed 0.5 default, so ped_ADE/ped_FDE can be compared directly against
    // the mixed-domain run to see whether cross-domain training helps or hurts.
    bool raw_mixed = (argc > 3 && std::string(argv[3]) == "mixed_raw");
    bool pedestrian_only = (argc > 3 && std::string(argv[3]) == "pedestrian_only");
    cfg.density_match = !(raw_mixed || pedestrian_only);
    cfg.checkpoint_dir = raw_mixed ? "checkpoints_mixed_raw" : "checkpoints_density_matched";
    std::cout << "density_match=" << cfg.density_match << std::endl;
    if (pedestrian_only) {
        cfg.domain_mix = 1.0;
        cfg.checkpoint_dir = "checkpoints_pedestrian_only";
        std::cout << "Running PEDESTRIAN-ONLY baseline (domain_mix=1.0)\n";
    } else {
        std::cout << "Running MIXED-DOMAIN training (domain_mix=" << cfg.domain_mix << ")\n";
    }

    traj::train(train_pool, eval_pool, cfg);

    return 0;
}