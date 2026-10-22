#include "encoder.hpp"
#include <cmath>

namespace traj {

using torch::indexing::Slice;
using torch::indexing::None;

// ---------------------------------------------------------------- DenseGAT

DenseGATLayerImpl::DenseGATLayerImpl(int64_t node_dim, int64_t edge_dim, int64_t out_dim,
                                       int64_t n_heads, double dropout_p)
    : n_heads_(n_heads), head_dim_(out_dim / n_heads) {
    TORCH_CHECK(out_dim % n_heads == 0, "out_dim must be divisible by n_heads");

    q_proj = register_module("q_proj", torch::nn::Linear(node_dim, out_dim));
    k_proj = register_module("k_proj", torch::nn::Linear(node_dim, out_dim));
    v_proj = register_module("v_proj", torch::nn::Linear(node_dim, out_dim));
    edge_proj = register_module("edge_proj", torch::nn::Linear(edge_dim, n_heads));
    out_proj = register_module("out_proj", torch::nn::Linear(out_dim, out_dim));
    dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    norm = register_module("norm", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({out_dim})));

    use_residual_proj_ = (node_dim != out_dim);
    if (use_residual_proj_) {
        residual_proj = register_module("residual_proj", torch::nn::Linear(node_dim, out_dim));
    }
}

torch::Tensor DenseGATLayerImpl::forward(torch::Tensor node_feats, torch::Tensor edge_feats,
                                           torch::Tensor edge_mask) {
    int64_t B = node_feats.size(0), N = node_feats.size(1);
    int64_t H = n_heads_, D = head_dim_;

    auto q = q_proj(node_feats).view({B, N, H, D}).transpose(1, 2);  // (B,H,N,D)
    auto k = k_proj(node_feats).view({B, N, H, D}).transpose(1, 2);
    auto v = v_proj(node_feats).view({B, N, H, D}).transpose(1, 2);

    auto logits = torch::einsum("bhid,bhjd->bhij", {q, k}) / std::sqrt(static_cast<double>(D));
    auto edge_bias = edge_proj(edge_feats).permute({0, 3, 1, 2});  // (B,H,N,N)
    logits = logits + edge_bias;

    auto mask = edge_mask.unsqueeze(1);  // (B,1,N,N)
    logits = logits.masked_fill(mask == 0, -std::numeric_limits<double>::infinity());

    auto self_mask = torch::eye(N, node_feats.options()).to(torch::kBool).view({1, 1, N, N});
    logits = logits.masked_fill(self_mask, 0.0);

    auto attn = torch::softmax(logits, /*dim=*/-1);
    attn = dropout(attn);
    auto out = torch::einsum("bhij,bhjd->bhid", {attn, v});  // (B,H,N,D)
    out = out.transpose(1, 2).reshape({B, N, H * D});
    out = out_proj(out);

    auto residual = use_residual_proj_ ? residual_proj(node_feats) : node_feats;
    out = norm(out + residual);
    return out;
}

// ------------------------------------------------------------ TemporalEncoder

TemporalEncoderImpl::TemporalEncoderImpl(int64_t d_model, int64_t n_layers, int64_t n_heads,
                                            double dropout_p, int64_t max_len) {
    pos_embed = register_parameter("pos_embed", torch::randn({1, max_len, d_model}) * 0.02);

    auto layer_opts = torch::nn::TransformerEncoderLayerOptions(d_model, n_heads)
                           .dim_feedforward(4 * d_model)
                           .dropout(dropout_p);
    torch::nn::TransformerEncoderLayer layer(layer_opts);
    transformer = register_module("transformer",
        torch::nn::TransformerEncoder(torch::nn::TransformerEncoderOptions(layer, n_layers)));
}

torch::Tensor TemporalEncoderImpl::forward(torch::Tensor x, torch::Tensor key_padding_mask) {
    // x: (B_N, T, D). LibTorch's TransformerEncoder has no batch_first option
    // (unlike Python's nn.TransformerEncoderLayer(batch_first=True)), so we
    // transpose to the (T, B, D) layout it expects and back afterward.
    int64_t T = x.size(1);
    x = x + pos_embed.index({Slice(), Slice(None, T)});
    auto x_tbd = x.transpose(0, 1);  // (T, B_N, D)
    auto out = transformer->forward(x_tbd, /*src_mask=*/torch::Tensor(), key_padding_mask);
    return out.transpose(0, 1);  // back to (B_N, T, D)
}

// --------------------------------------------------------- SpatioTemporalEncoder

SpatioTemporalEncoderImpl::SpatioTemporalEncoderImpl(int64_t node_dim, int64_t edge_dim, int64_t gat_dim,
                                                        int64_t gat_layers, int64_t n_heads,
                                                        int64_t temporal_layers, double dropout) {
    int64_t in_dim = node_dim;
    for (int64_t i = 0; i < gat_layers; ++i) {
        auto layer = DenseGATLayer(in_dim, edge_dim, gat_dim, n_heads, dropout);
        register_module("gat_layer_" + std::to_string(i), layer);
        gat_layers_.push_back(layer);
        in_dim = gat_dim;
    }
    temporal = register_module("temporal", TemporalEncoder(gat_dim, temporal_layers, n_heads, dropout));
    out_dim = gat_dim;
}

torch::Tensor SpatioTemporalEncoderImpl::forward(torch::Tensor node_feats, torch::Tensor edge_feats,
                                                    torch::Tensor edge_mask, torch::Tensor agent_mask) {
    int64_t B = node_feats.size(0), T = node_feats.size(1), N = node_feats.size(2);

    std::vector<torch::Tensor> spatial_steps;
    spatial_steps.reserve(T);
    for (int64_t t = 0; t < T; ++t) {
        auto h_t = node_feats.index({Slice(), t});           // (B,N,node_dim)
        auto e_t = edge_feats.index({Slice(), t});            // (B,N,N,edge_dim)
        auto m_t = edge_mask.index({Slice(), t});              // (B,N,N)
        for (auto& layer : gat_layers_) {
            h_t = layer->forward(h_t, e_t, m_t);
        }
        spatial_steps.push_back(h_t);
    }
    auto spatial_out = torch::stack(spatial_steps, /*dim=*/1);  // (B,T,N,gat_dim)

    int64_t D = spatial_out.size(-1);
    auto flat = spatial_out.permute({0, 2, 1, 3}).reshape({B * N, T, D});  // (B*N,T,D)
    auto pad_mask = (agent_mask.permute({0, 2, 1}).reshape({B * N, T}) == 0);  // True=pad

    // agents never valid in this window would give an all-True padding mask
    // -> unmask position 0 so softmax over an empty row stays finite.
    auto all_pad = pad_mask.all(/*dim=*/1);
    pad_mask.index_put_({all_pad, 0}, false);

    auto temporal_out = temporal->forward(flat, pad_mask);         // (B*N,T,D)
    temporal_out = temporal_out.reshape({B, N, T, D});

    auto valid_counts = agent_mask.sum(/*dim=*/1).clamp_min(1);     // (B,N)
    auto last_idx = (agent_mask.cumsum(1) == valid_counts.unsqueeze(1)).to(torch::kFloat32);
    last_idx = std::get<1>(last_idx.max(/*dim=*/1));                // (B,N) argmax

    auto index = last_idx.view({B, N, 1, 1}).expand({-1, -1, 1, D}).to(torch::kInt64);
    auto h = torch::gather(temporal_out, /*dim=*/2, index).squeeze(2);  // (B,N,D)
    return h;
}

}  // namespace traj