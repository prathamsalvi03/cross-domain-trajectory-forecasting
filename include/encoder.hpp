#pragma once
#include <torch/torch.h>

namespace traj {

// Multi-head graph attention over a fully-connected (masked) agent graph,
// with edge features biasing the attention logits. Mirrors
// models/encoder.py's DenseGATLayer.
struct DenseGATLayerImpl : torch::nn::Module {
    DenseGATLayerImpl(int64_t node_dim, int64_t edge_dim, int64_t out_dim,
                       int64_t n_heads = 4, double dropout = 0.1);

    // node_feats: (B,N,node_dim), edge_feats: (B,N,N,edge_dim), edge_mask: (B,N,N)
    // returns: (B,N,out_dim)
    torch::Tensor forward(torch::Tensor node_feats, torch::Tensor edge_feats, torch::Tensor edge_mask);

    int64_t n_heads_, head_dim_;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr};
    torch::nn::Linear edge_proj{nullptr}, out_proj{nullptr}, residual_proj{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::LayerNorm norm{nullptr};
    bool use_residual_proj_;
};
TORCH_MODULE(DenseGATLayer);

// Per-agent temporal transformer over the observed history. Mirrors
// models/encoder.py's TemporalEncoder.
struct TemporalEncoderImpl : torch::nn::Module {
    TemporalEncoderImpl(int64_t d_model, int64_t n_layers = 2, int64_t n_heads = 4,
                          double dropout = 0.1, int64_t max_len = 64);

    // x: (B_N, T, d_model); key_padding_mask: (B_N, T) bool, true = PAD
    torch::Tensor forward(torch::Tensor x, torch::Tensor key_padding_mask);

    torch::Tensor pos_embed;
    torch::nn::TransformerEncoder transformer{nullptr};
};
TORCH_MODULE(TemporalEncoder);

// Ties together per-timestep DenseGAT -> per-agent TemporalEncoder -> pooled
// fusion. Mirrors models/encoder.py's SpatioTemporalEncoder.
struct SpatioTemporalEncoderImpl : torch::nn::Module {
    SpatioTemporalEncoderImpl(int64_t node_dim = 5, int64_t edge_dim = 4, int64_t gat_dim = 64,
                                int64_t gat_layers = 2, int64_t n_heads = 4,
                                int64_t temporal_layers = 2, double dropout = 0.1);

    // node_feats: (B,T,N,node_dim), edge_feats: (B,T,N,N,edge_dim),
    // edge_mask: (B,T,N,N), agent_mask: (B,T,N) -> h: (B,N,out_dim)
    torch::Tensor forward(torch::Tensor node_feats, torch::Tensor edge_feats,
                           torch::Tensor edge_mask, torch::Tensor agent_mask);

    std::vector<DenseGATLayer> gat_layers_;
    TemporalEncoder temporal{nullptr};
    int64_t out_dim;
};
TORCH_MODULE(SpatioTemporalEncoder);

}  // namespace traj