#pragma once
#include "../../tensor/tensor.hpp"
#include "../../llaisys/llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include <vector>
namespace llaisys::models {
class Qwen2Model {
public:
    LlaisysQwen2Meta meta;
    llaisysDeviceType_t device_type;
    int device_id;
    // Weight handles (llaisysTensor_t = LlaisysTensor*)
    llaisysTensor_t in_embed = nullptr;
    llaisysTensor_t out_embed = nullptr;
    llaisysTensor_t out_norm_w = nullptr;
    // Per-layer weight handles
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
    // KV Cache
    std::vector<tensor_t> k_cache;
    std::vector<tensor_t> v_cache;
    // Intermediate buffers
    tensor_t hidden;
    tensor_t attn_norm_out;
    tensor_t q_buf;
    tensor_t k_buf;
    tensor_t v_buf;
    tensor_t q_rope;
    tensor_t k_rope;
    tensor_t attn_out;
    tensor_t o_buf;
    tensor_t mlp_norm_out;
    tensor_t gate_buf;
    tensor_t up_buf;
    tensor_t swiglu_out;
    tensor_t down_buf;
    tensor_t logits;
    tensor_t max_idx;
    tensor_t max_val;
    tensor_t pos_ids;
    tensor_t embed_out;
    // Weights struct for C API
    LlaisysQwen2Weights weights_struct;
    // KV cache length
    size_t kv_len = 0;
    Qwen2Model(const LlaisysQwen2Meta *meta_, llaisysDeviceType_t device, int dev_id);
    ~Qwen2Model() = default;
    void initBuffers();
    LlaisysQwen2Weights *getWeights();
    int64_t infer(int64_t *token_ids, size_t ntoken);
private:
    tensor_t getTensor(llaisysTensor_t h) { return h ? h->tensor : nullptr; }
    void forwardLayer(size_t layer_idx, size_t seq_len, size_t past_len);
};
} // namespace llaisys::models
