#include "model.hpp"
#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../ops/argmax/op.hpp"
#include <cmath>
#include <cstring>
namespace llaisys::models {

// Device-aware copy between two tensors residing on the same device.
static void copyTensor(tensor_t dst, tensor_t src) {
    size_t bytes = src->numel() * src->elementSize();
    if (dst->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(dst->data(), src->data(), bytes);
    } else {
        llaisys::core::context().setDevice(dst->deviceType(), dst->deviceId());
        auto api = llaisys::core::context().runtime().api();
        api->memcpy_sync(dst->data(), src->data(), bytes, LLAISYS_MEMCPY_D2D);
    }
}

// Copy tensor data back to a host pointer.
static void copyToHost(void *host_dst, tensor_t src) {
    size_t bytes = src->numel() * src->elementSize();
    if (src->deviceType() == LLAISYS_DEVICE_CPU) {
        std::memcpy(host_dst, src->data(), bytes);
    } else {
        llaisys::core::context().setDevice(src->deviceType(), src->deviceId());
        auto api = llaisys::core::context().runtime().api();
        api->memcpy_sync(host_dst, src->data(), bytes, LLAISYS_MEMCPY_D2H);
        api->device_synchronize();
    }
}
Qwen2Model::Qwen2Model(const LlaisysQwen2Meta *meta_, llaisysDeviceType_t device, int dev_id)
    : device_type(device), device_id(dev_id) {
    std::memcpy(&meta, meta_, sizeof(LlaisysQwen2Meta));
    size_t n = meta.nlayer;
    attn_norm_w.resize(n, nullptr);
    attn_q_w.resize(n, nullptr);
    attn_q_b.resize(n, nullptr);
    attn_k_w.resize(n, nullptr);
    attn_k_b.resize(n, nullptr);
    attn_v_w.resize(n, nullptr);
    attn_v_b.resize(n, nullptr);
    attn_o_w.resize(n, nullptr);
    mlp_norm_w.resize(n, nullptr);
    mlp_gate_w.resize(n, nullptr);
    mlp_up_w.resize(n, nullptr);
    mlp_down_w.resize(n, nullptr);
    k_cache.resize(n);
    v_cache.resize(n);
}
void Qwen2Model::initBuffers() {
    size_t seq_len = 1;
    size_t maxseq = meta.maxseq;
    for (size_t i = 0; i < meta.nlayer; i++) {
        k_cache[i] = Tensor::create({maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
        v_cache[i] = Tensor::create({maxseq, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    }
    hidden = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
    attn_norm_out = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
    q_buf = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device_type, device_id);
    k_buf = Tensor::create({seq_len, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    v_buf = Tensor::create({seq_len, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    q_rope = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device_type, device_id);
    k_rope = Tensor::create({seq_len, meta.nkvh, meta.dh}, meta.dtype, device_type, device_id);
    attn_out = Tensor::create({seq_len, meta.nh, meta.dh}, meta.dtype, device_type, device_id);
    o_buf = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
    mlp_norm_out = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
    gate_buf = Tensor::create({seq_len, meta.di}, meta.dtype, device_type, device_id);
    up_buf = Tensor::create({seq_len, meta.di}, meta.dtype, device_type, device_id);
    swiglu_out = Tensor::create({seq_len, meta.di}, meta.dtype, device_type, device_id);
    down_buf = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
    logits = Tensor::create({1, meta.voc}, meta.dtype, device_type, device_id);
    max_idx = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
    max_val = Tensor::create({1}, meta.dtype, device_type, device_id);
    pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
    embed_out = Tensor::create({seq_len, meta.hs}, meta.dtype, device_type, device_id);
}
LlaisysQwen2Weights *Qwen2Model::getWeights() {
    weights_struct.in_embed = in_embed;
    weights_struct.out_embed = out_embed;
    weights_struct.out_norm_w = out_norm_w;
    weights_struct.attn_norm_w = attn_norm_w.data();
    weights_struct.attn_q_w = attn_q_w.data();
    weights_struct.attn_q_b = attn_q_b.data();
    weights_struct.attn_k_w = attn_k_w.data();
    weights_struct.attn_k_b = attn_k_b.data();
    weights_struct.attn_v_w = attn_v_w.data();
    weights_struct.attn_v_b = attn_v_b.data();
    weights_struct.attn_o_w = attn_o_w.data();
    weights_struct.mlp_norm_w = mlp_norm_w.data();
    weights_struct.mlp_gate_w = mlp_gate_w.data();
    weights_struct.mlp_up_w = mlp_up_w.data();
    weights_struct.mlp_down_w = mlp_down_w.data();
    return &weights_struct;
}
void Qwen2Model::forwardLayer(size_t layer_idx, size_t seq_len, size_t past_len) {
    size_t total_len = past_len + seq_len;
    tensor_t attn_norm_w_t = getTensor(attn_norm_w[layer_idx]);
    tensor_t q_w = getTensor(attn_q_w[layer_idx]);
    tensor_t q_b = getTensor(attn_q_b[layer_idx]);
    tensor_t k_w = getTensor(attn_k_w[layer_idx]);
    tensor_t k_b = getTensor(attn_k_b[layer_idx]);
    tensor_t v_w = getTensor(attn_v_w[layer_idx]);
    tensor_t v_b = getTensor(attn_v_b[layer_idx]);
    tensor_t o_w = getTensor(attn_o_w[layer_idx]);
    tensor_t m_norm_w = getTensor(mlp_norm_w[layer_idx]);
    tensor_t gate_w = getTensor(mlp_gate_w[layer_idx]);
    tensor_t up_w = getTensor(mlp_up_w[layer_idx]);
    tensor_t down_w = getTensor(mlp_down_w[layer_idx]);
    ops::rms_norm(attn_norm_out, hidden, attn_norm_w_t, meta.epsilon);
    auto q_2d = q_buf->view({seq_len, meta.nh * meta.dh});
    ops::linear(q_2d, attn_norm_out, q_w, q_b);
    auto k_2d = k_buf->view({seq_len, meta.nkvh * meta.dh});
    ops::linear(k_2d, attn_norm_out, k_w, k_b);
    auto v_2d = v_buf->view({seq_len, meta.nkvh * meta.dh});
    ops::linear(v_2d, attn_norm_out, v_w, v_b);
    ops::rope(q_rope, q_buf, pos_ids, meta.theta);
    ops::rope(k_rope, k_buf, pos_ids, meta.theta);
    // Update KV cache
    auto k_dst = k_cache[layer_idx]->slice(0, past_len, total_len);
    copyTensor(k_dst, k_rope);
    auto v_dst = v_cache[layer_idx]->slice(0, past_len, total_len);
    copyTensor(v_dst, v_buf);
    float scale = 1.0f / sqrtf(static_cast<float>(meta.dh));
    auto k_full = k_cache[layer_idx]->slice(0, 0, total_len);
    auto v_full = v_cache[layer_idx]->slice(0, 0, total_len);
    ops::self_attention(attn_out, q_rope, k_full, v_full, scale);
    auto attn_out_2d = attn_out->view({seq_len, meta.nh * meta.dh});
    ops::linear(o_buf, attn_out_2d, o_w, nullptr);
    // Residual add (in-place element-wise add works for both CPU and CUDA dispatch)
    ops::add(hidden, hidden, o_buf);
    ops::rms_norm(mlp_norm_out, hidden, m_norm_w, meta.epsilon);
    ops::linear(gate_buf, mlp_norm_out, gate_w, nullptr);
    ops::linear(up_buf, mlp_norm_out, up_w, nullptr);
    ops::swiglu(swiglu_out, gate_buf, up_buf);
    ops::linear(down_buf, swiglu_out, down_w, nullptr);
    ops::add(hidden, hidden, down_buf);
}
int64_t Qwen2Model::infer(int64_t *token_ids, size_t ntoken) {
    // Sync global weights from weights_struct (populated by Python via getWeights()).
    // getWeights() is called once at model creation when these members are still null,
    // and Python writes the loaded handles into weights_struct directly. The model's
    // own members would otherwise stay null, so refresh them here.
    in_embed = weights_struct.in_embed;
    out_embed = weights_struct.out_embed;
    out_norm_w = weights_struct.out_norm_w;

    tensor_t in_emb = getTensor(in_embed);
    tensor_t out_norm = getTensor(out_norm_w);
    tensor_t lm_head = getTensor(out_embed);
    for (size_t t = 0; t < ntoken; t++) {
        size_t past_len = kv_len;
        int64_t pos = static_cast<int64_t>(kv_len);
        pos_ids->load(&pos);
        int64_t tid = token_ids[t];
        tensor_t token_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64, device_type, device_id);
        token_tensor->load(&tid);
        ops::embedding(embed_out, token_tensor, in_emb);
        copyTensor(hidden, embed_out);
        for (size_t i = 0; i < meta.nlayer; i++) {
            forwardLayer(i, 1, past_len);
        }
        kv_len++;
    }
    // Only compute logits for the last token
    ops::rms_norm(attn_norm_out, hidden, out_norm, meta.epsilon);
    ops::linear(logits, attn_norm_out, lm_head, nullptr);
    ops::argmax(max_idx, max_val, logits);
    int64_t result;
    copyToHost(&result, max_idx);
    return result;
}
} // namespace llaisys::models
