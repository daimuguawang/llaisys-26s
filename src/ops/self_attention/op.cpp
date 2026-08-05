// #include "op.hpp"

// namespace llaisys::ops {
// void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cfloat>

namespace llaisys::ops {

template <typename T>
void self_attention_cpu_impl(T* attn_val, const T* q, const T* k, const T* v,
                              float scale, size_t qlen, size_t kvlen,
                              size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t n_rep = n_heads / n_kv_heads;
    // causal mask offset: when kvlen > qlen (KV cache), q position i can see up to i + (kvlen - qlen)
    size_t causal_offset = kvlen - qlen;

    for (size_t h = 0; h < n_heads; h++) {
        size_t kv_h = h / n_rep;

        for (size_t i = 0; i < qlen; i++) {
            const T* q_ptr = q + (i * n_heads + h) * head_dim;

            std::vector<float> scores(kvlen);
            for (size_t j = 0; j < kvlen; j++) {
                const T* k_ptr = k + (j * n_kv_heads + kv_h) * head_dim;
                float dot = 0.0f;
                for (size_t d = 0; d < head_dim; d++) {
                    dot += utils::cast<float>(q_ptr[d]) * utils::cast<float>(k_ptr[d]);
                }
                scores[j] = dot * scale;
            }

            // Causal mask: mask positions j > i + causal_offset
            for (size_t j = i + 1 + causal_offset; j < kvlen; j++) {
                scores[j] = -FLT_MAX;
            }

            // Online softmax (numerically stable)
            float max_score = -FLT_MAX;
            for (size_t j = 0; j < kvlen; j++) {
                max_score = std::max(max_score, scores[j]);
            }
            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] /= sum_exp;
            }

            T* out_ptr = attn_val + (i * n_heads + h) * head_dim;
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    const T* v_ptr = v + (j * n_kv_heads + kv_h) * head_dim;
                    sum += scores[j] * utils::cast<float>(v_ptr[d]);
                }
                out_ptr[d] = utils::cast<T>(sum);
            }
        }
    }
}

void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "SelfAttention: all tensors must be contiguous.");
    ASSERT(attn_val->dtype() == q->dtype() && q->dtype() == k->dtype() && k->dtype() == v->dtype(),
           "SelfAttention: dtypes must match.");
    ASSERT(q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3 && attn_val->ndim() == 3,
           "SelfAttention: all tensors must be 3D.");

    size_t qlen = q->shape()[0];
    size_t n_heads = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t kvlen = k->shape()[0];
    size_t n_kv_heads = k->shape()[1];
    size_t k_head_dim = k->shape()[2];

    ASSERT(v->shape()[0] == kvlen && v->shape()[1] == n_kv_heads && v->shape()[2] == head_dim,
           "SelfAttention: v shape mismatch.");
    ASSERT(k_head_dim == head_dim, "SelfAttention: k/v head_dim must match q.");
    ASSERT(attn_val->shape()[0] == qlen && attn_val->shape()[1] == n_heads && attn_val->shape()[2] == head_dim,
           "SelfAttention: attn_val shape mismatch.");
    ASSERT(n_heads % n_kv_heads == 0, "SelfAttention: n_heads must be divisible by n_kv_heads.");
    ASSERT(kvlen >= qlen, "SelfAttention: kvlen must be >= qlen for causal masking.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte* out_ptr = attn_val->data();
        const std::byte* q_ptr = q->data();
        const std::byte* k_ptr = k->data();
        const std::byte* v_ptr = v->data();

        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            self_attention_cpu_impl(reinterpret_cast<float*>(out_ptr),
                                   reinterpret_cast<const float*>(q_ptr),
                                   reinterpret_cast<const float*>(k_ptr),
                                   reinterpret_cast<const float*>(v_ptr),
                                   scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
            break;
        case LLAISYS_DTYPE_F16:
            self_attention_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr),
                                   reinterpret_cast<const fp16_t*>(q_ptr),
                                   reinterpret_cast<const fp16_t*>(k_ptr),
                                   reinterpret_cast<const fp16_t*>(v_ptr),
                                   scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            self_attention_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr),
                                   reinterpret_cast<const bf16_t*>(q_ptr),
                                   reinterpret_cast<const bf16_t*>(k_ptr),
                                   reinterpret_cast<const bf16_t*>(v_ptr),
                                   scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
        return;
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops