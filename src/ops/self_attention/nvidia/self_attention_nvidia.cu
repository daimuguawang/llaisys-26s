#include "self_attention_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

#include <cfloat>
#include <cmath>

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
float to_float(T v) {
    if constexpr (std::is_same_v<T, half>) {
        return static_cast<float>(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return static_cast<float>(v);
    } else {
        return v;
    }
}

template <typename T>
T from_float(float v) {
    if constexpr (std::is_same_v<T, half>) {
        return static_cast<half>(v);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return static_cast<__nv_bfloat16>(v);
    } else {
        return v;
    }
}

// Causal self-attention with GQA (host-side implementation)
//   q:[qlen,n_heads,head_dim], k/v:[kvlen,n_kv_heads,head_dim], out:[qlen,n_heads,head_dim]
template <typename T>
void self_attention_host_impl(T *attn_val, const T *q, const T *k, const T *v,
                              float scale, size_t qlen, size_t kvlen,
                              size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t n_rep = n_heads / n_kv_heads;
    size_t causal_offset = kvlen - qlen;

    for (size_t i = 0; i < qlen; i++) {
        for (size_t h = 0; h < n_heads; h++) {
            size_t kv_h = h / n_rep;

            const T *q_ptr = q + (i * n_heads + h) * head_dim;
            T *out_ptr = attn_val + (i * n_heads + h) * head_dim;

            // Compute scores
            std::vector<float> scores(kvlen);
            for (size_t j = 0; j < kvlen; j++) {
                if (j > i + causal_offset) {
                    scores[j] = -FLT_MAX;
                } else {
                    const T *k_ptr = k + (j * n_kv_heads + kv_h) * head_dim;
                    float dot = 0.0f;
                    for (size_t d = 0; d < head_dim; d++) {
                        dot += to_float(q_ptr[d]) * to_float(k_ptr[d]);
                    }
                    scores[j] = dot * scale;
                }
            }

            // Softmax (numerically stable)
            float max_score = -FLT_MAX;
            for (size_t j = 0; j < kvlen; j++) {
                max_score = std::max(max_score, scores[j]);
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }

            // Normalize
            for (size_t j = 0; j < kvlen; j++) {
                scores[j] /= sum_exp;
            }

            // Weighted sum of v
            for (size_t d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (size_t j = 0; j < kvlen; j++) {
                    const T *v_ptr = v + (j * n_kv_heads + kv_h) * head_dim;
                    sum += scores[j] * to_float(v_ptr[d]);
                }
                out_ptr[d] = from_float<T>(sum);
            }
        }
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, float scale,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim) {
    cudaDeviceSynchronize();

    size_t q_numel = qlen * n_heads * head_dim;
    size_t kv_numel = kvlen * n_kv_heads * head_dim;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_q = mxm_copy_to_host(reinterpret_cast<const float *>(q), q_numel);
        auto h_k = mxm_copy_to_host(reinterpret_cast<const float *>(k), kv_numel);
        auto h_v = mxm_copy_to_host(reinterpret_cast<const float *>(v), kv_numel);
        std::vector<float> h_out(q_numel);
        self_attention_host_impl(h_out.data(), h_q.data(), h_k.data(), h_v.data(),
                                 scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<float *>(attn_val), h_out.data(), q_numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_q = mxm_copy_to_host(reinterpret_cast<const half *>(q), q_numel);
        auto h_k = mxm_copy_to_host(reinterpret_cast<const half *>(k), kv_numel);
        auto h_v = mxm_copy_to_host(reinterpret_cast<const half *>(v), kv_numel);
        std::vector<half> h_out(q_numel);
        self_attention_host_impl(h_out.data(), h_q.data(), h_k.data(), h_v.data(),
                                 scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<half *>(attn_val), h_out.data(), q_numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_q = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(q), q_numel);
        auto h_k = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(k), kv_numel);
        auto h_v = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(v), kv_numel);
        std::vector<__nv_bfloat16> h_out(q_numel);
        self_attention_host_impl(h_out.data(), h_q.data(), h_k.data(), h_v.data(),
                                 scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<__nv_bfloat16 *>(attn_val), h_out.data(), q_numel);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

#else  // NVIDIA CUDA

template <typename T>
__device__ __forceinline__ float dev_to_float(T v) {
    if constexpr (std::is_same<T, half>::value) {
        return __half2float(v);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return __bfloat162float(v);
    } else {
        return v;
    }
}

template <typename T>
__device__ __forceinline__ T dev_from_float(float v) {
    if constexpr (std::is_same<T, half>::value) {
        return __float2half(v);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
        return __float2bfloat16(v);
    } else {
        return v;
    }
}

// Causal self-attention with GQA.
//   q:[qlen,n_heads,head_dim], k/v:[kvlen,n_kv_heads,head_dim], out:[qlen,n_heads,head_dim]
// One block per (query position i, head h). Scores for the block live in dynamic
// shared memory. All math is done in float to match the CPU reference.
template <typename T>
__global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T *v,
                                      float scale, size_t qlen, size_t kvlen,
                                      size_t n_heads, size_t n_kv_heads, size_t head_dim) {
    size_t i = blockIdx.x;
    size_t h = blockIdx.y;
    if (i >= qlen || h >= n_heads) return;

    size_t n_rep = n_heads / n_kv_heads;
    size_t kv_h = h / n_rep;
    size_t causal_offset = kvlen - qlen;

    extern __shared__ float scores[]; // size == kvlen

    const T *q_ptr = q + (i * n_heads + h) * head_dim;
    T *out_ptr = attn_val + (i * n_heads + h) * head_dim;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = (stride + WARP_SIZE - 1) / WARP_SIZE;

    // Pass 1: score[j] = dot(q[i,h,:], k[j,kv_h,:]) * scale
    for (size_t j = (size_t)tid; j < kvlen; j += (size_t)stride) {
        const T *k_ptr = k + (j * n_kv_heads + kv_h) * head_dim;
        float dot = 0.0f;
        for (size_t d = 0; d < head_dim; d++) {
            dot += dev_to_float(q_ptr[d]) * dev_to_float(k_ptr[d]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Causal mask: positions j > i + causal_offset are forbidden.
    for (size_t j = (size_t)tid; j < kvlen; j += (size_t)stride) {
        if (j > i + causal_offset) {
            scores[j] = -FLT_MAX;
        }
    }
    __syncthreads();

    // Find row max (numerically stable softmax).
    float local_max = -FLT_MAX;
    for (size_t j = (size_t)tid; j < kvlen; j += (size_t)stride) {
        local_max = fmaxf(local_max, scores[j]);
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }
    __shared__ float s_reduce[WARP_SIZE];
    if (lane == 0) s_reduce[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        float total = (lane < num_warps) ? s_reduce[lane] : -FLT_MAX;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total = fmaxf(total, __shfl_down_sync(0xffffffff, total, offset));
        }
        if (lane == 0) s_reduce[0] = total;
    }
    __syncthreads();
    float max_score = s_reduce[0];

    // exp(score - max) and sum.
    float local_sum = 0.0f;
    for (size_t j = (size_t)tid; j < kvlen; j += (size_t)stride) {
        scores[j] = expf(scores[j] - max_score);
        local_sum += scores[j];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) s_reduce[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float total = (lane < num_warps) ? s_reduce[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0) s_reduce[0] = total;
    }
    __syncthreads();
    float sum_exp = s_reduce[0];

    // Normalize scores (matches CPU: scores[j] /= sum_exp before the weighted sum).
    for (size_t j = (size_t)tid; j < kvlen; j += (size_t)stride) {
        scores[j] /= sum_exp;
    }
    __syncthreads();

    // Weighted sum of v along kvlen, accumulated in float (same order as CPU).
    for (size_t d = (size_t)tid; d < head_dim; d += (size_t)stride) {
        float sum = 0.0f;
        for (size_t j = 0; j < kvlen; j++) {
            const T *v_ptr = v + (j * n_kv_heads + kv_h) * head_dim;
            sum += scores[j] * dev_to_float(v_ptr[d]);
        }
        out_ptr[d] = dev_from_float<T>(sum);
    }
}

void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                    const std::byte *v, llaisysDataType_t type, float scale,
                    size_t qlen, size_t kvlen, size_t n_heads, size_t n_kv_heads,
                    size_t head_dim) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    int threads = (int)head_dim;
    if (threads > 256) threads = 256;
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE; // full warps for shuffle reduction

    dim3 grid((unsigned)qlen, (unsigned)n_heads);
    size_t shared_bytes = kvlen * sizeof(float);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        self_attention_kernel<float><<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        self_attention_kernel<half><<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<half *>(attn_val), reinterpret_cast<const half *>(q),
            reinterpret_cast<const half *>(k), reinterpret_cast<const half *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        self_attention_kernel<__nv_bfloat16><<<grid, threads, shared_bytes, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(attn_val), reinterpret_cast<const __nv_bfloat16 *>(q),
            reinterpret_cast<const __nv_bfloat16 *>(k), reinterpret_cast<const __nv_bfloat16 *>(v),
            scale, qlen, kvlen, n_heads, n_kv_heads, head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
