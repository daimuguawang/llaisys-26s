#include "rope_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

#include <cmath>

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
void rope_host_impl(T *out, const T *in, const int64_t *pos_ids,
                    float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        for (size_t h = 0; h < n_heads; h++) {
            const T *x = in + (s * n_heads + h) * head_dim;
            T *y = out + (s * n_heads + h) * head_dim;
            for (size_t j = 0; j < half_dim; j++) {
                float exponent = static_cast<float>(2 * j) / static_cast<float>(head_dim);
                float freqs = static_cast<float>(pos) / std::pow(theta, exponent);
                float cos_val = std::cos(freqs);
                float sin_val = std::sin(freqs);

                float a = static_cast<float>(x[j]);
                float b = static_cast<float>(x[j + half_dim]);

                if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) {
                    y[j] = static_cast<T>(a * cos_val - b * sin_val);
                    y[j + half_dim] = static_cast<T>(b * cos_val + a * sin_val);
                } else {
                    y[j] = a * cos_val - b * sin_val;
                    y[j + half_dim] = b * cos_val + a * sin_val;
                }
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    cudaDeviceSynchronize();

    size_t numel = seq_len * n_heads * head_dim;
    size_t num_pos = seq_len;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const float *>(in), numel);
        auto h_pos_ids = mxm_copy_to_host(pos_ids, num_pos);
        std::vector<float> h_out(numel);
        rope_host_impl(h_out.data(), h_in.data(), h_pos_ids.data(), theta, seq_len, n_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<float *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const half *>(in), numel);
        auto h_pos_ids = mxm_copy_to_host(pos_ids, num_pos);
        std::vector<half> h_out(numel);
        rope_host_impl(h_out.data(), h_in.data(), h_pos_ids.data(), theta, seq_len, n_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<half *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(in), numel);
        auto h_pos_ids = mxm_copy_to_host(pos_ids, num_pos);
        std::vector<__nv_bfloat16> h_out(numel);
        rope_host_impl(h_out.data(), h_in.data(), h_pos_ids.data(), theta, seq_len, n_heads, head_dim);
        mxm_copy_to_device(reinterpret_cast<__nv_bfloat16 *>(out), h_out.data(), numel);
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

// RoPE. Layout: in/out [seq_len, n_heads, head_dim].
// grid(seq_len, n_heads), blockDim covers head_dim/2 pairs.
// freq = pos / pow(theta, 2j / head_dim) computed in float, matching the CPU.
template <typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids,
                            float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t s = blockIdx.x;
    size_t h = blockIdx.y;
    if (s >= seq_len || h >= n_heads) return;

    size_t half_dim = head_dim / 2;
    int64_t pos = pos_ids[s];

    const T *x = in + (s * n_heads + h) * head_dim;
    T *y = out + (s * n_heads + h) * head_dim;

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    for (size_t j = tid; j < half_dim; j += stride) {
        float exponent = static_cast<float>(2 * j) / static_cast<float>(head_dim);
        float freqs = static_cast<float>(pos) / powf(theta, exponent);
        float cos_val = cosf(freqs);
        float sin_val = sinf(freqs);

        float a = dev_to_float(x[j]);
        float b = dev_to_float(x[j + half_dim]);

        y[j] = dev_from_float<T>(a * cos_val - b * sin_val);
        y[j + half_dim] = dev_from_float<T>(b * cos_val + a * sin_val);
    }
}

void rope(std::byte *out, const std::byte *in, const int64_t *pos_ids,
          llaisysDataType_t type, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    int threads = (int)(head_dim / 2);
    if (threads > 256) threads = 256;
    if (threads < 1) threads = 1;

    dim3 grid((unsigned)seq_len, (unsigned)n_heads);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rope_kernel<float><<<grid, threads, 0, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            pos_ids, theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_F16:
        rope_kernel<half><<<grid, threads, 0, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
            pos_ids, theta, seq_len, n_heads, head_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        rope_kernel<__nv_bfloat16><<<grid, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            pos_ids, theta, seq_len, n_heads, head_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
