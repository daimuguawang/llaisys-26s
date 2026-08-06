#include "rms_norm_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

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

// One block per row. Each thread accumulates a partial sum-of-squares, then a
// warp/block reduction produces the RMS factor. Matches the CPU path which
// computes in float: rms = 1 / sqrt(mean(x^2) + eps).
template <typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight,
                                float eps, size_t rows, size_t cols) {
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const T *x = in + row * cols;
    T *y = out + row * cols;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int num_warps = (stride + 31) / 32;

    float sum_sq = 0.0f;
    for (size_t j = (size_t)tid; j < cols; j += (size_t)stride) {
        float v = dev_to_float(x[j]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sum[32];
    if (lane == 0) warp_sum[warp_id] = sum_sq;
    __syncthreads();

    // Final reduction in warp 0
    if (warp_id == 0) {
        float total = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xffffffff, total, offset);
        }
        if (lane == 0) {
            float mean_sq = total / static_cast<float>(cols);
            warp_sum[0] = 1.0f / sqrtf(mean_sq + eps);
        }
    }
    __syncthreads();

    float rms = warp_sum[0];
    for (size_t j = (size_t)tid; j < cols; j += (size_t)stride) {
        float v = dev_to_float(x[j]);
        float w = dev_to_float(weight[j]);
        y[j] = dev_from_float<T>(v * rms * w);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t rows, size_t cols) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    int threads = (int)cols;
    if (threads > 1024) threads = 1024;
    if (threads < 32) threads = 32;
    threads = ((threads + 31) / 32) * 32; // round up to a full-warp multiple

    int blocks = (int)rows;
    if (blocks < 1) blocks = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        rms_norm_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight), eps, rows, cols);
        break;
    case LLAISYS_DTYPE_F16:
        rms_norm_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(in),
            reinterpret_cast<const half *>(weight), eps, rows, cols);
        break;
    case LLAISYS_DTYPE_BF16:
        rms_norm_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(in),
            reinterpret_cast<const __nv_bfloat16 *>(weight), eps, rows, cols);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
