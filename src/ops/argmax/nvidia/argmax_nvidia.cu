#include "argmax_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cfloat>

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

// Single-block reduction (1024 threads) with grid-stride loop over the input.
// Tracks both the max value and its (lowest) index, matching CPU behaviour
// where strict greater-than comparison keeps the first occurrence.
template <typename T>
__global__ void argmax_kernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float sval[1024];
    __shared__ int64_t sidx[1024];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    float local_max = -FLT_MAX;
    int64_t local_idx = 0;
    for (size_t i = (size_t)tid; i < numel; i += (size_t)stride) {
        float v = dev_to_float(vals[i]);
        if (v > local_max) {
            local_max = v;
            local_idx = (int64_t)i;
        }
    }

    sval[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Block reduction: keep the higher value; on ties keep the lower index.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float other_val = sval[tid + s];
            int64_t other_idx = sidx[tid + s];
            if (other_val > sval[tid] ||
                (other_val == sval[tid] && other_idx < sidx[tid])) {
                sval[tid] = other_val;
                sidx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = sidx[0];
        *max_val = dev_from_float<T>(sval[0]);
    }
}

void argmax(int64_t *max_idx, std::byte *max_val, const std::byte *vals,
            llaisysDataType_t type, size_t numel) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    const int threads = 1024;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        argmax_kernel<float><<<1, threads, 0, stream>>>(
            max_idx, reinterpret_cast<float *>(max_val),
            reinterpret_cast<const float *>(vals), numel);
        break;
    case LLAISYS_DTYPE_F16:
        argmax_kernel<half><<<1, threads, 0, stream>>>(
            max_idx, reinterpret_cast<half *>(max_val),
            reinterpret_cast<const half *>(vals), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        argmax_kernel<__nv_bfloat16><<<1, threads, 0, stream>>>(
            max_idx, reinterpret_cast<__nv_bfloat16 *>(max_val),
            reinterpret_cast<const __nv_bfloat16 *>(vals), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    // Result is read back synchronously by the host.
    llaisys::core::context().runtime().synchronize();
}

} // namespace llaisys::ops::nvidia
