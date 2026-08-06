#include "add_nvidia.hpp"

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

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        float va = dev_to_float(a[i]);
        float vb = dev_to_float(b[i]);
        c[i] = dev_from_float<T>(va + vb);
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    const int threads = 256;
    int blocks = (int)((numel + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        add_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(c), reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b), numel);
        break;
    case LLAISYS_DTYPE_F16:
        add_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(c), reinterpret_cast<const half *>(a),
            reinterpret_cast<const half *>(b), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        add_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(c), reinterpret_cast<const __nv_bfloat16 *>(a),
            reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
