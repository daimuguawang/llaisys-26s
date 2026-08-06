#include "add_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
void add_host_impl(T *c, const T *a, const T *b, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float va = static_cast<float>(a[i]);
        float vb = static_cast<float>(b[i]);
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) {
            c[i] = static_cast<T>(va + vb);
        } else {
            c[i] = a[i] + b[i];
        }
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    cudaDeviceSynchronize();

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_a = mxm_copy_to_host(reinterpret_cast<const float *>(a), numel);
        auto h_b = mxm_copy_to_host(reinterpret_cast<const float *>(b), numel);
        std::vector<float> h_c(numel);
        add_host_impl(h_c.data(), h_a.data(), h_b.data(), numel);
        mxm_copy_to_device(reinterpret_cast<float *>(c), h_c.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_a = mxm_copy_to_host(reinterpret_cast<const half *>(a), numel);
        auto h_b = mxm_copy_to_host(reinterpret_cast<const half *>(b), numel);
        std::vector<half> h_c(numel);
        add_host_impl(h_c.data(), h_a.data(), h_b.data(), numel);
        mxm_copy_to_device(reinterpret_cast<half *>(c), h_c.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_a = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(a), numel);
        auto h_b = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(b), numel);
        std::vector<__nv_bfloat16> h_c(numel);
        add_host_impl(h_c.data(), h_a.data(), h_b.data(), numel);
        mxm_copy_to_device(reinterpret_cast<__nv_bfloat16 *>(c), h_c.data(), numel);
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

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
