#include "swiglu_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
void swiglu_host_impl(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = static_cast<float>(gate[i]);
        float u = static_cast<float>(up[i]);
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
        float result = u * g * sigmoid_g;
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) {
            out[i] = static_cast<T>(result);
        } else {
            out[i] = result;
        }
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up,
            llaisysDataType_t type, size_t numel) {
    cudaDeviceSynchronize();

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_gate = mxm_copy_to_host(reinterpret_cast<const float *>(gate), numel);
        auto h_up = mxm_copy_to_host(reinterpret_cast<const float *>(up), numel);
        std::vector<float> h_out(numel);
        swiglu_host_impl(h_out.data(), h_gate.data(), h_up.data(), numel);
        mxm_copy_to_device(reinterpret_cast<float *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_gate = mxm_copy_to_host(reinterpret_cast<const half *>(gate), numel);
        auto h_up = mxm_copy_to_host(reinterpret_cast<const half *>(up), numel);
        std::vector<half> h_out(numel);
        swiglu_host_impl(h_out.data(), h_gate.data(), h_up.data(), numel);
        mxm_copy_to_device(reinterpret_cast<half *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_gate = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(gate), numel);
        auto h_up = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        std::vector<__nv_bfloat16> h_out(numel);
        swiglu_host_impl(h_out.data(), h_gate.data(), h_up.data(), numel);
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

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, size_t numel) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        float g = dev_to_float(gate[i]);
        float u = dev_to_float(up[i]);
        // SwiGLU: out = up * gate * sigmoid(gate) = up * gate / (1 + exp(-gate))
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        out[i] = dev_from_float<T>(u * g * sigmoid_g);
    }
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    const int threads = 256;
    int blocks = (int)((numel + 255) / 256);
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        swiglu_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out), reinterpret_cast<const float *>(gate),
            reinterpret_cast<const float *>(up), numel);
        break;
    case LLAISYS_DTYPE_F16:
        swiglu_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(out), reinterpret_cast<const half *>(gate),
            reinterpret_cast<const half *>(up), numel);
        break;
    case LLAISYS_DTYPE_BF16:
        swiglu_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), reinterpret_cast<const __nv_bfloat16 *>(gate),
            reinterpret_cast<const __nv_bfloat16 *>(up), numel);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
