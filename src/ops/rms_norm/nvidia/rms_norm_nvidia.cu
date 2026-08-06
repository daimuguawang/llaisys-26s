#include "rms_norm_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
void rms_norm_host_impl(T *out, const T *in, const T *weight,
                        float eps, size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; row++) {
        const T *x = in + row * cols;
        T *y = out + row * cols;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float v = static_cast<float>(x[j]);
            sum_sq += v * v;
        }
        float mean_sq = sum_sq / static_cast<float>(cols);
        float rms = 1.0f / std::sqrt(mean_sq + eps);

        for (size_t j = 0; j < cols; j++) {
            float v = static_cast<float>(x[j]);
            float w = static_cast<float>(weight[j]);
            if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>) {
                y[j] = static_cast<T>(v * rms * w);
            } else {
                y[j] = v * rms * w;
            }
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight,
              llaisysDataType_t type, float eps, size_t rows, size_t cols) {
    cudaDeviceSynchronize();
    size_t numel = rows * cols;

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const float *>(in), numel);
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const float *>(weight), cols);
        std::vector<float> h_out(numel);
        rms_norm_host_impl(h_out.data(), h_in.data(), h_weight.data(), eps, rows, cols);
        mxm_copy_to_device(reinterpret_cast<float *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const half *>(in), numel);
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const half *>(weight), cols);
        std::vector<half> h_out(numel);
        rms_norm_host_impl(h_out.data(), h_in.data(), h_weight.data(), eps, rows, cols);
        mxm_copy_to_device(reinterpret_cast<half *>(out), h_out.data(), numel);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_in = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(in), numel);
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(weight), cols);
        std::vector<__nv_bfloat16> h_out(numel);
        rms_norm_host_impl(h_out.data(), h_in.data(), h_weight.data(), eps, rows, cols);
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
    int warp_id = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int num_warps = (stride + WARP_SIZE - 1) / WARP_SIZE;

    float sum_sq = 0.0f;
    for (size_t j = (size_t)tid; j < cols; j += (size_t)stride) {
        float v = dev_to_float(x[j]);
        sum_sq += v * v;
    }

    // Warp reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float warp_sum[WARP_SIZE];
    if (lane == 0) warp_sum[warp_id] = sum_sq;
    __syncthreads();

    // Final reduction in warp 0
    if (warp_id == 0) {
        float total = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
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
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

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

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
