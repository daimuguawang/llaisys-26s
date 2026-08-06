#include "linear_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../device/nvidia/nvidia_resource.cuh"
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

// out[m, n] += bias[n]. 2D grid over (M, N). Computed in float for F16/BF16 to
// match the CPU path (cuBLAS already overwrote `out` with beta=0).
template <typename T>
__global__ void bias_add_kernel(T *out, const T *bias, size_t M, size_t N) {
    size_t n = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t m = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float v = dev_to_float(out[m * N + n]);
    float b = dev_to_float(bias[n]);
    out[m * N + n] = dev_from_float<T>(v + b);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t type, size_t M, size_t N, size_t K,
            llaisysStream_t stream) {
    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
    cublasHandle_t handle = llaisys::device::nvidia::getCublasHandle(stream);

    cudaDataType_t cuda_type;
    switch (type) {
    case LLAISYS_DTYPE_F32:
        cuda_type = CUDA_R_32F;
        break;
    case LLAISYS_DTYPE_F16:
        cuda_type = CUDA_R_16F;
        break;
    case LLAISYS_DTYPE_BF16:
        cuda_type = CUDA_R_16BF;
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
        return;
    }

    // Y = X * W^T + b. X:[M,K] (row-major), W:[N,K] (row-major), Y:[M,N].
    // cuBLAS is column-major, so we compute Y^T = W * X^T via:
    //   op(A)=W (CUBLAS_OP_T, lda=K), op(B)=X (CUBLAS_OP_N, ldb=K), C=Y (ldc=N)
    // giving a (N x M) result written into Y interpreted row-major as [M, N].
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 (int)N, (int)M, (int)K,
                 &alpha,
                 weight, cuda_type, (int)K,
                 in, cuda_type, (int)K,
                 &beta,
                 out, cuda_type, (int)N,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    if (bias != nullptr) {
        dim3 block(16, 16);
        dim3 grid((unsigned)((N + 15) / 16), (unsigned)((M + 15) / 16));
        switch (type) {
        case LLAISYS_DTYPE_F32:
            bias_add_kernel<float><<<grid, block, 0, cu_stream>>>(
                reinterpret_cast<float *>(out),
                reinterpret_cast<const float *>(bias), M, N);
            break;
        case LLAISYS_DTYPE_F16:
            bias_add_kernel<half><<<grid, block, 0, cu_stream>>>(
                reinterpret_cast<half *>(out),
                reinterpret_cast<const half *>(bias), M, N);
            break;
        case LLAISYS_DTYPE_BF16:
            bias_add_kernel<__nv_bfloat16><<<grid, block, 0, cu_stream>>>(
                reinterpret_cast<__nv_bfloat16 *>(out),
                reinterpret_cast<const __nv_bfloat16 *>(bias), M, N);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(type);
            break;
        }
    }
}

} // namespace llaisys::ops::nvidia
