#pragma once

// CUDA/MXMACA Compatibility Header for MetaX C500
// When compiled with MXMACA, maps CUDA API to MXMACA mc_* API.

#ifdef USE_MXMACA

#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_types.h>
#include <mcblas/mcblas.h>
#include <cstdlib>
#include <cstring>
#include <vector>

// Type mappings
using half = mcblas_half;
using __nv_bfloat16 = mcblas_bfloat16;

// Memcpy kind type mapping
using cudaMemcpyKind = mcMemcpyKind;
#define cudaMemcpyHostToHost mcMemcpyHostToHost
#define cudaMemcpyHostToDevice mcMemcpyHostToDevice
#define cudaMemcpyDeviceToHost mcMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice mcMemcpyDeviceToDevice
#define cudaMemcpyDefault mcMemcpyDefault

// Function mappings: mc_* -> cuda* names for code compatibility
#define cudaMalloc mcMalloc
#define cudaFree mcFree
#define cudaMallocHost mcMallocHost
#define cudaFreeHost mcFreeHost
#define cudaMemcpy mcMemcpy
#define cudaMemcpyAsync mcMemcpyAsync
#define cudaSetDevice mcSetDevice
#define cudaGetDevice mcGetDevice
#define cudaGetDeviceCount mcGetDeviceCount
#define cudaDeviceSynchronize mcDeviceSynchronize
#define cudaStream_t mcStream_t
#define cudaStreamCreate mcStreamCreate
#define cudaStreamDestroy mcStreamDestroy
#define cudaStreamSynchronize mcStreamSynchronize
#define cudaSuccess mcSuccess

// cuBLAS -> mcBLAS mappings
#define cublasHandle_t mcblasHandle_t
#define cublasCreate mcblasCreate
#define cublasDestroy mcblasDestroy
#define cublasSetStream mcblasSetStream
#define cublasGemmEx mcblasGemmEx
#define cublasSgemm mcblasSgemm
#define cublasHgemm mcblasHgemm
#define CUBLAS_OP_T MCBLAS_OP_T
#define CUBLAS_OP_N MCBLAS_OP_N
#define CUBLAS_COMPUTE_32F MCBLAS_COMPUTE_32F
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MCBLAS_GEMM_DEFAULT

// Data type mappings
using cudaDataType_t = macaDataType_t;
#define CUDA_R_32F MACA_R_32F
#define CUDA_R_16F MACA_R_16F
#define CUDA_R_16BF MACA_R_16BF

// Float conversion functions (MXMACA uses native half/bfloat16)
#define __half2float(v) static_cast<float>((v))
#define __float2half(v) mcblas_half(v)
#define __bfloat162float(v) static_cast<float>((v))
#define __float2bfloat16(v) mcblas_bfloat16(v)

// Warp size for C500 (64 instead of 32)
#define WARP_SIZE 64

// Warp shuffle compatibility:
// C500 uses __shfl_down without mask parameter
#define __shfl_down_sync(mask, val, offset) __shfl_down(val, offset)

// MXMACA host-side utility: copy device data to host buffer
template <typename T>
std::vector<T> mxm_copy_to_host(const T* dev_ptr, size_t count) {
    std::vector<T> host_buf(count);
    mcMemcpy(host_buf.data(), dev_ptr, count * sizeof(T), mcMemcpyDeviceToHost);
    return host_buf;
}

// MXMACA host-side utility: copy host buffer to device
template <typename T>
void mxm_copy_to_device(T* dev_ptr, const T* host_ptr, size_t count) {
    mcMemcpy(dev_ptr, host_ptr, count * sizeof(T), mcMemcpyHostToDevice);
}

// MXMACA host-side utility: copy device data to host byte buffer
inline std::vector<std::byte> mxm_bytes_to_host(const std::byte* dev_ptr, size_t bytes) {
    std::vector<std::byte> host_buf(bytes);
    mcMemcpy(host_buf.data(), dev_ptr, bytes, mcMemcpyDeviceToHost);
    return host_buf;
}

// MXMACA host-side utility: copy host byte buffer to device
inline void mxm_bytes_to_device(std::byte* dev_ptr, const std::byte* host_ptr, size_t bytes) {
    mcMemcpy(dev_ptr, host_ptr, bytes, mcMemcpyHostToDevice);
}

#else  // NVIDIA CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

#endif  // USE_MXMACA
