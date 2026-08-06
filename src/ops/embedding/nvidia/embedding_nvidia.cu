#include "embedding_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_kernel(T *out, const int64_t *index, const T *weight,
                                 size_t num_indices, size_t embedding_dim) {
    size_t row = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;
    for (size_t r = row; r < num_indices; r += gridDim.x) {
        int64_t idx = index[r];
        const T *src = weight + (size_t)idx * embedding_dim;
        T *dst = out + r * embedding_dim;
        for (size_t j = tid; j < embedding_dim; j += stride) {
            dst[j] = src[j];
        }
    }
}

void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embedding_dim) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(llaisys::core::context().runtime().stream());

    const int threads = 256;
    int blocks = (int)num_indices;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    switch (type) {
    case LLAISYS_DTYPE_F32:
        embedding_kernel<float><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float *>(out), index,
            reinterpret_cast<const float *>(weight), num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_F16:
        embedding_kernel<half><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<half *>(out), index,
            reinterpret_cast<const half *>(weight), num_indices, embedding_dim);
        break;
    case LLAISYS_DTYPE_BF16:
        embedding_kernel<__nv_bfloat16><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(out), index,
            reinterpret_cast<const __nv_bfloat16 *>(weight), num_indices, embedding_dim);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
