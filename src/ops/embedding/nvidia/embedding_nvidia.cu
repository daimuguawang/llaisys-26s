#include "embedding_nvidia.hpp"

#include "../../../core/llaisys_core.hpp"
#include "../../../utils.hpp"
#include "../../../device/nvidia/cuda_compat.hpp"

namespace llaisys::ops::nvidia {

#ifdef USE_MXMACA
// MXMACA implementation: host-side computation with device memory copy

template <typename T>
void embedding_host_impl(T *out, const int64_t *index, const T *weight,
                         size_t num_indices, size_t embedding_dim) {
    for (size_t r = 0; r < num_indices; r++) {
        int64_t idx = index[r];
        const T *src = weight + (size_t)idx * embedding_dim;
        T *dst = out + r * embedding_dim;
        for (size_t j = 0; j < embedding_dim; j++) {
            dst[j] = src[j];
        }
    }
}

void embedding(std::byte *out, const int64_t *index, const std::byte *weight,
               llaisysDataType_t type, size_t num_indices, size_t embedding_dim) {
    cudaDeviceSynchronize();

    // Copy index to host first to find max index
    auto h_index = mxm_copy_to_host(index, num_indices);

    size_t weight_rows = 0;
    // Determine weight size from indices
    if (num_indices > 0) {
        // Find max index to determine weight size needed
        int64_t max_idx = 0;
        for (size_t r = 0; r < num_indices; r++) {
            if (h_index[r] > max_idx) max_idx = h_index[r];
        }
        weight_rows = max_idx + 1;
    }

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const float *>(weight), weight_rows * embedding_dim);
        std::vector<float> h_out(num_indices * embedding_dim);
        embedding_host_impl(h_out.data(), h_index.data(), h_weight.data(), num_indices, embedding_dim);
        mxm_copy_to_device(reinterpret_cast<float *>(out), h_out.data(), num_indices * embedding_dim);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const half *>(weight), weight_rows * embedding_dim);
        std::vector<half> h_out(num_indices * embedding_dim);
        embedding_host_impl(h_out.data(), h_index.data(), h_weight.data(), num_indices, embedding_dim);
        mxm_copy_to_device(reinterpret_cast<half *>(out), h_out.data(), num_indices * embedding_dim);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        auto h_weight = mxm_copy_to_host(reinterpret_cast<const __nv_bfloat16 *>(weight), weight_rows * embedding_dim);
        std::vector<__nv_bfloat16> h_out(num_indices * embedding_dim);
        embedding_host_impl(h_out.data(), h_index.data(), h_weight.data(), num_indices, embedding_dim);
        mxm_copy_to_device(reinterpret_cast<__nv_bfloat16 *>(out), h_out.data(), num_indices * embedding_dim);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

#else  // NVIDIA CUDA

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

#endif  // USE_MXMACA

} // namespace llaisys::ops::nvidia
