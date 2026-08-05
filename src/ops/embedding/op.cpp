// #include "op.hpp"

// namespace llaisys::ops {
// void embedding(tensor_t out, tensor_t index, tensor_t weight) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cstdint>
#include <cstring>

namespace llaisys::ops {

template <typename T>
void embedding_cpu_impl(T* out, const int64_t* index, const T* weight,
                        size_t num_indices, size_t embedding_dim) {
    for (size_t i = 0; i < num_indices; i++) {
        int64_t idx = index[i];
        const T* src = weight + idx * embedding_dim;
        T* dst = out + i * embedding_dim;
        for (size_t j = 0; j < embedding_dim; j++) {
            dst[j] = src[j];
        }
    }
}

void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be int64.");
    ASSERT(out->dtype() == weight->dtype(), "Embedding: out and weight must have same dtype.");
    ASSERT(index->ndim() == 1, "Embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "Embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "Embedding: out must be 2D.");

    size_t num_indices = index->shape()[0];
    size_t embedding_dim = weight->shape()[1];

    ASSERT(out->shape()[0] == num_indices, "Embedding: out shape mismatch.");
    ASSERT(out->shape()[1] == embedding_dim, "Embedding: out shape mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const int64_t* idx_ptr = reinterpret_cast<const int64_t*>(index->data());
        std::byte* out_ptr = out->data();
        const std::byte* w_ptr = weight->data();

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            embedding_cpu_impl(reinterpret_cast<float*>(out_ptr), idx_ptr,
                              reinterpret_cast<const float*>(w_ptr), num_indices, embedding_dim);
            break;
        case LLAISYS_DTYPE_F16:
            embedding_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr), idx_ptr,
                              reinterpret_cast<const fp16_t*>(w_ptr), num_indices, embedding_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            embedding_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr), idx_ptr,
                              reinterpret_cast<const bf16_t*>(w_ptr), num_indices, embedding_dim);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
        return;
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops