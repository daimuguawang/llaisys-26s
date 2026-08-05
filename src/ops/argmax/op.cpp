// #include "op.hpp"

// namespace llaisys::ops {
// void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cstdint>
#include <cfloat>

namespace llaisys::ops {

template <typename T>
void argmax_cpu_impl(int64_t* max_idx, T* max_val, const T* vals, size_t numel) {
    float max_val_f = -FLT_MAX;
    int64_t max_idx_v = 0;
    for (size_t i = 0; i < numel; i++) {
        float v = utils::cast<float>(vals[i]);
        if (v > max_val_f) {
            max_val_f = v;
            max_idx_v = static_cast<int64_t>(i);
        }
    }
    *max_val = utils::cast<T>(max_val_f);
    *max_idx = max_idx_v;
}

void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "Argmax: all tensors must be contiguous.");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be int64.");
    ASSERT(max_val->dtype() == vals->dtype(), "Argmax: max_val and vals must have same dtype.");

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        std::byte* val_ptr = max_val->data();
        const std::byte* vals_ptr = vals->data();
        size_t numel = vals->numel();

        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            argmax_cpu_impl(idx_ptr, reinterpret_cast<float*>(val_ptr),
                           reinterpret_cast<const float*>(vals_ptr), numel);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_cpu_impl(idx_ptr, reinterpret_cast<fp16_t*>(val_ptr),
                           reinterpret_cast<const fp16_t*>(vals_ptr), numel);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_cpu_impl(idx_ptr, reinterpret_cast<bf16_t*>(val_ptr),
                           reinterpret_cast<const bf16_t*>(vals_ptr), numel);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
        return;
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
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
