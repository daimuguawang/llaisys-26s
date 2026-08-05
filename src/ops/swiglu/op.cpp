// #include "op.hpp"

// namespace llaisys::ops {
// void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>

namespace llaisys::ops {

template <typename T>
void swiglu_cpu_impl(T* out, const T* gate, const T* up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        // SwiGLU: out = up * gate * sigmoid(gate) = up * gate / (1 + exp(-gate))
        float sigmoid_g = 1.0f / (1.0f + std::exp(-g));
        out[i] = utils::cast<T>(u * g * sigmoid_g);
    }
}

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "SwiGLU: all tensors must be contiguous.");
    ASSERT(out->dtype() == gate->dtype() && gate->dtype() == up->dtype(),
           "SwiGLU: dtypes must match.");
    ASSERT(out->shape() == gate->shape() && gate->shape() == up->shape(),
           "SwiGLU: all tensors must have same shape.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte* out_ptr = out->data();
        const std::byte* gate_ptr = gate->data();
        const std::byte* up_ptr = up->data();
        size_t numel = out->numel();

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            swiglu_cpu_impl(reinterpret_cast<float*>(out_ptr),
                           reinterpret_cast<const float*>(gate_ptr),
                           reinterpret_cast<const float*>(up_ptr), numel);
            break;
        case LLAISYS_DTYPE_F16:
            swiglu_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr),
                           reinterpret_cast<const fp16_t*>(gate_ptr),
                           reinterpret_cast<const fp16_t*>(up_ptr), numel);
            break;
        case LLAISYS_DTYPE_BF16:
            swiglu_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr),
                           reinterpret_cast<const bf16_t*>(gate_ptr),
                           reinterpret_cast<const bf16_t*>(up_ptr), numel);
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