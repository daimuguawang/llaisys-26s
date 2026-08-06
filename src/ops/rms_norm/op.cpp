// #include "op.hpp"

// namespace llaisys::ops {
// void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.hpp"
#endif

namespace llaisys::ops {

template <typename T>
void rms_norm_cpu_impl(T* out, const T* in, const T* weight, float eps,
                       size_t rows, size_t cols) {
    for (size_t row = 0; row < rows; row++) {
        const T* x = in + row * cols;
        T* y = out + row * cols;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(x[j]);
            sum_sq += v * v;
        }
        float mean_sq = sum_sq / static_cast<float>(cols);
        float rms = 1.0f / std::sqrt(mean_sq + eps);

        // Normalize and scale
        for (size_t j = 0; j < cols; j++) {
            float v = utils::cast<float>(x[j]);
            float w = utils::cast<float>(weight[j]);
            y[j] = utils::cast<T>(v * rms * w);
        }
    }
}

void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");
    ASSERT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(),
           "RMSNorm: dtypes must match.");
    ASSERT(in->ndim() == 2 && out->ndim() == 2, "RMSNorm: in and out must be 2D.");
    ASSERT(weight->ndim() == 1, "RMSNorm: weight must be 1D.");

    size_t rows = in->shape()[0];
    size_t cols = in->shape()[1];

    ASSERT(out->shape()[0] == rows && out->shape()[1] == cols, "RMSNorm: out shape mismatch.");
    ASSERT(weight->shape()[0] == cols, "RMSNorm: weight shape mismatch.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte* out_ptr = out->data();
        const std::byte* in_ptr = in->data();
        const std::byte* w_ptr = weight->data();

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            rms_norm_cpu_impl(reinterpret_cast<float*>(out_ptr),
                             reinterpret_cast<const float*>(in_ptr),
                             reinterpret_cast<const float*>(w_ptr),
                             eps, rows, cols);
            break;
        case LLAISYS_DTYPE_F16:
            rms_norm_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr),
                             reinterpret_cast<const fp16_t*>(in_ptr),
                             reinterpret_cast<const fp16_t*>(w_ptr),
                             eps, rows, cols);
            break;
        case LLAISYS_DTYPE_BF16:
            rms_norm_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr),
                             reinterpret_cast<const bf16_t*>(in_ptr),
                             reinterpret_cast<const bf16_t*>(w_ptr),
                             eps, rows, cols);
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
        nvidia::rms_norm(out->data(), in->data(), weight->data(), out->dtype(), eps, rows, cols);
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops