// #include "op.hpp"

// namespace llaisys::ops {
// void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {

template <typename T>
void linear_cpu_impl(T* out, const T* in, const T* weight, const T* bias,
                     size_t M, size_t N, size_t K) {
    // Y = X * W^T + b
    // out: [M, N], in: [M, K], weight: [N, K], bias: [N] or nullptr
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                float x = utils::cast<float>(in[m * K + k]);
                float w = utils::cast<float>(weight[n * K + k]);
                sum += x * w;
            }
            if (bias != nullptr) {
                sum += utils::cast<float>(bias[n]);
            }
            out[m * N + n] = utils::cast<T>(sum);
        }
    }
}

void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: out, in, weight must be contiguous.");
    ASSERT(out->dtype() == in->dtype() && in->dtype() == weight->dtype(),
           "Linear: dtypes must match.");
    ASSERT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2,
           "Linear: all tensors must be 2D for now.");

    size_t M = in->shape()[0];
    size_t K = in->shape()[1];
    size_t N = weight->shape()[0];

    ASSERT(weight->shape()[1] == K, "Linear: weight shape mismatch.");
    ASSERT(out->shape()[0] == M && out->shape()[1] == N, "Linear: out shape mismatch.");

    bool has_bias = (bias != nullptr);
    if (has_bias) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
        ASSERT(bias->dtype() == out->dtype(), "Linear: bias dtype mismatch.");
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == N, "Linear: bias shape mismatch.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte* out_ptr = out->data();
        const std::byte* in_ptr = in->data();
        const std::byte* w_ptr = weight->data();
        const std::byte* b_ptr = has_bias ? bias->data() : nullptr;

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            linear_cpu_impl(reinterpret_cast<float*>(out_ptr),
                           reinterpret_cast<const float*>(in_ptr),
                           reinterpret_cast<const float*>(w_ptr),
                           has_bias ? reinterpret_cast<const float*>(b_ptr) : nullptr,
                           M, N, K);
            break;
        case LLAISYS_DTYPE_F16:
            linear_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr),
                           reinterpret_cast<const fp16_t*>(in_ptr),
                           reinterpret_cast<const fp16_t*>(w_ptr),
                           has_bias ? reinterpret_cast<const fp16_t*>(b_ptr) : nullptr,
                           M, N, K);
            break;
        case LLAISYS_DTYPE_BF16:
            linear_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr),
                           reinterpret_cast<const bf16_t*>(in_ptr),
                           reinterpret_cast<const bf16_t*>(w_ptr),
                           has_bias ? reinterpret_cast<const bf16_t*>(b_ptr) : nullptr,
                           M, N, K);
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
        nvidia::linear(out->data(), in->data(), weight->data(),
                       has_bias ? bias->data() : nullptr, out->dtype(), M, N, K,
                       llaisys::core::context().runtime().stream());
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
