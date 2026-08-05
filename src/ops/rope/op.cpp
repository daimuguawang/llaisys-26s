// #include "op.hpp"

// namespace llaisys::ops {
// void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
//     TO_BE_IMPLEMENTED();
// }
// } // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include <cmath>
#include <cstdint>

namespace llaisys::ops {

template <typename T>
void rope_cpu_impl(T* out, const T* in, const int64_t* pos_ids, float theta,
                    size_t seq_len, size_t n_heads, size_t head_dim) {
    size_t half_dim = head_dim / 2;

    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        for (size_t h = 0; h < n_heads; h++) {
            const T* x = in + (s * n_heads + h) * head_dim;
            T* y = out + (s * n_heads + h) * head_dim;

            for (size_t j = 0; j < half_dim; j++) {
                // Compute angle: phi = pos / theta^(2j / head_dim)
                float exponent = static_cast<float>(2 * j) / static_cast<float>(head_dim);
                float freqs = static_cast<float>(pos) / std::pow(theta, exponent);
                float cos_val = std::cos(freqs);
                float sin_val = std::sin(freqs);

                float a = utils::cast<float>(x[j]);
                float b = utils::cast<float>(x[j + half_dim]);

                // a' = a*cos - b*sin
                // b' = b*cos + a*sin
                y[j] = utils::cast<T>(a * cos_val - b * sin_val);
                y[j + half_dim] = utils::cast<T>(b * cos_val + a * sin_val);
            }
        }
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");
    ASSERT(out->dtype() == in->dtype(), "RoPE: out and in dtypes must match.");
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be int64.");
    ASSERT(in->ndim() == 3 && out->ndim() == 3, "RoPE: in and out must be 3D [seqlen, nhead, d].");
    ASSERT(pos_ids->ndim() == 1, "RoPE: pos_ids must be 1D.");

    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    ASSERT(out->shape()[0] == seq_len && out->shape()[1] == n_heads && out->shape()[2] == head_dim,
           "RoPE: out shape mismatch.");
    ASSERT(pos_ids->shape()[0] == seq_len, "RoPE: pos_ids length mismatch.");
    ASSERT(head_dim % 2 == 0, "RoPE: head_dim must be even.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        std::byte* out_ptr = out->data();
        const std::byte* in_ptr = in->data();
        const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());

        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            rope_cpu_impl(reinterpret_cast<float*>(out_ptr),
                         reinterpret_cast<const float*>(in_ptr),
                         pos_ptr, theta, seq_len, n_heads, head_dim);
            break;
        case LLAISYS_DTYPE_F16:
            rope_cpu_impl(reinterpret_cast<fp16_t*>(out_ptr),
                         reinterpret_cast<const fp16_t*>(in_ptr),
                         pos_ptr, theta, seq_len, n_heads, head_dim);
            break;
        case LLAISYS_DTYPE_BF16:
            rope_cpu_impl(reinterpret_cast<bf16_t*>(out_ptr),
                         reinterpret_cast<const bf16_t*>(in_ptr),
                         pos_ptr, theta, seq_len, n_heads, head_dim);
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