#include "nvidia_resource.cuh"

#include "cuda_compat.hpp"

namespace llaisys::device::nvidia {

Resource::Resource(int device_id) : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    // A handle may also be created here if a per-resource handle is desired,
    // but inference currently uses the thread-local handle from getCublasHandle().
    _cublas_handle = nullptr;
}

Resource::~Resource() {
    if (_cublas_handle != nullptr) {
        cublasDestroy(_cublas_handle);
        _cublas_handle = nullptr;
    }
}

cublasHandle_t getCublasHandle(llaisysStream_t stream) {
    thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    cudaStream_t s = (stream != nullptr) ? reinterpret_cast<cudaStream_t>(stream) : 0;
    cublasSetStream(handle, s);
    return handle;
}

} // namespace llaisys::device::nvidia
