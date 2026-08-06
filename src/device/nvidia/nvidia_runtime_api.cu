#include "../runtime_api.hpp"

#include "cuda_compat.hpp"
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

void setDevice(int device_id) {
    cudaSetDevice(device_id);
}

void deviceSynchronize() {
    cudaDeviceSynchronize();
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
    }
}

void streamSynchronize(llaisysStream_t stream) {
    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
}

void *mallocDevice(size_t size) {
    void *ptr = nullptr;
    if (size == 0) return nullptr;
    cudaMalloc(&ptr, size);
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

void *mallocHost(size_t size) {
    void *ptr = nullptr;
    if (size == 0) return nullptr;
    // Pinned (page-locked) host memory for fast async transfers.
    cudaMallocHost(&ptr, size);
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

static cudaMemcpyKind toCudaKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H: return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D: return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H: return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D: return cudaMemcpyDeviceToDevice;
    default: return cudaMemcpyDefault;
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaMemcpy(dst, src, size, toCudaKind(kind));
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaStream_t s = (stream != nullptr) ? reinterpret_cast<cudaStream_t>(stream) : 0;
    cudaMemcpyAsync(dst, src, size, toCudaKind(kind), s);
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
