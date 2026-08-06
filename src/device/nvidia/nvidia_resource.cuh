#pragma once

#include "../device_resource.hpp"
#include "llaisys.h"

#include "cuda_compat.hpp"

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
public:
    Resource(int device_id);
    ~Resource();

private:
    cublasHandle_t _cublas_handle;
};

// Lazily create / fetch a thread-local cuBLAS/rocBLAS handle bound to the current
// device and the given stream. The stream is (re)bound on every call so
// that BLAS work is ordered against the active runtime stream.
cublasHandle_t getCublasHandle(llaisysStream_t stream);
} // namespace llaisys::device::nvidia
