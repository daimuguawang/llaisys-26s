#pragma once
#include "llaisys.h"
#include <cstddef>
#include <cstdint>
namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight, llaisysDataType_t type, size_t num_indices, size_t embedding_dim);
}
