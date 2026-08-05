#include "llaisys/models/qwen2.h"
#include "../models/qwen2/model.hpp"
#include <cstdlib>
#include <cstring>
using namespace llaisys::models;
__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        (void)device_ids;
        (void)ndevice;
        auto *model = new Qwen2Model(meta, device, device_ids ? device_ids[0] : 0);
        model->initBuffers();
        return reinterpret_cast<struct LlaisysQwen2Model *>(model);
    }
    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model *model) {
        if (model) {
            delete reinterpret_cast<Qwen2Model *>(model);
        }
    }
    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model *model) {
        auto *m = reinterpret_cast<Qwen2Model *>(model);
        return m->getWeights();
    }
    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        auto *m = reinterpret_cast<Qwen2Model *>(model);
        return m->infer(token_ids, ntoken);
    }
}
