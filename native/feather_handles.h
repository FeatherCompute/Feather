#ifndef FEATHER_HANDLES_H
#define FEATHER_HANDLES_H

#include "feather_c_api.h"

enum FeHandleKind {
    FE_HANDLE_CONTEXT = 1,
    FE_HANDLE_BUFFER = 2,
    FE_HANDLE_TEXTURE = 3,
    FE_HANDLE_SAMPLER = 4,
    FE_HANDLE_KERNEL = 5,
    FE_HANDLE_GRAPHICS_PIPELINE = 6,
    FE_HANDLE_AD_KERNEL = 7,
    FE_HANDLE_TENSOR = 8
};

#endif
