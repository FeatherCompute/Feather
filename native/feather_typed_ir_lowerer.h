#pragma once

#include "feather_typed_ir.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <IR/Module.h>

namespace Feather::TypedIR {

struct ResourceInfo {
    uint32_t binding = 0;
    uint8_t kind = 0;
    uint8_t access = 0;
    std::string name;
    std::string element_type;
    GPU::Runtime::PixelFormat texture_format = GPU::Runtime::PixelFormat::RGBA8;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    bool sampled = false;
};

struct PushConstantInfo {
    uint32_t binding = 0;
    void* data = nullptr;
    size_t size = 0;
    size_t alignment = 0;
};

struct LoweringInputs {
    uint8_t shader_kind = 0;
    int32_t group_x = 1;
    int32_t group_y = 1;
    int32_t group_z = 1;
    bool bounds_check = false;
    int32_t logical_x = 0;
    int32_t logical_y = 0;
    int32_t logical_z = 0;
    int32_t* logical_x_data = nullptr;
    int32_t* logical_y_data = nullptr;
    int32_t* logical_z_data = nullptr;
    std::vector<ResourceInfo> resources;
    std::vector<PushConstantInfo> push_constants;
};

std::unique_ptr<GPU::IR::Module> TryLowerToEasyGpuModule(
    const Module& typed, const LoweringInputs& inputs, std::string* error = nullptr);

} // namespace Feather::TypedIR
