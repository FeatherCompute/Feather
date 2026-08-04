#pragma once

#include "feather_luisa_xir.h"
#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace Feather::Luisa {

struct HostBufferBinding {
    uint32_t binding = 0;
    uint8_t access = 0;
    uint32_t stride = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

struct HostTextureBinding {
    uint32_t binding = 0;
    uint8_t kind = 0;
    uint8_t access = 0;
    uint32_t width = 1;
    uint32_t height = 1;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t pixel_format = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

struct DispatchInputs {
    uint32_t group_x = 1;
    uint32_t group_y = 1;
    uint32_t group_z = 1;
    uint32_t logical_x = 1;
    uint32_t logical_y = 1;
    uint32_t logical_z = 1;
    std::string runtime_directory;
};

struct AdGradientBinding {
    uint32_t source_binding = 0;
    uint32_t element_count = 0;
    uint32_t component_count = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

std::string RuntimeDirectory();

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> buffers, std::span<HostTextureBinding> textures,
              const DispatchInputs& dispatch, const AdInputs* ad_inputs = nullptr,
              std::span<AdGradientBinding> gradients = {}, std::string* error = nullptr);

} // namespace Feather::Luisa
