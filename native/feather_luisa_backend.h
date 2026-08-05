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
    uint64_t shader_cache_key = 0;
    std::string backend_name = "vk";
    std::string runtime_directory;
};

struct AdGradientBinding {
    uint32_t source_binding = 0;
    uint32_t element_count = 0;
    uint32_t component_count = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

std::string RuntimeDirectory();

// Owns the Luisa Context/Device/Stream lifetime for Feather's native context.
// Shutdown is called from context/runtime teardown; Abandon is used on process
// exit when C++ destructors may run after the dynamic backend has unloaded.
void Shutdown();
void Abandon() noexcept;

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> buffers, std::span<HostTextureBinding> textures,
              const DispatchInputs& dispatch, const AdInputs* ad_inputs = nullptr,
              std::span<AdGradientBinding> gradients = {}, std::string* error = nullptr);

} // namespace Feather::Luisa
