#pragma once

#include "feather_luisa_xir.h"
#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Feather::Luisa {

#if defined(__APPLE__)
inline constexpr std::string_view DefaultBackendName = "metal";
#else
inline constexpr std::string_view DefaultBackendName = "vk";
#endif

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
    std::string backend_name = std::string{DefaultBackendName};
    std::string runtime_directory;
};

struct RasterDispatchInputs {
    uint32_t vertex_count = 3;
    uint32_t viewport_x = 0;
    uint32_t viewport_y = 0;
    uint32_t viewport_width = 1;
    uint32_t viewport_height = 1;
    uint32_t scissor_x = 0;
    uint32_t scissor_y = 0;
    uint32_t scissor_width = 1;
    uint32_t scissor_height = 1;
    uint32_t cull_mode = 0;
    uint32_t front_face = 0;
    uint32_t depth_test = 0;
    uint32_t depth_write = 0;
    uint32_t depth_compare = 1;
    uint32_t depth_clamp = 0;
    uint32_t stencil_test = 0;
    uint32_t stencil_front_fail = 0;
    uint32_t stencil_front_pass = 0;
    uint32_t stencil_front_depth_fail = 0;
    uint32_t stencil_front_compare = 7;
    uint32_t stencil_back_fail = 0;
    uint32_t stencil_back_pass = 0;
    uint32_t stencil_back_depth_fail = 0;
    uint32_t stencil_back_compare = 7;
    uint32_t stencil_read_mask = ~0u;
    uint32_t stencil_write_mask = ~0u;
    uint32_t stencil_reference = 0;
    uint32_t clear_depth = 0;
    uint32_t clear_stencil = 0;
    float clear_depth_value = 1.0f;
    uint32_t clear_color = 0;
    float clear_color_r = 0.0f;
    float clear_color_g = 0.0f;
    float clear_color_b = 0.0f;
    float clear_color_a = 1.0f;
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

// Experimental compute triangle assembly/raster stage between generated graphics FEIR stages.
bool DispatchVerticalRaster(HostBufferBinding vertices, HostTextureBinding target,
                            HostTextureBinding* depth, const RasterDispatchInputs& raster,
                            const DispatchInputs& dispatch,
                            std::vector<unsigned char>* fragment_varyings,
                            std::vector<unsigned char>* fragment_coverage,
                            std::string* error = nullptr);

} // namespace Feather::Luisa
