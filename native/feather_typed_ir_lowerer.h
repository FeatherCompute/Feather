#pragma once

#include "feather_typed_ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace Feather::TypedIR {

struct ResourceInfo {
    uint32_t binding = 0;
    uint8_t kind = 0;
    uint8_t access = 0;
    std::string name;
    std::string element_type;
    uint32_t element_count = 0;
    uint32_t texture_format = 3u;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    bool sampled = false;
    uint32_t sampler_min_filter = 0;
    uint32_t sampler_mag_filter = 0;
    uint32_t sampler_mipmap_mode = 0;
    uint32_t sampler_address_u = 0;
    uint32_t sampler_address_v = 0;
    uint32_t sampler_address_w = 0;
    bool sampler_anisotropy = false;
};

struct PushConstantInfo {
    uint32_t binding = 0;
    void* data = nullptr;
    size_t size = 0;
    size_t alignment = 0;
};

struct GraphicsBlendInfo {
    bool enable = false;
    uint32_t src_color = 1;
    uint32_t dst_color = 0;
    uint32_t color_op = 0;
    uint32_t src_alpha = 1;
    uint32_t dst_alpha = 0;
    uint32_t alpha_op = 0;
    uint32_t write_mask = 15;
};

struct GraphicsColorTargetInfo {
    uint32_t binding = NoIndex;
    uint32_t return_field = NoIndex;
    GraphicsBlendInfo blend;
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
    bool enable_fused_multiply_add = false;
    int32_t* logical_x_data = nullptr;
    int32_t* logical_y_data = nullptr;
    int32_t* logical_z_data = nullptr;
    uint32_t stage_input_binding = NoIndex;
    uint32_t stage_output_binding = NoIndex;
    uint32_t stage_coverage_binding = NoIndex;
    uint32_t graphics_vertex_count = 0;
    uint32_t graphics_first_instance = 0;
    uint32_t graphics_sample_count = 1;
    uint32_t graphics_sample_index = 0;
    std::vector<GraphicsColorTargetInfo> graphics_color_targets;
    std::vector<ResourceInfo> resources;
    std::vector<PushConstantInfo> push_constants;
    std::vector<std::vector<unsigned char>> push_constant_storage;
    bool dynamic_push_constants = false;
};

} // namespace Feather::TypedIR
