#ifndef FEATHER_C_API_H
#define FEATHER_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <cstdbool>
extern "C" {
#else
#include <stdbool.h>
#endif

#if defined(_WIN32)
#if defined(FEATHER_BUILDING_DLL)
#define FE_API __declspec(dllexport)
#else
#define FE_API __declspec(dllimport)
#endif
#else
#define FE_API __attribute__((visibility("default")))
#endif

typedef uint64_t FeContextHandle;
typedef uint64_t FeBufferHandle;
typedef uint64_t FeTextureHandle;
typedef uint64_t FeSamplerHandle;
typedef uint64_t FeKernelHandle;
typedef uint64_t FeGraphicsPipelineHandle;
typedef uint64_t FeADKernelHandle;
typedef uint64_t FeTensorHandle;
typedef uint64_t FeWindowHandle;
typedef uint64_t FeTexturePresenterHandle;

typedef enum FeResult {
    FE_OK = 0,
    FE_ERROR_UNKNOWN = 1,
    FE_ERROR_INVALID_ARGUMENT = 2,
    FE_ERROR_INVALID_HANDLE = 3,
    FE_ERROR_BACKEND_UNAVAILABLE = 4,
    FE_ERROR_SHADER_COMPILE_FAILED = 5,
    FE_ERROR_OUT_OF_MEMORY = 6,
    FE_ERROR_UNSUPPORTED = 7
} FeResult;

typedef enum FeDispatchPath {
    FE_DISPATCH_PATH_NONE = 0,
    FE_DISPATCH_PATH_TYPED_EASYGPU = 1,
    FE_DISPATCH_PATH_CPU_REFERENCE_FALLBACK = 2,
    FE_DISPATCH_PATH_GRAPHICS_FALLBACK = 3,
    FE_DISPATCH_PATH_REJECTED = 4
} FeDispatchPath;

typedef struct FeBackendCaps {
    uint32_t backend_type;
    uint32_t max_work_group_size_x;
    uint32_t max_work_group_size_y;
    uint32_t max_work_group_size_z;
    uint32_t supports_graphics;
    uint32_t supports_ad;
    uint32_t supports_nn;
    uint32_t supports_window;
    uint32_t supports_depth_clamp;
    uint32_t supports_non_fill_polygon_mode;
} FeBackendCaps;

typedef struct FeWindowDesc {
    uint32_t width;
    uint32_t height;
    const char* title;
    uint32_t resizable;
    uint32_t visible;
    uint32_t vsync;
    uint32_t high_dpi;
    uint32_t center_on_create;
} FeWindowDesc;

typedef struct FeWindowEvent {
    uint32_t kind;
    uint32_t key;
    uint32_t mouse_button;
    uint32_t modifiers;
    uint32_t pressed;
    int32_t x;
    int32_t y;
    int32_t dx;
    int32_t dy;
    float scroll_x;
    float scroll_y;
    uint32_t width;
    uint32_t height;
    uint32_t codepoint;
} FeWindowEvent;

typedef struct FeBufferDesc {
    uint64_t size_in_bytes;
    uint32_t mode;
    uint32_t element_stride;
} FeBufferDesc;

typedef struct FeTexture2DDesc {
    uint32_t width;
    uint32_t height;
    uint32_t mip_levels;
    uint32_t pixel_format;
    uint32_t access;
} FeTexture2DDesc;

typedef struct FeTexture3DDesc {
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t mip_levels;
    uint32_t pixel_format;
    uint32_t access;
} FeTexture3DDesc;

typedef struct FeSamplerDesc {
    uint32_t min_filter;
    uint32_t mag_filter;
    uint32_t mipmap_mode;
    uint32_t address_u;
    uint32_t address_v;
    uint32_t address_w;
    float mip_lod_bias;
    float min_lod;
    float max_lod;
    uint32_t anisotropy_enable;
    float max_anisotropy;
    uint32_t compare_enable;
    uint32_t compare_op;
    uint32_t border_color;
} FeSamplerDesc;

typedef struct FeGraphicsStencilFaceDesc {
    uint32_t fail_op;
    uint32_t pass_op;
    uint32_t depth_fail_op;
    uint32_t compare_op;
} FeGraphicsStencilFaceDesc;

typedef struct FeGraphicsColorBlendAttachmentDesc {
    uint32_t blend_enable;
    uint32_t src_color;
    uint32_t dst_color;
    uint32_t color_op;
    uint32_t src_alpha;
    uint32_t dst_alpha;
    uint32_t alpha_op;
    uint32_t write_mask;
} FeGraphicsColorBlendAttachmentDesc;

typedef struct FeKernelCreateDesc {
    const void* ir_data;
    uint64_t ir_size;
    const char* debug_name;
    bool auto_diff;
    bool bounds_check;
} FeKernelCreateDesc;

typedef struct FeGraphicsPipelineCreateDesc {
    const void* ir_data;
    uint64_t ir_size;
    const void* vertex_ir_data;
    uint64_t vertex_ir_size;
    const void* fragment_ir_data;
    uint64_t fragment_ir_size;
    uint32_t topology;
    uint32_t sample_count;
    uint32_t color_attachment_count;
    uint32_t depth_test;
    uint32_t depth_write;
    uint32_t depth_compare;
    uint32_t stencil_test;
    FeGraphicsStencilFaceDesc stencil_front;
    FeGraphicsStencilFaceDesc stencil_back;
    uint32_t stencil_read_mask;
    uint32_t stencil_write_mask;
    uint32_t stencil_reference;
    uint32_t blend_enable;
    uint32_t blend_src_color;
    uint32_t blend_dst_color;
    uint32_t blend_color_op;
    uint32_t blend_src_alpha;
    uint32_t blend_dst_alpha;
    uint32_t blend_alpha_op;
    uint32_t blend_write_mask;
    uint32_t color_blend_attachment_count;
    FeGraphicsColorBlendAttachmentDesc color_blend_attachments[8];
    uint32_t cull_mode;
    uint32_t front_face;
    uint32_t polygon_mode;
    uint32_t depth_clamp;
    const char* debug_name;
} FeGraphicsPipelineCreateDesc;

typedef struct FeGraphicsDrawDesc {
    const FeTextureHandle* color_targets;
    uint32_t color_target_count;
    FeTextureHandle depth_target;
    uint32_t count;
    FeBufferHandle index_buffer;
    uint32_t indexed;
    uint32_t wait;
    uint32_t viewport_enabled;
    uint32_t viewport_x;
    uint32_t viewport_y;
    uint32_t viewport_width;
    uint32_t viewport_height;
    uint32_t scissor_enabled;
    uint32_t scissor_x;
    uint32_t scissor_y;
    uint32_t scissor_width;
    uint32_t scissor_height;
    uint32_t clear_depth;
    float clear_depth_value;
    uint32_t depth_load_op;
} FeGraphicsDrawDesc;

typedef struct FeProfilerQueryResult {
    uint64_t count;
    double min_time_ms;
    double max_time_ms;
    double average_time_ms;
    double total_time_ms;
} FeProfilerQueryResult;

typedef struct FeADGradientInfo {
    char name[128];
    char resource_name[128];
    char element_type[64];
    char easygpu_name[64];
    uint32_t source_binding;
    uint32_t gradient_binding;
    uint32_t element_count;
    uint32_t element_stride;
    uint64_t byte_size;
    uint32_t component_count;
    uint32_t reserved;
} FeADGradientInfo;

FE_API FeResult fe_context_get_default(FeContextHandle* out_context);
FE_API FeResult fe_context_initialize(FeContextHandle context);
FE_API FeResult fe_context_shutdown(FeContextHandle context);
FE_API FeResult fe_context_get_backend_type(FeContextHandle context, uint32_t* out_backend);
FE_API FeResult fe_context_get_caps(FeContextHandle context, FeBackendCaps* out_caps);
FE_API FeResult fe_get_last_error(char* buffer, size_t buffer_size, size_t* out_required_size);
FE_API FeResult fe_runtime_shutdown(void);
FE_API FeResult fe_runtime_process_exit(void);

FE_API FeResult fe_window_create(const FeWindowDesc* desc, FeWindowHandle* out_window);
FE_API FeResult fe_window_destroy(FeWindowHandle window);
FE_API FeResult fe_window_is_open(FeWindowHandle window, bool* out_is_open);
FE_API FeResult fe_window_close(FeWindowHandle window);
FE_API FeResult fe_window_poll_events(FeWindowHandle window);
FE_API FeResult fe_window_wait_events(FeWindowHandle window);
FE_API FeResult fe_window_poll_event(FeWindowHandle window, FeWindowEvent* out_event, bool* out_has_event);
FE_API FeResult fe_window_get_size(FeWindowHandle window, uint32_t* out_width, uint32_t* out_height);
FE_API FeResult fe_window_set_title(FeWindowHandle window, const char* title);
FE_API FeResult fe_window_set_vsync(FeWindowHandle window, bool enabled);
FE_API FeResult fe_window_is_key_down(FeWindowHandle window, uint32_t key, bool* out_is_down);
FE_API FeResult fe_window_is_mouse_down(FeWindowHandle window, uint32_t mouse_button, bool* out_is_down);
FE_API FeResult fe_window_get_mouse_position(FeWindowHandle window, int32_t* out_x, int32_t* out_y);
FE_API FeResult fe_window_get_mouse_scroll(FeWindowHandle window, float* out_x, float* out_y);
FE_API FeResult fe_window_present_pixels(FeWindowHandle window, const uint32_t* pixels, uint32_t width,
                                         uint32_t height);

FE_API FeResult fe_texture_presenter_create(FeWindowHandle window, FeTexturePresenterHandle* out_presenter);
FE_API FeResult fe_texture_presenter_destroy(FeTexturePresenterHandle presenter);
FE_API FeResult fe_texture_presenter_present_texture(FeTexturePresenterHandle presenter, FeTextureHandle texture,
                                                     uint32_t mode);
FE_API FeResult fe_texture_presenter_present_pixels(FeTexturePresenterHandle presenter, const uint32_t* pixels,
                                                    uint32_t width, uint32_t height);

FE_API FeResult fe_buffer_create(FeContextHandle context, const FeBufferDesc* desc, const void* initial_data,
                                 FeBufferHandle* out_buffer);
FE_API FeResult fe_buffer_destroy(FeBufferHandle buffer);
FE_API FeResult fe_buffer_upload(FeBufferHandle buffer, uint64_t offset, uint64_t size, const void* data);
FE_API FeResult fe_buffer_download(FeBufferHandle buffer, uint64_t offset, uint64_t size, void* out_data);
FE_API FeResult fe_buffer_map(FeBufferHandle buffer, uint32_t mode, void** out_ptr);
FE_API FeResult fe_buffer_unmap(FeBufferHandle buffer);

FE_API FeResult fe_texture2d_create(FeContextHandle context, const FeTexture2DDesc* desc, const void* initial_data,
                                    FeTextureHandle* out_texture);
FE_API FeResult fe_texture3d_create(FeContextHandle context, const FeTexture3DDesc* desc, const void* initial_data,
                                    FeTextureHandle* out_texture);
FE_API FeResult fe_texture_destroy(FeTextureHandle texture);
FE_API FeResult fe_texture2d_upload(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                    const void* data);
FE_API FeResult fe_texture2d_download(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                      void* out_data);
FE_API FeResult fe_texture3d_upload(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
                                    uint32_t height, uint32_t depth, const void* data);
FE_API FeResult fe_texture3d_download(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
                                      uint32_t height, uint32_t depth, void* out_data);
FE_API FeResult fe_texture_generate_mipmaps(FeTextureHandle texture);

FE_API FeResult fe_sampler_create(FeContextHandle context, const FeSamplerDesc* desc, FeSamplerHandle* out_sampler);
FE_API FeResult fe_sampler_destroy(FeSamplerHandle sampler);

FE_API FeResult fe_kernel_create_from_ir(FeContextHandle context, const FeKernelCreateDesc* desc,
                                         FeKernelHandle* out_kernel);
FE_API FeResult fe_kernel_destroy(FeKernelHandle kernel);
FE_API FeResult fe_kernel_bind_buffer(FeKernelHandle kernel, uint32_t binding, FeBufferHandle buffer);
FE_API FeResult fe_kernel_bind_texture(FeKernelHandle kernel, uint32_t binding, FeTextureHandle texture);
FE_API FeResult fe_kernel_bind_sampler(FeKernelHandle kernel, uint32_t binding, FeSamplerHandle sampler);
FE_API FeResult fe_kernel_set_push_constants(FeKernelHandle kernel, const void* data, uint64_t size);
FE_API FeResult fe_kernel_dispatch(FeKernelHandle kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                   uint32_t logical_x, uint32_t logical_y, uint32_t logical_z, bool wait);
FE_API FeResult fe_kernel_get_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size, size_t* out_required_size);
FE_API FeResult fe_kernel_get_optimized_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size,
                                             size_t* out_required_size);
FE_API FeResult fe_kernel_get_last_dispatch_path(FeKernelHandle kernel, uint32_t* out_path);
FE_API FeResult fe_kernel_get_ad_gradient_count(FeKernelHandle kernel, uint32_t* out_count);
FE_API FeResult fe_kernel_get_ad_gradient_info(FeKernelHandle kernel, uint32_t index, FeADGradientInfo* out_info);
FE_API FeResult fe_kernel_read_ad_gradient(FeKernelHandle kernel, uint32_t index, uint64_t offset, uint64_t size,
                                           void* out_data);
FE_API FeResult fe_kernel_reduce_ad_gradient_to_buffer(FeKernelHandle kernel, uint32_t index,
                                                       FeBufferHandle destination, uint64_t destination_offset,
                                                       uint64_t destination_size);
FE_API FeResult fe_kernel_get_ad_backward_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size,
                                               size_t* out_required_size);

FE_API FeResult fe_graphics_pipeline_create_from_ir(FeContextHandle context, const FeGraphicsPipelineCreateDesc* desc,
                                                    FeGraphicsPipelineHandle* out_pipeline);
FE_API FeResult fe_graphics_pipeline_destroy(FeGraphicsPipelineHandle pipeline);
FE_API FeResult fe_graphics_pipeline_set_vertex_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer,
                                                       uint32_t stride);
FE_API FeResult fe_graphics_pipeline_set_index_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer);
FE_API FeResult fe_graphics_pipeline_bind_buffer(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                 FeBufferHandle buffer);
FE_API FeResult fe_graphics_pipeline_bind_texture(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                  FeTextureHandle texture);
FE_API FeResult fe_graphics_pipeline_bind_sampler(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                  FeSamplerHandle sampler);
FE_API FeResult fe_graphics_pipeline_set_push_constants(FeGraphicsPipelineHandle pipeline, const void* data,
                                                        uint64_t size);
FE_API FeResult fe_graphics_pipeline_draw(FeGraphicsPipelineHandle pipeline, FeTextureHandle color_target,
                                          FeTextureHandle depth_target, uint32_t vertex_count, bool wait);
FE_API FeResult fe_graphics_pipeline_draw_ex(FeGraphicsPipelineHandle pipeline, const FeGraphicsDrawDesc* desc);
FE_API FeResult fe_graphics_pipeline_draw_indexed(FeGraphicsPipelineHandle pipeline, FeTextureHandle color_target,
                                                  FeTextureHandle depth_target, FeBufferHandle index_buffer,
                                                  uint32_t index_count, bool wait);
FE_API FeResult fe_graphics_pipeline_get_last_dispatch_path(FeGraphicsPipelineHandle pipeline, uint32_t* out_path);

FE_API FeResult fe_profiler_set_enabled(bool enabled);
FE_API FeResult fe_profiler_is_enabled(bool* out_enabled);
FE_API FeResult fe_profiler_clear(void);
FE_API FeResult fe_profiler_get_total_time(double* out_total_time_ms);
FE_API FeResult fe_profiler_query(const char* name, FeProfilerQueryResult* out_result);
FE_API FeResult fe_profiler_get_formatted(char* buffer, size_t buffer_size, size_t* out_required_size);

FE_API uint32_t fe_ir_bridge_contract_version(void);
FE_API FeResult fe_ir_validate(const void* ir_data, uint64_t ir_size);

#ifdef __cplusplus
}
#endif

#endif
