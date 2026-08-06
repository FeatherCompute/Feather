#pragma once

#include "feather_luisa_xir.h"
#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <cstddef>
#include <cstdint>
#include <array>
#include <functional>
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
    uint64_t resident_key = 0;
    bool upload = true;
    bool download = true;
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
    uint64_t resident_key = 0;
    bool upload = true;
    bool download = true;
    bool generate_mipmaps = false;
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
    bool synchronize = true;
    bool reuse_if_inputs_clean = false;
    bool* execution_skipped = nullptr;
    uint64_t execution_cache_key = 0;
    uint64_t context_key = 1;
    uint32_t device_index = UINT32_MAX;
    uint64_t stream_key = 0;
    uint64_t fence_key = 0;
    bool retain_fence = false;
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
    uint32_t polygon_mode = 0;
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
    uint32_t vertices_per_instance = 3;
    uint32_t vertex_domain = 3;
    uint32_t sample_count = 1;
    uint32_t opaque_fragment = 0;
};

struct AdGradientBinding {
    uint32_t source_binding = 0;
    uint32_t element_count = 0;
    uint32_t component_count = 0;
    std::vector<unsigned char>* bytes = nullptr;
};

struct DeviceInfo {
    std::string backend_name;
    std::string device_name;
    uint32_t device_index = 0;
    uint32_t compute_warp_size = 0;
};

enum class NativeTextureHandleKind : uint32_t {
    Unknown = 0u,
    MetalTexture = 1u,
    VulkanImage = 2u,
    Direct3D12Resource = 3u,
};

struct NativeTextureInfo {
    void* handle = nullptr;
    NativeTextureHandleKind kind = NativeTextureHandleKind::Unknown;
};

std::string RuntimeDirectory();
std::vector<DeviceInfo> EnumerateDevices(std::string_view runtime_directory);
bool ValidateDevice(std::string_view runtime_directory, std::string_view backend_name,
                    uint32_t device_index, DeviceInfo* info, std::string* error);

// Owns the Luisa Context/Device/Stream lifetime for Feather's native context.
// Shutdown is called from context/runtime teardown; Abandon is used on process
// exit when C++ destructors may run after the dynamic backend has unloaded.
void Shutdown();
void Shutdown(uint64_t context_key);
void Abandon() noexcept;

bool CreateStream(uint64_t context_key, std::string_view runtime_directory, std::string_view backend_name,
                  uint32_t device_index, uint64_t stream_key, std::string* error = nullptr);
bool DestroyStream(uint64_t context_key, uint64_t stream_key, std::string* error = nullptr);
bool Synchronize(uint64_t context_key, std::string* error = nullptr);
bool SynchronizeStream(uint64_t context_key, uint64_t stream_key, std::string* error = nullptr);
bool WaitFence(uint64_t context_key, uint64_t stream_key, uint64_t fence_key, std::string* error = nullptr);
bool IsFenceCompleted(uint64_t context_key, uint64_t fence_key, bool* completed, std::string* error = nullptr);
bool SynchronizeFence(uint64_t context_key, uint64_t fence_key, std::string* error = nullptr);
bool DestroyFence(uint64_t context_key, uint64_t fence_key, std::string* error = nullptr);

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> buffers, std::span<HostTextureBinding> textures,
              const DispatchInputs& dispatch, const AdInputs* ad_inputs = nullptr,
              std::span<AdGradientBinding> gradients = {}, std::string* error = nullptr);

bool PrepareGraphicsFragment(const TypedIR::Module& module,
                             const TypedIR::LoweringInputs& lowering,
                             std::span<HostBufferBinding> buffers,
                             std::span<HostTextureBinding> textures,
                             const DispatchInputs& dispatch,
                             uint64_t callable_cache_key,
                             std::string* error = nullptr);

// Experimental compute triangle assembly/raster stage between generated graphics FEIR stages.
bool DispatchVerticalRaster(HostBufferBinding vertices, HostTextureBinding target,
                            std::span<const uint32_t> vertex_indices,
                            HostTextureBinding* depth, const RasterDispatchInputs& raster,
                            const DispatchInputs& dispatch,
                            uint64_t varying_resident_key, uint64_t coverage_resident_key,
                            uint64_t geometry_cache_key, bool rebuild_geometry,
                            std::span<const uint64_t> fragment_callable_keys,
                            std::span<const uint64_t> fragment_target_keys,
                            std::vector<unsigned char>* fragment_varyings,
                            std::vector<unsigned char>* fragment_coverage,
                            std::string* error = nullptr);

bool DownloadResidentTexture(uint64_t context_key, uint64_t resident_key, void* destination, size_t size,
                             std::string* error = nullptr);

bool DownloadResidentTextureAsync(uint64_t context_key, uint64_t resident_key, void* destination, size_t size,
                                  std::function<void()> completion,
                                  std::string* error = nullptr);

bool PresentResidentTexture(uint64_t context_key, uint64_t presenter_key, uint64_t resident_key,
                            uint64_t native_display, uint64_t native_window,
                            uint32_t width, uint32_t height, bool vsync,
                            NativeTextureInfo* native_texture,
                            std::string* error = nullptr);

bool PresentHostTexture(uint64_t context_key, std::string_view runtime_directory,
                        std::string_view backend_name, uint32_t device_index,
                        uint64_t presenter_key, const void* pixels, size_t size,
                        uint64_t native_display, uint64_t native_window,
                        uint32_t width, uint32_t height, bool vsync,
                        std::string* error = nullptr);

bool DestroyPresenter(uint64_t context_key, uint64_t presenter_key,
                      std::string* error = nullptr);

bool ResolveMultisampleTexture(uint64_t context_key, std::span<const uint64_t> sample_keys,
                               const HostTextureBinding& target,
                               bool synchronize, std::string* error = nullptr);

bool ClearMultisampleTexture(uint64_t context_key, std::span<const uint64_t> sample_keys,
                             const HostTextureBinding& target,
                             const std::array<float, 4u>& color,
                             std::string* error = nullptr);

} // namespace Feather::Luisa
