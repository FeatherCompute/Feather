#include "feather_luisa_backend.h"
#include "feather_luisa_xir.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <luisa/ast/function.h>
#include <luisa/ast/type.h>
#include <luisa/core/binary_io.h>
#include <luisa/dsl/dispatch_indirect.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/dispatch_buffer.h>
#include <luisa/runtime/event.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/rtx/accel.h>
#include <luisa/runtime/rtx/mesh.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/swapchain.h>
#include <luisa/runtime/volume.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/dce.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/gvn.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/passes/simplify_cfg.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

namespace Feather::Luisa {
namespace {

using namespace luisa::compute;
using namespace luisa::compute::xir;

constexpr uint8_t kResourceBuffer = 1;
constexpr uint8_t kResourceAccel = 7;
constexpr uint8_t kResourceTexture2D = 2;
constexpr uint8_t kResourcePushConstant = 5;
constexpr uint8_t kResourceTexture3D = 6;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessReadWrite = 3;

void ensure_luisa_spirv_optimization_preset() noexcept {
    static std::once_flag once;
    std::call_once(once, [] {
        // LC 0.9.0 has no high-level optimization enum, so use SPIRV-Tools'
        // strongest stable `full`
        // recipe by default while preserving an explicit caller override.
        if (std::getenv("LUISA_SPIRV_OPT_PASSES") != nullptr) return;
#if defined(_WIN32)
        _putenv_s("LUISA_SPIRV_OPT_PASSES", "full");
#else
        setenv("LUISA_SPIRV_OPT_PASSES", "full", 1);
#endif
    });
}

class MemoryBinaryStream final : public luisa::BinaryStream {
private:
    std::vector<std::byte> data_;
    size_t position_ = 0u;

public:
    explicit MemoryBinaryStream(luisa::span<const std::byte> data)
        : data_{data.begin(), data.end()} {}

    [[nodiscard]] size_t length() const noexcept override { return data_.size(); }
    [[nodiscard]] size_t pos() const noexcept override { return position_; }

    void read(luisa::span<std::byte> destination) noexcept override {
        if (position_ > data_.size() || destination.size() > data_.size() - position_) {
            std::memset(destination.data(), 0, destination.size());
            position_ = data_.size();
            return;
        }
        std::memcpy(destination.data(), data_.data() + position_, destination.size());
        position_ += destination.size();
    }
};

// LC's default I/O opens one LMDB environment per Device. A process-wide cache
// keeps multiple Feather contexts independent without reopening the same LMDB path.
class ProcessBinaryIO final : public luisa::BinaryIO {
private:
    std::filesystem::path cache_directory_;
    std::filesystem::path data_directory_;
    mutable std::mutex mutex_;

    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_entry(
        const std::filesystem::path& directory, luisa::string_view name) const noexcept {
        std::scoped_lock lock{mutex_};
        auto path = std::filesystem::path{std::string{name.data(), name.size()}};
        if (!path.is_absolute()) path = directory / path;
        std::ifstream file{path, std::ios::binary | std::ios::ate};
        if (!file) return nullptr;
        const auto end = file.tellg();
        if (end <= 0) return nullptr;
        std::vector<std::byte> data(static_cast<size_t>(end));
        file.seekg(0, std::ios::beg);
        if (!file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()))) {
            return nullptr;
        }
        return luisa::make_unique<MemoryBinaryStream>(
            luisa::span<const std::byte>{data.data(), data.size()});
    }

    luisa::filesystem::path write_entry(
        const std::filesystem::path& directory, luisa::string_view name,
        luisa::span<const std::byte> data) const noexcept {
        std::scoped_lock lock{mutex_};
        auto path = std::filesystem::path{std::string{name.data(), name.size()}};
        if (!path.is_absolute()) path = directory / path;
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec) return {};
        std::ofstream file{path, std::ios::binary | std::ios::trunc};
        if (!file.write(reinterpret_cast<const char*>(data.data()),
                        static_cast<std::streamsize>(data.size()))) {
            return {};
        }
        return path;
    }

public:
    explicit ProcessBinaryIO(std::filesystem::path runtime_directory)
        : cache_directory_{runtime_directory / ".feather-cache"},
          data_directory_{runtime_directory / ".feather-data"} {}

    void clear_shader_cache() const noexcept override {
        std::scoped_lock lock{mutex_};
        std::error_code ec;
        std::filesystem::remove_all(cache_directory_, ec);
    }

    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_shader_bytecode(
        luisa::string_view name) const noexcept override { return read_entry(data_directory_, name); }
    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_shader_cache(
        luisa::string_view name) const noexcept override { return read_entry(cache_directory_, name); }
    [[nodiscard]] luisa::unique_ptr<luisa::BinaryStream> read_internal_shader(
        luisa::string_view name) const noexcept override { return read_entry(data_directory_, name); }

    luisa::filesystem::path write_shader_bytecode(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        return write_entry(data_directory_, name, data);
    }
    luisa::filesystem::path write_shader_cache(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        return write_entry(cache_directory_, name, data);
    }
    luisa::filesystem::path write_internal_shader(
        luisa::string_view name, luisa::span<const std::byte> data) const noexcept override {
        return write_entry(data_directory_, name, data);
    }
};

ProcessBinaryIO& process_binary_io(std::string_view runtime_directory) noexcept {
    static auto* mutex = new std::mutex{};
    static auto* instances = new std::unordered_map<std::string, std::unique_ptr<ProcessBinaryIO>>{};
    std::scoped_lock lock{*mutex};
    std::error_code ec;
    auto path = std::filesystem::absolute(std::filesystem::path{runtime_directory}, ec).lexically_normal();
    if (ec) path = std::filesystem::path{runtime_directory}.lexically_normal();
    auto key = path.string();
    auto& instance = (*instances)[key];
    if (instance == nullptr) instance = std::make_unique<ProcessBinaryIO>(std::move(path));
    return *instance;
}

bool has_vulkan_backend(const std::filesystem::path& directory) {
#if defined(_WIN32)
    return std::filesystem::exists(directory / "luisa-backend-vk.dll");
#else
    return std::filesystem::exists(directory / "libluisa-backend-vk.so") ||
           std::filesystem::exists(directory / "libluisa-backend-vk.dylib");
#endif
}

std::string resolve_runtime_directory(std::filesystem::path module_path) {
    auto directory = module_path.parent_path();
    if (has_vulkan_backend(directory))
        return directory.string();
    auto build_bin = directory / "bin";
    return has_vulkan_backend(build_bin) ? build_bin.string() : directory.string();
}

size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1u) / alignment * alignment;
}

bool feir_layout(const TypedIR::Module& module, uint32_t id, size_t* size, size_t* alignment) {
    if (id >= module.types.size() || size == nullptr || alignment == nullptr) return false;
    const auto& source = module.types[id];
    switch (source.kind) {
    case 1:
        *size = source.b / 8u;
        *alignment = *size;
        return source.b == 32u;
    case 2: {
        size_t element_size = 0;
        size_t element_alignment = 0;
        if (!feir_layout(module, source.a, &element_size, &element_alignment)) return false;
        *size = element_size * source.b;
        *alignment = source.b == 2u ? 8u : 16u;
        return true;
    }
    case 3: {
        const auto column_stride = source.b == 2u ? 8u : 16u;
        *size = source.b * column_stride;
        *alignment = column_stride;
        return source.b >= 2u && source.b <= 4u;
    }
    case 4:
        if (source.a >= module.structs.size()) return false;
        *size = module.structs[source.a].size_in_bytes;
        *alignment = module.structs[source.a].alignment;
        return true;
    case 5: {
        size_t element_size = 0;
        size_t element_alignment = 0;
        if (source.b == TypedIR::NoIndex || !feir_layout(module, source.a, &element_size, &element_alignment)) return false;
        *size = align_up(element_size, element_alignment) * source.b;
        *alignment = element_alignment;
        return true;
    }
    default:
        return false;
    }
}

bool repack_value(const TypedIR::Module& module, uint32_t id, const Type* device_type,
                  const unsigned char* source, unsigned char* destination, bool to_device) {
    if (id >= module.types.size() || device_type == nullptr || source == nullptr || destination == nullptr) return false;
    const auto& type = module.types[id];
    size_t source_size = 0;
    size_t source_alignment = 0;
    if (!feir_layout(module, id, &source_size, &source_alignment)) return false;
    auto copy_direction = [&](const void* feir, void* device, size_t bytes) {
        if (to_device) std::memcpy(device, feir, bytes);
        else std::memcpy(const_cast<void*>(feir), device, bytes);
    };
    switch (type.kind) {
    case 1:
        if (type.a == 0u) {
            if (to_device) {
                const bool value = *reinterpret_cast<const uint32_t*>(source) != 0u;
                std::memcpy(destination, &value, sizeof(value));
            } else {
                bool value = false;
                std::memcpy(&value, destination, sizeof(value));
                const uint32_t packed = value ? 1u : 0u;
                std::memcpy(const_cast<unsigned char*>(source), &packed, sizeof(packed));
            }
            return true;
        }
        copy_direction(source, destination, std::min(source_size, device_type->size()));
        return true;
    case 2:
        if (module.types[type.a].kind == 1u && module.types[type.a].a == 0u) {
            for (uint32_t i = 0; i < type.b; ++i) {
                if (!repack_value(module, type.a, Type::of<bool>(), source + i * sizeof(uint32_t), destination + i, to_device))
                    return false;
            }
            return true;
        }
        copy_direction(source, destination, source_size);
        return true;
    case 3:
        copy_direction(source, destination, source_size);
        return true;
    case 4: {
        if (type.a >= module.structs.size() || !device_type->is_structure()) return false;
        const auto& structure = module.structs[type.a];
        auto members = device_type->members();
        if (members.size() != structure.field_count) return false;
        size_t device_offset = 0;
        for (uint32_t i = 0; i < structure.field_count; ++i) {
            const auto& field = module.struct_fields[structure.first_field + i];
            device_offset = align_up(device_offset, members[i]->alignment());
            if (!repack_value(module, field.type_id, members[i], source + field.offset,
                              destination + device_offset, to_device)) return false;
            device_offset += members[i]->size();
        }
        return true;
    }
    case 5: {
        if (!device_type->is_array()) return false;
        size_t feir_element_size = 0;
        size_t feir_element_alignment = 0;
        if (!feir_layout(module, type.a, &feir_element_size, &feir_element_alignment)) return false;
        const auto feir_stride = align_up(feir_element_size, feir_element_alignment);
        const auto device_stride = align_up(device_type->element()->size(), device_type->element()->alignment());
        for (uint32_t i = 0; i < type.b; ++i)
            if (!repack_value(module, type.a, device_type->element(), source + i * feir_stride,
                              destination + i * device_stride, to_device)) return false;
        return true;
    }
    default:
        return false;
    }
}

std::optional<PixelStorage> pixel_storage(uint32_t format) {
    switch (format) {
    case 1: return PixelStorage::BYTE1;
    case 2: return PixelStorage::BYTE2;
    case 3: return PixelStorage::BYTE4;
    case 5: return PixelStorage::HALF1;
    case 6: return PixelStorage::HALF2;
    case 7: return PixelStorage::HALF4;
    case 8: return PixelStorage::FLOAT1;
    case 9: return PixelStorage::FLOAT2;
    case 10: return PixelStorage::FLOAT4;
    default: return std::nullopt;
    }
}

using RuntimeTexture = std::variant<Image<float>, Volume<float>>;

template<size_t>
using RasterUIntArgument = uint32_t;

template<typename>
struct RasterShaderType;

template<size_t... I>
struct RasterShaderType<std::index_sequence<I...>> {
    using type = Shader3D<ByteBuffer, Buffer<uint32_t>, ByteBuffer, Buffer<float>,
                          Buffer<float>, Buffer<uint32_t>, RasterUIntArgument<I>..., float>;
};

using RasterShader = RasterShaderType<std::make_index_sequence<29u>>::type;
using MipmapShader = Shader2D<Image<float>, Image<float>>;

template<size_t>
using FastRasterUIntArgument = uint32_t;

template<typename>
struct FastRasterShaderType;

template<size_t... I>
struct FastRasterShaderType<std::index_sequence<I...>> {
    using type = Shader2D<ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>, Buffer<uint32_t>,
                          FastRasterUIntArgument<I>...>;
};

using FastRasterShader = FastRasterShaderType<std::make_index_sequence<14u>>::type;
using FastRasterInitShader = Shader3D<Buffer<uint32_t>, Buffer<uint32_t>, Buffer<float>,
                                      Buffer<float>, uint32_t, float>;

template<size_t>
using TileAssemblyUIntArgument = uint32_t;

template<typename>
struct TileAssemblyShaderType;

template<size_t... I>
struct TileAssemblyShaderType<std::index_sequence<I...>> {
    using type = Shader1D<ByteBuffer, Buffer<uint32_t>, ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                          TileAssemblyUIntArgument<I>...>;
};

using TileAssemblyShader = TileAssemblyShaderType<std::make_index_sequence<15u>>::type;

template<size_t>
using TileRasterUIntArgument = uint32_t;

template<typename>
struct TileRasterShaderType;

template<size_t... I>
struct TileRasterShaderType<std::index_sequence<I...>> {
    using type = Shader3D<ByteBuffer, ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                          Buffer<uint32_t>, Buffer<float>, ByteBuffer, Buffer<float>,
                          TileRasterUIntArgument<I>...>;
};

using TileRasterShader = TileRasterShaderType<std::make_index_sequence<8u>>::type;

template<typename>
struct SharedTileRasterShaderType;

template<size_t... I>
struct SharedTileRasterShaderType<std::index_sequence<I...>> {
    using type = Shader2D<ByteBuffer, ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                          Buffer<uint32_t>, Buffer<uint32_t>, Buffer<float>, ByteBuffer, Buffer<float>,
                          TileRasterUIntArgument<I>..., uint32_t, float>;
};

using SharedTileRasterShader = SharedTileRasterShaderType<std::make_index_sequence<8u>>::type;
template<typename>
struct FusedTileRasterShaderType;

template<size_t... I>
struct FusedTileRasterShaderType<std::index_sequence<I...>> {
    using type = Shader2D<ByteBuffer, ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                          Buffer<uint32_t>, Buffer<uint32_t>, Buffer<float>,
                          Image<float>,
                          TileRasterUIntArgument<I>..., uint32_t, float, uint32_t, float4>;
};

using FusedTileRasterShader = FusedTileRasterShaderType<std::make_index_sequence<8u>>::type;
using FastRasterResolveShader = Shader3D<ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                                         Buffer<float>, ByteBuffer, Buffer<float>,
                                         uint32_t, uint32_t, uint32_t, uint32_t>;
using TileResetShader = Shader1D<Buffer<uint32_t>, Buffer<uint32_t>, Buffer<uint32_t>,
                                 Buffer<uint32_t>, uint32_t>;
using TilePrefixShader = Shader1D<Buffer<uint32_t>, Buffer<uint32_t>, Buffer<uint32_t>,
                                  Buffer<uint32_t>, IndirectDispatchBuffer, uint32_t, uint32_t>;
using TileFillShader = Shader1D<ByteBuffer, Buffer<uint32_t>, Buffer<uint32_t>,
                                Buffer<uint32_t>, Buffer<uint32_t>, Buffer<uint32_t>,
                                uint32_t, uint32_t, uint32_t, uint32_t>;
using MsaaResolveShader = Shader2D<Image<float>, Image<float>, Image<float>, Image<float>, Image<float>>;
using MsaaClearShader = Shader2D<Image<float>, Image<float>, Image<float>, Image<float>, float4>;
using MsaaLoadShader = Shader2D<Image<float>, Image<float>, Image<float>, Image<float>, Image<float>>;
using AdGradientReduceShader = Shader1D<ByteBuffer, ByteBuffer, uint32_t, uint32_t, uint32_t>;

constexpr auto kMaximumClippedVertices = 12u;
constexpr auto kClippedPrimitiveStride = 16u;
constexpr auto kRasterTileSize = 8u;
constexpr auto kRasterMicroCellSize = 2u;
constexpr auto kRasterMicroCellsPerAxis = kRasterTileSize / kRasterMicroCellSize;
constexpr auto kRasterPrimitiveRecordSize = 192u;
constexpr auto kRasterPrimitiveEdge0Offset = 144u;
constexpr auto kRasterPrimitiveEdge1Offset = 160u;
constexpr auto kRasterPrimitiveEdge2Offset = 176u;
constexpr auto kRasterPrimitiveExpansion = 10u;
constexpr auto kInitialTileReferencesPerTriangle = 32u;
constexpr auto kSharedPrimitiveBatchSize = 64u;

void clip_homogeneous_triangle(ArrayFloat4<kMaximumClippedVertices>& positions,
                               ArrayFloat3<kMaximumClippedVertices>& source_weights,
                               UInt& count) noexcept {
    Bool trivially_inside = def(true);
    Bool trivially_outside = def(false);
    for (auto plane = 0u; plane < 6u; ++plane) {
        Bool any_inside = def(false);
        Bool all_inside = def(true);
        for (auto vertex = 0u; vertex < 3u; ++vertex) {
            const auto p = positions[vertex];
            Float distance = def(0.0f);
            switch (plane) {
            case 0u: distance = p.x + p.w; break;
            case 1u: distance = p.w - p.x; break;
            case 2u: distance = p.y + p.w; break;
            case 3u: distance = p.w - p.y; break;
            case 4u: distance = p.z; break;
            default: distance = p.w - p.z; break;
            }
            const auto inside = distance >= 0.0f;
            any_inside |= inside;
            all_inside &= inside;
        }
        trivially_inside &= all_inside;
        trivially_outside |= !any_inside;
    }
    $if (trivially_outside) { count = 0u; }
    $elif (!trivially_inside) {
        for (auto plane = 0u; plane < 6u; ++plane) {
            ArrayFloat4<kMaximumClippedVertices> output_positions;
            ArrayFloat3<kMaximumClippedVertices> output_weights;
            UInt output_count = 0u;
            const auto distance = [plane](Float4 p) noexcept {
            switch (plane) {
            case 0u: return def(p.x + p.w);
            case 1u: return def(p.w - p.x);
            case 2u: return def(p.y + p.w);
            case 3u: return def(p.w - p.y);
            case 4u: return def(p.z);
            default: return def(p.w - p.z);
            }
            };
            UInt i = 0u;
            $while (i < count) {
            const auto previous_index = ite(i == 0u, count - 1u, i - 1u);
            Float4 previous_position = positions[previous_index];
            Float4 current_position = positions[i];
            Float3 previous_weights = source_weights[previous_index];
            Float3 current_weights = source_weights[i];
            Float previous_distance = distance(previous_position);
            Float current_distance = distance(current_position);
            const auto previous_inside = previous_distance >= 0.0f;
            const auto current_inside = current_distance >= 0.0f;
            $if (previous_inside != current_inside) {
                const auto t = previous_distance / (previous_distance - current_distance);
                output_positions[output_count] = lerp(previous_position, current_position, t);
                output_weights[output_count] = lerp(previous_weights, current_weights, t);
                output_count += 1u;
            };
            $if (current_inside) {
                output_positions[output_count] = current_position;
                output_weights[output_count] = current_weights;
                output_count += 1u;
            };
            i += 1u;
            };
            count = output_count;
            UInt copy_index = 0u;
            $while (copy_index < output_count) {
            positions[copy_index] = output_positions[copy_index];
            source_weights[copy_index] = output_weights[copy_index];
            copy_index += 1u;
            };
        }
    };
}

class RuntimeState {
public:
  using Completion = std::function<bool(std::string*)>;

  struct CachedKernel {
      std::unique_ptr<Shader3D<>> shader;
      std::vector<ByteBuffer*> buffers;
      std::vector<ByteBuffer*> push_constants;
      std::vector<RuntimeTexture*> textures;
      std::vector<ByteBuffer*> gradients;
      std::vector<std::unique_ptr<ByteBuffer>> owned_buffers;
      std::vector<std::unique_ptr<ByteBuffer>> owned_push_constants;
      std::vector<RuntimeTexture> owned_textures;
      std::vector<std::unique_ptr<ByteBuffer>> owned_gradients;
      uint64_t execution_cache_key = ~0ull;
  };

    struct CachedFragment {
        luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> callable;
        const Type* varying_type = nullptr;
        const Type* return_type = nullptr;
    };

    struct CachedRasterGeometry {
        uint64_t geometry_key = 0u;
        uint32_t primitive_count = 0u;
        uint32_t reference_count = 0u;
    };

    struct ResidentTexture {
        std::unique_ptr<RuntimeTexture> resource;
        uint8_t kind = 0;
        PixelStorage storage = PixelStorage::BYTE1;
        uint3 size{};
        uint32_t mip_levels = 1;
    };

    struct PresentTarget {
        uint64_t display = 0u;
        uint64_t window = 0u;
        uint2 size{};
        bool vsync = true;
        Stream stream;
        Swapchain swapchain;
        Event handoff;
        Image<float> staging;
    };

private:
  struct FenceState {
      Event event;
      uint64_t fence = 0u;
      uint64_t stream_key = 0u;
      uint64_t sequence = 0u;
      Completion completion;
      bool finalized = false;
      bool retained = false;
      std::string error;
  };

    std::unique_ptr<Context> context_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<Stream> stream_;
    std::unordered_map<uint64_t, std::unique_ptr<Stream>> streams_;
    std::unordered_set<uint64_t> stream_keys_;
    std::unordered_map<uint64_t, FenceState> fences_;
    uint64_t next_fence_sequence_ = 1u;
    std::string runtime_directory_;
    std::string backend_name_;
    uint32_t device_index_ = UINT32_MAX;
    std::unordered_map<uint64_t, std::unique_ptr<CachedKernel>> kernels_;
    std::unordered_map<uint64_t, CachedFragment> fragment_callables_;
    std::unordered_map<uint64_t, CachedRasterGeometry> raster_geometries_;
    std::unordered_map<uint64_t, std::unique_ptr<RasterShader>> raster_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<FastRasterShader>> fast_raster_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<FastRasterResolveShader>> fast_raster_resolve_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<TileAssemblyShader>> tile_assembly_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<TileRasterShader>> tile_raster_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<SharedTileRasterShader>> shared_tile_raster_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<FusedTileRasterShader>> fused_tile_raster_shaders_;
    std::unordered_map<uint64_t, std::unique_ptr<ByteBuffer>> resident_buffers_;
    std::unordered_map<uint64_t, ResidentTexture> resident_textures_;
    std::unordered_map<uint64_t, std::unique_ptr<PresentTarget>> present_targets_;
    std::unique_ptr<MipmapShader> mipmap_shader_;
    std::unique_ptr<FastRasterInitShader> fast_raster_init_shader_;
    std::unique_ptr<TileResetShader> tile_reset_shader_;
    std::unique_ptr<TilePrefixShader> tile_prefix_shader_;
    std::unique_ptr<TileFillShader> tile_fill_shader_;
    std::unique_ptr<IndirectDispatchBuffer> tile_fill_dispatch_buffer_;
    std::unique_ptr<MsaaResolveShader> msaa_resolve_shader_;
    std::unique_ptr<MsaaClearShader> msaa_clear_shader_;
    std::unique_ptr<MsaaLoadShader> msaa_load_shader_;
    std::unique_ptr<AdGradientReduceShader> ad_gradient_reduce_shader_;

public:
    struct AccelState {
        std::vector<std::unique_ptr<Buffer<float3>>> vertex_buffers;
        std::vector<std::unique_ptr<Buffer<Triangle>>> index_buffers;
        std::vector<std::unique_ptr<Mesh>> meshes;
        std::unique_ptr<Accel> accel;
    };
    std::unordered_map<uint64_t, std::unique_ptr<AccelState>> accels_;

    RuntimeState() = default;
    RuntimeState(const RuntimeState &) = delete;
    RuntimeState &operator=(const RuntimeState &) = delete;

    void ensure(std::string_view runtime_directory, std::string_view backend_name, uint32_t device_index) {
        if (context_ != nullptr && runtime_directory_ == runtime_directory &&
            backend_name_ == backend_name && device_index_ == device_index) return;
        reset();
        // Vulkan device enumeration must happen before a Vulkan Device is live.
        // Warm Feather's discovery cache so later managed discovery remains safe.
        (void)EnumerateDevices(runtime_directory);
        runtime_directory_ = runtime_directory;
        backend_name_ = backend_name;
        device_index_ = device_index;
        context_ = std::make_unique<Context>(runtime_directory_);
        DeviceConfig config{};
        config.binary_io = &process_binary_io(runtime_directory_);
        if (device_index != UINT32_MAX) {
            config.device_index = device_index;
        }
        device_ = std::make_unique<Device>(context_->create_device(backend_name_, &config));
        stream_ = std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE));
        for (const auto key : stream_keys_) {
            streams_.emplace(key, std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE)));
        }
    }

    Device &device() noexcept { return *device_; }
    Stream &stream() noexcept { return *stream_; }
    Stream* stream(uint64_t key) noexcept {
        if (key == 0u)
            return stream_.get();
        const auto found = streams_.find(key);
        return found == streams_.end() ? nullptr : found->second.get();
    }

    [[nodiscard]] bool has_stream(uint64_t key) const noexcept {
        return key == 0u || stream_keys_.contains(key);
    }

    bool create_stream(uint64_t key, std::string* error) {
        if (key == 0u || streams_.contains(key)) {
            if (error != nullptr)
                *error = "Luisa stream handle is invalid or already exists";
            return false;
        }
        stream_keys_.emplace(key);
        streams_.emplace(key, std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE)));
        return true;
    }

    bool finalize_fence(FenceState& state, std::string* error) {
        if (!state.finalized) {
            state.finalized = true;
            if (state.completion && !state.completion(&state.error) && state.error.empty()) {
                state.error = "Luisa asynchronous dispatch finalization failed";
            }
            state.completion = {};
        }
        if (!state.error.empty()) {
            if (error != nullptr)
                *error = state.error;
            return false;
        }
        return true;
    }

    bool finalize_stream_through(uint64_t stream_key, uint64_t sequence, bool release_orphans, std::string* error) {
        std::vector<std::pair<uint64_t, uint64_t>> ordered;
        for (const auto& [key, fence] : fences_) {
            if (fence.stream_key == stream_key && fence.sequence <= sequence) {
                ordered.emplace_back(fence.sequence, key);
            }
        }
        std::ranges::sort(ordered);
        bool succeeded = true;
        std::vector<uint64_t> completed_orphans;
        for (const auto [fence_sequence, key] : ordered) {
            (void)fence_sequence;
            auto& fence = fences_.at(key);
            if (!finalize_fence(fence, error))
                succeeded = false;
            if (release_orphans && !fence.retained)
                completed_orphans.push_back(key);
        }
        for (const auto key : completed_orphans)
            fences_.erase(key);
        return succeeded;
    }

    bool collect_completed_orphans(uint64_t stream_key, std::string* error) {
        uint64_t completed_sequence = 0u;
        for (const auto& [key, fence] : fences_) {
            (void)key;
            if (fence.stream_key == stream_key && !fence.retained &&
                fence.event.is_completed(fence.fence)) {
                completed_sequence = std::max(completed_sequence, fence.sequence);
            }
        }
        return completed_sequence == 0u ||
               finalize_stream_through(stream_key, completed_sequence, true, error);
    }

    bool signal_fence(uint64_t stream_key, uint64_t fence_key, bool retained, Completion completion,
                      std::string* error) {
        auto* target = stream(stream_key);
        if (target == nullptr || fence_key == 0u || fences_.contains(fence_key)) {
            if (error != nullptr)
                *error = "Luisa stream or fence handle is invalid";
            return false;
        }
        if (!collect_completed_orphans(stream_key, error)) return false;
        auto event = device_->create_event();
        *target << event.signal();
        const auto value = event.last_fence();
        fences_.emplace(fence_key, FenceState{std::move(event),
                                              value,
                                              stream_key,
                                              next_fence_sequence_++,
                                              std::move(completion),
                                              false,
                                              retained,
                                              {}});
        return true;
    }

    bool is_fence_completed(uint64_t key, bool* completed, std::string* error) {
        const auto found = fences_.find(key);
        if (found == fences_.end() || completed == nullptr) {
            if (error != nullptr)
                *error = "Luisa fence handle is invalid";
            return false;
        }
        if (found->second.finalized) {
            *completed = true;
            return finalize_fence(found->second, error);
        }
        *completed = found->second.event.is_completed(found->second.fence);
        return !*completed || finalize_stream_through(found->second.stream_key, found->second.sequence, false, error);
    }

    bool synchronize_fence(uint64_t key, std::string* error) {
        const auto found = fences_.find(key);
        if (found == fences_.end()) {
            if (error != nullptr)
                *error = "Luisa fence handle is invalid";
            return false;
        }
        if (found->second.finalized)
            return finalize_fence(found->second, error);
        auto* producer = stream(found->second.stream_key);
        if (producer == nullptr) {
            if (error != nullptr)
                *error = "Luisa fence producer stream is unavailable";
            return false;
        }
        producer->synchronize();
        return finalize_stream_through(found->second.stream_key, found->second.sequence, false, error);
    }

    bool wait_fence(uint64_t stream_key, uint64_t fence_key, std::string* error) {
        const auto found = fences_.find(fence_key);
        if (!has_stream(stream_key) || found == fences_.end()) {
            if (error != nullptr)
                *error = "Luisa stream or fence handle is invalid";
            return false;
        }
        if (found->second.finalized)
            return finalize_fence(found->second, error);
        // LC 35a06cb exposes device-side event waits, but completion of Feather's
        // host readback/repacking is part of the fence contract too. Synchronize
        // the producer stream before returning so both GPU and host-visible state
        // are ordered consistently on every backend.
        auto* producer = stream(found->second.stream_key);
        if (producer == nullptr) {
            if (error != nullptr)
                *error = "Luisa fence producer stream is unavailable";
            return false;
        }
        producer->synchronize();
        return finalize_stream_through(found->second.stream_key, found->second.sequence, false, error);
    }

    bool synchronize_stream(uint64_t key, std::string* error) {
        auto* target = stream(key);
        if (target == nullptr) {
            if (!has_device() && has_stream(key))
                return finalize_stream_through(key, std::numeric_limits<uint64_t>::max(), true, error);
            if (error != nullptr)
                *error = "Luisa stream handle is invalid";
            return false;
        }
        target->synchronize();
        return finalize_stream_through(key, std::numeric_limits<uint64_t>::max(), true, error);
    }

    bool synchronize(std::string* error) {
        if (!has_device())
            return true;
        bool succeeded = synchronize_stream(0u, error);
        for (const auto& [key, value] : streams_) {
            (void)value;
            if (!synchronize_stream(key, error))
                succeeded = false;
        }
        for (const auto& [key, target] : present_targets_) {
            (void)key;
            target->stream.synchronize();
        }
        return succeeded;
    }

    bool destroy_fence(uint64_t key, std::string* error) {
        if (!synchronize_fence(key, error))
            return false;
        fences_.erase(key);
        return true;
    }

    bool destroy_stream(uint64_t key, std::string* error) {
        if (key == 0u || !has_stream(key)) {
            if (error != nullptr)
                *error = "Luisa stream handle is invalid";
            return false;
        }
        if (!has_device()) {
            stream_keys_.erase(key);
            return true;
        }
        if (!synchronize_stream(key, error))
            return false;
        streams_.erase(key);
        stream_keys_.erase(key);
        return true;
    }
    [[nodiscard]] bool has_device() const noexcept { return device_ != nullptr; }
    [[nodiscard]] std::string_view backend_name() const noexcept { return backend_name_; }
    [[nodiscard]] uint32_t device_index() const noexcept { return device_index_; }

    CachedKernel *find(uint64_t key) noexcept {
        const auto it = kernels_.find(key);
        return it == kernels_.end() ? nullptr : it->second.get();
    }

    void insert(uint64_t key, std::unique_ptr<CachedKernel> kernel) {
        kernels_[key] = std::move(kernel);
    }

    CachedFragment* find_fragment(uint64_t key) noexcept {
        const auto found = fragment_callables_.find(key);
        return found == fragment_callables_.end() ? nullptr : &found->second;
    }

    void insert_fragment(uint64_t key, CachedFragment fragment) {
        fragment_callables_[key] = std::move(fragment);
    }

    CachedRasterGeometry* find_raster_geometry(uint64_t key) noexcept {
        const auto found = raster_geometries_.find(key);
        return found == raster_geometries_.end() ? nullptr : &found->second;
    }

    void insert_raster_geometry(uint64_t key, CachedRasterGeometry geometry) {
        raster_geometries_[key] = geometry;
    }

    RasterShader* find_raster(uint64_t key) noexcept {
        const auto found = raster_shaders_.find(key);
        return found == raster_shaders_.end() ? nullptr : found->second.get();
    }

    RasterShader* insert_raster(uint64_t key, RasterShader shader) {
        auto entry = std::make_unique<RasterShader>(std::move(shader));
        auto* result = entry.get();
        raster_shaders_[key] = std::move(entry);
        return result;
    }

    FastRasterShader* find_fast_raster(uint64_t key) noexcept {
        const auto found = fast_raster_shaders_.find(key);
        return found == fast_raster_shaders_.end() ? nullptr : found->second.get();
    }

    FastRasterShader* insert_fast_raster(uint64_t key, FastRasterShader shader) {
        auto entry = std::make_unique<FastRasterShader>(std::move(shader));
        auto* result = entry.get();
        fast_raster_shaders_[key] = std::move(entry);
        return result;
    }

    FastRasterResolveShader* find_fast_raster_resolve(uint64_t key) noexcept {
        const auto found = fast_raster_resolve_shaders_.find(key);
        return found == fast_raster_resolve_shaders_.end() ? nullptr : found->second.get();
    }

    FastRasterResolveShader* insert_fast_raster_resolve(uint64_t key, FastRasterResolveShader shader) {
        auto entry = std::make_unique<FastRasterResolveShader>(std::move(shader));
        auto* result = entry.get();
        fast_raster_resolve_shaders_[key] = std::move(entry);
        return result;
    }

    TileAssemblyShader* find_tile_assembly(uint64_t key) noexcept {
        const auto found = tile_assembly_shaders_.find(key);
        return found == tile_assembly_shaders_.end() ? nullptr : found->second.get();
    }

    TileAssemblyShader* insert_tile_assembly(uint64_t key, TileAssemblyShader shader) {
        auto entry = std::make_unique<TileAssemblyShader>(std::move(shader));
        auto* result = entry.get();
        tile_assembly_shaders_[key] = std::move(entry);
        return result;
    }

    TileRasterShader* find_tile_raster(uint64_t key) noexcept {
        const auto found = tile_raster_shaders_.find(key);
        return found == tile_raster_shaders_.end() ? nullptr : found->second.get();
    }

    TileRasterShader* insert_tile_raster(uint64_t key, TileRasterShader shader) {
        auto entry = std::make_unique<TileRasterShader>(std::move(shader));
        auto* result = entry.get();
        tile_raster_shaders_[key] = std::move(entry);
        return result;
    }

    SharedTileRasterShader* find_shared_tile_raster(uint64_t key) noexcept {
        const auto found = shared_tile_raster_shaders_.find(key);
        return found == shared_tile_raster_shaders_.end() ? nullptr : found->second.get();
    }

    SharedTileRasterShader* insert_shared_tile_raster(uint64_t key, SharedTileRasterShader shader) {
        auto entry = std::make_unique<SharedTileRasterShader>(std::move(shader));
        auto* result = entry.get();
        shared_tile_raster_shaders_[key] = std::move(entry);
        return result;
    }

    FusedTileRasterShader* find_fused_tile_raster(uint64_t key) noexcept {
        const auto found = fused_tile_raster_shaders_.find(key);
        return found == fused_tile_raster_shaders_.end() ? nullptr : found->second.get();
    }

    FusedTileRasterShader* insert_fused_tile_raster(uint64_t key, FusedTileRasterShader shader) {
        auto entry = std::make_unique<FusedTileRasterShader>(std::move(shader));
        auto* result = entry.get();
        fused_tile_raster_shaders_[key] = std::move(entry);
        return result;
    }

    FastRasterInitShader* fast_raster_init_shader() {
        if (fast_raster_init_shader_ == nullptr) {
            auto shader = device_->compile<3>([](BufferUInt depth_bits, BufferUInt owners,
                                                 BufferFloat depth_values, BufferFloat coverage,
                                                 UInt clear_depth, Float clear_value) noexcept {
                const auto id = dispatch_id();
                const auto pixel = id.y * dispatch_size().x + id.x;
                const auto index = pixel * dispatch_size().z + id.z;
                const auto depth = ite(clear_depth != 0u, clear_value, depth_values.read(index));
                depth_bits.write(index, depth.bitcast<uint>());
                depth_values.write(index, depth);
                owners.write(index, ~0u);
                coverage.write(index, 0.0f);
            });
            fast_raster_init_shader_ = std::make_unique<FastRasterInitShader>(std::move(shader));
        }
        return fast_raster_init_shader_.get();
    }

    TileResetShader* tile_reset_shader() {
        if (tile_reset_shader_ == nullptr) {
            auto shader = device_->compile<1>([](BufferUInt counts, BufferUInt offsets,
                                                 BufferUInt cursors, BufferUInt stats,
                                                 UInt tile_count) noexcept {
                const auto index = dispatch_id().x;
                $if (index < tile_count) {
                    counts.write(index, 0u);
                    offsets.write(index, 0u);
                    cursors.write(index, 0u);
                };
                $if (index < 3u) { stats.write(index, 0u); };
            });
            tile_reset_shader_ = std::make_unique<TileResetShader>(std::move(shader));
        }
        return tile_reset_shader_.get();
    }

    TilePrefixShader* tile_prefix_shader() {
        if (tile_prefix_shader_ == nullptr) {
            auto shader = device_->compile<1>([](BufferUInt counts, BufferUInt offsets,
                                                 BufferUInt cursors, BufferUInt stats,
                                                 IndirectDispatchBufferVar fill_dispatch,
                                                 UInt tile_count, UInt reference_capacity) noexcept {
                $if (dispatch_id().x == 0u) {
                    UInt total = 0u;
                    UInt tile = 0u;
                    $while (tile < tile_count) {
                        offsets.write(tile, total);
                        cursors.write(tile, 0u);
                        total += counts.read(tile);
                        tile += 1u;
                    };
                    stats.write(1u, total);
                    $if (total > reference_capacity) { stats.write(2u, 1u); };
                    fill_dispatch.set_dispatch_count(1u);
                    fill_dispatch.set_kernel(
                        0u, make_uint3(64u, 1u, 1u),
                        make_uint3(stats.read(0u), 1u, 1u));
                };
            });
            tile_prefix_shader_ = std::make_unique<TilePrefixShader>(std::move(shader));
        }
        return tile_prefix_shader_.get();
    }

    TileFillShader* tile_fill_shader() {
        if (tile_fill_shader_ == nullptr) {
            auto shader = device_->compile<1>([](ByteBufferVar primitives, BufferUInt offsets,
                                                 BufferUInt cursors, BufferUInt indices,
                                                 BufferUInt masks, BufferUInt stats, UInt primitive_capacity,
                                                 UInt tile_width, UInt reference_capacity,
                                                 UInt precise_masks) noexcept {
                set_block_size(64u, 1u, 1u);
                const auto primitive = dispatch_id().x;
                const auto primitive_count = luisa::compute::min(stats.read(0u), primitive_capacity);
                $if (primitive < primitive_count) {
                    const auto base = primitive * kRasterPrimitiveRecordSize;
                    const auto bounds = primitives.read<uint4>(base + 112u);
                    const auto pixel_bounds = primitives.read<float4>(base + 128u);
                    const auto screen_ab = primitives.read<float4>(base);
                    const auto screen_c_metadata = primitives.read<float4>(base + 16u);
                    const auto edge0 = primitives.read<float4>(base + kRasterPrimitiveEdge0Offset);
                    const auto edge1 = primitives.read<float4>(base + kRasterPrimitiveEdge1Offset);
                    const auto edge2 = primitives.read<float4>(base + kRasterPrimitiveEdge2Offset);
                    UInt tile_y = bounds.y;
                    $while (tile_y <= bounds.w) {
                        UInt tile_x = bounds.x;
                        $while (tile_x <= bounds.z) {
                            const auto tile = tile_y * tile_width + tile_x;
                            const auto slot = cursors.atomic(tile).fetch_add(1u);
                            const auto destination = offsets.read(tile) + slot;
                            $if (destination < reference_capacity) {
                                UInt mask0 = 0u;
                                UInt mask1 = 0u;
                                for (uint32_t micro_y = 0u; micro_y < kRasterMicroCellsPerAxis; ++micro_y) {
                                    for (uint32_t micro_x = 0u; micro_x < kRasterMicroCellsPerAxis; ++micro_x) {
                                        const auto origin = make_float2(
                                            (tile_x * kRasterTileSize + micro_x * kRasterMicroCellSize).cast<float>(),
                                            (tile_y * kRasterTileSize + micro_y * kRasterMicroCellSize).cast<float>());
                                        const auto end = origin + static_cast<float>(kRasterMicroCellSize - 1u);
                                        Bool overlaps = pixel_bounds.x <= end.x &
                                                        pixel_bounds.z >= origin.x &
                                                        pixel_bounds.y <= end.y &
                                                        pixel_bounds.w >= origin.y;
                                        $if (precise_masks != 0u & overlaps) {
                                            const auto edge_maximum = [&](Float4 edge, Float2 reference) noexcept {
                                                const auto sample_origin = origin + 0.125f;
                                                const auto sample_limit = origin +
                                                    (static_cast<float>(kRasterMicroCellSize) - 0.125f);
                                                return edge.x * (ite(edge.x >= 0.0f, sample_limit.x, sample_origin.x) - reference.x) +
                                                       edge.y * (ite(edge.y >= 0.0f, sample_limit.y, sample_origin.y) - reference.y);
                                            };
                                            overlaps = edge_maximum(edge0, screen_ab.zw()) >= 0.0f &
                                                       edge_maximum(edge1, screen_c_metadata.xy()) >= 0.0f &
                                                       edge_maximum(edge2, screen_ab.xy()) >= 0.0f;
                                        };
                                        const auto bit = micro_y * kRasterMicroCellsPerAxis + micro_x;
                                        if (bit < 32u) mask0 |= ite(overlaps, 1u << bit, 0u);
                                        else mask1 |= ite(overlaps, 1u << (bit - 32u), 0u);
                                    }
                                }
                                indices.write(destination, primitive);
                                const auto mask_base = destination * 2u;
                                masks.write(mask_base, mask0);
                                masks.write(mask_base + 1u, mask1);
                            }
                            $else { stats.write(2u, 1u); };
                            tile_x += 1u;
                        };
                        tile_y += 1u;
                    };
                };
            });
            tile_fill_shader_ = std::make_unique<TileFillShader>(std::move(shader));
        }
        return tile_fill_shader_.get();
    }

    IndirectDispatchBuffer& tile_fill_dispatch_buffer() {
        if (tile_fill_dispatch_buffer_ == nullptr) {
            tile_fill_dispatch_buffer_ = std::make_unique<IndirectDispatchBuffer>(
                device_->create_indirect_dispatch_buffer(1u));
        }
        return *tile_fill_dispatch_buffer_;
    }

    ByteBuffer* resident_buffer(uint64_t key, size_t size, bool* created = nullptr) {
        if (created != nullptr)
            *created = false;
        if (key == 0u || size == 0u) return nullptr;
        if (const auto found = resident_buffers_.find(key); found != resident_buffers_.end()) {
            return found->second->size_bytes() == size ? found->second.get() : nullptr;
        }
        auto buffer = std::make_unique<ByteBuffer>(device_->create_byte_buffer(size));
        auto* result = buffer.get();
        resident_buffers_.emplace(key, std::move(buffer));
        if (created != nullptr)
            *created = true;
        return result;
    }

    bool reduce_ad_gradient(std::span<const unsigned char> gradient, uint32_t element_count,
                            uint32_t component_count, uint64_t destination_key,
                            std::vector<unsigned char>* destination, uint64_t destination_offset,
                            uint64_t destination_size, bool upload_destination,
                            std::string* error) {
        const auto scalar_slots = static_cast<uint64_t>(element_count) * component_count;
        if (destination == nullptr || scalar_slots == 0u || gradient.size() % sizeof(float) != 0u ||
            destination_offset % sizeof(float) != 0u) {
            if (error != nullptr) *error = "AD gradient reduction has an invalid scalar layout";
            return false;
        }
        const auto total_scalars = gradient.size() / sizeof(float);
        if (total_scalars % scalar_slots != 0u || scalar_slots > UINT32_MAX ||
            total_scalars / scalar_slots > UINT32_MAX ||
            destination_size != scalar_slots * sizeof(float) ||
            destination_offset > destination->size() ||
            destination_size > destination->size() - destination_offset) {
            if (error != nullptr) *error = "AD gradient cannot be reduced to the requested destination range";
            return false;
        }
        bool destination_created = false;
        auto* destination_buffer = resident_buffer(destination_key, destination->size(), &destination_created);
        if (destination_buffer == nullptr) {
            if (error != nullptr) *error = "Luisa AD reduction destination buffer is unavailable";
            return false;
        }
        if (ad_gradient_reduce_shader_ == nullptr) {
            auto shader = device_->compile<1>([](ByteBufferVar source, ByteBufferVar target,
                                                 UInt slots, UInt dispatch_count,
                                                 UInt destination_byte_offset) noexcept {
                const auto slot = dispatch_id().x;
                $if (slot < slots) {
                    Float sum = 0.0f;
                    $for (index, dispatch_count) {
                        sum += source.read<float>((index * slots + slot) * 4u);
                    };
                    target.write(destination_byte_offset + slot * 4u, sum);
                };
            });
            ad_gradient_reduce_shader_ = std::make_unique<AdGradientReduceShader>(std::move(shader));
        }
        auto source_buffer = device_->create_byte_buffer(gradient.size());
        if (destination_created || upload_destination)
            *stream_ << destination_buffer->copy_from(destination->data());
        *stream_ << source_buffer.copy_from(gradient.data())
                 << (*ad_gradient_reduce_shader_)(source_buffer, *destination_buffer,
                                                  static_cast<uint32_t>(scalar_slots),
                                                  static_cast<uint32_t>(total_scalars / scalar_slots),
                                                  static_cast<uint32_t>(destination_offset))
                        .dispatch(static_cast<uint32_t>(scalar_slots))
                 << destination_buffer->copy_to(destination->data())
                 << luisa::compute::synchronize();
        return true;
    }

    RuntimeTexture* resident_texture(uint64_t key, uint8_t kind, PixelStorage storage, uint3 size, uint32_t mip_levels,
                                     bool* created = nullptr) {
        if (created != nullptr)
            *created = false;
        if (key == 0u || size.x == 0u || size.y == 0u || size.z == 0u) return nullptr;
        if (const auto found = resident_textures_.find(key); found != resident_textures_.end()) {
            const auto& entry = found->second;
            return entry.kind == kind && entry.storage == storage && entry.size.x == size.x &&
                           entry.size.y == size.y && entry.size.z == size.z &&
                           entry.mip_levels == mip_levels
                       ? entry.resource.get()
                       : nullptr;
        }
        std::unique_ptr<RuntimeTexture> resource;
        if (kind == kResourceTexture2D) {
            resource = std::make_unique<RuntimeTexture>(
                device_->create_image<float>(storage, make_uint2(size), mip_levels, true));
        } else if (kind == kResourceTexture3D) {
            resource = std::make_unique<RuntimeTexture>(
                device_->create_volume<float>(storage, size, mip_levels, true));
        } else {
            return nullptr;
        }
        auto* result = resource.get();
        resident_textures_.emplace(
            key, ResidentTexture{std::move(resource), kind, storage, size, mip_levels});
        if (created != nullptr)
            *created = true;
        return result;
    }

    bool generate_mipmaps(RuntimeTexture& resource, uint32_t mip_levels, std::string* error) {
        auto* image = std::get_if<Image<float>>(&resource);
        if (image == nullptr) {
            if (error != nullptr) *error = "Luisa mipmap generation currently supports 2D textures";
            return false;
        }
        if (mip_levels <= 1u) return true;
        if (mipmap_shader_ == nullptr) {
            auto shader = device_->compile<2>([](ImageFloat source, ImageFloat destination) noexcept {
                const auto destination_pixel = dispatch_id().xy();
                const auto source_origin = destination_pixel * 2u;
                const auto source_limit = source.size() - 1u;
                const auto p00 = luisa::compute::min(source_origin, source_limit);
                const auto p10 = luisa::compute::min(source_origin + make_uint2(1u, 0u), source_limit);
                const auto p01 = luisa::compute::min(source_origin + make_uint2(0u, 1u), source_limit);
                const auto p11 = luisa::compute::min(source_origin + 1u, source_limit);
                destination.write(destination_pixel,
                                  (source.read(p00) + source.read(p10) +
                                   source.read(p01) + source.read(p11)) * 0.25f);
            });
            mipmap_shader_ = std::make_unique<MipmapShader>(std::move(shader));
        }
        for (uint32_t level = 1u; level < mip_levels; ++level) {
            *stream_ << (*mipmap_shader_)(image->view(level - 1u), image->view(level))
                            .dispatch(image->view(level).size());
        }
        return true;
    }

    bool resolve_multisample(std::span<const uint64_t> sample_keys,
                             const HostTextureBinding& target, bool synchronize_stream,
                             std::string* error) {
        if (sample_keys.size() != 4u || target.resident_key == 0u) {
            if (error != nullptr) *error = "Luisa multisample resolve requires four resident samples";
            return false;
        }
        std::array<Image<float>*, 4u> sources{};
        for (size_t i = 0u; i < sources.size(); ++i) {
            const auto found = resident_textures_.find(sample_keys[i]);
            if (found == resident_textures_.end() || found->second.kind != kResourceTexture2D) {
                if (error != nullptr) *error = "Luisa multisample source texture is unavailable";
                return false;
            }
            sources[i] = std::get_if<Image<float>>(found->second.resource.get());
            if (sources[i] == nullptr) {
                if (error != nullptr) *error = "Luisa multisample source must be a 2D texture";
                return false;
            }
        }
        const auto storage = pixel_storage(target.pixel_format);
        if (!storage) {
            if (error != nullptr) *error = "Luisa multisample target format is unsupported";
            return false;
        }
        auto* destination_resource = resident_texture(
            target.resident_key, kResourceTexture2D, *storage,
            make_uint3(target.width, target.height, 1u), target.mip_levels);
        auto* destination = destination_resource == nullptr
                                ? nullptr
                                : std::get_if<Image<float>>(destination_resource);
        if (destination == nullptr) {
            if (error != nullptr) *error = "Luisa multisample destination texture is unavailable";
            return false;
        }
        if (msaa_resolve_shader_ == nullptr) {
            auto shader = device_->compile<2>([](ImageFloat s0, ImageFloat s1,
                                                 ImageFloat s2, ImageFloat s3,
                                                 ImageFloat output) noexcept {
                const auto pixel = dispatch_id().xy();
                const auto destination = make_uint2(
                    pixel.x, dispatch_size().y - 1u - pixel.y);
                output.write(destination, (s0.read(pixel) + s1.read(pixel) +
                                           s2.read(pixel) + s3.read(pixel)) * 0.25f);
            });
            msaa_resolve_shader_ = std::make_unique<MsaaResolveShader>(std::move(shader));
        }
        *stream_ << (*msaa_resolve_shader_)(
                        *sources[0], *sources[1], *sources[2], *sources[3], *destination)
                        .dispatch(target.width, target.height);
        if (synchronize_stream)
            *stream_ << luisa::compute::synchronize();
        return true;
    }

    bool clear_multisample(std::span<const uint64_t> sample_keys,
                           const HostTextureBinding& target, float4 color,
                           std::string* error) {
        if (sample_keys.size() != 4u) {
            if (error != nullptr) *error = "Luisa multisample clear requires four resident samples";
            return false;
        }
        const auto storage = pixel_storage(target.pixel_format);
        if (!storage) {
            if (error != nullptr) *error = "Luisa multisample clear format is unsupported";
            return false;
        }
        std::array<Image<float>*, 4u> samples{};
        for (size_t i = 0u; i < samples.size(); ++i) {
            auto* resource = resident_texture(
                sample_keys[i], kResourceTexture2D, *storage,
                make_uint3(target.width, target.height, 1u), target.mip_levels);
            samples[i] = resource == nullptr ? nullptr : std::get_if<Image<float>>(resource);
            if (samples[i] == nullptr) {
                if (error != nullptr) *error = "Luisa multisample clear target is unavailable";
                return false;
            }
        }
        if (msaa_clear_shader_ == nullptr) {
            auto shader = device_->compile<2>([](ImageFloat s0, ImageFloat s1,
                                                 ImageFloat s2, ImageFloat s3,
                                                 Float4 value) noexcept {
                const auto pixel = dispatch_id().xy();
                s0.write(pixel, value);
                s1.write(pixel, value);
                s2.write(pixel, value);
                s3.write(pixel, value);
            });
            msaa_clear_shader_ = std::make_unique<MsaaClearShader>(std::move(shader));
        }
        *stream_ << (*msaa_clear_shader_)(
                        *samples[0], *samples[1], *samples[2], *samples[3], color)
                        .dispatch(target.width, target.height);
        return true;
    }

    bool load_multisample(std::span<const uint64_t> sample_keys,
                          const HostTextureBinding& target,
                          std::string* error) {
        if (sample_keys.size() != 4u || target.resident_key == 0u || target.bytes == nullptr) {
            if (error != nullptr) *error = "Luisa multisample load requires a resident source and four samples";
            return false;
        }
        const auto storage = pixel_storage(target.pixel_format);
        if (!storage) {
            if (error != nullptr) *error = "Luisa multisample load format is unsupported";
            return false;
        }
        bool source_created = false;
        auto* source_resource = resident_texture(
            target.resident_key, kResourceTexture2D, *storage,
            make_uint3(target.width, target.height, 1u), target.mip_levels, &source_created);
        auto* source = source_resource == nullptr ? nullptr : std::get_if<Image<float>>(source_resource);
        if (source == nullptr) {
            if (error != nullptr) *error = "Luisa multisample load source is unavailable";
            return false;
        }
        std::array<Image<float>*, 4u> samples{};
        for (size_t i = 0u; i < samples.size(); ++i) {
            auto* resource = resident_texture(
                sample_keys[i], kResourceTexture2D, *storage,
                make_uint3(target.width, target.height, 1u), target.mip_levels);
            samples[i] = resource == nullptr ? nullptr : std::get_if<Image<float>>(resource);
            if (samples[i] == nullptr) {
                if (error != nullptr) *error = "Luisa multisample load target is unavailable";
                return false;
            }
        }
        if (target.upload || source_created) *stream_ << source->copy_from(target.bytes->data());
        if (msaa_load_shader_ == nullptr) {
            auto shader = device_->compile<2>([](ImageFloat input, ImageFloat s0, ImageFloat s1,
                                                 ImageFloat s2, ImageFloat s3) noexcept {
                const auto pixel = dispatch_id().xy();
                const auto source_pixel = make_uint2(pixel.x, dispatch_size().y - 1u - pixel.y);
                const auto value = input.read(source_pixel);
                s0.write(pixel, value);
                s1.write(pixel, value);
                s2.write(pixel, value);
                s3.write(pixel, value);
            });
            msaa_load_shader_ = std::make_unique<MsaaLoadShader>(std::move(shader));
        }
        *stream_ << (*msaa_load_shader_)(
                        *source, *samples[0], *samples[1], *samples[2], *samples[3])
                        .dispatch(target.width, target.height);
        return true;
    }

    bool download_texture(uint64_t key, void* destination, size_t size, std::string* error) {
        const auto found = resident_textures_.find(key);
        if (found == resident_textures_.end() || destination == nullptr) {
            if (error != nullptr) *error = "Luisa resident texture is unavailable";
            return false;
        }
        bool size_matches = false;
        std::visit([&](auto& texture) {
            const auto expected = pixel_storage_size(found->second.storage, found->second.size);
            size_matches = expected == size;
            if (size_matches) *stream_ << texture.copy_to(destination);
        }, *found->second.resource);
        if (!size_matches) {
            if (error != nullptr) *error = "Luisa resident texture size changed";
            return false;
        }
        *stream_ << luisa::compute::synchronize();
        return true;
    }

    bool download_texture_async(uint64_t key, void* destination, size_t size,
                                std::function<void()> completion, std::string* error) {
        const auto found = resident_textures_.find(key);
        if (found == resident_textures_.end() || destination == nullptr || !completion) {
            if (error != nullptr) *error = "Luisa resident texture async download is unavailable";
            return false;
        }
        bool size_matches = false;
        std::visit([&](auto& texture) {
            const auto expected = pixel_storage_size(found->second.storage, found->second.size);
            size_matches = expected == size;
            if (size_matches) {
                *stream_ << texture.copy_to(destination)
                         << [completion = std::move(completion)]() mutable { completion(); };
            }
        }, *found->second.resource);
        if (!size_matches) {
            if (error != nullptr) *error = "Luisa resident texture async download size changed";
            return false;
        }
        return true;
    }

    PresentTarget* present_target(uint64_t key, uint64_t display, uint64_t window,
                                  uint32_t width, uint32_t height, bool vsync,
                                  std::string* error) {
        if (key == 0u || window == 0u || width == 0u || height == 0u) {
            if (error != nullptr) *error = "Luisa presenter requires a native window and positive dimensions";
            return nullptr;
        }
        if (backend_name_ != "metal" && backend_name_ != "vk" && backend_name_ != "dx") {
            if (error != nullptr) *error = "The selected Luisa backend does not expose swapchain presentation";
            return nullptr;
        }
        const auto found = present_targets_.find(key);
        if (found != present_targets_.end()) {
            const auto& target = *found->second;
            if (target.display == display && target.window == window &&
                target.size.x == width && target.size.y == height && target.vsync == vsync) {
                return found->second.get();
            }
            found->second->stream.synchronize();
            present_targets_.erase(found);
        }
        auto present_stream = device_->create_stream(StreamTag::GRAPHICS);
        auto swapchain = device_->create_swapchain(
            present_stream,
            SwapchainOption{.display = display,
                            .window = window,
                            .size = make_uint2(width, height),
                            .wants_hdr = false,
                            .wants_vsync = vsync,
                            .wants_transparent = false,
                            .back_buffer_count = 3u});
        if (!swapchain) {
            if (error != nullptr) *error = "Luisa failed to create a swapchain for the native window";
            return nullptr;
        }
        auto target = std::make_unique<PresentTarget>();
        target->display = display;
        target->window = window;
        target->size = make_uint2(width, height);
        target->vsync = vsync;
        target->stream = std::move(present_stream);
        target->swapchain = std::move(swapchain);
        target->handoff = device_->create_event();
        auto* result = target.get();
        present_targets_.emplace(key, std::move(target));
        return result;
    }

    bool present_resident_texture(uint64_t presenter_key, uint64_t texture_key,
                                  uint64_t display, uint64_t window,
                                  uint32_t width, uint32_t height, bool vsync,
                                  NativeTextureInfo* native_texture, std::string* error) {
        const auto found = resident_textures_.find(texture_key);
        auto* image = found == resident_textures_.end() ? nullptr :
                          std::get_if<Image<float>>(found->second.resource.get());
        if (image == nullptr || image->size().x != width || image->size().y != height) {
            if (error != nullptr) *error = "Luisa resident 2D texture is unavailable for presentation";
            return false;
        }
        auto* target = present_target(presenter_key, display, window, width, height, vsync, error);
        if (target == nullptr) return false;
        *stream_ << target->handoff.signal();
        target->stream << target->handoff.wait(target->handoff.last_fence())
                       << target->swapchain.present(image->view());
        if (native_texture != nullptr) {
            native_texture->handle = image->native_handle();
            native_texture->kind = backend_name_ == "metal" ? NativeTextureHandleKind::MetalTexture :
                                   backend_name_ == "vk" ? NativeTextureHandleKind::VulkanImage :
                                   backend_name_ == "dx" ? NativeTextureHandleKind::Direct3D12Resource :
                                                            NativeTextureHandleKind::Unknown;
        }
        return true;
    }

    bool present_host_texture(uint64_t presenter_key, const void* pixels, size_t size,
                              uint64_t display, uint64_t window,
                              uint32_t width, uint32_t height, bool vsync,
                              std::string* error) {
        if (pixels == nullptr || size != static_cast<size_t>(width) * height * sizeof(uint32_t)) {
            if (error != nullptr) *error = "Luisa host presentation requires a complete RGBA8 image";
            return false;
        }
        auto* target = present_target(presenter_key, display, window, width, height, vsync, error);
        if (target == nullptr) return false;
        if (!target->staging || target->staging.size().x != width || target->staging.size().y != height) {
            target->stream.synchronize();
            target->staging = device_->create_image<float>(PixelStorage::BYTE4, make_uint2(width, height));
        }
        target->stream << target->staging.copy_from(pixels)
                       << target->swapchain.present(target->staging.view());
        return true;
    }

    bool destroy_presenter(uint64_t key, std::string* error) {
        const auto found = present_targets_.find(key);
        if (found == present_targets_.end()) return true;
        found->second->stream.synchronize();
        present_targets_.erase(found);
        if (error != nullptr) error->clear();
        return true;
    }

    void reset() noexcept {
        if (stream_ != nullptr) {
            std::string ignored;
            (void)synchronize(&ignored);
        }
        for (auto& [key, fence] : fences_) {
            (void)key;
            fence.event = Event{};
        }
        streams_.clear();
        kernels_.clear();
        fragment_callables_.clear();
        raster_geometries_.clear();
        raster_shaders_.clear();
        fast_raster_shaders_.clear();
        fast_raster_resolve_shaders_.clear();
        tile_assembly_shaders_.clear();
        tile_raster_shaders_.clear();
        shared_tile_raster_shaders_.clear();
        fused_tile_raster_shaders_.clear();
        present_targets_.clear();
        accels_.clear();
        resident_buffers_.clear();
        resident_textures_.clear();
        mipmap_shader_.reset();
        fast_raster_init_shader_.reset();
        tile_reset_shader_.reset();
        tile_prefix_shader_.reset();
        tile_fill_shader_.reset();
        tile_fill_dispatch_buffer_.reset();
        msaa_resolve_shader_.reset();
        msaa_clear_shader_.reset();
        msaa_load_shader_.reset();
        stream_.reset();
        device_.reset();
        context_.reset();
        runtime_directory_.clear();
        backend_name_.clear();
        device_index_ = UINT32_MAX;
    }

    void abandon() noexcept {
        // Deliberately leak runtime objects during process teardown: their
        // destructors call into dynamically loaded Luisa/Vulkan code.
        (void)context_.release();
        (void)device_.release();
        (void)stream_.release();
        for (auto& [key, stream] : streams_) {
            (void)key;
            (void)stream.release();
        }
        streams_.clear();
        stream_keys_.clear();
        for (auto& [key, target] : present_targets_) {
            (void)key;
            (void)target->staging.release();
            (void)target->handoff.release();
            (void)target->swapchain.release();
            (void)target->stream.release();
        }
        present_targets_.clear();
        for (auto& [key, fence] : fences_) {
            (void)key;
            (void)fence.event.release();
        }
        fences_.clear();
        // The owner is intentionally leaked by runtime_registry(); leave cached
        // shaders and resources untouched so their destructors cannot run after
        // the dynamically loaded Luisa backend has been unloaded.
        runtime_directory_.clear();
        backend_name_.clear();
        device_index_ = UINT32_MAX;
    }
};

class RuntimeRegistry {
private:
    std::unordered_map<uint64_t, std::unique_ptr<RuntimeState>> states_;

public:
    RuntimeState& prepare(uint64_t context_key, std::string_view runtime_directory,
                          std::string_view backend_name, uint32_t device_index) {
        if (backend_name == "vk") {
            for (auto& [key, state] : states_) {
                if (key != context_key && state->has_device() && state->backend_name() == "vk") {
                    state->reset();
                }
            }
        }
        auto& state = states_[context_key];
        if (state == nullptr) state = std::make_unique<RuntimeState>();
        state->ensure(runtime_directory, backend_name, device_index);
        return *state;
    }

    RuntimeState* find(uint64_t context_key) noexcept {
        const auto found = states_.find(context_key);
        return found == states_.end() ? nullptr : found->second.get();
    }

    RuntimeState* find_active(std::string_view backend_name, uint32_t device_index) noexcept {
        for (auto& [key, state] : states_) {
            (void)key;
            if (state->has_device() && state->backend_name() == backend_name &&
                (state->device_index() == device_index ||
                 (state->device_index() == UINT32_MAX && device_index == 0u))) {
                return state.get();
            }
        }
        return nullptr;
    }

    bool has_active(std::string_view backend_name) const noexcept {
        return std::any_of(states_.begin(), states_.end(), [backend_name](const auto& entry) {
            return entry.second->has_device() && entry.second->backend_name() == backend_name;
        });
    }

    void erase(uint64_t context_key) {
        const auto found = states_.find(context_key);
        if (found == states_.end()) return;
        found->second->reset();
        states_.erase(found);
    }

    void reset() noexcept {
        for (auto& [key, state] : states_) {
            (void)key;
            state->reset();
        }
        states_.clear();
    }

    void abandon() noexcept {
        for (auto& [key, state] : states_) {
            (void)key;
            state->abandon();
        }
        states_.clear();
    }
};

RuntimeRegistry &runtime_registry() {
    // The native backend may be unloaded before C++ static destructors run.
    // Keep the owner itself alive until process exit; explicit Shutdown handles
    // normal context teardown and Abandon releases only the dynamically loaded
    // runtime objects on the process-exit path.
    static auto *registry = new RuntimeRegistry();
    return *registry;
}

uint64_t next_runtime_key() noexcept {
    static std::atomic<uint64_t> counter{1000u};
    return counter.fetch_add(1u, std::memory_order_relaxed);
}

std::recursive_mutex& runtime_mutex() noexcept {
    // LC devices, streams, command lists, and Feather's resident caches are not
    // safe for concurrent host mutation. Keep submissions serialized while the
    // commands themselves remain asynchronous on their selected LC streams.
    static auto* mutex = new std::recursive_mutex{};
    return *mutex;
}

} // namespace

void Shutdown() {
    std::scoped_lock lock{runtime_mutex()};
    runtime_registry().reset();
}

void Shutdown(uint64_t context_key) {
    std::scoped_lock lock{runtime_mutex()};
    runtime_registry().erase(context_key);
}

bool CreateStream(uint64_t context_key, std::string_view runtime_directory, std::string_view backend_name,
                  uint32_t device_index, uint64_t stream_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto& state = runtime_registry().prepare(context_key, runtime_directory, backend_name, device_index);
    return state.create_stream(stream_key, error);
}

bool DestroyStream(uint64_t context_key, uint64_t stream_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr)
            *error = "Luisa context has no runtime state";
        return false;
    }
    return state->destroy_stream(stream_key, error);
}

bool CreateAccel(uint64_t context_key, std::string_view runtime_directory,
                 std::string_view backend_name, uint32_t device_index,
                 std::span<const AccelMeshDesc> meshes,
                 uint64_t* accel_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) { error->clear(); }
    if (accel_key == nullptr) {
        if (error != nullptr) { *error = "accel key output is null"; }
        return false;
    }
    if (meshes.empty()) {
        if (error != nullptr) { *error = "accel requires at least one mesh"; }
        return false;
    }
    auto& state = runtime_registry().prepare(
        context_key, runtime_directory, backend_name, device_index);
    auto& device = state.device();
    auto& stream = state.stream();
    auto accel_state = std::make_unique<RuntimeState::AccelState>();
    accel_state->vertex_buffers.reserve(meshes.size());
    accel_state->index_buffers.reserve(meshes.size());
    accel_state->meshes.reserve(meshes.size());
    for (auto& mesh_desc : meshes) {
        if (mesh_desc.vertex_count == 0u || mesh_desc.vertices == nullptr ||
            mesh_desc.index_count == 0u || mesh_desc.indices == nullptr ||
            mesh_desc.index_count % 3u != 0u) {
            if (error != nullptr) {
                *error = "accel mesh requires float3 vertices and uint3 index triplets";
            }
            return false;
        }
        // FEIR carries tightly packed float3 vertices (12-byte stride), but
        // luisa::compute::float3 is 16-byte aligned SIMD storage; casting the
        // packed bytes as float3 would misalign every vertex, so pack them.
        std::vector<float3> packed_vertices;
        packed_vertices.reserve(mesh_desc.vertex_count);
        {
            const auto* src = reinterpret_cast<const float*>(mesh_desc.vertices);
            for (uint32_t i = 0u; i < mesh_desc.vertex_count; ++i) {
                packed_vertices.emplace_back(src[i * 3u], src[i * 3u + 1u], src[i * 3u + 2u]);
            }
        }
        auto vertex_buffer = device.create_buffer<float3>(mesh_desc.vertex_count);
        stream << vertex_buffer.copy_from(luisa::span<const float3>{packed_vertices})
               << synchronize();
        auto index_buffer = device.create_buffer<Triangle>(mesh_desc.index_count / 3u);
        stream << index_buffer.copy_from(
                      luisa::span<const Triangle>{
                          reinterpret_cast<const Triangle*>(mesh_desc.indices),
                          mesh_desc.index_count / 3u})
               << synchronize();
        auto mesh = device.create_mesh(vertex_buffer, index_buffer);
        stream << mesh.build() << synchronize();
        accel_state->vertex_buffers.emplace_back(
            std::make_unique<Buffer<float3>>(std::move(vertex_buffer)));
        accel_state->index_buffers.emplace_back(
            std::make_unique<Buffer<Triangle>>(std::move(index_buffer)));
        accel_state->meshes.emplace_back(
            std::make_unique<Mesh>(std::move(mesh)));
    }
    auto accel = device.create_accel();
    for (auto& mesh : accel_state->meshes) {
        accel.emplace_back(*mesh);
    }
    stream << accel.build() << synchronize();
    accel_state->accel = std::make_unique<Accel>(std::move(accel));
    auto key = next_runtime_key();
    state.accels_.emplace(key, std::move(accel_state));
    *accel_key = key;
    return true;
}

bool DestroyAccel(uint64_t context_key, uint64_t accel_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) { error->clear(); }
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) { *error = "Luisa context has no runtime state"; }
        return false;
    }
    if (state->accels_.erase(accel_key) == 0u) {
        if (error != nullptr) { *error = "accel key not found"; }
        return false;
    }
    return true;
}

bool Synchronize(uint64_t context_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    return state == nullptr || state->synchronize(error);
}

bool SynchronizeStream(uint64_t context_key, uint64_t stream_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr)
            *error = "Luisa context has no runtime state";
        return false;
    }
    return state->synchronize_stream(stream_key, error);
}

bool WaitFence(uint64_t context_key, uint64_t stream_key, uint64_t fence_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr)
            *error = "Luisa context has no runtime state";
        return false;
    }
    return state->wait_fence(stream_key, fence_key, error);
}

bool IsFenceCompleted(uint64_t context_key, uint64_t fence_key, bool* completed, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr)
            *error = "Luisa context has no runtime state";
        return false;
    }
    return state->is_fence_completed(fence_key, completed, error);
}

bool SynchronizeFence(uint64_t context_key, uint64_t fence_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr)
            *error = "Luisa context has no runtime state";
        return false;
    }
    return state->synchronize_fence(fence_key, error);
}

bool DestroyFence(uint64_t context_key, uint64_t fence_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    auto* state = runtime_registry().find(context_key);
    if (state == nullptr)
        return true;
    return state->destroy_fence(fence_key, error);
}

void Abandon() noexcept {
    std::scoped_lock lock{runtime_mutex()};
    runtime_registry().abandon();
}

bool DownloadResidentTexture(uint64_t context_key, uint64_t resident_key, void* destination, size_t size,
                             std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->download_texture(resident_key, destination, size, error);
}

bool DownloadResidentTextureAsync(uint64_t context_key, uint64_t resident_key, void* destination, size_t size,
                                  std::function<void()> completion, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->download_texture_async(
        resident_key, destination, size, std::move(completion), error);
}

bool PresentResidentTexture(uint64_t context_key, uint64_t presenter_key, uint64_t resident_key,
                            uint64_t native_display, uint64_t native_window,
                            uint32_t width, uint32_t height, bool vsync,
                            NativeTextureInfo* native_texture, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->present_resident_texture(
        presenter_key, resident_key, native_display, native_window,
        width, height, vsync, native_texture, error);
}

bool PresentHostTexture(uint64_t context_key, std::string_view runtime_directory,
                        std::string_view backend_name, uint32_t device_index,
                        uint64_t presenter_key, const void* pixels, size_t size,
                        uint64_t native_display, uint64_t native_window,
                        uint32_t width, uint32_t height, bool vsync,
                        std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    auto& state = runtime_registry().prepare(
        context_key, runtime_directory, backend_name, device_index);
    return state.present_host_texture(
        presenter_key, pixels, size, native_display, native_window,
        width, height, vsync, error);
}

bool DestroyPresenter(uint64_t context_key, uint64_t presenter_key, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) return true;
    return state->destroy_presenter(presenter_key, error);
}

bool ResolveMultisampleTexture(uint64_t context_key, std::span<const uint64_t> sample_keys,
                               const HostTextureBinding& target,
                               bool synchronize, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->resolve_multisample(sample_keys, target, synchronize, error);
}

bool ClearMultisampleTexture(uint64_t context_key, std::span<const uint64_t> sample_keys,
                             const HostTextureBinding& target,
                             const std::array<float, 4u>& color,
                             std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->clear_multisample(
        sample_keys, target, make_float4(color[0], color[1], color[2], color[3]), error);
}

bool LoadMultisampleTexture(uint64_t context_key, std::span<const uint64_t> sample_keys,
                            const HostTextureBinding& target,
                            std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto state = runtime_registry().find(context_key);
    if (state == nullptr) {
        if (error != nullptr) *error = "Luisa context has no resident state";
        return false;
    }
    return state->load_multisample(sample_keys, target, error);
}

std::string RuntimeDirectory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&RuntimeDirectory), &module) == 0) {
        return {};
    }
    std::vector<char> path(MAX_PATH);
    for (;;) {
        const auto size = GetModuleFileNameA(module, path.data(), static_cast<DWORD>(path.size()));
        if (size == 0)
            return {};
        if (size < path.size() - 1u) {
            return resolve_runtime_directory(std::filesystem::path{std::string_view{path.data(), size}});
        }
        path.resize(path.size() * 2u);
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&RuntimeDirectory), &info) == 0 || info.dli_fname == nullptr) {
        return {};
    }
    return resolve_runtime_directory(std::filesystem::path{info.dli_fname});
#endif
}

std::vector<DeviceInfo> EnumerateDevices(std::string_view runtime_directory) {
    static std::mutex mutex;
    static std::string cached_runtime_directory;
    static std::vector<DeviceInfo> cached_devices;
    static bool initialized = false;
    std::lock_guard lock{mutex};
    if (initialized && cached_runtime_directory == runtime_directory) return cached_devices;

    Context context{runtime_directory};
    const auto installed = context.installed_backends();
    constexpr std::array<std::string_view, 4u> supported_backends{"metal", "vk", "cuda", "hip"};
    std::vector<DeviceInfo> devices;
    for (const auto backend : supported_backends) {
        const auto available = std::any_of(installed.begin(), installed.end(), [backend](const auto& name) {
            return std::string_view{name.data(), name.size()} == backend;
        });
        if (!available) continue;
        const auto names = context.backend_device_names(backend);
        for (uint32_t index = 0u; index < names.size(); ++index) {
            devices.emplace_back(DeviceInfo{
                std::string{backend}, std::string{names[index]}, index, 0u});
        }
    }
    cached_runtime_directory = runtime_directory;
    cached_devices = std::move(devices);
    initialized = true;
    return cached_devices;
}

bool ValidateDevice(std::string_view runtime_directory, std::string_view backend_name,
                    uint32_t device_index, DeviceInfo* info, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    const auto devices = EnumerateDevices(runtime_directory);
    const auto selected = std::find_if(devices.begin(), devices.end(), [&](const auto& candidate) {
        return candidate.backend_name == backend_name && candidate.device_index == device_index;
    });
    if (selected == devices.end()) {
        if (error != nullptr) {
            *error = "Luisa device index " + std::to_string(device_index) +
                     " is not available for backend '" + std::string{backend_name} + "'.";
        }
        return false;
    }

    if (auto state = runtime_registry().find_active(backend_name, device_index)) {
        if (info != nullptr) {
            *info = *selected;
            info->compute_warp_size = state->device().compute_warp_size();
        }
        return true;
    }
    if (backend_name == "vk" && runtime_registry().has_active("vk")) {
        if (info != nullptr) *info = *selected;
        return true;
    }

    Context context{runtime_directory};
    DeviceConfig config{};
    config.binary_io = &process_binary_io(runtime_directory);
    config.device_index = device_index;
    auto device = context.create_device(backend_name, &config);
    if (!device) {
        if (error != nullptr) {
            *error = "Luisa failed to create device " + std::to_string(device_index) +
                     " for backend '" + std::string{backend_name} + "'.";
        }
        return false;
    }
    if (info != nullptr) {
        *info = *selected;
        info->compute_warp_size = device.compute_warp_size();
    }
    return true;
}

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> host_buffers, std::span<HostTextureBinding> host_textures,
              const DispatchInputs& dispatch, const AdInputs* ad_inputs,
              std::span<AdGradientBinding> gradients,
              std::span<HostAccelBinding> host_accels, std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr)
        error->clear();
    if (dispatch.execution_skipped != nullptr) *dispatch.execution_skipped = false;
    if (!dispatch.synchronize && !gradients.empty()) {
        if (error != nullptr) *error = "asynchronous Luisa dispatch cannot return gradients";
        return false;
    }
    auto &state = runtime_registry().prepare(
        dispatch.context_key, dispatch.runtime_directory, dispatch.backend_name, dispatch.device_index);
    auto *cached = dispatch.shader_cache_key == 0u ? nullptr : state.find(dispatch.shader_cache_key);
    const auto inputs_clean = std::none_of(
                                  host_buffers.begin(), host_buffers.end(),
                                  [](const auto& binding) { return binding.upload; }) &&
                              std::none_of(
                                  host_textures.begin(), host_textures.end(),
                                  [](const auto& binding) { return binding.upload; });
    if (dispatch.reuse_if_inputs_clean && cached != nullptr && cached->shader != nullptr &&
        inputs_clean && cached->execution_cache_key == dispatch.execution_cache_key &&
        ad_inputs == nullptr && gradients.empty()) {
        if (dispatch.execution_skipped != nullptr) *dispatch.execution_skipped = true;
        return true;
    }
    ensure_luisa_spirv_optimization_preset();
    auto xir_module = std::make_shared<xir::Module>();
    std::vector<BufferLayout> buffer_layouts;
    std::vector<AdGradientLayout> gradient_layouts;
    auto* kernel = LowerToXir(module, lowering, *xir_module, &buffer_layouts, ad_inputs, &gradient_layouts, error);
    if (kernel == nullptr)
        return false;
    auto verification = xir_verify_module(xir_module.get());
    if (!verification.succeeded()) {
        if (error != nullptr) {
            *error = "generated Luisa XIR failed verification: ";
            error->append(verification.errors.front().message.data(), verification.errors.front().message.size());
            if (const auto* instruction = verification.errors.front().instruction;
                instruction != nullptr && instruction->isa<xir::ArithmeticInst>()) {
                const auto* arithmetic = static_cast<const xir::ArithmeticInst*>(instruction);
                error->append(" [FEDIAG op=");
                error->append(luisa::to_string(arithmetic->op()));
                error->append(" result=");
                error->append(arithmetic->type() == nullptr ? "<null>" : arithmetic->type()->description());
                for (auto operand_use : arithmetic->operand_uses()) {
                    const auto* value = operand_use->value();
                    error->append(" arg=");
                    error->append(value == nullptr || value->type() == nullptr
                                      ? "<null>"
                                      : value->type()->description());
                }
                error->append("]");
            }
        }
        return false;
    }

    auto &device = state.device();
    auto* dispatch_stream = state.stream(dispatch.stream_key);
    if (dispatch_stream == nullptr) {
        if (error != nullptr)
            *error = "Luisa dispatch stream handle is invalid";
        return false;
    }
    auto& stream = *dispatch_stream;
    bool cache_hit = cached != nullptr && cached->shader != nullptr;
    std::vector<ByteBuffer *> runtime_buffers;
    std::vector<std::unique_ptr<ByteBuffer>> owned_buffers;
    std::vector<ByteBuffer *> runtime_push_constants;
    std::vector<std::unique_ptr<ByteBuffer>> owned_push_constants;
    std::vector<std::vector<unsigned char>> staged_push_constants;
    std::vector<std::vector<unsigned char>> staged_bytes;
    std::vector<HostBufferBinding*> staged_bindings;
    std::vector<luisa::compute::Function::Binding> bound_arguments;
    std::vector<RuntimeTexture *> runtime_textures;
    std::vector<RuntimeTexture> owned_textures;
    std::vector<HostTextureBinding*> staged_textures;
    std::vector<ByteBuffer *> runtime_gradients;
    std::vector<std::unique_ptr<ByteBuffer>> owned_gradients;
    std::vector<std::vector<unsigned char>> staged_gradients;
    runtime_buffers.reserve(host_buffers.size());
    owned_buffers.reserve(host_buffers.size());
    runtime_push_constants.reserve(lowering.push_constants.size());
    owned_push_constants.reserve(lowering.push_constants.size());
    staged_push_constants.reserve(lowering.push_constants.size());
    staged_bytes.reserve(host_buffers.size());
    staged_bindings.reserve(host_buffers.size());
    bound_arguments.reserve(lowering.resources.size());
    runtime_textures.reserve(host_textures.size());
    owned_textures.reserve(host_textures.size());
    staged_textures.reserve(host_textures.size());
    runtime_gradients.reserve(gradient_layouts.size());
    owned_gradients.reserve(gradient_layouts.size());
    staged_gradients.reserve(gradient_layouts.size());

    size_t buffer_index = 0;
    size_t push_constant_index = 0;
    size_t texture_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind == kResourcePushConstant && lowering.dynamic_push_constants) {
            const auto found = std::find_if(
                lowering.push_constants.begin(), lowering.push_constants.end(),
                [&](const auto& push) { return push.binding == resource.binding; });
            const auto layout = std::find_if(
                buffer_layouts.begin(), buffer_layouts.end(),
                [&](const auto& candidate) { return candidate.binding == resource.binding; });
            if (found == lowering.push_constants.end() || found->data == nullptr ||
                layout == buffer_layouts.end() || layout->device_type == nullptr) {
                if (error != nullptr) *error = "Luisa dynamic push constant layout is missing";
                return false;
            }
            staged_push_constants.emplace_back(layout->device_type->size(), 0u);
            auto& packed = staged_push_constants.back();
            if (!repack_value(module, layout->feir_type_id, layout->device_type,
                              static_cast<const unsigned char*>(found->data), packed.data(), true)) {
                if (error != nullptr) *error = "Luisa failed to repack a dynamic push constant";
                return false;
            }
            ByteBuffer* runtime = nullptr;
            if (cache_hit) {
                if (push_constant_index >= cached->push_constants.size() ||
                    cached->push_constants[push_constant_index]->size_bytes() != packed.size()) {
                    if (error != nullptr) *error = "Luisa shader cache push constant layout changed";
                    return false;
                }
                runtime = cached->push_constants[push_constant_index];
            } else {
                owned_push_constants.emplace_back(
                    std::make_unique<ByteBuffer>(device.create_byte_buffer(packed.size())));
                runtime = owned_push_constants.back().get();
            }
            stream << runtime->copy_from(packed.data());
            if (!cache_hit) {
                bound_arguments.emplace_back(
                    luisa::compute::Function::BufferBinding{runtime->handle(), 0u, runtime->size_bytes()});
            }
            runtime_push_constants.push_back(runtime);
            ++push_constant_index;
            continue;
        }
        if (resource.kind == kResourceAccel) {
            auto found = std::find_if(host_accels.begin(), host_accels.end(),
                                      [&](const auto& binding) { return binding.binding == resource.binding; });
            if (found == host_accels.end()) {
                if (error != nullptr) *error = "Luisa accel binding is missing";
                return false;
            }
            auto accel_state = state.accels_.find(found->accel_key);
            if (accel_state == state.accels_.end() || accel_state->second->accel == nullptr) {
                if (error != nullptr) {
                    *error = "Luisa accel binding references an unknown acceleration structure (key " +
                             std::to_string(found->accel_key) + ", " + std::to_string(state.accels_.size()) +
                             " accels registered, context_key " + std::to_string(dispatch.context_key) + ")";
                }
                return false;
            }
            if (!cache_hit) {
                bound_arguments.emplace_back(
                    luisa::compute::Function::AccelBinding{
                        accel_state->second->accel->handle()});
            }
            continue;
        }
        if (resource.kind == kResourceTexture2D || resource.kind == kResourceTexture3D) {
            auto found = std::find_if(host_textures.begin(), host_textures.end(),
                                      [&](const auto& binding) { return binding.binding == resource.binding; });
            auto storage = found == host_textures.end() ? std::nullopt : pixel_storage(found->pixel_format);
            if (found == host_textures.end() || found->bytes == nullptr || found->bytes->empty() || !storage) {
                if (error != nullptr) *error = "Luisa texture binding is missing or uses an unsupported pixel format";
                return false;
            }
            RuntimeTexture *runtime = nullptr;
            bool resident_created = false;
            if (cache_hit) {
                if (texture_index >= cached->textures.size()) {
                    if (error != nullptr) *error = "Luisa shader cache resource layout changed";
                    return false;
                }
                runtime = cached->textures[texture_index];
            } else if (found->resident_key != 0u) {
                runtime = state.resident_texture(found->resident_key, resource.kind, *storage,
                                                 make_uint3(found->width, found->height, found->depth),
                                                 found->mip_levels, &resident_created);
                if (runtime == nullptr) {
                    if (error != nullptr) *error = "Luisa resident texture layout changed";
                    return false;
                }
            } else if (resource.kind == kResourceTexture2D) {
                owned_textures.emplace_back(device.create_image<float>(
                    *storage, luisa::make_uint2(found->width, found->height), found->mip_levels, true));
                runtime = &owned_textures.back();
            } else {
                owned_textures.emplace_back(device.create_volume<float>(
                    *storage, luisa::make_uint3(found->width, found->height, found->depth), found->mip_levels, true));
                runtime = &owned_textures.back();
            }
            const auto upload = found->upload || resident_created;
            runtime_textures.push_back(runtime);
            std::visit(
                [&](auto& texture) {
                    if (upload)
                        stream << texture.copy_from(found->bytes->data());
                    if (!cache_hit)
                        bound_arguments.emplace_back(luisa::compute::Function::TextureBinding{texture.handle(), 0u});
                },
                *runtime);
            if (upload && found->generate_mipmaps && !state.generate_mipmaps(*runtime, found->mip_levels, error)) {
                return false;
            }
            staged_textures.push_back(&*found);
            ++texture_index;
            continue;
        }
        if (resource.kind != kResourceBuffer)
            continue;
        auto found = std::find_if(host_buffers.begin(), host_buffers.end(),
                                  [&](const auto& binding) { return binding.binding == resource.binding; });
        if (found == host_buffers.end() || found->bytes == nullptr || found->stride == 0 || found->bytes->empty() ||
            found->bytes->size() % found->stride != 0) {
            if (error != nullptr)
                *error = "Luisa buffer binding is missing or has an unsupported stride";
            return false;
        }
        const auto layout = std::find_if(buffer_layouts.begin(), buffer_layouts.end(),
                                         [&](const auto& candidate) { return candidate.binding == resource.binding; });
        if (layout == buffer_layouts.end() || layout->device_type == nullptr) {
            if (error != nullptr) *error = "Luisa buffer layout metadata is missing";
            return false;
        }
        const auto count = found->bytes->size() / found->stride;
        const auto device_size = count * layout->device_type->size();
        ByteBuffer *runtime = nullptr;
        bool resident_created = false;
        if (cache_hit) {
            if (buffer_index >= cached->buffers.size() ||
                cached->buffers[buffer_index]->size_bytes() != device_size) {
                if (error != nullptr) *error = "Luisa shader cache buffer layout changed";
                return false;
            }
            runtime = cached->buffers[buffer_index];
        } else if (found->resident_key != 0u) {
            runtime = state.resident_buffer(found->resident_key, device_size, &resident_created);
            if (runtime == nullptr) {
                if (error != nullptr) *error = "Luisa resident buffer layout changed";
                return false;
            }
        } else {
            owned_buffers.emplace_back(std::make_unique<ByteBuffer>(device.create_byte_buffer(device_size)));
            runtime = owned_buffers.back().get();
        }
        const auto upload = found->upload || resident_created;
        staged_bytes.emplace_back(upload || found->download ? device_size : 0u, 0u);
        auto& packed = staged_bytes.back();
        if (upload) {
            for (size_t i = 0; i < count; ++i) {
                if (!repack_value(module, layout->feir_type_id, layout->device_type,
                                  found->bytes->data() + i * found->stride,
                                  packed.data() + i * layout->device_type->size(), true)) {
                    if (error != nullptr)
                        *error = "Luisa failed to repack a Feather buffer element";
                    return false;
                }
            }
            stream << runtime->copy_from(packed.data());
        }
        if (!cache_hit)
            bound_arguments.emplace_back(
                luisa::compute::Function::BufferBinding{runtime->handle(), 0u, runtime->size_bytes()});
        runtime_buffers.push_back(runtime);
        staged_bindings.push_back(&*found);
        ++buffer_index;
    }

    size_t gradient_index = 0;
    for (const auto& layout : gradient_layouts) {
        auto found = std::find_if(gradients.begin(), gradients.end(), [&](const auto& gradient) {
            return gradient.source_binding == layout.source_binding;
        });
        if (found == gradients.end() || found->bytes == nullptr || found->element_count != layout.element_count ||
            found->component_count == 0u || layout.device_type == nullptr) {
            if (error != nullptr) *error = "Luisa AD gradient output metadata is missing or inconsistent";
            return false;
        }
        const auto value_count = static_cast<size_t>(dispatch.logical_x) * layout.element_count;
        staged_gradients.emplace_back(value_count * layout.device_type->size(), 0u);
        ByteBuffer *runtime = nullptr;
        if (cache_hit) {
            if (gradient_index >= cached->gradients.size() ||
                cached->gradients[gradient_index]->size_bytes() != staged_gradients.back().size()) {
                if (error != nullptr) *error = "Luisa shader cache gradient layout changed";
                return false;
            }
            runtime = cached->gradients[gradient_index];
        } else {
            owned_gradients.emplace_back(
                std::make_unique<ByteBuffer>(device.create_byte_buffer(staged_gradients.back().size())));
            runtime = owned_gradients.back().get();
        }
        stream << runtime->copy_from(staged_gradients.back().data());
        if (!cache_hit)
            bound_arguments.emplace_back(
                luisa::compute::Function::BufferBinding{runtime->handle(), 0u, runtime->size_bytes()});
        runtime_gradients.push_back(runtime);
        ++gradient_index;
    }

    if (!cache_hit && ad_inputs == nullptr) {
        // The XIR-to-AST translator applies the kernel's bound arguments positionally to every
        // function it translates, so a callable's own parameters would be mistaken for resource
        // bindings. Destructuring before inlining permits multi-block callable bodies to be
        // inlined into the kernel, leaving only the function the bindings actually describe.
        auto destructured = destructure_cfg_pass_run_on_module(xir_module.get());
        if (destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure XIR control flow before callable inlining";
            return false;
        }
        auto inlined = inline_all_pass_run_on_module(xir_module.get());
        if (inlined.skipped_recursive_callable_count != 0u ||
            inlined.skipped_structured_call_count != 0u ||
            inlined.skipped_constrained_call_count != 0u ||
            inlined.skipped_metadata_call_count != 0u ||
            inlined.skipped_declaration_call_count != 0u ||
            inlined.rejected_malformed_call_count != 0u) {
            if (error != nullptr) *error = "Luisa could not inline the generated FEIR callable graph";
            return false;
        }
        // Inlined FEIR callable graphs carry the full weight of the generated
        // C# kernel, including loop-unrolled bodies with duplicated
        // expressions and dead stores. Simplify before structural
        // normalization: GVN collapses duplicated computation, DCE removes
        // dead instructions, and CFG simplification merges empty blocks, which
        // keeps the restructure pass working on a much smaller module.
        static_cast<void>(gvn_pass_run_on_module(xir_module.get()));
        static_cast<void>(dce_pass_run_on_module(xir_module.get()));
        static_cast<void>(simplify_cfg_pass_run_on_module(xir_module.get()));
        static_cast<void>(dce_pass_run_on_module(xir_module.get()));
        xir_to_ast_normalize_module(xir_module.get());
        auto inlined_verification = xir_verify_module(xir_module.get());
        if (!inlined_verification.succeeded()) {
            if (error != nullptr) {
                *error = "Luisa inlined XIR failed verification: ";
                error->append(inlined_verification.errors.front().message.data(),
                              inlined_verification.errors.front().message.size());
            }
            return false;
        }
    }
    if (!cache_hit && ad_inputs != nullptr) {
        auto preinline_destructured = destructure_cfg_pass_run_on_module(xir_module.get());
        if (preinline_destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure XIR control flow before AD callable inlining";
            return false;
        }
        auto inlined =
            inline_all_pass_run_on_module(xir_module.get(), InlineOptions{.allow_autodiff_scope_in_caller = true});
        if (inlined.skipped_recursive_callable_count != 0u ||
            inlined.skipped_structured_call_count != 0u ||
            inlined.skipped_constrained_call_count != 0u ||
            inlined.rejected_malformed_call_count != 0u) {
            if (error != nullptr) *error = "Luisa could not inline the complete FEIR callable graph before autodiff";
            return false;
        }
        // Same simplification rationale as the non-AD path: the inlined
        // callable graph is huge (loop-unrolled C# kernel bodies), and the
        // restructure pass before autodiff stays far cheaper on the cleaned
        // module.
        static_cast<void>(gvn_pass_run_on_module(xir_module.get()));
        static_cast<void>(dce_pass_run_on_module(xir_module.get()));
        static_cast<void>(simplify_cfg_pass_run_on_module(xir_module.get()));
        static_cast<void>(dce_pass_run_on_module(xir_module.get()));
        auto destructured = destructure_cfg_pass_run_on_module(xir_module.get());
        if (destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure XIR control flow before autodiff";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(xir_module.get()));
        auto restructured = restructure_cfg_pass_run_on_module(xir_module.get());
        if (!restructured.succeeded()) {
            if (error != nullptr) *error = "Luisa failed to restructure XIR control flow before autodiff";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(xir_module.get()));
        auto ad = autodiff_pass_run_on_module(xir_module.get());
        if (ad.transformed_scope_count == 0u) {
            if (error != nullptr) *error = "Luisa XIR autodiff did not transform the generated AD scope";
            return false;
        }
        auto ad_verification = xir_verify_module(xir_module.get());
        if (!ad_verification.succeeded()) {
            if (error != nullptr) {
                *error = "Luisa autodiff output failed XIR verification: ";
                error->append(ad_verification.errors.front().message.data(), ad_verification.errors.front().message.size());
            }
            return false;
        }
        auto ad_destructured = destructure_cfg_pass_run_on_module(xir_module.get());
        if (ad_destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure autodiff output for AST translation";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(xir_module.get()));
        auto ad_restructured = restructure_cfg_pass_run_on_module(xir_module.get());
        if (!ad_restructured.succeeded()) {
            if (error != nullptr) *error = "Luisa failed to restructure autodiff output for AST translation";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(xir_module.get()));
    }
    std::unique_ptr<Shader3D<>> shader;
    if (!cache_hit) {
        auto ast = xir_to_ast_translate(
            *kernel, XIR2ASTConfig{.strict = true,
                                   .bound_arguments = luisa::span<const luisa::compute::Function::Binding>{
                                       bound_arguments.data(), bound_arguments.size()}});
        if (ast == nullptr) {
            if (error != nullptr)
                *error = "Luisa failed to translate generated XIR to its executable AST";
            return false;
        }
        shader = std::make_unique<Shader3D<>>(
            device.create<Shader3D<>>(luisa::compute::Function{ast.get()}, ShaderOption{}));
        if (dispatch.shader_cache_key != 0u) {
            auto entry = std::make_unique<RuntimeState::CachedKernel>();
            entry->shader = std::move(shader);
            entry->buffers = runtime_buffers;
            entry->push_constants = runtime_push_constants;
            entry->textures = runtime_textures;
            entry->gradients = runtime_gradients;
            entry->owned_buffers = std::move(owned_buffers);
            entry->owned_push_constants = std::move(owned_push_constants);
            entry->owned_textures = std::move(owned_textures);
            entry->owned_gradients = std::move(owned_gradients);
            state.insert(dispatch.shader_cache_key, std::move(entry));
            cached = state.find(dispatch.shader_cache_key);
            cache_hit = true;
        }
    }
    auto &cached_shader = cache_hit ? *cached->shader : *shader;
    stream << cached_shader().dispatch(luisa::make_uint3(dispatch.logical_x, dispatch.logical_y, dispatch.logical_z));
    if (cached != nullptr) cached->execution_cache_key = dispatch.execution_cache_key;

    size_t staged_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceBuffer)
            continue;
        auto* found = staged_bindings[staged_index];
        auto* runtime = runtime_buffers[staged_index++];
        if ((resource.access != kAccessWrite && resource.access != kAccessReadWrite) || !found->download)
            continue;
        auto& packed = staged_bytes[staged_index - 1u];
        stream << runtime->copy_to(packed.data());
    }
    size_t output_texture_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceTexture2D && resource.kind != kResourceTexture3D) continue;
        auto* found = staged_textures[output_texture_index];
        auto* runtime = runtime_textures[output_texture_index++];
        if ((resource.access != kAccessWrite && resource.access != kAccessReadWrite) || !found->download) continue;
        std::visit([&](auto& texture) {
            stream << texture.copy_to(found->bytes->data());
        }, *runtime);
    }
    for (size_t i = 0; i < gradient_layouts.size(); ++i) {
        auto& packed = staged_gradients[i];
        stream << runtime_gradients[i]->copy_to(packed.data());
    }
    if (!dispatch.synchronize) {
        if (dispatch.fence_key == 0u) {
            if (error != nullptr)
                *error = "asynchronous Luisa dispatch requires a fence handle";
            return false;
        }
        auto resources = lowering.resources;
        auto bindings = std::vector<HostBufferBinding>{host_buffers.begin(), host_buffers.end()};
        auto completion = [typed_module = xir_module, source_module = std::make_shared<TypedIR::Module>(module),
                           resources = std::move(resources), layouts = std::move(buffer_layouts),
                           bindings = std::move(bindings), staged = std::move(staged_bytes),
                           push_constants = std::move(staged_push_constants), buffers = std::move(owned_buffers),
                           owned_push = std::move(owned_push_constants), textures = std::move(owned_textures),
                           gradients = std::move(owned_gradients),
                           uncached_shader = std::move(shader)](std::string* completion_error) mutable {
            (void)typed_module;
            (void)push_constants;
            (void)buffers;
            (void)owned_push;
            (void)textures;
            (void)gradients;
            (void)uncached_shader;
            size_t staged_index = 0u;
            for (const auto& resource : resources) {
                if (resource.kind != kResourceBuffer)
                    continue;
                const auto found = std::find_if(bindings.begin(), bindings.end(), [&](const auto& binding) {
                    return binding.binding == resource.binding;
                });
                if (found == bindings.end() || staged_index >= staged.size()) {
                    if (completion_error != nullptr) {
                        *completion_error = "Luisa asynchronous buffer completion metadata is missing";
                    }
                    return false;
                }
                auto& packed = staged[staged_index++];
                if ((resource.access != kAccessWrite && resource.access != kAccessReadWrite) || !found->download) {
                    continue;
                }
                const auto layout = std::find_if(layouts.begin(), layouts.end(), [&](const auto& candidate) {
                    return candidate.binding == resource.binding;
                });
                if (layout == layouts.end() || found->bytes == nullptr) {
                    if (completion_error != nullptr) {
                        *completion_error = "Luisa asynchronous buffer layout is missing";
                    }
                    return false;
                }
                const auto count = found->bytes->size() / found->stride;
                for (size_t i = 0u; i < count; ++i) {
                    if (!repack_value(*source_module, layout->feir_type_id, layout->device_type,
                                      found->bytes->data() + i * found->stride,
                                      packed.data() + i * layout->device_type->size(), false)) {
                        if (completion_error != nullptr) {
                            *completion_error = "Luisa failed to restore an asynchronous buffer element layout";
                        }
                        return false;
                    }
                }
            }
            return true;
        };
        auto completion_lifetime = std::make_shared<decltype(completion)>(std::move(completion));
        return state.signal_fence(
            dispatch.stream_key, dispatch.fence_key, dispatch.retain_fence,
            [completion_lifetime](std::string* completion_error) { return (*completion_lifetime)(completion_error); },
            error);
    }
    stream << synchronize();
    if (!state.finalize_stream_through(dispatch.stream_key, std::numeric_limits<uint64_t>::max(), true, error)) {
        return false;
    }

    staged_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceBuffer) continue;
        auto* found = staged_bindings[staged_index];
        auto& packed = staged_bytes[staged_index++];
        if ((resource.access != kAccessWrite && resource.access != kAccessReadWrite) || !found->download) continue;
        const auto layout = std::find_if(buffer_layouts.begin(), buffer_layouts.end(),
                                         [&](const auto& candidate) { return candidate.binding == resource.binding; });
        const auto count = found->bytes->size() / found->stride;
        for (size_t i = 0; i < count; ++i) {
            if (!repack_value(module, layout->feir_type_id, layout->device_type,
                              found->bytes->data() + i * found->stride,
                              packed.data() + i * layout->device_type->size(), false)) {
                if (error != nullptr) *error = "Luisa failed to restore a Feather buffer element layout";
                return false;
            }
        }
    }
    for (size_t i = 0; i < gradient_layouts.size(); ++i) {
        auto& packed = staged_gradients[i];
        auto found = std::find_if(gradients.begin(), gradients.end(), [&](const auto& gradient) {
            return gradient.source_binding == gradient_layouts[i].source_binding;
        });
        const auto value_count = static_cast<size_t>(dispatch.logical_x) * found->element_count;
        const auto packed_stride = static_cast<size_t>(found->component_count) * sizeof(float);
        found->bytes->assign(value_count * packed_stride, 0u);
        for (size_t value = 0; value < value_count; ++value) {
            std::memcpy(found->bytes->data() + value * packed_stride,
                        packed.data() + value * gradient_layouts[i].device_type->size(), packed_stride);
        }
    }
    return true;
}

bool ReduceAdGradient(uint64_t context_key, std::string_view runtime_directory,
                      std::string_view backend_name, uint32_t device_index,
                      std::span<const unsigned char> gradient, uint32_t element_count,
                      uint32_t component_count, uint64_t destination_key,
                      std::vector<unsigned char>* destination, uint64_t destination_offset,
                      uint64_t destination_size, bool upload_destination,
                      std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    if (error != nullptr) error->clear();
    auto& state = runtime_registry().prepare(
        context_key, runtime_directory, backend_name, device_index);
    return state.reduce_ad_gradient(gradient, element_count, component_count,
                                    destination_key, destination, destination_offset,
                                    destination_size, upload_destination, error);
}

bool PrepareGraphicsFragment(const TypedIR::Module& module,
                             const TypedIR::LoweringInputs& lowering,
                             std::span<HostBufferBinding> host_buffers,
                             std::span<HostTextureBinding> host_textures,
                             const DispatchInputs& dispatch,
                             uint64_t callable_cache_key,
                             std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    const auto trace_enabled = std::getenv("FEATHER_GRAPHICS_TRACE") != nullptr;
    const auto trace = [&](const char* step) {
        if (trace_enabled) std::fprintf(stderr, "[feather fused fragment] %s\n", step);
    };
    if (error != nullptr) error->clear();
    if (callable_cache_key == 0u) {
        if (error != nullptr) *error = "fused fragment callable requires a cache key";
        return false;
    }
    auto& state = runtime_registry().prepare(
        dispatch.context_key, dispatch.runtime_directory, dispatch.backend_name, dispatch.device_index);
    const auto inputs_clean = std::none_of(
                                  host_buffers.begin(), host_buffers.end(),
                                  [](const auto& binding) { return binding.upload; }) &&
                              std::none_of(
                                  host_textures.begin(), host_textures.end(),
                                  [](const auto& binding) { return binding.upload; });
    if (state.find_fragment(callable_cache_key) != nullptr && inputs_clean) return true;
    ensure_luisa_spirv_optimization_preset();
    xir::Module xir_module;
    std::vector<BufferLayout> buffer_layouts;
    const auto fragment = LowerGraphicsFragmentToXir(
        module, lowering, xir_module, &buffer_layouts, error);
    trace("lowered XIR");
    if (fragment.function == nullptr || fragment.varying_type == nullptr || fragment.return_type == nullptr) return false;
    auto verification = xir_verify_module(&xir_module);
    trace("verified XIR");
    if (!verification.succeeded()) {
        if (error != nullptr) {
            *error = "generated fused fragment XIR failed verification: ";
            error->append(verification.errors.front().message.data(),
                          verification.errors.front().message.size());
        }
        return false;
    }

    trace("runtime ready");
    auto& stream = state.stream();
    std::vector<luisa::compute::Function::Binding> bound_arguments;
    bound_arguments.reserve(lowering.resources.size());
    std::vector<std::vector<unsigned char>> staged_buffers;
    staged_buffers.reserve(host_buffers.size());

    for (const auto& resource : lowering.resources) {
        if (resource.kind == kResourceTexture2D || resource.kind == kResourceTexture3D) {
            auto found = std::find_if(host_textures.begin(), host_textures.end(),
                                      [&](const auto& binding) { return binding.binding == resource.binding; });
            const auto storage = found == host_textures.end()
                                     ? std::nullopt
                                     : pixel_storage(found->pixel_format);
            if (found == host_textures.end() || found->resident_key == 0u || found->bytes == nullptr ||
                found->bytes->empty() || !storage) {
                if (error != nullptr) *error = "fused fragment texture binding is not resident";
                return false;
            }
            bool resident_created = false;
            auto* runtime = state.resident_texture(found->resident_key, resource.kind, *storage,
                                                   make_uint3(found->width, found->height, found->depth),
                                                   found->mip_levels, &resident_created);
            if (runtime == nullptr) {
                if (error != nullptr) *error = "fused fragment resident texture layout changed";
                return false;
            }
            const auto upload = found->upload || resident_created;
            std::visit(
                [&](auto& texture) {
                    if (upload)
                        stream << texture.copy_from(found->bytes->data());
                    bound_arguments.emplace_back(luisa::compute::Function::TextureBinding{texture.handle(), 0u});
                },
                *runtime);
            if (upload && found->generate_mipmaps && !state.generate_mipmaps(*runtime, found->mip_levels, error)) {
                return false;
            }
            continue;
        }
        if (resource.kind != kResourceBuffer) continue;
        auto found = std::find_if(host_buffers.begin(), host_buffers.end(),
                                  [&](const auto& binding) { return binding.binding == resource.binding; });
        const auto layout = std::find_if(buffer_layouts.begin(), buffer_layouts.end(),
                                         [&](const auto& candidate) { return candidate.binding == resource.binding; });
        if (found == host_buffers.end() || found->resident_key == 0u || found->bytes == nullptr ||
            found->stride == 0u || found->bytes->empty() || found->bytes->size() % found->stride != 0u ||
            layout == buffer_layouts.end() || layout->device_type == nullptr) {
            if (error != nullptr) *error = "fused fragment buffer binding is not resident";
            return false;
        }
        const auto count = found->bytes->size() / found->stride;
        const auto device_size = count * layout->device_type->size();
        bool resident_created = false;
        auto* runtime = state.resident_buffer(found->resident_key, device_size, &resident_created);
        if (runtime == nullptr) {
            if (error != nullptr) *error = "fused fragment resident buffer layout changed";
            return false;
        }
        if (found->upload || resident_created) {
            auto& packed = staged_buffers.emplace_back(device_size, 0u);
            for (size_t i = 0u; i < count; ++i) {
                if (!repack_value(module, layout->feir_type_id, layout->device_type,
                                  found->bytes->data() + i * found->stride,
                                  packed.data() + i * layout->device_type->size(), true)) {
                    if (error != nullptr) *error = "fused fragment buffer repacking failed";
                    return false;
                }
            }
            stream << runtime->copy_from(packed.data());
        }
        bound_arguments.emplace_back(
            luisa::compute::Function::BufferBinding{runtime->handle(), 0u, runtime->size_bytes()});
    }

    trace("resources bound");

    if (state.find_fragment(callable_cache_key) != nullptr) return true;
    auto ast = xir_to_ast_translate(
        *fragment.function,
        XIR2ASTConfig{.strict = true,
                      .bound_arguments = luisa::span<const luisa::compute::Function::Binding>{
                          bound_arguments.data(), bound_arguments.size()}});
    trace("translated AST");
    if (ast == nullptr) {
        if (error != nullptr) *error = "Luisa failed to translate the fused fragment callable";
        return false;
    }
    state.insert_fragment(callable_cache_key,
                          RuntimeState::CachedFragment{std::move(ast), fragment.varying_type,
                                                       fragment.return_type});
    return true;
}

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
                            std::string* error) {
    std::scoped_lock lock{runtime_mutex()};
    // Opt-in compute-only triangle assembly and raster stage between generated vertex and
    // fragment FEIR dispatches.
    if (error != nullptr) error->clear();
    if (vertices.bytes == nullptr || target.bytes == nullptr || fragment_varyings == nullptr ||
        fragment_coverage == nullptr || vertices.stride < sizeof(float) * 4u || raster.vertex_count < 3u ||
        raster.vertex_count % 3u != 0u || vertices.resident_key == 0u || vertex_indices.empty() ||
        raster.vertices_per_instance == 0u || raster.vertex_domain == 0u ||
        raster.vertex_count % raster.vertices_per_instance != 0u ||
        vertex_indices.size() != raster.vertices_per_instance ||
        vertices.bytes->size() < vertices.stride * raster.vertex_domain *
                                     (raster.vertex_count / raster.vertices_per_instance) ||
        target.width == 0u || target.height == 0u ||
        target.depth != 1u || (raster.sample_count != 1u && raster.sample_count != 4u)) {
        if (error != nullptr) *error = "vertical raster bindings are incomplete";
        return false;
    }
    if (target.pixel_format != 3u && target.pixel_format != 4u && target.pixel_format != 10u) {
        if (error != nullptr) *error = "vertical raster supports only Rgba8, Bgra8, and Rgba32Float targets";
        return false;
    }
    const auto pixel_count = static_cast<size_t>(target.width) * target.height;
    const auto sample_element_count = pixel_count * raster.sample_count;
    if (target.bytes->size() < pixel_count * (target.pixel_format == 10u ? sizeof(float4) : sizeof(uint32_t))) {
        if (error != nullptr) *error = "vertical raster color storage is too small";
        return false;
    }
    if (depth != nullptr &&
        (depth->bytes == nullptr || (depth->pixel_format != 100u && depth->pixel_format != 101u) ||
         depth->width != target.width ||
         depth->height != target.height || depth->bytes->size() < pixel_count * sizeof(float))) {
        if (error != nullptr) *error = "vertical raster depth storage must be matching Depth32Float or Depth24Stencil8";
        return false;
    }

    auto &runtime = runtime_registry().prepare(
        dispatch.context_key, dispatch.runtime_directory, dispatch.backend_name, dispatch.device_index);
    auto &device = runtime.device();
    auto &stream = runtime.stream();
    std::vector<RuntimeState::CachedFragment*> fused_fragments;
    if (!fragment_callable_keys.empty()) {
        if (fragment_callable_keys.size() != 1u || raster.sample_count != 4u ||
            fragment_target_keys.size() != 4u) {
            if (error != nullptr) *error = "fused fragment bindings require one callable and four MSAA targets";
            return false;
        }
        fused_fragments.reserve(fragment_callable_keys.size());
        for (auto key : fragment_callable_keys) {
            auto* fragment = runtime.find_fragment(key);
            if (fragment == nullptr || fragment->callable == nullptr || fragment->varying_type == nullptr ||
                fragment->return_type != Type::vector(Type::of<float>(), 4u) ||
                fragment->varying_type->size() != vertices.stride) {
                if (error != nullptr) *error = "fused fragment callable is missing or has a different varying layout";
                return false;
            }
            fused_fragments.push_back(fragment);
        }
    }
    std::array<Image<float>*, 4u> fused_targets{};
    Image<float>* fused_resolved_target = nullptr;
    if (!fragment_target_keys.empty()) {
        const auto storage = pixel_storage(target.pixel_format);
        if (!storage) {
            if (error != nullptr) *error = "fused fragment target format is unsupported";
            return false;
        }
        for (size_t i = 0u; i < fused_targets.size(); ++i) {
            auto* resource = runtime.resident_texture(
                fragment_target_keys[i], kResourceTexture2D, *storage,
                make_uint3(target.width, target.height, 1u), target.mip_levels);
            fused_targets[i] = resource == nullptr ? nullptr : std::get_if<Image<float>>(resource);
            if (fused_targets[i] == nullptr) {
                if (error != nullptr) *error = "fused fragment MSAA target is unavailable";
                return false;
            }
        }
        auto* resolved_resource = runtime.resident_texture(
            target.resident_key, kResourceTexture2D, *storage,
            make_uint3(target.width, target.height, 1u), target.mip_levels);
        fused_resolved_target = resolved_resource == nullptr
                                    ? nullptr
                                    : std::get_if<Image<float>>(resolved_resource);
        if (fused_resolved_target == nullptr) {
            if (error != nullptr) *error = "fused fragment resolved target is unavailable";
            return false;
        }
    }

    if (vertices.stride % sizeof(float) != 0u) {
        if (error != nullptr) *error = "vertical raster varying stride must be float-aligned";
        return false;
    }
    auto* vertex_buffer = runtime.resident_buffer(vertices.resident_key, vertices.bytes->size());
    const auto index_resident_key = coverage_resident_key ^ 0x696e6469636573ull;
    auto* index_bytes = runtime.resident_buffer(index_resident_key, vertex_indices.size_bytes());
    if (vertex_buffer == nullptr || index_bytes == nullptr) {
        if (error != nullptr) *error = "vertical raster resident vertex/index buffers are unavailable";
        return false;
    }
    auto index_buffer = index_bytes->view().as<uint32_t>();
    const auto use_fast_raster = raster.vertex_count >= 512u && depth != nullptr &&
                                 raster.polygon_mode == 0u && raster.depth_test != 0u &&
                                 raster.depth_write != 0u && raster.depth_compare == 1u &&
                                 raster.stencil_test == 0u && raster.opaque_fragment != 0u;
    const auto use_fused_fragment = use_fast_raster && !fused_fragments.empty();
    const auto varying_byte_size = sample_element_count * vertices.stride;
    const auto coverage_byte_size = sample_element_count * sizeof(float);
    std::vector<unsigned char> host_varyings;
    std::vector<float> host_coverage;
    if (!use_fused_fragment) {
        host_varyings.resize(varying_byte_size, 0u);
        host_coverage.resize(sample_element_count, 0.0f);
    }
    auto* varying_buffer = runtime.resident_buffer(varying_resident_key, varying_byte_size);
    auto* coverage_bytes = runtime.resident_buffer(coverage_resident_key, coverage_byte_size);
    if (varying_buffer == nullptr || coverage_bytes == nullptr) {
        if (error != nullptr) *error = "vertical raster resident fragment buffers are unavailable";
        return false;
    }
    auto coverage_buffer = coverage_bytes->view().as<float>();
    const auto depth_element_count = depth == nullptr ? 1u : sample_element_count;
    const auto needs_host_depth = !use_fused_fragment || raster.clear_depth == 0u;
    std::vector<float> host_depth;
    std::vector<uint32_t> host_stencil;
    if (needs_host_depth) {
        host_depth.resize(depth_element_count, 1.0f);
        host_stencil.resize(depth_element_count, 0u);
    }
    if (depth != nullptr && needs_host_depth) {
        if (depth->pixel_format == 101u) {
            const auto* source = reinterpret_cast<const float*>(depth->bytes->data());
            for (size_t pixel = 0u; pixel < pixel_count; ++pixel) {
                for (uint32_t sample = 0u; sample < raster.sample_count; ++sample) {
                    host_depth[pixel * raster.sample_count + sample] = source[pixel];
                }
            }
        } else {
            const auto* packed = reinterpret_cast<const uint32_t*>(depth->bytes->data());
            constexpr auto inverse_depth24 = 1.0f / 16777215.0f;
            for (size_t i = 0u; i < pixel_count; ++i) {
                for (uint32_t sample = 0u; sample < raster.sample_count; ++sample) {
                    const auto index = i * raster.sample_count + sample;
                    host_depth[index] = static_cast<float>(packed[i] & 0x00ffffffu) * inverse_depth24;
                    host_stencil[index] = packed[i] >> 24u;
                }
            }
        }
    }
    const auto depth_resident_key = coverage_resident_key ^ 0x6465707468ull;
    const auto stencil_resident_key = coverage_resident_key ^ 0x7374656e63696cull;
    auto* depth_bytes = runtime.resident_buffer(depth_resident_key, depth_element_count * sizeof(float));
    auto* stencil_bytes = runtime.resident_buffer(stencil_resident_key, depth_element_count * sizeof(uint32_t));
    if (depth_bytes == nullptr || stencil_bytes == nullptr) {
        if (error != nullptr) *error = "vertical raster resident depth buffers are unavailable";
        return false;
    }
    auto depth_buffer = depth_bytes->view().as<float>();
    auto stencil_buffer = stencil_bytes->view().as<uint32_t>();
    if (const auto* profile = std::getenv("FEATHER_RASTER_PROFILE_STAGES");
        profile != nullptr && profile[0] != '\0' && std::strcmp(profile, "0") != 0) {
        std::fprintf(stderr,
                     "[feather raster profile] route=%s vertices=%u samples=%u depth=%u/%u compare=%u stencil=%u opaque=%u\n",
                     use_fast_raster ? "triangle-driven" : "pixel-driven", raster.vertex_count,
                     raster.sample_count, raster.depth_test, raster.depth_write, raster.depth_compare,
                     raster.stencil_test, raster.opaque_fragment);
    }
    if (!use_fast_raster) stream << index_bytes->copy_from(vertex_indices.data());
    if (depth != nullptr && (!use_fast_raster || raster.clear_depth == 0u)) {
        stream << depth_buffer.copy_from(host_depth.data())
               << stencil_buffer.copy_from(host_stencil.data());
    }

    const auto varying_stride = vertices.stride;
    const auto vertex_count = raster.vertex_count;
    const auto has_depth = depth != nullptr;
    uint64_t raster_shader_key = 1469598103934665603ull;
    const auto mix_shader_key = [&](uint64_t value) {
        raster_shader_key ^= value;
        raster_shader_key *= 1099511628211ull;
    };
    mix_shader_key(varying_stride);
    mix_shader_key(vertex_count);
    mix_shader_key(raster.vertices_per_instance);
    mix_shader_key(raster.vertex_domain);
    mix_shader_key(has_depth ? 1u : 0u);
    mix_shader_key(raster.sample_count);
    for (auto key : fragment_callable_keys) mix_shader_key(key);
    if (use_fast_raster) {
        const auto* profile_value = std::getenv("FEATHER_RASTER_PROFILE_STAGES");
        const auto profile_stages = profile_value != nullptr && profile_value[0] != '\0' &&
                                    std::strcmp(profile_value, "0") != 0;
        auto profile_start = std::chrono::steady_clock::now();
        const auto profile_checkpoint = [&](const char* stage) {
            if (!profile_stages) return;
            stream << synchronize();
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(now - profile_start).count();
            std::fprintf(stderr, "[feather raster profile] %s %.3f ms\n", stage, elapsed);
            profile_start = now;
        };
        const auto triangle_count = raster.vertex_count / 3u;
        if (triangle_count > std::numeric_limits<uint32_t>::max() / kRasterPrimitiveExpansion) {
            if (error != nullptr) *error = "tile raster primitive capacity overflow";
            return false;
        }
        const auto primitive_capacity = triangle_count * kRasterPrimitiveExpansion;
        const auto tile_width = (target.width + kRasterTileSize - 1u) / kRasterTileSize;
        const auto tile_height = (target.height + kRasterTileSize - 1u) / kRasterTileSize;
        const auto tile_count = tile_width * tile_height;
        if (triangle_count > std::numeric_limits<uint32_t>::max() /
                                 kInitialTileReferencesPerTriangle) {
            if (error != nullptr) *error = "tile raster reference capacity overflow";
            return false;
        }
        const auto reference_capacity = triangle_count * kInitialTileReferencesPerTriangle;
        const auto primitive_key = coverage_resident_key ^ 0x74696c657072696dull;
        const auto tile_count_key = coverage_resident_key ^ 0x74696c65636f756eull;
        const auto tile_offset_key = coverage_resident_key ^ 0x74696c656f666673ull;
        const auto tile_cursor_key = coverage_resident_key ^ 0x74696c6563757273ull;
        const auto tile_index_key = coverage_resident_key ^ 0x74696c65696e6478ull;
        const auto tile_mask_key = coverage_resident_key ^ 0x74696c656d61736bull;
        const auto tile_stats_key = coverage_resident_key ^ 0x74696c6573746174ull;
        auto* primitive_bytes = runtime.resident_buffer(
            primitive_key, static_cast<size_t>(primitive_capacity) * kRasterPrimitiveRecordSize);
        auto* tile_count_bytes = runtime.resident_buffer(
            tile_count_key, static_cast<size_t>(tile_count) * sizeof(uint32_t));
        auto* tile_offset_bytes = runtime.resident_buffer(
            tile_offset_key, static_cast<size_t>(tile_count) * sizeof(uint32_t));
        auto* tile_cursor_bytes = runtime.resident_buffer(
            tile_cursor_key, static_cast<size_t>(tile_count) * sizeof(uint32_t));
        auto* tile_index_bytes = runtime.resident_buffer(
            tile_index_key, static_cast<size_t>(reference_capacity) * sizeof(uint32_t));
        auto* tile_mask_bytes = runtime.resident_buffer(
            tile_mask_key, static_cast<size_t>(reference_capacity) * 2u * sizeof(uint32_t));
        auto* tile_stats_bytes = runtime.resident_buffer(tile_stats_key, 3u * sizeof(uint32_t));
        if (primitive_bytes == nullptr || tile_count_bytes == nullptr || tile_offset_bytes == nullptr ||
            tile_cursor_bytes == nullptr || tile_index_bytes == nullptr || tile_mask_bytes == nullptr ||
            tile_stats_bytes == nullptr) {
            if (error != nullptr) *error = "tile raster resident storage is unavailable";
            return false;
        }
        auto tile_counts = tile_count_bytes->view().as<uint32_t>();
        auto tile_offsets = tile_offset_bytes->view().as<uint32_t>();
        auto tile_cursors = tile_cursor_bytes->view().as<uint32_t>();
        auto tile_indices = tile_index_bytes->view().as<uint32_t>();
        auto tile_masks = tile_mask_bytes->view().as<uint32_t>();
        auto tile_stats = tile_stats_bytes->view().as<uint32_t>();
        const auto* cached_geometry = runtime.find_raster_geometry(coverage_resident_key);
        const auto reuse_geometry = !rebuild_geometry && geometry_cache_key != 0u &&
                                    cached_geometry != nullptr &&
                                    cached_geometry->geometry_key == geometry_cache_key;
        if (!reuse_geometry) {
            stream << index_bytes->copy_from(vertex_indices.data())
                   << (*runtime.tile_reset_shader())(
                          tile_counts, tile_offsets, tile_cursors, tile_stats, tile_count)
                          .dispatch(std::max(tile_count, 3u));
            profile_checkpoint("setup");

            const auto assembly_key = raster_shader_key ^ 0x74696c6561736d62ull;
            auto* assembly_shader = runtime.find_tile_assembly(assembly_key);
            if (assembly_shader == nullptr) {
                auto compiled = device.compile<1>(
                [varying_stride, vertices_per_instance = raster.vertices_per_instance,
                 vertex_domain = raster.vertex_domain, fused_record = use_fused_fragment](
                    ByteBufferVar vertex_buffer, BufferUInt index_buffer, ByteBufferVar primitives,
                    BufferUInt tile_counts, BufferUInt stats,
                    UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                    UInt scissor_x, UInt scissor_y, UInt scissor_width, UInt scissor_height,
                    UInt target_width, UInt target_height, UInt cull_mode, UInt front_face,
                    UInt primitive_capacity, UInt tile_width, UInt tile_height) noexcept {
                    const auto triangle = dispatch_id().x;
                    const auto raster_base = triangle * 3u;
                    const auto instance = raster_base / vertices_per_instance;
                    const auto local_base = raster_base % vertices_per_instance;
                    const auto source_a = instance * vertex_domain + index_buffer.read(local_base);
                    const auto source_b = instance * vertex_domain + index_buffer.read(local_base + 1u);
                    const auto source_c = instance * vertex_domain + index_buffer.read(local_base + 2u);
                    ArrayFloat4<kMaximumClippedVertices> positions;
                    ArrayFloat3<kMaximumClippedVertices> weights;
                    positions[0u] = vertex_buffer.read<float4>(source_a * varying_stride);
                    positions[1u] = vertex_buffer.read<float4>(source_b * varying_stride);
                    positions[2u] = vertex_buffer.read<float4>(source_c * varying_stride);
                    weights[0u] = make_float3(1.0f, 0.0f, 0.0f);
                    weights[1u] = make_float3(0.0f, 1.0f, 0.0f);
                    weights[2u] = make_float3(0.0f, 0.0f, 1.0f);
                    UInt clipped_count = 3u;
                    clip_homogeneous_triangle(positions, weights, clipped_count);
                    const auto viewport_origin = make_float2(
                        viewport_x.cast<float>(), viewport_y.cast<float>());
                    const auto viewport_size = make_float2(
                        viewport_width.cast<float>(), viewport_height.cast<float>());
                    const auto to_screen = [&](auto ndc) noexcept {
                        return viewport_origin + make_float2(
                            (ndc.x + 1.0f) * 0.5f * viewport_size.x,
                            (1.0f - ndc.y) * 0.5f * viewport_size.y);
                    };
                    const auto lower_bound = make_float2(
                        scissor_x.cast<float>(), scissor_y.cast<float>());
                    const auto upper_bound = make_float2(
                        luisa::compute::min(scissor_x + scissor_width, target_width).cast<float>() - 1.0f,
                        luisa::compute::min(scissor_y + scissor_height, target_height).cast<float>() - 1.0f);
                    UInt fan_index = 1u;
                    $while (fan_index + 1u < clipped_count) {
                        Float4 a = positions[0u];
                        Float4 b = positions[fan_index];
                        Float4 c = positions[fan_index + 1u];
                        Float2 pa = def(a.xy() / a.w);
                        Float2 pb = def(b.xy() / b.w);
                        Float2 pc = def(c.xy() / c.w);
                        Float area = def((pb.x - pa.x) * (pc.y - pa.y) -
                                         (pb.y - pa.y) * (pc.x - pa.x));
                        Bool front = area < 0.0f;
                        $if (front_face != 0u) { front = !front; };
                        Bool culled = def(false);
                        $if (cull_mode == 1u) { culled = front; }
                        $elif (cull_mode == 2u) { culled = !front; }
                        $elif (cull_mode == 3u) { culled = true; };
                    Float2 sa = def(to_screen(pa));
                    Float2 sb = def(to_screen(pb));
                    Float2 sc = def(to_screen(pc));
                    Float raster_area = def(area);
                    if (fused_record) {
                        constexpr auto subpixel_scale = 256.0f;
                        sa = floor(sa * subpixel_scale + 0.5f) / subpixel_scale;
                        sb = floor(sb * subpixel_scale + 0.5f) / subpixel_scale;
                        sc = floor(sc * subpixel_scale + 0.5f) / subpixel_scale;
                        raster_area = (sb.x - sa.x) * (sc.y - sa.y) -
                                      (sb.y - sa.y) * (sc.x - sa.x);
                        front = raster_area > 0.0f;
                        $if (front_face != 0u) { front = !front; };
                        culled = false;
                        $if (cull_mode == 1u) { culled = front; }
                        $elif (cull_mode == 2u) { culled = !front; }
                        $elif (cull_mode == 3u) { culled = true; };
                    }
                        const auto minimum = luisa::compute::max(
                            luisa::compute::floor(luisa::compute::min(sa, luisa::compute::min(sb, sc))),
                            lower_bound);
                        const auto maximum = luisa::compute::min(
                            luisa::compute::ceil(luisa::compute::max(sa, luisa::compute::max(sb, sc))),
                            upper_bound);
                        $if (!culled & abs(raster_area) > 1e-7f & all(minimum <= maximum)) {
                            const auto tile_minimum = luisa::compute::min(
                                minimum.cast<uint2>() / kRasterTileSize,
                                make_uint2(tile_width - 1u, tile_height - 1u));
                            const auto tile_maximum = luisa::compute::min(
                                maximum.cast<uint2>() / kRasterTileSize,
                                make_uint2(tile_width - 1u, tile_height - 1u));
                            const auto primitive = stats.atomic(0u).fetch_add(1u);
                            $if (primitive < primitive_capacity) {
                                const auto base = primitive * kRasterPrimitiveRecordSize;
                                if (fused_record) {
                                    const auto positive = raster_area > 0.0f;
                                    const auto orientation = ite(positive, 1.0f, -1.0f);
                                    const auto inverse_raster_area = 1.0f / raster_area;
                                    const auto edge_coefficients = [&](Float2 edge_a, Float2 edge_b,
                                                                       Bool top_left) noexcept {
                                        const auto coefficient_x = (edge_a.y - edge_b.y) * orientation;
                                        const auto coefficient_y = (edge_b.x - edge_a.x) * orientation;
                                        return make_float4(
                                            coefficient_x, coefficient_y,
                                            ite(top_left, 1.0f, 0.0f), inverse_raster_area);
                                    };
                                    const auto top_left_bc = ite(
                                        positive,
                                        (sc.y < sb.y) | ((sc.y == sb.y) & (sc.x > sb.x)),
                                        (sb.y < sc.y) | ((sb.y == sc.y) & (sb.x > sc.x)));
                                    const auto top_left_ca = ite(
                                        positive,
                                        (sa.y < sc.y) | ((sa.y == sc.y) & (sa.x > sc.x)),
                                        (sc.y < sa.y) | ((sc.y == sa.y) & (sc.x > sa.x)));
                                    const auto top_left_ab = ite(
                                        positive,
                                        (sb.y < sa.y) | ((sb.y == sa.y) & (sb.x > sa.x)),
                                        (sa.y < sb.y) | ((sa.y == sb.y) & (sa.x > sb.x)));
                                    primitives.write(base, make_float4(sa, sb));
                                    primitives.write(base + 16u, make_float4(
                                        sc, 1.0f / raster_area, 1.0f / a.w));
                                    primitives.write(base + 32u, make_float4(
                                        1.0f / b.w, 1.0f / c.w, a.z / a.w, b.z / b.w));
                                    primitives.write(base + 48u, make_float4(
                                        c.z / c.w, weights[0u].x, weights[0u].y, weights[0u].z));
                                    primitives.write(base + 64u, make_float4(weights[fan_index], 0.0f));
                                    primitives.write(base + 80u, make_float4(weights[fan_index + 1u], 0.0f));
                                    primitives.write(base + kRasterPrimitiveEdge0Offset,
                                                     edge_coefficients(sb, sc, top_left_bc));
                                    primitives.write(base + kRasterPrimitiveEdge1Offset,
                                                     edge_coefficients(sc, sa, top_left_ca));
                                    primitives.write(base + kRasterPrimitiveEdge2Offset,
                                                     edge_coefficients(sa, sb, top_left_ab));
                                } else {
                                    primitives.write(base, a);
                                    primitives.write(base + 16u, b);
                                    primitives.write(base + 32u, c);
                                    primitives.write(base + 48u, make_float4(weights[0u], 0.0f));
                                    primitives.write(base + 64u, make_float4(weights[fan_index], 0.0f));
                                    primitives.write(base + 80u, make_float4(weights[fan_index + 1u], 0.0f));
                                }
                                primitives.write(base + 96u, make_uint4(
                                    source_a, source_b, source_c,
                                    triangle * kClippedPrimitiveStride + fan_index - 1u));
                                primitives.write(base + 112u, make_uint4(
                                    tile_minimum.x, tile_minimum.y, tile_maximum.x, tile_maximum.y));
                                primitives.write(base + 128u, make_float4(minimum, maximum));
                                UInt tile_y = tile_minimum.y;
                                $while (tile_y <= tile_maximum.y) {
                                    UInt tile_x = tile_minimum.x;
                                    $while (tile_x <= tile_maximum.x) {
                                        tile_counts.atomic(tile_y * tile_width + tile_x).fetch_add(1u);
                                        tile_x += 1u;
                                    };
                                    tile_y += 1u;
                                };
                            }
                            $else { stats.write(2u, 1u); };
                        };
                        fan_index += 1u;
                    };
                    });
                assembly_shader = runtime.insert_tile_assembly(assembly_key, std::move(compiled));
            }
            profile_checkpoint("assembly compile");
            stream << (*assembly_shader)(
                          *vertex_buffer, index_buffer, *primitive_bytes, tile_counts, tile_stats,
                          raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height,
                          raster.scissor_x, raster.scissor_y, raster.scissor_width, raster.scissor_height,
                          target.width, target.height, raster.cull_mode, raster.front_face,
                          primitive_capacity, tile_width, tile_height)
                          .dispatch(triangle_count);
            profile_checkpoint("clip and tile count");
            auto& tile_fill_dispatch = runtime.tile_fill_dispatch_buffer();
            stream << (*runtime.tile_prefix_shader())(
                          tile_counts, tile_offsets, tile_cursors, tile_stats, tile_fill_dispatch,
                          tile_count, reference_capacity)
                          .dispatch(1u)
                   << (*runtime.tile_fill_shader())(
                          *primitive_bytes, tile_offsets, tile_cursors, tile_indices, tile_masks, tile_stats,
                          primitive_capacity, tile_width, reference_capacity,
                          use_fused_fragment ? 1u : 0u)
                          .dispatch(tile_fill_dispatch, 0u, primitive_capacity);
            std::array<uint32_t, 3u> host_tile_stats{};
            if (profile_stages) stream << tile_stats_bytes->copy_to(host_tile_stats.data());
            profile_checkpoint("prefix and tile fill");
            if (profile_stages) {
                std::fprintf(stderr,
                             "[feather raster profile] primitives=%u references=%u capacity=%u overflow=%u\n",
                             host_tile_stats[0], host_tile_stats[1], reference_capacity, host_tile_stats[2]);
            }
            runtime.insert_raster_geometry(
                coverage_resident_key,
                RuntimeState::CachedRasterGeometry{
                    geometry_cache_key, host_tile_stats[0], host_tile_stats[1]});
        } else {
            profile_checkpoint("geometry cache hit");
            if (profile_stages) {
                std::fprintf(stderr,
                             "[feather raster profile] reused geometry primitives=%u references=%u capacity=%u\n",
                             cached_geometry->primitive_count, cached_geometry->reference_count,
                             reference_capacity);
            }
        }

        if (use_fused_fragment) {
            std::vector<luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder>> fragment_functions;
            fragment_functions.reserve(fused_fragments.size());
            for (auto* fragment : fused_fragments) fragment_functions.push_back(fragment->callable);
            const auto* varying_type = fused_fragments.front()->varying_type;
            const auto fused_raster_key = raster_shader_key ^ 0x667573656474696cull;
            auto* fused_raster_shader = runtime.find_fused_tile_raster(fused_raster_key);
            if (fused_raster_shader == nullptr) {
                auto compiled = device.compile<2>(
                    [varying_stride, sample_count = raster.sample_count,
                     fragment_functions = std::move(fragment_functions), varying_type](
                        ByteBufferVar vertex_buffer, ByteBufferVar primitives,
                        BufferUInt tile_counts, BufferUInt tile_offsets, BufferUInt tile_indices,
                        BufferUInt tile_masks, BufferFloat depth_values,
                        ImageFloat resolved_output,
                        UInt tile_width, UInt reference_capacity, UInt target_width, UInt target_height,
                        UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                        UInt clear_depth, Float clear_depth_value, UInt load_color,
                        Float4 clear_color) noexcept {
                        set_block_size(kRasterTileSize, kRasterTileSize, 1u);
                        Shared<float4> shared_primitive_data{kSharedPrimitiveBatchSize * 7u};
                        Shared<uint4> shared_primitive_sources{kSharedPrimitiveBatchSize};
                        Shared<uint> shared_primitive_indices{kSharedPrimitiveBatchSize};
                        Shared<uint2> shared_primitive_masks{kSharedPrimitiveBatchSize};
                        const auto local_id = thread_id().xy();
                        const auto lane = local_id.y * kRasterTileSize + local_id.x;
                        const auto id = block_id().xy() * kRasterTileSize + local_id;
                        const auto valid_pixel = id.x < target_width & id.y < target_height;
                        const auto safe_id = luisa::compute::min(
                            id, make_uint2(target_width - 1u, target_height - 1u));
                        const auto tile = block_id().y * tile_width + block_id().x;
                        const auto begin = tile_offsets.read(tile);
                        const auto end = luisa::compute::min(
                            begin + tile_counts.read(tile), reference_capacity);
                        ArrayFloat<4u> best_depth;
                        ArrayUInt<4u> best_order;
                        ArrayUInt<4u> best_primitive;
                        ArrayUInt3<4u> best_sources;
                        ArrayFloat4<4u> sample_colors;
                        const auto output_pixel = make_uint2(
                            safe_id.x, target_height - 1u - safe_id.y);
                        const auto initial_color = ite(
                            load_color != 0u, resolved_output.read(output_pixel), clear_color);
                        for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                            const auto pixel_index = (safe_id.y * target_width + safe_id.x) * sample_count + sample;
                            $if (clear_depth != 0u) {
                                best_depth[sample] = clear_depth_value;
                            } $else {
                                best_depth[sample] = depth_values.read(pixel_index);
                            };
                            best_order[sample] = ~0u;
                            best_primitive[sample] = ~0u;
                            best_sources[sample] = make_uint3(0u);
                            sample_colors[sample] = initial_color;
                        }
                        const auto viewport_origin = make_float2(
                            viewport_x.cast<float>(), viewport_y.cast<float>());
                        const auto viewport_size = make_float2(
                            viewport_width.cast<float>(), viewport_height.cast<float>());
                        UInt batch_begin = begin;
                        $while (batch_begin < end) {
                            const auto batch_count = luisa::compute::min(
                                static_cast<uint32_t>(kSharedPrimitiveBatchSize), end - batch_begin);
                            $if (lane < batch_count) {
                                const auto primitive = tile_indices.read(batch_begin + lane);
                                const auto base = primitive * kRasterPrimitiveRecordSize;
                                const auto shared_base = lane * 7u;
                                shared_primitive_data.write(shared_base, primitives.read<float4>(base));
                                shared_primitive_data.write(shared_base + 1u, primitives.read<float4>(base + 16u));
                                shared_primitive_data.write(shared_base + 2u, primitives.read<float4>(base + 32u));
                                shared_primitive_data.write(shared_base + 3u, primitives.read<float4>(base + 48u));
                                shared_primitive_data.write(
                                    shared_base + 4u,
                                    primitives.read<float4>(base + kRasterPrimitiveEdge0Offset));
                                shared_primitive_data.write(
                                    shared_base + 5u,
                                    primitives.read<float4>(base + kRasterPrimitiveEdge1Offset));
                                shared_primitive_data.write(
                                    shared_base + 6u,
                                    primitives.read<float4>(base + kRasterPrimitiveEdge2Offset));
                                shared_primitive_sources.write(lane, primitives.read<uint4>(base + 96u));
                                shared_primitive_indices.write(lane, primitive);
                                const auto mask_base = (batch_begin + lane) * 2u;
                                shared_primitive_masks.write(
                                    lane, make_uint2(tile_masks.read(mask_base), tile_masks.read(mask_base + 1u)));
                            };
                            sync_block();
                            UInt local_primitive = 0u;
                            $while (local_primitive < batch_count) {
                                const auto shared_base = local_primitive * 7u;
                                const auto screen_ab = shared_primitive_data.read(shared_base);
                                const auto screen_c_metadata = shared_primitive_data.read(shared_base + 1u);
                                const auto interpolation = shared_primitive_data.read(shared_base + 2u);
                                const auto source_metadata = shared_primitive_data.read(shared_base + 3u);
                                const auto edge0 = shared_primitive_data.read(shared_base + 4u);
                                const auto edge1 = shared_primitive_data.read(shared_base + 5u);
                                const auto edge2 = shared_primitive_data.read(shared_base + 6u);
                                const auto sources = shared_primitive_sources.read(local_primitive);
                                const auto primitive = shared_primitive_indices.read(local_primitive);
                                const auto quadrant = local_id.y / kRasterMicroCellSize *
                                                          kRasterMicroCellsPerAxis +
                                                      local_id.x / kRasterMicroCellSize;
                                const auto mask = shared_primitive_masks.read(local_primitive);
                                const auto mask_word = ite(quadrant < 32u, mask.x, mask.y);
                                const auto active = (mask_word & (1u << (quadrant & 31u))) != 0u & valid_pixel;
                                $if (active) {
                                    for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                                        float2 sample_offset = make_float2(0.5f);
                                        if (sample_count == 4u) {
                                            constexpr std::array<float2, 4u> offsets{
                                                make_float2(0.375f, 0.875f), make_float2(0.875f, 0.625f),
                                                make_float2(0.125f, 0.375f), make_float2(0.625f, 0.125f)};
                                            sample_offset = offsets[sample];
                                        }
                                        const auto viewport_pixel = make_float2(id) + sample_offset;
                                        const auto edge_value0 = edge0.x * (viewport_pixel.x - screen_ab.z) +
                                                                 edge0.y * (viewport_pixel.y - screen_ab.w);
                                        const auto edge_value1 = edge1.x * (viewport_pixel.x - screen_c_metadata.x) +
                                                                 edge1.y * (viewport_pixel.y - screen_c_metadata.y);
                                        const auto edge_value2 = edge2.x * (viewport_pixel.x - screen_ab.x) +
                                                                 edge2.y * (viewport_pixel.y - screen_ab.y);
                                        const auto covered =
                                            ((edge_value0 > 0.0f) | ((edge_value0 == 0.0f) & (edge0.z != 0.0f))) &
                                            ((edge_value1 > 0.0f) | ((edge_value1 == 0.0f) & (edge1.z != 0.0f))) &
                                            ((edge_value2 > 0.0f) | ((edge_value2 == 0.0f) & (edge2.z != 0.0f)));
                                        $if (covered) {
                                                const auto inverse_area = abs(screen_c_metadata.z);
                                                const auto w0 = edge_value0 * inverse_area;
                                                const auto w1 = edge_value1 * inverse_area;
                                                const auto w2 = edge_value2 * inverse_area;
                                                const auto candidate_depth = w0 * interpolation.z +
                                                                             w1 * interpolation.w +
                                                                             w2 * source_metadata.x;
                                                const auto wins = candidate_depth < best_depth[sample] |
                                                                  ((candidate_depth == best_depth[sample]) &
                                                                   (sources.w < best_order[sample]));
                                                $if (wins) {
                                                    best_depth[sample] = candidate_depth;
                                                    best_order[sample] = sources.w;
                                                    best_primitive[sample] = primitive;
                                                    best_sources[sample] = sources.xyz();
                                                };
                                        };
                                    }
                                };
                                local_primitive += 1u;
                            };
                            sync_block();
                            batch_begin += batch_count;
                        };

                        for (uint32_t representative = 0u; representative < sample_count; ++representative) {
                            Bool already_shaded = def(false);
                            for (uint32_t previous = 0u; previous < representative; ++previous) {
                                already_shaded |= best_primitive[previous] == best_primitive[representative];
                            }
                            $if (best_order[representative] != ~0u & !already_shaded) {
                                const auto primitive_base =
                                    best_primitive[representative] * kRasterPrimitiveRecordSize;
                                const auto screen_ab = primitives.read<float4>(primitive_base);
                                const auto screen_c_metadata = primitives.read<float4>(primitive_base + 16u);
                                const auto interpolation = primitives.read<float4>(primitive_base + 32u);
                                const auto source_weight_a = primitives.read<float4>(primitive_base + 48u).yzw();
                                const auto source_weight_b = primitives.read<float4>(primitive_base + 64u).xyz();
                                const auto source_weight_c = primitives.read<float4>(primitive_base + 80u).xyz();
                                const auto pa = screen_ab.xy();
                                const auto pb = screen_ab.zw();
                                const auto pc = screen_c_metadata.xy();
                                const auto positive = screen_c_metadata.z > 0.0f;
                                auto source_weight_at = [&](Float2 screen_pixel) noexcept {
                                    const auto edge0_raw = (pb.x - screen_pixel.x) * (pc.y - screen_pixel.y) -
                                                           (pb.y - screen_pixel.y) * (pc.x - screen_pixel.x);
                                    const auto edge1_raw = (pc.x - screen_pixel.x) * (pa.y - screen_pixel.y) -
                                                           (pc.y - screen_pixel.y) * (pa.x - screen_pixel.x);
                                    const auto edge2_raw = (pa.x - screen_pixel.x) * (pb.y - screen_pixel.y) -
                                                           (pa.y - screen_pixel.y) * (pb.x - screen_pixel.x);
                                    const auto q0 = ite(positive, edge0_raw, -edge0_raw) *
                                                    screen_c_metadata.w;
                                    const auto q1 = ite(positive, edge1_raw, -edge1_raw) *
                                                    interpolation.x;
                                    const auto q2 = ite(positive, edge2_raw, -edge2_raw) *
                                                    interpolation.y;
                                    return (source_weight_a * q0 + source_weight_b * q1 +
                                            source_weight_c * q2) / (q0 + q1 + q2);
                                };
                                const auto pair_x = (id.x / 2u) * 2u;
                                const auto pair_y = (id.y / 2u) * 2u;
                                const auto pair_x1 = luisa::compute::min(pair_x + 1u, target_width - 1u);
                                const auto pair_y1 = luisa::compute::min(pair_y + 1u, target_height - 1u);
                                const auto current_weight = source_weight_at(make_float2(id) + 0.5f);
                                const auto x_is_first = (id.x & 1u) == 0u;
                                const auto y_is_first = (id.y & 1u) == 0u;
                                const auto x_neighbor = ite(x_is_first, pair_x1, pair_x);
                                const auto y_neighbor = ite(y_is_first, pair_y1, pair_y);
                                const auto x_neighbor_weight = source_weight_at(
                                    make_float2(x_neighbor.cast<float>() + 0.5f,
                                                id.y.cast<float>() + 0.5f));
                                const auto y_neighbor_weight = source_weight_at(
                                    make_float2(id.x.cast<float>() + 0.5f,
                                                y_neighbor.cast<float>() + 0.5f));
                                const auto x0_weight = ite(x_is_first, current_weight, x_neighbor_weight);
                                const auto x1_weight = ite(x_is_first, x_neighbor_weight, current_weight);
                                const auto y0_weight = ite(y_is_first, current_weight, y_neighbor_weight);
                                const auto y1_weight = ite(y_is_first, y_neighbor_weight, current_weight);
                                auto* builder = luisa::compute::detail::FunctionBuilder::current();
                                using VaryingExpressions = std::array<const Expression*, 5u>;
                                const VaryingExpressions source_weights{
                                    current_weight.expression(), x0_weight.expression(), x1_weight.expression(),
                                    y0_weight.expression(), y1_weight.expression()};
                                std::function<VaryingExpressions(const Type*, size_t)> build_varyings;
                                build_varyings = [&](const Type* type, size_t offset) -> VaryingExpressions {
                                    const auto source_a = best_sources[representative].x * varying_stride +
                                                          static_cast<uint32_t>(offset);
                                    const auto source_b = best_sources[representative].y * varying_stride +
                                                          static_cast<uint32_t>(offset);
                                    const auto source_c = best_sources[representative].z * varying_stride +
                                                          static_cast<uint32_t>(offset);
                                    const auto interpolate = [&](auto va, auto vb, auto vc) {
                                        VaryingExpressions values{};
                                        for (size_t i = 0u; i < values.size(); ++i) {
                                            const Expr<float3> weight{source_weights[i]};
                                            values[i] = (va * weight.x + vb * weight.y + vc * weight.z).expression();
                                        }
                                        return values;
                                    };
                                    if (type == Type::of<float>()) {
                                        return interpolate(vertex_buffer.read<float>(source_a),
                                                           vertex_buffer.read<float>(source_b),
                                                           vertex_buffer.read<float>(source_c));
                                    }
                                    if (type->is_vector() && type->element() == Type::of<float>()) {
                                        if (type->dimension() == 2u) {
                                            return interpolate(vertex_buffer.read<float2>(source_a),
                                                               vertex_buffer.read<float2>(source_b),
                                                               vertex_buffer.read<float2>(source_c));
                                        }
                                        if (type->dimension() == 3u) {
                                            return interpolate(vertex_buffer.read<float3>(source_a),
                                                               vertex_buffer.read<float3>(source_b),
                                                               vertex_buffer.read<float3>(source_c));
                                        }
                                        if (type->dimension() == 4u) {
                                            return interpolate(vertex_buffer.read<float4>(source_a),
                                                               vertex_buffer.read<float4>(source_b),
                                                               vertex_buffer.read<float4>(source_c));
                                        }
                                    }
                                    VaryingExpressions locals{};
                                    for (auto& local : locals) local = builder->local(type);
                                    if (type->is_vector() || type->is_array()) {
                                        const auto* element = type->element();
                                        const auto stride = align_up(element->size(), element->alignment());
                                        for (uint32_t i = 0u; i < type->dimension(); ++i) {
                                            auto* index = builder->literal(Type::of<uint32_t>(), i);
                                            const auto members = build_varyings(element, offset + i * stride);
                                            for (size_t value = 0u; value < locals.size(); ++value) {
                                                builder->assign(builder->access(element, locals[value], index),
                                                                members[value]);
                                            }
                                        }
                                        return locals;
                                    }
                                    if (type->is_structure()) {
                                        size_t member_offset = 0u;
                                        for (uint32_t i = 0u; i < type->members().size(); ++i) {
                                            const auto* member_type = type->members()[i];
                                            member_offset = align_up(member_offset, member_type->alignment());
                                            const auto members = build_varyings(member_type, offset + member_offset);
                                            for (size_t value = 0u; value < locals.size(); ++value) {
                                                builder->assign(builder->member(member_type, locals[value], i),
                                                                members[value]);
                                            }
                                            member_offset += member_type->size();
                                        }
                                        return locals;
                                    }
                                    LUISA_ERROR_WITH_LOCATION(
                                        "Unsupported fused fragment varying type {}.", type->description());
                                };
                                const auto arguments = build_varyings(varying_type, 0u);
                                auto* color_expression = builder->call(
                                    Type::vector(Type::of<float>(), 4u),
                                    luisa::compute::Function{fragment_functions.front().get()}, arguments);
                                const auto color = def<float4>(color_expression);
                                for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                                    $if (best_order[sample] != ~0u &
                                         best_primitive[sample] == best_primitive[representative]) {
                                        sample_colors[sample] = color;
                                        const auto pixel_index =
                                            (id.y * target_width + id.x) * sample_count + sample;
                                        depth_values.write(pixel_index, best_depth[sample]);
                                    };
                                }
                            };
                        }
                        $if (valid_pixel) {
                            const auto quantize_unorm8 = [](Float4 color) noexcept {
                                return round(clamp(color, 0.0f, 1.0f) * 255.0f) * (1.0f / 255.0f);
                            };
                            const auto resolved = (quantize_unorm8(sample_colors[0u]) +
                                                   quantize_unorm8(sample_colors[1u]) +
                                                   quantize_unorm8(sample_colors[2u]) +
                                                   quantize_unorm8(sample_colors[3u])) * 0.25f;
                            resolved_output.write(
                                make_uint2(id.x, target_height - 1u - id.y), resolved);
                        };
                    });
                fused_raster_shader = runtime.insert_fused_tile_raster(
                    fused_raster_key, std::move(compiled));
            }
            profile_checkpoint("fused tile raster compile");
            stream << (*fused_raster_shader)(
                          *vertex_buffer, *primitive_bytes, tile_counts, tile_offsets, tile_indices,
                          tile_masks, depth_buffer,
                          *fused_resolved_target,
                          tile_width, reference_capacity,
                          target.width, target.height, raster.viewport_x, raster.viewport_y,
                          raster.viewport_width, raster.viewport_height,
                          raster.clear_depth, raster.clear_depth_value,
                          raster.load_color,
                          make_float4(raster.clear_color_r, raster.clear_color_g,
                                      raster.clear_color_b, raster.clear_color_a))
                          .dispatch(target.width, target.height);
            if (dispatch.synchronize) stream << synchronize();
            profile_checkpoint("fused tile raster and fragment");
            fragment_varyings->clear();
            fragment_coverage->clear();
            return true;
        }

        const auto shared_raster_key = raster_shader_key ^ 0x7368617265647469ull;
        auto* shared_raster_shader = runtime.find_shared_tile_raster(shared_raster_key);
        if (shared_raster_shader == nullptr) {
            auto compiled = device.compile<2>(
                [varying_stride, sample_count = raster.sample_count](
                    ByteBufferVar vertex_buffer, ByteBufferVar primitives,
                    BufferUInt tile_counts, BufferUInt tile_offsets, BufferUInt tile_indices,
                    BufferUInt tile_masks,
                    BufferFloat depth_values, ByteBufferVar varying_buffer, BufferFloat coverage_buffer,
                    UInt tile_width, UInt reference_capacity, UInt target_width, UInt target_height,
                    UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                    UInt clear_depth, Float clear_depth_value) noexcept {
                    set_block_size(kRasterTileSize, kRasterTileSize, 1u);
                    Shared<float4> shared_primitive_data{kSharedPrimitiveBatchSize * 6u};
                    Shared<uint4> shared_primitive_sources{kSharedPrimitiveBatchSize};
                    Shared<uint2> shared_primitive_masks{kSharedPrimitiveBatchSize};
                    const auto id = dispatch_id().xy();
                    const auto lane = thread_id().y * kRasterTileSize + thread_id().x;
                    const auto tile = block_id().y * tile_width + block_id().x;
                    const auto begin = tile_offsets.read(tile);
                    const auto end = luisa::compute::min(
                        begin + tile_counts.read(tile), reference_capacity);
                    ArrayFloat<4u> best_depth;
                    ArrayUInt<4u> best_order;
                    ArrayUInt3<4u> best_sources;
                    ArrayFloat3<4u> best_source_weight;
                    for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                        const auto pixel_index = (id.y * target_width + id.x) * sample_count + sample;
                        best_depth[sample] = ite(
                            clear_depth != 0u, clear_depth_value, depth_values.read(pixel_index));
                        best_order[sample] = ~0u;
                        best_sources[sample] = make_uint3(0u);
                        best_source_weight[sample] = make_float3(0.0f);
                        coverage_buffer.write(pixel_index, 0.0f);
                    }
                    const auto viewport_origin = make_float2(
                        viewport_x.cast<float>(), viewport_y.cast<float>());
                    const auto viewport_size = make_float2(
                        viewport_width.cast<float>(), viewport_height.cast<float>());
                    UInt batch_begin = begin;
                    $while (batch_begin < end) {
                        const auto batch_count = luisa::compute::min(
                            static_cast<uint32_t>(kSharedPrimitiveBatchSize), end - batch_begin);
                        $if (lane < batch_count) {
                            const auto primitive = tile_indices.read(batch_begin + lane);
                            const auto base = primitive * kRasterPrimitiveRecordSize;
                            const auto shared_base = lane * 6u;
                            shared_primitive_data.write(shared_base, primitives.read<float4>(base));
                            shared_primitive_data.write(shared_base + 1u, primitives.read<float4>(base + 16u));
                            shared_primitive_data.write(shared_base + 2u, primitives.read<float4>(base + 32u));
                            shared_primitive_data.write(shared_base + 3u, primitives.read<float4>(base + 48u));
                            shared_primitive_data.write(shared_base + 4u, primitives.read<float4>(base + 64u));
                            shared_primitive_data.write(shared_base + 5u, primitives.read<float4>(base + 80u));
                            shared_primitive_sources.write(lane, primitives.read<uint4>(base + 96u));
                            const auto mask_base = (batch_begin + lane) * 2u;
                            shared_primitive_masks.write(
                                lane, make_uint2(tile_masks.read(mask_base), tile_masks.read(mask_base + 1u)));
                        };
                        sync_block();
                        UInt local_primitive = 0u;
                        $while (local_primitive < batch_count) {
                            const auto shared_base = local_primitive * 6u;
                            const auto a = shared_primitive_data.read(shared_base);
                            const auto b = shared_primitive_data.read(shared_base + 1u);
                            const auto c = shared_primitive_data.read(shared_base + 2u);
                            const auto source_weight_a = shared_primitive_data.read(shared_base + 3u).xyz();
                            const auto source_weight_b = shared_primitive_data.read(shared_base + 4u).xyz();
                            const auto source_weight_c = shared_primitive_data.read(shared_base + 5u).xyz();
                            const auto sources = shared_primitive_sources.read(local_primitive);
                            const auto quadrant = thread_id().y / kRasterMicroCellSize *
                                                      kRasterMicroCellsPerAxis +
                                                  thread_id().x / kRasterMicroCellSize;
                            const auto mask = shared_primitive_masks.read(local_primitive);
                            const auto mask_word = ite(quadrant < 32u, mask.x, mask.y);
                            const auto active = (mask_word & (1u << (quadrant & 31u))) != 0u;
                            $if (active) {
                            const auto pa = a.xy() / a.w;
                            const auto pb = b.xy() / b.w;
                            const auto pc = c.xy() / c.w;
                            const auto area = (pb.x - pa.x) * (pc.y - pa.y) -
                                              (pb.y - pa.y) * (pc.x - pa.x);
                            const auto primitive_minimum = luisa::compute::min(
                                pa, luisa::compute::min(pb, pc));
                            const auto primitive_maximum = luisa::compute::max(
                                pa, luisa::compute::max(pb, pc));
                            const auto positive = area > 0.0f;
                            const auto top_left_bc = ite(
                                positive,
                                (pc.y > pb.y) | ((pc.y == pb.y) & (pc.x < pb.x)),
                                (pb.y > pc.y) | ((pb.y == pc.y) & (pb.x < pc.x)));
                            const auto top_left_ca = ite(
                                positive,
                                (pa.y > pc.y) | ((pa.y == pc.y) & (pa.x < pc.x)),
                                (pc.y > pa.y) | ((pc.y == pa.y) & (pc.x < pa.x)));
                            const auto top_left_ab = ite(
                                positive,
                                (pb.y > pa.y) | ((pb.y == pa.y) & (pb.x < pa.x)),
                                (pa.y > pb.y) | ((pa.y == pb.y) & (pa.x < pb.x)));
                            for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                                float2 sample_offset = make_float2(0.5f);
                                if (sample_count == 4u) {
                                    constexpr std::array<float2, 4u> offsets{
                                        make_float2(0.375f, 0.125f), make_float2(0.875f, 0.375f),
                                        make_float2(0.125f, 0.625f), make_float2(0.625f, 0.875f)};
                                    sample_offset = offsets[sample];
                                }
                                const auto viewport_pixel = make_float2(id) + sample_offset - viewport_origin;
                                const auto p = make_float2(
                                    viewport_pixel.x / viewport_size.x * 2.0f - 1.0f,
                                    1.0f - viewport_pixel.y / viewport_size.y * 2.0f);
                                $if (all(p >= primitive_minimum) & all(p <= primitive_maximum)) {
                                    const auto edge0_raw = (pb.x - p.x) * (pc.y - p.y) -
                                                           (pb.y - p.y) * (pc.x - p.x);
                                    const auto edge1_raw = (pc.x - p.x) * (pa.y - p.y) -
                                                           (pc.y - p.y) * (pa.x - p.x);
                                    const auto edge2_raw = (pa.x - p.x) * (pb.y - p.y) -
                                                           (pa.y - p.y) * (pb.x - p.x);
                                    const auto edge0 = ite(positive, edge0_raw, -edge0_raw);
                                    const auto edge1 = ite(positive, edge1_raw, -edge1_raw);
                                    const auto edge2 = ite(positive, edge2_raw, -edge2_raw);
                                    const auto covered =
                                        ((edge0 > 0.0f) | ((edge0 == 0.0f) & top_left_bc)) &
                                        ((edge1 > 0.0f) | ((edge1 == 0.0f) & top_left_ca)) &
                                        ((edge2 > 0.0f) | ((edge2 == 0.0f) & top_left_ab));
                                    $if (covered) {
                                        const auto inverse_area = 1.0f / abs(area);
                                        const auto w0 = edge0 * inverse_area;
                                        const auto w1 = edge1 * inverse_area;
                                        const auto w2 = edge2 * inverse_area;
                                        const auto candidate_depth = w0 * (a.z / a.w) +
                                                                     w1 * (b.z / b.w) +
                                                                     w2 * (c.z / c.w);
                                        const auto wins = candidate_depth < best_depth[sample] |
                                                          ((candidate_depth == best_depth[sample]) &
                                                           (sources.w < best_order[sample]));
                                        $if (wins) {
                                            const auto q0 = w0 / a.w;
                                            const auto q1 = w1 / b.w;
                                            const auto q2 = w2 / c.w;
                                            const auto divisor = q0 + q1 + q2;
                                            best_depth[sample] = candidate_depth;
                                            best_order[sample] = sources.w;
                                            best_sources[sample] = sources.xyz();
                                            best_source_weight[sample] =
                                                (source_weight_a * q0 + source_weight_b * q1 +
                                                 source_weight_c * q2) /
                                                divisor;
                                        };
                                    };
                                };
                            }
                            };
                            local_primitive += 1u;
                        };
                        sync_block();
                        batch_begin += batch_count;
                    };
                    for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                        $if (best_order[sample] != ~0u) {
                            const auto pixel_index = (id.y * target_width + id.x) * sample_count + sample;
                            const auto output_base = pixel_index * varying_stride;
                            uint32_t offset = 0u;
                            for (; offset + sizeof(float4) <= varying_stride; offset += sizeof(float4)) {
                                const auto va = vertex_buffer.read<float4>(
                                    best_sources[sample].x * varying_stride + offset);
                                const auto vb = vertex_buffer.read<float4>(
                                    best_sources[sample].y * varying_stride + offset);
                                const auto vc = vertex_buffer.read<float4>(
                                    best_sources[sample].z * varying_stride + offset);
                                varying_buffer.write(
                                    output_base + offset,
                                    va * best_source_weight[sample].x +
                                    vb * best_source_weight[sample].y +
                                    vc * best_source_weight[sample].z);
                            }
                            for (; offset < varying_stride; offset += sizeof(float)) {
                                const auto va = vertex_buffer.read<float>(
                                    best_sources[sample].x * varying_stride + offset);
                                const auto vb = vertex_buffer.read<float>(
                                    best_sources[sample].y * varying_stride + offset);
                                const auto vc = vertex_buffer.read<float>(
                                    best_sources[sample].z * varying_stride + offset);
                                varying_buffer.write(
                                    output_base + offset,
                                    va * best_source_weight[sample].x +
                                    vb * best_source_weight[sample].y +
                                    vc * best_source_weight[sample].z);
                            }
                            depth_values.write(pixel_index, best_depth[sample]);
                            coverage_buffer.write(pixel_index, 1.0f);
                        };
                    }
                });
            shared_raster_shader = runtime.insert_shared_tile_raster(shared_raster_key, std::move(compiled));
        }
        profile_checkpoint("shared tile raster compile");
        stream << (*shared_raster_shader)(
                      *vertex_buffer, *primitive_bytes, tile_counts, tile_offsets, tile_indices,
                      tile_masks, depth_buffer, *varying_buffer, coverage_buffer,
                      tile_width, reference_capacity, target.width, target.height,
                      raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height,
                      raster.clear_depth, raster.clear_depth_value)
                      .dispatch(target.width, target.height);
        if (profile_stages) stream << coverage_bytes->copy_to(host_coverage.data());
        profile_checkpoint("shared tile raster and varying resolve");
        if (profile_stages) {
            const auto covered = static_cast<size_t>(std::count_if(
                host_coverage.begin(), host_coverage.end(), [](float value) { return value != 0.0f; }));
            std::fprintf(stderr, "[feather raster profile] covered_samples=%zu/%zu\n",
                         covered, host_coverage.size());
        }
        *fragment_varyings = std::move(host_varyings);
        fragment_coverage->resize(sample_element_count * sizeof(float));
        return true;

        const auto tile_raster_key = raster_shader_key ^ 0x74696c6572617374ull;
        auto* tile_raster_shader = runtime.find_tile_raster(tile_raster_key);
        if (tile_raster_shader == nullptr) {
            auto compiled = device.compile<3>(
                [varying_stride, sample_count = raster.sample_count](
                    ByteBufferVar vertex_buffer, ByteBufferVar primitives,
                    BufferUInt tile_counts, BufferUInt tile_offsets, BufferUInt tile_indices,
                    BufferFloat depth_values, ByteBufferVar varying_buffer, BufferFloat coverage_buffer,
                    UInt tile_width, UInt reference_capacity, UInt target_width, UInt target_height,
                    UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height) noexcept {
                    const auto id = dispatch_id();
                    const auto pixel_index = (id.y * target_width + id.x) * sample_count + id.z;
                    Float best_depth = depth_values.read(pixel_index);
                    UInt best_order = ~0u;
                    UInt3 best_sources = make_uint3(0u);
                    Float3 best_source_weight = make_float3(0.0f);
                    Float2 sample_position = def(make_float2(0.5f));
                    if (sample_count == 4u) {
                        $if (id.z == 0u) { sample_position = make_float2(0.375f, 0.125f); }
                        $elif (id.z == 1u) { sample_position = make_float2(0.875f, 0.375f); }
                        $elif (id.z == 2u) { sample_position = make_float2(0.125f, 0.625f); }
                        $else { sample_position = make_float2(0.625f, 0.875f); };
                    }
                    const auto viewport_origin = make_float2(
                        viewport_x.cast<float>(), viewport_y.cast<float>());
                    const auto viewport_size = make_float2(
                        viewport_width.cast<float>(), viewport_height.cast<float>());
                    const auto viewport_pixel = make_float2(id.xy()) + sample_position - viewport_origin;
                    const auto p = make_float2(
                        viewport_pixel.x / viewport_size.x * 2.0f - 1.0f,
                        1.0f - viewport_pixel.y / viewport_size.y * 2.0f);
                    const auto tile = (id.y / kRasterTileSize) * tile_width + id.x / kRasterTileSize;
                    const auto begin = tile_offsets.read(tile);
                    const auto end = luisa::compute::min(
                        begin + tile_counts.read(tile), reference_capacity);
                    UInt reference = begin;
                    $while (reference < end) {
                        const auto primitive = tile_indices.read(reference);
                        const auto base = primitive * kRasterPrimitiveRecordSize;
                        const auto a = primitives.read<float4>(base);
                        const auto b = primitives.read<float4>(base + 16u);
                        const auto c = primitives.read<float4>(base + 32u);
                        const auto pa = a.xy() / a.w;
                        const auto pb = b.xy() / b.w;
                        const auto pc = c.xy() / c.w;
                        const auto area = (pb.x - pa.x) * (pc.y - pa.y) -
                                          (pb.y - pa.y) * (pc.x - pa.x);
                        const auto positive = area > 0.0f;
                        const auto edge0_raw = (pb.x - p.x) * (pc.y - p.y) -
                                               (pb.y - p.y) * (pc.x - p.x);
                        const auto edge1_raw = (pc.x - p.x) * (pa.y - p.y) -
                                               (pc.y - p.y) * (pa.x - p.x);
                        const auto edge2_raw = (pa.x - p.x) * (pb.y - p.y) -
                                               (pa.y - p.y) * (pb.x - p.x);
                        const auto edge0 = ite(positive, edge0_raw, -edge0_raw);
                        const auto edge1 = ite(positive, edge1_raw, -edge1_raw);
                        const auto edge2 = ite(positive, edge2_raw, -edge2_raw);
                        const auto top_left_bc = ite(
                            positive,
                            (pc.y > pb.y) | ((pc.y == pb.y) & (pc.x < pb.x)),
                            (pb.y > pc.y) | ((pb.y == pc.y) & (pb.x < pc.x)));
                        const auto top_left_ca = ite(
                            positive,
                            (pa.y > pc.y) | ((pa.y == pc.y) & (pa.x < pc.x)),
                            (pc.y > pa.y) | ((pc.y == pa.y) & (pc.x < pa.x)));
                        const auto top_left_ab = ite(
                            positive,
                            (pb.y > pa.y) | ((pb.y == pa.y) & (pb.x < pa.x)),
                            (pa.y > pb.y) | ((pa.y == pb.y) & (pa.x < pb.x)));
                        const auto covered =
                            ((edge0 > 0.0f) | ((edge0 == 0.0f) & top_left_bc)) &
                            ((edge1 > 0.0f) | ((edge1 == 0.0f) & top_left_ca)) &
                            ((edge2 > 0.0f) | ((edge2 == 0.0f) & top_left_ab));
                        $if (covered) {
                            const auto inverse_area = 1.0f / abs(area);
                            const auto w0 = edge0 * inverse_area;
                            const auto w1 = edge1 * inverse_area;
                            const auto w2 = edge2 * inverse_area;
                            const auto candidate_depth = w0 * (a.z / a.w) +
                                                         w1 * (b.z / b.w) +
                                                         w2 * (c.z / c.w);
                            const auto sources = primitives.read<uint4>(base + 96u);
                            const auto wins = candidate_depth < best_depth |
                                              ((candidate_depth == best_depth) & (sources.w < best_order));
                            $if (wins) {
                                const auto q0 = w0 / a.w;
                                const auto q1 = w1 / b.w;
                                const auto q2 = w2 / c.w;
                                const auto divisor = q0 + q1 + q2;
                                const auto source_weight_a = primitives.read<float4>(base + 48u).xyz();
                                const auto source_weight_b = primitives.read<float4>(base + 64u).xyz();
                                const auto source_weight_c = primitives.read<float4>(base + 80u).xyz();
                                best_depth = candidate_depth;
                                best_order = sources.w;
                                best_sources = sources.xyz();
                                best_source_weight =
                                    (source_weight_a * q0 + source_weight_b * q1 + source_weight_c * q2) /
                                    divisor;
                            };
                        };
                        reference += 1u;
                    };
                    $if (best_order != ~0u) {
                        const auto output_base = pixel_index * varying_stride;
                        uint32_t offset = 0u;
                        for (; offset + sizeof(float4) <= varying_stride; offset += sizeof(float4)) {
                            const auto va = vertex_buffer.read<float4>(best_sources.x * varying_stride + offset);
                            const auto vb = vertex_buffer.read<float4>(best_sources.y * varying_stride + offset);
                            const auto vc = vertex_buffer.read<float4>(best_sources.z * varying_stride + offset);
                            varying_buffer.write(
                                output_base + offset,
                                va * best_source_weight.x + vb * best_source_weight.y + vc * best_source_weight.z);
                        }
                        for (; offset < varying_stride; offset += sizeof(float)) {
                            const auto va = vertex_buffer.read<float>(best_sources.x * varying_stride + offset);
                            const auto vb = vertex_buffer.read<float>(best_sources.y * varying_stride + offset);
                            const auto vc = vertex_buffer.read<float>(best_sources.z * varying_stride + offset);
                            varying_buffer.write(
                                output_base + offset,
                                va * best_source_weight.x + vb * best_source_weight.y + vc * best_source_weight.z);
                        }
                        depth_values.write(pixel_index, best_depth);
                        coverage_buffer.write(pixel_index, 1.0f);
                    };
                });
            tile_raster_shader = runtime.insert_tile_raster(tile_raster_key, std::move(compiled));
        }
        profile_checkpoint("tile raster compile");
        stream << (*tile_raster_shader)(
                      *vertex_buffer, *primitive_bytes, tile_counts, tile_offsets, tile_indices,
                      depth_buffer, *varying_buffer, coverage_buffer,
                      tile_width, reference_capacity, target.width, target.height,
                      raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height)
                      .dispatch(target.width, target.height, raster.sample_count);
        if (profile_stages) stream << coverage_bytes->copy_to(host_coverage.data());
        profile_checkpoint("tile raster and varying resolve");
        if (profile_stages) {
            const auto covered = static_cast<size_t>(std::count_if(
                host_coverage.begin(), host_coverage.end(), [](float value) { return value != 0.0f; }));
            std::fprintf(stderr, "[feather raster profile] covered_samples=%zu/%zu\n",
                         covered, host_coverage.size());
        }
        *fragment_varyings = std::move(host_varyings);
        fragment_coverage->resize(sample_element_count * sizeof(float));
        return true;
    }
    if (false && use_fast_raster) {
        const auto* profile_value = std::getenv("FEATHER_RASTER_PROFILE_STAGES");
        const auto profile_stages = profile_value != nullptr && profile_value[0] != '\0' &&
                                    std::strcmp(profile_value, "0") != 0;
        auto profile_start = std::chrono::steady_clock::now();
        const auto profile_checkpoint = [&](const char* stage) {
            if (!profile_stages) return;
            stream << synchronize();
            const auto now = std::chrono::steady_clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(now - profile_start).count();
            std::fprintf(stderr, "[feather raster profile] %s %.3f ms\n", stage, elapsed);
            profile_start = now;
        };
        const auto depth_bits_key = coverage_resident_key ^ 0x6465707468626974ull;
        const auto owner_key = coverage_resident_key ^ 0x7072696d6f776e72ull;
        auto* depth_bits_bytes = runtime.resident_buffer(
            depth_bits_key, sample_element_count * sizeof(uint32_t));
        auto* owner_bytes = runtime.resident_buffer(
            owner_key, sample_element_count * sizeof(uint32_t));
        if (depth_bits_bytes == nullptr || owner_bytes == nullptr) {
            if (error != nullptr) *error = "fast raster ownership buffers are unavailable";
            return false;
        }
        auto depth_bits = depth_bits_bytes->view().as<uint32_t>();
        auto owners = owner_bytes->view().as<uint32_t>();
        stream << (*runtime.fast_raster_init_shader())(
                      depth_bits, owners, depth_buffer, coverage_buffer,
                      raster.clear_depth, raster.clear_depth_value)
                      .dispatch(target.width, target.height, raster.sample_count);
        profile_checkpoint("sample setup");

        uint64_t fast_shader_key = raster_shader_key ^ 0x6661737472617374ull;
        auto* fast_shader = runtime.find_fast_raster(fast_shader_key);
        if (fast_shader == nullptr) {
            auto compiled = device.compile<2>(
                [varying_stride, vertex_count, vertices_per_instance = raster.vertices_per_instance,
                 vertex_domain = raster.vertex_domain, sample_count = raster.sample_count](
                    ByteBufferVar vertex_buffer, BufferUInt index_buffer,
                    BufferUInt depth_bits, BufferUInt owners,
                    UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                    UInt scissor_x, UInt scissor_y, UInt scissor_width, UInt scissor_height,
                    UInt target_width, UInt target_height, UInt cull_mode, UInt front_face,
                    UInt depth_clamp, UInt pass) noexcept {
                    const auto triangle = dispatch_id().x;
                    const auto sample = dispatch_id().y;
                    const auto raster_base = triangle * 3u;
                    const auto instance = raster_base / vertices_per_instance;
                    const auto local_base = raster_base % vertices_per_instance;
                    const auto source_a = instance * vertex_domain + index_buffer.read(local_base);
                    const auto source_b = instance * vertex_domain + index_buffer.read(local_base + 1u);
                    const auto source_c = instance * vertex_domain + index_buffer.read(local_base + 2u);
                    ArrayFloat4<kMaximumClippedVertices> clipped_positions;
                    ArrayFloat3<kMaximumClippedVertices> clipped_weights;
                    clipped_positions[0u] = vertex_buffer.read<float4>(source_a * varying_stride);
                    clipped_positions[1u] = vertex_buffer.read<float4>(source_b * varying_stride);
                    clipped_positions[2u] = vertex_buffer.read<float4>(source_c * varying_stride);
                    clipped_weights[0u] = make_float3(1.0f, 0.0f, 0.0f);
                    clipped_weights[1u] = make_float3(0.0f, 1.0f, 0.0f);
                    clipped_weights[2u] = make_float3(0.0f, 0.0f, 1.0f);
                    UInt clipped_count = 3u;
                    clip_homogeneous_triangle(clipped_positions, clipped_weights, clipped_count);
                    const auto viewport_origin = make_float2(
                        viewport_x.cast<float>(), viewport_y.cast<float>());
                    const auto viewport_size = make_float2(
                        viewport_width.cast<float>(), viewport_height.cast<float>());
                    const auto to_screen = [&](auto ndc) noexcept {
                        return viewport_origin + make_float2(
                            (ndc.x + 1.0f) * 0.5f * viewport_size.x,
                            (1.0f - ndc.y) * 0.5f * viewport_size.y);
                    };
                    const auto lower_bound = make_float2(
                        scissor_x.cast<float>(), scissor_y.cast<float>());
                    const auto upper_bound = make_float2(
                        luisa::compute::min(scissor_x + scissor_width, target_width).cast<float>() - 1.0f,
                        luisa::compute::min(scissor_y + scissor_height, target_height).cast<float>() - 1.0f);
                    Float2 sample_position = def(make_float2(0.5f));
                    if (sample_count == 4u) {
                        $if (sample == 0u) { sample_position = make_float2(0.375f, 0.125f); }
                        $elif (sample == 1u) { sample_position = make_float2(0.875f, 0.375f); }
                        $elif (sample == 2u) { sample_position = make_float2(0.125f, 0.625f); }
                        $else { sample_position = make_float2(0.625f, 0.875f); };
                    }

                    UInt fan_index = 1u;
                    $while (fan_index + 1u < clipped_count) {
                        Float4 a = clipped_positions[0u];
                        Float4 b = clipped_positions[fan_index];
                        Float4 c = clipped_positions[fan_index + 1u];
                        Float2 pa = def(a.xy() / a.w);
                        Float2 pb = def(b.xy() / b.w);
                        Float2 pc = def(c.xy() / c.w);
                        Float area = def((pb.x - pa.x) * (pc.y - pa.y) -
                                         (pb.y - pa.y) * (pc.x - pa.x));
                        Bool front = area < 0.0f;
                        $if (front_face != 0u) { front = !front; };
                        Bool culled = def(false);
                        $if (cull_mode == 1u) { culled = front; }
                        $elif (cull_mode == 2u) { culled = !front; }
                        $elif (cull_mode == 3u) { culled = true; };
                        Float2 sa = def(to_screen(pa));
                        Float2 sb = def(to_screen(pb));
                        Float2 sc = def(to_screen(pc));
                        const auto minimum = luisa::compute::max(
                            luisa::compute::floor(luisa::compute::min(sa, luisa::compute::min(sb, sc))),
                            lower_bound);
                        const auto maximum = luisa::compute::min(
                            luisa::compute::ceil(luisa::compute::max(sa, luisa::compute::max(sb, sc))),
                            upper_bound);
                        $if (!culled & abs(area) > 1e-7f & all(minimum <= maximum)) {
                            UInt y = minimum.y.cast<uint>();
                            $while (y <= maximum.y.cast<uint>()) {
                                UInt x = minimum.x.cast<uint>();
                                $while (x <= maximum.x.cast<uint>()) {
                                    const auto viewport_pixel = make_float2(
                                        x.cast<float>(), y.cast<float>()) + sample_position - viewport_origin;
                                    const auto p = make_float2(
                                        viewport_pixel.x / viewport_size.x * 2.0f - 1.0f,
                                        1.0f - viewport_pixel.y / viewport_size.y * 2.0f);
                                    Float edge0_raw = def((pb.x - p.x) * (pc.y - p.y) -
                                                          (pb.y - p.y) * (pc.x - p.x));
                                    Float edge1_raw = def((pc.x - p.x) * (pa.y - p.y) -
                                                          (pc.y - p.y) * (pa.x - p.x));
                                    Float edge2_raw = def((pa.x - p.x) * (pb.y - p.y) -
                                                          (pa.y - p.y) * (pb.x - p.x));
                                    const auto positive = area > 0.0f;
                                    Float edge0 = def(ite(positive, edge0_raw, -edge0_raw));
                                    Float edge1 = def(ite(positive, edge1_raw, -edge1_raw));
                                    Float edge2 = def(ite(positive, edge2_raw, -edge2_raw));
                                    const auto top_left_bc = ite(
                                        positive,
                                        (pc.y > pb.y) | ((pc.y == pb.y) & (pc.x < pb.x)),
                                        (pb.y > pc.y) | ((pb.y == pc.y) & (pb.x < pc.x)));
                                    const auto top_left_ca = ite(
                                        positive,
                                        (pa.y > pc.y) | ((pa.y == pc.y) & (pa.x < pc.x)),
                                        (pc.y > pa.y) | ((pc.y == pa.y) & (pc.x < pa.x)));
                                    const auto top_left_ab = ite(
                                        positive,
                                        (pb.y > pa.y) | ((pb.y == pa.y) & (pb.x < pa.x)),
                                        (pa.y > pb.y) | ((pa.y == pb.y) & (pa.x < pb.x)));
                                    const auto covered =
                                        ((edge0 > 0.0f) | ((edge0 == 0.0f) & top_left_bc)) &
                                        ((edge1 > 0.0f) | ((edge1 == 0.0f) & top_left_ca)) &
                                        ((edge2 > 0.0f) | ((edge2 == 0.0f) & top_left_ab));
                                    $if (covered) {
                                        Float inverse_area = def(1.0f / abs(area));
                                        Float w0 = def(edge0 * inverse_area);
                                        Float w1 = def(edge1 * inverse_area);
                                        Float w2 = def(edge2 * inverse_area);
                                        Float candidate_depth = w0 * (a.z / a.w) +
                                                                w1 * (b.z / b.w) +
                                                                w2 * (c.z / c.w);
                                        const auto depth_in_clip = candidate_depth >= 0.0f & candidate_depth <= 1.0f;
                                        $if (depth_clamp != 0u) {
                                            candidate_depth = clamp(candidate_depth, 0.0f, 1.0f);
                                        };
                                        $if ((depth_clamp != 0u) | depth_in_clip) {
                                            const auto pixel = y * target_width + x;
                                            const auto output_index = pixel * sample_count + sample;
                                            const auto candidate_bits = candidate_depth.bitcast<uint>();
                                            const auto primitive = triangle * kClippedPrimitiveStride + fan_index - 1u;
                                            $if (pass == 0u) {
                                                depth_bits.atomic(output_index).fetch_min(candidate_bits);
                                            }
                                            $elif (depth_bits.read(output_index) == candidate_bits) {
                                                owners.atomic(output_index).fetch_min(primitive);
                                            };
                                        };
                                    };
                                    x += 1u;
                                };
                                y += 1u;
                            };
                        };
                        fan_index += 1u;
                    };
                });
            fast_shader = runtime.insert_fast_raster(fast_shader_key, std::move(compiled));
        }
        profile_checkpoint("coverage shader compile");
        const auto triangle_count = raster.vertex_count / 3u;
        for (uint32_t pass = 0u; pass < 2u; ++pass) {
            stream << (*fast_shader)(
                          *vertex_buffer, index_buffer, depth_bits, owners,
                          raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height,
                          raster.scissor_x, raster.scissor_y, raster.scissor_width, raster.scissor_height,
                          target.width, target.height, raster.cull_mode, raster.front_face,
                          raster.depth_clamp, pass)
                          .dispatch(triangle_count, raster.sample_count);
            profile_checkpoint(pass == 0u ? "depth arbitration" : "primitive ownership");
        }

        const auto resolve_shader_key = fast_shader_key ^ 0x7265736f6c7665ull;
        auto* resolve_shader = runtime.find_fast_raster_resolve(resolve_shader_key);
        if (resolve_shader == nullptr) {
            auto compiled = device.compile<3>(
                [varying_stride, vertices_per_instance = raster.vertices_per_instance,
                 vertex_domain = raster.vertex_domain, sample_count = raster.sample_count](
                    ByteBufferVar vertex_buffer, BufferUInt index_buffer, BufferUInt owners,
                    BufferFloat depth_values, ByteBufferVar varying_buffer, BufferFloat coverage_buffer,
                    UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height) noexcept {
                    const auto id = dispatch_id();
                    const auto pixel_index = (id.y * dispatch_size().x + id.x) * sample_count + id.z;
                    const auto primitive = owners.read(pixel_index);
                    $if (primitive != ~0u) {
                        const auto triangle = primitive / kClippedPrimitiveStride;
                        const auto fan_index = primitive % kClippedPrimitiveStride + 1u;
                        const auto raster_base = triangle * 3u;
                        const auto instance = raster_base / vertices_per_instance;
                        const auto local_base = raster_base % vertices_per_instance;
                        UInt source_a = def(instance * vertex_domain + index_buffer.read(local_base));
                        UInt source_b = def(instance * vertex_domain + index_buffer.read(local_base + 1u));
                        UInt source_c = def(instance * vertex_domain + index_buffer.read(local_base + 2u));
                        ArrayFloat4<kMaximumClippedVertices> clipped_positions;
                        ArrayFloat3<kMaximumClippedVertices> clipped_weights;
                        clipped_positions[0u] = vertex_buffer.read<float4>(source_a * varying_stride);
                        clipped_positions[1u] = vertex_buffer.read<float4>(source_b * varying_stride);
                        clipped_positions[2u] = vertex_buffer.read<float4>(source_c * varying_stride);
                        clipped_weights[0u] = make_float3(1.0f, 0.0f, 0.0f);
                        clipped_weights[1u] = make_float3(0.0f, 1.0f, 0.0f);
                        clipped_weights[2u] = make_float3(0.0f, 0.0f, 1.0f);
                        UInt clipped_count = 3u;
                        clip_homogeneous_triangle(clipped_positions, clipped_weights, clipped_count);
                        Float4 a = clipped_positions[0u];
                        Float4 b = clipped_positions[fan_index];
                        Float4 c = clipped_positions[fan_index + 1u];
                        Float3 source_weight_a = clipped_weights[0u];
                        Float3 source_weight_b = clipped_weights[fan_index];
                        Float3 source_weight_c = clipped_weights[fan_index + 1u];
                        Float2 pa = def(a.xy() / a.w);
                        Float2 pb = def(b.xy() / b.w);
                        Float2 pc = def(c.xy() / c.w);
                        Float area = def((pb.x - pa.x) * (pc.y - pa.y) -
                                         (pb.y - pa.y) * (pc.x - pa.x));
                        Float2 sample_position = def(make_float2(0.5f));
                        if (sample_count == 4u) {
                            $if (id.z == 0u) { sample_position = make_float2(0.375f, 0.125f); }
                            $elif (id.z == 1u) { sample_position = make_float2(0.875f, 0.375f); }
                            $elif (id.z == 2u) { sample_position = make_float2(0.125f, 0.625f); }
                            $else { sample_position = make_float2(0.625f, 0.875f); };
                        }
                        Float2 viewport_pixel = def(make_float2(id.xy()) + sample_position -
                            make_float2(viewport_x.cast<float>(), viewport_y.cast<float>()));
                        Float2 p = def(make_float2(
                            viewport_pixel.x / viewport_width.cast<float>() * 2.0f - 1.0f,
                            1.0f - viewport_pixel.y / viewport_height.cast<float>() * 2.0f));
                        Float edge0 = def(abs((pb.x - p.x) * (pc.y - p.y) -
                                               (pb.y - p.y) * (pc.x - p.x)));
                        Float edge1 = def(abs((pc.x - p.x) * (pa.y - p.y) -
                                               (pc.y - p.y) * (pa.x - p.x)));
                        Float edge2 = def(abs((pa.x - p.x) * (pb.y - p.y) -
                                               (pa.y - p.y) * (pb.x - p.x)));
                        Float inverse_area = def(1.0f / abs(area));
                        Float w0 = def(edge0 * inverse_area);
                        Float w1 = def(edge1 * inverse_area);
                        Float w2 = def(edge2 * inverse_area);
                        Float q0 = def(w0 / a.w);
                        Float q1 = def(w1 / b.w);
                        Float q2 = def(w2 / c.w);
                        Float divisor = def(q0 + q1 + q2);
                        Float3 source_weight = def(
                            (source_weight_a * q0 + source_weight_b * q1 + source_weight_c * q2) /
                            divisor);
                        UInt output_base = def(pixel_index * varying_stride);
                        uint32_t offset = 0u;
                        for (; offset + sizeof(float4) <= varying_stride; offset += sizeof(float4)) {
                            Float4 va = def(vertex_buffer.read<float4>(source_a * varying_stride + offset));
                            Float4 vb = def(vertex_buffer.read<float4>(source_b * varying_stride + offset));
                            Float4 vc = def(vertex_buffer.read<float4>(source_c * varying_stride + offset));
                            Float4 interpolated = def(
                                va * source_weight.x + vb * source_weight.y + vc * source_weight.z);
                            varying_buffer.write(output_base + offset, interpolated);
                        }
                        for (; offset < varying_stride; offset += sizeof(float)) {
                            Float va = def(vertex_buffer.read<float>(source_a * varying_stride + offset));
                            Float vb = def(vertex_buffer.read<float>(source_b * varying_stride + offset));
                            Float vc = def(vertex_buffer.read<float>(source_c * varying_stride + offset));
                            Float interpolated = def(
                                va * source_weight.x + vb * source_weight.y + vc * source_weight.z);
                            varying_buffer.write(output_base + offset, interpolated);
                        }
                        Float output_depth = def(
                            w0 * (a.z / a.w) + w1 * (b.z / b.w) + w2 * (c.z / c.w));
                        depth_values.write(pixel_index, output_depth);
                        coverage_buffer.write(pixel_index, 1.0f);
                    };
                });
            resolve_shader = runtime.insert_fast_raster_resolve(resolve_shader_key, std::move(compiled));
        }
        profile_checkpoint("resolve shader compile");
        stream << (*resolve_shader)(
                      *vertex_buffer, index_buffer, owners, depth_buffer,
                      *varying_buffer, coverage_buffer,
                      raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height)
                      .dispatch(target.width, target.height, raster.sample_count);
        profile_checkpoint("varying resolve");
        *fragment_varyings = std::move(host_varyings);
        fragment_coverage->resize(sample_element_count * sizeof(float));
        return true;
    }
    auto* shader = runtime.find_raster(raster_shader_key);
    if (shader == nullptr) {
        auto compiled = device.compile<3>([varying_stride, vertex_count, vertices_per_instance = raster.vertices_per_instance,
                                           vertex_domain = raster.vertex_domain, has_depth,
                                           sample_count = raster.sample_count](ByteBufferVar vertex_buffer,
                                       BufferUInt index_buffer, ByteBufferVar varying_buffer, BufferFloat coverage_buffer,
                                       BufferFloat depth_buffer, BufferUInt stencil_buffer,
                                       UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                                       UInt scissor_x, UInt scissor_y, UInt scissor_width, UInt scissor_height,
                                       UInt cull_mode, UInt front_face, UInt polygon_mode,
                                       UInt depth_test, UInt depth_write,
                                       UInt depth_compare, UInt depth_clamp, UInt stencil_test,
                                       UInt stencil_front_fail, UInt stencil_front_pass,
                                       UInt stencil_front_depth_fail, UInt stencil_front_compare,
                                       UInt stencil_back_fail, UInt stencil_back_pass,
                                       UInt stencil_back_depth_fail, UInt stencil_back_compare,
                                       UInt stencil_read_mask, UInt stencil_write_mask, UInt stencil_reference,
                                       UInt clear_depth, UInt clear_stencil, Float clear_depth_value) noexcept {
        const auto pixel = dispatch_id().xy();
        const auto sample = dispatch_id().z;
        const auto pixel_index = (pixel.y * dispatch_size().x + pixel.x) * sample_count + sample;
        coverage_buffer.write(pixel_index, 0.0f);
        Float current_depth = def(1.0f);
        UInt current_stencil = def(0u);
        if (has_depth) {
            current_depth = depth_buffer.read(pixel_index);
            current_stencil = stencil_buffer.read(pixel_index);
        }
        $if (clear_depth != 0u) {
            current_depth = clear_depth_value;
            depth_buffer.write(pixel_index, current_depth);
        };
        $if (clear_stencil != 0u) {
            current_stencil = 0u;
            stencil_buffer.write(pixel_index, current_stencil);
        };

        const auto in_scissor = pixel.x >= scissor_x & pixel.y >= scissor_y &
                                 pixel.x < scissor_x + scissor_width & pixel.y < scissor_y + scissor_height;
        Float2 sample_position = def(make_float2(0.5f));
        if (sample_count == 4u) {
            $if (sample == 0u) { sample_position = make_float2(0.375f, 0.125f); }
            $elif (sample == 1u) { sample_position = make_float2(0.875f, 0.375f); }
            $elif (sample == 2u) { sample_position = make_float2(0.125f, 0.625f); }
            $else { sample_position = make_float2(0.625f, 0.875f); };
        }
        const auto viewport_pixel = make_float2(pixel) + sample_position -
                                    make_float2(viewport_x.cast<float>(), viewport_y.cast<float>());
        const auto viewport_size =
            make_float2(viewport_width.cast<float>(), viewport_height.cast<float>());
        const auto p = make_float2(
            viewport_pixel.x / viewport_size.x * 2.0f - 1.0f,
            1.0f - viewport_pixel.y / viewport_size.y * 2.0f);
        for (uint32_t triangle = 0u; triangle < vertex_count / 3u; ++triangle) {
            const auto raster_base = triangle * 3u;
            const auto instance = raster_base / vertices_per_instance;
            const auto local_base = raster_base % vertices_per_instance;
            const auto source_a = instance * vertex_domain + index_buffer.read(local_base);
            const auto source_b = instance * vertex_domain + index_buffer.read(local_base + 1u);
            const auto source_c = instance * vertex_domain + index_buffer.read(local_base + 2u);
            const auto a = vertex_buffer.read<float4>(source_a * varying_stride);
            const auto b = vertex_buffer.read<float4>(source_b * varying_stride);
            const auto c = vertex_buffer.read<float4>(source_c * varying_stride);
            const auto valid_w = a.w > 1e-7f & b.w > 1e-7f & c.w > 1e-7f;
            const auto pa = a.xy() / a.w;
            const auto pb = b.xy() / b.w;
            const auto pc = c.xy() / c.w;
            const auto area = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
            Bool front = area < 0.0f;// target-space Y points down
            $if (front_face != 0u) { front = !front; };
            Bool culled = def(false);
            $if (cull_mode == 1u) { culled = front; }
            $elif (cull_mode == 2u) { culled = !front; }
            $elif (cull_mode == 3u) { culled = true; };

            $if (valid_w & !culled & in_scissor & abs(area) > 1e-7f) {
                const auto edge0_raw = (pb.x - p.x) * (pc.y - p.y) - (pb.y - p.y) * (pc.x - p.x);
                const auto edge1_raw = (pc.x - p.x) * (pa.y - p.y) - (pc.y - p.y) * (pa.x - p.x);
                const auto edge2_raw = (pa.x - p.x) * (pb.y - p.y) - (pa.y - p.y) * (pb.x - p.x);
                const auto positive = area > 0.0f;
                const auto edge0 = ite(positive, edge0_raw, -edge0_raw);
                const auto edge1 = ite(positive, edge1_raw, -edge1_raw);
                const auto edge2 = ite(positive, edge2_raw, -edge2_raw);
                const auto top_left_bc = ite(positive,
                    (pc.y > pb.y) | ((pc.y == pb.y) & (pc.x < pb.x)),
                    (pb.y > pc.y) | ((pb.y == pc.y) & (pb.x < pc.x)));
                const auto top_left_ca = ite(positive,
                    (pa.y > pc.y) | ((pa.y == pc.y) & (pa.x < pc.x)),
                    (pc.y > pa.y) | ((pc.y == pa.y) & (pc.x < pa.x)));
                const auto top_left_ab = ite(positive,
                    (pb.y > pa.y) | ((pb.y == pa.y) & (pb.x < pa.x)),
                    (pa.y > pb.y) | ((pa.y == pb.y) & (pa.x < pb.x)));
                const auto filled = ((edge0 > 0.0f) | ((edge0 == 0.0f) & top_left_bc)) &
                                    ((edge1 > 0.0f) | ((edge1 == 0.0f) & top_left_ca)) &
                                    ((edge2 > 0.0f) | ((edge2 == 0.0f) & top_left_ab));
                Bool covered = filled;
                $if (polygon_mode == 1u) {
                    const auto distance0 = edge0 / length(pc - pb);
                    const auto distance1 = edge1 / length(pa - pc);
                    const auto distance2 = edge2 / length(pb - pa);
                    const auto nearest01 = ite(distance0 < distance1, distance0, distance1);
                    const auto nearest = ite(nearest01 < distance2, nearest01, distance2);
                    const auto viewport_min = ite(viewport_width < viewport_height,
                                                   viewport_width, viewport_height);
                    const auto line_width = 2.0f /
                        viewport_min.cast<float>();
                    covered = filled & (nearest <= line_width);
                }
                $elif (polygon_mode == 2u) {
                    const auto half_pixel = make_float2(1.0f / viewport_width.cast<float>(),
                                                        1.0f / viewport_height.cast<float>());
                    const auto near_a = all(abs(p - pa) <= half_pixel);
                    const auto near_b = all(abs(p - pb) <= half_pixel);
                    const auto near_c = all(abs(p - pc) <= half_pixel);
                    covered = near_a | near_b | near_c;
                };
                $if (covered) {
                    const auto inverse_area = 1.0f / abs(area);
                    const auto w0 = edge0 * inverse_area;
                    const auto w1 = edge1 * inverse_area;
                    const auto w2 = edge2 * inverse_area;
                    Float candidate_depth = w0 * (a.z / a.w) + w1 * (b.z / b.w) + w2 * (c.z / c.w);
                    const auto depth_in_clip = candidate_depth >= 0.0f & candidate_depth <= 1.0f;
                    $if (depth_clamp != 0u) { candidate_depth = clamp(candidate_depth, 0.0f, 1.0f); };
                    const auto stencil_fail_op = ite(front, stencil_front_fail, stencil_back_fail);
                    const auto stencil_pass_op = ite(front, stencil_front_pass, stencil_back_pass);
                    const auto stencil_depth_fail_op = ite(front, stencil_front_depth_fail, stencil_back_depth_fail);
                    const auto stencil_compare = ite(front, stencil_front_compare, stencil_back_compare);
                    Bool stencil_pass = def(true);
                    $if (stencil_test != 0u) {
                        const auto reference = stencil_reference & stencil_read_mask;
                        const auto stencil = current_stencil & stencil_read_mask;
                        $if (stencil_compare == 0u) { stencil_pass = false; }
                        $elif (stencil_compare == 1u) { stencil_pass = reference < stencil; }
                        $elif (stencil_compare == 2u) { stencil_pass = reference == stencil; }
                        $elif (stencil_compare == 3u) { stencil_pass = reference <= stencil; }
                        $elif (stencil_compare == 4u) { stencil_pass = reference > stencil; }
                        $elif (stencil_compare == 5u) { stencil_pass = reference != stencil; }
                        $elif (stencil_compare == 6u) { stencil_pass = reference >= stencil; }
                        $else { stencil_pass = true; };
                    };
                    Bool depth_pass = def(true);
                    $if (stencil_pass & (depth_test != 0u)) {
                        $if (depth_compare == 0u) { depth_pass = false; }
                        $elif (depth_compare == 1u) { depth_pass = candidate_depth < current_depth; }
                        $elif (depth_compare == 2u) { depth_pass = candidate_depth == current_depth; }
                        $elif (depth_compare == 3u) { depth_pass = candidate_depth <= current_depth; }
                        $elif (depth_compare == 4u) { depth_pass = candidate_depth > current_depth; }
                        $elif (depth_compare == 5u) { depth_pass = candidate_depth != current_depth; }
                        $elif (depth_compare == 6u) { depth_pass = candidate_depth >= current_depth; }
                        $else { depth_pass = true; };
                    };
                    $if (depth_clamp == 0u & !depth_in_clip) { depth_pass = false; };
                    $if (stencil_test != 0u) {
                        UInt operation = ite(stencil_pass, stencil_pass_op, stencil_fail_op);
                        $if (stencil_pass & !depth_pass) { operation = stencil_depth_fail_op; };
                        UInt stencil_result = current_stencil;
                        $if (operation == 1u) { stencil_result = 0u; }
                        $elif (operation == 2u) { stencil_result = stencil_reference; }
                        $elif (operation == 3u) { stencil_result = min(current_stencil + 1u, 255u); }
                        $elif (operation == 4u) { stencil_result = ite(current_stencil == 0u, 0u, current_stencil - 1u); }
                        $elif (operation == 5u) { stencil_result = ~current_stencil; }
                        $elif (operation == 6u) { stencil_result = (current_stencil + 1u) & 255u; }
                        $elif (operation == 7u) { stencil_result = (current_stencil - 1u) & 255u; };
                        current_stencil = (current_stencil & ~stencil_write_mask) |
                                          (stencil_result & stencil_write_mask);
                        current_stencil &= 255u;
                        stencil_buffer.write(pixel_index, current_stencil);
                    };
                    $if (stencil_pass & depth_pass) {
                        const auto q0 = w0 / a.w;
                        const auto q1 = w1 / b.w;
                        const auto q2 = w2 / c.w;
                        const auto varying = (a * q0 + b * q1 + c * q2) / (q0 + q1 + q2);
                        const auto output_base = pixel_index * varying_stride;
                        for (uint32_t lane = 0u; lane < varying_stride / sizeof(float); ++lane) {
                            const auto offset = lane * static_cast<uint32_t>(sizeof(float));
                            const auto va = vertex_buffer.read<float>(source_a * varying_stride + offset);
                            const auto vb = vertex_buffer.read<float>(source_b * varying_stride + offset);
                            const auto vc = vertex_buffer.read<float>(source_c * varying_stride + offset);
                            varying_buffer.write(output_base + offset,
                                                 (va * q0 + vb * q1 + vc * q2) / (q0 + q1 + q2));
                        }
                        coverage_buffer.write(pixel_index, 1.0f);
                        $if (depth_write != 0u) {
                            current_depth = candidate_depth;
                            depth_buffer.write(pixel_index, current_depth);
                        };
                    };
                };
            };
        }
    });
        shader = runtime.insert_raster(raster_shader_key, std::move(compiled));
    }
    stream << (*shader)(*vertex_buffer, index_buffer, *varying_buffer, coverage_buffer, depth_buffer, stencil_buffer,
                     raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height,
                     raster.scissor_x, raster.scissor_y, raster.scissor_width, raster.scissor_height,
                     raster.cull_mode, raster.front_face, raster.polygon_mode,
                     raster.depth_test, raster.depth_write,
                     raster.depth_compare, raster.depth_clamp, raster.stencil_test,
                     raster.stencil_front_fail, raster.stencil_front_pass,
                     raster.stencil_front_depth_fail, raster.stencil_front_compare,
                     raster.stencil_back_fail, raster.stencil_back_pass,
                     raster.stencil_back_depth_fail, raster.stencil_back_compare,
                     raster.stencil_read_mask, raster.stencil_write_mask, raster.stencil_reference,
                     raster.clear_depth, raster.clear_stencil, raster.clear_depth_value)
                  .dispatch(target.width, target.height, raster.sample_count);
    if (depth != nullptr) {
        stream << depth_bytes->copy_to(host_depth.data());
        if (depth->pixel_format == 100u) stream << stencil_bytes->copy_to(host_stencil.data());
    }
    const auto* profile_stages = std::getenv("FEATHER_RASTER_PROFILE_STAGES");
    if (depth != nullptr || (profile_stages != nullptr && profile_stages[0] != '\0' &&
                             std::strcmp(profile_stages, "0") != 0)) {
        stream << synchronize();
    }
    if (depth != nullptr && depth->pixel_format == 101u) {
        auto* output = reinterpret_cast<float*>(depth->bytes->data());
        for (size_t pixel = 0u; pixel < pixel_count; ++pixel) {
            auto value = host_depth[pixel * raster.sample_count];
            for (uint32_t sample = 1u; sample < raster.sample_count; ++sample) {
                value = std::min(value, host_depth[pixel * raster.sample_count + sample]);
            }
            output[pixel] = value;
        }
    } else if (depth != nullptr) {
        auto* packed = reinterpret_cast<uint32_t*>(depth->bytes->data());
        for (size_t i = 0u; i < pixel_count; ++i) {
            auto resolved_depth = host_depth[i * raster.sample_count];
            auto resolved_stencil = host_stencil[i * raster.sample_count];
            for (uint32_t sample = 1u; sample < raster.sample_count; ++sample) {
                const auto index = i * raster.sample_count + sample;
                if (host_depth[index] < resolved_depth) {
                    resolved_depth = host_depth[index];
                    resolved_stencil = host_stencil[index];
                }
            }
            const auto encoded_depth = static_cast<uint32_t>(
                std::clamp(resolved_depth, 0.0f, 1.0f) * 16777215.0f + 0.5f);
            packed[i] = (resolved_stencil << 24u) | (encoded_depth & 0x00ffffffu);
        }
    }
    *fragment_varyings = std::move(host_varyings);
    fragment_coverage->resize(sample_element_count * sizeof(float));
    return true;
}

} // namespace Feather::Luisa
