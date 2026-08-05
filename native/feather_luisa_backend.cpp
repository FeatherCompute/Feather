#include "feather_luisa_backend.h"
#include "feather_luisa_xir.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <variant>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <luisa/ast/function.h>
#include <luisa/ast/type.h>
#include <luisa/dsl/sugar.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>
#include <luisa/xir/instructions/arithmetic.h>
#include <luisa/xir/module.h>
#include <luisa/xir/passes/autodiff.h>
#include <luisa/xir/passes/destructure_cfg.h>
#include <luisa/xir/passes/inline.h>
#include <luisa/xir/passes/reg2mem.h>
#include <luisa/xir/passes/restructure_cfg.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

namespace Feather::Luisa {
namespace {

using namespace luisa::compute;
using namespace luisa::compute::xir;

constexpr uint8_t kResourceBuffer = 1;
constexpr uint8_t kResourceTexture2D = 2;
constexpr uint8_t kResourceTexture3D = 6;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessReadWrite = 3;

void ensure_luisa_spirv_optimization_preset() noexcept {
    static std::once_flag once;
    std::call_once(once, [] {
        // EasyGPU's production Ultra preset is stronger than SPIRV-Tools' maintained
        // -O recipe. LC 0.9.0 has no Ultra enum, so use its strongest stable `full`
        // recipe by default while preserving an explicit caller override.
        if (std::getenv("LUISA_SPIRV_OPT_PASSES") != nullptr) return;
#if defined(_WIN32)
        _putenv_s("LUISA_SPIRV_OPT_PASSES", "full");
#else
        setenv("LUISA_SPIRV_OPT_PASSES", "full", 1);
#endif
    });
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
    case 3:
        *size = source.b * 16u;
        *alignment = 16u;
        return true;
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

class RuntimeState {
public:
    struct CachedKernel {
        std::unique_ptr<Shader3D<>> shader;
        std::vector<std::unique_ptr<ByteBuffer>> buffers;
        std::vector<RuntimeTexture> textures;
        std::vector<std::unique_ptr<ByteBuffer>> gradients;
    };

private:
    std::unique_ptr<Context> context_;
    std::unique_ptr<Device> device_;
    std::unique_ptr<Stream> stream_;
    std::string runtime_directory_;
    std::string backend_name_;
    std::unordered_map<uint64_t, std::unique_ptr<CachedKernel>> kernels_;

public:
    RuntimeState() = default;
    RuntimeState(const RuntimeState &) = delete;
    RuntimeState &operator=(const RuntimeState &) = delete;

    void ensure(std::string_view runtime_directory, std::string_view backend_name) {
        if (context_ != nullptr && runtime_directory_ == runtime_directory && backend_name_ == backend_name) return;
        reset();
        runtime_directory_ = runtime_directory;
        backend_name_ = backend_name;
        context_ = std::make_unique<Context>(runtime_directory_);
        device_ = std::make_unique<Device>(context_->create_device(backend_name_));
        stream_ = std::make_unique<Stream>(device_->create_stream(StreamTag::COMPUTE));
    }

    Device &device() noexcept { return *device_; }
    Stream &stream() noexcept { return *stream_; }

    CachedKernel *find(uint64_t key) noexcept {
        const auto it = kernels_.find(key);
        return it == kernels_.end() ? nullptr : it->second.get();
    }

    void insert(uint64_t key, std::unique_ptr<CachedKernel> kernel) {
        kernels_[key] = std::move(kernel);
    }

    void reset() noexcept {
        if (stream_ != nullptr) stream_->synchronize();
        kernels_.clear();
        stream_.reset();
        device_.reset();
        context_.reset();
        runtime_directory_.clear();
        backend_name_.clear();
    }

    void abandon() noexcept {
        // Deliberately leak runtime objects during process teardown: their
        // destructors call into dynamically loaded Luisa/Vulkan code.
        (void)context_.release();
        (void)device_.release();
        (void)stream_.release();
        // The owner is intentionally leaked by runtime_state(); leave cached
        // shaders and resources untouched so their destructors cannot run after
        // the dynamically loaded Luisa backend has been unloaded.
        runtime_directory_.clear();
        backend_name_.clear();
    }
};

RuntimeState &runtime_state() {
    // The native backend may be unloaded before C++ static destructors run.
    // Keep the owner itself alive until process exit; explicit Shutdown handles
    // normal context teardown and Abandon releases only the dynamically loaded
    // runtime objects on the process-exit path.
    static auto *state = new RuntimeState();
    return *state;
}

} // namespace

void Shutdown() {
    runtime_state().reset();
}

void Abandon() noexcept {
    runtime_state().abandon();
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

bool Dispatch(const TypedIR::Module& module, const TypedIR::LoweringInputs& lowering,
              std::span<HostBufferBinding> host_buffers, std::span<HostTextureBinding> host_textures,
              const DispatchInputs& dispatch, const AdInputs* ad_inputs,
              std::span<AdGradientBinding> gradients, std::string* error) {
    if (error != nullptr)
        error->clear();
    ensure_luisa_spirv_optimization_preset();
    xir::Module xir_module;
    std::vector<BufferLayout> buffer_layouts;
    std::vector<AdGradientLayout> gradient_layouts;
    auto* kernel = LowerToXir(module, lowering, xir_module, &buffer_layouts,
                              ad_inputs, &gradient_layouts, error);
    if (kernel == nullptr)
        return false;
    auto verification = xir_verify_module(&xir_module);
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

    auto &runtime = runtime_state();
    runtime.ensure(dispatch.runtime_directory, dispatch.backend_name);
    auto &device = runtime.device();
    auto &stream = runtime.stream();
    auto *cached = dispatch.shader_cache_key == 0u ? nullptr : runtime.find(dispatch.shader_cache_key);
    bool cache_hit = cached != nullptr && cached->shader != nullptr;
    std::vector<ByteBuffer *> runtime_buffers;
    std::vector<std::unique_ptr<ByteBuffer>> owned_buffers;
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
    size_t texture_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind == kResourceTexture2D || resource.kind == kResourceTexture3D) {
            auto found = std::find_if(host_textures.begin(), host_textures.end(),
                                      [&](const auto& binding) { return binding.binding == resource.binding; });
            auto storage = found == host_textures.end() ? std::nullopt : pixel_storage(found->pixel_format);
            if (found == host_textures.end() || found->bytes == nullptr || found->bytes->empty() || !storage) {
                if (error != nullptr) *error = "Luisa texture binding is missing or uses an unsupported pixel format";
                return false;
            }
            RuntimeTexture *runtime = nullptr;
            if (cache_hit) {
                if (texture_index >= cached->textures.size()) {
                    if (error != nullptr) *error = "Luisa shader cache resource layout changed";
                    return false;
                }
                runtime = &cached->textures[texture_index];
            } else if (resource.kind == kResourceTexture2D) {
                owned_textures.emplace_back(device.create_image<float>(
                    *storage, luisa::make_uint2(found->width, found->height), found->mip_levels, true));
                runtime = &owned_textures.back();
            } else {
                owned_textures.emplace_back(device.create_volume<float>(
                    *storage, luisa::make_uint3(found->width, found->height, found->depth), found->mip_levels, true));
                runtime = &owned_textures.back();
            }
            runtime_textures.push_back(runtime);
            std::visit([&](auto& texture) {
                stream << texture.copy_from(found->bytes->data()) << synchronize();
                if (!cache_hit)
                    bound_arguments.emplace_back(luisa::compute::Function::TextureBinding{texture.handle(), 0u});
            }, *runtime);
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
        staged_bytes.emplace_back(count * layout->device_type->size(), 0u);
        auto& packed = staged_bytes.back();
        for (size_t i = 0; i < count; ++i) {
            if (!repack_value(module, layout->feir_type_id, layout->device_type,
                              found->bytes->data() + i * found->stride,
                              packed.data() + i * layout->device_type->size(), true)) {
                if (error != nullptr) *error = "Luisa failed to repack a Feather buffer element";
                return false;
            }
        }
        ByteBuffer *runtime = nullptr;
        if (cache_hit) {
            if (buffer_index >= cached->buffers.size() || cached->buffers[buffer_index]->size_bytes() != packed.size()) {
                if (error != nullptr) *error = "Luisa shader cache buffer layout changed";
                return false;
            }
            runtime = cached->buffers[buffer_index].get();
        } else {
            owned_buffers.emplace_back(std::make_unique<ByteBuffer>(device.create_byte_buffer(packed.size())));
            runtime = owned_buffers.back().get();
        }
        stream << runtime->copy_from(packed.data()) << synchronize();
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
            runtime = cached->gradients[gradient_index].get();
        } else {
            owned_gradients.emplace_back(
                std::make_unique<ByteBuffer>(device.create_byte_buffer(staged_gradients.back().size())));
            runtime = owned_gradients.back().get();
        }
        stream << runtime->copy_from(staged_gradients.back().data()) << synchronize();
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
        auto destructured = destructure_cfg_pass_run_on_module(&xir_module);
        if (destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure XIR control flow before callable inlining";
            return false;
        }
        auto inlined = inline_all_pass_run_on_module(&xir_module);
        if (inlined.skipped_recursive_callable_count != 0u ||
            inlined.skipped_structured_call_count != 0u ||
            inlined.skipped_constrained_call_count != 0u ||
            inlined.skipped_metadata_call_count != 0u ||
            inlined.skipped_declaration_call_count != 0u ||
            inlined.rejected_malformed_call_count != 0u) {
            if (error != nullptr) *error = "Luisa could not inline the generated FEIR callable graph";
            return false;
        }
        xir_to_ast_normalize_module(&xir_module);
        auto inlined_verification = xir_verify_module(&xir_module);
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
        xir_to_ast_normalize_module(&xir_module);
        auto destructured = destructure_cfg_pass_run_on_module(&xir_module);
        if (destructured.error_count != 0u) {
            if (error != nullptr) *error = "Luisa failed to destructure XIR control flow before autodiff";
            return false;
        }
        auto inlined = inline_all_pass_run_on_module(
            &xir_module, InlineOptions{.allow_autodiff_scope_in_caller = true});
        if (inlined.skipped_recursive_callable_count != 0u ||
            inlined.skipped_structured_call_count != 0u ||
            inlined.skipped_constrained_call_count != 0u ||
            inlined.rejected_malformed_call_count != 0u) {
            if (error != nullptr) *error = "Luisa could not inline the complete FEIR callable graph before autodiff";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(&xir_module));
        auto restructured = restructure_cfg_pass_run_on_module(&xir_module);
        if (!restructured.succeeded()) {
            if (error != nullptr) *error = "Luisa failed to restructure XIR control flow before autodiff";
            return false;
        }
        static_cast<void>(reg2mem_pass_run_on_module(&xir_module));
        auto ad = autodiff_pass_run_on_module(&xir_module);
        if (ad.transformed_scope_count == 0u) {
            if (error != nullptr) *error = "Luisa XIR autodiff did not transform the generated AD scope";
            return false;
        }
        auto ad_verification = xir_verify_module(&xir_module);
        if (!ad_verification.succeeded()) {
            if (error != nullptr) {
                *error = "Luisa autodiff output failed XIR verification: ";
                error->append(ad_verification.errors.front().message.data(), ad_verification.errors.front().message.size());
            }
            return false;
        }
        xir_to_ast_normalize_module(&xir_module);
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
            entry->buffers = std::move(owned_buffers);
            entry->textures = std::move(owned_textures);
            entry->gradients = std::move(owned_gradients);
            runtime.insert(dispatch.shader_cache_key, std::move(entry));
            cached = runtime.find(dispatch.shader_cache_key);
            cache_hit = true;
        }
    }
    auto &cached_shader = cache_hit ? *cached->shader : *shader;
    stream << cached_shader().dispatch(luisa::make_uint3(dispatch.logical_x, dispatch.logical_y, dispatch.logical_z))
           << synchronize();

    size_t staged_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceBuffer)
            continue;
        auto* found = staged_bindings[staged_index];
        auto* runtime = runtime_buffers[staged_index++];
        if (resource.access != kAccessWrite && resource.access != kAccessReadWrite)
            continue;
        auto& packed = staged_bytes[staged_index - 1u];
        stream << runtime->copy_to(packed.data()) << synchronize();
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
    size_t output_texture_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceTexture2D && resource.kind != kResourceTexture3D) continue;
        auto* found = staged_textures[output_texture_index];
        auto* runtime = runtime_textures[output_texture_index++];
        if (resource.access != kAccessWrite && resource.access != kAccessReadWrite) continue;
        std::visit([&](auto& texture) {
            stream << texture.copy_to(found->bytes->data()) << synchronize();
        }, *runtime);
    }
    for (size_t i = 0; i < gradient_layouts.size(); ++i) {
        auto& packed = staged_gradients[i];
        stream << runtime_gradients[i]->copy_to(packed.data()) << synchronize();
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

bool DispatchVerticalRaster(HostBufferBinding vertices, HostTextureBinding target,
                            HostTextureBinding* depth, const RasterDispatchInputs& raster,
                            const DispatchInputs& dispatch,
                            std::vector<unsigned char>* fragment_varyings,
                            std::vector<unsigned char>* fragment_coverage,
                            std::string* error) {
    // Opt-in compute-only triangle assembly and raster stage between generated vertex and
    // fragment FEIR dispatches.
    if (error != nullptr) error->clear();
    if (vertices.bytes == nullptr || target.bytes == nullptr || fragment_varyings == nullptr ||
        fragment_coverage == nullptr || vertices.stride < sizeof(float) * 4u || raster.vertex_count < 3u ||
        raster.vertex_count % 3u != 0u || vertices.bytes->size() < vertices.stride * raster.vertex_count ||
        target.width == 0u || target.height == 0u ||
        target.depth != 1u) {
        if (error != nullptr) *error = "vertical raster bindings are incomplete";
        return false;
    }
    if (target.pixel_format != 3u && target.pixel_format != 4u && target.pixel_format != 10u) {
        if (error != nullptr) *error = "vertical raster supports only Rgba8, Bgra8, and Rgba32Float targets";
        return false;
    }
    const auto pixel_count = static_cast<size_t>(target.width) * target.height;
    if (target.bytes->size() < pixel_count * (target.pixel_format == 10u ? sizeof(float4) : sizeof(uint32_t))) {
        if (error != nullptr) *error = "vertical raster color storage is too small";
        return false;
    }
    if (depth != nullptr &&
        (depth->bytes == nullptr || depth->pixel_format != 101u || depth->width != target.width ||
         depth->height != target.height || depth->bytes->size() < pixel_count * sizeof(float))) {
        if (error != nullptr) *error = "vertical raster depth storage must be matching Depth32Float";
        return false;
    }

    auto &runtime = runtime_state();
    runtime.ensure(dispatch.runtime_directory, dispatch.backend_name);
    auto &device = runtime.device();
    auto &stream = runtime.stream();

    if (vertices.stride % sizeof(float) != 0u) {
        if (error != nullptr) *error = "vertical raster varying stride must be float-aligned";
        return false;
    }
    auto vertex_buffer = device.create_byte_buffer(vertices.stride * raster.vertex_count);
    std::vector<unsigned char> host_varyings(pixel_count * vertices.stride, 0u);
    std::vector<float> host_coverage(pixel_count, 0.0f);
    auto varying_buffer = device.create_byte_buffer(host_varyings.size());
    auto coverage_buffer = device.create_buffer<float>(pixel_count);
    std::vector<float> host_depth(pixel_count, 1.0f);
    if (depth != nullptr) {
        std::memcpy(host_depth.data(), depth->bytes->data(), pixel_count * sizeof(float));
    }
    auto depth_buffer = device.create_buffer<float>(pixel_count);
    auto storage = target.pixel_format == 10u ? PixelStorage::FLOAT4 : PixelStorage::BYTE4;
    auto image = device.create_image<float>(storage, make_uint2(target.width, target.height), 1u, true);
    stream << vertex_buffer.copy_from(vertices.bytes->data())
           << varying_buffer.copy_from(host_varyings.data())
           << coverage_buffer.copy_from(host_coverage.data())
           << depth_buffer.copy_from(host_depth.data())
           << image.copy_from(target.bytes->data());

    const auto varying_stride = vertices.stride;
    const auto vertex_count = raster.vertex_count;
    auto shader = device.compile<2>([varying_stride, vertex_count](ImageFloat output, ByteBufferVar vertex_buffer,
                                       ByteBufferVar varying_buffer, BufferFloat coverage_buffer,
                                       BufferFloat depth_buffer,
                                       UInt viewport_x, UInt viewport_y, UInt viewport_width, UInt viewport_height,
                                       UInt scissor_x, UInt scissor_y, UInt scissor_width, UInt scissor_height,
                                       UInt cull_mode, UInt front_face, UInt depth_test, UInt depth_write,
                                       UInt depth_compare, UInt clear_depth, Float clear_depth_value,
                                       UInt clear_color, Float4 clear_color_value) noexcept {
        const auto pixel = dispatch_id().xy();
        const auto pixel_index = pixel.y * dispatch_size().x + pixel.x;
        coverage_buffer.write(pixel_index, 0.0f);
        $if (clear_color != 0u) {
            output.write(pixel, clear_color_value);
        };
        Float current_depth = depth_buffer.read(pixel_index);
        $if (clear_depth != 0u) {
            current_depth = clear_depth_value;
            depth_buffer.write(pixel_index, current_depth);
        };

        const auto in_scissor = pixel.x >= scissor_x & pixel.y >= scissor_y &
                                 pixel.x < scissor_x + scissor_width & pixel.y < scissor_y + scissor_height;
        const auto viewport_pixel = make_float2(pixel) + 0.5f -
                                    make_float2(viewport_x.cast<float>(), viewport_y.cast<float>());
        const auto viewport_size =
            make_float2(viewport_width.cast<float>(), viewport_height.cast<float>());
        const auto p = make_float2(
            viewport_pixel.x / viewport_size.x * 2.0f - 1.0f,
            1.0f - viewport_pixel.y / viewport_size.y * 2.0f);
        for (uint32_t triangle = 0u; triangle < vertex_count / 3u; ++triangle) {
            const auto triangle_base = triangle * 3u * varying_stride;
            const auto a = vertex_buffer.read<float4>(triangle_base);
            const auto b = vertex_buffer.read<float4>(triangle_base + varying_stride);
            const auto c = vertex_buffer.read<float4>(triangle_base + varying_stride * 2u);
            const auto valid_w = abs(a.w) > 1e-7f & abs(b.w) > 1e-7f & abs(c.w) > 1e-7f;
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
                const auto w0 = ((pb.x - p.x) * (pc.y - p.y) - (pb.y - p.y) * (pc.x - p.x)) / area;
                const auto w1 = ((pc.x - p.x) * (pa.y - p.y) - (pc.y - p.y) * (pa.x - p.x)) / area;
                const auto w2 = 1.0f - w0 - w1;
                $if (w0 >= 0.0f & w1 >= 0.0f & w2 >= 0.0f) {
                    const auto candidate_depth = w0 * (a.z / a.w) + w1 * (b.z / b.w) + w2 * (c.z / c.w);
                    Bool depth_pass = def(true);
                    $if (depth_test != 0u) {
                        $if (depth_compare == 0u) { depth_pass = false; }
                        $elif (depth_compare == 1u) { depth_pass = candidate_depth < current_depth; }
                        $elif (depth_compare == 2u) { depth_pass = candidate_depth == current_depth; }
                        $elif (depth_compare == 3u) { depth_pass = candidate_depth <= current_depth; }
                        $elif (depth_compare == 4u) { depth_pass = candidate_depth > current_depth; }
                        $elif (depth_compare == 5u) { depth_pass = candidate_depth != current_depth; }
                        $elif (depth_compare == 6u) { depth_pass = candidate_depth >= current_depth; }
                        $else { depth_pass = true; };
                    };
                    $if (depth_pass) {
                        const auto q0 = w0 / a.w;
                        const auto q1 = w1 / b.w;
                        const auto q2 = w2 / c.w;
                        const auto varying = (a * q0 + b * q1 + c * q2) / (q0 + q1 + q2);
                        output.write(pixel, varying);
                        const auto output_base = pixel_index * varying_stride;
                        for (uint32_t lane = 0u; lane < varying_stride / sizeof(float); ++lane) {
                            const auto offset = lane * static_cast<uint32_t>(sizeof(float));
                            const auto va = vertex_buffer.read<float>(triangle_base + offset);
                            const auto vb = vertex_buffer.read<float>(triangle_base + varying_stride + offset);
                            const auto vc = vertex_buffer.read<float>(triangle_base + varying_stride * 2u + offset);
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
    stream << shader(image, vertex_buffer, varying_buffer, coverage_buffer, depth_buffer,
                     raster.viewport_x, raster.viewport_y, raster.viewport_width, raster.viewport_height,
                     raster.scissor_x, raster.scissor_y, raster.scissor_width, raster.scissor_height,
                     raster.cull_mode, raster.front_face, raster.depth_test, raster.depth_write,
                     raster.depth_compare, raster.clear_depth, raster.clear_depth_value,
                     raster.clear_color,
                     make_float4(raster.clear_color_r, raster.clear_color_g,
                                 raster.clear_color_b, raster.clear_color_a))
                  .dispatch(target.width, target.height)
           << image.copy_to(target.bytes->data())
           << varying_buffer.copy_to(host_varyings.data())
           << coverage_buffer.copy_to(host_coverage.data());
    if (depth != nullptr) {
        stream << depth_buffer.copy_to(depth->bytes->data());
    }
    stream << synchronize();
    *fragment_varyings = std::move(host_varyings);
    fragment_coverage->resize(pixel_count * sizeof(float));
    std::memcpy(fragment_coverage->data(), host_coverage.data(), fragment_coverage->size());
    return true;
}

} // namespace Feather::Luisa
