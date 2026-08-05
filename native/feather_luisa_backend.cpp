#include "feather_luisa_backend.h"
#include "feather_luisa_xir.h"

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
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

} // namespace

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

    Context context{dispatch.runtime_directory};
    auto device = context.create_device("vk");
    auto stream = device.create_stream(StreamTag::COMPUTE);
    std::vector<ByteBuffer> runtime_buffers;
    std::vector<std::vector<unsigned char>> staged_bytes;
    std::vector<HostBufferBinding*> staged_bindings;
    std::vector<luisa::compute::Function::Binding> bound_arguments;
    std::vector<RuntimeTexture> runtime_textures;
    std::vector<HostTextureBinding*> staged_textures;
    std::vector<ByteBuffer> runtime_gradients;
    std::vector<std::vector<unsigned char>> staged_gradients;
    runtime_buffers.reserve(host_buffers.size());
    staged_bytes.reserve(host_buffers.size());
    staged_bindings.reserve(host_buffers.size());
    bound_arguments.reserve(lowering.resources.size());
    runtime_textures.reserve(host_textures.size());
    staged_textures.reserve(host_textures.size());
    runtime_gradients.reserve(gradient_layouts.size());
    staged_gradients.reserve(gradient_layouts.size());

    for (const auto& resource : lowering.resources) {
        if (resource.kind == kResourceTexture2D || resource.kind == kResourceTexture3D) {
            auto found = std::find_if(host_textures.begin(), host_textures.end(),
                                      [&](const auto& binding) { return binding.binding == resource.binding; });
            auto storage = found == host_textures.end() ? std::nullopt : pixel_storage(found->pixel_format);
            if (found == host_textures.end() || found->bytes == nullptr || found->bytes->empty() || !storage) {
                if (error != nullptr) *error = "Luisa texture binding is missing or uses an unsupported pixel format";
                return false;
            }
            if (resource.kind == kResourceTexture2D) {
                runtime_textures.emplace_back(device.create_image<float>(
                    *storage, luisa::make_uint2(found->width, found->height), found->mip_levels, true));
            } else {
                runtime_textures.emplace_back(device.create_volume<float>(
                    *storage, luisa::make_uint3(found->width, found->height, found->depth), found->mip_levels, true));
            }
            auto& runtime = runtime_textures.back();
            std::visit([&](auto& texture) {
                stream << texture.copy_from(found->bytes->data()) << synchronize();
                bound_arguments.emplace_back(luisa::compute::Function::TextureBinding{texture.handle(), 0u});
            }, runtime);
            staged_textures.push_back(&*found);
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
        runtime_buffers.emplace_back(device.create_byte_buffer(packed.size()));
        auto& runtime = runtime_buffers.back();
        stream << runtime.copy_from(packed.data()) << synchronize();
        bound_arguments.emplace_back(
            luisa::compute::Function::BufferBinding{runtime.handle(), 0u, runtime.size_bytes()});
        staged_bindings.push_back(&*found);
    }

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
        runtime_gradients.emplace_back(device.create_byte_buffer(staged_gradients.back().size()));
        auto& runtime = runtime_gradients.back();
        stream << runtime.copy_from(staged_gradients.back().data()) << synchronize();
        bound_arguments.emplace_back(
            luisa::compute::Function::BufferBinding{runtime.handle(), 0u, runtime.size_bytes()});
    }

    if (ad_inputs == nullptr) {
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
    if (ad_inputs != nullptr) {
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
    auto ast = xir_to_ast_translate(
        *kernel, XIR2ASTConfig{.strict = true,
                               .bound_arguments = luisa::span<const luisa::compute::Function::Binding>{
                                   bound_arguments.data(), bound_arguments.size()}});
    if (ast == nullptr) {
        if (error != nullptr)
            *error = "Luisa failed to translate generated XIR to its executable AST";
        return false;
    }
    auto shader = device.create<Shader3D<>>(luisa::compute::Function{ast.get()}, ShaderOption{});
    stream << shader().dispatch(luisa::make_uint3(dispatch.logical_x, dispatch.logical_y, dispatch.logical_z))
           << synchronize();

    size_t staged_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceBuffer)
            continue;
        auto* found = staged_bindings[staged_index];
        auto& runtime = runtime_buffers[staged_index++];
        if (resource.access != kAccessWrite && resource.access != kAccessReadWrite)
            continue;
        auto& packed = staged_bytes[staged_index - 1u];
        stream << runtime.copy_to(packed.data()) << synchronize();
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
    size_t texture_index = 0;
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kResourceTexture2D && resource.kind != kResourceTexture3D) continue;
        auto* found = staged_textures[texture_index];
        auto& runtime = runtime_textures[texture_index++];
        if (resource.access != kAccessWrite && resource.access != kAccessReadWrite) continue;
        std::visit([&](auto& texture) {
            stream << texture.copy_to(found->bytes->data()) << synchronize();
        }, runtime);
    }
    for (size_t i = 0; i < gradient_layouts.size(); ++i) {
        auto& packed = staged_gradients[i];
        stream << runtime_gradients[i].copy_to(packed.data()) << synchronize();
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

} // namespace Feather::Luisa
