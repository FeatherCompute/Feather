#include "feather_luisa_backend.h"
#include "feather_luisa_xir.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/byte_buffer.h>
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/volume.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/module.h>
#include <luisa/xir/translators/xir2ast.h>
#include <luisa/xir/verifier.h>

namespace Feather::Luisa {
namespace {

using namespace luisa::compute;
using namespace luisa::compute::xir;

constexpr uint8_t kTypePrimitive = 1;
constexpr uint8_t kStatementBlock = 1;
constexpr uint8_t kStatementAssignment = 3;
constexpr uint8_t kStatementReturn = 11;
constexpr uint8_t kExpressionLiteral = 1;
constexpr uint8_t kExpressionResourceElement = 5;
constexpr uint8_t kExpressionBinary = 7;
constexpr uint8_t kExpressionBuiltin = 18;
constexpr uint8_t kLValueResourceElement = 4;
constexpr uint8_t kResourceBuffer = 1;
constexpr uint8_t kResourceTexture2D = 2;
constexpr uint8_t kResourceTexture3D = 6;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessReadWrite = 3;
constexpr uint8_t kBuiltinThreadIndexX = 1;

const Type* scalar_type(const TypedIR::Module& module, uint32_t type_id) {
    if (type_id >= module.types.size())
        return nullptr;
    const auto& type = module.types[type_id];
    if (type.kind != kTypePrimitive || type.b != 32)
        return nullptr;
    switch (type.a) {
    case 0:
        return Type::of<bool>();
    case 1:
        return Type::of<int32_t>();
    case 2:
        return Type::of<uint32_t>();
    case 3:
        return Type::of<float>();
    default:
        return nullptr;
    }
}

ArithmeticOp binary_op(uint32_t op, bool* valid) {
    *valid = true;
    switch (op) {
    case 0:
        return ArithmeticOp::BINARY_ADD;
    case 1:
        return ArithmeticOp::BINARY_SUB;
    case 2:
        return ArithmeticOp::BINARY_MUL;
    case 3:
        return ArithmeticOp::BINARY_DIV;
    case 4:
        return ArithmeticOp::BINARY_MOD;
    case 5:
        return ArithmeticOp::BINARY_BIT_AND;
    case 6:
        return ArithmeticOp::BINARY_BIT_OR;
    case 7:
        return ArithmeticOp::BINARY_BIT_XOR;
    case 8:
        return ArithmeticOp::BINARY_SHIFT_LEFT;
    case 9:
        return ArithmeticOp::BINARY_SHIFT_RIGHT;
    default:
        *valid = false;
        return ArithmeticOp::BINARY_ADD;
    }
}

class Lowerer {
  public:
    Lowerer(const TypedIR::Module& module, const TypedIR::LoweringInputs& inputs, std::string* error)
        : module_(module), inputs_(inputs), error_(error) {}

    KernelFunction* lower(xir::Module& xir_module) {
        if (module_.entry_function >= module_.functions.size())
            return fail("missing FEIR entry function"), nullptr;
        const auto& entry = module_.functions[module_.entry_function];
        if (entry.kind != 0 || entry.name_id >= module_.strings.size() ||
            entry.body_statement_index >= module_.statements.size()) {
            return fail("M2.1 Luisa slice requires a one-dimensional compute entry"), nullptr;
        }
        if (inputs_.group_x <= 0 || inputs_.group_y != 1 || inputs_.group_z != 1) {
            return fail("M2.1 Luisa slice requires a one-dimensional thread group"), nullptr;
        }
        if (inputs_.group_x > 1024) {
            return fail("Luisa XIR requires at most 1024 threads in a thread group"), nullptr;
        }

        xir_module_ = &xir_module;
        kernel_ = xir_module.create_kernel();
        kernel_->set_name(module_.strings[entry.name_id]);
        // Luisa XIR requires 32-thread granularity. This slice observes global dispatch IDs only,
        // so rounding the declared 1D group does not alter its semantics.
        const auto block_x = std::max(32u, (static_cast<uint32_t>(inputs_.group_x) + 31u) & ~31u);
        kernel_->set_block_size(luisa::make_uint3(block_x, 1u, 1u));

        for (const auto& resource : inputs_.resources) {
            if (resource.kind != kResourceBuffer)
                return fail("M2.1 Luisa slice supports buffer resources only"), nullptr;
            const Type* element = nullptr;
            if (resource.element_type == "float" || resource.element_type == "System.Single") {
                element = Type::of<float>();
            } else if (resource.element_type == "int" || resource.element_type == "System.Int32") {
                element = Type::of<int32_t>();
            } else if (resource.element_type == "uint" || resource.element_type == "System.UInt32") {
                element = Type::of<uint32_t>();
            }
            if (element == nullptr || !element->is_scalar() || element->size() != 4u) {
                return fail("M2.1 Luisa slice requires 32-bit scalar buffer elements"), nullptr;
            }
            auto* argument = kernel_->create_resource_argument(Type::buffer(element));
            resources_.emplace(resource.name, Resource{argument, element, resource.binding, resource.access});
        }

        auto* body = kernel_->create_body_block();
        builder_.set_insertion_point(body);
        if (!lower_statement(entry.body_statement_index))
            return nullptr;
        if (!builder_.is_insertion_point_terminator())
            builder_.return_void();
        return kernel_;
    }

  private:
    struct Resource {
        ResourceArgument* argument;
        const Type* element_type;
        uint32_t binding;
        uint8_t access;
    };

    bool fail(std::string message) {
        if (error_ != nullptr && error_->empty())
            *error_ = std::move(message);
        return false;
    }

    std::string_view string(uint32_t id) const {
        return id < module_.strings.size() ? std::string_view{module_.strings[id]} : std::string_view{};
    }

    Value* index_value(uint32_t expression_id) {
        auto* value = lower_expression(expression_id);
        return value == nullptr ? nullptr : builder_.static_cast_if_necessary(Type::of<uint32_t>(), value);
    }

    Value* literal(const TypedIR::Expression& expression, const Type* type) {
        auto text = string(expression.name_id);
        if (type->is_float()) {
            std::string copy{text};
            char* end = nullptr;
            const auto value = std::strtof(copy.c_str(), &end);
            if (end == copy.c_str() || (*end != '\0' && !((*end == 'f' || *end == 'F') && end[1] == '\0')))
                return fail("invalid float literal in FEIR"), nullptr;
            return xir_module_->create_constant(type, &value);
        }
        if (type->is_int32()) {
            int32_t value{};
            auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size())
                return fail("invalid int literal in FEIR"), nullptr;
            return xir_module_->create_constant(type, &value);
        }
        if (type->is_uint32()) {
            if (!text.empty() && (text.back() == 'u' || text.back() == 'U'))
                text.remove_suffix(1u);
            uint32_t value{};
            auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec != std::errc{} || end != text.data() + text.size())
                return fail("invalid uint literal in FEIR"), nullptr;
            return xir_module_->create_constant(type, &value);
        }
        return fail("unsupported literal type in M2.1 Luisa slice"), nullptr;
    }

    Value* lower_expression(uint32_t id) {
        if (id >= module_.expressions.size())
            return fail("FEIR expression index is out of range"), nullptr;
        const auto& expression = module_.expressions[id];
        const auto* type = scalar_type(module_, expression.type_id);
        if (type == nullptr)
            return fail("M2.1 Luisa expression is not a 32-bit scalar"), nullptr;
        switch (expression.kind) {
        case kExpressionLiteral:
            return literal(expression, type);
        case kExpressionResourceElement: {
            const auto found = resources_.find(std::string{string(expression.name_id)});
            if (found == resources_.end())
                return fail("FEIR buffer expression names an unknown resource"), nullptr;
            auto* index = index_value(expression.a);
            return index == nullptr ? nullptr
                                    : builder_.call(type, ResourceReadOp::BUFFER_READ, {found->second.argument, index});
        }
        case kExpressionBinary: {
            auto* left = lower_expression(expression.a);
            auto* right = lower_expression(expression.b);
            if (left == nullptr || right == nullptr)
                return nullptr;
            bool valid = false;
            const auto op = binary_op(expression.op, &valid);
            if (!valid)
                return fail("unsupported FEIR binary operation in M2.1 Luisa slice"), nullptr;
            return builder_.call(type, op, {left, right});
        }
        case kExpressionBuiltin: {
            if (expression.op != kBuiltinThreadIndexX)
                return fail("M2.1 Luisa slice supports ThreadIds.X only"), nullptr;
            const uint32_t zero = 0;
            auto* index = xir_module_->create_constant(Type::of<uint32_t>(), &zero);
            auto* x =
                builder_.call(Type::of<uint32_t>(), ArithmeticOp::EXTRACT, {xir_module_->create_dispatch_id(), index});
            return builder_.static_cast_if_necessary(type, x);
        }
        default:
            return fail("unsupported FEIR expression kind in M2.1 Luisa slice"), nullptr;
        }
    }

    bool lower_statement(uint32_t id) {
        if (id >= module_.statements.size())
            return fail("FEIR statement index is out of range");
        const auto& statement = module_.statements[id];
        if (statement.kind == kStatementBlock) {
            if (statement.child_count > 0 &&
                (statement.first_child == TypedIR::NoIndex || statement.first_child > module_.children.size() ||
                 statement.child_count > module_.children.size() - statement.first_child)) {
                return fail("FEIR block child range is invalid");
            }
            for (uint32_t i = 0; i < statement.child_count; ++i) {
                if (!lower_statement(module_.children[statement.first_child + i]))
                    return false;
            }
            return true;
        }
        if (statement.kind == kStatementReturn && statement.a == TypedIR::NoIndex) {
            builder_.return_void();
            return true;
        }
        if (statement.kind != kStatementAssignment || statement.a >= module_.lvalues.size()) {
            return fail("unsupported FEIR statement kind in M2.1 Luisa slice");
        }
        const auto& target = module_.lvalues[statement.a];
        if (target.kind != kLValueResourceElement)
            return fail("M2.1 Luisa assignment target must be a buffer element");
        const auto found = resources_.find(std::string{string(target.name_id)});
        if (found == resources_.end() ||
            (found->second.access != kAccessWrite && found->second.access != kAccessReadWrite)) {
            return fail("FEIR assignment target is not a writable buffer resource");
        }
        auto* index = index_value(target.a);
        auto* value = lower_expression(statement.b);
        if (index == nullptr || value == nullptr)
            return false;
        builder_.call(ResourceWriteOp::BUFFER_WRITE, {found->second.argument, index, value});
        return true;
    }

    const TypedIR::Module& module_;
    const TypedIR::LoweringInputs& inputs_;
    std::string* error_;
    xir::Module* xir_module_ = nullptr;
    KernelFunction* kernel_ = nullptr;
    XIRBuilder builder_;
    std::unordered_map<std::string, Resource> resources_;
};

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
              const DispatchInputs& dispatch, std::string* error) {
    if (error != nullptr)
        error->clear();
    xir::Module xir_module;
    std::vector<BufferLayout> buffer_layouts;
    auto* kernel = LowerToXir(module, lowering, xir_module, &buffer_layouts, error);
    if (kernel == nullptr)
        return false;
    auto verification = xir_verify_module(&xir_module);
    if (!verification.succeeded()) {
        if (error != nullptr) {
            *error = "generated Luisa XIR failed verification: ";
            error->append(verification.errors.front().message.data(), verification.errors.front().message.size());
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
    runtime_buffers.reserve(host_buffers.size());
    staged_bytes.reserve(host_buffers.size());
    staged_bindings.reserve(host_buffers.size());
    bound_arguments.reserve(lowering.resources.size());
    runtime_textures.reserve(host_textures.size());
    staged_textures.reserve(host_textures.size());

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

    xir_to_ast_normalize_module(&xir_module);
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
    return true;
}

} // namespace Feather::Luisa
