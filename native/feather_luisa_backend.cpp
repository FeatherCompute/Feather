#include "feather_luisa_backend.h"

#include <algorithm>
#include <charconv>
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
#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/shader.h>
#include <luisa/runtime/stream.h>
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

using RuntimeBuffer = std::variant<Buffer<float>, Buffer<int32_t>, Buffer<uint32_t>>;

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

template <typename F> decltype(auto) visit_buffer(RuntimeBuffer& buffer, F&& f) {
    return std::visit([&](auto& typed) -> decltype(auto) { return f(typed); }, buffer);
}

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
              std::span<HostBufferBinding> host_buffers, const DispatchInputs& dispatch, std::string* error) {
    if (error != nullptr)
        error->clear();
    if (dispatch.logical_y != 1 || dispatch.logical_z != 1) {
        if (error != nullptr)
            *error = "M2.1 Luisa slice requires a one-dimensional dispatch";
        return false;
    }

    xir::Module xir_module;
    Lowerer lowerer{module, lowering, error};
    auto* kernel = lowerer.lower(xir_module);
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
    std::vector<RuntimeBuffer> runtime_buffers;
    std::vector<luisa::compute::Function::Binding> bound_arguments;
    runtime_buffers.reserve(lowering.resources.size());
    bound_arguments.reserve(lowering.resources.size());

    for (const auto& resource : lowering.resources) {
        auto found = std::find_if(host_buffers.begin(), host_buffers.end(),
                                  [&](const auto& binding) { return binding.binding == resource.binding; });
        if (found == host_buffers.end() || found->bytes == nullptr || found->stride != 4 || found->bytes->empty() ||
            found->bytes->size() % found->stride != 0) {
            if (error != nullptr)
                *error = "Luisa buffer binding is missing or has an unsupported stride";
            return false;
        }
        const auto count = found->bytes->size() / found->stride;
        if (resource.element_type == "float" || resource.element_type == "System.Single") {
            runtime_buffers.emplace_back(device.create_buffer<float>(count));
        } else if (resource.element_type == "int" || resource.element_type == "System.Int32") {
            runtime_buffers.emplace_back(device.create_buffer<int32_t>(count));
        } else if (resource.element_type == "uint" || resource.element_type == "System.UInt32") {
            runtime_buffers.emplace_back(device.create_buffer<uint32_t>(count));
        } else {
            if (error != nullptr)
                *error = "M2.1 Luisa slice supports float, int, and uint buffers only";
            return false;
        }
        auto& runtime = runtime_buffers.back();
        visit_buffer(runtime, [&](auto& typed) {
            using T = buffer_element_t<std::remove_cvref_t<decltype(typed)>>;
            auto values = luisa::span<const T>{reinterpret_cast<const T*>(found->bytes->data()), count};
            stream << typed.copy_from(values) << synchronize();
            bound_arguments.emplace_back(
                luisa::compute::Function::BufferBinding{typed.handle(), 0u, typed.size_bytes()});
        });
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
    auto shader = device.create<Shader1D<>>(luisa::compute::Function{ast.get()}, ShaderOption{});
    stream << shader().dispatch(dispatch.logical_x) << synchronize();

    for (size_t i = 0; i < lowering.resources.size(); ++i) {
        const auto& resource = lowering.resources[i];
        if (resource.access != kAccessWrite && resource.access != kAccessReadWrite)
            continue;
        auto found = std::find_if(host_buffers.begin(), host_buffers.end(),
                                  [&](const auto& binding) { return binding.binding == resource.binding; });
        const auto count = found->bytes->size() / found->stride;
        visit_buffer(runtime_buffers[i], [&](auto& typed) {
            using T = buffer_element_t<std::remove_cvref_t<decltype(typed)>>;
            auto values = luisa::span<T>{reinterpret_cast<T*>(found->bytes->data()), count};
            stream << typed.copy_to(values) << synchronize();
        });
    }
    return true;
}

} // namespace Feather::Luisa
