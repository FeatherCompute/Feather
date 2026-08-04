#include "feather_luisa_xir.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <luisa/ast/type.h>
#include <luisa/xir/builder.h>

namespace Feather::Luisa {
namespace {

using namespace luisa::compute;
using namespace luisa::compute::xir;

constexpr uint8_t kResourceBuffer = 1;
constexpr uint8_t kResourcePushConstant = 5;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessReadWrite = 3;

constexpr uint8_t kTypePrimitive = 1;
constexpr uint8_t kTypeVector = 2;
constexpr uint8_t kTypeMatrix = 3;
constexpr uint8_t kTypeStruct = 4;
constexpr uint8_t kTypeArray = 5;
constexpr uint8_t kTypeResourceWrapper = 6;
constexpr uint8_t kTypeVoid = 7;

constexpr uint8_t kStatementBlock = 1;
constexpr uint8_t kStatementLocalDeclaration = 2;
constexpr uint8_t kStatementAssignment = 3;
constexpr uint8_t kStatementCompoundAssignment = 4;
constexpr uint8_t kStatementIf = 5;
constexpr uint8_t kStatementFor = 6;
constexpr uint8_t kStatementWhile = 7;
constexpr uint8_t kStatementDoWhile = 8;
constexpr uint8_t kStatementBreak = 9;
constexpr uint8_t kStatementContinue = 10;
constexpr uint8_t kStatementReturn = 11;
constexpr uint8_t kStatementExpression = 12;
constexpr uint8_t kStatementBarrier = 13;
constexpr uint8_t kStatementIncrementDecrement = 14;
constexpr uint8_t kStatementSharedMemoryDeclaration = 15;

constexpr uint8_t kExpressionLiteral = 1;
constexpr uint8_t kExpressionLocal = 2;
constexpr uint8_t kExpressionParameter = 3;
constexpr uint8_t kExpressionField = 4;
constexpr uint8_t kExpressionResourceElement = 5;
constexpr uint8_t kExpressionUnary = 6;
constexpr uint8_t kExpressionBinary = 7;
constexpr uint8_t kExpressionComparison = 8;
constexpr uint8_t kExpressionLogical = 9;
constexpr uint8_t kExpressionConditional = 10;
constexpr uint8_t kExpressionConversion = 11;
constexpr uint8_t kExpressionConstructor = 12;
constexpr uint8_t kExpressionIntrinsic = 13;
constexpr uint8_t kExpressionCallableCall = 14;
constexpr uint8_t kExpressionSwizzle = 15;
constexpr uint8_t kExpressionMemberAccess = 16;
constexpr uint8_t kExpressionIndexAccess = 17;
constexpr uint8_t kExpressionBuiltin = 18;
constexpr uint8_t kExpressionPushConstant = 19;
constexpr uint8_t kExpressionMatrixColumn = 20;
constexpr uint8_t kExpressionSharedMemoryElement = 21;
constexpr uint8_t kExpressionAtomic = 22;
constexpr uint8_t kExpressionTextureSample = 23;

constexpr uint8_t kLValueLocal = 1;
constexpr uint8_t kLValueParameter = 2;
constexpr uint8_t kLValueField = 3;
constexpr uint8_t kLValueResourceElement = 4;
constexpr uint8_t kLValueSwizzle = 5;
constexpr uint8_t kLValueMemberAccess = 6;
constexpr uint8_t kLValueIndexAccess = 7;
constexpr uint8_t kLValueMatrixColumn = 8;
constexpr uint8_t kLValueSharedMemoryElement = 9;

class Lowerer {
  public:
    Lowerer(const TypedIR::Module& module, const TypedIR::LoweringInputs& inputs,
            xir::Module& xir_module, std::vector<BufferLayout>* buffer_layouts, std::string* error)
        : module_{module}, inputs_{inputs}, xir_module_{xir_module}, buffer_layouts_{buffer_layouts}, error_{error} {}

    KernelFunction* lower() {
        if (module_.entry_function >= module_.functions.size())
            return fail("FEIR entry function is missing"), nullptr;
        const auto& entry = module_.functions[module_.entry_function];
        if (entry.kind > 2 || entry.body_statement_index >= module_.statements.size())
            return fail("Luisa forward backend requires a compute FEIR entry"), nullptr;
        if (inputs_.group_x <= 0 || inputs_.group_y <= 0 || inputs_.group_z <= 0)
            return fail("FEIR thread-group dimensions must be positive"), nullptr;
        const auto group_threads = static_cast<uint64_t>(inputs_.group_x) *
                                   static_cast<uint64_t>(inputs_.group_y) *
                                   static_cast<uint64_t>(inputs_.group_z);
        if (group_threads > 1024u)
            return fail("Luisa XIR supports at most 1024 threads per group"), nullptr;

        kernel_ = xir_module_.create_kernel();
        kernel_->set_name(std::string{string(entry.name_id)});
        const auto block_x = group_threads < 32u
                                 ? static_cast<uint32_t>(inputs_.group_x) * static_cast<uint32_t>(32u / group_threads)
                                 : static_cast<uint32_t>(inputs_.group_x);
        kernel_->set_block_size(luisa::make_uint3(block_x, static_cast<uint32_t>(inputs_.group_y),
                                                  static_cast<uint32_t>(inputs_.group_z)));

        if (!register_resources()) return nullptr;
        builder_.set_insertion_point(kernel_->create_body_block());
        if (!emit_bounds_guard(entry.kind) || !lower_statement(entry.body_statement_index)) return nullptr;
        if (!builder_.is_insertion_point_terminator()) builder_.return_void();
        return kernel_;
    }

  private:
    struct Resource {
        ResourceArgument* argument = nullptr;
        const Type* element_type = nullptr;
        uint32_t binding = 0;
        uint8_t kind = 0;
        uint8_t access = 0;
    };

    struct Address {
        Value* pointer = nullptr;
        Resource* resource = nullptr;
        Value* resource_index = nullptr;
        const Type* root_type = nullptr;
        std::vector<Value*> indices;
        explicit operator bool() const noexcept { return pointer != nullptr || resource != nullptr; }
    };

    struct LoopTargets {
        BasicBlock* break_target = nullptr;
        BasicBlock* continue_target = nullptr;
    };

    bool fail(std::string message) {
        if (error_ != nullptr && error_->empty()) *error_ = std::move(message);
        return false;
    }

    std::string_view string(uint32_t id) const {
        return id < module_.strings.size() ? std::string_view{module_.strings[id]} : std::string_view{};
    }

    Constant* index_constant(uint32_t value) { return xir_module_.create_constant(Type::of<uint32_t>(), &value); }

    const Type* type(uint32_t id) {
        if (id >= module_.types.size()) return nullptr;
        if (const auto found = types_.find(id); found != types_.end()) return found->second;
        const auto& source = module_.types[id];
        const Type* result = nullptr;
        switch (source.kind) {
        case kTypePrimitive:
            if (source.b != 32u) break;
            switch (source.a) {
            case 0: result = Type::of<bool>(); break;
            case 1: result = Type::of<int32_t>(); break;
            case 2: result = Type::of<uint32_t>(); break;
            case 3: result = Type::of<float>(); break;
            default: break;
            }
            break;
        case kTypeVector:
            if (auto* element = type(source.a); element != nullptr && source.b >= 2u && source.b <= 4u)
                result = Type::vector(element, source.b);
            break;
        case kTypeMatrix:
            if (source.b == source.c && source.b >= 2u && source.b <= 4u && type(source.a) == Type::of<float>())
                result = Type::matrix(source.b);
            break;
        case kTypeArray:
            if (auto* element = type(source.a); element != nullptr && source.b != TypedIR::NoIndex && source.b > 0u)
                result = Type::array(element, source.b);
            break;
        case kTypeStruct:
            result = struct_type(source.a);
            break;
        case kTypeResourceWrapper:
            result = type(source.b);
            break;
        case kTypeVoid:
            result = nullptr;
            break;
        default:
            break;
        }
        if (result != nullptr) types_.emplace(id, result);
        return result;
    }

    const Type* struct_type(uint32_t id) {
        if (id >= module_.structs.size()) return nullptr;
        if (const auto found = struct_types_.find(id); found != struct_types_.end()) return found->second;
        if (struct_visiting_[id]) return nullptr;
        struct_visiting_[id] = true;
        const auto& structure = module_.structs[id];
        std::vector<const Type*> members;
        members.reserve(structure.field_count);
        for (uint32_t i = 0; i < structure.field_count; ++i) {
            const auto& field = module_.struct_fields[structure.first_field + i];
            auto* member = type(field.type_id);
            if (member == nullptr) {
                struct_visiting_[id] = false;
                return nullptr;
            }
            members.push_back(member);
        }
        auto* result = Type::structure(structure.alignment,
                                       luisa::span<const Type* const>{members.data(), members.size()});
        struct_visiting_[id] = false;
        struct_types_.emplace(id, result);
        return result;
    }

    const Type* type_from_name(std::string_view name) {
        auto matches = [&](std::string_view simple) {
            return name == simple || name.ends_with(std::string{"."} + std::string{simple}) ||
                   name.ends_with(std::string{"::"} + std::string{simple});
        };
        if (name == "bool" || name == "System.Boolean") return Type::of<bool>();
        if (name == "int" || name == "System.Int32") return Type::of<int32_t>();
        if (name == "uint" || name == "System.UInt32") return Type::of<uint32_t>();
        if (name == "float" || name == "System.Single") return Type::of<float>();
        for (auto n = 2u; n <= 4u; ++n) {
            if (matches("bool" + std::to_string(n))) return Type::vector(Type::of<bool>(), n);
            if (matches("int" + std::to_string(n))) return Type::vector(Type::of<int32_t>(), n);
            if (matches("uint" + std::to_string(n))) return Type::vector(Type::of<uint32_t>(), n);
            if (matches("float" + std::to_string(n))) return Type::vector(Type::of<float>(), n);
            if (matches("float" + std::to_string(n) + "x" + std::to_string(n))) return Type::matrix(n);
        }
        for (uint32_t i = 0; i < module_.structs.size(); ++i) {
            const auto& structure = module_.structs[i];
            auto simple = string(structure.name_id);
            auto qualified = string(structure.fully_qualified_name_id);
            if (name == simple || name == qualified ||
                (qualified.starts_with("global::") && name == qualified.substr(8u)) ||
                (!simple.empty() && name.ends_with(std::string{"."} + std::string{simple}))) return struct_type(i);
        }
        return nullptr;
    }

    std::optional<uint32_t> field_index(uint32_t owner_type, std::string_view name) const {
        if (owner_type >= module_.types.size()) return std::nullopt;
        const auto& source = module_.types[owner_type];
        if (source.kind == kTypeMatrix && name.size() == 2u && (name[0] == 'C' || name[0] == 'c') &&
            name[1] >= '0' && name[1] <= '3') return static_cast<uint32_t>(name[1] - '0');
        if (source.kind != kTypeStruct || source.a >= module_.structs.size()) return std::nullopt;
        const auto& structure = module_.structs[source.a];
        for (uint32_t i = 0; i < structure.field_count; ++i)
            if (string(module_.struct_fields[structure.first_field + i].name_id) == name) return i;
        return std::nullopt;
    }

    bool register_resources() {
        struct_visiting_.resize(module_.structs.size());
        for (const auto& source : inputs_.resources) {
            if (source.kind == kResourcePushConstant) continue;
            if (source.kind != kResourceBuffer)
                return fail("Luisa XIR buffer translator received a non-buffer resource");
            auto* element = type_from_name(source.element_type);
            if (element == nullptr) return fail("Luisa cannot resolve FEIR buffer element type '" + source.element_type + "'");
            auto* argument = kernel_->create_resource_argument(Type::buffer(element));
            resources_.emplace(source.name, Resource{argument, element, source.binding, source.kind, source.access});
            if (buffer_layouts_ != nullptr) {
                auto source_type = TypedIR::NoIndex;
                for (uint32_t i = 0; i < module_.types.size(); ++i) {
                    if (type(i) == element) { source_type = i; break; }
                }
                if (source_type == TypedIR::NoIndex)
                    return fail("Luisa cannot match a buffer element to its FEIR type record");
                buffer_layouts_->push_back(BufferLayout{source.binding, source_type, element});
            }
        }
        return true;
    }

    bool argument_range(const TypedIR::Expression& expression) const {
        return expression.argument_count == 0u
                   ? expression.first_argument == TypedIR::NoIndex
                   : expression.first_argument != TypedIR::NoIndex && expression.first_argument <= module_.arguments.size() &&
                         expression.argument_count <= module_.arguments.size() - expression.first_argument;
    }

    std::vector<Value*> arguments(const TypedIR::Expression& expression) {
        std::vector<Value*> values;
        if (!argument_range(expression)) return fail("FEIR expression has an invalid argument range"), values;
        values.reserve(expression.argument_count);
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            auto* value = lower_expression(module_.arguments[expression.first_argument + i]);
            if (value == nullptr) return {};
            values.push_back(value);
        }
        return values;
    }

    Value* literal(const TypedIR::Expression& expression, const Type* result_type) {
        auto text = string(expression.name_id);
        if (result_type->is_bool()) {
            if (text == "true") return xir_module_.create_constant_one(result_type);
            if (text == "false") return xir_module_.create_constant_zero(result_type);
        } else if (result_type->is_float()) {
            std::string copy{text};
            char* end = nullptr;
            const auto value = std::strtof(copy.c_str(), &end);
            if (end != copy.c_str()) return xir_module_.create_constant(result_type, &value);
        } else if (result_type->is_int32()) {
            int32_t value{};
            auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec == std::errc{} && end == text.data() + text.size()) return xir_module_.create_constant(result_type, &value);
        } else if (result_type->is_uint32()) {
            if (!text.empty() && (text.back() == 'u' || text.back() == 'U')) text.remove_suffix(1u);
            uint32_t value{};
            auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            if (ec == std::errc{} && end == text.data() + text.size()) return xir_module_.create_constant(result_type, &value);
        }
        return fail("invalid FEIR literal '" + std::string{text} + "'"), nullptr;
    }

    static std::optional<ArithmeticOp> binary_op(uint32_t op) {
        constexpr std::array ops{ArithmeticOp::BINARY_ADD, ArithmeticOp::BINARY_SUB, ArithmeticOp::BINARY_MUL,
                                 ArithmeticOp::BINARY_DIV, ArithmeticOp::BINARY_MOD, ArithmeticOp::BINARY_BIT_AND,
                                 ArithmeticOp::BINARY_BIT_OR, ArithmeticOp::BINARY_BIT_XOR,
                                 ArithmeticOp::BINARY_SHIFT_LEFT, ArithmeticOp::BINARY_SHIFT_RIGHT};
        return op < ops.size() ? std::optional{ops[op]} : std::nullopt;
    }

    static std::optional<ArithmeticOp> compare_op(uint32_t op) {
        constexpr std::array ops{ArithmeticOp::BINARY_EQUAL, ArithmeticOp::BINARY_NOT_EQUAL,
                                 ArithmeticOp::BINARY_LESS, ArithmeticOp::BINARY_LESS_EQUAL,
                                 ArithmeticOp::BINARY_GREATER, ArithmeticOp::BINARY_GREATER_EQUAL};
        return op < ops.size() ? std::optional{ops[op]} : std::nullopt;
    }

    Value* extract(Value* base, const Type* result_type, std::vector<Value*> indices) {
        indices.insert(indices.begin(), base);
        return builder_.call(result_type, ArithmeticOp::EXTRACT, indices);
    }

    Value* matrix_multiply(Value* left, Value* right, const Type* result_type) {
        auto scalar = [](const Type* value) { return value->is_matrix() ? value->element() : value->element(); };
        const auto* element = scalar(result_type);
        auto dot_row = [&](Value* matrix, Value* vector, uint32_t row) -> Value* {
            Value* sum = nullptr;
            for (uint32_t column = 0; column < matrix->type()->dimension(); ++column) {
                auto* matrix_value = extract(matrix, element, {index_constant(column), index_constant(row)});
                auto* vector_value = extract(vector, element, {index_constant(column)});
                auto* product = builder_.call(element, ArithmeticOp::BINARY_MUL, {matrix_value, vector_value});
                sum = sum == nullptr ? product : builder_.call(element, ArithmeticOp::BINARY_ADD, {sum, product});
            }
            return sum;
        };
        auto multiply_matrix_vector = [&](Value* matrix, Value* vector) -> Value* {
            std::vector<Value*> rows;
            rows.reserve(matrix->type()->dimension());
            for (uint32_t row = 0; row < matrix->type()->dimension(); ++row)
                rows.push_back(dot_row(matrix, vector, row));
            return builder_.call(Type::vector(element, matrix->type()->dimension()), ArithmeticOp::AGGREGATE, rows);
        };
        if (left->type()->is_matrix() && right->type()->is_vector())
            return multiply_matrix_vector(left, right);
        if (left->type()->is_matrix() && right->type()->is_matrix()) {
            std::vector<Value*> columns;
            columns.reserve(right->type()->dimension());
            auto* column_type = Type::vector(element, right->type()->dimension());
            for (uint32_t column = 0; column < right->type()->dimension(); ++column) {
                auto* right_column = extract(right, column_type, {index_constant(column)});
                columns.push_back(multiply_matrix_vector(left, right_column));
            }
            return builder_.call(result_type, ArithmeticOp::AGGREGATE, columns);
        }
        if (left->type()->is_vector() && right->type()->is_matrix()) {
            std::vector<Value*> values;
            values.reserve(right->type()->dimension());
            auto* column_type = Type::vector(element, right->type()->dimension());
            for (uint32_t column = 0; column < right->type()->dimension(); ++column) {
                auto* right_column = extract(right, column_type, {index_constant(column)});
                Value* sum = nullptr;
                for (uint32_t row = 0; row < right->type()->dimension(); ++row) {
                    auto* a = extract(left, element, {index_constant(row)});
                    auto* b = extract(right_column, element, {index_constant(row)});
                    auto* product = builder_.call(element, ArithmeticOp::BINARY_MUL, {a, b});
                    sum = sum == nullptr ? product : builder_.call(element, ArithmeticOp::BINARY_ADD, {sum, product});
                }
                values.push_back(sum);
            }
            return builder_.call(result_type, ArithmeticOp::AGGREGATE, values);
        }
        return fail("invalid FEIR matrix multiplication operands"), nullptr;
    }

    Value* lower_expression(uint32_t id) {
        if (id >= module_.expressions.size()) return fail("FEIR expression index is out of range"), nullptr;
        const auto& expression = module_.expressions[id];
        auto* result_type = type(expression.type_id);
        if (result_type == nullptr) return fail("FEIR expression has an unsupported XIR type"), nullptr;
        switch (expression.kind) {
        case kExpressionLiteral:
            return literal(expression, result_type);
        case kExpressionLocal:
        case kExpressionParameter: {
            const auto found = locals_.find(std::string{string(expression.name_id)});
            return found == locals_.end() ? (fail("FEIR references an unknown local"), nullptr)
                                          : builder_.load(result_type, found->second);
        }
        case kExpressionResourceElement: {
            const auto found = resources_.find(std::string{string(expression.name_id)});
            auto* index = lower_expression(expression.a);
            if (found == resources_.end() || index == nullptr) return fail("FEIR buffer read is invalid"), nullptr;
            index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
            return builder_.call(result_type, ResourceReadOp::BUFFER_READ, {found->second.argument, index});
        }
        case kExpressionUnary: {
            auto* value = lower_expression(expression.a);
            if (value == nullptr) return nullptr;
            if (expression.op == 0u) return builder_.call(result_type, ArithmeticOp::UNARY_MINUS, {value});
            if (expression.op == 2u) return builder_.call(result_type, ArithmeticOp::UNARY_BIT_NOT, {value});
            if (expression.op == 1u)
                return builder_.call(result_type, ArithmeticOp::BINARY_EQUAL,
                                     {value, xir_module_.create_constant_zero(result_type)});
            return fail("unsupported FEIR unary operation"), nullptr;
        }
        case kExpressionBinary: {
            auto* left = lower_expression(expression.a);
            auto* right = lower_expression(expression.b);
            auto op = binary_op(expression.op);
            if (left == nullptr || right == nullptr || !op) return fail("invalid FEIR binary operation"), nullptr;
            if (expression.op == 2u && (left->type()->is_matrix() || right->type()->is_matrix()))
                return matrix_multiply(left, right, result_type);
            return builder_.call(result_type, *op, {left, right});
        }
        case kExpressionComparison: {
            auto* left = lower_expression(expression.a);
            auto* right = lower_expression(expression.b);
            auto op = compare_op(expression.op);
            return left == nullptr || right == nullptr || !op ? (fail("invalid FEIR comparison"), nullptr)
                                                               : builder_.call(result_type, *op, {left, right});
        }
        case kExpressionLogical: {
            auto* left = lower_expression(expression.a);
            auto* right = lower_expression(expression.b);
            if (left == nullptr || right == nullptr || expression.op > 1u) return fail("invalid FEIR logical operation"), nullptr;
            return builder_.call(result_type, expression.op == 0u ? ArithmeticOp::BINARY_BIT_AND : ArithmeticOp::BINARY_BIT_OR,
                                 {left, right});
        }
        case kExpressionConditional: {
            auto* condition = lower_expression(expression.a);
            auto* when_true = lower_expression(expression.b);
            auto* when_false = lower_expression(expression.c);
            return condition == nullptr || when_true == nullptr || when_false == nullptr
                       ? nullptr
                       : builder_.call(result_type, ArithmeticOp::SELECT, {when_false, when_true, condition});
        }
        case kExpressionConversion: {
            auto* value = lower_expression(expression.a);
            return value == nullptr ? nullptr : builder_.static_cast_if_necessary(result_type, value);
        }
        case kExpressionConstructor: {
            auto values = arguments(expression);
            if (values.empty()) return fail("FEIR aggregate constructor requires arguments"), nullptr;
            if (values.size() == 1u && values.front()->type()->is_scalar() && result_type->is_vector()) {
                values.resize(result_type->dimension(), values.front());
            } else if (result_type->is_vector()) {
                std::vector<Value*> flattened;
                for (auto* value : values) {
                    if (value->type()->is_scalar()) flattened.push_back(value);
                    else for (uint32_t i = 0; i < value->type()->dimension(); ++i)
                        flattened.push_back(extract(value, value->type()->element(), {index_constant(i)}));
                }
                values = std::move(flattened);
            }
            return builder_.call(result_type, ArithmeticOp::AGGREGATE, values);
        }
        case kExpressionField:
        case kExpressionMemberAccess: {
            auto* base = lower_expression(expression.a);
            if (base == nullptr || expression.a >= module_.expressions.size()) return nullptr;
            auto field = field_index(module_.expressions[expression.a].type_id, string(expression.name_id));
            return field ? extract(base, result_type, {index_constant(*field)})
                         : (fail("FEIR member name is not present in its aggregate type"), nullptr);
        }
        case kExpressionIndexAccess: {
            auto* base = lower_expression(expression.a);
            auto* index = lower_expression(expression.b);
            return base == nullptr || index == nullptr ? nullptr : extract(base, result_type, {index});
        }
        case kExpressionMatrixColumn: {
            auto* base = lower_expression(expression.a);
            auto* index = lower_expression(expression.b);
            return base == nullptr || index == nullptr ? nullptr : extract(base, result_type, {index});
        }
        case kExpressionSwizzle: {
            auto* base = lower_expression(expression.a);
            if (base == nullptr) return nullptr;
            std::vector<Value*> operands{base};
            for (char c : string(expression.name_id)) {
                c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                auto position = std::string_view{"xyzw"}.find(c);
                if (position == std::string_view::npos) position = std::string_view{"rgba"}.find(c);
                if (position == std::string_view::npos) position = std::string_view{"stpq"}.find(c);
                if (position == std::string_view::npos) return fail("invalid FEIR swizzle"), nullptr;
                operands.push_back(index_constant(static_cast<uint32_t>(position)));
            }
            return operands.size() == 2u ? extract(base, result_type, {operands[1]})
                                         : builder_.call(result_type, ArithmeticOp::SHUFFLE, operands);
        }
        case kExpressionBuiltin:
            return lower_builtin(expression.op, result_type);
        case kExpressionPushConstant:
            return lower_push_constant(expression, result_type);
        case kExpressionSharedMemoryElement: {
            auto found = shared_.find(std::string{string(expression.name_id)});
            auto* index = lower_expression(expression.a);
            if (found == shared_.end() || index == nullptr) return fail("invalid FEIR shared-memory read"), nullptr;
            auto* pointer = builder_.gep(result_type, found->second, {index});
            return builder_.load(result_type, pointer);
        }
        case kExpressionIntrinsic:
            return lower_intrinsic(expression, result_type);
        case kExpressionCallableCall:
            return fail("FEIR callable translation is staged for the callable batch"), nullptr;
        case kExpressionAtomic:
            return lower_atomic(expression, result_type);
        case kExpressionTextureSample:
            return fail("FEIR texture sampling is staged for the texture batch"), nullptr;
        default:
            return fail("unsupported FEIR expression kind " + std::to_string(expression.kind)), nullptr;
        }
    }

    Value* lower_builtin(uint32_t builtin, const Type* result_type) {
        Value* vector = nullptr;
        uint32_t component = 0;
        if (builtin >= 1u && builtin <= 3u) { vector = xir_module_.create_dispatch_id(); component = builtin - 1u; }
        else if (builtin >= 4u && builtin <= 6u) { vector = xir_module_.create_thread_id(); component = builtin - 4u; }
        else if (builtin >= 7u && builtin <= 9u) { vector = xir_module_.create_block_id(); component = builtin - 7u; }
        else if (builtin >= 10u && builtin <= 12u) { vector = xir_module_.create_dispatch_size(); component = builtin - 10u; }
        else if (builtin >= 13u && builtin <= 15u) {
            const uint32_t sizes[]{static_cast<uint32_t>(inputs_.group_x), static_cast<uint32_t>(inputs_.group_y),
                                   static_cast<uint32_t>(inputs_.group_z)};
            return builder_.static_cast_if_necessary(result_type,
                                                      xir_module_.create_constant(Type::of<uint32_t>(), &sizes[builtin - 13u]));
        }
        if (vector == nullptr) return fail("unsupported compute builtin"), nullptr;
        auto* value = extract(vector, Type::of<uint32_t>(), {index_constant(component)});
        return builder_.static_cast_if_necessary(result_type, value);
    }

    Value* lower_push_constant(const TypedIR::Expression& expression, const Type* result_type) {
        const TypedIR::PushConstantInfo* found = nullptr;
        for (const auto& push : inputs_.push_constants) if (push.binding == expression.op) found = &push;
        if (found == nullptr || found->data == nullptr || found->size < result_type->size())
            return fail("FEIR push constant is missing or has the wrong layout"), nullptr;
        return xir_module_.create_constant(result_type, found->data);
    }

    static std::optional<ArithmeticOp> intrinsic_op(std::string_view name) {
        struct Entry { std::string_view suffix; ArithmeticOp op; };
        static constexpr Entry entries[]{
            {".Sin", ArithmeticOp::SIN}, {".Cos", ArithmeticOp::COS}, {".Tan", ArithmeticOp::TAN},
            {".Exp", ArithmeticOp::EXP}, {".Log", ArithmeticOp::LOG}, {".Sqrt", ArithmeticOp::SQRT},
            {".InverseSqrt", ArithmeticOp::RSQRT}, {".Length", ArithmeticOp::LENGTH},
            {".Normalize", ArithmeticOp::NORMALIZE}, {".Abs", ArithmeticOp::ABS},
            {".Floor", ArithmeticOp::FLOOR}, {".Ceil", ArithmeticOp::CEIL}, {".Round", ArithmeticOp::ROUND},
            {".Fract", ArithmeticOp::FRACT}, {".Pow", ArithmeticOp::POW}, {".Min", ArithmeticOp::MIN},
            {".Max", ArithmeticOp::MAX}, {".Clamp", ArithmeticOp::CLAMP}, {".Lerp", ArithmeticOp::LERP},
            {".Mix", ArithmeticOp::LERP}, {".Smoothstep", ArithmeticOp::SMOOTHSTEP}, {".Dot", ArithmeticOp::DOT},
            {".Cross", ArithmeticOp::CROSS}, {".Reflect", ArithmeticOp::REFLECT},
            {".Transpose", ArithmeticOp::MATRIX_TRANSPOSE}, {".Determinant", ArithmeticOp::MATRIX_DETERMINANT},
            {".Inverse", ArithmeticOp::MATRIX_INVERSE}, {".Hadamard", ArithmeticOp::MATRIX_COMP_MUL}};
        for (auto entry : entries) if (name.ends_with(entry.suffix)) return entry.op;
        if (name.ends_with(".Mul")) return ArithmeticOp::MATRIX_LINALG_MUL;
        if (name.ends_with(".Saturate")) return ArithmeticOp::SATURATE;
        return std::nullopt;
    }

    Value* lower_intrinsic(const TypedIR::Expression& expression, const Type* result_type) {
        auto op = intrinsic_op(string(expression.name_id));
        auto values = arguments(expression);
        if (!op || values.empty()) return fail("unsupported FEIR intrinsic '" + std::string{string(expression.name_id)} + "'"), nullptr;
        if ((*op == ArithmeticOp::MIN || *op == ArithmeticOp::MAX || *op == ArithmeticOp::CLAMP) && result_type->is_vector()) {
            for (size_t i = 1; i < values.size(); ++i)
                if (values[i]->type()->is_scalar()) {
                    std::vector<Value*> splat(result_type->dimension(), values[i]);
                    values[i] = builder_.call(result_type, ArithmeticOp::AGGREGATE, splat);
                }
        }
        if (*op == ArithmeticOp::MATRIX_LINALG_MUL && values.size() == 2u)
            return matrix_multiply(values[0], values[1], result_type);
        return builder_.call(result_type, *op, values);
    }

    std::optional<Address> address(uint32_t id) {
        if (id >= module_.lvalues.size()) return fail("FEIR l-value index is out of range"), std::nullopt;
        const auto& lvalue = module_.lvalues[id];
        if (lvalue.kind == kLValueLocal || lvalue.kind == kLValueParameter) {
            auto found = locals_.find(std::string{string(lvalue.name_id)});
            if (found == locals_.end()) return fail("FEIR l-value names an unknown local"), std::nullopt;
            return Address{.pointer = found->second, .root_type = found->second->type()};
        }
        if (lvalue.kind == kLValueResourceElement) {
            auto found = resources_.find(std::string{string(lvalue.name_id)});
            auto* index = lower_expression(lvalue.a);
            if (found == resources_.end() || index == nullptr) return fail("invalid FEIR buffer l-value"), std::nullopt;
            return Address{.resource = &found->second, .resource_index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index),
                           .root_type = found->second.element_type};
        }
        if (lvalue.kind == kLValueSharedMemoryElement) {
            auto found = shared_.find(std::string{string(lvalue.name_id)});
            auto* index = lower_expression(lvalue.a);
            if (found == shared_.end() || index == nullptr) return fail("invalid FEIR shared-memory l-value"), std::nullopt;
            auto* result_type = type(lvalue.type_id);
            return Address{.pointer = builder_.gep(result_type, found->second, {index}), .root_type = result_type};
        }
        if (lvalue.kind == kLValueField || lvalue.kind == kLValueMemberAccess || lvalue.kind == kLValueIndexAccess) {
            auto base = address(lvalue.a);
            if (!base) return std::nullopt;
            if (lvalue.kind == kLValueIndexAccess) {
                auto* index = lower_expression(lvalue.b);
                if (index == nullptr) return std::nullopt;
                base->indices.push_back(index);
            } else {
                auto field = field_index(module_.lvalues[lvalue.a].type_id, string(lvalue.name_id));
                if (!field) return fail("invalid FEIR aggregate l-value member"), std::nullopt;
                base->indices.push_back(index_constant(*field));
            }
            return base;
        }
        return fail("unsupported FEIR l-value kind " + std::to_string(lvalue.kind)), std::nullopt;
    }

    Value* read_address(const Address& address, const Type* result_type) {
        if (address.pointer != nullptr) {
            auto* pointer = address.indices.empty() ? address.pointer
                                                    : builder_.gep(result_type, address.pointer, address.indices);
            return builder_.load(result_type, pointer);
        }
        auto* root = builder_.call(address.root_type, ResourceReadOp::BUFFER_READ,
                                   {address.resource->argument, address.resource_index});
        return address.indices.empty() ? root : extract(root, result_type, address.indices);
    }

    bool write_address(const Address& address, const Type* value_type, Value* value) {
        if (address.pointer != nullptr) {
            auto* pointer = address.indices.empty() ? address.pointer
                                                    : builder_.gep(value_type, address.pointer, address.indices);
            builder_.store(pointer, value);
            return true;
        }
        if (address.resource->access != kAccessWrite && address.resource->access != kAccessReadWrite)
            return fail("FEIR writes a read-only buffer");
        if (!address.indices.empty()) {
            auto* root = builder_.call(address.root_type, ResourceReadOp::BUFFER_READ,
                                       {address.resource->argument, address.resource_index});
            std::vector<Value*> operands{root, value};
            operands.insert(operands.end(), address.indices.begin(), address.indices.end());
            value = builder_.call(address.root_type, ArithmeticOp::INSERT, operands);
        }
        builder_.call(ResourceWriteOp::BUFFER_WRITE, {address.resource->argument, address.resource_index, value});
        return true;
    }

    Value* lower_atomic(const TypedIR::Expression& expression, const Type* result_type) {
        auto target = address(expression.a);
        auto values = arguments(expression);
        if (!target || values.empty() || (expression.op == 8u && values.size() != 2u))
            return fail("invalid FEIR atomic expression"), nullptr;
        constexpr std::array ops{AtomicOp::FETCH_ADD, AtomicOp::FETCH_SUB, AtomicOp::FETCH_MIN, AtomicOp::FETCH_MAX,
                                 AtomicOp::FETCH_AND, AtomicOp::FETCH_OR, AtomicOp::FETCH_XOR, AtomicOp::EXCHANGE,
                                 AtomicOp::COMPARE_EXCHANGE};
        if (expression.op >= ops.size()) return fail("unsupported FEIR atomic operation"), nullptr;
        if (target->resource != nullptr) {
            std::vector<Value*> indices{target->resource_index};
            indices.insert(indices.end(), target->indices.begin(), target->indices.end());
            return builder_.call(result_type, ops[expression.op], target->resource->argument, indices, values);
        }
        auto* pointer = target->indices.empty() ? target->pointer : builder_.gep(result_type, target->pointer, target->indices);
        return builder_.call(result_type, ops[expression.op], pointer, {}, values);
    }

    bool emit_bounds_guard(uint8_t dimensions) {
        if (!inputs_.bounds_check) return true;
        Value* outside = nullptr;
        const uint32_t logical[]{static_cast<uint32_t>(inputs_.logical_x), static_cast<uint32_t>(inputs_.logical_y),
                                 static_cast<uint32_t>(inputs_.logical_z)};
        for (uint32_t axis = 0; axis <= dimensions; ++axis) {
            auto* id = extract(xir_module_.create_dispatch_id(), Type::of<uint32_t>(), {index_constant(axis)});
            auto* limit = xir_module_.create_constant(Type::of<uint32_t>(), &logical[axis]);
            auto* current = builder_.call(Type::of<bool>(), ArithmeticOp::BINARY_GREATER_EQUAL, {id, limit});
            outside = outside == nullptr ? current
                                         : builder_.call(Type::of<bool>(), ArithmeticOp::BINARY_BIT_OR, {outside, current});
        }
        auto* branch = builder_.if_(outside);
        auto* merge = branch->create_merge_block();
        builder_.set_insertion_point(branch->create_true_block());
        builder_.return_void();
        builder_.set_insertion_point(branch->create_false_block());
        builder_.br(merge);
        builder_.set_insertion_point(merge);
        return true;
    }

    bool lower_statement(uint32_t id) {
        if (id >= module_.statements.size()) return fail("FEIR statement index is out of range");
        const auto& statement = module_.statements[id];
        switch (statement.kind) {
        case kStatementBlock:
            if (statement.child_count == 0u) return true;
            if (statement.first_child == TypedIR::NoIndex || statement.first_child > module_.children.size() ||
                statement.child_count > module_.children.size() - statement.first_child) return fail("invalid FEIR block range");
            for (uint32_t i = 0; i < statement.child_count; ++i) {
                if (builder_.is_insertion_point_terminator()) break;
                if (!lower_statement(module_.children[statement.first_child + i])) return false;
            }
            return true;
        case kStatementLocalDeclaration: {
            auto name = std::string{string(statement.name_id)};
            auto* local_type = type(statement.op);
            if (name.empty() || local_type == nullptr) return fail("invalid FEIR local declaration");
            auto* local = builder_.alloca_local(local_type);
            local->set_name(name);
            locals_[name] = local;
            if (statement.a != TypedIR::NoIndex) {
                auto* initial = lower_expression(statement.a);
                if (initial == nullptr) return false;
                builder_.store(local, initial);
            }
            return true;
        }
        case kStatementSharedMemoryDeclaration: {
            auto name = std::string{string(statement.name_id)};
            auto* element = type(statement.op);
            if (name.empty() || element == nullptr || statement.a == 0u) return fail("invalid FEIR shared declaration");
            auto* memory = builder_.alloca_shared(Type::array(element, statement.a));
            memory->set_name(name);
            shared_[name] = memory;
            uses_group_semantics_ = true;
            return validate_exact_block_size();
        }
        case kStatementAssignment:
        case kStatementCompoundAssignment:
        case kStatementIncrementDecrement: {
            auto target = address(statement.a);
            if (!target) return false;
            auto* value_type = type(module_.lvalues[statement.a].type_id);
            if (value_type == nullptr) return fail("invalid FEIR assignment type");
            Value* value = nullptr;
            if (statement.kind == kStatementAssignment) value = lower_expression(statement.b);
            else {
                auto* old = read_address(*target, value_type);
                Value* right = nullptr;
                auto op = binary_op(statement.op);
                if (statement.kind == kStatementIncrementDecrement) {
                    right = xir_module_.create_constant_one(value_type);
                    op = (statement.op & 1u) != 0u ? ArithmeticOp::BINARY_ADD : ArithmeticOp::BINARY_SUB;
                } else right = lower_expression(statement.b);
                if (old == nullptr || right == nullptr || !op) return fail("invalid FEIR compound assignment");
                value = builder_.call(value_type, *op, {old, right});
            }
            return value != nullptr && write_address(*target, value_type, value);
        }
        case kStatementIf:
            return lower_if(statement);
        case kStatementFor:
            if (statement.a != TypedIR::NoIndex && !lower_statement(statement.a)) return false;
            return lower_loop(statement.op, statement.b, statement.c, false);
        case kStatementWhile:
            return lower_loop(statement.b, statement.a, TypedIR::NoIndex, false);
        case kStatementDoWhile:
            return lower_loop(statement.a, statement.b, TypedIR::NoIndex, true);
        case kStatementBreak:
            if (loops_.empty()) return fail("FEIR break appears outside a loop");
            builder_.break_(loops_.back().break_target);
            return true;
        case kStatementContinue:
            if (loops_.empty()) return fail("FEIR continue appears outside a loop");
            builder_.continue_(loops_.back().continue_target);
            return true;
        case kStatementReturn:
            if (statement.a == TypedIR::NoIndex) builder_.return_void();
            else {
                auto* value = lower_expression(statement.a);
                if (value == nullptr) return false;
                builder_.return_(value);
            }
            return true;
        case kStatementExpression:
            return lower_expression(statement.a) != nullptr;
        case kStatementBarrier:
            uses_group_semantics_ = true;
            if (!validate_exact_block_size()) return false;
            builder_.synchronize_block();
            return true;
        default:
            return fail("unsupported FEIR statement kind " + std::to_string(statement.kind));
        }
    }

    bool validate_exact_block_size() {
        const auto count = static_cast<uint64_t>(inputs_.group_x) * static_cast<uint64_t>(inputs_.group_y) *
                           static_cast<uint64_t>(inputs_.group_z);
        return count >= 32u && count % 32u == 0u
                   ? true
                   : fail("Luisa local/shared semantics require an exact FEIR group size that is a multiple of 32");
    }

    bool lower_if(const TypedIR::Statement& statement) {
        auto* condition = lower_expression(statement.a);
        if (condition == nullptr) return false;
        auto* branch = builder_.if_(builder_.static_cast_if_necessary(Type::of<bool>(), condition));
        auto* merge = branch->create_merge_block();
        builder_.set_insertion_point(branch->create_true_block());
        if (!lower_statement(statement.b)) return false;
        if (!builder_.is_insertion_point_terminator()) builder_.br(merge);
        builder_.set_insertion_point(branch->create_false_block());
        if (statement.c != TypedIR::NoIndex && !lower_statement(statement.c)) return false;
        if (!builder_.is_insertion_point_terminator()) builder_.br(merge);
        builder_.set_insertion_point(merge);
        return true;
    }

    bool lower_loop(uint32_t body_id, uint32_t condition_id, uint32_t update_id, bool do_first) {
        auto* loop = builder_.loop();
        auto* prepare = loop->create_prepare_block();
        auto* body = loop->create_body_block();
        auto* update = loop->create_update_block();
        auto* merge = loop->create_merge_block();
        builder_.set_insertion_point(prepare);
        if (do_first) builder_.br(body);
        else if (condition_id == TypedIR::NoIndex) builder_.br(body);
        else {
            auto* condition = lower_expression(condition_id);
            if (condition == nullptr) return false;
            builder_.cond_br(builder_.static_cast_if_necessary(Type::of<bool>(), condition), body, merge);
        }
        loops_.push_back({merge, update});
        builder_.set_insertion_point(body);
        if (!lower_statement(body_id)) return false;
        if (!builder_.is_insertion_point_terminator()) builder_.br(update);
        builder_.set_insertion_point(update);
        if (update_id != TypedIR::NoIndex && !lower_statement(update_id)) return false;
        if (!builder_.is_insertion_point_terminator()) {
            if (!do_first) builder_.br(prepare);
            else {
                auto* condition = lower_expression(condition_id);
                if (condition == nullptr) return false;
                builder_.cond_br(builder_.static_cast_if_necessary(Type::of<bool>(), condition), body, merge);
            }
        }
        loops_.pop_back();
        builder_.set_insertion_point(merge);
        return true;
    }

    const TypedIR::Module& module_;
    const TypedIR::LoweringInputs& inputs_;
    xir::Module& xir_module_;
    std::vector<BufferLayout>* buffer_layouts_ = nullptr;
    std::string* error_ = nullptr;
    KernelFunction* kernel_ = nullptr;
    XIRBuilder builder_;
    std::unordered_map<uint32_t, const Type*> types_;
    std::unordered_map<uint32_t, const Type*> struct_types_;
    std::vector<bool> struct_visiting_;
    std::unordered_map<std::string, Resource> resources_;
    std::unordered_map<std::string, Value*> locals_;
    std::unordered_map<std::string, Value*> shared_;
    std::vector<LoopTargets> loops_;
    bool uses_group_semantics_ = false;
};

} // namespace

KernelFunction* LowerToXir(const TypedIR::Module& module, const TypedIR::LoweringInputs& inputs,
                           xir::Module& xir_module, std::vector<BufferLayout>* buffer_layouts, std::string* error) {
    if (error != nullptr) error->clear();
    if (buffer_layouts != nullptr) buffer_layouts->clear();
    return Lowerer{module, inputs, xir_module, buffer_layouts, error}.lower();
}

} // namespace Feather::Luisa
