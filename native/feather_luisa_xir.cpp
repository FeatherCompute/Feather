#include "feather_luisa_xir.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <numeric>
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
constexpr uint8_t kResourceTexture2D = 2;
constexpr uint8_t kResourceSampler = 3;
constexpr uint8_t kResourcePushConstant = 5;
constexpr uint8_t kResourceTexture3D = 6;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessReadWrite = 3;

constexpr uint8_t kTypePrimitive = 1;
constexpr uint8_t kTypeVector = 2;
constexpr uint8_t kTypeMatrix = 3;
constexpr uint8_t kTypeStruct = 4;
constexpr uint8_t kTypeArray = 5;
constexpr uint8_t kTypeResourceWrapper = 6;
constexpr uint8_t kTypeVoid = 7;
constexpr uint32_t kTypeResourceBuffer = 0;
constexpr uint32_t kTypeResourceTexture2D = 1;
constexpr uint32_t kTypeResourceTexture3D = 2;
constexpr uint32_t kTypeResourceSampler = 3;

constexpr uint8_t kFunctionCallable = 5;
constexpr uint8_t kFunctionVertex = 3;
constexpr uint8_t kFunctionFragment = 4;

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
            xir::Module& xir_module, std::vector<BufferLayout>* buffer_layouts,
            const AdInputs* ad_inputs, std::vector<AdGradientLayout>* ad_gradient_layouts,
            std::string* error)
        : module_{module}, inputs_{inputs}, xir_module_{xir_module}, buffer_layouts_{buffer_layouts},
          ad_inputs_{ad_inputs}, ad_gradient_layouts_{ad_gradient_layouts}, error_{error} {}

    KernelFunction* lower() {
        if (module_.entry_function >= module_.functions.size())
            return fail("FEIR entry function is missing"), nullptr;
        const auto& entry = module_.functions[module_.entry_function];
        const auto graphics_stage = entry.kind == kFunctionVertex || entry.kind == kFunctionFragment;
        if ((!graphics_stage && entry.kind > 2u) ||
            (graphics_stage && inputs_.stage_output_binding == TypedIR::NoIndex) ||
            entry.body_statement_index >= module_.statements.size())
            return fail("Luisa forward backend requires a compute FEIR entry"), nullptr;
        if (inputs_.group_x <= 0 || inputs_.group_y <= 0 || inputs_.group_z <= 0)
            return fail("FEIR thread-group dimensions must be positive"), nullptr;
        const auto group_threads = static_cast<uint64_t>(inputs_.group_x) *
                                   static_cast<uint64_t>(inputs_.group_y) *
                                   static_cast<uint64_t>(inputs_.group_z);
        if (group_threads > 1024u)
            return fail("Luisa XIR supports at most 1024 threads per group"), nullptr;

        kernel_ = xir_module_.create_kernel();
        function_ = kernel_;
        kernel_->set_name(std::string{string(entry.name_id)});
        logical_groups_per_block_ = static_cast<uint32_t>(32u / std::gcd<uint64_t>(group_threads, 32u));
        if (group_threads * logical_groups_per_block_ > 1024u)
            return fail("FEIR thread group cannot be represented at Luisa's 32-thread granularity"), nullptr;
        const auto block_x = static_cast<uint32_t>(inputs_.group_x) * logical_groups_per_block_;
        kernel_->set_block_size(luisa::make_uint3(block_x, static_cast<uint32_t>(inputs_.group_y),
                                                  static_cast<uint32_t>(inputs_.group_z)));

        if (!register_resources() || !register_ad_resources() || !stage_callables() || !lower_callable_bodies()) return nullptr;
        if (graphics_stage && !register_graphics_stage(entry)) return nullptr;
        builder_.set_insertion_point(kernel_->create_body_block());
        if (!emit_bounds_guard(graphics_stage ? 0u : entry.kind)) return nullptr;
        if (graphics_stage && !bind_graphics_stage_parameters(entry)) return nullptr;
        BasicBlock* ad_merge = nullptr;
        if (ad_inputs_ != nullptr) {
            auto* scope = builder_.autodiff_scope();
            ad_merge = scope->create_merge_block();
            builder_.set_insertion_point(scope->create_entry_block());
            inside_ad_scope_ = true;
            if (!begin_ad_scope()) return nullptr;
        }
        if (!lower_statement(entry.body_statement_index)) return nullptr;
        if (ad_inputs_ != nullptr) {
            if (!finish_ad_scope()) return nullptr;
            if (!builder_.is_insertion_point_terminator()) builder_.br(ad_merge);
            builder_.set_insertion_point(ad_merge);
            inside_ad_scope_ = false;
        }
        if (!builder_.is_insertion_point_terminator()) builder_.return_void();
        return kernel_;
    }

    GraphicsFragmentXir lower_graphics_fragment() {
        if (module_.entry_function >= module_.functions.size()) {
            fail("FEIR fragment entry function is missing");
            return {};
        }
        const auto& entry = module_.functions[module_.entry_function];
        if (entry.kind != kFunctionFragment || entry.body_statement_index >= module_.statements.size()) {
            fail("fused Luisa raster requires a fragment FEIR entry");
            return {};
        }
        auto* return_type = type(entry.return_type_id);
        if (return_type == nullptr) {
            fail("fused fragment return type is unsupported");
            return {};
        }
        fragment_callable_ = xir_module_.create_callable(return_type);
        function_ = fragment_callable_;
        fragment_callable_->set_name(std::string{string(entry.name_id)} + "_fused_raster");
        if (!register_resources() || !stage_callables() || !lower_callable_bodies() ||
            !register_fused_fragment_stage(entry)) {
            return {};
        }
        builder_.set_insertion_point(fragment_callable_->create_body_block());
        if (!bind_fused_fragment_parameters(entry) || !lower_statement(entry.body_statement_index)) return {};
        if (!builder_.is_insertion_point_terminator()) builder_.return_void();
        return {fragment_callable_, fragment_parameter_type_, return_type};
    }

  private:
    struct Resource {
        ResourceArgument* argument = nullptr;
        const Type* element_type = nullptr;
        uint32_t binding = 0;
        uint32_t element_count = 0;
        uint8_t kind = 0;
        uint8_t access = 0;
    };

    struct Sampler {
        Value* filter = nullptr;
        Value* address = nullptr;
    };

    struct CallableParameter {
        Argument* argument = nullptr;
        Argument* sampler_address = nullptr;
        uint32_t type_id = TypedIR::NoIndex;
        uint8_t direction = 0;
    };

    struct CallableRecord {
        CallableFunction* function = nullptr;
        uint32_t function_id = TypedIR::NoIndex;
        std::vector<CallableParameter> parameters;
    };

    struct Address {
        Value* pointer = nullptr;
        Resource* resource = nullptr;
        Value* resource_index = nullptr;
        uint32_t resource_index_expression = TypedIR::NoIndex;
        const Type* root_type = nullptr;
        std::vector<Value*> indices;
        explicit operator bool() const noexcept { return pointer != nullptr || resource != nullptr; }
    };

    struct LoopTargets {
        BasicBlock* break_target = nullptr;
        BasicBlock* continue_target = nullptr;
    };

    struct StageColorOutput {
        Resource* resource = nullptr;
        uint32_t return_field = TypedIR::NoIndex;
        TypedIR::GraphicsBlendInfo blend;
    };

    struct AdResource {
        Resource* source = nullptr;
        ResourceArgument* gradient = nullptr;
        uint32_t element_count = 0;
        Value* parameter = nullptr;
    };

    struct AdIndexTerm {
        uint32_t expression = TypedIR::NoIndex;
        int sign = 1;
        std::string key;
    };

    struct AdIndex {
        std::vector<AdIndexTerm> terms;
        int64_t offset = 0;
    };

    struct AdLocalSlot {
        Value* pointer = nullptr;
        AdIndex index;
        bool written = false;
    };

    struct AdLocalBuffer {
        Resource* resource = nullptr;
        uint32_t element_count = 0;
        std::unordered_map<std::string, AdLocalSlot> slots;
    };

    bool fail(std::string message) {
        if (error_ != nullptr && error_->empty()) *error_ = std::move(message);
        return false;
    }

    std::string_view string(uint32_t id) const {
        return id < module_.strings.size() ? std::string_view{module_.strings[id]} : std::string_view{};
    }

    Constant* index_constant(uint32_t value) { return xir_module_.create_constant(Type::of<uint32_t>(), &value); }

    bool is_feir_matrix_type(uint32_t id) const {
        return id < module_.types.size() && module_.types[id].kind == kTypeMatrix;
    }

    uint32_t feir_matrix_dimension(uint32_t id) const {
        return is_feir_matrix_type(id) ? module_.types[id].b : 0u;
    }

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
            if (source.b == source.c && source.b >= 2u && source.b <= 4u && type(source.a) == Type::of<float>()) {
                result = ad_inputs_ == nullptr
                             ? Type::matrix(source.b)
                             : Type::array(Type::vector(Type::of<float>(), source.b), source.b);
            }
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

    const Type* texture_element_type(uint32_t id) {
        if (id >= module_.types.size()) return nullptr;
        return module_.types[id].kind == kTypeStruct ? Type::vector(Type::of<float>(), 4u) : type(id);
    }

    bool is_texture_expression(uint32_t id) const {
        if (id >= module_.expressions.size()) return false;
        const auto& expression = module_.expressions[id];
        if (expression.kind == kExpressionTextureSample) return true;
        if (expression.kind == kExpressionResourceElement) {
            const auto resource = resources_.find(std::string{string(expression.name_id)});
            return resource != resources_.end() &&
                   (resource->second.kind == kResourceTexture2D || resource->second.kind == kResourceTexture3D);
        }
        if (expression.kind != kExpressionIndexAccess || expression.a >= module_.expressions.size()) return false;
        const auto& base = module_.expressions[expression.a];
        if (base.kind != kExpressionLocal && base.kind != kExpressionParameter) return false;
        const auto resource = resources_.find(std::string{string(base.name_id)});
        return resource != resources_.end() &&
               (resource->second.kind == kResourceTexture2D || resource->second.kind == kResourceTexture3D);
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
            if (source.kind == kResourcePushConstant && !inputs_.dynamic_push_constants) continue;
            if (source.kind == kResourceSampler) {
                if (source.sampler_min_filter > 1u || source.sampler_mag_filter > 1u ||
                    source.sampler_mipmap_mode > 1u || source.sampler_address_u > 3u ||
                    source.sampler_address_v > 3u || source.sampler_address_w > 3u)
                    return fail("Luisa received an invalid Feather sampler descriptor");
                if (source.sampler_min_filter != source.sampler_mag_filter)
                    return fail("Luisa XIR cannot represent different minification and magnification filters");
                if (source.sampler_address_u != source.sampler_address_v)
                    return fail("Luisa XIR cannot represent different U/V sampler address modes");
                const auto filter = source.sampler_anisotropy ? 3u
                                    : source.sampler_min_filter == 0u ? 0u
                                    : source.sampler_mipmap_mode == 0u ? 1u : 2u;
                const uint32_t addresses[]{0u, 1u, 2u, 3u};
                samplers_.emplace(source.name,
                                  Sampler{index_constant(filter), index_constant(addresses[source.sampler_address_u])});
                continue;
            }
            const auto texture = source.kind == kResourceTexture2D || source.kind == kResourceTexture3D;
            const auto push_constant = source.kind == kResourcePushConstant;
            if (source.kind != kResourceBuffer && !texture && !push_constant)
                return fail("Luisa XIR received an unsupported forward resource");
            auto* element = texture ? Type::vector(Type::of<float>(), 4u) : type_from_name(source.element_type);
            if (element == nullptr) return fail("Luisa cannot resolve FEIR resource element type '" + source.element_type + "'");
            auto* resource_type = texture ? Type::texture(Type::of<float>(), source.kind == kResourceTexture2D ? 2u : 3u)
                                          : Type::buffer(element);
            auto* argument = function_->create_resource_argument(resource_type);
            resources_.emplace(source.name, Resource{argument, element, source.binding,
                                                     source.element_count, source.kind, source.access});
            if (!texture && buffer_layouts_ != nullptr) {
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

    Resource* resource_by_binding(uint32_t binding) {
        const auto found = std::find_if(resources_.begin(), resources_.end(), [&](auto& entry) {
            return entry.second.binding == binding;
        });
        return found == resources_.end() ? nullptr : &found->second;
    }

    bool register_graphics_stage(const TypedIR::Function& entry) {
        stage_output_ = resource_by_binding(inputs_.stage_output_binding);
        if (entry.kind == kFunctionFragment) {
            if (inputs_.graphics_color_targets.empty()) {
                return fail("fragment stage requires at least one color target");
            }
            const auto& return_type = module_.types[entry.return_type_id];
            for (const auto& target : inputs_.graphics_color_targets) {
                auto* resource = resource_by_binding(target.binding);
                if (resource == nullptr || resource->kind != kResourceTexture2D ||
                    resource->access != kAccessReadWrite ||
                    resource->element_type != Type::vector(Type::of<float>(), 4u)) {
                    return fail("fragment color target does not match a read-write float4 texture");
                }
                if (target.return_field == TypedIR::NoIndex) {
                    if (inputs_.graphics_color_targets.size() != 1u ||
                        resource->element_type != type(entry.return_type_id)) {
                        return fail("scalar fragment output requires exactly one matching color target");
                    }
                } else {
                    if (return_type.kind != kTypeStruct || return_type.a >= module_.structs.size()) {
                        return fail("MRT fragment output must be a struct");
                    }
                    const auto& structure = module_.structs[return_type.a];
                    if (target.return_field >= structure.field_count ||
                        structure.first_field == TypedIR::NoIndex ||
                        structure.first_field + target.return_field >= module_.struct_fields.size() ||
                        type(module_.struct_fields[structure.first_field + target.return_field].type_id) !=
                            resource->element_type) {
                        return fail("MRT fragment field does not match its color target");
                    }
                }
                stage_color_outputs_.push_back(StageColorOutput{resource, target.return_field, target.blend});
            }
            stage_output_ = stage_color_outputs_.front().resource;
            stage_input_ = resource_by_binding(inputs_.stage_input_binding);
            stage_coverage_ = resource_by_binding(inputs_.stage_coverage_binding);
            if (entry.parameter_count != 1u || entry.first_parameter == TypedIR::NoIndex ||
                entry.first_parameter >= module_.parameters.size() || stage_input_ == nullptr ||
                stage_input_->kind != kResourceBuffer ||
                stage_input_->element_type != type(module_.parameters[entry.first_parameter].type_id) ||
                stage_coverage_ == nullptr || stage_coverage_->kind != kResourceBuffer ||
                stage_coverage_->element_type != Type::of<float>()) {
                return fail("fragment stage requires one matching varying input buffer");
            }
        } else {
            if (stage_output_ == nullptr || stage_output_->kind != kResourceBuffer ||
                stage_output_->access != kAccessWrite ||
                stage_output_->element_type != type(entry.return_type_id)) {
                return fail("vertex stage output buffer does not match the FEIR return type");
            }
            if (entry.parameter_count != 0u) return fail("vertex stage entry parameters are unsupported");
        }
        return true;
    }

    bool register_fused_fragment_stage(const TypedIR::Function& entry) {
        if (entry.parameter_count != 1u || entry.first_parameter == TypedIR::NoIndex ||
            entry.first_parameter >= module_.parameters.size()) {
            return fail("fused fragment stage requires one varying parameter");
        }
        const auto& parameter = module_.parameters[entry.first_parameter];
        fragment_parameter_type_ = type(parameter.type_id);
        if (parameter.direction != 0u || fragment_parameter_type_ == nullptr) {
            return fail("fused fragment varying parameter metadata is invalid");
        }
        fragment_parameter_argument_ = fragment_callable_->create_value_argument(fragment_parameter_type_);
        for (auto& argument : fragment_neighbor_arguments_) {
            argument = fragment_callable_->create_value_argument(fragment_parameter_type_);
        }
        return true;
    }

    bool bind_graphics_stage_parameters(const TypedIR::Function& entry) {
        if (entry.kind != kFunctionFragment) return true;
        const auto& parameter = module_.parameters[entry.first_parameter];
        const auto name = std::string{string(parameter.name_id)};
        auto* parameter_type = type(parameter.type_id);
        if (name.empty() || parameter.direction != 0u || parameter_type == nullptr) {
            return fail("fragment varying parameter metadata is invalid");
        }
        auto* dispatch = xir_module_.create_dispatch_id();
        auto* x = extract(dispatch, Type::of<uint32_t>(), {index_constant(0u)});
        auto* y = extract(dispatch, Type::of<uint32_t>(), {index_constant(1u)});
        auto* width = extract(xir_module_.create_dispatch_size(), Type::of<uint32_t>(), {index_constant(0u)});
        auto* row = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {y, width});
        auto* index = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {row, x});
        if (inputs_.graphics_sample_count > 1u) {
            index = builder_.call(
                Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                {builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                               {index, index_constant(inputs_.graphics_sample_count)}),
                 index_constant(inputs_.graphics_sample_index)});
        }
        auto* value = builder_.call(parameter_type, ResourceReadOp::BUFFER_READ, {stage_input_->argument, index});
        auto* local = builder_.alloca_local(parameter_type);
        builder_.store(local, value);
        locals_.emplace(name, local);
        fragment_parameter_name_ = name;
        fragment_parameter_type_ = parameter_type;
        auto* coverage = builder_.call(Type::of<float>(), ResourceReadOp::BUFFER_READ,
                                       {stage_coverage_->argument, index});
        auto* zero = xir_module_.create_constant_zero(Type::of<float>());
        auto* uncovered = builder_.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL, {coverage, zero});
        auto* guard = builder_.if_(uncovered);
        auto* merge = guard->create_merge_block();
        builder_.set_insertion_point(guard->create_true_block());
        builder_.return_void();
        builder_.set_insertion_point(guard->create_false_block());
        builder_.br(merge);
        builder_.set_insertion_point(merge);
        return true;
    }

    bool bind_fused_fragment_parameters(const TypedIR::Function& entry) {
        const auto& parameter = module_.parameters[entry.first_parameter];
        const auto name = std::string{string(parameter.name_id)};
        if (name.empty()) return fail("fused fragment varying parameter name is missing");
        auto* local = builder_.alloca_local(fragment_parameter_type_);
        builder_.store(local, fragment_parameter_argument_);
        locals_.emplace(name, local);
        fragment_parameter_name_ = name;
        return true;
    }

    bool register_ad_resources() {
        if (ad_inputs_ == nullptr) return true;
        if (ad_inputs_->loss_name.empty() || ad_inputs_->parameters.empty())
            return fail("Luisa AD requires one loss and at least one parameter buffer");
        for (const auto& parameter : ad_inputs_->parameters) {
            auto source = std::find_if(resources_.begin(), resources_.end(), [&](const auto& entry) {
                return entry.second.binding == parameter.source_binding;
            });
            if (source == resources_.end() || source->second.kind != kResourceBuffer ||
                parameter.element_count == 0u ||
                (!source->second.element_type->is_float() && !source->second.element_type->is_vector()))
                return fail("Luisa AD parameter must identify a non-empty float buffer");
            auto* gradient = function_->create_resource_argument(Type::buffer(source->second.element_type));
            ad_resources_.emplace(parameter.source_binding,
                                  AdResource{&source->second, gradient, parameter.element_count});
            if (ad_gradient_layouts_ != nullptr)
                ad_gradient_layouts_->push_back(
                    AdGradientLayout{parameter.source_binding, parameter.element_count, source->second.element_type});
        }
        for (auto& [name, resource] : resources_) {
            if (resource.kind != kResourceBuffer || ad_resources_.contains(resource.binding) ||
                resource.access != kAccessReadWrite || resource.element_count == 0u ||
                (!resource.element_type->is_float() && !resource.element_type->is_vector()) ||
                !resource_is_read(name) || !resource_is_written(name)) continue;
            ad_local_buffers_.emplace(resource.binding,
                                      AdLocalBuffer{&resource, resource.element_count});
        }
        return true;
    }

    std::string raw_expression_key(uint32_t id, uint32_t depth = 0u) const {
        if (id >= module_.expressions.size() || depth > 64u) return {};
        if (auto constant = integer_expression(id)) return "#" + std::to_string(*constant);
        const auto& expression = module_.expressions[id];
        std::string key = std::to_string(expression.kind) + ":" + std::to_string(expression.op) + ":" +
                          std::to_string(expression.type_id) + ":" + std::string{string(expression.name_id)};
        const auto append = [&](uint32_t child) {
            key += "(" + raw_expression_key(child, depth + 1u) + ")";
        };
        switch (expression.kind) {
        case kExpressionUnary:
        case kExpressionConversion:
        case kExpressionField:
        case kExpressionMemberAccess:
        case kExpressionResourceElement:
            append(expression.a);
            break;
        case kExpressionBinary:
        case kExpressionComparison:
        case kExpressionLogical:
        case kExpressionIndexAccess:
        case kExpressionMatrixColumn:
            append(expression.a);
            append(expression.b);
            break;
        case kExpressionConditional:
            append(expression.a);
            append(expression.b);
            append(expression.c);
            break;
        case kExpressionConstructor:
        case kExpressionIntrinsic:
        case kExpressionCallableCall:
            if (expression.first_argument != TypedIR::NoIndex &&
                expression.first_argument <= module_.arguments.size() &&
                expression.argument_count <= module_.arguments.size() - expression.first_argument) {
                for (uint32_t i = 0u; i < expression.argument_count; ++i)
                    append(module_.arguments[expression.first_argument + i]);
            }
            break;
        default:
            break;
        }
        return key;
    }

    bool collect_ad_index_terms(uint32_t id, int sign, uint32_t depth,
                                std::vector<AdIndexTerm>& terms, int64_t& offset) const {
        if (id >= module_.expressions.size() || depth > 64u) return false;
        if (auto constant = integer_expression(id)) {
            auto sum = sign > 0 ? checked_add(offset, *constant)
                                : checked_subtract(offset, *constant);
            if (!sum) return false;
            offset = *sum;
            return true;
        }
        const auto& expression = module_.expressions[id];
        if (expression.kind == kExpressionBinary && (expression.op == 0u || expression.op == 1u)) {
            return collect_ad_index_terms(expression.a, sign, depth + 1u, terms, offset) &&
                   collect_ad_index_terms(expression.b, expression.op == 0u ? sign : -sign,
                                          depth + 1u, terms, offset);
        }
        auto key = raw_expression_key(id, depth);
        if (key.empty()) return false;
        terms.emplace_back(AdIndexTerm{id, sign, std::move(key)});
        return true;
    }

    std::optional<AdIndex> canonical_ad_index(uint32_t id) const {
        AdIndex index;
        if (!collect_ad_index_terms(id, 1, 0u, index.terms, index.offset)) return std::nullopt;
        std::sort(index.terms.begin(), index.terms.end(), [](const auto& left, const auto& right) {
            return left.sign != right.sign ? left.sign < right.sign : left.key < right.key;
        });
        return index;
    }

    static std::string ad_index_key(const AdIndex& index) {
        std::string key = "offset=" + std::to_string(index.offset);
        for (const auto& term : index.terms)
            key += term.sign > 0 ? "+(" + term.key + ")" : "-(" + term.key + ")";
        return key;
    }

    static bool ad_index_terms_include(const AdIndex& index, const AdIndex& subset) {
        const auto less = [](const auto& left, const auto& right) {
            return left.sign != right.sign ? left.sign < right.sign : left.key < right.key;
        };
        return std::includes(index.terms.begin(), index.terms.end(),
                             subset.terms.begin(), subset.terms.end(), less);
    }

    Value* lower_ad_index(const AdIndex& index) {
        const auto magnitude = index.offset < 0
                                   ? static_cast<uint64_t>(-(index.offset + 1)) + 1u
                                   : static_cast<uint64_t>(index.offset);
        if (magnitude > std::numeric_limits<uint32_t>::max()) return nullptr;
        auto value = static_cast<uint32_t>(magnitude);
        Value* result = xir_module_.create_constant_zero(Type::of<uint32_t>());
        if (value != 0u) {
            auto* constant = xir_module_.create_constant(Type::of<uint32_t>(), &value);
            result = builder_.call(Type::of<uint32_t>(),
                                   index.offset < 0 ? ArithmeticOp::BINARY_SUB : ArithmeticOp::BINARY_ADD,
                                   {result, constant});
        }
        for (const auto& term : index.terms) {
            auto* term_value = lower_expression(term.expression);
            if (term_value == nullptr) return nullptr;
            term_value = builder_.static_cast_if_necessary(Type::of<uint32_t>(), term_value);
            result = builder_.call(Type::of<uint32_t>(),
                                   term.sign > 0 ? ArithmeticOp::BINARY_ADD : ArithmeticOp::BINARY_SUB,
                                   {result, term_value});
        }
        return result;
    }

    Value* select_ad_local_read(const AdLocalBuffer& local, const AdIndex& read_index,
                                Value* runtime_index, Value* fallback) {
        Value* result = fallback;
        std::vector<std::pair<std::string_view, const AdLocalSlot*>> candidates;
        candidates.reserve(local.slots.size());
        for (const auto& [key, slot] : local.slots) {
            if (!slot.written || slot.index.terms.size() >= read_index.terms.size() ||
                !ad_index_terms_include(read_index, slot.index)) continue;
            candidates.emplace_back(key, &slot);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            return left.first < right.first;
        });
        for (const auto& [_, slot] : candidates) {
            auto* candidate_index = lower_ad_index(slot->index);
            if (candidate_index == nullptr) return nullptr;
            auto* matches = builder_.call(Type::of<bool>(), ArithmeticOp::BINARY_EQUAL,
                                          {runtime_index, candidate_index});
            auto* candidate = builder_.load(local.resource->element_type, slot->pointer);
            result = builder_.call(local.resource->element_type, ArithmeticOp::SELECT,
                                   {result, candidate, matches});
        }
        return candidates.empty() ? nullptr : result;
    }

    Value* ad_local_slot(Resource& resource, Value* runtime_index,
                         uint32_t index_expression, Value* initial,
                         Value** selected_result = nullptr) {
        auto local = ad_local_buffers_.find(resource.binding);
        if (local == ad_local_buffers_.end()) return nullptr;
        auto index = canonical_ad_index(index_expression);
        if (!index) return nullptr;
        auto key = ad_index_key(*index);
        if (auto found = local->second.slots.find(key); found != local->second.slots.end()) {
            if (initial == nullptr) found->second.written = true;
            return found->second.pointer;
        }
        if (initial != nullptr && selected_result != nullptr) {
            if (auto* selected = select_ad_local_read(local->second, *index, runtime_index, initial);
                selected != nullptr) {
                *selected_result = selected;
                return nullptr;
            }
        }
        XIRBuilder allocation_builder;
        allocation_builder.set_insertion_point(
            function_->definition()->body_block()->instructions().head_sentinel());
        auto* slot = allocation_builder.alloca_local(resource.element_type);
        local->second.slots.emplace(key, AdLocalSlot{slot, std::move(*index), initial == nullptr});
        if (initial != nullptr) builder_.store(slot, initial);
        return slot;
    }

    Value* track_ad_read(Resource& resource, Value* index, Value* value,
                         uint32_t index_expression = TypedIR::NoIndex) {
        if (!inside_ad_scope_ || value == nullptr) return value;
        if (auto parameter = ad_resources_.find(resource.binding);
            parameter != ad_resources_.end() && parameter->second.parameter != nullptr)
            return extract(parameter->second.parameter, resource.element_type, {index});
        Value* selected = nullptr;
        if (auto* slot = ad_local_slot(resource, index, index_expression, value, &selected); slot != nullptr)
            return builder_.load(resource.element_type, slot);
        if (selected != nullptr) return selected;
        return value;
    }

    bool resource_is_read(std::string_view name) const {
        for (const auto& expression : module_.expressions) {
            if (expression.kind == kExpressionResourceElement && string(expression.name_id) == name)
                return true;
            if (expression.kind == kExpressionIndexAccess && expression.a < module_.expressions.size()) {
                const auto& base = module_.expressions[expression.a];
                if ((base.kind == kExpressionLocal || base.kind == kExpressionParameter) &&
                    string(base.name_id) == name) return true;
            }
        }
        return false;
    }

    bool resource_is_written(std::string_view name) const {
        for (const auto& lvalue : module_.lvalues) {
            if (lvalue.kind == kLValueResourceElement && string(lvalue.name_id) == name)
                return true;
        }
        return false;
    }

    bool begin_ad_scope() {
        for (auto& [_, resource] : ad_resources_) {
            auto* parameter_type = Type::array(resource.source->element_type, resource.element_count);
            std::vector<Value*> elements;
            elements.reserve(resource.element_count);
            for (uint32_t i = 0u; i < resource.element_count; ++i) {
                elements.emplace_back(builder_.call(
                    resource.source->element_type, ResourceReadOp::BUFFER_READ,
                    {resource.source->argument, index_constant(i)}));
            }
            resource.parameter = builder_.call(parameter_type, ArithmeticOp::AGGREGATE, elements);
            builder_.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_REQUIRES_GRADIENT,
                          {resource.parameter});
        }
        return true;
    }

    bool finish_ad_scope() {
        auto loss = locals_.find(ad_inputs_->loss_name);
        if (loss == locals_.end() || loss->second->type() != Type::of<float>())
            return fail("Luisa AD loss annotation does not resolve to a scalar float local");
        auto* loss_value = builder_.load(Type::of<float>(), loss->second);
        auto* one = xir_module_.create_constant_one(Type::of<float>());
        builder_.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_GRADIENT_MARKER, {loss_value, one});
        builder_.call(Type::of<void>(), AutodiffIntrinsicOp::AUTODIFF_BACKWARD, {});

        auto* dispatch_id = xir_module_.create_dispatch_id();
        auto* thread = extract(dispatch_id, Type::of<uint32_t>(), {index_constant(0u)});
        for (const auto& [_, resource] : ad_resources_) {
            auto* gradient_type = Type::array(resource.source->element_type, resource.element_count);
            auto* gradient = builder_.call(gradient_type, AutodiffIntrinsicOp::AUTODIFF_GRADIENT,
                                           {resource.parameter});
            auto* count = index_constant(resource.element_count);
            auto* base = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {thread, count});
            for (uint32_t i = 0u; i < resource.element_count; ++i) {
                auto* index = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                                            {base, index_constant(i)});
                auto* element = extract(gradient, resource.source->element_type, {index_constant(i)});
                builder_.call(ResourceWriteOp::BUFFER_WRITE, {resource.gradient, index, element});
            }
        }
        return true;
    }

    bool is_void_type(uint32_t id) const {
        return id < module_.types.size() && module_.types[id].kind == kTypeVoid;
    }

    bool stage_callables() {
        for (uint32_t function_id = 0; function_id < module_.functions.size(); ++function_id) {
            const auto& source = module_.functions[function_id];
            if (source.kind != kFunctionCallable) continue;
            if ((!is_void_type(source.return_type_id) && type(source.return_type_id) == nullptr) ||
                source.body_statement_index >= module_.statements.size() ||
                (source.parameter_count != 0u &&
                 (source.first_parameter == TypedIR::NoIndex || source.first_parameter > module_.parameters.size() ||
                  source.parameter_count > module_.parameters.size() - source.first_parameter)))
                return fail("FEIR callable has an invalid return type, body, or parameter range");
            auto name = std::string{string(source.mangled_name_id)};
            if (name.empty() || callables_.contains(name)) return fail("FEIR callable name is missing or duplicated");
            auto* callable = xir_module_.create_callable(is_void_type(source.return_type_id) ? nullptr
                                                                                              : type(source.return_type_id));
            callable->set_name(name);
            CallableRecord record{.function = callable, .function_id = function_id};
            record.parameters.reserve(source.parameter_count);
            for (uint32_t i = 0; i < source.parameter_count; ++i) {
                const auto& parameter = module_.parameters[source.first_parameter + i];
                if (parameter.type_id >= module_.types.size()) return fail("FEIR callable parameter type is out of range");
                const auto& parameter_type = module_.types[parameter.type_id];
                CallableParameter lowered{.type_id = parameter.type_id, .direction = parameter.direction};
                if (parameter_type.kind == kTypeResourceWrapper) {
                    if (parameter.direction != 0u) return fail("FEIR callable resource parameters must be inputs");
                    if (parameter_type.a == kTypeResourceSampler) {
                        lowered.argument = callable->create_value_argument(Type::of<uint32_t>());
                        lowered.sampler_address = callable->create_value_argument(Type::of<uint32_t>());
                    } else {
                        auto* element = parameter_type.a == kTypeResourceTexture2D ||
                                                parameter_type.a == kTypeResourceTexture3D
                                            ? texture_element_type(parameter_type.b)
                                            : type(parameter_type.b);
                        if (element == nullptr) return fail("FEIR callable resource element type is unsupported");
                        const Type* resource_type = nullptr;
                        if (parameter_type.a == kTypeResourceBuffer) resource_type = Type::buffer(element);
                        else if (parameter_type.a == kTypeResourceTexture2D)
                            resource_type = Type::texture(Type::of<float>(), 2u);
                        else if (parameter_type.a == kTypeResourceTexture3D)
                            resource_type = Type::texture(Type::of<float>(), 3u);
                        else return fail("FEIR callable resource kind is unsupported");
                        lowered.argument = callable->create_resource_argument(resource_type);
                    }
                } else {
                    auto* parameter_value_type = type(parameter.type_id);
                    if (parameter_value_type == nullptr) return fail("FEIR callable value parameter type is unsupported");
                    lowered.argument = parameter.direction == 0u
                                           ? static_cast<Argument*>(callable->create_value_argument(parameter_value_type))
                                           : static_cast<Argument*>(callable->create_reference_argument(parameter_value_type));
                }
                record.parameters.push_back(lowered);
            }
            callables_.emplace(std::move(name), std::move(record));
        }
        return true;
    }

    bool lower_callable_bodies() {
        for (auto& [name, callable] : callables_) {
            const auto& source = module_.functions[callable.function_id];
            auto saved_locals = std::move(locals_);
            auto saved_constant_integers = std::move(constant_integers_);
            auto saved_resources = std::move(resources_);
            auto saved_samplers = std::move(samplers_);
            auto saved_shared = std::move(shared_);
            auto saved_loops = std::move(loops_);
            locals_.clear();
            constant_integers_.clear();
            resources_.clear();
            samplers_.clear();
            shared_.clear();
            loops_.clear();
            builder_.set_insertion_point(callable.function->create_body_block());
            for (uint32_t i = 0; i < source.parameter_count; ++i) {
                const auto& parameter = module_.parameters[source.first_parameter + i];
                const auto& lowered = callable.parameters[i];
                const auto parameter_name = std::string{string(parameter.name_id)};
                const auto& parameter_type = module_.types[parameter.type_id];
                if (parameter_name.empty()) return fail("FEIR callable parameter name is missing");
                if (parameter_type.kind == kTypeResourceWrapper) {
                    if (parameter_type.a == kTypeResourceSampler) {
                        samplers_.emplace(parameter_name, Sampler{lowered.argument, lowered.sampler_address});
                    } else {
                        const uint8_t kind = parameter_type.a == kTypeResourceBuffer ? kResourceBuffer
                                             : parameter_type.a == kTypeResourceTexture2D ? kResourceTexture2D
                                                                                         : kResourceTexture3D;
                        const uint8_t access = parameter_type.c == 0u ? 1u : parameter_type.c == 1u ? 2u
                                               : parameter_type.c == 2u ? 3u : 4u;
                        auto* element = kind == kResourceTexture2D || kind == kResourceTexture3D
                                            ? texture_element_type(parameter_type.b)
                                            : type(parameter_type.b);
                        resources_.emplace(parameter_name,
                            Resource{static_cast<ResourceArgument*>(lowered.argument), element, 0u, 0u, kind, access});
                    }
                } else if (parameter.direction == 0u) {
                    auto* storage = builder_.alloca_local(type(parameter.type_id));
                    builder_.store(storage, lowered.argument);
                    locals_.emplace(parameter_name, storage);
                } else {
                    locals_.emplace(parameter_name, lowered.argument);
                }
            }
            if (!lower_statement(source.body_statement_index)) return false;
            if (!builder_.is_insertion_point_terminator()) {
                if (is_void_type(source.return_type_id)) builder_.return_void();
                else return fail("non-void FEIR callable does not terminate with a return");
            }
            locals_ = std::move(saved_locals);
            constant_integers_ = std::move(saved_constant_integers);
            resources_ = std::move(saved_resources);
            samplers_ = std::move(saved_samplers);
            shared_ = std::move(saved_shared);
            loops_ = std::move(saved_loops);
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

    // Shifts keep the right-hand operand independent of the result shape, so they must not be splatted.
    static bool is_shift_op(ArithmeticOp op) {
        return op == ArithmeticOp::BINARY_SHIFT_LEFT || op == ArithmeticOp::BINARY_SHIFT_RIGHT ||
               op == ArithmeticOp::BINARY_ROTATE_LEFT || op == ArithmeticOp::BINARY_ROTATE_RIGHT;
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

    // C# permits mixed vector/scalar arithmetic such as `float3 * float`, while XIR requires every
    // operand of an elementwise arithmetic instruction to match the result type exactly. Broadcast the
    // scalar side into a vector of the result shape so the generated module passes verification.
    Value* splat_to(Value* value, const Type* result_type) {
        if (value == nullptr || result_type == nullptr) return value;
        if (!result_type->is_vector() || !value->type()->is_scalar()) return value;
        auto* element = builder_.static_cast_if_necessary(result_type->element(), value);
        std::vector<Value*> components(result_type->dimension(), element);
        return builder_.call(result_type, ArithmeticOp::AGGREGATE, components);
    }

    Value* graphics_blend_factor(uint32_t factor, Value* source, Value* destination,
                                 uint32_t component) {
        auto* float_type = Type::of<float>();
        auto* zero = xir_module_.create_constant_zero(float_type);
        auto* one = xir_module_.create_constant_one(float_type);
        auto component_of = [&](Value* value, uint32_t index) {
            return extract(value, float_type, {index_constant(index)});
        };
        Value* value = nullptr;
        bool invert = false;
        switch (factor) {
        case 0u: return zero;
        case 1u: return one;
        case 2u: value = component_of(source, component); break;
        case 3u: value = component_of(source, component); invert = true; break;
        case 4u: value = component_of(destination, component); break;
        case 5u: value = component_of(destination, component); invert = true; break;
        case 6u: value = component_of(source, 3u); break;
        case 7u: value = component_of(source, 3u); invert = true; break;
        case 8u: value = component_of(destination, 3u); break;
        case 9u: value = component_of(destination, 3u); invert = true; break;
        default: return fail("compute raster received an invalid blend factor"), nullptr;
        }
        return invert ? builder_.call(float_type, ArithmeticOp::BINARY_SUB, {one, value}) : value;
    }

    Value* graphics_blend_component(Value* source, Value* destination, uint32_t component,
                                    uint32_t source_factor, uint32_t destination_factor,
                                    uint32_t operation) {
        auto* float_type = Type::of<float>();
        auto* source_value = extract(source, float_type, {index_constant(component)});
        auto* destination_value = extract(destination, float_type, {index_constant(component)});
        if (operation == 3u || operation == 4u) {
            return builder_.call(float_type,
                                 operation == 3u ? ArithmeticOp::MIN : ArithmeticOp::MAX,
                                 {source_value, destination_value});
        }
        auto* source_scale = graphics_blend_factor(source_factor, source, destination, component);
        auto* destination_scale = graphics_blend_factor(destination_factor, source, destination, component);
        if (source_scale == nullptr || destination_scale == nullptr) return nullptr;
        auto* source_term = builder_.call(float_type, ArithmeticOp::BINARY_MUL,
                                          {source_value, source_scale});
        auto* destination_term = builder_.call(float_type, ArithmeticOp::BINARY_MUL,
                                               {destination_value, destination_scale});
        switch (operation) {
        case 0u:
            return builder_.call(float_type, ArithmeticOp::BINARY_ADD,
                                 {source_term, destination_term});
        case 1u:
            return builder_.call(float_type, ArithmeticOp::BINARY_SUB,
                                 {source_term, destination_term});
        case 2u:
            return builder_.call(float_type, ArithmeticOp::BINARY_SUB,
                                 {destination_term, source_term});
        default:
            return fail("compute raster received an invalid blend operation"), nullptr;
        }
    }

    Value* apply_graphics_blend(Value* source, Value* destination,
                                const TypedIR::GraphicsBlendInfo& blend) {
        auto* float4_type = Type::vector(Type::of<float>(), 4u);
        if (source == nullptr || destination == nullptr || source->type() != float4_type ||
            destination->type() != float4_type) {
            return fail("compute raster blending requires float4 colors"), nullptr;
        }
        std::vector<Value*> components;
        components.reserve(4u);
        for (uint32_t component = 0u; component < 4u; ++component) {
            auto* destination_value = extract(destination, Type::of<float>(), {index_constant(component)});
            Value* output = nullptr;
            if (!blend.enable) {
                output = extract(source, Type::of<float>(), {index_constant(component)});
            } else if (component < 3u) {
                output = graphics_blend_component(source, destination, component,
                                                  blend.src_color, blend.dst_color, blend.color_op);
            } else {
                output = graphics_blend_component(source, destination, component,
                                                  blend.src_alpha, blend.dst_alpha, blend.alpha_op);
            }
            if (output == nullptr) return nullptr;
            components.push_back((blend.write_mask & (1u << component)) != 0u
                                     ? output
                                     : destination_value);
        }
        return builder_.call(float4_type, ArithmeticOp::AGGREGATE, components);
    }

    Value* convert(Value* value, const Type* result_type) {
        if (value == nullptr || result_type == nullptr) return nullptr;
        if (value->type() == result_type) return value;
        if (value->type()->is_scalar() && result_type->is_scalar())
            return builder_.static_cast_if_necessary(result_type, value);
        if (value->type()->is_vector() && result_type->is_vector() &&
            value->type()->dimension() == result_type->dimension()) {
            std::vector<Value*> components;
            components.reserve(result_type->dimension());
            for (uint32_t i = 0; i < result_type->dimension(); ++i) {
                auto* component = extract(value, value->type()->element(), {index_constant(i)});
                components.push_back(builder_.static_cast_if_necessary(result_type->element(), component));
            }
            return builder_.call(result_type, ArithmeticOp::AGGREGATE, components);
        }
        return fail("unsupported FEIR numeric conversion"), nullptr;
    }

    Value* matrix_multiply(Value* left, Value* right, const Type* result_type,
                           bool left_matrix, bool right_matrix, uint32_t dimension) {
        const auto* element = Type::of<float>();
        auto dot_row = [&](Value* matrix, Value* vector, uint32_t row) -> Value* {
            Value* sum = nullptr;
            for (uint32_t column = 0; column < dimension; ++column) {
                auto* matrix_value = extract(matrix, element, {index_constant(column), index_constant(row)});
                auto* vector_value = extract(vector, element, {index_constant(column)});
                auto* product = builder_.call(element, ArithmeticOp::BINARY_MUL, {matrix_value, vector_value});
                sum = sum == nullptr ? product : builder_.call(element, ArithmeticOp::BINARY_ADD, {sum, product});
            }
            return sum;
        };
        auto multiply_matrix_vector = [&](Value* matrix, Value* vector) -> Value* {
            std::vector<Value*> rows;
            rows.reserve(dimension);
            for (uint32_t row = 0; row < dimension; ++row)
                rows.push_back(dot_row(matrix, vector, row));
            return builder_.call(Type::vector(element, dimension), ArithmeticOp::AGGREGATE, rows);
        };
        if (left_matrix && !right_matrix)
            return multiply_matrix_vector(left, right);
        if (left_matrix && right_matrix) {
            std::vector<Value*> columns;
            columns.reserve(dimension);
            auto* column_type = Type::vector(element, dimension);
            for (uint32_t column = 0; column < dimension; ++column) {
                auto* right_column = extract(right, column_type, {index_constant(column)});
                columns.push_back(multiply_matrix_vector(left, right_column));
            }
            return builder_.call(result_type, ArithmeticOp::AGGREGATE, columns);
        }
        if (!left_matrix && right_matrix) {
            std::vector<Value*> values;
            values.reserve(dimension);
            auto* column_type = Type::vector(element, dimension);
            for (uint32_t column = 0; column < dimension; ++column) {
                auto* right_column = extract(right, column_type, {index_constant(column)});
                Value* sum = nullptr;
                for (uint32_t row = 0; row < dimension; ++row) {
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
        if (expression.kind == kExpressionTextureSample) {
            result_type = texture_element_type(expression.type_id);
        } else if (expression.kind == kExpressionResourceElement) {
            const auto resource = resources_.find(std::string{string(expression.name_id)});
            if (resource != resources_.end() &&
                (resource->second.kind == kResourceTexture2D || resource->second.kind == kResourceTexture3D))
                result_type = resource->second.element_type;
        } else if (expression.kind == kExpressionIndexAccess && expression.a < module_.expressions.size()) {
            const auto& base = module_.expressions[expression.a];
            if (base.kind == kExpressionLocal || base.kind == kExpressionParameter) {
                const auto resource = resources_.find(std::string{string(base.name_id)});
                if (resource != resources_.end() &&
                    (resource->second.kind == kResourceTexture2D || resource->second.kind == kResourceTexture3D))
                    result_type = resource->second.element_type;
            }
        } else if ((expression.kind == kExpressionField || expression.kind == kExpressionMemberAccess) &&
                   is_texture_expression(expression.a)) {
            result_type = Type::of<float>();
        }
        if (result_type == nullptr && !(expression.kind == kExpressionCallableCall && is_void_type(expression.type_id)))
            return fail("FEIR expression kind " + std::to_string(expression.kind) + " has unsupported type " +
                        std::to_string(expression.type_id)), nullptr;
        switch (expression.kind) {
        case kExpressionLiteral:
            return literal(expression, result_type);
        case kExpressionLocal:
        case kExpressionParameter: {
            auto name = std::string{string(expression.name_id)};
            if (ad_inputs_ != nullptr && is_integer_type(expression.type_id)) {
                if (const auto constant = constant_integers_.find(name);
                    constant != constant_integers_.end()) {
                    if (module_.types[expression.type_id].a == 1u) {
                        const auto value = static_cast<int32_t>(constant->second);
                        return xir_module_.create_constant(result_type, &value);
                    }
                    const auto value = static_cast<uint32_t>(constant->second);
                    return xir_module_.create_constant(result_type, &value);
                }
            }
            const auto found = locals_.find(name);
            return found == locals_.end() ? (fail("FEIR references an unknown local"), nullptr)
                                          : builder_.load(result_type, found->second);
        }
        case kExpressionResourceElement: {
            const auto found = resources_.find(std::string{string(expression.name_id)});
            auto* index = lower_expression(expression.a);
            if (found == resources_.end() || index == nullptr) return fail("FEIR resource read is invalid"), nullptr;
            if (found->second.kind == kResourceTexture2D || found->second.kind == kResourceTexture3D) {
                const auto dimension = found->second.kind == kResourceTexture2D ? 2u : 3u;
                auto* coord_type = Type::vector(Type::of<uint32_t>(), dimension);
                index = convert(index, coord_type);
                auto* texture_value = builder_.call(found->second.element_type,
                    dimension == 2u ? ResourceReadOp::TEXTURE2D_READ : ResourceReadOp::TEXTURE3D_READ,
                    {found->second.argument, index});
                return texture_value->type() == result_type ? texture_value
                                                            : builder_.bit_cast_if_necessary(result_type, texture_value);
            }
            index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
            return track_ad_read(found->second, index,
                                 builder_.call(result_type, ResourceReadOp::BUFFER_READ,
                                               {found->second.argument, index}),
                                 expression.a);
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
            const auto left_matrix = expression.a < module_.expressions.size() &&
                                     is_feir_matrix_type(module_.expressions[expression.a].type_id);
            const auto right_matrix = expression.b < module_.expressions.size() &&
                                      is_feir_matrix_type(module_.expressions[expression.b].type_id);
            if (expression.op == 2u && (left_matrix || right_matrix)) {
                const auto dimension = left_matrix
                                           ? feir_matrix_dimension(module_.expressions[expression.a].type_id)
                                           : feir_matrix_dimension(module_.expressions[expression.b].type_id);
                return matrix_multiply(left, right, result_type, left_matrix, right_matrix, dimension);
            }
            if (is_shift_op(*op)) return builder_.call(result_type, *op, {splat_to(left, result_type), right});
            return builder_.call(result_type, *op, {splat_to(left, result_type), splat_to(right, result_type)});
        }
        case kExpressionComparison: {
            auto* left = lower_expression(expression.a);
            auto* right = lower_expression(expression.b);
            auto op = compare_op(expression.op);
            if (left == nullptr || right == nullptr || !op) return fail("invalid FEIR comparison"), nullptr;
            // Comparisons yield a boolean result, so the operands are broadcast to the wider operand
            // shape rather than to the result type.
            const auto* operand_type = left->type()->is_vector() ? left->type() : right->type();
            return builder_.call(result_type, *op, {splat_to(left, operand_type), splat_to(right, operand_type)});
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
            return convert(value, result_type);
        }
        case kExpressionConstructor: {
            auto values = arguments(expression);
            if (values.empty()) return fail("FEIR aggregate constructor requires arguments"), nullptr;
            if (is_feir_matrix_type(expression.type_id)) {
                const auto dimension = feir_matrix_dimension(expression.type_id);
                if (values.size() == dimension * dimension &&
                    std::all_of(values.begin(), values.end(), [](const auto* value) {
                        return value->type()->is_scalar();
                    })) {
                    std::vector<Value*> columns;
                    columns.reserve(dimension);
                    const auto* column_type = result_type->is_matrix()
                                                  ? Type::vector(result_type->element(), dimension)
                                                  : result_type->element();
                    for (uint32_t column = 0u; column < dimension; ++column) {
                        const auto first = values.begin() + static_cast<std::ptrdiff_t>(column * dimension);
                        std::vector<Value*> column_values(
                            first, first + static_cast<std::ptrdiff_t>(dimension));
                        columns.push_back(builder_.call(column_type, ArithmeticOp::AGGREGATE, column_values));
                    }
                    values = std::move(columns);
                }
                const auto* column_type = result_type->is_matrix()
                                              ? Type::vector(result_type->element(), dimension)
                                              : result_type->element();
                if (values.size() != dimension ||
                    std::any_of(values.begin(), values.end(), [&](const auto* value) {
                        return value->type() != column_type;
                    })) return fail("FEIR matrix constructor requires column vectors or column-major scalars"), nullptr;
            } else if (values.size() == 1u && values.front()->type()->is_scalar() && result_type->is_vector()) {
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
            if (expression.a < module_.expressions.size()) {
                const auto& base_expression = module_.expressions[expression.a];
                if (base_expression.kind == kExpressionLocal || base_expression.kind == kExpressionParameter) {
                    const auto resource = resources_.find(std::string{string(base_expression.name_id)});
                    if (resource != resources_.end()) {
                        auto* index = lower_expression(expression.b);
                        if (index == nullptr) return nullptr;
                        if (resource->second.kind == kResourceTexture2D || resource->second.kind == kResourceTexture3D) {
                            const auto dimension = resource->second.kind == kResourceTexture2D ? 2u : 3u;
                            index = convert(index, Type::vector(Type::of<uint32_t>(), dimension));
                            return builder_.call(result_type,
                                dimension == 2u ? ResourceReadOp::TEXTURE2D_READ : ResourceReadOp::TEXTURE3D_READ,
                                {resource->second.argument, index});
                        }
                        index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
                        return track_ad_read(resource->second, index,
                                             builder_.call(result_type, ResourceReadOp::BUFFER_READ,
                                                           {resource->second.argument, index}),
                                             expression.b);
                    }
                }
            }
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
            index = shared_index(index, found->second.length);
            auto* pointer = builder_.gep(result_type, found->second.pointer, {index});
            return builder_.load(result_type, pointer);
        }
        case kExpressionIntrinsic:
            return lower_intrinsic(expression, result_type);
        case kExpressionCallableCall:
            return lower_callable_call(expression, result_type);
        case kExpressionAtomic:
            return lower_atomic(expression, result_type);
        case kExpressionTextureSample:
            return lower_texture_sample(expression, result_type);
        default:
            return fail("unsupported FEIR expression kind " + std::to_string(expression.kind)), nullptr;
        }
    }

    Value* lower_builtin(uint32_t builtin, const Type* result_type) {
        Value* vector = nullptr;
        uint32_t component = 0;
        if (builtin >= 1u && builtin <= 3u) { vector = xir_module_.create_dispatch_id(); component = builtin - 1u; }
        else if (builtin >= 4u && builtin <= 6u) {
            component = builtin - 4u;
            auto* dispatch = extract(xir_module_.create_dispatch_id(), Type::of<uint32_t>(), {index_constant(component)});
            const uint32_t sizes[]{static_cast<uint32_t>(inputs_.group_x), static_cast<uint32_t>(inputs_.group_y),
                                   static_cast<uint32_t>(inputs_.group_z)};
            auto* divisor = xir_module_.create_constant(Type::of<uint32_t>(), &sizes[component]);
            auto* value = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {dispatch, divisor});
            return builder_.static_cast_if_necessary(result_type, value);
        }
        else if (builtin >= 7u && builtin <= 9u) {
            component = builtin - 7u;
            auto* dispatch = extract(xir_module_.create_dispatch_id(), Type::of<uint32_t>(), {index_constant(component)});
            const uint32_t sizes[]{static_cast<uint32_t>(inputs_.group_x), static_cast<uint32_t>(inputs_.group_y),
                                   static_cast<uint32_t>(inputs_.group_z)};
            auto* divisor = xir_module_.create_constant(Type::of<uint32_t>(), &sizes[component]);
            auto* value = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {dispatch, divisor});
            return builder_.static_cast_if_necessary(result_type, value);
        }
        else if (builtin >= 10u && builtin <= 12u) { vector = xir_module_.create_dispatch_size(); component = builtin - 10u; }
        else if (builtin >= 13u && builtin <= 15u) {
            const uint32_t sizes[]{static_cast<uint32_t>(inputs_.group_x), static_cast<uint32_t>(inputs_.group_y),
                                   static_cast<uint32_t>(inputs_.group_z)};
            return builder_.static_cast_if_necessary(result_type,
                                                      xir_module_.create_constant(Type::of<uint32_t>(), &sizes[builtin - 13u]));
        }
        else if (builtin == 16u) {
            auto* value = extract(xir_module_.create_dispatch_id(), Type::of<uint32_t>(), {index_constant(0u)});
            if (inputs_.graphics_vertex_count != 0u) {
                auto* count = xir_module_.create_constant(Type::of<uint32_t>(), &inputs_.graphics_vertex_count);
                value = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MOD, {value, count});
            }
            return builder_.static_cast_if_necessary(result_type, value);
        }
        else if (builtin == 17u) {
            Value* value = xir_module_.create_constant_zero(Type::of<uint32_t>());
            if (inputs_.graphics_vertex_count != 0u) {
                auto* dispatch = extract(xir_module_.create_dispatch_id(), Type::of<uint32_t>(), {index_constant(0u)});
                auto* count = xir_module_.create_constant(Type::of<uint32_t>(), &inputs_.graphics_vertex_count);
                value = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {dispatch, count});
                if (inputs_.graphics_first_instance != 0u) {
                    auto* first = xir_module_.create_constant(Type::of<uint32_t>(), &inputs_.graphics_first_instance);
                    value = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {value, first});
                }
            }
            return builder_.static_cast_if_necessary(result_type, value);
        }
        if (vector == nullptr) return fail("unsupported compute builtin"), nullptr;
        auto* value = extract(vector, Type::of<uint32_t>(), {index_constant(component)});
        return builder_.static_cast_if_necessary(result_type, value);
    }

    Value* lower_push_constant(const TypedIR::Expression& expression, const Type* result_type) {
        if (inputs_.dynamic_push_constants) {
            auto* resource = resource_by_binding(expression.op);
            if (resource == nullptr || resource->kind != kResourcePushConstant ||
                resource->element_type != result_type) {
                return fail("FEIR dynamic push constant has the wrong resource layout"), nullptr;
            }
            return builder_.call(result_type, ResourceReadOp::BUFFER_READ,
                                 {resource->argument, index_constant(0u)});
        }
        const TypedIR::PushConstantInfo* found = nullptr;
        for (const auto& push : inputs_.push_constants) if (push.binding == expression.op) found = &push;
        if (found == nullptr || found->data == nullptr || found->size < result_type->size())
            return fail("FEIR push constant is missing or has the wrong layout"), nullptr;
        return xir_module_.create_constant(result_type, found->data);
    }

    Value* lower_callable_call(const TypedIR::Expression& expression, const Type* result_type) {
        const auto found = callables_.find(std::string{string(expression.name_id)});
        if (found == callables_.end() || !argument_range(expression) ||
            expression.argument_count != found->second.parameters.size())
            return fail("FEIR callable call has an unknown target or invalid argument range"), nullptr;
        std::vector<Value*> values;
        struct Writeback {
            Address address;
            Value* temporary;
            const Type* type;
        };
        std::vector<Writeback> writebacks;
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            const auto expression_id = module_.arguments[expression.first_argument + i];
            const auto& parameter = found->second.parameters[i];
            const auto& parameter_type = module_.types[parameter.type_id];
            if (parameter_type.kind == kTypeResourceWrapper) {
                if (expression_id >= module_.expressions.size())
                    return fail("FEIR callable resource argument is out of range"), nullptr;
                const auto& argument = module_.expressions[expression_id];
                if (argument.kind != kExpressionLocal && argument.kind != kExpressionParameter)
                    return fail("FEIR callable resource argument is not a direct resource reference"), nullptr;
                const auto name = std::string{string(argument.name_id)};
                if (parameter_type.a == kTypeResourceSampler) {
                    const auto sampler = samplers_.find(name);
                    if (sampler == samplers_.end()) return fail("FEIR callable sampler argument is unknown"), nullptr;
                    values.push_back(sampler->second.filter);
                    values.push_back(sampler->second.address);
                } else {
                    const auto resource = resources_.find(name);
                    if (resource == resources_.end()) return fail("FEIR callable resource argument is unknown"), nullptr;
                    values.push_back(resource->second.argument);
                }
            } else {
                if (parameter.direction == 0u) {
                    auto* value = lower_expression(expression_id);
                    if (value == nullptr) return nullptr;
                    values.push_back(value);
                } else {
                    auto target = expression_address(expression_id);
                    if (!target) return fail("FEIR callable reference argument is not assignable"), nullptr;
                    if (target->pointer != nullptr && target->indices.empty()) {
                        values.push_back(target->pointer);
                    } else {
                        auto* value_type = type(parameter.type_id);
                        auto* temporary = builder_.alloca_local(value_type);
                        if (parameter.direction == 2u) {
                            auto* initial = read_address(*target, value_type);
                            if (initial == nullptr) return nullptr;
                            builder_.store(temporary, initial);
                        }
                        values.push_back(temporary);
                        writebacks.push_back(Writeback{std::move(*target), temporary, value_type});
                    }
                }
            }
        }
        auto* result = builder_.call(result_type, found->second.function, values);
        for (const auto& writeback : writebacks) {
            auto* value = builder_.load(writeback.type, writeback.temporary);
            if (!write_address(writeback.address, writeback.type, value)) return nullptr;
        }
        return result;
    }

    Value* lower_texture_sample(const TypedIR::Expression& expression, const Type* result_type) {
        const uint32_t expected_count = expression.op == 0u ? 3u : expression.op == 1u ? 4u :
                                        expression.op == 2u ? 5u : 0u;
        if (expected_count == 0u || expression.argument_count != expected_count || !argument_range(expression))
            return fail("FEIR texture sample has an invalid operation or argument range"), nullptr;
        const auto texture_id = module_.arguments[expression.first_argument];
        const auto sampler_id = module_.arguments[expression.first_argument + 1u];
        if (texture_id >= module_.expressions.size() || sampler_id >= module_.expressions.size())
            return fail("FEIR texture sample resource reference is out of range"), nullptr;
        const auto& texture_expression = module_.expressions[texture_id];
        const auto& sampler_expression = module_.expressions[sampler_id];
        if ((texture_expression.kind != kExpressionLocal && texture_expression.kind != kExpressionParameter) ||
            (sampler_expression.kind != kExpressionLocal && sampler_expression.kind != kExpressionParameter))
            return fail("FEIR texture sampling requires direct resource references"), nullptr;
        const auto texture = resources_.find(std::string{string(texture_expression.name_id)});
        const auto sampler = samplers_.find(std::string{string(sampler_expression.name_id)});
        if (texture == resources_.end() || texture->second.kind != kResourceTexture2D || sampler == samplers_.end())
            return fail("FEIR texture sample references an unknown sampled texture or sampler"), nullptr;

        auto* uv = lower_expression(module_.arguments[expression.first_argument + 2u]);
        if (uv == nullptr || uv->type() != Type::vector(Type::of<float>(), 2u))
            return fail("FEIR texture sample requires float2 coordinates"), nullptr;
        std::vector<Value*> operands{texture->second.argument, uv};
        ResourceQueryOp op = ResourceQueryOp::TEXTURE2D_SAMPLE;
        if (expression.op == 1u) {
            auto* lod = lower_expression(module_.arguments[expression.first_argument + 3u]);
            if (lod == nullptr || lod->type() != Type::of<float>())
                return fail("FEIR texture SampleLevel requires a float LOD"), nullptr;
            operands.push_back(lod);
            op = ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL;
        } else if (expression.op == 2u) {
            auto* ddx = lower_expression(module_.arguments[expression.first_argument + 3u]);
            auto* ddy = lower_expression(module_.arguments[expression.first_argument + 4u]);
            if (ddx == nullptr || ddy == nullptr || ddx->type() != uv->type() || ddy->type() != uv->type())
                return fail("FEIR texture SampleGrad requires float2 gradients"), nullptr;
            operands.push_back(ddx);
            operands.push_back(ddy);
            op = ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD;
        }
        operands.push_back(sampler->second.filter);
        operands.push_back(sampler->second.address);
        auto* sampled = builder_.call(Type::vector(Type::of<float>(), 4u), op, operands);
        return sampled->type() == result_type ? sampled : builder_.bit_cast_if_necessary(result_type, sampled);
    }

    Value* lower_fragment_derivative(uint32_t expression_id, const Type* result_type, bool along_x) {
        if (fragment_callable_ != nullptr) {
            if (fragment_parameter_name_.empty() || fragment_parameter_type_ == nullptr) {
                return fail("fused fragment derivatives require a varying parameter"), nullptr;
            }
            auto evaluate = [&](Value* varying) -> Value* {
                auto* temporary = builder_.alloca_local(fragment_parameter_type_);
                builder_.store(temporary, varying);
                auto found = locals_.find(fragment_parameter_name_);
                if (found == locals_.end()) return nullptr;
                auto* saved = found->second;
                found->second = temporary;
                auto* value = lower_expression(expression_id);
                found->second = saved;
                return value;
            };
            const auto offset = along_x ? 0u : 2u;
            auto* first = evaluate(fragment_neighbor_arguments_[offset]);
            auto* second = evaluate(fragment_neighbor_arguments_[offset + 1u]);
            if (first == nullptr || second == nullptr || first->type() != result_type ||
                second->type() != result_type) {
                return fail("fused fragment derivative expression has an unsupported type"), nullptr;
            }
            return builder_.call(result_type, ArithmeticOp::BINARY_SUB, {second, first});
        }
        if (stage_input_ == nullptr || fragment_parameter_name_.empty() || fragment_parameter_type_ == nullptr) {
            return fail("fragment derivatives require a varying input"), nullptr;
        }
        auto* dispatch = xir_module_.create_dispatch_id();
        auto* dispatch_size = xir_module_.create_dispatch_size();
        auto* x = extract(dispatch, Type::of<uint32_t>(), {index_constant(0u)});
        auto* y = extract(dispatch, Type::of<uint32_t>(), {index_constant(1u)});
        auto* width = extract(dispatch_size, Type::of<uint32_t>(), {index_constant(0u)});
        auto* height = extract(dispatch_size, Type::of<uint32_t>(), {index_constant(1u)});
        auto* two = index_constant(2u);
        auto* one = index_constant(1u);
        auto* axis = along_x ? x : y;
        auto* extent = along_x ? width : height;
        auto* pair = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                                   {builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {axis, two}), two});
        auto* extent_last = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_SUB, {extent, one});
        auto* pair_end = builder_.call(Type::of<uint32_t>(), ArithmeticOp::MIN,
                                       {builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {pair, one}),
                                        extent_last});
        auto evaluate = [&](Value* sample_x, Value* sample_y) -> Value* {
            auto* row = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {sample_y, width});
            auto* index = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {row, sample_x});
            if (inputs_.graphics_sample_count > 1u) {
                index = builder_.call(
                    Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD,
                    {builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL,
                                   {index, index_constant(inputs_.graphics_sample_count)}),
                     index_constant(inputs_.graphics_sample_index)});
            }
            auto* varying = builder_.call(fragment_parameter_type_, ResourceReadOp::BUFFER_READ,
                                          {stage_input_->argument, index});
            auto* temporary = builder_.alloca_local(fragment_parameter_type_);
            builder_.store(temporary, varying);
            auto found = locals_.find(fragment_parameter_name_);
            if (found == locals_.end()) return nullptr;
            auto* saved = found->second;
            found->second = temporary;
            auto* value = lower_expression(expression_id);
            found->second = saved;
            return value;
        };
        auto* first = evaluate(along_x ? pair : x, along_x ? y : pair);
        auto* second = evaluate(along_x ? pair_end : x, along_x ? y : pair_end);
        if (first == nullptr || second == nullptr || first->type() != result_type || second->type() != result_type) {
            return fail("fragment derivative expression has an unsupported type"), nullptr;
        }
        return builder_.call(result_type, ArithmeticOp::BINARY_SUB, {second, first});
    }

    Value* shared_index(Value* logical_index, uint32_t length) {
        if (logical_groups_per_block_ == 1u) return logical_index;
        auto* physical_x = extract(xir_module_.create_thread_id(), Type::of<uint32_t>(), {index_constant(0u)});
        const auto group_x = static_cast<uint32_t>(inputs_.group_x);
        auto* group_width = xir_module_.create_constant(Type::of<uint32_t>(), &group_x);
        auto* subgroup = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_DIV, {physical_x, group_width});
        auto* stride = xir_module_.create_constant(Type::of<uint32_t>(), &length);
        auto* offset = builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_MUL, {subgroup, stride});
        auto* index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), logical_index);
        return builder_.call(Type::of<uint32_t>(), ArithmeticOp::BINARY_ADD, {offset, index});
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
        const auto name = string(expression.name_id);
        auto op = intrinsic_op(name);
        if (name.ends_with(".Ddx") || name.ends_with(".Ddy")) {
            if (!argument_range(expression) || expression.argument_count != 1u) {
                return fail("fragment derivative intrinsic requires one argument"), nullptr;
            }
            return lower_fragment_derivative(module_.arguments[expression.first_argument], result_type,
                                             name.ends_with(".Ddx"));
        }
        auto values = arguments(expression);
        if (!op || values.empty()) return fail("unsupported FEIR intrinsic '" + std::string{string(expression.name_id)} + "'"), nullptr;
        // Keep domain-invalid logarithm sentinels as runtime values. LC's Metal
        // AST builder folds a literal Log(negative) to NaN while emitting the
        // shader, and Metal's literal printer rejects that non-finite constant.
        // Adding a dynamic zero preserves the runtime result (including NaN for
        // invalid labels) without making the compiler materialize NaN in MSL.
        if (*op == ArithmeticOp::LOG && values.front()->type()->is_scalar() &&
            values.front()->type()->is_float() && expression.argument_count != 0u &&
            expression.first_argument != TypedIR::NoIndex &&
            expression.first_argument < module_.arguments.size() &&
            module_.arguments[expression.first_argument] < module_.expressions.size()) {
            const auto& argument = module_.expressions[module_.arguments[expression.first_argument]];
            if (argument.kind == kExpressionLiteral) {
                auto text = std::string{string(argument.name_id)};
                char* end = nullptr;
                const auto literal_value = std::strtof(text.c_str(), &end);
                if (end != text.c_str() && literal_value < 0.0f) {
                    auto* dispatch = xir_module_.create_dispatch_id();
                    auto* lane = extract(dispatch, Type::of<uint32_t>(), {index_constant(0u)});
                    auto* dynamic_lane = builder_.static_cast_if_necessary(Type::of<float>(), lane);
                    auto* zero = xir_module_.create_constant_zero(Type::of<float>());
                    auto* dynamic_zero = builder_.call(Type::of<float>(), ArithmeticOp::BINARY_MUL,
                                                       {dynamic_lane, zero});
                    values.front() = builder_.call(Type::of<float>(), ArithmeticOp::BINARY_ADD,
                                                   {values.front(), dynamic_zero});
                }
            }
        }
        if ((*op == ArithmeticOp::MIN || *op == ArithmeticOp::MAX || *op == ArithmeticOp::CLAMP) && result_type->is_vector()) {
            for (size_t i = 1; i < values.size(); ++i)
                if (values[i]->type()->is_scalar()) {
                    std::vector<Value*> splat(result_type->dimension(), values[i]);
                    values[i] = builder_.call(result_type, ArithmeticOp::AGGREGATE, splat);
                }
        }
        if (*op == ArithmeticOp::MATRIX_LINALG_MUL && values.size() == 2u) {
            if (expression.first_argument == TypedIR::NoIndex || expression.first_argument > module_.arguments.size() ||
                expression.argument_count > module_.arguments.size() - expression.first_argument)
                return fail("matrix multiplication has an invalid FEIR argument range"), nullptr;
            const auto left_id = module_.arguments[expression.first_argument];
            const auto right_id = module_.arguments[expression.first_argument + 1u];
            if (left_id >= module_.expressions.size() || right_id >= module_.expressions.size())
                return fail("matrix multiplication has an invalid FEIR argument"), nullptr;
            const auto left_matrix = is_feir_matrix_type(module_.expressions[left_id].type_id);
            const auto right_matrix = is_feir_matrix_type(module_.expressions[right_id].type_id);
            const auto dimension = left_matrix ? feir_matrix_dimension(module_.expressions[left_id].type_id)
                                               : feir_matrix_dimension(module_.expressions[right_id].type_id);
            return matrix_multiply(values[0], values[1], result_type, left_matrix, right_matrix, dimension);
        }
        return builder_.call(result_type, *op, values);
    }

    std::optional<Address> address(uint32_t id) {
        if (id >= module_.lvalues.size()) return fail("FEIR l-value index is out of range"), std::nullopt;
        const auto& lvalue = module_.lvalues[id];
        if (lvalue.kind == kLValueLocal || lvalue.kind == kLValueParameter) {
            const auto name = std::string{string(lvalue.name_id)};
            if (auto resource = resources_.find(name); resource != resources_.end())
                return Address{.resource = &resource->second, .root_type = resource->second.element_type};
            auto found = locals_.find(name);
            if (found == locals_.end()) return fail("FEIR l-value names an unknown local"), std::nullopt;
            return Address{.pointer = found->second, .root_type = found->second->type()};
        }
        if (lvalue.kind == kLValueResourceElement) {
            auto found = resources_.find(std::string{string(lvalue.name_id)});
            auto* index = lower_expression(lvalue.a);
            if (found == resources_.end() || index == nullptr) return fail("invalid FEIR buffer l-value"), std::nullopt;
            if (found->second.kind == kResourceTexture2D || found->second.kind == kResourceTexture3D) {
                const auto dimension = found->second.kind == kResourceTexture2D ? 2u : 3u;
                index = convert(index, Type::vector(Type::of<uint32_t>(), dimension));
            } else index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
            return Address{.resource = &found->second, .resource_index = index,
                           .resource_index_expression = lvalue.a,
                           .root_type = found->second.element_type};
        }
        if (lvalue.kind == kLValueSharedMemoryElement) {
            auto found = shared_.find(std::string{string(lvalue.name_id)});
            auto* index = lower_expression(lvalue.a);
            if (found == shared_.end() || index == nullptr) return fail("invalid FEIR shared-memory l-value"), std::nullopt;
            auto* result_type = type(lvalue.type_id);
            index = shared_index(index, found->second.length);
            return Address{.pointer = builder_.gep(result_type, found->second.pointer, {index}), .root_type = result_type};
        }
        if (lvalue.kind == kLValueField || lvalue.kind == kLValueMemberAccess || lvalue.kind == kLValueIndexAccess) {
            auto base = address(lvalue.a);
            if (!base) return std::nullopt;
            if (lvalue.kind == kLValueIndexAccess) {
                auto* index = lower_expression(lvalue.b);
                if (index == nullptr) return std::nullopt;
                if (base->resource != nullptr && base->resource_index == nullptr) {
                    if (base->resource->kind == kResourceTexture2D || base->resource->kind == kResourceTexture3D) {
                        const auto dimension = base->resource->kind == kResourceTexture2D ? 2u : 3u;
                        index = convert(index, Type::vector(Type::of<uint32_t>(), dimension));
                    } else index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
                    base->resource_index = index;
                    base->resource_index_expression = lvalue.b;
                } else base->indices.push_back(index);
            } else {
                auto field = field_index(module_.lvalues[lvalue.a].type_id, string(lvalue.name_id));
                if (!field) return fail("invalid FEIR aggregate l-value member"), std::nullopt;
                base->indices.push_back(index_constant(*field));
            }
            return base;
        }
        return fail("unsupported FEIR l-value kind " + std::to_string(lvalue.kind)), std::nullopt;
    }

    std::optional<Address> expression_address(uint32_t id) {
        if (id >= module_.expressions.size()) return std::nullopt;
        const auto& expression = module_.expressions[id];
        if (expression.kind == kExpressionLocal || expression.kind == kExpressionParameter) {
            const auto name = std::string{string(expression.name_id)};
            auto found = locals_.find(name);
            return found == locals_.end() ? std::nullopt
                                          : std::optional{Address{.pointer = found->second,
                                                                  .root_type = found->second->type()}};
        }
        if (expression.kind == kExpressionResourceElement) {
            auto found = resources_.find(std::string{string(expression.name_id)});
            auto* index = lower_expression(expression.a);
            if (found == resources_.end() || index == nullptr) return std::nullopt;
            index = builder_.static_cast_if_necessary(Type::of<uint32_t>(), index);
            return Address{.resource = &found->second, .resource_index = index,
                           .resource_index_expression = expression.a,
                           .root_type = found->second.element_type};
        }
        if (expression.kind == kExpressionField || expression.kind == kExpressionMemberAccess ||
            expression.kind == kExpressionIndexAccess || expression.kind == kExpressionMatrixColumn) {
            auto base = expression_address(expression.a);
            if (!base) return std::nullopt;
            if (expression.kind == kExpressionIndexAccess || expression.kind == kExpressionMatrixColumn) {
                auto* index = lower_expression(expression.b);
                if (index == nullptr) return std::nullopt;
                base->indices.push_back(index);
            } else {
                auto field = field_index(module_.expressions[expression.a].type_id, string(expression.name_id));
                if (!field) return std::nullopt;
                base->indices.push_back(index_constant(*field));
            }
            return base;
        }
        return std::nullopt;
    }

    Value* read_address(const Address& address, const Type* result_type) {
        if (address.pointer != nullptr) {
            auto* pointer = address.indices.empty() ? address.pointer
                                                    : builder_.gep(result_type, address.pointer, address.indices);
            return builder_.load(result_type, pointer);
        }
        if (address.resource_index == nullptr) return fail("FEIR resource l-value is missing an element index"), nullptr;
        Value* root = nullptr;
        if (address.resource->kind == kResourceTexture2D || address.resource->kind == kResourceTexture3D) {
            root = builder_.call(address.root_type,
                                 address.resource->kind == kResourceTexture2D ? ResourceReadOp::TEXTURE2D_READ
                                                                             : ResourceReadOp::TEXTURE3D_READ,
                                 {address.resource->argument, address.resource_index});
        } else root = track_ad_read(*address.resource, address.resource_index,
                                    builder_.call(address.root_type, ResourceReadOp::BUFFER_READ,
                                                  {address.resource->argument, address.resource_index}),
                                    address.resource_index_expression);
        return address.indices.empty() ? root : extract(root, result_type, address.indices);
    }

    bool write_address(const Address& address, const Type* value_type, Value* value) {
        if (address.pointer != nullptr) {
            auto* pointer = address.indices.empty() ? address.pointer
                                                    : builder_.gep(value_type, address.pointer, address.indices);
            builder_.store(pointer, value);
            return true;
        }
        if (address.resource_index == nullptr) return fail("FEIR resource l-value is missing an element index");
        if (address.resource->access != kAccessWrite && address.resource->access != kAccessReadWrite)
            return fail("FEIR writes a read-only buffer");
        if (address.resource->kind == kResourceTexture2D || address.resource->kind == kResourceTexture3D) {
            if (!address.indices.empty()) return fail("nested texture-element l-values are unsupported");
            builder_.call(address.resource->kind == kResourceTexture2D ? ResourceWriteOp::TEXTURE2D_WRITE
                                                                       : ResourceWriteOp::TEXTURE3D_WRITE,
                          {address.resource->argument, address.resource_index, value});
            return true;
        }
        if (!address.indices.empty()) {
            auto* root = track_ad_read(*address.resource, address.resource_index,
                                       builder_.call(address.root_type, ResourceReadOp::BUFFER_READ,
                                                     {address.resource->argument, address.resource_index}),
                                       address.resource_index_expression);
            std::vector<Value*> operands{root, value};
            operands.insert(operands.end(), address.indices.begin(), address.indices.end());
            value = builder_.call(address.root_type, ArithmeticOp::INSERT, operands);
        }
        if (inside_ad_scope_) {
            if (auto* slot = ad_local_slot(*address.resource, address.resource_index,
                                           address.resource_index_expression, nullptr);
                slot != nullptr) builder_.store(slot, value);
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

    bool is_integer_type(uint32_t type_id) const {
        return type_id < module_.types.size() && module_.types[type_id].kind == kTypePrimitive &&
               module_.types[type_id].b == 32u &&
               (module_.types[type_id].a == 1u || module_.types[type_id].a == 2u);
    }

    static std::optional<int64_t> checked_add(int64_t left, int64_t right) {
        if ((right > 0 && left > std::numeric_limits<int64_t>::max() - right) ||
            (right < 0 && left < std::numeric_limits<int64_t>::min() - right)) return std::nullopt;
        return left + right;
    }

    static std::optional<int64_t> checked_subtract(int64_t left, int64_t right) {
        if ((right < 0 && left > std::numeric_limits<int64_t>::max() + right) ||
            (right > 0 && left < std::numeric_limits<int64_t>::min() + right)) return std::nullopt;
        return left - right;
    }

    static std::optional<int64_t> checked_multiply(int64_t left, int64_t right) {
        if (left == 0 || right == 0) return 0;
        if ((left == -1 && right == std::numeric_limits<int64_t>::min()) ||
            (right == -1 && left == std::numeric_limits<int64_t>::min())) return std::nullopt;
        if (left > 0) {
            if ((right > 0 && left > std::numeric_limits<int64_t>::max() / right) ||
                (right < 0 && right < std::numeric_limits<int64_t>::min() / left)) return std::nullopt;
        } else {
            if ((right > 0 && left < std::numeric_limits<int64_t>::min() / right) ||
                (right < 0 && left < std::numeric_limits<int64_t>::max() / right)) return std::nullopt;
        }
        return left * right;
    }

    std::optional<int64_t> integer_expression(uint32_t id) const {
        if (id >= module_.expressions.size()) return std::nullopt;
        const auto& expression = module_.expressions[id];
        if (!is_integer_type(expression.type_id) && expression.kind != kExpressionConversion) return std::nullopt;
        switch (expression.kind) {
        case kExpressionLiteral: {
            auto text = string(expression.name_id);
            if (!text.empty() && (text.back() == 'u' || text.back() == 'U')) text.remove_suffix(1u);
            int64_t value{};
            auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
            return ec == std::errc{} && end == text.data() + text.size() ? std::optional{value} : std::nullopt;
        }
        case kExpressionLocal:
        case kExpressionParameter: {
            const auto found = constant_integers_.find(std::string{string(expression.name_id)});
            return found == constant_integers_.end() ? std::nullopt : std::optional{found->second};
        }
        case kExpressionPushConstant: {
            if (inputs_.dynamic_push_constants || !is_integer_type(expression.type_id)) return std::nullopt;
            const TypedIR::PushConstantInfo* found = nullptr;
            for (const auto& push : inputs_.push_constants)
                if (push.binding == expression.op) found = &push;
            if (found == nullptr || found->data == nullptr || found->size < sizeof(uint32_t)) return std::nullopt;
            if (module_.types[expression.type_id].a == 1u) {
                int32_t value{};
                std::memcpy(&value, found->data, sizeof(value));
                return value;
            }
            uint32_t value{};
            std::memcpy(&value, found->data, sizeof(value));
            return value;
        }
        case kExpressionUnary: {
            auto value = integer_expression(expression.a);
            if (!value) return std::nullopt;
            if (expression.op == 0u) return checked_subtract(0, *value);
            if (expression.op == 2u) return ~*value;
            return std::nullopt;
        }
        case kExpressionBinary: {
            auto left = integer_expression(expression.a);
            auto right = integer_expression(expression.b);
            if (!left || !right) return std::nullopt;
            switch (expression.op) {
            case 0u: return checked_add(*left, *right);
            case 1u: return checked_subtract(*left, *right);
            case 2u: return checked_multiply(*left, *right);
            case 3u:
                if (*right == 0 || (*left == std::numeric_limits<int64_t>::min() && *right == -1))
                    return std::nullopt;
                return *left / *right;
            case 4u:
                if (*right == 0 || (*left == std::numeric_limits<int64_t>::min() && *right == -1))
                    return std::nullopt;
                return *left % *right;
            default: return std::nullopt;
            }
        }
        case kExpressionConversion:
            return integer_expression(expression.a);
        default:
            return std::nullopt;
        }
    }

    static bool compare_integers(uint32_t op, int64_t left, int64_t right) {
        switch (op) {
        case 0u: return left == right;
        case 1u: return left != right;
        case 2u: return left < right;
        case 3u: return left <= right;
        case 4u: return left > right;
        case 5u: return left >= right;
        default: return false;
        }
    }

    const TypedIR::Statement* single_statement(uint32_t id) const {
        if (id >= module_.statements.size()) return nullptr;
        const auto* statement = &module_.statements[id];
        while (statement->kind == kStatementBlock) {
            if (statement->child_count != 1u || statement->first_child == TypedIR::NoIndex ||
                statement->first_child >= module_.children.size()) return nullptr;
            id = module_.children[statement->first_child];
            if (id >= module_.statements.size()) return nullptr;
            statement = &module_.statements[id];
        }
        return statement;
    }

    std::optional<std::string> direct_local_lvalue(uint32_t id) const {
        if (id >= module_.lvalues.size()) return std::nullopt;
        const auto& lvalue = module_.lvalues[id];
        if (lvalue.kind != kLValueLocal && lvalue.kind != kLValueParameter) return std::nullopt;
        auto name = std::string{string(lvalue.name_id)};
        return name.empty() ? std::nullopt : std::optional{std::move(name)};
    }

    bool expression_references_local(uint32_t id, std::string_view name) const {
        if (id >= module_.expressions.size()) return false;
        const auto& expression = module_.expressions[id];
        if ((expression.kind == kExpressionLocal || expression.kind == kExpressionParameter) &&
            string(expression.name_id) == name) return true;
        if (expression.a != TypedIR::NoIndex && expression_references_local(expression.a, name)) return true;
        if (expression.b != TypedIR::NoIndex && expression_references_local(expression.b, name)) return true;
        if (expression.c != TypedIR::NoIndex && expression_references_local(expression.c, name)) return true;
        if (expression.first_argument != TypedIR::NoIndex &&
            expression.first_argument <= module_.arguments.size() &&
            expression.argument_count <= module_.arguments.size() - expression.first_argument) {
            for (uint32_t i = 0u; i < expression.argument_count; ++i)
                if (expression_references_local(module_.arguments[expression.first_argument + i], name)) return true;
        }
        return false;
    }

    bool statement_contains_loop_control(uint32_t id) const {
        if (id >= module_.statements.size()) return true;
        const auto& statement = module_.statements[id];
        if (statement.kind == kStatementBreak || statement.kind == kStatementContinue) return true;
        auto contains = [&](uint32_t child) {
            return child != TypedIR::NoIndex && statement_contains_loop_control(child);
        };
        switch (statement.kind) {
        case kStatementBlock:
            if (statement.child_count != 0u &&
                (statement.first_child == TypedIR::NoIndex || statement.first_child > module_.children.size() ||
                 statement.child_count > module_.children.size() - statement.first_child)) return true;
            for (uint32_t i = 0u; i < statement.child_count; ++i)
                if (statement_contains_loop_control(module_.children[statement.first_child + i])) return true;
            return false;
        case kStatementIf: return contains(statement.b) || contains(statement.c);
        case kStatementFor: return contains(statement.a) || contains(statement.c) || contains(statement.op);
        case kStatementWhile: return contains(statement.b);
        case kStatementDoWhile: return contains(statement.a);
        default: return false;
        }
    }

    bool statement_writes_local(uint32_t id, std::string_view name) const {
        if (id >= module_.statements.size()) return true;
        const auto& statement = module_.statements[id];
        if (statement.kind == kStatementLocalDeclaration && string(statement.name_id) == name) return true;
        if (statement.kind == kStatementAssignment || statement.kind == kStatementCompoundAssignment ||
            statement.kind == kStatementIncrementDecrement) {
            auto target = direct_local_lvalue(statement.a);
            if (target && *target == name) return true;
        }
        auto writes = [&](uint32_t child) {
            return child != TypedIR::NoIndex && statement_writes_local(child, name);
        };
        switch (statement.kind) {
        case kStatementBlock:
            if (statement.child_count != 0u &&
                (statement.first_child == TypedIR::NoIndex || statement.first_child > module_.children.size() ||
                 statement.child_count > module_.children.size() - statement.first_child)) return true;
            for (uint32_t i = 0u; i < statement.child_count; ++i)
                if (statement_writes_local(module_.children[statement.first_child + i], name)) return true;
            return false;
        case kStatementIf: return writes(statement.b) || writes(statement.c);
        case kStatementFor: return writes(statement.a) || writes(statement.c) || writes(statement.op);
        case kStatementWhile: return writes(statement.b);
        case kStatementDoWhile: return writes(statement.a);
        default: return false;
        }
    }

    std::optional<int64_t> for_step(const TypedIR::Statement& statement, std::string_view induction) const {
        const auto* step = single_statement(statement.c);
        if (step == nullptr) return std::nullopt;
        auto target = direct_local_lvalue(step->a);
        if (!target || *target != induction) return std::nullopt;
        if (step->kind == kStatementIncrementDecrement)
            return (step->op & 1u) != 0u ? 1 : -1;
        if (step->kind == kStatementCompoundAssignment && (step->op == 0u || step->op == 1u)) {
            auto amount = integer_expression(step->b);
            if (!amount) return std::nullopt;
            return step->op == 0u ? amount : checked_subtract(0, *amount);
        }
        if (step->kind != kStatementAssignment || step->b >= module_.expressions.size()) return std::nullopt;
        const auto& value = module_.expressions[step->b];
        if (value.kind != kExpressionBinary || (value.op != 0u && value.op != 1u)) return std::nullopt;
        const auto references = [&](uint32_t id) {
            return id < module_.expressions.size() &&
                   (module_.expressions[id].kind == kExpressionLocal ||
                    module_.expressions[id].kind == kExpressionParameter) &&
                   string(module_.expressions[id].name_id) == induction;
        };
        if (references(value.a)) {
            auto amount = integer_expression(value.b);
            if (!amount) return std::nullopt;
            return value.op == 0u ? amount : checked_subtract(0, *amount);
        }
        if (value.op == 0u && references(value.b)) return integer_expression(value.a);
        return std::nullopt;
    }

    struct StaticForLoop { uint32_t trip_count = 0u; };

    std::optional<StaticForLoop> analyze_static_for(const TypedIR::Statement& statement) const {
        if (statement.a == TypedIR::NoIndex || statement.b == TypedIR::NoIndex ||
            statement.c == TypedIR::NoIndex || statement.op == TypedIR::NoIndex ||
            statement_contains_loop_control(statement.op)) return std::nullopt;
        const auto* initializer = single_statement(statement.a);
        if (initializer == nullptr) return std::nullopt;
        std::string induction;
        uint32_t induction_type = TypedIR::NoIndex;
        uint32_t initial_expression = TypedIR::NoIndex;
        if (initializer->kind == kStatementLocalDeclaration) {
            induction = std::string{string(initializer->name_id)};
            induction_type = initializer->op;
            initial_expression = initializer->a;
        } else if (initializer->kind == kStatementAssignment) {
            auto target = direct_local_lvalue(initializer->a);
            if (!target) return std::nullopt;
            induction = std::move(*target);
            induction_type = module_.lvalues[initializer->a].type_id;
            initial_expression = initializer->b;
        } else return std::nullopt;
        if (induction.empty() || !is_integer_type(induction_type) ||
            initial_expression == TypedIR::NoIndex || statement_writes_local(statement.op, induction))
            return std::nullopt;
        auto initial = integer_expression(initial_expression);
        auto step = for_step(statement, induction);
        if (!initial || !step || *step == 0 || statement.b >= module_.expressions.size()) return std::nullopt;
        const auto& condition = module_.expressions[statement.b];
        if (condition.kind != kExpressionComparison || condition.op > 5u) return std::nullopt;
        const auto is_induction = [&](uint32_t id) {
            return id < module_.expressions.size() &&
                   (module_.expressions[id].kind == kExpressionLocal ||
                    module_.expressions[id].kind == kExpressionParameter) &&
                   string(module_.expressions[id].name_id) == induction;
        };
        const auto induction_left = is_induction(condition.a);
        const auto induction_right = is_induction(condition.b);
        if (induction_left == induction_right) return std::nullopt;
        const auto bound_id = induction_left ? condition.b : condition.a;
        if (expression_references_local(bound_id, induction)) return std::nullopt;
        auto bound = integer_expression(bound_id);
        if (!bound) return std::nullopt;
        const auto unsigned_induction = module_.types[induction_type].a == 2u;
        auto in_range = [&](int64_t value) {
            return unsigned_induction ? value >= 0 && value <= std::numeric_limits<uint32_t>::max()
                                      : value >= std::numeric_limits<int32_t>::min() &&
                                            value <= std::numeric_limits<int32_t>::max();
        };
        if (!in_range(*initial)) return std::nullopt;
        auto value = *initial;
        for (uint32_t trips = 0u; trips <= 64u; ++trips) {
            const auto condition_true = induction_left
                                            ? compare_integers(condition.op, value, *bound)
                                            : compare_integers(condition.op, *bound, value);
            if (!condition_true) return StaticForLoop{trips};
            if (trips == 64u) return std::nullopt;
            auto next = checked_add(value, *step);
            if (!next || !in_range(*next)) return std::nullopt;
            value = *next;
        }
        return std::nullopt;
    }

    void update_constant_integer(const TypedIR::Statement& statement,
                                 std::optional<int64_t> value = std::nullopt) {
        if (ad_inputs_ == nullptr) return;
        std::optional<std::string> name;
        if (statement.kind == kStatementLocalDeclaration) {
            auto local = std::string{string(statement.name_id)};
            if (!local.empty()) name = std::move(local);
        } else if (statement.kind == kStatementAssignment ||
                   statement.kind == kStatementCompoundAssignment ||
                   statement.kind == kStatementIncrementDecrement) {
            name = direct_local_lvalue(statement.a);
        }
        if (!name) return;
        if (value) constant_integers_[*name] = *value;
        else constant_integers_.erase(*name);
    }

    std::optional<int64_t> assigned_integer_value(const TypedIR::Statement& statement) const {
        if (ad_inputs_ == nullptr) return std::nullopt;
        if (statement.kind == kStatementLocalDeclaration)
            return is_integer_type(statement.op) && statement.a != TypedIR::NoIndex
                       ? integer_expression(statement.a)
                       : std::nullopt;
        auto name = direct_local_lvalue(statement.a);
        if (!name || statement.a >= module_.lvalues.size() ||
            !is_integer_type(module_.lvalues[statement.a].type_id)) return std::nullopt;
        if (statement.kind == kStatementAssignment) return integer_expression(statement.b);
        const auto old = constant_integers_.find(*name);
        if (old == constant_integers_.end()) return std::nullopt;
        if (statement.kind == kStatementIncrementDecrement)
            return (statement.op & 1u) != 0u ? checked_add(old->second, 1) : checked_subtract(old->second, 1);
        if (statement.kind != kStatementCompoundAssignment) return std::nullopt;
        auto right = integer_expression(statement.b);
        if (!right) return std::nullopt;
        switch (statement.op) {
        case 0u: return checked_add(old->second, *right);
        case 1u: return checked_subtract(old->second, *right);
        case 2u: return checked_multiply(old->second, *right);
        case 3u:
            if (*right == 0 || (old->second == std::numeric_limits<int64_t>::min() && *right == -1))
                return std::nullopt;
            return old->second / *right;
        case 4u:
            if (*right == 0 || (old->second == std::numeric_limits<int64_t>::min() && *right == -1))
                return std::nullopt;
            return old->second % *right;
        default: return std::nullopt;
        }
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
            auto constant = assigned_integer_value(statement);
            auto* local = builder_.alloca_local(local_type);
            local->set_name(name);
            locals_[name] = local;
            if (statement.a != TypedIR::NoIndex) {
                auto* initial = lower_expression(statement.a);
                if (initial == nullptr) return false;
                builder_.store(local, initial);
            }
            update_constant_integer(statement, constant);
            return true;
        }
        case kStatementSharedMemoryDeclaration: {
            auto name = std::string{string(statement.name_id)};
            auto* element = type(statement.op);
            if (name.empty() || element == nullptr || statement.a == 0u) return fail("invalid FEIR shared declaration");
            auto* memory = builder_.alloca_shared(Type::array(element, statement.a * logical_groups_per_block_));
            memory->set_name(name);
            shared_[name] = SharedMemory{memory, statement.a};
            uses_group_semantics_ = true;
            return true;
        }
        case kStatementAssignment:
        case kStatementCompoundAssignment:
        case kStatementIncrementDecrement: {
            auto constant = assigned_integer_value(statement);
            auto target = address(statement.a);
            if (!target) return false;
            auto* value_type = type(module_.lvalues[statement.a].type_id);
            if (value_type == nullptr && target->resource != nullptr && target->indices.empty() &&
                (target->resource->kind == kResourceTexture2D || target->resource->kind == kResourceTexture3D)) {
                // Luisa textures are represented as float4 even when Feather exposes an RGBA
                // struct with byte channels. A whole texel assignment therefore uses the
                // resource's mapped element type rather than the host-side struct layout.
                value_type = target->root_type;
            }
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
                const auto left_matrix = is_feir_matrix_type(module_.lvalues[statement.a].type_id);
                const auto right_matrix = statement.b < module_.expressions.size() &&
                                          is_feir_matrix_type(module_.expressions[statement.b].type_id);
                if (*op == ArithmeticOp::BINARY_MUL && (left_matrix || right_matrix)) {
                    const auto dimension = left_matrix
                                               ? feir_matrix_dimension(module_.lvalues[statement.a].type_id)
                                               : feir_matrix_dimension(module_.expressions[statement.b].type_id);
                    value = matrix_multiply(old, right, value_type, left_matrix, right_matrix, dimension);
                } else if (is_shift_op(*op)) value = builder_.call(value_type, *op, {old, right});
                else value = builder_.call(value_type, *op, {splat_to(old, value_type), splat_to(right, value_type)});
            }
            if (value == nullptr || !write_address(*target, value_type, value)) return false;
            update_constant_integer(statement, constant);
            return true;
        }
        case kStatementIf:
            return lower_if(statement);
        case kStatementFor: {
            auto unrolled = ad_inputs_ == nullptr ? std::nullopt : analyze_static_for(statement);
            if (statement.a != TypedIR::NoIndex && !lower_statement(statement.a)) return false;
            if (unrolled) {
                for (uint32_t i = 0u; i < unrolled->trip_count; ++i) {
                    if (!lower_statement(statement.op)) return false;
                    if (builder_.is_insertion_point_terminator()) return true;
                    if (!lower_statement(statement.c)) return false;
                }
                return true;
            }
            auto result = lower_loop(statement.op, statement.b, statement.c, false);
            if (ad_inputs_ != nullptr) constant_integers_.clear();
            return result;
        }
        case kStatementWhile: {
            auto result = lower_loop(statement.b, statement.a, TypedIR::NoIndex, false);
            if (ad_inputs_ != nullptr) constant_integers_.clear();
            return result;
        }
        case kStatementDoWhile: {
            auto result = lower_loop(statement.a, statement.b, TypedIR::NoIndex, true);
            if (ad_inputs_ != nullptr) constant_integers_.clear();
            return result;
        }
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
                if (stage_output_ != nullptr) {
                    if (!stage_color_outputs_.empty()) {
                        Value* coordinate = fragment_coordinate_argument_;
                        if (coordinate == nullptr) {
                            auto* dispatch = xir_module_.create_dispatch_id();
                            auto* x = extract(dispatch, Type::of<uint32_t>(), {index_constant(0u)});
                            auto* y = extract(dispatch, Type::of<uint32_t>(), {index_constant(1u)});
                            coordinate = builder_.call(
                                Type::vector(Type::of<uint32_t>(), 2u), ArithmeticOp::AGGREGATE, {x, y});
                        }
                        for (const auto& output : stage_color_outputs_) {
                            auto* source = output.return_field == TypedIR::NoIndex
                                               ? value
                                               : extract(value, output.resource->element_type,
                                                         {index_constant(output.return_field)});
                            auto* destination = builder_.call(
                                output.resource->element_type, ResourceReadOp::TEXTURE2D_READ,
                                {output.resource->argument, coordinate});
                            source = apply_graphics_blend(source, destination, output.blend);
                            if (source == nullptr) return false;
                            builder_.call(ResourceWriteOp::TEXTURE2D_WRITE,
                                          {output.resource->argument, coordinate, source});
                        }
                    } else {
                        auto* dispatch = xir_module_.create_dispatch_id();
                        auto* index = extract(dispatch, Type::of<uint32_t>(), {index_constant(0u)});
                        builder_.call(ResourceWriteOp::BUFFER_WRITE, {stage_output_->argument, index, value});
                    }
                    builder_.return_void();
                } else {
                    builder_.return_(value);
                }
            }
            return true;
        case kStatementExpression:
            return lower_expression(statement.a) != nullptr;
        case kStatementBarrier:
            uses_group_semantics_ = true;
            builder_.synchronize_block();
            return true;
        default:
            return fail("unsupported FEIR statement kind " + std::to_string(statement.kind));
        }
    }

    bool lower_if(const TypedIR::Statement& statement) {
        auto* condition = lower_expression(statement.a);
        if (condition == nullptr) return false;
        auto constants_before = constant_integers_;
        auto* branch = builder_.if_(builder_.static_cast_if_necessary(Type::of<bool>(), condition));
        auto* merge = branch->create_merge_block();
        builder_.set_insertion_point(branch->create_true_block());
        if (!lower_statement(statement.b)) return false;
        auto constants_true = constant_integers_;
        if (!builder_.is_insertion_point_terminator()) builder_.br(merge);
        builder_.set_insertion_point(branch->create_false_block());
        constant_integers_ = constants_before;
        if (statement.c != TypedIR::NoIndex && !lower_statement(statement.c)) return false;
        auto constants_false = constant_integers_;
        if (!builder_.is_insertion_point_terminator()) builder_.br(merge);
        builder_.set_insertion_point(merge);
        constant_integers_.clear();
        for (const auto& [name, value] : constants_true) {
            const auto found = constants_false.find(name);
            if (found != constants_false.end() && found->second == value)
                constant_integers_.emplace(name, value);
        }
        return true;
    }

    bool lower_loop(uint32_t body_id, uint32_t condition_id, uint32_t update_id, bool do_first) {
        Value* first_iteration = nullptr;
        if (do_first) {
            first_iteration = builder_.alloca_local(Type::of<bool>());
            builder_.store(first_iteration, xir_module_.create_constant_one(Type::of<bool>()));
        }
        auto* loop = builder_.loop();
        auto* prepare = loop->create_prepare_block();
        auto* body = loop->create_body_block();
        auto* update = loop->create_update_block();
        auto* merge = loop->create_merge_block();
        builder_.set_insertion_point(prepare);
        if (condition_id == TypedIR::NoIndex) builder_.br(body);
        else {
            auto* condition = lower_expression(condition_id);
            if (condition == nullptr) return false;
            if (do_first) {
                auto* first = builder_.load(Type::of<bool>(), first_iteration);
                condition = builder_.call(Type::of<bool>(), ArithmeticOp::BINARY_BIT_OR,
                                          {first, builder_.static_cast_if_necessary(Type::of<bool>(), condition)});
            }
            builder_.cond_br(builder_.static_cast_if_necessary(Type::of<bool>(), condition), body, merge);
        }
        loops_.push_back({merge, update});
        builder_.set_insertion_point(body);
        if (!lower_statement(body_id)) return false;
        if (!builder_.is_insertion_point_terminator()) builder_.br(update);
        builder_.set_insertion_point(update);
        if (update_id != TypedIR::NoIndex && !lower_statement(update_id)) return false;
        if (!builder_.is_insertion_point_terminator()) {
            if (do_first) builder_.store(first_iteration, xir_module_.create_constant_zero(Type::of<bool>()));
            builder_.br(prepare);
        }
        loops_.pop_back();
        builder_.set_insertion_point(merge);
        return true;
    }

    const TypedIR::Module& module_;
    const TypedIR::LoweringInputs& inputs_;
    xir::Module& xir_module_;
    std::vector<BufferLayout>* buffer_layouts_ = nullptr;
    const AdInputs* ad_inputs_ = nullptr;
    std::vector<AdGradientLayout>* ad_gradient_layouts_ = nullptr;
    std::string* error_ = nullptr;
    KernelFunction* kernel_ = nullptr;
    FunctionDefinition* function_ = nullptr;
    CallableFunction* fragment_callable_ = nullptr;
    XIRBuilder builder_;
    std::unordered_map<uint32_t, const Type*> types_;
    std::unordered_map<uint32_t, const Type*> struct_types_;
    std::vector<bool> struct_visiting_;
    std::unordered_map<std::string, Resource> resources_;
    std::unordered_map<std::string, Sampler> samplers_;
    std::unordered_map<std::string, CallableRecord> callables_;
    std::unordered_map<std::string, Value*> locals_;
    std::unordered_map<std::string, int64_t> constant_integers_;
    struct SharedMemory { Value* pointer; uint32_t length; };
    std::unordered_map<std::string, SharedMemory> shared_;
    std::vector<LoopTargets> loops_;
    std::unordered_map<uint32_t, AdResource> ad_resources_;
    std::unordered_map<uint32_t, AdLocalBuffer> ad_local_buffers_;
    bool inside_ad_scope_ = false;
    bool uses_group_semantics_ = false;
    Resource* stage_input_ = nullptr;
    Resource* stage_output_ = nullptr;
    std::vector<StageColorOutput> stage_color_outputs_;
    Resource* stage_coverage_ = nullptr;
    std::string fragment_parameter_name_;
    const Type* fragment_parameter_type_ = nullptr;
    ValueArgument* fragment_parameter_argument_ = nullptr;
    std::array<ValueArgument*, 4u> fragment_neighbor_arguments_{};
    ValueArgument* fragment_coordinate_argument_ = nullptr;
    uint32_t logical_groups_per_block_ = 1u;
};

} // namespace

KernelFunction* LowerToXir(const TypedIR::Module& module, const TypedIR::LoweringInputs& inputs,
                           xir::Module& xir_module, std::vector<BufferLayout>* buffer_layouts,
                           const AdInputs* ad_inputs, std::vector<AdGradientLayout>* ad_gradient_layouts,
                           std::string* error) {
    if (error != nullptr) error->clear();
    if (buffer_layouts != nullptr) buffer_layouts->clear();
    if (ad_gradient_layouts != nullptr) ad_gradient_layouts->clear();
    return Lowerer{module, inputs, xir_module, buffer_layouts, ad_inputs, ad_gradient_layouts, error}.lower();
}

GraphicsFragmentXir LowerGraphicsFragmentToXir(
    const TypedIR::Module& module, const TypedIR::LoweringInputs& inputs,
    xir::Module& xir_module, std::vector<BufferLayout>* buffer_layouts,
    std::string* error) {
    if (error != nullptr) error->clear();
    if (buffer_layouts != nullptr) buffer_layouts->clear();
    return Lowerer{module, inputs, xir_module, buffer_layouts, nullptr, nullptr, error}
        .lower_graphics_fragment();
}

} // namespace Feather::Luisa
