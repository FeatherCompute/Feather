#include "feather_typed_ir_lowerer.h"

#include <array>
#include <cctype>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Feather::TypedIR {
namespace {

constexpr uint8_t kResourceKindBuffer = 1;
constexpr uint8_t kResourceKindTexture2D = 2;
constexpr uint8_t kResourceKindSampler = 3;
constexpr uint8_t kResourceKindPushConstant = 5;
constexpr uint8_t kResourceKindTexture3D = 6;

constexpr uint8_t kAccessRead = 1;
constexpr uint8_t kAccessWrite = 2;
constexpr uint8_t kAccessSample = 4;

constexpr uint8_t kFunctionCompute1D = 0;
constexpr uint8_t kFunctionCompute2D = 1;
constexpr uint8_t kFunctionCompute3D = 2;
constexpr uint8_t kFunctionCallable = 5;

constexpr uint8_t kTypePrimitive = 1;
constexpr uint8_t kTypeVector = 2;
constexpr uint8_t kTypeMatrix = 3;
constexpr uint8_t kTypeStruct = 4;
constexpr uint8_t kTypeArray = 5;
constexpr uint8_t kTypeVoid = 7;

constexpr uint8_t kPrimitiveBool = 0;
constexpr uint8_t kPrimitiveInt = 1;
constexpr uint8_t kPrimitiveUInt = 2;
constexpr uint8_t kPrimitiveFloat = 3;

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

constexpr uint32_t kBarrierWorkgroup = 0;
constexpr uint32_t kBarrierMemory = 1;
constexpr uint32_t kBarrierFull = 2;

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

bool IsLocalLikeLValue(uint8_t kind) {
    return kind == kLValueLocal || kind == kLValueParameter;
}

struct RegisteredResource {
    GPU::IR::ResourceId id = GPU::IR::InvalidResourceId;
    uint8_t kind = 0;
    uint8_t access = 0;
};

GPU::IR::ResourceAccess ToResourceAccess(uint8_t access) {
    switch (access) {
    case kAccessRead:
        return GPU::IR::ResourceAccess::Read;
    case kAccessWrite:
        return GPU::IR::ResourceAccess::Write;
    default:
        return GPU::IR::ResourceAccess::ReadWrite;
    }
}

GPU::IR::Type TypeFromName(const std::string& name) {
    if (name == "System.Boolean" || name == "bool") {
        return GPU::IR::Type::Bool();
    }
    if (name == "System.Int32" || name == "int") {
        return GPU::IR::Type::Int();
    }
    if (name == "System.UInt32" || name == "uint") {
        return GPU::IR::Type::UInt();
    }
    if (name == "System.Single" || name == "float") {
        return GPU::IR::Type::Float();
    }
    if (name == "Feather.Math.int2" || name == "global::Feather.Math.int2" || name == "int2") {
        return GPU::IR::Type::Int2();
    }
    if (name == "Feather.Math.int3" || name == "global::Feather.Math.int3" || name == "int3") {
        return GPU::IR::Type::Int3();
    }
    if (name == "Feather.Math.int4" || name == "global::Feather.Math.int4" || name == "int4") {
        return GPU::IR::Type::Int4();
    }
    if (name == "Feather.Math.uint2" || name == "global::Feather.Math.uint2" || name == "uint2") {
        return GPU::IR::Type::UInt2();
    }
    if (name == "Feather.Math.uint3" || name == "global::Feather.Math.uint3" || name == "uint3") {
        return GPU::IR::Type::UInt3();
    }
    if (name == "Feather.Math.uint4" || name == "global::Feather.Math.uint4" || name == "uint4") {
        return GPU::IR::Type::UInt4();
    }
    if (name == "Feather.Math.bool2" || name == "global::Feather.Math.bool2" || name == "bool2") {
        return GPU::IR::Type::Bool2();
    }
    if (name == "Feather.Math.bool3" || name == "global::Feather.Math.bool3" || name == "bool3") {
        return GPU::IR::Type::Bool3();
    }
    if (name == "Feather.Math.bool4" || name == "global::Feather.Math.bool4" || name == "bool4") {
        return GPU::IR::Type::Bool4();
    }
    if (name == "Feather.Math.float2" || name == "global::Feather.Math.float2" || name == "float2") {
        return GPU::IR::Type::Float2();
    }
    if (name == "Feather.Math.float3" || name == "global::Feather.Math.float3" || name == "float3") {
        return GPU::IR::Type::Float3();
    }
    if (name == "Feather.Math.float4" || name == "global::Feather.Math.float4" || name == "float4") {
        return GPU::IR::Type::Float4();
    }
    if (name == "Feather.Math.float2x2" || name == "global::Feather.Math.float2x2" || name == "float2x2") {
        return GPU::IR::Type::Float2x2();
    }
    if (name == "Feather.Math.float3x3" || name == "global::Feather.Math.float3x3" || name == "float3x3") {
        return GPU::IR::Type::Float3x3();
    }
    if (name == "Feather.Math.float4x4" || name == "global::Feather.Math.float4x4" || name == "float4x4") {
        return GPU::IR::Type::Float4x4();
    }

    return {};
}

std::string SanitizeGlslIdentifier(std::string name) {
    if (name.empty()) {
        return {};
    }

    for (auto& ch : name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_') {
            ch = '_';
        }
    }
    if (std::isdigit(static_cast<unsigned char>(name.front()))) {
        name.insert(name.begin(), '_');
    }
    static const std::unordered_set<std::string_view> reserved = {
        "active", "asm", "atomic_uint", "attribute", "bool", "break", "buffer", "bvec2", "bvec3", "bvec4",
        "case", "cast", "centroid", "class", "coherent", "common", "const", "continue", "default", "discard",
        "dmat2", "dmat2x2", "dmat2x3", "dmat2x4", "dmat3", "dmat3x2", "dmat3x3", "dmat3x4", "dmat4",
        "dmat4x2", "dmat4x3", "dmat4x4", "do", "double", "dvec2", "dvec3", "dvec4", "else", "enum",
        "extern", "external", "false", "filter", "fixed", "flat", "float", "for", "fvec2", "fvec3", "fvec4",
        "goto", "half", "highp", "hvec2", "hvec3", "hvec4", "if", "image", "image1D", "image1DArray",
        "image2D", "image2DArray", "image2DMS", "image2DMSArray", "image2DRect", "image3D", "imageBuffer",
        "in", "inline", "inout", "input", "int", "interface", "invariant", "isampler1D", "isampler1DArray",
        "isampler2D", "isampler2DArray", "isampler2DMS", "isampler2DMSArray", "isampler2DRect", "isampler3D",
        "isamplerBuffer", "isamplerCube", "isamplerCubeArray", "ivec2", "ivec3", "ivec4", "layout", "long",
        "lowp", "mat2", "mat2x2", "mat2x3", "mat2x4", "mat3", "mat3x2", "mat3x3", "mat3x4", "mat4",
        "mat4x2", "mat4x3", "mat4x4", "mediump", "namespace", "noinline", "noperspective", "out", "output",
        "packed", "partition", "patch", "precision", "public", "readonly", "resource", "restrict", "return",
        "sample", "sampler", "sampler1D", "sampler1DArray", "sampler1DArrayShadow", "sampler1DShadow",
        "sampler2D", "sampler2DArray", "sampler2DArrayShadow", "sampler2DMS", "sampler2DMSArray",
        "sampler2DRect", "sampler2DRectShadow", "sampler2DShadow", "sampler3D", "samplerBuffer",
        "samplerCube", "samplerCubeArray", "samplerCubeArrayShadow", "samplerCubeShadow", "shared", "short",
        "sizeof", "smooth", "static", "struct", "subroutine", "superp", "switch", "template", "texture",
        "this", "true", "typedef", "uimage1D", "uimage1DArray", "uimage2D", "uimage2DArray", "uimage2DMS",
        "uimage2DMSArray", "uimage2DRect", "uimage3D", "uimageBuffer", "uint", "uniform", "union",
        "unsigned", "usampler1D", "usampler1DArray", "usampler2D", "usampler2DArray", "usampler2DMS",
        "usampler2DMSArray", "usampler2DRect", "usampler3D", "usamplerBuffer", "usamplerCube",
        "usamplerCubeArray", "using", "uvec2", "uvec3", "uvec4", "varying", "vec2", "vec3", "vec4", "void",
        "volatile", "while", "writeonly",
        "gl_FragCoord", "gl_FragDepth", "gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_LocalInvocationIndex",
        "gl_NumWorkGroups", "gl_Position", "gl_WorkGroupID", "gl_WorkGroupSize"
    };
    if (reserved.find(name) != reserved.end() || name.rfind("gl_", 0) == 0 || name.rfind("__", 0) == 0) {
        name = "fe_" + name;
    }
    return name;
}

class Lowerer {
public:
    Lowerer(const Module& typed, const LoweringInputs& inputs, std::string* error)
        : typed_(typed), inputs_(inputs), error_(error) {
    }

    std::unique_ptr<GPU::IR::Module> Lower() {
        if (inputs_.shader_kind < 1 || inputs_.shader_kind > 3 ||
            inputs_.group_x <= 0 || inputs_.group_y <= 0 || inputs_.group_z <= 0 ||
            typed_.entry_function >= typed_.functions.size()) {
            Fail("invalid section 7 module header, shader kind, workgroup size, or entry function");
            return nullptr;
        }

        const auto& entry = typed_.functions[typed_.entry_function];
        if (entry.kind > kFunctionCompute3D || entry.body_statement_index >= typed_.statements.size()) {
            Fail("invalid section 7 entry function record or body statement index");
            return nullptr;
        }

        builder_.BeginComputeKernel(
            static_cast<uint32_t>(inputs_.group_x),
            static_cast<uint32_t>(inputs_.group_y),
            static_cast<uint32_t>(inputs_.group_z),
            DimensionFor(entry.kind));

        if (!RegisterResources() || !RegisterCallables() || !EmitBoundsCheckGuard(entry.kind) ||
            !LowerStatement(entry.body_statement_index)) {
            Fail("section 7 typed IR lowerer failed before EasyGPU module creation");
            return nullptr;
        }

        return std::make_unique<GPU::IR::Module>(builder_.GetModule());
    }

private:
    uint32_t DimensionFor(uint8_t function_kind) const {
        switch (function_kind) {
        case kFunctionCompute2D:
            return 2;
        case kFunctionCompute3D:
            return 3;
        default:
            return 1;
        }
    }

    bool Fail(std::string message) const {
        if (error_ != nullptr && error_->empty()) {
            *error_ = std::move(message);
        }

        return false;
    }

    GPU::IR::ValueId InvalidValue(std::string message) const {
        Fail(std::move(message));
        return GPU::IR::InvalidValueId;
    }

    bool RegisterResources() {
        for (const auto& resource : inputs_.resources) {
            GPU::IR::ResourceId id = GPU::IR::InvalidResourceId;
            if (resource.kind == kResourceKindBuffer) {
                const auto type = TypeFromName(resource.element_type);
                if (!type.IsValid()) {
                    return Fail("buffer resource '" + resource.name + "' uses unsupported element type '" +
                                resource.element_type + "'");
                }

                id = builder_.AddBuffer(resource.binding, type, ToResourceAccess(resource.access),
                                        BufferName(resource.binding));
            } else if (resource.kind == kResourceKindTexture2D || resource.kind == kResourceKindTexture3D) {
                const auto is_texture3d = resource.kind == kResourceKindTexture3D;
                if (resource.width == 0 || resource.height == 0 ||
                    (is_texture3d ? resource.depth == 0 : resource.depth != 1)) {
                    return Fail(std::string(is_texture3d ? "texture3D" : "texture2D") +
                                " resource '" + resource.name + "' has invalid dimensions " +
                                std::to_string(resource.width) + "x" + std::to_string(resource.height) +
                                "x" + std::to_string(resource.depth));
                }

                const auto texture_type = TextureElementTypeFromName(resource.element_type);
                if (!texture_type.IsValid()) {
                    return Fail(std::string(is_texture3d ? "texture3D" : "texture2D") +
                                " resource '" + resource.name + "' uses unsupported element type '" +
                                resource.element_type + "'");
                }

                if (is_texture3d) {
                    id = builder_.AddTexture3D(resource.binding, texture_type, ToResourceAccess(resource.access),
                                               TextureName(resource.binding), resource.texture_format,
                                               resource.width, resource.height, resource.depth,
                                               resource.sampled || resource.access == kAccessSample);
                } else {
                    id = builder_.AddTexture2D(resource.binding, texture_type, ToResourceAccess(resource.access),
                                               TextureName(resource.binding), resource.texture_format,
                                               resource.width, resource.height,
                                               resource.sampled || resource.access == kAccessSample);
                }
            } else if (resource.kind == kResourceKindPushConstant) {
                const auto type = TypeFromName(resource.element_type);
                if (!type.IsValid()) {
                    return Fail("push constant resource '" + resource.name + "' uses unsupported element type '" +
                                resource.element_type + "'");
                }

                const auto* push_constant = FindPushConstant(resource.binding);
                if (push_constant == nullptr || push_constant->size == 0 || push_constant->alignment == 0) {
                    return Fail("push constant resource '" + resource.name +
                                "' is missing packed data, size, or alignment");
                }

                id = builder_.AddPushConstant(resource.binding, type, PushConstantName(resource.binding),
                                              push_constant->data, push_constant->size, push_constant->alignment);
            } else if (resource.kind == kResourceKindSampler) {
                id = static_cast<GPU::IR::ResourceId>(resource.binding + 1000);
            } else {
                return Fail("resource '" + resource.name + "' has unsupported kind " +
                            std::to_string(resource.kind));
            }

            if (id == GPU::IR::InvalidResourceId) {
                return Fail("EasyGPU rejected resource '" + resource.name + "'");
            }

            resources_by_name_[resource.name] = id;
            resource_infos_by_name_[resource.name] = RegisteredResource{id, resource.kind, resource.access};
            resources_by_binding_[resource.binding] = id;
            resource_infos_by_binding_[resource.binding] = RegisteredResource{id, resource.kind, resource.access};
        }

        if (!RegisterBoundsCheckResources()) {
            return false;
        }

        return true;
    }

    bool RegisterBoundsCheckResources() {
        if (!inputs_.bounds_check) {
            return true;
        }

        std::array<int32_t*, 3> values{
            inputs_.logical_x_data,
            inputs_.logical_y_data,
            inputs_.logical_z_data
        };
        for (uint32_t axis = 0; axis < logical_size_resource_.size(); ++axis) {
            if (values[axis] == nullptr) {
                return Fail("hidden logical dispatch-size data is missing");
            }

            logical_size_resource_[axis] = builder_.AddPushConstant(
                UINT32_MAX - axis,
                GPU::IR::Type::Int(),
                "__feather_dispatch_size_" + std::to_string(axis),
                values[axis],
                sizeof(int32_t),
                alignof(int32_t));
            if (logical_size_resource_[axis] == GPU::IR::InvalidResourceId) {
                return Fail("EasyGPU rejected hidden logical dispatch-size push constants");
            }
        }

        return true;
    }

    bool RegisterCallables() {
        // Callable bodies can reference callables declared later in the C# struct.
        // Register every mangled symbol before lowering any body so nested calls are
        // resolved by the typed call graph instead of depending on source order.
        for (uint32_t function_id = 0; function_id < typed_.functions.size(); ++function_id) {
            const auto& function = typed_.functions[function_id];
            if (function.kind != kFunctionCallable) {
                continue;
            }

            const auto* raw_name = GetString(function.mangled_name_id);
            if (raw_name == nullptr || raw_name->empty() || function.body_statement_index >= typed_.statements.size()) {
                return false;
            }

            const auto name = SanitizeGlslIdentifier(*raw_name);
            const auto return_type = ToModuleType(function.return_type_id);
            if (name.empty() || !return_type.IsValid()) {
                return false;
            }

            callable_names_[*raw_name] = name;
        }

        for (uint32_t function_id = 0; function_id < typed_.functions.size(); ++function_id) {
            const auto& function = typed_.functions[function_id];
            if (function.kind != kFunctionCallable) {
                continue;
            }

            const auto* raw_name = GetString(function.mangled_name_id);
            if (raw_name == nullptr || raw_name->empty() || function.body_statement_index >= typed_.statements.size()) {
                return false;
            }

            const auto name = SanitizeGlslIdentifier(*raw_name);
            const auto return_type = ToModuleType(function.return_type_id);
            if (name.empty() || !return_type.IsValid()) {
                return false;
            }

            auto previous_local_values = local_values_;
            auto previous_declared_locals = declared_locals_;
            auto previous_local_glsl_names = local_glsl_names_;
            auto previous_shared_values = shared_values_;
            std::vector<std::pair<std::string, GPU::IR::Type>> parameters;
            parameters.reserve(function.parameter_count);

            if (function.parameter_count > 0) {
                if (function.first_parameter == NoIndex ||
                    function.first_parameter > typed_.parameters.size() ||
                    function.parameter_count > typed_.parameters.size() - function.first_parameter) {
                    return false;
                }
            }

            for (uint32_t i = 0; i < function.parameter_count; ++i) {
                const auto& parameter = typed_.parameters[function.first_parameter + i];
                const auto* parameter_name = GetString(parameter.name_id);
                const auto parameter_type = ToModuleType(parameter.type_id);
                if (parameter_name == nullptr || parameter_name->empty() || !parameter_type.IsValid() ||
                    parameter.direction != 0) {
                    return false;
                }

                const auto sanitized_parameter_name = UniqueGlslName(*parameter_name);
                if (sanitized_parameter_name.empty()) {
                    return false;
                }

                parameters.emplace_back(sanitized_parameter_name, parameter_type);
                local_values_[*parameter_name] = builder_.LocalVariable(parameter_type, sanitized_parameter_name);
                declared_locals_[*parameter_name] = parameter_type;
                local_glsl_names_[*parameter_name] = sanitized_parameter_name;
            }

            auto body = LowerCallableStatementList(function.body_statement_index);
            local_values_ = std::move(previous_local_values);
            declared_locals_ = std::move(previous_declared_locals);
            local_glsl_names_ = std::move(previous_local_glsl_names);
            shared_values_ = std::move(previous_shared_values);
            if (!body.has_value()) {
                return false;
            }

            const auto callable_id = builder_.AddCallable(
                name, return_type, std::move(parameters), std::move(body->statements), std::move(body->blocks));
            if (callable_id == GPU::IR::InvalidFunctionId) {
                return false;
            }
        }

        return true;
    }

    GPU::IR::Type TypeFromName(const std::string& name) const {
        auto primitive = Feather::TypedIR::TypeFromName(name);
        if (primitive.IsValid()) {
            return primitive;
        }

        for (uint32_t i = 0; i < typed_.structs.size(); ++i) {
            const auto& structure = typed_.structs[i];
            const auto* simple = GetString(structure.name_id);
            const auto* qualified = GetString(structure.fully_qualified_name_id);
            if ((simple != nullptr && *simple == name) || (qualified != nullptr && *qualified == name) ||
                (qualified != nullptr && StripGlobalPrefix(*qualified) == StripGlobalPrefix(name))) {
                return StructType(i);
            }
        }

        return {};
    }

    GPU::IR::Type StructType(uint32_t struct_index) const {
        if (struct_index >= typed_.structs.size()) {
            return {};
        }

        const auto& structure = typed_.structs[struct_index];
        const auto* raw_name = GetString(structure.name_id);
        if (raw_name == nullptr) {
            return {};
        }

        const auto type_name = SanitizeGlslIdentifier(*raw_name);
        std::vector<uint32_t> visiting;
        std::unordered_set<uint32_t> emitted;
        std::vector<std::pair<std::string, std::string>> definitions;
        if (!CollectStructDefinitions(struct_index, visiting, emitted, definitions) || definitions.empty()) {
            return {};
        }

        auto definition = std::move(definitions.back());
        definitions.pop_back();
        if (type_name.empty() || definition.first != type_name || definition.second.empty()) {
            return {};
        }

        return GPU::IR::Type::Struct(type_name, std::move(definition.second), std::move(definitions));
    }

    bool CollectStructDefinitions(
        uint32_t struct_index,
        std::vector<uint32_t>& visiting,
        std::unordered_set<uint32_t>& emitted,
        std::vector<std::pair<std::string, std::string>>& definitions) const {
        if (struct_index >= typed_.structs.size()) {
            return false;
        }
        if (emitted.find(struct_index) != emitted.end()) {
            return true;
        }
        if (std::find(visiting.begin(), visiting.end(), struct_index) != visiting.end()) {
            return false;
        }

        const auto& structure = typed_.structs[struct_index];
        const auto* raw_name = GetString(structure.name_id);
        if (raw_name == nullptr) {
            return false;
        }

        visiting.push_back(struct_index);
        if (structure.field_count > 0) {
            if (structure.first_field == NoIndex ||
                structure.first_field > typed_.struct_fields.size() ||
                structure.field_count > typed_.struct_fields.size() - structure.first_field) {
                visiting.pop_back();
                return false;
            }

            for (uint32_t i = 0; i < structure.field_count; ++i) {
                const auto& field = typed_.struct_fields[structure.first_field + i];
                if (field.type_id >= typed_.types.size()) {
                    visiting.pop_back();
                    return false;
                }

                if (!CollectStructTypeDependencies(field.type_id, visiting, emitted, definitions)) {
                    visiting.pop_back();
                    return false;
                }
            }
        }

        const auto type_name = SanitizeGlslIdentifier(*raw_name);
        auto definition = BuildStructDefinition(struct_index, type_name);
        if (type_name.empty() || definition.empty()) {
            visiting.pop_back();
            return false;
        }

        emitted.insert(struct_index);
        visiting.pop_back();
        definitions.emplace_back(type_name, std::move(definition));
        return true;
    }

    std::string BuildStructDefinition(uint32_t struct_index, const std::string& type_name) const {
        if (struct_index >= typed_.structs.size() || type_name.empty()) {
            return {};
        }

        const auto& structure = typed_.structs[struct_index];
        if (structure.field_count == 0) {
            return "struct " + type_name + " {\n};\n";
        }
        if (structure.first_field == NoIndex ||
            structure.first_field > typed_.struct_fields.size() ||
            structure.field_count > typed_.struct_fields.size() - structure.first_field) {
            return {};
        }

        std::string definition = "struct " + type_name + " {\n";
        for (uint32_t i = 0; i < structure.field_count; ++i) {
            const auto& field = typed_.struct_fields[structure.first_field + i];
            const auto field_decl = GlslStructFieldTypeAndSuffix(field.type_id);
            const auto* field_name = GetString(field.name_id);
            if (field_decl.first.empty() || field_name == nullptr || field_name->empty()) {
                return {};
            }

            definition += "    " + field_decl.first + " " + SanitizeGlslIdentifier(*field_name) + field_decl.second + ";\n";
        }

        definition += "};\n";
        return definition;
    }

    bool CollectStructTypeDependencies(
        uint32_t type_id,
        std::vector<uint32_t>& visiting,
        std::unordered_set<uint32_t>& emitted,
        std::vector<std::pair<std::string, std::string>>& definitions) const {
        if (type_id >= typed_.types.size()) {
            return false;
        }

        const auto& type = typed_.types[type_id];
        if (type.kind == kTypeStruct) {
            return CollectStructDefinitions(type.a, visiting, emitted, definitions);
        }
        if (type.kind == kTypeArray) {
            return CollectStructTypeDependencies(type.a, visiting, emitted, definitions);
        }

        return true;
    }

    static std::string StripGlobalPrefix(const std::string& value) {
        constexpr std::string_view prefix = "global::";
        return value.rfind(prefix, 0) == 0 ? value.substr(prefix.size()) : value;
    }

    static std::string GlslTypeName(const GPU::IR::Type& type) {
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool:
            return "bool";
        case GPU::IR::Type::Kind::Int:
            return "int";
        case GPU::IR::Type::Kind::UInt:
            return "uint";
        case GPU::IR::Type::Kind::Float:
            return "float";
        case GPU::IR::Type::Kind::Bool2:
            return "bvec2";
        case GPU::IR::Type::Kind::Bool3:
            return "bvec3";
        case GPU::IR::Type::Kind::Bool4:
            return "bvec4";
        case GPU::IR::Type::Kind::Int2:
            return "ivec2";
        case GPU::IR::Type::Kind::Int3:
            return "ivec3";
        case GPU::IR::Type::Kind::Int4:
            return "ivec4";
        case GPU::IR::Type::Kind::UInt2:
            return "uvec2";
        case GPU::IR::Type::Kind::UInt3:
            return "uvec3";
        case GPU::IR::Type::Kind::UInt4:
            return "uvec4";
        case GPU::IR::Type::Kind::Float2:
            return "vec2";
        case GPU::IR::Type::Kind::Float3:
            return "vec3";
        case GPU::IR::Type::Kind::Float4:
            return "vec4";
        case GPU::IR::Type::Kind::Float2x2:
            return "mat2";
        case GPU::IR::Type::Kind::Float3x3:
            return "mat3";
        case GPU::IR::Type::Kind::Float4x4:
            return "mat4";
        case GPU::IR::Type::Kind::Struct:
            return type.typeName;
        default:
            return {};
        }
    }

    std::pair<std::string, std::string> GlslStructFieldTypeAndSuffix(uint32_t type_id) const {
        if (type_id >= typed_.types.size()) {
            return {};
        }

        const auto& type = typed_.types[type_id];
        if (type.kind == kTypeArray) {
            if (type.b == NoIndex || type.b == 0) {
                return {};
            }

            auto element = GlslStructFieldTypeAndSuffix(type.a);
            if (element.first.empty()) {
                return {};
            }

            element.second += "[" + std::to_string(type.b) + "]";
            return element;
        }

        auto module_type = ToModuleType(type_id);
        auto glsl_type = GlslTypeName(module_type);
        return glsl_type.empty()
                   ? std::pair<std::string, std::string>{}
                   : std::pair<std::string, std::string>{std::move(glsl_type), {}};
    }

    const PushConstantInfo* FindPushConstant(uint32_t binding) const {
        for (const auto& push_constant : inputs_.push_constants) {
            if (push_constant.binding == binding) {
                return &push_constant;
            }
        }

        return nullptr;
    }

    bool LowerStatement(uint32_t statement_id) {
        if (statement_id >= typed_.statements.size()) {
            return Fail("statement index " + std::to_string(statement_id) + " is outside the section 7 statement table");
        }

        const auto& statement = typed_.statements[statement_id];
        switch (statement.kind) {
        case kStatementBlock:
            return LowerBlock(statement);
        case kStatementLocalDeclaration:
            return LowerLocalDeclaration(statement);
        case kStatementAssignment:
            return LowerAssignment(statement);
        case kStatementCompoundAssignment:
            return LowerCompoundAssignment(statement);
        case kStatementIf:
            return LowerIf(statement);
        case kStatementFor:
            return LowerFor(statement);
        case kStatementWhile:
            return LowerWhile(statement);
        case kStatementDoWhile:
            return LowerDoWhile(statement);
        case kStatementBreak:
            EmitBreak();
            return true;
        case kStatementContinue:
            EmitContinue();
            return true;
        case kStatementReturn:
            return LowerReturn(statement);
        case kStatementExpression: {
            if (statement.a >= typed_.expressions.size()) {
                return Fail("expression statement references expression index " + std::to_string(statement.a) +
                            " outside the section 7 expression table");
            }

            const auto value = BuildExpression(statement.a);
            if (value == GPU::IR::InvalidValueId) {
                return false;
            }

            EmitExpression(value);
            return true;
        }
        case kStatementBarrier:
            if (!EmitBarrier(statement.op)) {
                return Fail("barrier statement uses unsupported barrier kind " + std::to_string(statement.op));
            }
            return true;
        case kStatementIncrementDecrement:
            return LowerIncrementDecrement(statement);
        case kStatementSharedMemoryDeclaration:
            return LowerSharedMemoryDeclaration(statement);
        default:
            return Fail("unsupported section 7 statement kind " + std::to_string(statement.kind));
        }
    }

    bool LowerBlock(const Statement& block) {
        if (block.child_count == 0) {
            return block.first_child == NoIndex;
        }
        if (block.first_child == NoIndex || block.first_child > typed_.children.size() ||
            block.child_count > typed_.children.size() - block.first_child) {
            return false;
        }

        for (uint32_t i = 0; i < block.child_count; ++i) {
            if (!LowerStatement(typed_.children[block.first_child + i])) {
                return false;
            }
        }

        return true;
    }

    bool EmitBoundsCheckGuard(uint8_t function_kind) {
        if (!inputs_.bounds_check) {
            return true;
        }

        const auto logical_x = builder_.PushConstant(logical_size_resource_[0]);
        const auto condition_x = builder_.Compare(GPU::IR::CompareOp::GreaterEqual, builder_.ThreadIndexX(), logical_x);
        if (logical_x == GPU::IR::InvalidValueId || condition_x == GPU::IR::InvalidValueId) {
            return false;
        }

        auto condition = condition_x;
        if (function_kind >= kFunctionCompute2D) {
            const auto logical_y = builder_.PushConstant(logical_size_resource_[1]);
            const auto condition_y = builder_.Compare(GPU::IR::CompareOp::GreaterEqual, builder_.ThreadIndexY(), logical_y);
            condition = builder_.Binary(GPU::IR::BinaryOp::LogicalOr, condition, condition_y);
            if (logical_y == GPU::IR::InvalidValueId || condition_y == GPU::IR::InvalidValueId ||
                condition == GPU::IR::InvalidValueId) {
                return false;
            }
        }

        if (function_kind >= kFunctionCompute3D) {
            const auto logical_z = builder_.PushConstant(logical_size_resource_[2]);
            const auto condition_z = builder_.Compare(GPU::IR::CompareOp::GreaterEqual, builder_.ThreadIndexZ(), logical_z);
            condition = builder_.Binary(GPU::IR::BinaryOp::LogicalOr, condition, condition_z);
            if (logical_z == GPU::IR::InvalidValueId || condition_z == GPU::IR::InvalidValueId ||
                condition == GPU::IR::InvalidValueId) {
                return false;
            }
        }

        std::vector<GPU::IR::Statement> then_statements;
        GPU::IR::Statement return_statement;
        return_statement.kind = GPU::IR::Statement::Kind::Return;
        then_statements.push_back(return_statement);
        EmitIf(condition, AddBlock(std::move(then_statements)), GPU::IR::InvalidBlockId);
        return true;
    }

    bool LowerLocalDeclaration(const Statement& statement) {
        const auto* name = GetString(statement.name_id);
        if (name == nullptr || statement.op >= typed_.types.size()) {
            return false;
        }

        const auto type = ToModuleType(statement.op);
        if (!type.IsValid()) {
            return false;
        }

        const auto glsl_name = UniqueGlslName(*name);
        if (glsl_name.empty()) {
            return false;
        }

        if (statement.a == NoIndex) {
            local_values_[*name] = builder_.LocalVariable(type, glsl_name);
            declared_locals_[*name] = type;
            local_glsl_names_[*name] = glsl_name;
            EmitLocalDeclaration(type, glsl_name, GPU::IR::InvalidValueId);
            return local_values_[*name] != GPU::IR::InvalidValueId;
        }

        const auto value = BuildExpression(statement.a);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        local_values_[*name] = builder_.LocalVariable(type, glsl_name);
        declared_locals_[*name] = type;
        local_glsl_names_[*name] = glsl_name;
        EmitLocalDeclaration(type, glsl_name, value);
        return local_values_[*name] != GPU::IR::InvalidValueId;
    }

    bool LowerSharedMemoryDeclaration(const Statement& statement) {
        const auto* name = GetString(statement.name_id);
        if (name == nullptr || name->empty() || statement.a == 0 || statement.op >= typed_.types.size()) {
            return false;
        }

        const auto type = ToModuleType(statement.op);
        if (!type.IsValid()) {
            return false;
        }

        const auto sanitized = SanitizeGlslIdentifier(*name);
        shared_values_[*name] = SharedMemoryInfo{type, sanitized};
        EmitSharedMemoryDeclaration(type, static_cast<uint32_t>(statement.a), sanitized);
        return true;
    }

    bool LowerAssignment(const Statement& statement) {
        if (statement.a >= typed_.lvalues.size() || statement.b >= typed_.expressions.size()) {
            return false;
        }

        const auto value = BuildExpression(statement.b);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto& target = typed_.lvalues[statement.a];
        if (IsLocalLikeLValue(target.kind)) {
            const auto destination = BuildLValueRead(statement.a);
            if (destination == GPU::IR::InvalidValueId) {
                return false;
            }

            EmitStore(destination, value);
            return true;
        }

        const auto destination = BuildLValueAddress(statement.a);
        if (destination == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(destination, value);
        return true;
    }

    bool LowerCompoundAssignment(const Statement& statement) {
        if (statement.a >= typed_.lvalues.size() || statement.b >= typed_.expressions.size()) {
            return false;
        }

        GPU::IR::BinaryOp op{};
        if (!TryMapBinaryOp(statement.op, &op)) {
            return false;
        }

        const auto left = BuildLValueRead(statement.a);
        const auto right = BuildExpression(statement.b);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto value = builder_.Binary(op, left, right);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto& target = typed_.lvalues[statement.a];
        if (IsLocalLikeLValue(target.kind)) {
            const auto destination = BuildLValueRead(statement.a);
            if (destination == GPU::IR::InvalidValueId) {
                return false;
            }

            EmitStore(destination, value);
            return true;
        }

        const auto address = BuildLValueAddress(statement.a);
        if (address == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(address, value);
        return true;
    }

    bool LowerIncrementDecrement(const Statement& statement) {
        if (statement.a >= typed_.lvalues.size()) {
            return false;
        }

        const auto type = ToModuleType(typed_.lvalues[statement.a].type_id);
        if (!type.IsValid()) {
            return false;
        }

        const auto current = BuildLValueRead(statement.a);
        const auto one = builder_.Literal(type, "1");
        if (current == GPU::IR::InvalidValueId || one == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto op = (statement.op & 1u) != 0 ? GPU::IR::BinaryOp::Add : GPU::IR::BinaryOp::Sub;
        const auto value = builder_.Binary(op, current, one);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto& target = typed_.lvalues[statement.a];
        if (IsLocalLikeLValue(target.kind)) {
            const auto destination = BuildLValueRead(statement.a);
            if (destination == GPU::IR::InvalidValueId) {
                return false;
            }

            EmitStore(destination, value);
            return true;
        }

        const auto address = BuildLValueAddress(statement.a);
        if (address == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(address, value);
        return true;
    }

    bool LowerIf(const Statement& statement) {
        if (statement.a >= typed_.expressions.size() || statement.b >= typed_.statements.size()) {
            return false;
        }

        const auto condition = BuildExpression(statement.a);
        if (condition == GPU::IR::InvalidValueId) {
            return false;
        }

        auto then_statements = LowerStatementList(statement.b);
        if (!then_statements.has_value()) {
            return false;
        }

        auto else_block_id = GPU::IR::InvalidBlockId;
        if (statement.c != NoIndex) {
            auto else_statements = LowerStatementList(statement.c);
            if (!else_statements.has_value()) {
                return false;
            }

            else_block_id = AddBlock(std::move(*else_statements));
        }

        const auto then_block_id = AddBlock(std::move(*then_statements));
        EmitIf(condition, then_block_id, else_block_id);
        return true;
    }

    bool LowerFor(const Statement& statement) {
        if ((statement.b != NoIndex && statement.b >= typed_.expressions.size()) ||
            statement.op >= typed_.statements.size()) {
            return false;
        }

        const auto outer_local_values = local_values_;
        const auto outer_declared_locals = declared_locals_;
        const auto outer_local_glsl_names = local_glsl_names_;
        const auto outer_shared_values = shared_values_;

        auto init_block_id = GPU::IR::InvalidBlockId;
        if (statement.a != NoIndex) {
            if (statement.a >= typed_.statements.size()) {
                return false;
            }
            auto init_statements = LowerStatementListKeepingLocals(statement.a);
            if (!init_statements.has_value()) {
                local_values_ = outer_local_values;
                declared_locals_ = outer_declared_locals;
                local_glsl_names_ = outer_local_glsl_names;
                shared_values_ = outer_shared_values;
                return false;
            }
            init_block_id = AddBlock(std::move(*init_statements));
        }

        const auto condition = statement.b == NoIndex
                                   ? builder_.Literal(GPU::IR::Type::Bool(), "true")
                                   : BuildExpression(statement.b);
        if (condition == GPU::IR::InvalidValueId) {
            local_values_ = outer_local_values;
            declared_locals_ = outer_declared_locals;
            local_glsl_names_ = outer_local_glsl_names;
            shared_values_ = outer_shared_values;
            return false;
        }

        auto step_block_id = GPU::IR::InvalidBlockId;
        if (statement.c != NoIndex) {
            if (statement.c >= typed_.statements.size()) {
                local_values_ = outer_local_values;
                declared_locals_ = outer_declared_locals;
                local_glsl_names_ = outer_local_glsl_names;
                shared_values_ = outer_shared_values;
                return false;
            }
            auto step_statements = LowerStatementList(statement.c);
            if (!step_statements.has_value()) {
                local_values_ = outer_local_values;
                declared_locals_ = outer_declared_locals;
                local_glsl_names_ = outer_local_glsl_names;
                shared_values_ = outer_shared_values;
                return false;
            }
            step_block_id = AddBlock(std::move(*step_statements));
        }

        auto body_statements = LowerStatementList(statement.op);
        if (!body_statements.has_value()) {
            local_values_ = outer_local_values;
            declared_locals_ = outer_declared_locals;
            local_glsl_names_ = outer_local_glsl_names;
            shared_values_ = outer_shared_values;
            return false;
        }

        const auto body_block_id = AddBlock(std::move(*body_statements));
        EmitFor(init_block_id, condition, step_block_id, body_block_id);
        local_values_ = outer_local_values;
        declared_locals_ = outer_declared_locals;
        local_glsl_names_ = outer_local_glsl_names;
        shared_values_ = outer_shared_values;
        return true;
    }

    bool LowerWhile(const Statement& statement) {
        if (statement.a >= typed_.expressions.size() || statement.b >= typed_.statements.size()) {
            return false;
        }

        const auto condition = BuildExpression(statement.a);
        auto body_statements = LowerStatementList(statement.b);
        if (condition == GPU::IR::InvalidValueId || !body_statements.has_value()) {
            return false;
        }

        const auto body_block_id = AddBlock(std::move(*body_statements));
        EmitWhile(condition, body_block_id);
        return true;
    }

    bool LowerDoWhile(const Statement& statement) {
        if (statement.a >= typed_.statements.size() || statement.b >= typed_.expressions.size()) {
            return false;
        }

        auto body_statements = LowerStatementList(statement.a);
        const auto condition = BuildExpression(statement.b);
        if (!body_statements.has_value() || condition == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto body_block_id = AddBlock(std::move(*body_statements));
        EmitDoWhile(body_block_id, condition);
        return true;
    }

    bool LowerReturn(const Statement& statement) {
        if (statement.a == NoIndex) {
            EmitReturn(GPU::IR::InvalidValueId);
            return true;
        }

        if (statement.a >= typed_.expressions.size()) {
            return false;
        }

        const auto value = BuildExpression(statement.a);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitReturn(value);
        return true;
    }

    struct CallableBody {
        std::vector<GPU::IR::Statement> statements;
        std::vector<GPU::IR::Block> blocks;
    };

    std::optional<std::vector<GPU::IR::Statement>> LowerStatementList(uint32_t statement_id) {
        const auto previous_capture = capture_;
        const auto previous_local_values = local_values_;
        const auto previous_declared_locals = declared_locals_;
        const auto previous_shared_values = shared_values_;
        std::vector<GPU::IR::Statement> captured;
        capture_ = &captured;
        const auto ok = LowerStatement(statement_id);
        capture_ = previous_capture;
        local_values_ = previous_local_values;
        declared_locals_ = previous_declared_locals;
        shared_values_ = previous_shared_values;
        if (!ok) {
            return std::nullopt;
        }

        return captured;
    }

    std::optional<CallableBody> LowerCallableStatementList(uint32_t statement_id) {
        const auto previous_capture = capture_;
        const auto previous_callable_blocks = callable_blocks_;
        std::vector<GPU::IR::Statement> captured;
        std::vector<GPU::IR::Block> blocks;
        capture_ = &captured;
        callable_blocks_ = &blocks;
        const auto ok = LowerStatement(statement_id);
        capture_ = previous_capture;
        callable_blocks_ = previous_callable_blocks;
        if (!ok) {
            return std::nullopt;
        }

        return CallableBody{std::move(captured), std::move(blocks)};
    }

    std::optional<std::vector<GPU::IR::Statement>> LowerStatementListKeepingLocals(uint32_t statement_id) {
        const auto previous_capture = capture_;
        const auto previous_shared_values = shared_values_;
        std::vector<GPU::IR::Statement> captured;
        capture_ = &captured;
        const auto ok = LowerStatement(statement_id);
        capture_ = previous_capture;
        shared_values_ = previous_shared_values;
        if (!ok) {
            return std::nullopt;
        }

        return captured;
    }

    GPU::IR::BlockId AddBlock(std::vector<GPU::IR::Statement> statements) {
        if (callable_blocks_ == nullptr) {
            return builder_.AddBlock(std::move(statements));
        }

        GPU::IR::Block block;
        block.id = static_cast<GPU::IR::BlockId>(callable_blocks_->size());
        block.statements = std::move(statements);
        callable_blocks_->push_back(std::move(block));
        return callable_blocks_->back().id;
    }

    void EmitStore(GPU::IR::ValueId target, GPU::IR::ValueId value) {
        if (capture_ == nullptr) {
            builder_.Store(target, value);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Store;
        statement.target = target;
        statement.value = value;
        capture_->push_back(statement);
    }

    void EmitExpression(GPU::IR::ValueId value) {
        if (capture_ == nullptr) {
            builder_.Expression(value);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Expression;
        statement.value = value;
        capture_->push_back(statement);
    }

    void EmitLocalDeclaration(GPU::IR::Type type, std::string name, GPU::IR::ValueId initializer) {
        if (capture_ == nullptr) {
            builder_.DeclareLocal(type, std::move(name), initializer);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::LocalDeclaration;
        statement.localType = type;
        statement.localName = std::move(name);
        statement.initializer = initializer;
        capture_->push_back(std::move(statement));
    }

    bool EmitBarrier(uint32_t kind) {
        GPU::IR::BarrierKind barrier_kind{};
        switch (kind) {
        case kBarrierWorkgroup:
            barrier_kind = GPU::IR::BarrierKind::Workgroup;
            break;
        case kBarrierMemory:
            barrier_kind = GPU::IR::BarrierKind::Memory;
            break;
        case kBarrierFull:
            barrier_kind = GPU::IR::BarrierKind::Full;
            break;
        default:
            return false;
        }

        if (capture_ == nullptr) {
            builder_.Barrier(barrier_kind);
            return true;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Barrier;
        statement.barrierKind = barrier_kind;
        capture_->push_back(statement);
        return true;
    }

    void EmitSharedMemoryDeclaration(GPU::IR::Type type, uint32_t count, std::string name) {
        if (capture_ == nullptr) {
            builder_.SharedMemoryDecl(type, count, std::move(name));
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::SharedMemoryDecl;
        statement.sharedType = std::move(type);
        statement.sharedCount = count;
        statement.sharedName = std::move(name);
        capture_->push_back(std::move(statement));
    }

    void EmitIf(GPU::IR::ValueId condition, GPU::IR::BlockId then_block_id, GPU::IR::BlockId else_block_id) {
        if (capture_ == nullptr) {
            builder_.If(condition, then_block_id, else_block_id);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::If;
        statement.condition = condition;
        statement.thenBlock = then_block_id;
        statement.elseBlock = else_block_id;
        capture_->push_back(statement);
    }

    void EmitFor(GPU::IR::BlockId init_block_id, GPU::IR::ValueId condition,
                 GPU::IR::BlockId step_block_id, GPU::IR::BlockId body_block_id) {
        if (capture_ == nullptr) {
            builder_.For(init_block_id, condition, step_block_id, body_block_id);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::For;
        statement.initBlock = init_block_id;
        statement.condition = condition;
        statement.stepBlock = step_block_id;
        statement.bodyBlock = body_block_id;
        capture_->push_back(statement);
    }

    void EmitWhile(GPU::IR::ValueId condition, GPU::IR::BlockId body_block_id) {
        if (capture_ == nullptr) {
            builder_.While(condition, body_block_id);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::While;
        statement.condition = condition;
        statement.bodyBlock = body_block_id;
        capture_->push_back(statement);
    }

    void EmitDoWhile(GPU::IR::BlockId body_block_id, GPU::IR::ValueId condition) {
        if (capture_ == nullptr) {
            builder_.DoWhile(body_block_id, condition);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::DoWhile;
        statement.bodyBlock = body_block_id;
        statement.condition = condition;
        capture_->push_back(statement);
    }

    void EmitBreak() {
        if (capture_ == nullptr) {
            builder_.Break();
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Break;
        capture_->push_back(statement);
    }

    void EmitContinue() {
        if (capture_ == nullptr) {
            builder_.Continue();
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Continue;
        capture_->push_back(statement);
    }

    void EmitReturn(GPU::IR::ValueId value) {
        if (capture_ == nullptr) {
            builder_.Return(value);
            return;
        }

        GPU::IR::Statement statement;
        statement.kind = GPU::IR::Statement::Kind::Return;
        statement.value = value;
        capture_->push_back(statement);
    }

    GPU::IR::ValueId BuildExpression(uint32_t expression_id) {
        if (expression_id >= typed_.expressions.size()) {
            return InvalidValue("expression index " + std::to_string(expression_id) +
                                " is outside the section 7 expression table");
        }

        const auto& expression = typed_.expressions[expression_id];
        switch (expression.kind) {
        case kExpressionLiteral:
            return BuildLiteral(expression);
        case kExpressionLocal:
        case kExpressionParameter:
            return BuildLocalReference(expression);
        case kExpressionField:
            return BuildFieldReference(expression);
        case kExpressionResourceElement:
            return BuildResourceElement(expression);
        case kExpressionUnary:
            return BuildUnary(expression);
        case kExpressionBinary:
            return BuildBinary(expression);
        case kExpressionComparison:
            return BuildComparison(expression);
        case kExpressionLogical:
            return BuildLogical(expression);
        case kExpressionConditional:
            return BuildConditional(expression);
        case kExpressionConversion:
            return BuildConversion(expression);
        case kExpressionConstructor:
            return BuildConstructor(expression);
        case kExpressionIntrinsic:
            return BuildIntrinsic(expression);
        case kExpressionCallableCall:
            return BuildCallableCall(expression);
        case kExpressionSwizzle:
            return BuildSwizzle(expression);
        case kExpressionIndexAccess:
            return BuildIndexAccess(expression);
        case kExpressionMemberAccess:
            return BuildMemberAccess(expression);
        case kExpressionMatrixColumn:
            return BuildMatrixColumn(expression);
        case kExpressionBuiltin:
            return BuildBuiltin(expression);
        case kExpressionPushConstant:
            return BuildPushConstant(expression);
        case kExpressionSharedMemoryElement:
            return BuildSharedMemoryElement(expression);
        case kExpressionAtomic:
            return BuildAtomic(expression);
        case kExpressionTextureSample:
            return BuildTextureSample(expression);
        default:
            return InvalidValue("unsupported section 7 expression kind " + std::to_string(expression.kind) +
                                " at expression index " + std::to_string(expression_id));
        }
    }

    GPU::IR::ValueId BuildLiteral(const Expression& expression) {
        const auto type = ToModuleType(expression.type_id);
        const auto* literal = GetString(expression.name_id);
        if (!type.IsValid() || literal == nullptr) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Literal(type, *literal);
    }

    GPU::IR::ValueId BuildLocalReference(const Expression& expression) {
        const auto* name = GetString(expression.name_id);
        if (name == nullptr) {
            return GPU::IR::InvalidValueId;
        }

        const auto mapped = local_values_.find(*name);
        return mapped == local_values_.end() ? GPU::IR::InvalidValueId : mapped->second;
    }

    GPU::IR::ValueId BuildResourceElement(const Expression& expression) {
        const auto* name = GetString(expression.name_id);
        if (name == nullptr) {
            return InvalidValue("resource element expression has an invalid resource-name string id");
        }

        const auto resource = resources_by_name_.find(*name);
        if (resource == resources_by_name_.end()) {
            return InvalidValue("resource element expression references unknown resource '" + *name + "'");
        }

        const auto index = BuildExpression(expression.a);
        if (index == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        const auto info = resource_infos_by_name_.find(*name);
        if (info != resource_infos_by_name_.end() &&
            (info->second.kind == kResourceKindTexture2D || info->second.kind == kResourceKindTexture3D)) {
            return BuildTextureElement(info->second.id, index);
        }

        return builder_.ResourceElement(resource->second, index);
    }

    GPU::IR::ValueId BuildFieldReference(const Expression& expression) {
        const auto instance = BuildExpression(expression.a);
        const auto result_type = ToModuleType(expression.type_id);
        const auto* member = GetString(expression.name_id);
        if (instance == GPU::IR::InvalidValueId || !result_type.IsValid() ||
            member == nullptr || member->empty()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.MemberAccess(instance, result_type, SanitizeGlslIdentifier(*member));
    }

    GPU::IR::ValueId BuildUnary(const Expression& expression) {
        GPU::IR::UnaryOp op{};
        if (!TryMapUnaryOp(expression.op, &op) || expression.a >= typed_.expressions.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto value = BuildExpression(expression.a);
        if (value == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Unary(op, value);
    }

    GPU::IR::ValueId BuildBinary(const Expression& expression) {
        GPU::IR::BinaryOp op{};
        if (!TryMapBinaryOp(expression.op, &op)) {
            return GPU::IR::InvalidValueId;
        }

        const auto left = BuildExpression(expression.a);
        const auto right = BuildExpression(expression.b);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Binary(op, left, right);
    }

    GPU::IR::ValueId BuildComparison(const Expression& expression) {
        GPU::IR::CompareOp op{};
        if (!TryMapCompareOp(expression.op, &op)) {
            return GPU::IR::InvalidValueId;
        }

        const auto left = BuildExpression(expression.a);
        const auto right = BuildExpression(expression.b);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Compare(op, left, right);
    }

    GPU::IR::ValueId BuildLogical(const Expression& expression) {
        GPU::IR::BinaryOp op{};
        if (!TryMapLogicalOp(expression.op, &op)) {
            return GPU::IR::InvalidValueId;
        }

        const auto left = BuildExpression(expression.a);
        const auto right = BuildExpression(expression.b);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Binary(op, left, right);
    }

    GPU::IR::ValueId BuildConditional(const Expression& expression) {
        const auto condition = BuildExpression(expression.a);
        const auto when_true = BuildExpression(expression.b);
        const auto when_false = BuildExpression(expression.c);
        if (condition == GPU::IR::InvalidValueId || when_true == GPU::IR::InvalidValueId ||
            when_false == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Ternary(condition, when_true, when_false);
    }

    GPU::IR::ValueId BuildConversion(const Expression& expression) {
        if (expression.a >= typed_.expressions.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto operand = BuildExpression(expression.a);
        const auto result_type = ToModuleType(expression.type_id);
        const auto conversion = ConversionName(result_type);
        if (operand == GPU::IR::InvalidValueId || !result_type.IsValid() || conversion.empty()) {
            return GPU::IR::InvalidValueId;
        }

        std::vector<GPU::IR::ValueId> arguments{operand};
        return builder_.Intrinsic(conversion, result_type, arguments);
    }

    GPU::IR::ValueId BuildBuiltin(const Expression& expression) {
        switch (expression.op) {
        case 1:
            return builder_.ThreadIndexX();
        case 2:
            return builder_.ThreadIndexY();
        case 3:
            return builder_.ThreadIndexZ();
        case 4:
            return builder_.LocalIndexX();
        case 5:
            return builder_.LocalIndexY();
        case 6:
            return builder_.LocalIndexZ();
        case 7:
            return builder_.GroupIdX();
        case 8:
            return builder_.GroupIdY();
        case 9:
            return builder_.GroupIdZ();
        case 10:
            return builder_.DispatchSizeX();
        case 11:
            return builder_.DispatchSizeY();
        case 12:
            return builder_.DispatchSizeZ();
        case 13:
            return builder_.GroupSizeX();
        case 14:
            return builder_.GroupSizeY();
        case 15:
            return builder_.GroupSizeZ();
        default:
            return GPU::IR::InvalidValueId;
        }
    }

    GPU::IR::ValueId BuildPushConstant(const Expression& expression) {
        if (const auto* name = GetString(expression.name_id)) {
            const auto resource = resources_by_name_.find(*name);
            if (resource != resources_by_name_.end()) {
                return builder_.PushConstant(resource->second);
            }
        }

        const auto by_binding = resources_by_binding_.find(expression.op);
        return by_binding == resources_by_binding_.end()
                   ? GPU::IR::InvalidValueId
                   : builder_.PushConstant(by_binding->second);
    }

    GPU::IR::ValueId BuildConstructor(const Expression& expression) {
        const auto result_type = ToModuleType(expression.type_id);
        const auto constructor = ConstructorName(result_type);
        if (!result_type.IsValid() || constructor.empty()) {
            return GPU::IR::InvalidValueId;
        }

        auto arguments = BuildArguments(expression);
        if (!arguments.has_value() || arguments->empty()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Intrinsic(constructor, result_type, *arguments);
    }

    GPU::IR::ValueId BuildIntrinsic(const Expression& expression) {
        const auto* symbol = GetString(expression.name_id);
        const auto intrinsic = symbol == nullptr ? std::string{} : IntrinsicName(*symbol);
        const auto result_type = ToModuleType(expression.type_id);
        if (intrinsic.empty() || !result_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        auto arguments = BuildArguments(expression);
        if (!arguments.has_value()) {
            return GPU::IR::InvalidValueId;
        }

        if (intrinsic == "matrix_multiply") {
            if (arguments->size() != 2) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.Binary(GPU::IR::BinaryOp::Mul, (*arguments)[0], (*arguments)[1]);
        }

        if (intrinsic == "clamp01") {
            if (arguments->size() != 1) {
                return GPU::IR::InvalidValueId;
            }

            const auto zero = builder_.Literal(result_type, "0");
            const auto one = builder_.Literal(result_type, "1");
            if (zero == GPU::IR::InvalidValueId || one == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            std::vector<GPU::IR::ValueId> clamp_arguments{(*arguments)[0], zero, one};
            return builder_.Intrinsic("clamp", result_type, clamp_arguments);
        }

        if (intrinsic == "clamp" && IsVectorType(result_type)) {
            if (arguments->size() != 3) {
                return GPU::IR::InvalidValueId;
            }

            const auto min_value = MaybeSplatScalarArgument(result_type, (*arguments)[1]);
            const auto max_value = MaybeSplatScalarArgument(result_type, (*arguments)[2]);
            if (min_value == GPU::IR::InvalidValueId || max_value == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            std::vector<GPU::IR::ValueId> clamp_arguments{(*arguments)[0], min_value, max_value};
            return builder_.Intrinsic("clamp", result_type, clamp_arguments);
        }

        return builder_.Intrinsic(intrinsic, result_type, *arguments);
    }

    GPU::IR::ValueId BuildTextureSample(const Expression& expression) {
        if (expression.argument_count != (expression.op == 1 ? 4u : 3u) ||
            expression.first_argument == NoIndex ||
            expression.first_argument > typed_.arguments.size() ||
            expression.argument_count > typed_.arguments.size() - expression.first_argument) {
            return InvalidValue("texture sample expression has invalid argument range or arity");
        }

        const auto texture_expression_id = typed_.arguments[expression.first_argument];
        const auto sampler_expression_id = typed_.arguments[expression.first_argument + 1];
        const auto uv_expression_id = typed_.arguments[expression.first_argument + 2];
        if (texture_expression_id >= typed_.expressions.size() ||
            sampler_expression_id >= typed_.expressions.size() ||
            uv_expression_id >= typed_.expressions.size()) {
            return InvalidValue("texture sample expression references an argument outside the expression table");
        }

        const auto texture = TextureResourceFromExpression(typed_.expressions[texture_expression_id]);
        const auto sampler = ResourceFromExpression(typed_.expressions[sampler_expression_id]);
        if (!texture.has_value() || texture->kind != kResourceKindTexture2D ||
            texture->access != kAccessSample ||
            !sampler.has_value() || sampler->kind != kResourceKindSampler) {
            return InvalidValue("texture sample expression requires a sampled texture2D resource and sampler resource");
        }

        const auto result_type = TextureSampleResultType(expression.type_id);
        const auto uv = BuildExpression(uv_expression_id);
        if (!result_type.IsValid() || uv == GPU::IR::InvalidValueId) {
            return InvalidValue("texture sample expression has unsupported result type or UV expression");
        }

        if (expression.op == 0) {
            return builder_.TextureSample(texture->id, result_type, uv);
        }

        const auto lod_expression_id = typed_.arguments[expression.first_argument + 3];
        if (lod_expression_id >= typed_.expressions.size()) {
            return InvalidValue("texture SampleLevel expression references an LOD argument outside the expression table");
        }

        const auto lod = BuildExpression(lod_expression_id);
        if (lod == GPU::IR::InvalidValueId) {
            return InvalidValue("texture SampleLevel expression has an invalid LOD expression");
        }

        return builder_.TextureSampleLevel(texture->id, result_type, uv, lod);
    }

    GPU::IR::ValueId BuildCallableCall(const Expression& expression) {
        const auto* raw_name = GetString(expression.name_id);
        if (raw_name == nullptr || raw_name->empty()) {
            return GPU::IR::InvalidValueId;
        }

        const auto mapped = callable_names_.find(*raw_name);
        if (mapped == callable_names_.end()) {
            return GPU::IR::InvalidValueId;
        }

        const auto result_type = ToModuleType(expression.type_id);
        if (!result_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        auto arguments = BuildArguments(expression);
        if (!arguments.has_value()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Call(mapped->second, result_type, *arguments);
    }

    GPU::IR::ValueId BuildAtomic(const Expression& expression) {
        if (expression.a >= typed_.lvalues.size()) {
            return InvalidValue("atomic expression references l-value index " + std::to_string(expression.a) +
                                " outside the section 7 l-value table");
        }

        GPU::IR::AtomicOp op{};
        if (!TryMapAtomicOp(expression.op, &op)) {
            return InvalidValue("atomic expression uses unsupported operation " + std::to_string(expression.op));
        }

        auto arguments = BuildArguments(expression);
        if (!arguments.has_value() ||
            arguments->size() != (op == GPU::IR::AtomicOp::CompareExchange ? 2u : 1u)) {
            return InvalidValue("atomic expression has invalid argument range or arity");
        }

        const auto target = BuildLValueAddress(expression.a);
        const auto result_type = ToModuleType(expression.type_id);
        const auto target_type = ToModuleType(typed_.lvalues[expression.a].type_id);
        if (target == GPU::IR::InvalidValueId || !result_type.IsValid() ||
            result_type.kind != GPU::IR::Type::Kind::Int ||
            target_type.kind != GPU::IR::Type::Kind::Int) {
            return InvalidValue("atomic expression requires an int result and int addressable l-value target");
        }

        for (const auto argument : *arguments) {
            if (argument >= builder_.GetModule().values.size() ||
                builder_.GetModule().values[argument].type.kind != GPU::IR::Type::Kind::Int) {
                return InvalidValue("atomic expression arguments must be int values");
            }
        }

        return builder_.Atomic(op, result_type, target, *arguments);
    }

    GPU::IR::ValueId BuildSwizzle(const Expression& expression) {
        if (expression.a >= typed_.expressions.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto vector = BuildExpression(expression.a);
        const auto result_type = ToModuleType(expression.type_id);
        const auto* components = GetString(expression.name_id);
        if (vector == GPU::IR::InvalidValueId || !result_type.IsValid() || components == nullptr) {
            return GPU::IR::InvalidValueId;
        }

        auto normalized = NormalizeSwizzle(*components);
        if (normalized.empty()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.Swizzle(vector, result_type, std::move(normalized));
    }

    GPU::IR::ValueId BuildMemberAccess(const Expression& expression) {
        const auto instance = BuildExpression(expression.a);
        const auto result_type = MemberAccessResultType(expression);
        const auto* member = GetString(expression.name_id);
        if (instance == GPU::IR::InvalidValueId || !result_type.IsValid() ||
            member == nullptr || member->empty()) {
            return GPU::IR::InvalidValueId;
        }

        if (const auto swizzle = TextureStructFieldSwizzle(*member); swizzle != nullptr) {
            const auto& values = builder_.GetModule().values;
            if (instance < values.size() && values[instance].type.kind == GPU::IR::Type::Kind::Float4) {
                return builder_.Swizzle(instance, result_type, swizzle);
            }
        }

        return builder_.MemberAccess(instance, result_type, SanitizeGlslIdentifier(*member));
    }

    GPU::IR::ValueId BuildIndexAccess(const Expression& expression) {
        const auto instance = BuildExpression(expression.a);
        const auto index = BuildExpression(expression.b);
        const auto result_type = ToModuleType(expression.type_id);
        if (instance == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId || !result_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.IndexAccess(instance, index, result_type);
    }

    GPU::IR::ValueId BuildMatrixColumn(const Expression& expression) {
        return BuildIndexAccess(expression);
    }

    GPU::IR::ValueId BuildSharedMemoryElement(const Expression& expression) {
        const auto* name = GetString(expression.name_id);
        if (name == nullptr || expression.a >= typed_.expressions.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto shared = shared_values_.find(*name);
        if (shared == shared_values_.end()) {
            return GPU::IR::InvalidValueId;
        }

        const auto index = BuildExpression(expression.a);
        const auto type = ToModuleType(expression.type_id);
        if (index == GPU::IR::InvalidValueId || !type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.SharedMemoryElement(type, shared->second.glsl_name, index);
    }

    GPU::IR::ValueId BuildTextureElement(GPU::IR::ResourceId resource, GPU::IR::ValueId index) {
        if (resource >= builder_.GetModule().resources.size() ||
            builder_.GetModule().resources[resource].kind != GPU::IR::ResourceKind::Texture ||
            index >= builder_.GetModule().values.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto& texture = builder_.GetModule().resources[resource];
        const auto index_type = builder_.GetModule().values[index].type;
        const auto is_texture3d = texture.textureDimension == 3;
        if (!is_texture3d && index_type.kind == GPU::IR::Type::Kind::Int2) {
            const auto x = builder_.Swizzle(index, GPU::IR::Type::Int(), "x");
            const auto y = builder_.Swizzle(index, GPU::IR::Type::Int(), "y");
            if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.TextureElement(resource, x, y);
        }

        if (is_texture3d && index_type.kind == GPU::IR::Type::Kind::Int3) {
            const auto x = builder_.Swizzle(index, GPU::IR::Type::Int(), "x");
            const auto y = builder_.Swizzle(index, GPU::IR::Type::Int(), "y");
            const auto z = builder_.Swizzle(index, GPU::IR::Type::Int(), "z");
            if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId ||
                z == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.TextureElement3D(resource, x, y, z);
        }

        if (!is_texture3d && index_type.kind == GPU::IR::Type::Kind::Int) {
            const auto width = builder_.Literal(GPU::IR::Type::Int(),
                std::to_string(texture.width));
            if (width == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            const auto x = builder_.Binary(GPU::IR::BinaryOp::Mod, index, width);
            const auto y = builder_.Binary(GPU::IR::BinaryOp::Div, index, width);
            if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.TextureElement(resource, x, y);
        }

        return GPU::IR::InvalidValueId;
    }

    std::optional<RegisteredResource> ResourceFromExpression(const Expression& expression) const {
        if (expression.kind != kExpressionLocal && expression.kind != kExpressionParameter) {
            return std::nullopt;
        }

        const auto* name = GetString(expression.name_id);
        if (name == nullptr) {
            return std::nullopt;
        }

        const auto found = resource_infos_by_name_.find(*name);
        return found == resource_infos_by_name_.end() ? std::nullopt : std::optional<RegisteredResource>(found->second);
    }

    std::optional<RegisteredResource> TextureResourceFromExpression(const Expression& expression) const {
        auto resource = ResourceFromExpression(expression);
        if (resource.has_value() &&
            (resource->kind == kResourceKindTexture2D || resource->kind == kResourceKindTexture3D)) {
            return resource;
        }

        return std::nullopt;
    }

    std::optional<std::vector<GPU::IR::ValueId>> BuildArguments(const Expression& expression) {
        if (expression.argument_count == 0) {
            if (expression.first_argument != NoIndex) {
                return std::nullopt;
            }

            return std::vector<GPU::IR::ValueId>{};
        }

        if (expression.first_argument == NoIndex ||
            expression.first_argument > typed_.arguments.size() ||
            expression.argument_count > typed_.arguments.size() - expression.first_argument) {
            return std::nullopt;
        }

        std::vector<GPU::IR::ValueId> arguments;
        arguments.reserve(expression.argument_count);
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            const auto expression_id = typed_.arguments[expression.first_argument + i];
            const auto value = BuildExpression(expression_id);
            if (value == GPU::IR::InvalidValueId) {
                return std::nullopt;
            }

            arguments.push_back(value);
        }

        return arguments;
    }

    GPU::IR::ValueId BuildLValueAddress(uint32_t lvalue_id) {
        if (lvalue_id >= typed_.lvalues.size()) {
            return InvalidValue("l-value index " + std::to_string(lvalue_id) +
                                " is outside the section 7 l-value table");
        }

        const auto& lvalue = typed_.lvalues[lvalue_id];
        switch (lvalue.kind) {
        case kLValueResourceElement:
            return BuildResourceLValueAddress(lvalue);
        case kLValueSharedMemoryElement:
            return BuildSharedMemoryLValueAddress(lvalue);
        case kLValueSwizzle:
            return BuildSwizzleLValueAddress(lvalue);
        case kLValueIndexAccess:
            return BuildIndexLValueAddress(lvalue);
        case kLValueMatrixColumn:
            return BuildMatrixColumnLValueAddress(lvalue);
        case kLValueField:
        case kLValueMemberAccess: {
            const auto instance = BuildLValueRead(lvalue.a);
            const auto type = ToModuleType(lvalue.type_id);
            const auto* member = GetString(lvalue.name_id);
            if (instance == GPU::IR::InvalidValueId || !type.IsValid() ||
                member == nullptr || member->empty()) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.MemberAccess(instance, type, SanitizeGlslIdentifier(*member));
        }
        default:
            return InvalidValue("unsupported addressable l-value kind " + std::to_string(lvalue.kind));
        }
    }

    GPU::IR::ValueId BuildResourceLValueAddress(const LValue& lvalue) {
        const auto* name = GetString(lvalue.name_id);
        if (name == nullptr) {
            return InvalidValue("resource l-value has an invalid resource-name string id");
        }

        const auto resource = resources_by_name_.find(*name);
        if (resource == resources_by_name_.end()) {
            return InvalidValue("resource l-value references unknown resource '" + *name + "'");
        }

        const auto index = BuildExpression(lvalue.a);
        if (index == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        const auto info = resource_infos_by_name_.find(*name);
        if (info != resource_infos_by_name_.end() &&
            (info->second.kind == kResourceKindTexture2D || info->second.kind == kResourceKindTexture3D)) {
            return BuildTextureElement(info->second.id, index);
        }

        return builder_.ResourceElement(resource->second, index);
    }

    GPU::IR::ValueId BuildSharedMemoryLValueAddress(const LValue& lvalue) {
        const auto* name = GetString(lvalue.name_id);
        if (name == nullptr) {
            return InvalidValue("shared-memory l-value has an invalid name string id");
        }

        const auto shared = shared_values_.find(*name);
        if (shared == shared_values_.end()) {
            return InvalidValue("shared-memory l-value references unknown shared memory '" + *name + "'");
        }

        const auto index = BuildExpression(lvalue.a);
        const auto type = ToModuleType(lvalue.type_id);
        if (index == GPU::IR::InvalidValueId || !type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.SharedMemoryElement(type, shared->second.glsl_name, index);
    }

    GPU::IR::ValueId BuildSwizzleLValueAddress(const LValue& lvalue) {
        const auto vector = BuildExpression(lvalue.a);
        const auto type = ToModuleType(lvalue.type_id);
        const auto* components = GetString(lvalue.name_id);
        if (vector == GPU::IR::InvalidValueId || !type.IsValid() ||
            components == nullptr || components->empty()) {
            return InvalidValue("swizzle l-value has invalid vector, result type, or component string");
        }

        auto normalized = NormalizeSwizzle(*components);
        if (normalized.empty() || HasDuplicateSwizzleComponent(normalized)) {
            return InvalidValue("swizzle l-value '" + *components +
                                "' is invalid or writes the same component more than once");
        }

        return builder_.Swizzle(vector, type, std::move(normalized));
    }

    GPU::IR::ValueId BuildIndexLValueAddress(const LValue& lvalue) {
        const auto instance = BuildLValueRead(lvalue.a);
        const auto index = BuildExpression(lvalue.b);
        const auto type = ToModuleType(lvalue.type_id);
        if (instance == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId || !type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.IndexAccess(instance, index, type);
    }

    GPU::IR::ValueId BuildMatrixColumnLValueAddress(const LValue& lvalue) {
        const auto instance = BuildExpression(lvalue.a);
        const auto index = BuildExpression(lvalue.b);
        const auto type = ToModuleType(lvalue.type_id);
        if (instance == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId || !type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder_.IndexAccess(instance, index, type);
    }

    GPU::IR::ValueId BuildLValueRead(uint32_t lvalue_id) {
        if (lvalue_id >= typed_.lvalues.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto& lvalue = typed_.lvalues[lvalue_id];
        if (IsLocalLikeLValue(lvalue.kind)) {
            const auto* name = GetString(lvalue.name_id);
            if (name == nullptr) {
                return GPU::IR::InvalidValueId;
            }

            const auto mapped = local_values_.find(*name);
            if (mapped != local_values_.end()) {
                return mapped->second;
            }
            const auto declared = declared_locals_.find(*name);
            if (declared != declared_locals_.end()) {
                auto glsl_name = local_glsl_names_.find(*name);
                auto value = builder_.LocalVariable(
                    declared->second,
                    glsl_name == local_glsl_names_.end() ? SanitizeGlslIdentifier(*name) : glsl_name->second);
                if (value != GPU::IR::InvalidValueId) {
                    local_values_[*name] = value;
                }
                return value;
            }
            return GPU::IR::InvalidValueId;
        }

        return BuildLValueAddress(lvalue_id);
    }

    GPU::IR::Type ToModuleType(uint32_t type_id) const {
        if (type_id >= typed_.types.size()) {
            return {};
        }

        const auto& type = typed_.types[type_id];
        if (type.kind == kTypePrimitive) {
            switch (type.a) {
            case kPrimitiveBool:
                return GPU::IR::Type::Bool();
            case kPrimitiveInt:
                return GPU::IR::Type::Int();
            case kPrimitiveUInt:
                return GPU::IR::Type::UInt();
            case kPrimitiveFloat:
                return GPU::IR::Type::Float();
            default:
                return {};
            }
        }

        if (type.kind == kTypeVector) {
            const auto element = ToModuleType(type.a);
            if (element.kind == GPU::IR::Type::Kind::Int) {
                switch (type.b) {
                case 2:
                    return GPU::IR::Type::Int2();
                case 3:
                    return GPU::IR::Type::Int3();
                case 4:
                    return GPU::IR::Type::Int4();
                default:
                    return {};
                }
            }
            if (element.kind == GPU::IR::Type::Kind::UInt) {
                switch (type.b) {
                case 2:
                    return GPU::IR::Type::UInt2();
                case 3:
                    return GPU::IR::Type::UInt3();
                case 4:
                    return GPU::IR::Type::UInt4();
                default:
                    return {};
                }
            }
            if (element.kind == GPU::IR::Type::Kind::Bool) {
                switch (type.b) {
                case 2:
                    return GPU::IR::Type::Bool2();
                case 3:
                    return GPU::IR::Type::Bool3();
                case 4:
                    return GPU::IR::Type::Bool4();
                default:
                    return {};
                }
            }
            if (element.kind == GPU::IR::Type::Kind::Float) {
                switch (type.b) {
                case 2:
                    return GPU::IR::Type::Float2();
                case 3:
                    return GPU::IR::Type::Float3();
                case 4:
                    return GPU::IR::Type::Float4();
                default:
                    return {};
                }
            }

            return {};
        }

        if (type.kind == kTypeMatrix) {
            const auto element = ToModuleType(type.a);
            if (element.kind != GPU::IR::Type::Kind::Float) {
                return {};
            }

            if (type.b == 2 && type.c == 2) {
                return GPU::IR::Type::Float2x2();
            }
            if (type.b == 3 && type.c == 3) {
                return GPU::IR::Type::Float3x3();
            }
            if (type.b == 4 && type.c == 4) {
                return GPU::IR::Type::Float4x4();
            }

            return {};
        }

        if (type.kind == kTypeStruct) {
            return StructType(type.a);
        }

        if (type.kind == kTypeArray) {
            return ToModuleType(type.a);
        }

        if (type.kind == kTypeVoid) {
            return GPU::IR::Type::Void();
        }

        return {};
    }

    GPU::IR::Type TextureElementType(GPU::IR::Type declared_type) const {
        if (declared_type.kind == GPU::IR::Type::Kind::Struct) {
            return GPU::IR::Type::Float4();
        }

        return declared_type;
    }

    GPU::IR::Type TextureElementTypeFromName(const std::string& name) const {
        const auto declared = Feather::TypedIR::TypeFromName(name);
        if (declared.IsValid()) {
            return TextureElementType(declared);
        }

        if (LooksLikeRgbaStructName(name)) {
            return GPU::IR::Type::Float4();
        }

        return {};
    }

    GPU::IR::Type TextureSampleResultType(uint32_t type_id) const {
        if (type_id >= typed_.types.size()) {
            return {};
        }

        const auto& type = typed_.types[type_id];
        if (type.kind == kTypeStruct && type.a < typed_.structs.size()) {
            const auto& structure = typed_.structs[type.a];
            const auto* simple = GetString(structure.name_id);
            const auto* qualified = GetString(structure.fully_qualified_name_id);
            if ((simple != nullptr && LooksLikeRgbaStructName(*simple)) ||
                (qualified != nullptr && LooksLikeRgbaStructName(*qualified))) {
                return GPU::IR::Type::Float4();
            }
        }

        return TextureElementType(ToModuleType(type_id));
    }

    GPU::IR::Type MemberAccessResultType(const Expression& expression) const {
        auto result_type = ToModuleType(expression.type_id);
        if (result_type.IsValid()) {
            return result_type;
        }

        const auto* member = GetString(expression.name_id);
        if (member != nullptr && TextureStructFieldSwizzle(*member) != nullptr) {
            return GPU::IR::Type::Float();
        }

        return {};
    }

    GPU::IR::ValueId MaybeSplatScalarArgument(GPU::IR::Type target_type, GPU::IR::ValueId argument) {
        if (argument >= builder_.GetModule().values.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto& argument_type = builder_.GetModule().values[argument].type;
        if (argument_type.kind == target_type.kind) {
            return argument;
        }
        if (argument_type.kind != GPU::IR::Type::Kind::Float || !IsVectorType(target_type)) {
            return GPU::IR::InvalidValueId;
        }

        const auto constructor = ConstructorName(target_type);
        if (constructor.empty()) {
            return GPU::IR::InvalidValueId;
        }

        std::vector<GPU::IR::ValueId> constructor_arguments{argument};
        return builder_.Intrinsic(constructor, target_type, constructor_arguments);
    }

    static bool IsVectorType(GPU::IR::Type type) {
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool2:
        case GPU::IR::Type::Kind::Bool3:
        case GPU::IR::Type::Kind::Bool4:
        case GPU::IR::Type::Kind::Int2:
        case GPU::IR::Type::Kind::Int3:
        case GPU::IR::Type::Kind::Int4:
        case GPU::IR::Type::Kind::UInt2:
        case GPU::IR::Type::Kind::UInt3:
        case GPU::IR::Type::Kind::UInt4:
        case GPU::IR::Type::Kind::Float2:
        case GPU::IR::Type::Kind::Float3:
        case GPU::IR::Type::Kind::Float4:
            return true;
        default:
            return false;
        }
    }

    const std::string* GetString(uint32_t id) const {
        return id < typed_.strings.size() ? &typed_.strings[id] : nullptr;
    }

    static std::string ConstructorName(GPU::IR::Type type) {
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool2:
            return "bvec2";
        case GPU::IR::Type::Kind::Bool3:
            return "bvec3";
        case GPU::IR::Type::Kind::Bool4:
            return "bvec4";
        case GPU::IR::Type::Kind::Int2:
            return "ivec2";
        case GPU::IR::Type::Kind::Int3:
            return "ivec3";
        case GPU::IR::Type::Kind::Int4:
            return "ivec4";
        case GPU::IR::Type::Kind::UInt2:
            return "uvec2";
        case GPU::IR::Type::Kind::UInt3:
            return "uvec3";
        case GPU::IR::Type::Kind::UInt4:
            return "uvec4";
        case GPU::IR::Type::Kind::Float2:
            return "vec2";
        case GPU::IR::Type::Kind::Float3:
            return "vec3";
        case GPU::IR::Type::Kind::Float4:
            return "vec4";
        case GPU::IR::Type::Kind::Float2x2:
            return "mat2";
        case GPU::IR::Type::Kind::Float3x3:
            return "mat3";
        case GPU::IR::Type::Kind::Float4x4:
            return "mat4";
        case GPU::IR::Type::Kind::Struct:
            return type.typeName;
        default:
            return {};
        }
    }

    static std::string ConversionName(GPU::IR::Type type) {
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool:
            return "bool";
        case GPU::IR::Type::Kind::Int:
            return "int";
        case GPU::IR::Type::Kind::UInt:
            return "uint";
        case GPU::IR::Type::Kind::Float:
            return "float";
        default:
            return ConstructorName(type);
        }
    }

    static std::string IntrinsicName(const std::string& symbol) {
        if (symbol == "global::Feather.Math.ShaderMath.Sin" || symbol == "global::Feather.Math.Hlsl.Sin") {
            return "sin";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Cos" || symbol == "global::Feather.Math.Hlsl.Cos") {
            return "cos";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Tan" || symbol == "global::Feather.Math.Hlsl.Tan") {
            return "tan";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Exp" || symbol == "global::Feather.Math.Hlsl.Exp") {
            return "exp";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Log" || symbol == "global::Feather.Math.Hlsl.Log") {
            return "log";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Sqrt" || symbol == "global::Feather.Math.Hlsl.Sqrt") {
            return "sqrt";
        }
        if (symbol == "global::Feather.Math.ShaderMath.InverseSqrt" ||
            symbol == "global::Feather.Math.Hlsl.InverseSqrt") {
            return "inversesqrt";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Length" || symbol == "global::Feather.Math.Hlsl.Length") {
            return "length";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Normalize" || symbol == "global::Feather.Math.Hlsl.Normalize") {
            return "normalize";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Abs" || symbol == "global::Feather.Math.Hlsl.Abs") {
            return "abs";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Floor" || symbol == "global::Feather.Math.Hlsl.Floor") {
            return "floor";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Ceil" || symbol == "global::Feather.Math.Hlsl.Ceil") {
            return "ceil";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Round") {
            return "round";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Fract" || symbol == "global::Feather.Math.Hlsl.Fract") {
            return "fract";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Saturate") {
            return "clamp01";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Pow" || symbol == "global::Feather.Math.Hlsl.Pow") {
            return "pow";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Min") {
            return "min";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Max") {
            return "max";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Clamp" || symbol == "global::Feather.Math.Hlsl.Clamp") {
            return "clamp";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Lerp" || symbol == "global::Feather.Math.Hlsl.Lerp" ||
            symbol == "global::Feather.Math.ShaderMath.Mix" || symbol == "global::Feather.Math.Hlsl.Mix") {
            return "mix";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Smoothstep") {
            return "smoothstep";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Dot" || symbol == "global::Feather.Math.Hlsl.Dot") {
            return "dot";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Cross" || symbol == "global::Feather.Math.Hlsl.Cross") {
            return "cross";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Mul" || symbol == "global::Feather.Math.Hlsl.Mul") {
            return "matrix_multiply";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Transpose" ||
            symbol == "global::Feather.Math.Hlsl.Transpose") {
            return "transpose";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Determinant") {
            return "determinant";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Inverse" ||
            symbol == "global::Feather.Math.Hlsl.Inverse") {
            return "inverse";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Hadamard") {
            return "matrixCompMult";
        }

        return {};
    }

    static std::string NormalizeSwizzle(const std::string& components) {
        std::string normalized;
        normalized.reserve(components.size());
        for (const auto component : components) {
            const auto lowered = static_cast<char>(
                std::tolower(static_cast<unsigned char>(component)));
            switch (lowered) {
            case 'x':
            case 'r':
            case 's':
                normalized.push_back('x');
                break;
            case 'y':
            case 'g':
            case 't':
                normalized.push_back('y');
                break;
            case 'z':
            case 'b':
            case 'p':
                normalized.push_back('z');
                break;
            case 'w':
            case 'a':
            case 'q':
                normalized.push_back('w');
                break;
            default:
                return {};
            }
        }

        return normalized;
    }

    static bool HasDuplicateSwizzleComponent(const std::string& components) {
        std::array<bool, 4> seen{};
        for (const auto component : components) {
            size_t index = 0;
            switch (component) {
            case 'x':
            case 'r':
            case 's':
                index = 0;
                break;
            case 'y':
            case 'g':
            case 't':
                index = 1;
                break;
            case 'z':
            case 'b':
            case 'p':
                index = 2;
                break;
            case 'w':
            case 'a':
            case 'q':
                index = 3;
                break;
            default:
                return true;
            }

            if (seen[index]) {
                return true;
            }
            seen[index] = true;
        }

        return false;
    }

    bool TryMapUnaryOp(uint32_t raw, GPU::IR::UnaryOp* op) const {
        if (op == nullptr) {
            return false;
        }

        switch (raw) {
        case 0:
            *op = GPU::IR::UnaryOp::Negate;
            return true;
        case 1:
            *op = GPU::IR::UnaryOp::LogicalNot;
            return true;
        case 2:
            *op = GPU::IR::UnaryOp::BitwiseNot;
            return true;
        default:
            return false;
        }
    }

    bool TryMapBinaryOp(uint32_t raw, GPU::IR::BinaryOp* op) const {
        if (op == nullptr) {
            return false;
        }

        switch (raw) {
        case 0:
            *op = GPU::IR::BinaryOp::Add;
            return true;
        case 1:
            *op = GPU::IR::BinaryOp::Sub;
            return true;
        case 2:
            *op = GPU::IR::BinaryOp::Mul;
            return true;
        case 3:
            *op = GPU::IR::BinaryOp::Div;
            return true;
        case 4:
            *op = GPU::IR::BinaryOp::Mod;
            return true;
        case 5:
            *op = GPU::IR::BinaryOp::BitAnd;
            return true;
        case 6:
            *op = GPU::IR::BinaryOp::BitOr;
            return true;
        case 7:
            *op = GPU::IR::BinaryOp::BitXor;
            return true;
        case 8:
            *op = GPU::IR::BinaryOp::ShiftLeft;
            return true;
        case 9:
            *op = GPU::IR::BinaryOp::ShiftRight;
            return true;
        default:
            return false;
        }
    }

    bool TryMapLogicalOp(uint32_t raw, GPU::IR::BinaryOp* op) const {
        if (op == nullptr) {
            return false;
        }

        switch (raw) {
        case 0:
            *op = GPU::IR::BinaryOp::LogicalAnd;
            return true;
        case 1:
            *op = GPU::IR::BinaryOp::LogicalOr;
            return true;
        default:
            return false;
        }
    }

    bool TryMapCompareOp(uint32_t raw, GPU::IR::CompareOp* op) const {
        if (op == nullptr) {
            return false;
        }

        switch (raw) {
        case 0:
            *op = GPU::IR::CompareOp::Equal;
            return true;
        case 1:
            *op = GPU::IR::CompareOp::NotEqual;
            return true;
        case 2:
            *op = GPU::IR::CompareOp::Less;
            return true;
        case 3:
            *op = GPU::IR::CompareOp::LessEqual;
            return true;
        case 4:
            *op = GPU::IR::CompareOp::Greater;
            return true;
        case 5:
            *op = GPU::IR::CompareOp::GreaterEqual;
            return true;
        default:
            return false;
        }
    }

    bool TryMapAtomicOp(uint32_t raw, GPU::IR::AtomicOp* op) const {
        if (op == nullptr) {
            return false;
        }

        switch (raw) {
        case 0:
            *op = GPU::IR::AtomicOp::Add;
            return true;
        case 1:
            *op = GPU::IR::AtomicOp::Sub;
            return true;
        case 2:
            *op = GPU::IR::AtomicOp::Min;
            return true;
        case 3:
            *op = GPU::IR::AtomicOp::Max;
            return true;
        case 4:
            *op = GPU::IR::AtomicOp::And;
            return true;
        case 5:
            *op = GPU::IR::AtomicOp::Or;
            return true;
        case 6:
            *op = GPU::IR::AtomicOp::Xor;
            return true;
        case 7:
            *op = GPU::IR::AtomicOp::Exchange;
            return true;
        case 8:
            *op = GPU::IR::AtomicOp::CompareExchange;
            return true;
        default:
            return false;
        }
    }

    std::string UniqueGlslName(const std::string& source_name) {
        auto base = SanitizeGlslIdentifier(source_name);
        if (base.empty()) {
            return {};
        }

        auto candidate = base;
        uint32_t suffix = 0;
        while (used_glsl_names_.find(candidate) != used_glsl_names_.end()) {
            ++suffix;
            candidate = base + "_" + std::to_string(suffix);
        }

        used_glsl_names_.insert(candidate);
        return candidate;
    }

    static std::string BufferName(uint32_t binding) {
        return "fe_" + std::to_string(binding);
    }

    static std::string PushConstantName(uint32_t binding) {
        return "pc_" + std::to_string(binding);
    }

    static std::string TextureName(uint32_t binding) {
        return "te_" + std::to_string(binding);
    }

    static bool LooksLikeRgbaStructName(const std::string& name) {
        return name.find("Rgba32") != std::string::npos ||
               name.find("Rgba") != std::string::npos;
    }

    static const char* TextureStructFieldSwizzle(const std::string& member) {
        if (member == "R" || member == "X" || member == "S") return "x";
        if (member == "G" || member == "Y" || member == "T") return "y";
        if (member == "B" || member == "Z" || member == "P") return "z";
        if (member == "A" || member == "W" || member == "Q") return "w";
        return nullptr;
    }

    const Module& typed_;
    const LoweringInputs& inputs_;
    std::string* error_;
    GPU::IR::ModuleBuilder builder_;
    std::unordered_map<std::string, GPU::IR::ResourceId> resources_by_name_;
    std::unordered_map<uint32_t, GPU::IR::ResourceId> resources_by_binding_;
    std::unordered_map<std::string, RegisteredResource> resource_infos_by_name_;
    std::unordered_map<uint32_t, RegisteredResource> resource_infos_by_binding_;
    std::unordered_map<std::string, GPU::IR::ValueId> local_values_;
    std::unordered_map<std::string, GPU::IR::Type> declared_locals_;
    std::unordered_map<std::string, std::string> local_glsl_names_;
    std::unordered_set<std::string> used_glsl_names_;
    std::unordered_map<std::string, std::string> callable_names_;
    struct SharedMemoryInfo {
        GPU::IR::Type type;
        std::string glsl_name;
    };
    std::unordered_map<std::string, SharedMemoryInfo> shared_values_;
    std::array<GPU::IR::ResourceId, 3> logical_size_resource_{
        GPU::IR::InvalidResourceId,
        GPU::IR::InvalidResourceId,
        GPU::IR::InvalidResourceId
    };
    std::vector<GPU::IR::Statement>* capture_ = nullptr;
    std::vector<GPU::IR::Block>* callable_blocks_ = nullptr;
};

} // namespace

std::unique_ptr<GPU::IR::Module> TryLowerToEasyGpuModule(
    const Module& typed, const LoweringInputs& inputs, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }

    return Lowerer(typed, inputs, error).Lower();
}

} // namespace Feather::TypedIR
