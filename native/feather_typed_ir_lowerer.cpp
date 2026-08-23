#include "feather_typed_ir_lowerer.h"

#include <algorithm>
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

constexpr uint32_t kDiagnosticExecutionHeat = 1;
constexpr uint32_t kDiagnosticLineValue = 2;
constexpr uint32_t kDiagnosticUbsan = 3;
constexpr uint32_t kDiagnosticPrintAssert = 4;
constexpr uint32_t kDiagnosticBranchDivergence = 5;
constexpr uint32_t kDiagnosticComputeTrace = 6;
constexpr uint32_t kDiagnosticCounterfactual = 7;
constexpr uint32_t kCounterfactualForceIfFalse = 1;

constexpr uint32_t kTraceEventFunctionEnter = 1;
constexpr uint32_t kTraceEventStatement = 2;
constexpr uint32_t kTraceEventValue = 3;
constexpr uint32_t kTraceEventBranchPredicate = 4;
constexpr uint32_t kTraceEventFunctionExit = 5;
constexpr uint32_t kTraceEventInvocationEnd = 6;
constexpr uint32_t kUbsanCheckFloatDivideByZero = 1u << 0;
constexpr uint32_t kUbsanCheckSqrtDomain = 1u << 1;
constexpr uint32_t kUbsanCheckLogDomain = 1u << 2;
constexpr uint32_t kUbsanCheckNonFinite = 1u << 3;
constexpr uint32_t kUbsanCheckBufferBounds = 1u << 4;

constexpr uint32_t kUbsanIssueDivideByZero = 1;
constexpr uint32_t kUbsanIssueSqrtDomain = 2;
constexpr uint32_t kUbsanIssueLogDomain = 3;
constexpr uint32_t kUbsanIssueNaN = 4;
constexpr uint32_t kUbsanIssueInfinity = 5;
constexpr uint32_t kUbsanIssueBufferOob = 6;

constexpr uint8_t kFunctionCompute1D = 0;
constexpr uint8_t kFunctionCompute2D = 1;
constexpr uint8_t kFunctionCompute3D = 2;
constexpr uint8_t kFunctionCallable = 5;

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
    GPU::IR::ResourceId element_count = GPU::IR::InvalidResourceId;
    GPU::IR::Type element_type;
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

std::optional<GPU::IR::CallableParameterDirection> ToCallableParameterDirection(uint8_t direction) {
    switch (direction) {
    case 0:
        return GPU::IR::CallableParameterDirection::In;
    case 1:
        return GPU::IR::CallableParameterDirection::Out;
    case 2:
        return GPU::IR::CallableParameterDirection::InOut;
    default:
        return std::nullopt;
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

        current_function_id_ = typed_.entry_function;

        if (!RegisterResources() || !ResolveCallableResourceBindings() || !RegisterCallables() ||
            !EmitBoundsCheckGuard(entry.kind) ||
            !EmitComputeTraceEntryStart(typed_.entry_function, entry.body_statement_index) ||
            !LowerStatement(entry.body_statement_index) ||
            !EmitComputeTraceEntryEnd(typed_.entry_function, entry.body_statement_index) ||
            (inputs_.diagnostic_mode == kDiagnosticBranchDivergence &&
             !branch_divergence_site_emitted_) ||
            (inputs_.diagnostic_mode == kDiagnosticCounterfactual &&
             !counterfactual_site_emitted_)) {
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
            GPU::IR::ResourceId element_count = GPU::IR::InvalidResourceId;
            GPU::IR::Type registered_type;
            if (resource.kind == kResourceKindBuffer) {
                const auto type = TypeFromName(resource.element_type);
                if (!type.IsValid()) {
                    return Fail("buffer resource '" + resource.name + "' uses unsupported element type '" +
                                resource.element_type + "'");
                }

                id = builder_.AddBuffer(resource.binding, type, ToResourceAccess(resource.access),
                                        BufferName(resource.binding));
                registered_type = type;
                if (inputs_.diagnostic_mode == kDiagnosticUbsan &&
                    (inputs_.diagnostic_flags & kUbsanCheckBufferBounds) != 0u &&
                    resource.binding != inputs_.diagnostic_binding) {
                    if (resource.element_count_data == nullptr || *resource.element_count_data == 0u) {
                        return Fail("UBSan buffer resource '" + resource.name +
                                    "' is missing its runtime element count");
                    }
                    const auto hidden_binding = UINT32_MAX - 16u - diagnostic_count_resource_count_++;
                    element_count = builder_.AddPushConstant(
                        hidden_binding,
                        GPU::IR::Type::UInt(),
                        "__feather_ubsan_buffer_count_" + std::to_string(resource.binding),
                        resource.element_count_data,
                        sizeof(uint32_t),
                        alignof(uint32_t));
                    if (element_count == GPU::IR::InvalidResourceId) {
                        return Fail("EasyGPU rejected the UBSan buffer-length push constant");
                    }
                }
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
            resource_infos_by_name_[resource.name] =
                RegisteredResource{id, resource.kind, resource.access, element_count, registered_type};
            resources_by_binding_[resource.binding] = id;
            resource_infos_by_binding_[resource.binding] =
                RegisteredResource{id, resource.kind, resource.access, element_count, registered_type};
        }

        if (inputs_.diagnostic_mode == kDiagnosticExecutionHeat ||
            inputs_.diagnostic_mode == kDiagnosticLineValue ||
            inputs_.diagnostic_mode == kDiagnosticUbsan ||
            inputs_.diagnostic_mode == kDiagnosticPrintAssert ||
            inputs_.diagnostic_mode == kDiagnosticBranchDivergence ||
            inputs_.diagnostic_mode == kDiagnosticComputeTrace) {
            const auto diagnostic = resources_by_binding_.find(inputs_.diagnostic_binding);
            if (diagnostic == resources_by_binding_.end() || inputs_.diagnostic_site_count == 0) {
                return Fail("diagnostic buffer is missing from the lowering inputs");
            }
            diagnostic_sites_resource_ = diagnostic->second;
            if (inputs_.diagnostic_mode == kDiagnosticLineValue &&
                inputs_.diagnostic_source_site >= inputs_.diagnostic_site_count) {
                return Fail("line-value source site is outside the configured diagnostic ABI");
            }
            if (inputs_.diagnostic_mode == kDiagnosticUbsan &&
                (inputs_.diagnostic_record_capacity == 0u || inputs_.diagnostic_flags == 0u)) {
                return Fail("UBSan stream capacity or enabled-check mask is invalid");
            }
            if (inputs_.diagnostic_mode == kDiagnosticPrintAssert &&
                (inputs_.diagnostic_record_capacity == 0u ||
                 inputs_.diagnostic_filter_mode > 1u ||
                 inputs_.diagnostic_logical_x == 0u ||
                 inputs_.diagnostic_logical_y == 0u ||
                 inputs_.diagnostic_logical_z == 0u ||
                 (inputs_.diagnostic_filter_mode == 1u &&
                  (inputs_.diagnostic_selected_x >= inputs_.diagnostic_logical_x ||
                   inputs_.diagnostic_selected_y >= inputs_.diagnostic_logical_y ||
                   inputs_.diagnostic_selected_z >= inputs_.diagnostic_logical_z)))) {
                return Fail("Print/Assert stream filter or immutable logical extent is invalid");
            }
            if (inputs_.diagnostic_mode == kDiagnosticBranchDivergence &&
                (inputs_.diagnostic_source_site >= inputs_.diagnostic_site_count ||
                 inputs_.diagnostic_record_capacity == 0u || inputs_.diagnostic_flags != 0x0fu)) {
                return Fail("branch-divergence source site, capacity, or subgroup feature contract is invalid");
            }
            if (inputs_.diagnostic_mode == kDiagnosticComputeTrace &&
                (inputs_.diagnostic_record_capacity == 0u || inputs_.diagnostic_flags != 0u)) {
                return Fail("compute-trace capacity or feature contract is invalid");
            }
        }

        if (inputs_.diagnostic_mode == kDiagnosticCounterfactual &&
            (inputs_.diagnostic_site_count == 0u ||
             inputs_.diagnostic_source_site >= inputs_.diagnostic_site_count ||
             inputs_.diagnostic_transform_kind != kCounterfactualForceIfFalse ||
             inputs_.diagnostic_flags != 0u)) {
            return Fail("counterfactual source site or transformation contract is invalid");
        }

        if (!RegisterBoundsCheckResources()) {
            return false;
        }

        return true;
    }

    bool RegisterBoundsCheckResources() {
        if (!inputs_.bounds_check && inputs_.diagnostic_mode != kDiagnosticPrintAssert) {
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

            logical_size_resource_[axis] = builder_.AddPushConstant(UINT32_MAX - axis, GPU::IR::Type::Int(),
                                                                    "__feather_dispatch_size_" + std::to_string(axis),
                                                                    values[axis], sizeof(int32_t), alignof(int32_t));
            if (logical_size_resource_[axis] == GPU::IR::InvalidResourceId) {
                return Fail("EasyGPU rejected hidden logical dispatch-size push constants");
            }
        }

        return true;
    }

    bool IsSupportedCallableResourceType(uint32_t type_id) const {
        if (type_id >= typed_.types.size()) {
            return false;
        }

        const auto& type = typed_.types[type_id];
        if (type.kind != kTypeResourceWrapper || type.b >= typed_.types.size()) {
            return false;
        }

        switch (type.a) {
        case kTypeResourceBuffer:
        case kTypeResourceTexture2D:
        case kTypeResourceTexture3D:
            return ToModuleType(type.b).IsValid();
        case kTypeResourceSampler:
            return true;
        default:
            return false;
        }
    }

    bool ResourceMatchesType(const RegisteredResource& resource, uint32_t type_id) const {
        if (!IsSupportedCallableResourceType(type_id)) {
            return false;
        }

        const auto& type = typed_.types[type_id];
        const auto expected_kind = type.a == kTypeResourceBuffer      ? kResourceKindBuffer
                                   : type.a == kTypeResourceTexture2D ? kResourceKindTexture2D
                                   : type.a == kTypeResourceTexture3D ? kResourceKindTexture3D
                                                                      : kResourceKindSampler;
        const auto expected_access = type.c == 0   ? kAccessRead
                                     : type.c == 1 ? kAccessWrite
                                     : type.c == 2 ? static_cast<uint8_t>(kAccessRead | kAccessWrite)
                                                   : kAccessSample;
        return resource.kind == expected_kind && resource.access == expected_access;
    }

    std::optional<RegisteredResource>
    ResolveResourceReference(uint32_t expression_id,
                             const std::unordered_map<std::string, RegisteredResource>& active_resources) const {
        if (expression_id >= typed_.expressions.size()) {
            return std::nullopt;
        }

        const auto& expression = typed_.expressions[expression_id];
        if (expression.kind != kExpressionLocal && expression.kind != kExpressionParameter) {
            return std::nullopt;
        }

        const auto* name = GetString(expression.name_id);
        if (name == nullptr) {
            return std::nullopt;
        }

        if (const auto active = active_resources.find(*name); active != active_resources.end()) {
            return active->second;
        }

        const auto info = resource_infos_by_name_.find(*name);
        return info == resource_infos_by_name_.end() ? std::nullopt : std::optional<RegisteredResource>{info->second};
    }

    std::optional<RegisteredResource> ResolveResourceReference(uint32_t expression_id) const {
        return ResolveResourceReference(expression_id, active_resource_parameters_);
    }

    bool CollectCallableCallsFromExpression(uint32_t expression_id, std::vector<uint32_t>& calls,
                                            std::unordered_set<uint32_t>& seen_expressions,
                                            std::unordered_set<uint32_t>& seen_lvalues) const {
        if (expression_id == NoIndex)
            return true;
        if (expression_id >= typed_.expressions.size())
            return false;
        if (!seen_expressions.insert(expression_id).second)
            return true;
        const auto& expression = typed_.expressions[expression_id];
        if (expression.kind == kExpressionCallableCall)
            calls.push_back(expression_id);

        auto expression_child = [&](uint32_t child) {
            return CollectCallableCallsFromExpression(child, calls, seen_expressions, seen_lvalues);
        };
        auto lvalue_child = [&](uint32_t child) {
            return CollectCallableCallsFromLValue(child, calls, seen_expressions, seen_lvalues);
        };
        auto argument_children = [&]() {
            if (expression.argument_count == 0)
                return expression.first_argument == NoIndex;
            if (expression.first_argument == NoIndex || expression.first_argument > typed_.arguments.size() ||
                expression.argument_count > typed_.arguments.size() - expression.first_argument)
                return false;
            for (uint32_t i = 0; i < expression.argument_count; ++i) {
                if (!expression_child(typed_.arguments[expression.first_argument + i]))
                    return false;
            }
            return true;
        };

        switch (expression.kind) {
        case kExpressionLiteral:
        case kExpressionLocal:
        case kExpressionParameter:
        case kExpressionBuiltin:
        case kExpressionPushConstant:
            return true;
        case kExpressionField:
        case kExpressionResourceElement:
        case kExpressionUnary:
        case kExpressionConversion:
        case kExpressionSwizzle:
        case kExpressionMemberAccess:
        case kExpressionSharedMemoryElement:
            return expression_child(expression.a);
        case kExpressionBinary:
        case kExpressionComparison:
        case kExpressionLogical:
        case kExpressionIndexAccess:
        case kExpressionMatrixColumn:
            return expression_child(expression.a) && expression_child(expression.b);
        case kExpressionConditional:
            return expression_child(expression.a) && expression_child(expression.b) && expression_child(expression.c);
        case kExpressionConstructor:
        case kExpressionIntrinsic:
        case kExpressionCallableCall:
        case kExpressionTextureSample:
            return argument_children();
        case kExpressionAtomic:
            return lvalue_child(expression.a) && argument_children();
        default:
            return false;
        }
    }

    bool CollectCallableCallsFromLValue(uint32_t lvalue_id, std::vector<uint32_t>& calls,
                                        std::unordered_set<uint32_t>& seen_expressions,
                                        std::unordered_set<uint32_t>& seen_lvalues) const {
        if (lvalue_id == NoIndex)
            return true;
        if (lvalue_id >= typed_.lvalues.size())
            return false;
        if (!seen_lvalues.insert(lvalue_id).second)
            return true;
        const auto& lvalue = typed_.lvalues[lvalue_id];
        auto expression_child = [&](uint32_t child) {
            return CollectCallableCallsFromExpression(child, calls, seen_expressions, seen_lvalues);
        };
        auto lvalue_child = [&](uint32_t child) {
            return CollectCallableCallsFromLValue(child, calls, seen_expressions, seen_lvalues);
        };
        switch (lvalue.kind) {
        case kLValueLocal:
        case kLValueParameter:
            return true;
        case kLValueField:
            return lvalue.a == NoIndex || lvalue_child(lvalue.a);
        case kLValueResourceElement:
        case kLValueSwizzle:
        case kLValueSharedMemoryElement:
            return expression_child(lvalue.a);
        case kLValueMemberAccess:
            return lvalue_child(lvalue.a);
        case kLValueIndexAccess:
            return lvalue_child(lvalue.a) && expression_child(lvalue.b);
        case kLValueMatrixColumn:
            return expression_child(lvalue.a) && expression_child(lvalue.b);
        default:
            return false;
        }
    }

    bool CollectCallableCallsFromStatement(uint32_t statement_id, std::vector<uint32_t>& calls,
                                           std::unordered_set<uint32_t>& seen_statements,
                                           std::unordered_set<uint32_t>& seen_expressions,
                                           std::unordered_set<uint32_t>& seen_lvalues) const {
        if (statement_id == NoIndex)
            return true;
        if (statement_id >= typed_.statements.size())
            return false;
        if (!seen_statements.insert(statement_id).second)
            return true;
        const auto& statement = typed_.statements[statement_id];
        auto stmt = [&](uint32_t child) {
            return CollectCallableCallsFromStatement(child, calls, seen_statements, seen_expressions, seen_lvalues);
        };
        auto expr = [&](uint32_t child) {
            return CollectCallableCallsFromExpression(child, calls, seen_expressions, seen_lvalues);
        };
        auto lvalue = [&](uint32_t child) {
            return CollectCallableCallsFromLValue(child, calls, seen_expressions, seen_lvalues);
        };
        switch (statement.kind) {
        case kStatementBlock:
            if (statement.child_count == 0)
                return statement.first_child == NoIndex;
            if (statement.first_child == NoIndex || statement.first_child > typed_.children.size() ||
                statement.child_count > typed_.children.size() - statement.first_child)
                return false;
            for (uint32_t i = 0; i < statement.child_count; ++i) {
                if (!stmt(typed_.children[statement.first_child + i]))
                    return false;
            }
            return true;
        case kStatementLocalDeclaration:
            return statement.a == NoIndex || expr(statement.a);
        case kStatementAssignment:
        case kStatementCompoundAssignment:
            return lvalue(statement.a) && expr(statement.b);
        case kStatementIf:
            return expr(statement.a) && stmt(statement.b) && (statement.c == NoIndex || stmt(statement.c));
        case kStatementFor:
            return (statement.a == NoIndex || stmt(statement.a)) && (statement.b == NoIndex || expr(statement.b)) &&
                   (statement.c == NoIndex || stmt(statement.c)) && stmt(statement.op);
        case kStatementWhile:
            return expr(statement.a) && stmt(statement.b);
        case kStatementDoWhile:
            return stmt(statement.a) && expr(statement.b);
        case kStatementReturn:
            return statement.a == NoIndex || expr(statement.a);
        case kStatementExpression:
            return expr(statement.a);
        case kStatementIncrementDecrement:
            return lvalue(statement.a);
        case kStatementBreak:
        case kStatementContinue:
        case kStatementBarrier:
        case kStatementSharedMemoryDeclaration:
            return true;
        default:
            return false;
        }
    }

    bool ResolveCallableResourceBindings() {
        std::unordered_map<uint32_t, std::vector<uint32_t>> calls_by_function;
        for (uint32_t function_id = 0; function_id < typed_.functions.size(); ++function_id) {
            std::unordered_set<uint32_t> seen_statements;
            std::unordered_set<uint32_t> seen_expressions;
            std::unordered_set<uint32_t> seen_lvalues;
            if (!CollectCallableCallsFromStatement(typed_.functions[function_id].body_statement_index,
                                                   calls_by_function[function_id], seen_statements, seen_expressions,
                                                   seen_lvalues)) {
                return Fail("callable resource binding could not traverse a function body");
            }
        }

        bool changed = true;
        while (changed) {
            changed = false;
            for (uint32_t owner_id = 0; owner_id < typed_.functions.size(); ++owner_id) {
                std::unordered_map<std::string, RegisteredResource> owner_resources;
                const auto& owner = typed_.functions[owner_id];
                if (owner.kind == kFunctionCallable) {
                    const auto* owner_name = GetString(owner.mangled_name_id);
                    if (owner_name == nullptr)
                        return false;
                    for (uint32_t i = 0; i < owner.parameter_count; ++i) {
                        const auto& parameter = typed_.parameters[owner.first_parameter + i];
                        const auto* parameter_name = GetString(parameter.name_id);
                        const auto binding = callable_resource_bindings_[*owner_name].find(i);
                        if (parameter_name != nullptr && binding != callable_resource_bindings_[*owner_name].end()) {
                            owner_resources[*parameter_name] = binding->second;
                        }
                    }
                }

                for (const auto expression_id : calls_by_function[owner_id]) {
                    const auto& expression = typed_.expressions[expression_id];

                    const auto* raw_name = GetString(expression.name_id);
                    const auto callable =
                        raw_name == nullptr ? typed_.callables.end() : typed_.callables.find(*raw_name);
                    if (callable == typed_.callables.end() ||
                        callable->second.function_index >= typed_.functions.size()) {
                        return Fail("callable resource binding references an unknown callable");
                    }

                    const auto& function = typed_.functions[callable->second.function_index];
                    if (function.parameter_count != expression.argument_count ||
                        (function.parameter_count > 0 &&
                         (function.first_parameter == NoIndex || expression.first_argument == NoIndex ||
                          function.first_parameter > typed_.parameters.size() ||
                          function.parameter_count > typed_.parameters.size() - function.first_parameter ||
                          expression.first_argument > typed_.arguments.size() ||
                          expression.argument_count > typed_.arguments.size() - expression.first_argument))) {
                        return Fail("callable resource binding has an invalid parameter or argument range");
                    }

                    for (uint32_t i = 0; i < function.parameter_count; ++i) {
                        const auto& parameter = typed_.parameters[function.first_parameter + i];
                        if (parameter.type_id >= typed_.types.size() ||
                            typed_.types[parameter.type_id].kind != kTypeResourceWrapper) {
                            continue;
                        }
                        if (!IsSupportedCallableResourceType(parameter.type_id) || parameter.direction != 0) {
                            return Fail(
                                "callable resource parameters require a supported input buffer, texture, or sampler");
                        }

                        const auto argument_id = typed_.arguments[expression.first_argument + i];
                        const auto resource = ResolveResourceReference(argument_id, owner_resources);
                        if (!resource.has_value()) {
                            continue;
                        }
                        if (!ResourceMatchesType(*resource, parameter.type_id)) {
                            return Fail("callable resource argument kind or access does not match its parameter");
                        }

                        auto& bindings = callable_resource_bindings_[*raw_name];
                        const auto existing = bindings.find(i);
                        if (existing != bindings.end() && existing->second.id != resource->id) {
                            return Fail("a callable resource parameter is called with multiple resource bindings");
                        }
                        if (existing == bindings.end()) {
                            bindings[i] = *resource;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (const auto& callable : typed_.callables) {
            if (callable.second.function_index >= typed_.functions.size()) {
                return false;
            }
            const auto& function = typed_.functions[callable.second.function_index];
            for (uint32_t i = 0; i < function.parameter_count; ++i) {
                const auto& parameter = typed_.parameters[function.first_parameter + i];
                if (parameter.type_id < typed_.types.size() &&
                    typed_.types[parameter.type_id].kind == kTypeResourceWrapper &&
                    callable_resource_bindings_[callable.first].find(i) ==
                        callable_resource_bindings_[callable.first].end()) {
                    return Fail("callable resource parameters must resolve to a kernel resource binding");
                }
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
            auto previous_active_resource_parameters = active_resource_parameters_;
            std::vector<GPU::IR::CallableParameter> parameters;
            parameters.reserve(function.parameter_count);

            if (function.parameter_count > 0) {
                if (function.first_parameter == NoIndex || function.first_parameter > typed_.parameters.size() ||
                    function.parameter_count > typed_.parameters.size() - function.first_parameter) {
                    return false;
                }
            }

            for (uint32_t i = 0; i < function.parameter_count; ++i) {
                const auto& parameter = typed_.parameters[function.first_parameter + i];
                const auto* parameter_name = GetString(parameter.name_id);
                if (parameter.type_id < typed_.types.size() &&
                    typed_.types[parameter.type_id].kind == kTypeResourceWrapper) {
                    const auto binding = callable_resource_bindings_[*raw_name].find(i);
                    if (parameter_name == nullptr || parameter_name->empty() ||
                        !IsSupportedCallableResourceType(parameter.type_id) || parameter.direction != 0 ||
                        binding == callable_resource_bindings_[*raw_name].end()) {
                        return false;
                    }

                    active_resource_parameters_[*parameter_name] = binding->second;
                    continue;
                }

                const auto parameter_type = ToModuleType(parameter.type_id);
                const auto parameter_direction = ToCallableParameterDirection(parameter.direction);
                if (parameter_name == nullptr || parameter_name->empty() || !parameter_type.IsValid() ||
                    !parameter_direction.has_value()) {
                    return false;
                }

                const auto sanitized_parameter_name = UniqueGlslName(*parameter_name);
                if (sanitized_parameter_name.empty()) {
                    return false;
                }

                parameters.push_back(
                    GPU::IR::CallableParameter{sanitized_parameter_name, parameter_type, *parameter_direction});
                local_values_[*parameter_name] = builder_.LocalVariable(parameter_type, sanitized_parameter_name);
                declared_locals_[*parameter_name] = parameter_type;
                local_glsl_names_[*parameter_name] = sanitized_parameter_name;
            }

            const auto previous_function_id = current_function_id_;
            current_function_id_ = function_id;
            auto body = LowerCallableStatementList(function.body_statement_index, function_id);
            current_function_id_ = previous_function_id;
            local_values_ = std::move(previous_local_values);
            declared_locals_ = std::move(previous_declared_locals);
            local_glsl_names_ = std::move(previous_local_glsl_names);
            shared_values_ = std::move(previous_shared_values);
            active_resource_parameters_ = std::move(previous_active_resource_parameters);
            if (!body.has_value()) {
                return false;
            }

            const auto callable_id = builder_.AddCallable(name, return_type, std::move(parameters),
                                                          std::move(body->statements), std::move(body->blocks));
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

        struct StatementIdScope {
            uint32_t& slot;
            uint32_t previous;
            ~StatementIdScope() { slot = previous; }
        } statement_scope{current_statement_id_, current_statement_id_};
        current_statement_id_ = statement_id;

        const auto& statement = typed_.statements[statement_id];
        if (statement.kind != kStatementBlock && diagnostic_site_suppression_depth_ == 0 &&
            !EmitDiagnosticSiteHit(statement_id)) {
            return false;
        }
        if (statement.kind != kStatementBlock &&
            !EmitComputeTraceEvent(
                statement_id,
                kTraceEventStatement,
                current_function_id_,
                ComputeTraceStatementSymbol(statement))) {
            return false;
        }
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

            const auto& expression = typed_.expressions[statement.a];
            const auto* expression_symbol = expression.kind == kExpressionIntrinsic
                                                ? GetString(expression.name_id)
                                                : nullptr;
            const bool is_gpu_debug_marker = expression_symbol != nullptr &&
                (*expression_symbol == "global::Feather.GpuDebug.Print" ||
                 *expression_symbol == "global::Feather.GpuDebug.Assert");
            auto value = BuildExpression(statement.a);
            if (value == GPU::IR::InvalidValueId) {
                return false;
            }

            value = MaterializeDiagnosticValue(
                statement_id, value, typed_.expressions[statement.a].type_id);
            if (value == GPU::IR::InvalidValueId) {
                return false;
            }

            if (!is_gpu_debug_marker) {
                EmitExpression(value);
            }
            return EmitLineValueRecord(
                       statement_id, value, typed_.expressions[statement.a].type_id) &&
                   EmitComputeTraceEvent(
                       statement_id,
                       kTraceEventValue,
                       current_function_id_,
                       ComputeTraceStatementSymbol(statement),
                       value,
                       typed_.expressions[statement.a].type_id);
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

    bool EmitDiagnosticSiteHit(uint32_t statement_id) {
        if (inputs_.diagnostic_mode != 1) {
            return true;
        }
        if (diagnostic_sites_resource_ == GPU::IR::InvalidResourceId ||
            statement_id >= inputs_.diagnostic_site_count) {
            return Fail("execution-heat diagnostic site is outside the configured buffer ABI");
        }

        const auto index = builder_.Literal(GPU::IR::Type::UInt(), std::to_string(statement_id) + "u");
        const auto target = builder_.ResourceElement(diagnostic_sites_resource_, index);
        const auto one = builder_.Literal(GPU::IR::Type::UInt(), "1u");
        if (index == GPU::IR::InvalidValueId || target == GPU::IR::InvalidValueId ||
            one == GPU::IR::InvalidValueId) {
            return Fail("execution-heat diagnostic counter address could not be lowered");
        }
        const std::array arguments{one};
        const auto increment = builder_.Atomic(
            GPU::IR::AtomicOp::Add,
            GPU::IR::Type::UInt(),
            target,
            arguments);
        if (increment == GPU::IR::InvalidValueId) {
            return Fail("execution-heat diagnostic atomic increment could not be lowered");
        }
        EmitExpression(increment);
        return true;
    }

    uint32_t ComputeTraceStatementSymbol(const Statement& statement) const {
        if (statement.name_id != NoIndex) {
            return statement.name_id;
        }
        if ((statement.kind == kStatementAssignment ||
             statement.kind == kStatementCompoundAssignment ||
             statement.kind == kStatementIncrementDecrement) &&
            statement.a < typed_.lvalues.size()) {
            return typed_.lvalues[statement.a].name_id;
        }
        if (statement.kind == kStatementExpression &&
            statement.a < typed_.expressions.size()) {
            return typed_.expressions[statement.a].name_id;
        }
        return NoIndex;
    }

    GPU::IR::ValueId MaterializeLineValueIfSelected(
        uint32_t statement_id,
        GPU::IR::ValueId value,
        uint32_t type_id) {
        if (inputs_.diagnostic_mode != 2 ||
            inputs_.diagnostic_source_site != statement_id ||
            diagnostic_site_suppression_depth_ != 0) {
            return value;
        }
        if (type_id >= typed_.types.size()) {
            return GPU::IR::InvalidValueId;
        }
        const auto type = ToModuleType(type_id);
        const auto name = UniqueGlslName("__feather_line_value_once");
        if (!type.IsValid() || name.empty()) {
            return GPU::IR::InvalidValueId;
        }
        const auto local = builder_.LocalVariable(type, name);
        if (local == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }
        EmitLocalDeclaration(type, name, value);
        return local;
    }

    bool EmitLineValueRecord(
        uint32_t statement_id,
        GPU::IR::ValueId value,
        uint32_t type_id) {
        if (inputs_.diagnostic_mode != 2 ||
            inputs_.diagnostic_source_site != statement_id ||
            diagnostic_site_suppression_depth_ != 0) {
            return true;
        }
        if (diagnostic_sites_resource_ == GPU::IR::InvalidResourceId ||
            type_id >= typed_.types.size()) {
            return Fail("line-value record target or type is unavailable");
        }

        const auto type = ToModuleType(type_id);
        GPU::IR::Type scalar_type;
        uint32_t type_tag = 0;
        uint32_t component_count = 0;
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool:
            scalar_type = GPU::IR::Type::Bool(); type_tag = 1; component_count = 1; break;
        case GPU::IR::Type::Kind::Int:
            scalar_type = GPU::IR::Type::Int(); type_tag = 2; component_count = 1; break;
        case GPU::IR::Type::Kind::UInt:
            scalar_type = GPU::IR::Type::UInt(); type_tag = 3; component_count = 1; break;
        case GPU::IR::Type::Kind::Float:
            scalar_type = GPU::IR::Type::Float(); type_tag = 4; component_count = 1; break;
        case GPU::IR::Type::Kind::Bool2:
        case GPU::IR::Type::Kind::Bool3:
        case GPU::IR::Type::Kind::Bool4:
            scalar_type = GPU::IR::Type::Bool(); type_tag = 1;
            component_count = type.kind == GPU::IR::Type::Kind::Bool2 ? 2u :
                              type.kind == GPU::IR::Type::Kind::Bool3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::Int2:
        case GPU::IR::Type::Kind::Int3:
        case GPU::IR::Type::Kind::Int4:
            scalar_type = GPU::IR::Type::Int(); type_tag = 2;
            component_count = type.kind == GPU::IR::Type::Kind::Int2 ? 2u :
                              type.kind == GPU::IR::Type::Kind::Int3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::UInt2:
        case GPU::IR::Type::Kind::UInt3:
        case GPU::IR::Type::Kind::UInt4:
            scalar_type = GPU::IR::Type::UInt(); type_tag = 3;
            component_count = type.kind == GPU::IR::Type::Kind::UInt2 ? 2u :
                              type.kind == GPU::IR::Type::Kind::UInt3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::Float2:
        case GPU::IR::Type::Kind::Float3:
        case GPU::IR::Type::Kind::Float4:
            scalar_type = GPU::IR::Type::Float(); type_tag = 4;
            component_count = type.kind == GPU::IR::Type::Kind::Float2 ? 2u :
                              type.kind == GPU::IR::Type::Kind::Float3 ? 3u : 4u;
            break;
        default:
            return Fail("line-value selected type is not a 32-bit scalar or vector");
        }

        const auto uint_type = GPU::IR::Type::UInt();
        const auto one = builder_.Literal(uint_type, "1u");
        const auto zero = builder_.Literal(uint_type, "0u");
        const auto thread_x = builder_.ThreadIndexX();
        const auto thread_y = builder_.ThreadIndexY();
        const auto thread_z = builder_.ThreadIndexZ();
        const std::array x_args{thread_x};
        const std::array y_args{thread_y};
        const std::array z_args{thread_z};
        const auto x = builder_.Intrinsic("uint", uint_type, x_args);
        const auto y = builder_.Intrinsic("uint", uint_type, y_args);
        const auto z = builder_.Intrinsic("uint", uint_type, z_args);
        const auto selected_x = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_x) + "u");
        const auto selected_y = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_y) + "u");
        const auto selected_z = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_z) + "u");
        auto selected = builder_.Compare(GPU::IR::CompareOp::Equal, x, selected_x);
        const auto y_selected = builder_.Compare(GPU::IR::CompareOp::Equal, y, selected_y);
        const auto z_selected = builder_.Compare(GPU::IR::CompareOp::Equal, z, selected_z);
        selected = builder_.Binary(GPU::IR::BinaryOp::LogicalAnd, selected, y_selected);
        selected = builder_.Binary(GPU::IR::BinaryOp::LogicalAnd, selected, z_selected);
        if (one == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId ||
            x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId ||
            z == GPU::IR::InvalidValueId || selected == GPU::IR::InvalidValueId) {
            return Fail("line-value invocation filter could not be lowered");
        }

        auto word = [&](uint32_t index) {
            const auto literal = builder_.Literal(uint_type, std::to_string(index) + "u");
            return builder_.ResourceElement(diagnostic_sites_resource_, literal);
        };
        std::vector<GPU::IR::Statement> writes;
        const auto previous_capture = capture_;
        capture_ = &writes;
        auto store_constant = [&](uint32_t index, uint32_t constant) {
            EmitStore(
                word(index),
                builder_.Literal(uint_type, std::to_string(constant) + "u"));
        };
        store_constant(0, 1u);
        store_constant(1, 1u);
        const auto occurrence_target = word(2);
        const std::array occurrence_arguments{one};
        const auto occurrence_increment = builder_.Atomic(
            GPU::IR::AtomicOp::Add,
            uint_type,
            occurrence_target,
            occurrence_arguments);
        EmitExpression(occurrence_increment);
        store_constant(3, statement_id);
        EmitStore(word(4), x);
        EmitStore(word(5), y);
        EmitStore(word(6), z);
        store_constant(7, type_tag);
        store_constant(8, component_count);

        constexpr std::string_view components = "xyzw";
        for (uint32_t component_index = 0; component_index < component_count; ++component_index) {
            auto component = value;
            if (component_count > 1) {
                component = builder_.Swizzle(
                    value,
                    scalar_type,
                    std::string(1, components[component_index]));
            }
            GPU::IR::ValueId raw = GPU::IR::InvalidValueId;
            if (type_tag == 1) {
                raw = builder_.Ternary(component, one, zero);
            } else if (type_tag == 2) {
                const std::array args{component};
                raw = builder_.Intrinsic("uint", uint_type, args);
            } else if (type_tag == 3) {
                raw = component;
            } else {
                const std::array args{component};
                raw = builder_.Intrinsic("floatBitsToUint", uint_type, args);
            }
            if (raw == GPU::IR::InvalidValueId) {
                capture_ = previous_capture;
                return Fail("line-value payload encoding could not be lowered");
            }
            EmitStore(word(9u + component_index), raw);
        }
        for (uint32_t index = 9u + component_count; index < 16u; ++index) {
            store_constant(index, 0u);
        }
        capture_ = previous_capture;

        if (occurrence_increment == GPU::IR::InvalidValueId) {
            return Fail("line-value occurrence counter could not be lowered");
        }
        EmitIf(selected, AddBlock(std::move(writes)), GPU::IR::InvalidBlockId);
        return true;
    }

    bool UbsanEnabled(uint32_t flag) const {
        return inputs_.diagnostic_mode == kDiagnosticUbsan &&
               (inputs_.diagnostic_flags & flag) != 0u;
    }

    std::optional<std::vector<GPU::IR::Statement>> CaptureDiagnosticStatements(
        const std::function<bool()>& emit) {
        const auto previous_capture = capture_;
        std::vector<GPU::IR::Statement> statements;
        capture_ = &statements;
        const auto succeeded = emit();
        capture_ = previous_capture;
        if (!succeeded) {
            return std::nullopt;
        }
        return statements;
    }

    GPU::IR::ValueId MaterializeOnce(
        GPU::IR::ValueId value,
        GPU::IR::Type type,
        std::string_view prefix) {
        if (value == GPU::IR::InvalidValueId || !type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }
        const auto name = UniqueGlslName(std::string(prefix));
        const auto local = builder_.LocalVariable(type, name);
        if (name.empty() || local == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }
        EmitLocalDeclaration(type, name, value);
        return local;
    }

    GPU::IR::ValueId ComputeTraceWord(uint32_t index) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(index) + "u");
        return literal == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, literal);
    }

    GPU::IR::ValueId ComputeTraceWord(GPU::IR::ValueId base, uint32_t offset) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(offset) + "u");
        const auto index = builder_.Binary(GPU::IR::BinaryOp::Add, base, literal);
        return literal == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, index);
    }

    GPU::IR::ValueId ComputeTraceSelectedPredicate() {
        const auto uint_type = GPU::IR::Type::UInt();
        const auto to_uint = [&](GPU::IR::ValueId value) {
            const std::array arguments{value};
            return value == GPU::IR::InvalidValueId
                       ? GPU::IR::InvalidValueId
                       : builder_.Intrinsic("uint", uint_type, arguments);
        };
        const auto x = to_uint(builder_.ThreadIndexX());
        const auto y = to_uint(builder_.ThreadIndexY());
        const auto z = to_uint(builder_.ThreadIndexZ());
        const auto selected_x = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_x) + "u");
        const auto selected_y = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_y) + "u");
        const auto selected_z = builder_.Literal(
            uint_type, std::to_string(inputs_.diagnostic_selected_z) + "u");
        auto selected = builder_.Compare(GPU::IR::CompareOp::Equal, x, selected_x);
        const auto y_selected = builder_.Compare(GPU::IR::CompareOp::Equal, y, selected_y);
        const auto z_selected = builder_.Compare(GPU::IR::CompareOp::Equal, z, selected_z);
        selected = builder_.Binary(GPU::IR::BinaryOp::LogicalAnd, selected, y_selected);
        selected = builder_.Binary(GPU::IR::BinaryOp::LogicalAnd, selected, z_selected);
        return x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId ||
                       z == GPU::IR::InvalidValueId || selected == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : selected;
    }

    bool BuildComputeTracePayload(
        GPU::IR::ValueId value,
        uint32_t type_id,
        uint32_t* type_tag,
        uint32_t* component_count,
        std::array<GPU::IR::ValueId, 4>* raw_components) {
        if (type_tag == nullptr || component_count == nullptr || raw_components == nullptr) {
            return false;
        }
        *type_tag = 0u;
        *component_count = 0u;
        const auto uint_type = GPU::IR::Type::UInt();
        const auto zero = builder_.Literal(uint_type, "0u");
        raw_components->fill(zero);
        if (zero == GPU::IR::InvalidValueId) {
            return false;
        }
        if (value == GPU::IR::InvalidValueId || type_id == NoIndex || type_id >= typed_.types.size()) {
            return true;
        }

        const auto type = ToModuleType(type_id);
        GPU::IR::Type scalar_type;
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool:
            scalar_type = GPU::IR::Type::Bool(); *type_tag = 1u; *component_count = 1u; break;
        case GPU::IR::Type::Kind::Int:
            scalar_type = GPU::IR::Type::Int(); *type_tag = 2u; *component_count = 1u; break;
        case GPU::IR::Type::Kind::UInt:
            scalar_type = GPU::IR::Type::UInt(); *type_tag = 3u; *component_count = 1u; break;
        case GPU::IR::Type::Kind::Float:
            scalar_type = GPU::IR::Type::Float(); *type_tag = 4u; *component_count = 1u; break;
        case GPU::IR::Type::Kind::Bool2:
        case GPU::IR::Type::Kind::Bool3:
        case GPU::IR::Type::Kind::Bool4:
            scalar_type = GPU::IR::Type::Bool(); *type_tag = 1u;
            *component_count = type.kind == GPU::IR::Type::Kind::Bool2 ? 2u :
                               type.kind == GPU::IR::Type::Kind::Bool3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::Int2:
        case GPU::IR::Type::Kind::Int3:
        case GPU::IR::Type::Kind::Int4:
            scalar_type = GPU::IR::Type::Int(); *type_tag = 2u;
            *component_count = type.kind == GPU::IR::Type::Kind::Int2 ? 2u :
                               type.kind == GPU::IR::Type::Kind::Int3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::UInt2:
        case GPU::IR::Type::Kind::UInt3:
        case GPU::IR::Type::Kind::UInt4:
            scalar_type = GPU::IR::Type::UInt(); *type_tag = 3u;
            *component_count = type.kind == GPU::IR::Type::Kind::UInt2 ? 2u :
                               type.kind == GPU::IR::Type::Kind::UInt3 ? 3u : 4u;
            break;
        case GPU::IR::Type::Kind::Float2:
        case GPU::IR::Type::Kind::Float3:
        case GPU::IR::Type::Kind::Float4:
            scalar_type = GPU::IR::Type::Float(); *type_tag = 4u;
            *component_count = type.kind == GPU::IR::Type::Kind::Float2 ? 2u :
                               type.kind == GPU::IR::Type::Kind::Float3 ? 3u : 4u;
            break;
        default:
            return true;
        }

        const auto one = builder_.Literal(uint_type, "1u");
        constexpr std::string_view components = "xyzw";
        for (uint32_t index = 0; index < *component_count; ++index) {
            auto component = value;
            if (*component_count > 1u) {
                component = builder_.Swizzle(
                    value, scalar_type, std::string(1, components[index]));
            }
            GPU::IR::ValueId raw = GPU::IR::InvalidValueId;
            if (*type_tag == 1u) {
                raw = builder_.Ternary(component, one, zero);
            } else if (*type_tag == 2u) {
                const std::array arguments{component};
                raw = builder_.Intrinsic("uint", uint_type, arguments);
            } else if (*type_tag == 3u) {
                raw = component;
            } else {
                const std::array arguments{component};
                raw = builder_.Intrinsic("floatBitsToUint", uint_type, arguments);
            }
            if (raw == GPU::IR::InvalidValueId) {
                return false;
            }
            (*raw_components)[index] = raw;
        }
        return one != GPU::IR::InvalidValueId;
    }

    bool EmitComputeTraceEvent(
        uint32_t source_site,
        uint32_t event_kind,
        uint32_t function_id,
        uint32_t symbol_id,
        GPU::IR::ValueId value = GPU::IR::InvalidValueId,
        uint32_t type_id = NoIndex) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        if (diagnostic_sites_resource_ == GPU::IR::InvalidResourceId ||
            source_site >= inputs_.diagnostic_site_count ||
            function_id >= typed_.functions.size()) {
            return Fail("compute-trace event identity is outside the configured FEIR ABI");
        }

        uint32_t type_tag = 0u;
        uint32_t component_count = 0u;
        std::array<GPU::IR::ValueId, 4> raw_components{};
        if (!BuildComputeTracePayload(
                value, type_id, &type_tag, &component_count, &raw_components)) {
            return Fail("compute-trace value payload could not be encoded");
        }
        const auto selected = ComputeTraceSelectedPredicate();
        if (selected == GPU::IR::InvalidValueId) {
            return Fail("compute-trace invocation filter could not be lowered");
        }

        const auto selected_statements = CaptureDiagnosticStatements([&] {
            const auto uint_type = GPU::IR::Type::UInt();
            const auto one = builder_.Literal(uint_type, "1u");
            const auto sixteen = builder_.Literal(uint_type, "16u");
            const auto header_words = builder_.Literal(uint_type, "16u");
            const std::array increment_arguments{one};
            const auto attempted = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                ComputeTraceWord(3u),
                increment_arguments);
            const auto slot = MaterializeOnce(
                attempted, uint_type, "__feather_compute_trace_slot");
            const auto capacity = ComputeTraceWord(2u);
            const auto has_capacity = builder_.Compare(
                GPU::IR::CompareOp::Less, slot, capacity);
            const auto record_offset = builder_.Binary(
                GPU::IR::BinaryOp::Mul, slot, sixteen);
            const auto record_base = builder_.Binary(
                GPU::IR::BinaryOp::Add, header_words, record_offset);
            if (one == GPU::IR::InvalidValueId || sixteen == GPU::IR::InvalidValueId ||
                header_words == GPU::IR::InvalidValueId || attempted == GPU::IR::InvalidValueId ||
                slot == GPU::IR::InvalidValueId || capacity == GPU::IR::InvalidValueId ||
                has_capacity == GPU::IR::InvalidValueId || record_offset == GPU::IR::InvalidValueId ||
                record_base == GPU::IR::InvalidValueId) {
                return false;
            }

            const auto committed = CaptureDiagnosticStatements([&] {
                const auto to_uint = [&](GPU::IR::ValueId input) {
                    const std::array arguments{input};
                    return builder_.Intrinsic("uint", uint_type, arguments);
                };
                const auto thread_x = to_uint(builder_.ThreadIndexX());
                const auto thread_y = to_uint(builder_.ThreadIndexY());
                const auto thread_z = to_uint(builder_.ThreadIndexZ());
                const auto site = builder_.Literal(uint_type, std::to_string(source_site) + "u");
                const auto kind = builder_.Literal(uint_type, std::to_string(event_kind) + "u");
                const auto function = builder_.Literal(uint_type, std::to_string(function_id) + "u");
                const auto symbol = builder_.Literal(uint_type, std::to_string(symbol_id) + "u");
                const auto value_type = builder_.Literal(uint_type, std::to_string(type_tag) + "u");
                const auto components = builder_.Literal(uint_type, std::to_string(component_count) + "u");
                const auto zero = builder_.Literal(uint_type, "0u");
                const auto depth = ComputeTraceWord(12u);
                if (thread_x == GPU::IR::InvalidValueId || thread_y == GPU::IR::InvalidValueId ||
                    thread_z == GPU::IR::InvalidValueId || site == GPU::IR::InvalidValueId ||
                    kind == GPU::IR::InvalidValueId || function == GPU::IR::InvalidValueId ||
                    symbol == GPU::IR::InvalidValueId || value_type == GPU::IR::InvalidValueId ||
                    components == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId ||
                    depth == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitStore(ComputeTraceWord(record_base, 0u), slot);
                EmitStore(ComputeTraceWord(record_base, 1u), site);
                EmitStore(ComputeTraceWord(record_base, 2u), kind);
                EmitStore(ComputeTraceWord(record_base, 3u), depth);
                EmitStore(ComputeTraceWord(record_base, 4u), function);
                EmitStore(ComputeTraceWord(record_base, 5u), symbol);
                EmitStore(ComputeTraceWord(record_base, 6u), value_type);
                EmitStore(ComputeTraceWord(record_base, 7u), components);
                for (uint32_t index = 0; index < raw_components.size(); ++index) {
                    EmitStore(ComputeTraceWord(record_base, 8u + index), raw_components[index]);
                }
                EmitStore(ComputeTraceWord(record_base, 12u), thread_x);
                EmitStore(ComputeTraceWord(record_base, 13u), thread_y);
                EmitStore(ComputeTraceWord(record_base, 14u), thread_z);
                EmitStore(ComputeTraceWord(record_base, 15u), zero);
                const auto committed_increment = builder_.Atomic(
                    GPU::IR::AtomicOp::Add,
                    uint_type,
                    ComputeTraceWord(4u),
                    increment_arguments);
                if (committed_increment == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitExpression(committed_increment);
                return true;
            });
            const auto dropped = CaptureDiagnosticStatements([&] {
                const auto dropped_increment = builder_.Atomic(
                    GPU::IR::AtomicOp::Add,
                    uint_type,
                    ComputeTraceWord(5u),
                    increment_arguments);
                if (dropped_increment == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitExpression(dropped_increment);
                return true;
            });
            if (!committed.has_value() || !dropped.has_value()) {
                return false;
            }
            EmitIf(
                has_capacity,
                AddBlock(std::move(*committed)),
                AddBlock(std::move(*dropped)));
            return true;
        });
        if (!selected_statements.has_value()) {
            return Fail("compute-trace bounded event write could not be lowered");
        }
        EmitIf(selected, AddBlock(std::move(*selected_statements)), GPU::IR::InvalidBlockId);
        return true;
    }

    bool EmitComputeTraceSelectedStore(uint32_t word_index, uint32_t constant) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        const auto selected = ComputeTraceSelectedPredicate();
        const auto statements = CaptureDiagnosticStatements([&] {
            const auto value = builder_.Literal(
                GPU::IR::Type::UInt(), std::to_string(constant) + "u");
            if (value == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitStore(ComputeTraceWord(word_index), value);
            return true;
        });
        if (selected == GPU::IR::InvalidValueId || !statements.has_value()) {
            return Fail("compute-trace selected header store could not be lowered");
        }
        EmitIf(selected, AddBlock(std::move(*statements)), GPU::IR::InvalidBlockId);
        return true;
    }

    bool EmitComputeTraceDepthDelta(bool increment) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        const auto selected = ComputeTraceSelectedPredicate();
        const auto statements = CaptureDiagnosticStatements([&] {
            const auto delta = builder_.Literal(
                GPU::IR::Type::UInt(), increment ? "1u" : "4294967295u");
            const std::array arguments{delta};
            const auto changed = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                GPU::IR::Type::UInt(),
                ComputeTraceWord(12u),
                arguments);
            if (delta == GPU::IR::InvalidValueId || changed == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitExpression(changed);
            return true;
        });
        if (selected == GPU::IR::InvalidValueId || !statements.has_value()) {
            return Fail("compute-trace call-depth transition could not be lowered");
        }
        EmitIf(selected, AddBlock(std::move(*statements)), GPU::IR::InvalidBlockId);
        return true;
    }

    bool EmitComputeTraceEntryStart(uint32_t function_id, uint32_t source_site) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        return EmitComputeTraceSelectedStore(6u, 1u) &&
               EmitComputeTraceSelectedStore(7u, 0u) &&
               EmitComputeTraceSelectedStore(12u, 0u) &&
               EmitComputeTraceEvent(
                   source_site, kTraceEventFunctionEnter, function_id, NoIndex);
    }

    bool EmitComputeTraceEntryEnd(uint32_t function_id, uint32_t source_site) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        return EmitComputeTraceEvent(
                   source_site, kTraceEventFunctionExit, function_id, NoIndex) &&
               EmitComputeTraceEvent(
                   source_site, kTraceEventInvocationEnd, function_id, NoIndex) &&
               EmitComputeTraceSelectedStore(7u, 1u);
    }

    bool EmitComputeTraceCallableStart(uint32_t function_id, uint32_t source_site) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        return EmitComputeTraceDepthDelta(true) &&
               EmitComputeTraceEvent(
                   source_site, kTraceEventFunctionEnter, function_id, NoIndex);
    }

    bool EmitComputeTraceCallableEnd(uint32_t function_id, uint32_t source_site) {
        if (inputs_.diagnostic_mode != kDiagnosticComputeTrace) {
            return true;
        }
        return EmitComputeTraceEvent(
                   source_site, kTraceEventFunctionExit, function_id, NoIndex) &&
               EmitComputeTraceDepthDelta(false);
    }

    GPU::IR::ValueId UbsanWord(uint32_t index) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(index) + "u");
        return literal == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, literal);
    }

    GPU::IR::ValueId UbsanWord(GPU::IR::ValueId base, uint32_t offset) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(offset) + "u");
        const auto index = builder_.Binary(GPU::IR::BinaryOp::Add, base, literal);
        return literal == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, index);
    }

    GPU::IR::ValueId UbsanUInt(GPU::IR::ValueId value) {
        const std::array arguments{value};
        return value == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.Intrinsic("uint", GPU::IR::Type::UInt(), arguments);
    }

    GPU::IR::ValueId BranchDivergenceWord(uint32_t index) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(index) + "u");
        return literal == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, literal);
    }

    GPU::IR::ValueId BranchDivergenceWord(GPU::IR::ValueId base, uint32_t offset) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(offset) + "u");
        const auto index = builder_.Binary(GPU::IR::BinaryOp::Add, base, literal);
        return literal == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, index);
    }

    bool EmitBranchDivergenceCapture(GPU::IR::ValueId predicate) {
        if (inputs_.diagnostic_mode != kDiagnosticBranchDivergence ||
            current_statement_id_ != inputs_.diagnostic_source_site) {
            return true;
        }
        if (branch_divergence_site_emitted_ ||
            diagnostic_sites_resource_ == GPU::IR::InvalidResourceId) {
            return Fail("branch-divergence source site was emitted more than once or has no stream");
        }

        const auto uint_type = GPU::IR::Type::UInt();
        const auto bool_type = GPU::IR::Type::Bool();
        const auto uint4_type = GPU::IR::Type::UInt4();
        const auto one = builder_.Literal(uint_type, "1u");
        const auto zero = builder_.Literal(uint_type, "0u");
        const auto always = builder_.Literal(bool_type, "true");
        const auto not_predicate = builder_.Unary(GPU::IR::UnaryOp::LogicalNot, predicate);
        auto active_mask = MaterializeOnce(
            builder_.SubgroupBallot(always), uint4_type, "feather_branch_active_mask");
        auto true_mask = MaterializeOnce(
            builder_.SubgroupBallot(predicate), uint4_type, "feather_branch_true_mask");
        auto active_count = MaterializeOnce(
            builder_.SubgroupBallotBitCount(active_mask), uint_type, "feather_branch_active_count");
        auto true_count = MaterializeOnce(
            builder_.SubgroupBallotBitCount(true_mask), uint_type, "feather_branch_true_count");
        auto false_count = MaterializeOnce(
            builder_.Binary(GPU::IR::BinaryOp::Sub, active_count, true_count),
            uint_type,
            "feather_branch_false_count");
        auto any_true = MaterializeOnce(
            builder_.SubgroupAny(predicate), bool_type, "feather_branch_any_true");
        auto any_false = MaterializeOnce(
            builder_.SubgroupAny(not_predicate), bool_type, "feather_branch_any_false");
        auto all_true = MaterializeOnce(
            builder_.SubgroupAll(predicate), bool_type, "feather_branch_all_true");
        auto all_false = MaterializeOnce(
            builder_.SubgroupAll(not_predicate), bool_type, "feather_branch_all_false");
        auto mixed = MaterializeOnce(
            builder_.Binary(GPU::IR::BinaryOp::LogicalAnd, any_true, any_false),
            bool_type,
            "feather_branch_mixed");
        auto elected = MaterializeOnce(
            builder_.SubgroupElect(), bool_type, "feather_branch_elected");
        auto subgroup_id = MaterializeOnce(
            builder_.SubgroupId(), uint_type, "feather_branch_subgroup_id");
        auto subgroup_size = MaterializeOnce(
            builder_.SubgroupSize(), uint_type, "feather_branch_subgroup_size");
        auto num_subgroups = MaterializeOnce(
            builder_.NumSubgroups(), uint_type, "feather_branch_num_subgroups");
        auto elected_lane = MaterializeOnce(
            builder_.SubgroupInvocationId(), uint_type, "feather_branch_elected_lane");
        const auto group_x = UbsanUInt(builder_.GroupIdX());
        const auto group_y = UbsanUInt(builder_.GroupIdY());
        const auto group_z = UbsanUInt(builder_.GroupIdZ());
        const auto mixed_u32 = builder_.Ternary(mixed, one, zero);
        const auto uniform_true_u32 = builder_.Ternary(all_true, one, zero);
        const auto uniform_false_u32 = builder_.Ternary(all_false, one, zero);
        if (one == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId ||
            always == GPU::IR::InvalidValueId || not_predicate == GPU::IR::InvalidValueId ||
            active_mask == GPU::IR::InvalidValueId || true_mask == GPU::IR::InvalidValueId ||
            active_count == GPU::IR::InvalidValueId || true_count == GPU::IR::InvalidValueId ||
            false_count == GPU::IR::InvalidValueId || any_true == GPU::IR::InvalidValueId ||
            any_false == GPU::IR::InvalidValueId || all_true == GPU::IR::InvalidValueId ||
            all_false == GPU::IR::InvalidValueId || mixed == GPU::IR::InvalidValueId ||
            elected == GPU::IR::InvalidValueId || subgroup_id == GPU::IR::InvalidValueId ||
            subgroup_size == GPU::IR::InvalidValueId || num_subgroups == GPU::IR::InvalidValueId ||
            elected_lane == GPU::IR::InvalidValueId || group_x == GPU::IR::InvalidValueId ||
            group_y == GPU::IR::InvalidValueId || group_z == GPU::IR::InvalidValueId ||
            mixed_u32 == GPU::IR::InvalidValueId || uniform_true_u32 == GPU::IR::InvalidValueId ||
            uniform_false_u32 == GPU::IR::InvalidValueId) {
            return Fail("branch-divergence subgroup predicate facts could not be lowered");
        }

        const auto elected_statements = CaptureDiagnosticStatements([&] {
            const std::array one_argument{one};
            const auto attempted = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                BranchDivergenceWord(3u),
                one_argument);
            const auto slot = MaterializeOnce(
                attempted, uint_type, "feather_branch_record_slot");
            const auto capacity = BranchDivergenceWord(2u);
            const auto has_capacity = builder_.Compare(
                GPU::IR::CompareOp::Less, slot, capacity);
            const auto twenty = builder_.Literal(uint_type, "20u");
            const auto sixteen = builder_.Literal(uint_type, "16u");
            const auto record_offset = builder_.Binary(
                GPU::IR::BinaryOp::Mul, slot, twenty);
            const auto record_base = builder_.Binary(
                GPU::IR::BinaryOp::Add, sixteen, record_offset);
            if (attempted == GPU::IR::InvalidValueId || slot == GPU::IR::InvalidValueId ||
                capacity == GPU::IR::InvalidValueId || has_capacity == GPU::IR::InvalidValueId ||
                twenty == GPU::IR::InvalidValueId || sixteen == GPU::IR::InvalidValueId ||
                record_offset == GPU::IR::InvalidValueId || record_base == GPU::IR::InvalidValueId) {
                return false;
            }

            const std::array<std::pair<uint32_t, GPU::IR::ValueId>, 7> aggregates{{
                {6u, one},
                {7u, mixed_u32},
                {8u, active_count},
                {9u, true_count},
                {10u, false_count},
                {11u, uniform_true_u32},
                {12u, uniform_false_u32}
            }};
            for (const auto& [word, amount] : aggregates) {
                const std::array arguments{amount};
                const auto increment = builder_.Atomic(
                    GPU::IR::AtomicOp::Add,
                    uint_type,
                    BranchDivergenceWord(word),
                    arguments);
                if (increment == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitExpression(increment);
            }

            const auto committed = CaptureDiagnosticStatements([&] {
                const std::array<GPU::IR::ValueId, 12> fixed_values{
                    builder_.Literal(uint_type, std::to_string(current_statement_id_) + "u"),
                    group_x,
                    group_y,
                    group_z,
                    subgroup_id,
                    subgroup_size,
                    num_subgroups,
                    elected_lane,
                    active_count,
                    true_count,
                    false_count,
                    mixed_u32
                };
                for (uint32_t offset = 0; offset < fixed_values.size(); ++offset) {
                    if (fixed_values[offset] == GPU::IR::InvalidValueId) {
                        return false;
                    }
                    EmitStore(BranchDivergenceWord(record_base, offset), fixed_values[offset]);
                }
                constexpr std::string_view components = "xyzw";
                for (uint32_t component = 0; component < 4u; ++component) {
                    const auto active_component = builder_.Swizzle(
                        active_mask, uint_type, std::string(1, components[component]));
                    const auto true_component = builder_.Swizzle(
                        true_mask, uint_type, std::string(1, components[component]));
                    if (active_component == GPU::IR::InvalidValueId ||
                        true_component == GPU::IR::InvalidValueId) {
                        return false;
                    }
                    EmitStore(BranchDivergenceWord(record_base, 12u + component), active_component);
                    EmitStore(BranchDivergenceWord(record_base, 16u + component), true_component);
                }
                const auto increment = builder_.Atomic(
                    GPU::IR::AtomicOp::Add,
                    uint_type,
                    BranchDivergenceWord(4u),
                    one_argument);
                if (increment == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitExpression(increment);
                return true;
            });
            const auto dropped = CaptureDiagnosticStatements([&] {
                const auto increment = builder_.Atomic(
                    GPU::IR::AtomicOp::Add,
                    uint_type,
                    BranchDivergenceWord(5u),
                    one_argument);
                if (increment == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitExpression(increment);
                return true;
            });
            if (!committed.has_value() || !dropped.has_value()) {
                return false;
            }
            EmitIf(
                has_capacity,
                AddBlock(std::move(*committed)),
                AddBlock(std::move(*dropped)));
            return true;
        });
        if (!elected_statements.has_value()) {
            return Fail("branch-divergence record and aggregate writes could not be lowered");
        }
        EmitIf(elected, AddBlock(std::move(*elected_statements)), GPU::IR::InvalidBlockId);
        branch_divergence_site_emitted_ = true;
        return true;
    }

    GPU::IR::ValueId UbsanFloatBits(GPU::IR::ValueId value) {
        const std::array arguments{value};
        return value == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.Intrinsic("floatBitsToUint", GPU::IR::Type::UInt(), arguments);
    }

    bool EmitUbsanIssue(
        uint32_t code,
        uint32_t source_site,
        GPU::IR::ValueId detail0,
        GPU::IR::ValueId detail1,
        GPU::IR::ValueId detail2) {
        if (inputs_.diagnostic_mode != kDiagnosticUbsan ||
            diagnostic_sites_resource_ == GPU::IR::InvalidResourceId ||
            source_site >= inputs_.diagnostic_site_count ||
            detail0 == GPU::IR::InvalidValueId ||
            detail1 == GPU::IR::InvalidValueId ||
            detail2 == GPU::IR::InvalidValueId) {
            return Fail("UBSan issue identity or payload is unavailable");
        }

        const auto uint_type = GPU::IR::Type::UInt();
        const auto one = builder_.Literal(uint_type, "1u");
        const auto eight = builder_.Literal(uint_type, "8u");
        const auto four = builder_.Literal(uint_type, "4u");
        const auto attempted_target = UbsanWord(0u);
        const std::array increment_arguments{one};
        const auto attempted = builder_.Atomic(
            GPU::IR::AtomicOp::Add,
            uint_type,
            attempted_target,
            increment_arguments);
        const auto slot = MaterializeOnce(attempted, uint_type, "__feather_ubsan_slot");
        const auto capacity = UbsanWord(3u);
        const auto has_capacity = builder_.Compare(GPU::IR::CompareOp::Less, slot, capacity);
        const auto record_offset = builder_.Binary(GPU::IR::BinaryOp::Mul, slot, eight);
        const auto record_base = builder_.Binary(GPU::IR::BinaryOp::Add, four, record_offset);
        if (one == GPU::IR::InvalidValueId || eight == GPU::IR::InvalidValueId ||
            four == GPU::IR::InvalidValueId || attempted == GPU::IR::InvalidValueId ||
            slot == GPU::IR::InvalidValueId || capacity == GPU::IR::InvalidValueId ||
            has_capacity == GPU::IR::InvalidValueId || record_offset == GPU::IR::InvalidValueId ||
            record_base == GPU::IR::InvalidValueId) {
            return Fail("UBSan bounded record reservation could not be lowered");
        }

        const auto committed = CaptureDiagnosticStatements([&] {
            const auto thread_x = UbsanUInt(builder_.ThreadIndexX());
            const auto thread_y = UbsanUInt(builder_.ThreadIndexY());
            const auto thread_z = UbsanUInt(builder_.ThreadIndexZ());
            const auto code_value = builder_.Literal(uint_type, std::to_string(code) + "u");
            const auto site_value = builder_.Literal(uint_type, std::to_string(source_site) + "u");
            if (thread_x == GPU::IR::InvalidValueId || thread_y == GPU::IR::InvalidValueId ||
                thread_z == GPU::IR::InvalidValueId || code_value == GPU::IR::InvalidValueId ||
                site_value == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitStore(UbsanWord(record_base, 0u), code_value);
            EmitStore(UbsanWord(record_base, 1u), site_value);
            EmitStore(UbsanWord(record_base, 2u), thread_x);
            EmitStore(UbsanWord(record_base, 3u), thread_y);
            EmitStore(UbsanWord(record_base, 4u), thread_z);
            EmitStore(UbsanWord(record_base, 5u), detail0);
            EmitStore(UbsanWord(record_base, 6u), detail1);
            EmitStore(UbsanWord(record_base, 7u), detail2);
            const auto committed_increment = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                UbsanWord(1u),
                increment_arguments);
            if (committed_increment == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitExpression(committed_increment);
            return true;
        });
        const auto dropped = CaptureDiagnosticStatements([&] {
            const auto dropped_increment = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                UbsanWord(2u),
                increment_arguments);
            if (dropped_increment == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitExpression(dropped_increment);
            return true;
        });
        if (!committed.has_value() || !dropped.has_value()) {
            return Fail("UBSan record commit or overflow accounting could not be lowered");
        }
        EmitIf(
            has_capacity,
            AddBlock(std::move(*committed)),
            AddBlock(std::move(*dropped)));
        return true;
    }

    GPU::IR::ValueId BuildUbsanSafeDivision(
        GPU::IR::ValueId left,
        GPU::IR::ValueId right,
        uint32_t source_site) {
        const auto float_type = GPU::IR::Type::Float();
        left = MaterializeOnce(left, float_type, "__feather_ubsan_div_left");
        right = MaterializeOnce(right, float_type, "__feather_ubsan_div_right");
        const auto zero = builder_.Literal(float_type, "0.0");
        const auto zero_uint = builder_.Literal(GPU::IR::Type::UInt(), "0u");
        const auto result_name = UniqueGlslName("__feather_ubsan_div_result");
        const auto result = builder_.LocalVariable(float_type, result_name);
        const auto denominator_is_zero = builder_.Compare(GPU::IR::CompareOp::Equal, right, zero);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId ||
            zero == GPU::IR::InvalidValueId || zero_uint == GPU::IR::InvalidValueId ||
            result_name.empty() || result == GPU::IR::InvalidValueId ||
            denominator_is_zero == GPU::IR::InvalidValueId) {
            return InvalidValue("UBSan division guard could not be lowered");
        }
        EmitLocalDeclaration(float_type, result_name, zero);

        const auto invalid = CaptureDiagnosticStatements([&] {
            return EmitUbsanIssue(
                kUbsanIssueDivideByZero,
                source_site,
                UbsanFloatBits(left),
                UbsanFloatBits(right),
                zero_uint);
        });
        const auto valid = CaptureDiagnosticStatements([&] {
            const auto raw = builder_.Binary(GPU::IR::BinaryOp::Div, left, right);
            if (raw == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitStore(result, raw);
            return true;
        });
        if (!invalid.has_value() || !valid.has_value()) {
            return InvalidValue("UBSan division branches could not be lowered");
        }
        EmitIf(
            denominator_is_zero,
            AddBlock(std::move(*invalid)),
            AddBlock(std::move(*valid)));
        return result;
    }

    GPU::IR::ValueId BuildUbsanSafeDomainIntrinsic(
        std::string_view intrinsic,
        uint32_t issue_code,
        GPU::IR::CompareOp comparison,
        GPU::IR::ValueId argument,
        uint32_t source_site) {
        const auto float_type = GPU::IR::Type::Float();
        argument = MaterializeOnce(argument, float_type, "__feather_ubsan_domain_input");
        const auto zero = builder_.Literal(float_type, "0.0");
        const auto zero_uint = builder_.Literal(GPU::IR::Type::UInt(), "0u");
        const auto invalid_domain = builder_.Compare(comparison, argument, zero);
        const auto result_name = UniqueGlslName("__feather_ubsan_domain_result");
        const auto result = builder_.LocalVariable(float_type, result_name);
        if (argument == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId ||
            zero_uint == GPU::IR::InvalidValueId || invalid_domain == GPU::IR::InvalidValueId ||
            result_name.empty() || result == GPU::IR::InvalidValueId) {
            return InvalidValue("UBSan domain guard could not be lowered");
        }
        EmitLocalDeclaration(float_type, result_name, zero);

        const auto invalid = CaptureDiagnosticStatements([&] {
            return EmitUbsanIssue(
                issue_code,
                source_site,
                UbsanFloatBits(argument),
                zero_uint,
                zero_uint);
        });
        const auto valid = CaptureDiagnosticStatements([&] {
            const std::array arguments{argument};
            const auto raw = builder_.Intrinsic(std::string(intrinsic), float_type, arguments);
            if (raw == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitStore(result, raw);
            return true;
        });
        if (!invalid.has_value() || !valid.has_value()) {
            return InvalidValue("UBSan domain branches could not be lowered");
        }
        EmitIf(
            invalid_domain,
            AddBlock(std::move(*invalid)),
            AddBlock(std::move(*valid)));
        return result;
    }

    GPU::IR::ValueId SanitizeUbsanFiniteValue(
        uint32_t source_site,
        GPU::IR::ValueId value,
        uint32_t type_id) {
        if (!UbsanEnabled(kUbsanCheckNonFinite) || type_id >= typed_.types.size() ||
            ToModuleType(type_id).kind != GPU::IR::Type::Kind::Float) {
            return value;
        }
        const auto float_type = GPU::IR::Type::Float();
        value = MaterializeOnce(value, float_type, "__feather_ubsan_finite_input");
        const auto result_name = UniqueGlslName("__feather_ubsan_finite_result");
        const auto result = builder_.LocalVariable(float_type, result_name);
        const auto zero = builder_.Literal(float_type, "0.0");
        const auto zero_uint = builder_.Literal(GPU::IR::Type::UInt(), "0u");
        const std::array arguments{value};
        const auto is_nan = builder_.Intrinsic("isnan", GPU::IR::Type::Bool(), arguments);
        const auto is_inf = builder_.Intrinsic("isinf", GPU::IR::Type::Bool(), arguments);
        if (value == GPU::IR::InvalidValueId || result_name.empty() ||
            result == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId ||
            zero_uint == GPU::IR::InvalidValueId || is_nan == GPU::IR::InvalidValueId ||
            is_inf == GPU::IR::InvalidValueId) {
            return InvalidValue("UBSan finite-value observation could not be lowered");
        }
        EmitLocalDeclaration(float_type, result_name, zero);

        const auto nan_branch = CaptureDiagnosticStatements([&] {
            return EmitUbsanIssue(
                kUbsanIssueNaN,
                source_site,
                UbsanFloatBits(value),
                zero_uint,
                zero_uint);
        });
        const auto inf_branch = CaptureDiagnosticStatements([&] {
            return EmitUbsanIssue(
                kUbsanIssueInfinity,
                source_site,
                UbsanFloatBits(value),
                zero_uint,
                zero_uint);
        });
        const auto finite_branch = CaptureDiagnosticStatements([&] {
            EmitStore(result, value);
            return true;
        });
        if (!nan_branch.has_value() || !inf_branch.has_value() || !finite_branch.has_value()) {
            return InvalidValue("UBSan finite-value branches could not be lowered");
        }
        const auto inf_or_finite = CaptureDiagnosticStatements([&] {
            EmitIf(
                is_inf,
                AddBlock(std::move(*inf_branch)),
                AddBlock(std::move(*finite_branch)));
            return true;
        });
        if (!inf_or_finite.has_value()) {
            return InvalidValue("UBSan finite-value fallback could not be lowered");
        }
        EmitIf(
            is_nan,
            AddBlock(std::move(*nan_branch)),
            AddBlock(std::move(*inf_or_finite)));
        return result;
    }

    GPU::IR::ValueId MaterializeDiagnosticValue(
        uint32_t statement_id,
        GPU::IR::ValueId value,
        uint32_t type_id) {
        value = SanitizeUbsanFiniteValue(statement_id, value, type_id);
        if (value == GPU::IR::InvalidValueId) {
            return value;
        }
        if (inputs_.diagnostic_mode == kDiagnosticComputeTrace) {
            if (type_id >= typed_.types.size()) {
                return GPU::IR::InvalidValueId;
            }
            const auto trace_type = ToModuleType(type_id);
            if (trace_type.kind != GPU::IR::Type::Kind::Void) {
                value = MaterializeOnce(
                    value,
                    trace_type,
                    "__feather_compute_trace_value");
                if (value == GPU::IR::InvalidValueId) {
                    return value;
                }
            }
        }
        return MaterializeLineValueIfSelected(statement_id, value, type_id);
    }

    GPU::IR::ValueId BuildUbsanCheckedResourceRead(
        const RegisteredResource& resource,
        GPU::IR::ValueId index,
        uint32_t index_type_id,
        uint32_t source_site) {
        if (!UbsanEnabled(kUbsanCheckBufferBounds) ||
            resource.kind != kResourceKindBuffer ||
            resource.element_count == GPU::IR::InvalidResourceId ||
            index_type_id >= typed_.types.size()) {
            return builder_.ResourceElement(resource.id, index);
        }

        const auto result_type = resource.element_type;
        const auto supported_result =
            result_type.kind == GPU::IR::Type::Kind::Bool ||
            result_type.kind == GPU::IR::Type::Kind::Int ||
            result_type.kind == GPU::IR::Type::Kind::UInt ||
            result_type.kind == GPU::IR::Type::Kind::Float;
        const auto index_type = ToModuleType(index_type_id);
        if (!supported_result ||
            (index_type.kind != GPU::IR::Type::Kind::Int &&
             index_type.kind != GPU::IR::Type::Kind::UInt)) {
            return builder_.ResourceElement(resource.id, index);
        }

        index = MaterializeOnce(index, index_type, "__feather_ubsan_buffer_index");
        const auto count = builder_.PushConstant(resource.element_count);
        GPU::IR::ValueId raw_index = index;
        GPU::IR::ValueId below_zero = GPU::IR::InvalidValueId;
        if (index_type.kind == GPU::IR::Type::Kind::Int) {
            raw_index = UbsanUInt(index);
            const auto signed_zero = builder_.Literal(GPU::IR::Type::Int(), "0");
            below_zero = builder_.Compare(GPU::IR::CompareOp::Less, index, signed_zero);
        }
        const auto beyond_end = builder_.Compare(GPU::IR::CompareOp::GreaterEqual, raw_index, count);
        auto invalid_index = beyond_end;
        if (below_zero != GPU::IR::InvalidValueId) {
            invalid_index = builder_.Binary(
                GPU::IR::BinaryOp::LogicalOr,
                below_zero,
                beyond_end);
        }
        const auto fallback_literal = result_type.kind == GPU::IR::Type::Kind::Bool ? "false" : "0";
        const auto fallback = builder_.Literal(result_type, fallback_literal);
        const auto zero_uint = builder_.Literal(GPU::IR::Type::UInt(), "0u");
        const auto result_name = UniqueGlslName("__feather_ubsan_buffer_value");
        const auto result = builder_.LocalVariable(result_type, result_name);
        if (index == GPU::IR::InvalidValueId || count == GPU::IR::InvalidValueId ||
            raw_index == GPU::IR::InvalidValueId || beyond_end == GPU::IR::InvalidValueId ||
            invalid_index == GPU::IR::InvalidValueId || fallback == GPU::IR::InvalidValueId ||
            zero_uint == GPU::IR::InvalidValueId || result_name.empty() ||
            result == GPU::IR::InvalidValueId) {
            return InvalidValue("UBSan buffer bounds guard could not be lowered");
        }
        EmitLocalDeclaration(result_type, result_name, fallback);

        const auto invalid = CaptureDiagnosticStatements([&] {
            return EmitUbsanIssue(
                kUbsanIssueBufferOob,
                source_site,
                raw_index,
                count,
                zero_uint);
        });
        const auto valid = CaptureDiagnosticStatements([&] {
            const auto loaded = builder_.ResourceElement(resource.id, index);
            if (loaded == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitStore(result, loaded);
            return true;
        });
        if (!invalid.has_value() || !valid.has_value()) {
            return InvalidValue("UBSan buffer bounds branches could not be lowered");
        }
        EmitIf(
            invalid_index,
            AddBlock(std::move(*invalid)),
            AddBlock(std::move(*valid)));
        return result;
    }

    bool EmitDiagnosticSiteHitsForSequence(uint32_t statement_id) {
        if (statement_id >= typed_.statements.size()) {
            return Fail("diagnostic statement sequence is outside the section 7 statement table");
        }

        const auto& statement = typed_.statements[statement_id];
        if (statement.kind != kStatementBlock) {
            return EmitDiagnosticSiteHit(statement_id);
        }
        if (statement.child_count == 0) {
            return statement.first_child == NoIndex;
        }
        if (statement.first_child == NoIndex || statement.first_child > typed_.children.size() ||
            statement.child_count > typed_.children.size() - statement.first_child) {
            return Fail("diagnostic statement sequence has an invalid block child range");
        }

        for (uint32_t index = 0; index < statement.child_count; ++index) {
            if (!EmitDiagnosticSiteHitsForSequence(typed_.children[statement.first_child + index])) {
                return false;
            }
        }
        return true;
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

        auto value = BuildExpression(statement.a);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        value = MaterializeDiagnosticValue(current_statement_id_, value, statement.op);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        local_values_[*name] = builder_.LocalVariable(type, glsl_name);
        declared_locals_[*name] = type;
        local_glsl_names_[*name] = glsl_name;
        EmitLocalDeclaration(type, glsl_name, value);
        return local_values_[*name] != GPU::IR::InvalidValueId &&
               EmitLineValueRecord(current_statement_id_, local_values_[*name], statement.op) &&
               EmitComputeTraceEvent(
                   current_statement_id_,
                   kTraceEventValue,
                   current_function_id_,
                   statement.name_id,
                   local_values_[*name],
                   statement.op);
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

        auto value = BuildExpression(statement.b);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        value = MaterializeDiagnosticValue(current_statement_id_, value,
                                           typed_.lvalues[statement.a].type_id);
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
            return EmitLineValueRecord(
                       current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
                   EmitComputeTraceEvent(
                       current_statement_id_,
                       kTraceEventValue,
                       current_function_id_,
                       target.name_id,
                       value,
                       target.type_id);
        }

        const auto destination = BuildLValueAddress(statement.a);
        if (destination == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(destination, value);
        return EmitLineValueRecord(
                   current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
               EmitComputeTraceEvent(
                   current_statement_id_,
                   kTraceEventValue,
                   current_function_id_,
                   target.name_id,
                   value,
                   target.type_id);
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

        const auto target_type_id = typed_.lvalues[statement.a].type_id;
        const auto target_type = ToModuleType(target_type_id);
        auto value = op == GPU::IR::BinaryOp::Div &&
                             UbsanEnabled(kUbsanCheckFloatDivideByZero) &&
                             target_type.kind == GPU::IR::Type::Kind::Float
                         ? BuildUbsanSafeDivision(left, right, current_statement_id_)
                         : builder_.Binary(op, left, right);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        value = MaterializeDiagnosticValue(current_statement_id_, value, target_type_id);
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
            return EmitLineValueRecord(
                       current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
                   EmitComputeTraceEvent(
                       current_statement_id_,
                       kTraceEventValue,
                       current_function_id_,
                       target.name_id,
                       value,
                       target.type_id);
        }

        const auto address = BuildLValueAddress(statement.a);
        if (address == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(address, value);
        return EmitLineValueRecord(
                   current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
               EmitComputeTraceEvent(
                   current_statement_id_,
                   kTraceEventValue,
                   current_function_id_,
                   target.name_id,
                   value,
                   target.type_id);
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
        auto value = builder_.Binary(op, current, one);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        value = MaterializeDiagnosticValue(current_statement_id_, value,
                                           typed_.lvalues[statement.a].type_id);
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
            return EmitLineValueRecord(
                       current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
                   EmitComputeTraceEvent(
                       current_statement_id_,
                       kTraceEventValue,
                       current_function_id_,
                       target.name_id,
                       value,
                       target.type_id);
        }

        const auto address = BuildLValueAddress(statement.a);
        if (address == GPU::IR::InvalidValueId) {
            return false;
        }

        EmitStore(address, value);
        return EmitLineValueRecord(
                   current_statement_id_, value, typed_.lvalues[statement.a].type_id) &&
               EmitComputeTraceEvent(
                   current_statement_id_,
                   kTraceEventValue,
                   current_function_id_,
                   target.name_id,
                   value,
                   target.type_id);
    }

    bool LowerIf(const Statement& statement) {
        if (statement.a >= typed_.expressions.size() || statement.b >= typed_.statements.size()) {
            return false;
        }

        auto condition = BuildExpression(statement.a);
        if (condition == GPU::IR::InvalidValueId) {
            return false;
        }
        if (inputs_.diagnostic_mode == kDiagnosticBranchDivergence &&
            current_statement_id_ == inputs_.diagnostic_source_site) {
            condition = MaterializeOnce(
                condition,
                GPU::IR::Type::Bool(),
                "feather_branch_predicate");
            if (condition == GPU::IR::InvalidValueId ||
                !EmitBranchDivergenceCapture(condition)) {
                return false;
            }
        }
        if (inputs_.diagnostic_mode == kDiagnosticComputeTrace) {
            condition = MaterializeOnce(
                condition,
                GPU::IR::Type::Bool(),
                "__feather_compute_trace_predicate");
            if (condition == GPU::IR::InvalidValueId ||
                !EmitComputeTraceEvent(
                    current_statement_id_,
                    kTraceEventBranchPredicate,
                    current_function_id_,
                    NoIndex,
                    condition,
                    typed_.expressions[statement.a].type_id)) {
                return false;
            }
        }
        if (inputs_.diagnostic_mode == kDiagnosticCounterfactual &&
            current_statement_id_ == inputs_.diagnostic_source_site) {
            if (inputs_.diagnostic_transform_kind != kCounterfactualForceIfFalse) {
                return Fail("counterfactual if transformation is unsupported");
            }
            condition = builder_.Literal(GPU::IR::Type::Bool(), "false");
            if (condition == GPU::IR::InvalidValueId) {
                return Fail("counterfactual false predicate could not be materialized");
            }
            counterfactual_site_emitted_ = true;
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
            // GLSL's for initializer is either a declaration or an expression list;
            // an atomic expression cannot precede a declaration in that header. Count
            // source-level initializer sites immediately before the loop, then suppress
            // the normal in-header diagnostic emission while lowering the initializer.
            if (inputs_.diagnostic_mode == 1 && !EmitDiagnosticSiteHitsForSequence(statement.a)) {
                return false;
            }
            auto init_statements = LowerStatementListKeepingLocals(
                statement.a,
                inputs_.diagnostic_mode == 1);
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
            if (inputs_.diagnostic_mode == kDiagnosticComputeTrace) {
                const bool ended = current_function_id_ == typed_.entry_function
                                       ? EmitComputeTraceEntryEnd(
                                             current_function_id_, current_statement_id_)
                                       : EmitComputeTraceCallableEnd(
                                             current_function_id_, current_statement_id_);
                if (!ended) {
                    return false;
                }
            }
            EmitReturn(GPU::IR::InvalidValueId);
            return true;
        }

        if (statement.a >= typed_.expressions.size()) {
            return false;
        }

        auto value = BuildExpression(statement.a);
        if (value == GPU::IR::InvalidValueId) {
            return false;
        }

        value = MaterializeDiagnosticValue(
            current_statement_id_, value, typed_.expressions[statement.a].type_id);
        if (value == GPU::IR::InvalidValueId ||
            !EmitLineValueRecord(
                current_statement_id_, value, typed_.expressions[statement.a].type_id)) {
            return false;
        }

        if (!EmitComputeTraceEvent(
                current_statement_id_,
                kTraceEventValue,
                current_function_id_,
                NoIndex,
                value,
                typed_.expressions[statement.a].type_id)) {
            return false;
        }
        if (inputs_.diagnostic_mode == kDiagnosticComputeTrace) {
            const bool ended = current_function_id_ == typed_.entry_function
                                   ? EmitComputeTraceEntryEnd(
                                         current_function_id_, current_statement_id_)
                                   : EmitComputeTraceCallableEnd(
                                         current_function_id_, current_statement_id_);
            if (!ended) {
                return false;
            }
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

    std::optional<CallableBody> LowerCallableStatementList(
        uint32_t statement_id,
        uint32_t function_id) {
        const auto previous_capture = capture_;
        const auto previous_callable_blocks = callable_blocks_;
        std::vector<GPU::IR::Statement> captured;
        std::vector<GPU::IR::Block> blocks;
        capture_ = &captured;
        callable_blocks_ = &blocks;
        const auto ok = EmitComputeTraceCallableStart(function_id, statement_id) &&
                        LowerStatement(statement_id) &&
                        EmitComputeTraceCallableEnd(function_id, statement_id);
        capture_ = previous_capture;
        callable_blocks_ = previous_callable_blocks;
        if (!ok) {
            return std::nullopt;
        }

        return CallableBody{std::move(captured), std::move(blocks)};
    }

    std::optional<std::vector<GPU::IR::Statement>> LowerStatementListKeepingLocals(
        uint32_t statement_id,
        bool suppress_diagnostic_site_hits = false) {
        const auto previous_capture = capture_;
        const auto previous_shared_values = shared_values_;
        if (suppress_diagnostic_site_hits) {
            ++diagnostic_site_suppression_depth_;
        }
        std::vector<GPU::IR::Statement> captured;
        capture_ = &captured;
        const auto ok = LowerStatement(statement_id);
        capture_ = previous_capture;
        shared_values_ = previous_shared_values;
        if (suppress_diagnostic_site_hits) {
            --diagnostic_site_suppression_depth_;
        }
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

        const auto active = active_resource_parameters_.find(*name);
        const auto info = active != active_resource_parameters_.end()
            ? std::optional<RegisteredResource>{active->second}
            : resource_infos_by_name_.find(*name) != resource_infos_by_name_.end()
                ? std::optional<RegisteredResource>{resource_infos_by_name_.find(*name)->second}
                : std::nullopt;
        if (!info.has_value()) {
            return InvalidValue("resource element expression references unknown resource '" + *name + "'");
        }

        const auto index = BuildExpression(expression.a);
        if (index == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        if (info->kind == kResourceKindTexture2D || info->kind == kResourceKindTexture3D) {
            return BuildTextureElement(info->id, index);
        }

        return BuildUbsanCheckedResourceRead(
            *info,
            index,
            expression.a < typed_.expressions.size()
                ? typed_.expressions[expression.a].type_id
                : NoIndex,
            current_statement_id_);
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

        const auto result_type = ToModuleType(expression.type_id);
        const auto fma_type =
            result_type.kind == GPU::IR::Type::Kind::Float || result_type.kind == GPU::IR::Type::Kind::Float2 ||
            result_type.kind == GPU::IR::Type::Kind::Float3 || result_type.kind == GPU::IR::Type::Kind::Float4;
        if (inputs_.enable_fused_multiply_add && op == GPU::IR::BinaryOp::Add && fma_type &&
            expression.a < typed_.expressions.size() && expression.b < typed_.expressions.size()) {
            const auto& multiply = typed_.expressions[expression.a];
            GPU::IR::BinaryOp multiply_op{};
            const auto has_result_type = [&](uint32_t expression_id) {
                return expression_id < typed_.expressions.size() &&
                       ToModuleType(typed_.expressions[expression_id].type_id).kind == result_type.kind;
            };
            if (multiply.kind == kExpressionBinary && TryMapBinaryOp(multiply.op, &multiply_op) &&
                multiply_op == GPU::IR::BinaryOp::Mul && has_result_type(expression.a) && has_result_type(multiply.a) &&
                has_result_type(multiply.b) && has_result_type(expression.b)) {
                const std::array arguments{BuildExpression(multiply.a), BuildExpression(multiply.b),
                                           BuildExpression(expression.b)};
                if (std::find(arguments.begin(), arguments.end(), GPU::IR::InvalidValueId) != arguments.end()) {
                    return GPU::IR::InvalidValueId;
                }
                return builder_.Intrinsic("fma", result_type, arguments);
            }
        }

        const auto left = BuildExpression(expression.a);
        const auto right = BuildExpression(expression.b);
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        if (op == GPU::IR::BinaryOp::Div &&
            UbsanEnabled(kUbsanCheckFloatDivideByZero) &&
            result_type.kind == GPU::IR::Type::Kind::Float &&
            expression.a < typed_.expressions.size() &&
            expression.b < typed_.expressions.size() &&
            ToModuleType(typed_.expressions[expression.a].type_id).kind == GPU::IR::Type::Kind::Float &&
            ToModuleType(typed_.expressions[expression.b].type_id).kind == GPU::IR::Type::Kind::Float) {
            return BuildUbsanSafeDivision(left, right, current_statement_id_);
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

    struct PrintAssertPayloadInfo {
        GPU::IR::Type scalar_type;
        uint32_t type_tag = 0;
        uint32_t component_count = 0;
    };

    std::optional<PrintAssertPayloadInfo> PrintAssertPayloadType(uint32_t type_id) const {
        const auto type = ToModuleType(type_id);
        PrintAssertPayloadInfo info;
        switch (type.kind) {
        case GPU::IR::Type::Kind::Bool:
            info = {GPU::IR::Type::Bool(), 1u, 1u};
            break;
        case GPU::IR::Type::Kind::Bool2:
        case GPU::IR::Type::Kind::Bool3:
        case GPU::IR::Type::Kind::Bool4:
            info = {GPU::IR::Type::Bool(), 1u,
                    type.kind == GPU::IR::Type::Kind::Bool2 ? 2u :
                    type.kind == GPU::IR::Type::Kind::Bool3 ? 3u : 4u};
            break;
        case GPU::IR::Type::Kind::Int:
            info = {GPU::IR::Type::Int(), 2u, 1u};
            break;
        case GPU::IR::Type::Kind::Int2:
        case GPU::IR::Type::Kind::Int3:
        case GPU::IR::Type::Kind::Int4:
            info = {GPU::IR::Type::Int(), 2u,
                    type.kind == GPU::IR::Type::Kind::Int2 ? 2u :
                    type.kind == GPU::IR::Type::Kind::Int3 ? 3u : 4u};
            break;
        case GPU::IR::Type::Kind::UInt:
            info = {GPU::IR::Type::UInt(), 3u, 1u};
            break;
        case GPU::IR::Type::Kind::UInt2:
        case GPU::IR::Type::Kind::UInt3:
        case GPU::IR::Type::Kind::UInt4:
            info = {GPU::IR::Type::UInt(), 3u,
                    type.kind == GPU::IR::Type::Kind::UInt2 ? 2u :
                    type.kind == GPU::IR::Type::Kind::UInt3 ? 3u : 4u};
            break;
        case GPU::IR::Type::Kind::Float:
            info = {GPU::IR::Type::Float(), 4u, 1u};
            break;
        case GPU::IR::Type::Kind::Float2:
        case GPU::IR::Type::Kind::Float3:
        case GPU::IR::Type::Kind::Float4:
            info = {GPU::IR::Type::Float(), 4u,
                    type.kind == GPU::IR::Type::Kind::Float2 ? 2u :
                    type.kind == GPU::IR::Type::Kind::Float3 ? 3u : 4u};
            break;
        default:
            return std::nullopt;
        }
        return info;
    }

    GPU::IR::ValueId PrintAssertWord(uint32_t index) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(index) + "u");
        return literal == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, literal);
    }

    GPU::IR::ValueId PrintAssertWord(GPU::IR::ValueId base, uint32_t offset) {
        const auto literal = builder_.Literal(
            GPU::IR::Type::UInt(), std::to_string(offset) + "u");
        const auto index = builder_.Binary(GPU::IR::BinaryOp::Add, base, literal);
        return literal == GPU::IR::InvalidValueId || index == GPU::IR::InvalidValueId
                   ? GPU::IR::InvalidValueId
                   : builder_.ResourceElement(diagnostic_sites_resource_, index);
    }

    GPU::IR::ValueId EncodePrintAssertComponent(
        GPU::IR::ValueId payload,
        const PrintAssertPayloadInfo& info,
        uint32_t component_index) {
        auto component = payload;
        if (info.component_count > 1u) {
            constexpr std::string_view components = "xyzw";
            component = builder_.Swizzle(
                payload,
                info.scalar_type,
                std::string(1, components[component_index]));
        }
        const auto uint_type = GPU::IR::Type::UInt();
        if (component == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }
        if (info.type_tag == 1u) {
            const auto one = builder_.Literal(uint_type, "1u");
            const auto zero = builder_.Literal(uint_type, "0u");
            return one == GPU::IR::InvalidValueId || zero == GPU::IR::InvalidValueId
                       ? GPU::IR::InvalidValueId
                       : builder_.Ternary(component, one, zero);
        }
        if (info.type_tag == 2u) {
            const std::array arguments{component};
            return builder_.Intrinsic("uint", uint_type, arguments);
        }
        if (info.type_tag == 3u) {
            return component;
        }
        const std::array arguments{component};
        return builder_.Intrinsic("floatBitsToUint", uint_type, arguments);
    }

    bool EmitPrintAssertRecord(
        uint32_t code,
        uint32_t severity,
        GPU::IR::ValueId x,
        GPU::IR::ValueId y,
        GPU::IR::ValueId z,
        GPU::IR::ValueId linear,
        GPU::IR::ValueId payload,
        const PrintAssertPayloadInfo& payload_info) {
        const auto uint_type = GPU::IR::Type::UInt();
        const auto one = builder_.Literal(uint_type, "1u");
        const auto sixteen = builder_.Literal(uint_type, "16u");
        const auto eight = builder_.Literal(uint_type, "8u");
        const std::array increment_arguments{one};
        const auto attempted = builder_.Atomic(
            GPU::IR::AtomicOp::Add,
            uint_type,
            PrintAssertWord(3u),
            increment_arguments);
        const auto slot = MaterializeOnce(
            attempted, uint_type, "feather_print_assert_slot");
        const auto capacity = PrintAssertWord(2u);
        const auto has_capacity = builder_.Compare(
            GPU::IR::CompareOp::Less, slot, capacity);
        const auto record_offset = builder_.Binary(
            GPU::IR::BinaryOp::Mul, slot, sixteen);
        const auto record_base = builder_.Binary(
            GPU::IR::BinaryOp::Add, eight, record_offset);
        if (one == GPU::IR::InvalidValueId || sixteen == GPU::IR::InvalidValueId ||
            eight == GPU::IR::InvalidValueId || attempted == GPU::IR::InvalidValueId ||
            slot == GPU::IR::InvalidValueId || capacity == GPU::IR::InvalidValueId ||
            has_capacity == GPU::IR::InvalidValueId || record_offset == GPU::IR::InvalidValueId ||
            record_base == GPU::IR::InvalidValueId) {
            return Fail("Print/Assert bounded record reservation could not be lowered");
        }

        const auto committed = CaptureDiagnosticStatements([&] {
            auto store_constant = [&](uint32_t offset, uint32_t value) {
                EmitStore(
                    PrintAssertWord(record_base, offset),
                    builder_.Literal(uint_type, std::to_string(value) + "u"));
            };
            store_constant(0u, code);
            store_constant(1u, current_statement_id_);
            store_constant(2u, 3u);
            store_constant(3u, severity);
            EmitStore(PrintAssertWord(record_base, 4u), x);
            EmitStore(PrintAssertWord(record_base, 5u), y);
            EmitStore(PrintAssertWord(record_base, 6u), z);
            EmitStore(PrintAssertWord(record_base, 7u), linear);
            store_constant(8u, payload_info.type_tag);
            store_constant(9u, payload_info.component_count);
            for (uint32_t component = 0; component < payload_info.component_count; ++component) {
                const auto raw = EncodePrintAssertComponent(payload, payload_info, component);
                if (raw == GPU::IR::InvalidValueId) {
                    return false;
                }
                EmitStore(PrintAssertWord(record_base, 10u + component), raw);
            }
            for (uint32_t component = payload_info.component_count; component < 4u; ++component) {
                store_constant(10u + component, 0u);
            }
            store_constant(14u, code == 2u ? 1u : 0u);
            store_constant(15u, 0u);
            const auto committed_increment = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                PrintAssertWord(4u),
                increment_arguments);
            if (committed_increment == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitExpression(committed_increment);
            return true;
        });
        const auto dropped = CaptureDiagnosticStatements([&] {
            const auto dropped_increment = builder_.Atomic(
                GPU::IR::AtomicOp::Add,
                uint_type,
                PrintAssertWord(5u),
                increment_arguments);
            if (dropped_increment == GPU::IR::InvalidValueId) {
                return false;
            }
            EmitExpression(dropped_increment);
            return true;
        });
        if (!committed.has_value() || !dropped.has_value()) {
            return Fail("Print/Assert record commit or drop accounting could not be lowered");
        }
        EmitIf(
            has_capacity,
            AddBlock(std::move(*committed)),
            AddBlock(std::move(*dropped)));
        return true;
    }

    bool EmitPrintAssertMask(GPU::IR::ValueId linear) {
        const auto uint_type = GPU::IR::Type::UInt();
        const auto one = builder_.Literal(uint_type, "1u");
        const uint32_t mask_base_index = 8u + inputs_.diagnostic_record_capacity * 16u;
        const auto mask_base = builder_.Literal(
            uint_type, std::to_string(mask_base_index) + "u");
        const auto four = builder_.Literal(uint_type, "4u");
        const auto cell_offset = builder_.Binary(
            GPU::IR::BinaryOp::Add, four, linear);
        const auto cell_index = builder_.Binary(
            GPU::IR::BinaryOp::Add, mask_base, cell_offset);
        const std::array increment_arguments{one};
        const auto failure_count = builder_.Atomic(
            GPU::IR::AtomicOp::Add,
            uint_type,
            PrintAssertWord(mask_base_index + 3u),
            increment_arguments);
        if (one == GPU::IR::InvalidValueId || mask_base == GPU::IR::InvalidValueId ||
            four == GPU::IR::InvalidValueId || cell_offset == GPU::IR::InvalidValueId ||
            cell_index == GPU::IR::InvalidValueId || failure_count == GPU::IR::InvalidValueId) {
            return Fail("Print/Assert assertion mask address could not be lowered");
        }
        EmitExpression(failure_count);
        EmitStore(builder_.ResourceElement(diagnostic_sites_resource_, cell_index), one);
        return true;
    }

    GPU::IR::ValueId BuildPrintAssertIntrinsic(
        const Expression& expression,
        std::string_view intrinsic,
        GPU::IR::Type result_type,
        const std::vector<GPU::IR::ValueId>& arguments) {
        const bool is_print = intrinsic == "feather_debug_print";
        if ((is_print && arguments.size() != 1u) ||
            (!is_print && arguments.size() != 1u && arguments.size() != 2u) ||
            expression.first_argument == NoIndex ||
            expression.first_argument > typed_.arguments.size() ||
            expression.argument_count > typed_.arguments.size() - expression.first_argument) {
            return InvalidValue("GpuDebug marker has an invalid argument layout");
        }

        const uint32_t payload_argument = is_print || arguments.size() == 1u ? 0u : 1u;
        const uint32_t payload_expression_id = typed_.arguments[
            expression.first_argument + payload_argument];
        if (payload_expression_id >= typed_.expressions.size()) {
            return InvalidValue("GpuDebug payload expression is outside the typed expression table");
        }
        const auto payload_info = PrintAssertPayloadType(
            typed_.expressions[payload_expression_id].type_id);
        if (!payload_info.has_value()) {
            return InvalidValue("GpuDebug payload must be a 32-bit scalar or 2-4 component vector");
        }

        auto condition = is_print
                             ? builder_.Literal(GPU::IR::Type::Bool(), "true")
                             : arguments[0];
        auto payload = arguments[payload_argument];
        if (condition == GPU::IR::InvalidValueId || payload == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }
        if (inputs_.diagnostic_mode != kDiagnosticPrintAssert) {
            return is_print ? payload : condition;
        }
        if (!is_print) {
            condition = MaterializeOnce(
                condition, GPU::IR::Type::Bool(), "feather_debug_condition");
        }
        if (!is_print && arguments.size() == 1u) {
            payload = condition;
        } else {
            payload = MaterializeOnce(
                payload,
                ToModuleType(typed_.expressions[payload_expression_id].type_id),
                "feather_debug_payload");
        }
        if (condition == GPU::IR::InvalidValueId || payload == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        if (diagnostic_sites_resource_ == GPU::IR::InvalidResourceId ||
            current_statement_id_ >= inputs_.diagnostic_site_count) {
            return InvalidValue("GpuDebug marker has no configured diagnostic stream or source site");
        }

        const auto uint_type = GPU::IR::Type::UInt();
        const auto x = UbsanUInt(builder_.ThreadIndexX());
        const auto y = UbsanUInt(builder_.ThreadIndexY());
        const auto z = UbsanUInt(builder_.ThreadIndexZ());
        const auto logical_x = UbsanUInt(
            builder_.PushConstant(logical_size_resource_[0]));
        const auto logical_y = UbsanUInt(
            builder_.PushConstant(logical_size_resource_[1]));
        const auto y_plus_z = builder_.Binary(
            GPU::IR::BinaryOp::Add,
            y,
            builder_.Binary(GPU::IR::BinaryOp::Mul, logical_y, z));
        const auto linear = builder_.Binary(
            GPU::IR::BinaryOp::Add,
            x,
            builder_.Binary(GPU::IR::BinaryOp::Mul, logical_x, y_plus_z));
        auto selected = builder_.Literal(GPU::IR::Type::Bool(), "true");
        if (inputs_.diagnostic_filter_mode == 1u) {
            const auto selected_x = builder_.Literal(
                uint_type, std::to_string(inputs_.diagnostic_selected_x) + "u");
            const auto selected_y = builder_.Literal(
                uint_type, std::to_string(inputs_.diagnostic_selected_y) + "u");
            const auto selected_z = builder_.Literal(
                uint_type, std::to_string(inputs_.diagnostic_selected_z) + "u");
            selected = builder_.Compare(GPU::IR::CompareOp::Equal, x, selected_x);
            selected = builder_.Binary(
                GPU::IR::BinaryOp::LogicalAnd,
                selected,
                builder_.Compare(GPU::IR::CompareOp::Equal, y, selected_y));
            selected = builder_.Binary(
                GPU::IR::BinaryOp::LogicalAnd,
                selected,
                builder_.Compare(GPU::IR::CompareOp::Equal, z, selected_z));
        }
        if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId ||
            z == GPU::IR::InvalidValueId || logical_x == GPU::IR::InvalidValueId ||
            logical_y == GPU::IR::InvalidValueId || y_plus_z == GPU::IR::InvalidValueId ||
            linear == GPU::IR::InvalidValueId || selected == GPU::IR::InvalidValueId) {
            return InvalidValue("GpuDebug invocation identity could not be lowered");
        }

        if (is_print) {
            const auto record = CaptureDiagnosticStatements([&] {
                return EmitPrintAssertRecord(
                    1u, 1u, x, y, z, linear, payload, *payload_info);
            });
            if (!record.has_value()) {
                return GPU::IR::InvalidValueId;
            }
            EmitIf(selected, AddBlock(std::move(*record)), GPU::IR::InvalidBlockId);
            return payload;
        }

        const auto failed = builder_.Unary(GPU::IR::UnaryOp::LogicalNot, condition);
        const auto failure = CaptureDiagnosticStatements([&] {
            if (!EmitPrintAssertMask(linear)) {
                return false;
            }
            const auto record = CaptureDiagnosticStatements([&] {
                return EmitPrintAssertRecord(
                    2u, 3u, x, y, z, linear, payload, *payload_info);
            });
            if (!record.has_value()) {
                return false;
            }
            EmitIf(selected, AddBlock(std::move(*record)), GPU::IR::InvalidBlockId);
            return true;
        });
        if (failed == GPU::IR::InvalidValueId || !failure.has_value()) {
            return InvalidValue("GpuDebug assertion instrumentation could not be lowered");
        }
        EmitIf(failed, AddBlock(std::move(*failure)), GPU::IR::InvalidBlockId);
        return condition;
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

        if (intrinsic == "feather_debug_print" || intrinsic == "feather_debug_assert") {
            return BuildPrintAssertIntrinsic(expression, intrinsic, result_type, *arguments);
        }

        if (intrinsic == "matrix_multiply") {
            if (arguments->size() != 2) {
                return GPU::IR::InvalidValueId;
            }

            return builder_.Binary(GPU::IR::BinaryOp::Mul, (*arguments)[0], (*arguments)[1]);
        }

        if (result_type.kind == GPU::IR::Type::Kind::Float && arguments->size() == 1) {
            if (intrinsic == "sqrt" && UbsanEnabled(kUbsanCheckSqrtDomain)) {
                return BuildUbsanSafeDomainIntrinsic(
                    intrinsic,
                    kUbsanIssueSqrtDomain,
                    GPU::IR::CompareOp::Less,
                    (*arguments)[0],
                    current_statement_id_);
            }
            if (intrinsic == "log" && UbsanEnabled(kUbsanCheckLogDomain)) {
                return BuildUbsanSafeDomainIntrinsic(
                    intrinsic,
                    kUbsanIssueLogDomain,
                    GPU::IR::CompareOp::LessEqual,
                    (*arguments)[0],
                    current_statement_id_);
            }
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

        const auto callable = typed_.callables.find(*raw_name);
        if (callable == typed_.callables.end() || callable->second.function_index >= typed_.functions.size()) {
            return GPU::IR::InvalidValueId;
        }

        const auto& function = typed_.functions[callable->second.function_index];
        if (function.parameter_count != expression.argument_count ||
            (function.parameter_count > 0 &&
             (function.first_parameter == NoIndex || expression.first_argument == NoIndex ||
              function.first_parameter > typed_.parameters.size() ||
              function.parameter_count > typed_.parameters.size() - function.first_parameter ||
              expression.first_argument > typed_.arguments.size() ||
              expression.argument_count > typed_.arguments.size() - expression.first_argument))) {
            return GPU::IR::InvalidValueId;
        }

        std::vector<GPU::IR::ValueId> arguments;
        arguments.reserve(expression.argument_count);
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            const auto argument_id = typed_.arguments[expression.first_argument + i];
            const auto& parameter = typed_.parameters[function.first_parameter + i];
            if (parameter.type_id < typed_.types.size() &&
                typed_.types[parameter.type_id].kind == kTypeResourceWrapper) {
                const auto resource = ResolveResourceReference(argument_id);
                const auto binding = callable_resource_bindings_[*raw_name].find(i);
                if (!resource.has_value() || binding == callable_resource_bindings_[*raw_name].end() ||
                    resource->id != binding->second.id) {
                    return GPU::IR::InvalidValueId;
                }
                continue;
            }

            const auto argument = BuildExpression(argument_id);
            if (argument == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }
            arguments.push_back(argument);
        }

        return builder_.Call(mapped->second, result_type, arguments);
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
        if (instance == GPU::IR::InvalidValueId || !result_type.IsValid() || member == nullptr || member->empty()) {
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
        const auto index = BuildExpression(expression.b);
        const auto result_type = ToModuleType(expression.type_id);
        if (index == GPU::IR::InvalidValueId || !result_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        if (const auto resource = ResolveResourceReference(expression.a); resource.has_value()) {
            if (resource->kind == kResourceKindBuffer) {
                return builder_.ResourceElement(resource->id, index);
            }
            if (resource->kind == kResourceKindTexture2D || resource->kind == kResourceKindTexture3D) {
                return BuildTextureElement(resource->id, index);
            }
        }

        const auto instance = BuildExpression(expression.a);
        if (instance == GPU::IR::InvalidValueId) {
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

        const auto active = active_resource_parameters_.find(*name);
        if (active != active_resource_parameters_.end()) {
            return active->second;
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

        const auto active = active_resource_parameters_.find(*name);
        const auto info = active != active_resource_parameters_.end()
                              ? std::optional<RegisteredResource>{active->second}
                          : resource_infos_by_name_.find(*name) != resource_infos_by_name_.end()
                              ? std::optional<RegisteredResource>{resource_infos_by_name_.find(*name)->second}
                              : std::nullopt;
        if (!info.has_value()) {
            return InvalidValue("resource l-value references unknown resource '" + *name + "'");
        }

        const auto index = BuildExpression(lvalue.a);
        if (index == GPU::IR::InvalidValueId) {
            return GPU::IR::InvalidValueId;
        }

        if (info->kind == kResourceKindTexture2D || info->kind == kResourceKindTexture3D) {
            return BuildTextureElement(info->id, index);
        }

        return builder_.ResourceElement(info->id, index);
    }

    std::optional<RegisteredResource> ResolveResourceLValue(uint32_t lvalue_id) const {
        if (lvalue_id >= typed_.lvalues.size()) {
            return std::nullopt;
        }
        const auto& lvalue = typed_.lvalues[lvalue_id];
        if (lvalue.kind != kLValueLocal && lvalue.kind != kLValueParameter) {
            return std::nullopt;
        }
        const auto* name = GetString(lvalue.name_id);
        if (name == nullptr) {
            return std::nullopt;
        }
        const auto active = active_resource_parameters_.find(*name);
        if (active != active_resource_parameters_.end()) {
            return active->second;
        }
        const auto resource = resource_infos_by_name_.find(*name);
        return resource == resource_infos_by_name_.end() ? std::nullopt
                                                         : std::optional<RegisteredResource>{resource->second};
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
        if (vector == GPU::IR::InvalidValueId || !type.IsValid() || components == nullptr || components->empty()) {
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
        if (const auto resource = ResolveResourceLValue(lvalue.a); resource.has_value()) {
            const auto index = BuildExpression(lvalue.b);
            if (index == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }
            if (resource->kind == kResourceKindBuffer) {
                return builder_.ResourceElement(resource->id, index);
            }
            if (resource->kind == kResourceKindTexture2D || resource->kind == kResourceKindTexture3D) {
                return BuildTextureElement(resource->id, index);
            }
            return GPU::IR::InvalidValueId;
        }

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
        if (symbol == "global::Feather.GpuDebug.Print") {
            return "feather_debug_print";
        }
        if (symbol == "global::Feather.GpuDebug.Assert") {
            return "feather_debug_assert";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Sin" || symbol == "global::Feather.Math.Hlsl.Sin") {
            return "sin";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Cos" || symbol == "global::Feather.Math.Hlsl.Cos") {
            return "cos";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Tan" || symbol == "global::Feather.Math.Hlsl.Tan") {
            return "tan";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Sinh" || symbol == "global::Feather.Math.Hlsl.Sinh") {
            return "sinh";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Cosh" || symbol == "global::Feather.Math.Hlsl.Cosh") {
            return "cosh";
        }
        if (symbol == "global::Feather.Math.ShaderMath.Tanh" || symbol == "global::Feather.Math.Hlsl.Tanh") {
            return "tanh";
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
        if (symbol == "global::Feather.Math.ShaderMath.Reflect" ||
            symbol == "global::Feather.Math.Hlsl.Reflect") {
            return "reflect";
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
    std::unordered_map<std::string, std::unordered_map<uint32_t, RegisteredResource>> callable_resource_bindings_;
    std::unordered_map<std::string, RegisteredResource> active_resource_parameters_;
    struct SharedMemoryInfo {
        GPU::IR::Type type;
        std::string glsl_name;
    };
    std::unordered_map<std::string, SharedMemoryInfo> shared_values_;
    std::array<GPU::IR::ResourceId, 3> logical_size_resource_{GPU::IR::InvalidResourceId, GPU::IR::InvalidResourceId,
                                                              GPU::IR::InvalidResourceId};
    GPU::IR::ResourceId diagnostic_sites_resource_ = GPU::IR::InvalidResourceId;
    uint32_t diagnostic_count_resource_count_ = 0;
    uint32_t diagnostic_site_suppression_depth_ = 0;
    uint32_t current_statement_id_ = NoIndex;
    uint32_t current_function_id_ = NoIndex;
    bool branch_divergence_site_emitted_ = false;
    bool counterfactual_site_emitted_ = false;
    std::vector<GPU::IR::Statement>* capture_ = nullptr;
    std::vector<GPU::IR::Block>* callable_blocks_ = nullptr;
};

} // namespace

std::unique_ptr<GPU::IR::Module> TryLowerToEasyGpuModule(const Module& typed, const LoweringInputs& inputs,
                                                         std::string* error) {
    if (error != nullptr) {
        error->clear();
    }

    return Lowerer(typed, inputs, error).Lower();
}

} // namespace Feather::TypedIR
