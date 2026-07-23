#include "feather_c_api.h"
#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <variant>
#include <type_traits>
#include <vector>

#include <AD/ADKernel.h>
#include <AD/AdjointGenerator.h>
#include <AD/GradientTape.h>
#include <Backend/Backend.h>
#include <IR/Module.h>
#include <Kernel/KernelBuildContext.h>
#include <Runtime/Context.h>
#include <Runtime/Texture.h>

#if FEATHER_BUILD_WINDOW
#include <Window/AppWindow.h>
#include <Window/Input.h>
#include <Window/TexturePresenter.h>
#include <Window/WindowConfig.h>
#include <Window/WindowEvents.h>
#endif

namespace {

void trace_graphics_step(const char* step);

#ifndef FEATHER_SHADER_OPTIMIZATION_LEVEL
#define FEATHER_SHADER_OPTIMIZATION_LEVEL GPU::Backend::ShaderOptimizationLevel::Ultra
#endif

constexpr GPU::Backend::ShaderOptimizationLevel kShaderOptimizationLevel = FEATHER_SHADER_OPTIMIZATION_LEVEL;
constexpr bool kEnableFusedMultiplyAdd = kShaderOptimizationLevel == GPU::Backend::ShaderOptimizationLevel::Ultra ||
                                         kShaderOptimizationLevel == GPU::Backend::ShaderOptimizationLevel::Extreme;
constexpr FeContextHandle kDefaultContext = 1;
constexpr uint8_t kIrOpcodeIf = 4;
constexpr uint8_t kIrOpcodeBeginBlock = 13;
constexpr uint8_t kIrOpcodeElse = 14;
constexpr uint8_t kIrOpcodeEndBlock = 15;
constexpr uint8_t kIrOpcodeWorkgroupBarrier = 16;
constexpr uint8_t kIrOpcodeMemoryBarrier = 17;
constexpr uint8_t kIrOpcodeFullBarrier = 18;
constexpr uint8_t kIrOpcodeAssignment = 2;
constexpr uint8_t kIrOpcodeFor = 5;
constexpr uint8_t kIrOpcodeWhile = 6;
constexpr uint8_t kIrOpcodeDo = 7;
constexpr uint8_t kIrOpcodeBreak = 8;
constexpr uint8_t kIrOpcodeContinue = 9;
constexpr uint8_t kIrOpcodeInvocation = 10;
constexpr uint8_t kIrOpcodeResourceAccess = 11;
constexpr uint8_t kIrOpcodeExpression = 12;
constexpr uint8_t kIrOpcodeLocalDeclaration = 1;
constexpr uint8_t kIrOpcodeReturn = 3;
constexpr uint8_t kIrOpcodeSharedMemoryDeclaration = 28;
constexpr uint8_t kIrOpcodeTextureSample = 29;
constexpr uint32_t kIrSectionControlFlowExpressions = 3;
constexpr uint32_t kIrSectionAdAnnotations = 4;
constexpr uint32_t kIrSectionLocalVariables = 5;
constexpr uint32_t kIrSectionCompoundAssignments = 6;
constexpr uint8_t kIrExpressionNodeKindComparison = 6;
constexpr uint8_t kIrExpressionNodeKindPushConstant = 5;
constexpr uint8_t kIrExpressionNodeKindLocalVariable = 7;
constexpr uint8_t kIrExpressionNodeKindShaderBuiltin = 8;
constexpr uint8_t kIrExpressionNodeKindTernary = 9;
constexpr uint8_t kIrExpressionNodeKindConstructor = 10;
constexpr uint8_t kIrExpressionNodeKindCallableCall = 11;
constexpr uint8_t kIrExpressionNodeKindTextureSample = 12;
constexpr uint8_t kIrExpressionNodeKindTextureSampleLevel = 13;
constexpr uint8_t kIrExpressionNodeKindGpuStructField = 14;
constexpr uint8_t kIrExpressionNodeKindMax = 14;
constexpr uint8_t kCfRoleIfCondition = 1;
constexpr uint8_t kCfRoleForCondition = 2;
constexpr uint8_t kCfRoleForInit = 3;
constexpr uint8_t kCfRoleForStep = 4;
constexpr uint8_t kCfRoleWhileCondition = 5;
constexpr uint8_t kCfRoleDoCondition = 6;
constexpr uint8_t kIrOperandKindElementwiseAssignment = 2;
constexpr uint8_t kIrOperandKindSymbol = 3;
constexpr uint64_t kIrSectionRecordSize = 8;
constexpr uint32_t kIrSectionElementwiseAssignments = 1;
constexpr uint32_t kIrSectionElementwiseExpressionAssignments = 2;
constexpr uint64_t kIrAssignmentHeaderSize = 4;
constexpr uint64_t kIrAssignmentRecordSize = 28;
constexpr uint64_t kIrExpressionAssignmentHeaderSize = 8;
constexpr uint64_t kIrExpressionAssignmentHeaderWithArgumentsSize = 12;
constexpr uint64_t kIrExpressionAssignmentRecordSize = 16;
constexpr uint64_t kIrExpressionNodeRecordSize = 28;
constexpr uint64_t kIrExpressionNodeRecordWithArgumentsSize = 40;
constexpr uint64_t kTypedIrHeaderSize = 104;
constexpr uint32_t kIrNoBinding = UINT32_MAX;
constexpr uint32_t kIrNoString = UINT32_MAX;
constexpr uint8_t kIrResourceKindBuffer = 1;
constexpr uint8_t kIrResourceKindTexture2D = 2;
constexpr uint8_t kIrResourceKindSampler = 3;
constexpr uint8_t kIrResourceKindPushConstant = 5;
constexpr uint8_t kIrResourceKindTexture3D = 6;
constexpr uint8_t kIrBlockKindGeneric = 0;
constexpr uint8_t kIrBlockKindIfTrue = 1;
constexpr uint8_t kIrBlockKindIfElse = 2;
constexpr uint32_t kIrAdRoleParameter = 0;
constexpr uint32_t kIrAdRoleLoss = 1;
constexpr uint32_t kIrAdSourceKindBufferElement = 1;
constexpr uint32_t kIrAdSourceKindLocal = 2;
constexpr uint8_t kTypedStatementBlock = 1;
constexpr uint8_t kTypedStatementLocalDeclaration = 2;
constexpr uint8_t kTypedStatementAssignment = 3;
constexpr uint8_t kTypedStatementCompoundAssignment = 4;
constexpr uint8_t kTypedStatementIf = 5;
constexpr uint8_t kTypedStatementFor = 6;
constexpr uint8_t kTypedStatementWhile = 7;
constexpr uint8_t kTypedStatementDoWhile = 8;
constexpr uint8_t kTypedStatementBreak = 9;
constexpr uint8_t kTypedStatementContinue = 10;
constexpr uint8_t kTypedStatementReturn = 11;
constexpr uint8_t kTypedStatementExpression = 12;
constexpr uint8_t kTypedStatementIncrementDecrement = 14;
constexpr uint8_t kTypedExpressionResourceElement = 5;
constexpr uint8_t kTypedExpressionUnary = 6;
constexpr uint8_t kTypedExpressionBinary = 7;
constexpr uint8_t kTypedExpressionComparison = 8;
constexpr uint8_t kTypedExpressionLogical = 9;
constexpr uint8_t kTypedExpressionConditional = 10;
constexpr uint8_t kTypedExpressionConversion = 11;
constexpr uint8_t kTypedExpressionConstructor = 12;
constexpr uint8_t kTypedExpressionIntrinsic = 13;
constexpr uint8_t kTypedExpressionCallableCall = 14;
constexpr uint8_t kTypedExpressionSwizzle = 15;
constexpr uint8_t kTypedExpressionMemberAccess = 16;
constexpr uint8_t kTypedExpressionIndexAccess = 17;
constexpr uint8_t kTypedExpressionMatrixColumn = 20;
constexpr uint8_t kTypedExpressionAtomic = 22;
constexpr uint8_t kTypedExpressionTextureSample = 23;
constexpr uint8_t kTypedLValueLocal = 1;
constexpr uint8_t kTypedLValueParameter = 2;
constexpr uint8_t kTypedLValueField = 3;
constexpr uint8_t kTypedLValueResourceElement = 4;
constexpr uint8_t kTypedLValueSwizzle = 5;
constexpr uint8_t kTypedLValueMemberAccess = 6;
constexpr uint8_t kTypedLValueIndexAccess = 7;
constexpr uint8_t kTypedLValueMatrixColumn = 8;
constexpr uint8_t kTypedLValueSharedMemoryElement = 9;
constexpr uint32_t kTypedStructFieldFlagPosition = 1u;
constexpr uint32_t kTypedStructFieldFlagColor = 2u;
constexpr uint32_t kTypedStructFieldColorIndexShift = 8u;
constexpr uint32_t kWindowEventResize = 1;
constexpr uint32_t kWindowEventClose = 2;
constexpr uint32_t kWindowEventKey = 3;
constexpr uint32_t kWindowEventCharInput = 4;
constexpr uint32_t kWindowEventMouseButton = 5;
constexpr uint32_t kWindowEventMouseMove = 6;
constexpr uint32_t kWindowEventMouseScroll = 7;
constexpr uint32_t kWindowEventFocus = 8;

struct BufferState {
    std::vector<unsigned char> bytes;
    uint32_t mode = 0;
    uint32_t stride = 0;
    GPU::Backend::BufferHandle backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
    bool host_dirty = false;
    bool device_dirty = false;
};

template <size_t N> struct FloatVectorValue {
    float components[N]{};
};

struct TextureState {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t pixel_format = 0;
    uint32_t access = 0;
    std::vector<unsigned char> bytes;
    GPU::Backend::TextureHandle backend_texture = GPU::Backend::INVALID_TEXTURE_HANDLE;
    bool host_dirty = false;
    bool device_dirty = false;
    bool mipmaps_requested = false;
    bool mipmaps_dirty = false;
};

struct SamplerState {
    FeSamplerDesc desc{};
};

struct GraphicsPushConstantLayoutEntry {
    uint32_t binding = UINT32_MAX;
    std::string name;
    size_t offset = 0;
    size_t size = 0;
};

struct GraphicsResourceBindingEntry {
    uint32_t source_binding = UINT32_MAX;
    uint32_t backend_binding = UINT32_MAX;
    uint8_t kind = 0;
    uint8_t access = 0;
    uint32_t sampler_binding = UINT32_MAX;
};

struct GraphicsResourceLayout {
    std::vector<GraphicsResourceBindingEntry> entries;
};

struct GraphicsPipelineCacheEntry {
    std::string key;
    GPU::Backend::ShaderHandle vertex_shader = GPU::Backend::INVALID_SHADER_HANDLE;
    GPU::Backend::ShaderHandle fragment_shader = GPU::Backend::INVALID_SHADER_HANDLE;
    GPU::Backend::PipelineHandle pipeline = GPU::Backend::INVALID_PIPELINE_HANDLE;
    uint32_t push_constant_size = 0;
};

struct ADGradientState {
    std::string name;
    std::string resource_name;
    std::string element_type;
    std::string easygpu_name;
    uint32_t source_binding = 0;
    uint32_t gradient_binding = 0;
    uint32_t element_count = 0;
    uint32_t element_stride = 0;
    uint32_t component_count = 0;
    size_t byte_size = 0;
    GPU::Backend::BufferHandle backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
};

struct KernelState {
    std::vector<unsigned char> ir;
    std::vector<unsigned char> push_constants;
    std::unordered_map<uint32_t, FeBufferHandle> buffers;
    std::unordered_map<uint32_t, FeTextureHandle> textures;
    std::unordered_map<uint32_t, FeSamplerHandle> samplers;
    std::string debug_name;
    bool auto_diff = false;
    bool bounds_check = false;
    int32_t logical_x = 0;
    int32_t logical_y = 0;
    int32_t logical_z = 0;
    std::vector<unsigned char> backward_ir;
    std::string last_ad_backward_glsl;
    std::vector<ADGradientState> ad_gradients;
    GPU::Backend::BufferHandle ad_adj_pool = GPU::Backend::INVALID_BUFFER_HANDLE;
    size_t ad_adj_pool_size = 0;
    FeDispatchPath last_dispatch_path = FE_DISPATCH_PATH_NONE;
};

struct ComputeKernelCache {
    std::unique_ptr<GPU::Kernel::KernelBuildContext> context;
};

struct IrResource {
    uint32_t binding = 0;
    uint8_t kind = 0;
    uint8_t access = 0;
    uint32_t name_string_id = 0;
    uint32_t element_type_string_id = 0;
};

struct IrInstruction {
    uint8_t opcode = 0;
    uint8_t operand_kind = 0;
    uint32_t operand_string_id = 0;
};

struct IrElementwiseAssignment {
    uint32_t instruction_index = 0;
    uint32_t destination_binding = 0;
    uint32_t left_binding = 0;
    uint32_t right_binding = UINT32_MAX;
    uint8_t operation = 0;
    uint8_t right_operand_kind = 0;
    uint32_t index_string_id = 0;
    uint32_t right_literal_string_id = UINT32_MAX;
};

struct IrExpressionAssignment {
    uint32_t instruction_index = 0;
    uint32_t destination_binding = 0;
    uint32_t index_string_id = UINT32_MAX;
    uint32_t root_node_index = UINT32_MAX;
};

struct IrControlFlowExpression {
    uint32_t instruction_index = 0;
    uint8_t role = 0;
    uint32_t root_node_index = UINT32_MAX;
};

struct IrCompoundAssignment {
    uint32_t instruction_index = 0;
    uint32_t destination_binding = 0;
    uint32_t index_string_id = 0;
    uint8_t operation = 0;
    uint32_t padding = 0;
    uint32_t root_node_index = UINT32_MAX;
};

struct IrLocalVariableDecl {
    uint32_t instruction_index = 0;
    uint32_t name_string_id = 0;
    uint32_t glsl_text_string_id = 0;
};

struct IrExpressionNode {
    uint8_t kind = 0;
    uint8_t operation = 0;
    uint32_t resource_binding = UINT32_MAX;
    uint32_t index_string_id = UINT32_MAX;
    uint32_t literal_string_id = UINT32_MAX;
    uint32_t type_string_id = UINT32_MAX;
    uint32_t left_node_index = UINT32_MAX;
    uint32_t right_node_index = UINT32_MAX;
    uint32_t symbol_string_id = UINT32_MAX;
    uint32_t first_argument_index = UINT32_MAX;
    uint32_t argument_count = 0;
};

struct IrAdAnnotation {
    uint32_t role = 0;
    uint32_t binding = kIrNoBinding;
    uint32_t name_string_id = kIrNoString;
    uint32_t resource_name_string_id = kIrNoString;
    uint32_t type_name_string_id = kIrNoString;
    uint32_t index_string_id = kIrNoString;
    uint32_t source_kind = 0;
    uint32_t element_count = 0;
};

// Lightweight section 7 expression record for CPU fallback evaluation.
struct ParsedS7Expr {
    uint8_t kind = 0;         // expression kind (1=literal, 3=param ref, 7=binary, 14=callable call)
    uint32_t a = UINT32_MAX;  // left child / index
    uint32_t b = UINT32_MAX;  // right child
    uint32_t c = UINT32_MAX;  // extra data
    uint32_t op = 0;          // operator / arg count
    uint32_t name_id = UINT32_MAX; // string table id for name/literal
    uint32_t first_argument = UINT32_MAX;
    uint32_t argument_count = 0;
};

struct ParsedS7Stmt {
    uint8_t kind = 0;         // statement kind (1=block, 11=return)
    uint32_t a = UINT32_MAX;  // body block / return expr
    uint32_t b = UINT32_MAX;  // extra data
    uint32_t c = UINT32_MAX;  // extra data
    uint32_t op = 0;          // return type id (for function records)
    uint32_t name_id = UINT32_MAX; // string table id
    uint32_t first_child = UINT32_MAX;
    uint32_t child_count = 0;
};

struct ParsedS7Function {
    uint8_t kind = 0;
    uint32_t name_id = UINT32_MAX;
    uint32_t mangled_name_id = UINT32_MAX;
    uint32_t return_type_id = UINT32_MAX;
    uint32_t first_parameter = UINT32_MAX;
    uint32_t parameter_count = 0;
    uint32_t body_statement_index = UINT32_MAX;
};

struct ParsedS7Param {
    uint8_t direction = 0;
    uint32_t name_id = UINT32_MAX;
    uint32_t type_id = UINT32_MAX;
};

struct ParsedS7Callable {
    std::string name;
    uint32_t function_index = UINT32_MAX;
};

struct ParsedIr {
    uint8_t shader_kind = 0;
    int32_t group_x = 1;
    int32_t group_y = 1;
    int32_t group_z = 1;
    std::vector<IrResource> resources;
    std::vector<IrInstruction> instructions;
    std::vector<IrElementwiseAssignment> elementwise_assignments;
    std::vector<IrExpressionAssignment> expression_assignments;
    std::vector<IrExpressionNode> expression_nodes;
    std::vector<uint32_t> expression_argument_indices;
    std::vector<IrControlFlowExpression> control_flow_expressions;
    std::vector<IrExpressionNode> control_flow_nodes;
    std::vector<uint32_t> control_flow_argument_indices;
    std::vector<uint32_t> ad_parameter_bindings;
    std::vector<uint32_t> ad_loss_bindings;
    std::vector<IrAdAnnotation> ad_annotations;
    std::vector<IrCompoundAssignment> compound_assignments;
    std::vector<IrExpressionNode> compound_assignment_nodes;
    std::vector<uint32_t> compound_assignment_args;
    std::vector<IrLocalVariableDecl> local_variable_decls;
    std::vector<std::string> strings;

    // Section 7 typed IR data (for callable dispatch)
    bool has_section7 = false;
    Feather::TypedIR::Module typed_module;
    uint32_t s7_entry_function = UINT32_MAX;
    std::vector<ParsedS7Function> s7_functions;
    std::vector<ParsedS7Param> s7_parameters;
    std::vector<ParsedS7Stmt> s7_stmts;
    std::vector<ParsedS7Expr> s7_exprs;
    std::vector<uint32_t> s7_children;
    std::vector<uint32_t> s7_arguments;
    std::vector<std::string> s7_strings;
    std::unordered_map<std::string, ParsedS7Callable> s7_callables; // mangled name → callable info
};

bool typed_ir_contains_unsupported_ad_control_flow(const Feather::TypedIR::Module& module, std::string* reason) {
    for (const auto& statement : module.statements) {
        switch (statement.kind) {
        case kTypedStatementWhile:
            if (reason != nullptr) {
                *reason = "while loops are not supported in differentiable kernels";
            }
            return true;
        case kTypedStatementDoWhile:
            if (reason != nullptr) {
                *reason = "do-while loops are not supported in differentiable kernels";
            }
            return true;
        case kTypedStatementBreak:
            if (reason != nullptr) {
                *reason = "break statements are not supported in differentiable kernels";
            }
            return true;
        case kTypedStatementContinue:
            if (reason != nullptr) {
                *reason = "continue statements are not supported in differentiable kernels";
            }
            return true;
        default:
            break;
        }
    }

    return false;
}

enum class FallbackExpressionKind { Copy, BufferBinaryBuffer, BufferBinaryLiteral };

struct FallbackAssignment {
    uint32_t destination_binding = UINT32_MAX;
    uint32_t left_binding = UINT32_MAX;
    uint32_t right_binding = UINT32_MAX;
    double literal_value = 0.0;
    char operation = 0;
    FallbackExpressionKind kind = FallbackExpressionKind::Copy;
};

struct GraphicsPipelineState {
    std::vector<unsigned char> ir;
    std::vector<unsigned char> vertex_ir;
    std::vector<unsigned char> fragment_ir;
    std::vector<unsigned char> push_constants;
    FeBufferHandle vertex_buffer = 0;
    FeBufferHandle index_buffer = 0;
    uint32_t vertex_stride = 0;
    uint32_t topology = 0;
    uint32_t sample_count = 1;
    uint32_t color_attachment_count = 1;
    uint32_t depth_test = 0;
    uint32_t depth_write = 0;
    uint32_t depth_compare = 1;
    uint32_t stencil_test = 0;
    FeGraphicsStencilFaceDesc stencil_front{};
    FeGraphicsStencilFaceDesc stencil_back{};
    uint32_t stencil_read_mask = UINT32_MAX;
    uint32_t stencil_write_mask = UINT32_MAX;
    uint32_t stencil_reference = 0;
    uint32_t blend_enable = 0;
    uint32_t blend_src_color = 1;
    uint32_t blend_dst_color = 0;
    uint32_t blend_color_op = 0;
    uint32_t blend_src_alpha = 1;
    uint32_t blend_dst_alpha = 0;
    uint32_t blend_alpha_op = 0;
    uint32_t blend_write_mask = 15;
    uint32_t color_blend_attachment_count = 0;
    std::array<FeGraphicsColorBlendAttachmentDesc, GPU::Backend::MAX_COLOR_ATTACHMENTS> color_blend_attachments{};
    uint32_t cull_mode = 0;
    uint32_t front_face = 0;
    uint32_t polygon_mode = 0;
    uint32_t depth_clamp = 0;
    std::unordered_map<uint32_t, FeBufferHandle> buffers;
    std::unordered_map<uint32_t, FeTextureHandle> textures;
    std::unordered_map<uint32_t, FeSamplerHandle> samplers;
    std::string debug_name;
    FeDispatchPath last_dispatch_path = FE_DISPATCH_PATH_NONE;
    bool graphics_lowered = false;
    std::string vertex_shader_source;
    std::string fragment_shader_source;
    GraphicsResourceLayout resource_layout;
    std::vector<GraphicsPushConstantLayoutEntry> push_constant_layout;
    std::vector<GraphicsPipelineCacheEntry> backend_cache;
};

#if FEATHER_BUILD_WINDOW
struct WindowState {
    std::unique_ptr<GPU::Window::AppWindow> window;
};

struct TexturePresenterState {
    FeWindowHandle window_handle = 0;
    std::unique_ptr<GPU::Window::TexturePresenter> presenter;
};
#endif

struct ProfilerRecord {
    std::string name;
    double elapsed_ms = 0.0;
    uint32_t group_x = 1;
    uint32_t group_y = 1;
    uint32_t group_z = 1;
};

struct ProfilerStats {
    uint64_t count = 0;
    double min_time_ms = 0.0;
    double max_time_ms = 0.0;
    double total_time_ms = 0.0;
};

std::mutex g_mutex;
std::atomic<uint64_t> g_next_handle{100};
std::atomic<bool> g_runtime_shutting_down{false};
std::unordered_map<FeBufferHandle, BufferState> g_buffers;
std::unordered_map<FeTextureHandle, TextureState> g_textures;
std::unordered_map<FeSamplerHandle, SamplerState> g_samplers;
std::unordered_map<FeKernelHandle, KernelState> g_kernels;
std::unordered_map<FeKernelHandle, ComputeKernelCache> g_compute_kernel_caches;
std::unordered_map<FeGraphicsPipelineHandle, GraphicsPipelineState> g_pipelines;
#if FEATHER_BUILD_WINDOW
std::unordered_map<FeWindowHandle, WindowState> g_windows;
std::unordered_map<FeTexturePresenterHandle, TexturePresenterState> g_texture_presenters;
#endif
bool g_profiler_enabled = false;
std::vector<ProfilerRecord> g_profiler_records;
std::unordered_map<std::string, ProfilerStats> g_profiler_stats;
thread_local std::string g_last_error;
thread_local FeResult g_last_result = FE_OK;

class EasyGpuShaderGuard {
  public:
    EasyGpuShaderGuard(GPU::Backend::Backend& backend, GPU::Backend::ShaderHandle shader)
        : backend_(backend), shader_(shader) {
    }

    ~EasyGpuShaderGuard() {
        if (shader_ != GPU::Backend::INVALID_SHADER_HANDLE) {
            backend_.DestroyShader(shader_);
        }
    }

    EasyGpuShaderGuard(const EasyGpuShaderGuard&) = delete;
    EasyGpuShaderGuard& operator=(const EasyGpuShaderGuard&) = delete;

  private:
    GPU::Backend::Backend& backend_;
    GPU::Backend::ShaderHandle shader_;
};

class EasyGpuPipelineGuard {
  public:
    EasyGpuPipelineGuard(GPU::Backend::Backend& backend, GPU::Backend::PipelineHandle pipeline)
        : backend_(backend), pipeline_(pipeline) {
    }

    ~EasyGpuPipelineGuard() {
        if (pipeline_ != GPU::Backend::INVALID_PIPELINE_HANDLE) {
            backend_.DestroyPipeline(pipeline_);
        }
    }

    EasyGpuPipelineGuard(const EasyGpuPipelineGuard&) = delete;
    EasyGpuPipelineGuard& operator=(const EasyGpuPipelineGuard&) = delete;

  private:
    GPU::Backend::Backend& backend_;
    GPU::Backend::PipelineHandle pipeline_;
};

class EasyGpuBufferGuard {
  public:
    EasyGpuBufferGuard(GPU::Backend::Backend& backend, GPU::Backend::BufferHandle buffer)
        : backend_(backend), buffer_(buffer) {
    }

    ~EasyGpuBufferGuard() {
        if (buffer_ != GPU::Backend::INVALID_BUFFER_HANDLE) {
            backend_.DestroyBuffer(buffer_);
        }
    }

    EasyGpuBufferGuard(const EasyGpuBufferGuard&) = delete;
    EasyGpuBufferGuard& operator=(const EasyGpuBufferGuard&) = delete;

    GPU::Backend::BufferHandle get() const {
        return buffer_;
    }

  private:
    GPU::Backend::Backend& backend_;
    GPU::Backend::BufferHandle buffer_;
};

void reset_compute_kernel_cache(ComputeKernelCache& cache, bool abandon_backend_resources) {
    if (abandon_backend_resources && cache.context != nullptr) {
        cache.context->SetCachedPipeline(GPU::Backend::INVALID_PIPELINE_HANDLE);
    }

    cache.context.reset();
}

void erase_compute_kernel_cache(FeKernelHandle kernel, bool abandon_backend_resources = false) {
    auto it = g_compute_kernel_caches.find(kernel);
    if (it == g_compute_kernel_caches.end()) {
        return;
    }

    reset_compute_kernel_cache(it->second, abandon_backend_resources);
    g_compute_kernel_caches.erase(it);
}

void clear_compute_kernel_caches(bool abandon_backend_resources = false) {
    for (auto& [handle, cache] : g_compute_kernel_caches) {
        (void)handle;
        reset_compute_kernel_cache(cache, abandon_backend_resources);
    }

    g_compute_kernel_caches.clear();
}

void destroy_graphics_pipeline_cache(GraphicsPipelineState& pipeline) {
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        pipeline.backend_cache.clear();
        return;
    }

    for (auto& entry : pipeline.backend_cache) {
        if (entry.pipeline != GPU::Backend::INVALID_PIPELINE_HANDLE) {
            backend->DestroyPipeline(entry.pipeline);
            entry.pipeline = GPU::Backend::INVALID_PIPELINE_HANDLE;
        }
        if (entry.vertex_shader != GPU::Backend::INVALID_SHADER_HANDLE) {
            backend->DestroyShader(entry.vertex_shader);
            entry.vertex_shader = GPU::Backend::INVALID_SHADER_HANDLE;
        }
        if (entry.fragment_shader != GPU::Backend::INVALID_SHADER_HANDLE) {
            backend->DestroyShader(entry.fragment_shader);
            entry.fragment_shader = GPU::Backend::INVALID_SHADER_HANDLE;
        }
    }
    pipeline.backend_cache.clear();
}

void release_ad_gradient_buffers_with_backend(KernelState& kernel, GPU::Backend::Backend* backend) {
    if (backend != nullptr) {
        std::unordered_set<GPU::Backend::BufferHandle> released;
        for (auto& gradient : kernel.ad_gradients) {
            if (gradient.backend_buffer != GPU::Backend::INVALID_BUFFER_HANDLE &&
                released.insert(gradient.backend_buffer).second) {
                backend->DestroyBuffer(gradient.backend_buffer);
            }
            gradient.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
        }

        if (kernel.ad_adj_pool != GPU::Backend::INVALID_BUFFER_HANDLE) {
            backend->DestroyBuffer(kernel.ad_adj_pool);
            kernel.ad_adj_pool = GPU::Backend::INVALID_BUFFER_HANDLE;
        }
    } else {
        for (auto& gradient : kernel.ad_gradients) {
            gradient.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
        }
        kernel.ad_adj_pool = GPU::Backend::INVALID_BUFFER_HANDLE;
    }

    kernel.ad_gradients.clear();
    kernel.ad_adj_pool_size = 0;
}

void destroy_backend_resources_for_shutdown() {
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend != nullptr) {
        try {
            backend->Finish();
        } catch (...) {
        }
    }

#if FEATHER_BUILD_WINDOW
    g_texture_presenters.clear();
    g_windows.clear();
#endif

    clear_compute_kernel_caches();

    for (auto& [handle, pipeline] : g_pipelines) {
        (void)handle;
        try {
            destroy_graphics_pipeline_cache(pipeline);
        } catch (...) {
            pipeline.backend_cache.clear();
        }
    }
    g_pipelines.clear();

    for (auto& [handle, kernel] : g_kernels) {
        (void)handle;
        try {
            release_ad_gradient_buffers_with_backend(kernel, backend);
        } catch (...) {
            release_ad_gradient_buffers_with_backend(kernel, nullptr);
        }
    }
    g_kernels.clear();

    if (backend != nullptr) {
        for (auto& [handle, texture] : g_textures) {
            (void)handle;
            if (texture.backend_texture != GPU::Backend::INVALID_TEXTURE_HANDLE) {
                try {
                    backend->DestroyTexture(texture.backend_texture);
                } catch (...) {
                }
                texture.backend_texture = GPU::Backend::INVALID_TEXTURE_HANDLE;
            }
        }

        for (auto& [handle, buffer] : g_buffers) {
            (void)handle;
            if (buffer.backend_buffer != GPU::Backend::INVALID_BUFFER_HANDLE) {
                try {
                    backend->DestroyBuffer(buffer.backend_buffer);
                } catch (...) {
                }
                buffer.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
            }
        }
    }

    g_textures.clear();
    g_buffers.clear();
    g_samplers.clear();
    g_profiler_records.clear();
    g_profiler_stats.clear();

    try {
        GPU::Runtime::Context::GetInstance().ShutdownBackend();
    } catch (...) {
    }
}

void abandon_native_resources_for_process_exit() {
#if FEATHER_BUILD_WINDOW
    for (auto& [handle, presenter] : g_texture_presenters) {
        (void)handle;
        (void)presenter.presenter.release();
    }
    g_texture_presenters.clear();

    for (auto& [handle, window] : g_windows) {
        (void)handle;
        (void)window.window.release();
    }
    g_windows.clear();
#endif

    clear_compute_kernel_caches(/*abandon_backend_resources*/ true);

    for (auto& [handle, pipeline] : g_pipelines) {
        (void)handle;
        for (auto& entry : pipeline.backend_cache) {
            entry.pipeline = GPU::Backend::INVALID_PIPELINE_HANDLE;
            entry.vertex_shader = GPU::Backend::INVALID_SHADER_HANDLE;
            entry.fragment_shader = GPU::Backend::INVALID_SHADER_HANDLE;
        }
        pipeline.backend_cache.clear();
    }
    g_pipelines.clear();

    for (auto& [handle, kernel] : g_kernels) {
        (void)handle;
        for (auto& gradient : kernel.ad_gradients) {
            gradient.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
        }
        kernel.ad_gradients.clear();
        kernel.ad_adj_pool = GPU::Backend::INVALID_BUFFER_HANDLE;
        kernel.ad_adj_pool_size = 0;
    }
    g_kernels.clear();

    for (auto& [handle, texture] : g_textures) {
        (void)handle;
        texture.backend_texture = GPU::Backend::INVALID_TEXTURE_HANDLE;
    }
    g_textures.clear();

    for (auto& [handle, buffer] : g_buffers) {
        (void)handle;
        buffer.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
    }
    g_buffers.clear();

    g_samplers.clear();
    g_profiler_records.clear();
    g_profiler_stats.clear();

    GPU::Runtime::Context::GetInstance().AbandonBackendForProcessExit();
}

uint64_t next_handle() {
    return g_next_handle.fetch_add(1, std::memory_order_relaxed);
}

FeResult fail(FeResult result, const char* message) {
    g_last_result = result;
    g_last_error = message == nullptr ? "Feather native call failed." : message;
    return result;
}

FeResult fail(FeResult result, const std::string& message) {
    g_last_result = result;
    g_last_error = message.empty() ? "Feather native call failed." : message;
    return result;
}

FeResult ok() {
    g_last_result = FE_OK;
    g_last_error.clear();
    return FE_OK;
}

std::string glsl_excerpt(const std::string& source, size_t first_line, size_t last_line) {
    std::ostringstream output;
    std::istringstream input(source);
    std::string line;
    size_t line_number = 1;
    while (std::getline(input, line)) {
        if (line_number >= first_line && line_number <= last_line) {
            output << std::setw(4) << line_number << ": " << line << '\n';
        }
        if (line_number > last_line) {
            break;
        }
        line_number++;
    }

    return output.str();
}

size_t glsl_error_line(const std::string& message) {
    const std::string prefix = "ERROR: 0:";
    const auto start = message.find(prefix);
    if (start == std::string::npos) {
        return 0;
    }

    const auto number_start = start + prefix.size();
    const auto number_end = message.find(':', number_start);
    if (number_end == std::string::npos || number_end <= number_start) {
        return 0;
    }

    try {
        return static_cast<size_t>(std::stoull(message.substr(number_start, number_end - number_start)));
    } catch (...) {
        return 0;
    }
}

std::string glsl_error_excerpt(const std::string& source, const std::string& message) {
    const auto line = glsl_error_line(message);
    if (line == 0) {
        return glsl_excerpt(source, 1, 120);
    }

    const auto first = line > 35 ? line - 35 : 1;
    return glsl_excerpt(source, first, line + 35);
}

uint32_t map_backend_type(GPU::Backend::BackendType type) {
    switch (type) {
    case GPU::Backend::BackendType::OpenGL:
        return 1;
    case GPU::Backend::BackendType::Vulkan:
        return 2;
    default:
        return 0;
    }
}

GPU::Backend::Backend* require_backend() {
    GPU::Runtime::AutoInitContext();
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        throw std::runtime_error("EasyGPU backend is unavailable.");
    }

    return backend;
}

std::string copy_debug_name(const char* debug_name, const char* fallback) {
    if (debug_name != nullptr && debug_name[0] != '\0') {
        return debug_name;
    }

    return fallback;
}

uint16_t read_u16(const unsigned char* data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t read_u32(const unsigned char* data) {
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) | (static_cast<uint32_t>(data[3]) << 24);
}

int32_t read_i32(const unsigned char* data) {
    return static_cast<int32_t>(read_u32(data));
}

bool checked_add_size(size_t a, size_t b, size_t* result) {
    if (SIZE_MAX - a < b) {
        return false;
    }

    *result = a + b;
    return true;
}

void adopt_typed_ir_module(const Feather::TypedIR::Module& source, ParsedIr* parsed) {
    parsed->has_section7 = true;
    parsed->typed_module = source;
    parsed->s7_entry_function = source.entry_function;

    parsed->s7_functions.clear();
    parsed->s7_functions.reserve(source.functions.size());
    for (const auto& function : source.functions) {
        ParsedS7Function target;
        target.kind = function.kind;
        target.name_id = function.name_id;
        target.mangled_name_id = function.mangled_name_id;
        target.return_type_id = function.return_type_id;
        target.first_parameter = function.first_parameter;
        target.parameter_count = function.parameter_count;
        target.body_statement_index = function.body_statement_index;
        parsed->s7_functions.push_back(target);
    }

    parsed->s7_parameters.clear();
    parsed->s7_parameters.reserve(source.parameters.size());
    for (const auto& parameter : source.parameters) {
        parsed->s7_parameters.push_back(ParsedS7Param{parameter.direction, parameter.name_id, parameter.type_id});
    }

    parsed->s7_stmts.clear();
    parsed->s7_stmts.reserve(source.statements.size());
    for (const auto& statement : source.statements) {
        ParsedS7Stmt target;
        target.kind = statement.kind;
        target.a = statement.a;
        target.b = statement.b;
        target.c = statement.c;
        target.op = statement.op;
        target.name_id = statement.name_id;
        target.first_child = statement.first_child;
        target.child_count = statement.child_count;
        parsed->s7_stmts.push_back(target);
    }

    parsed->s7_exprs.clear();
    parsed->s7_exprs.reserve(source.expressions.size());
    for (const auto& expression : source.expressions) {
        ParsedS7Expr target;
        target.kind = expression.kind;
        target.a = expression.a;
        target.b = expression.b;
        target.c = expression.c;
        target.op = expression.op;
        target.name_id = expression.name_id;
        target.first_argument = expression.first_argument;
        target.argument_count = expression.argument_count;
        parsed->s7_exprs.push_back(target);
    }

    parsed->s7_children = source.children;
    parsed->s7_arguments = source.arguments;
    parsed->s7_strings = source.strings;

    parsed->s7_callables.clear();
    for (const auto& [name, callable] : source.callables) {
        parsed->s7_callables[name] = ParsedS7Callable{callable.name, callable.function_index};
    }
}

size_t pixel_size(uint32_t format) {
    switch (format) {
    case 1:
        return 1;
    case 2:
        return 2;
    case 3:
    case 4:
        return 4;
    case 5:
        return 2;
    case 6:
        return 4;
    case 7:
        return 8;
    case 8:
        return 4;
    case 9:
        return 8;
    case 10:
        return 16;
    case 100:
        return 4;
    case 101:
        return 4;
    default:
        return 4;
    }
}

const char* pixel_format_name(uint32_t format) {
    switch (format) {
    case 1:
        return "R8";
    case 2:
        return "Rg8";
    case 3:
        return "Rgba8";
    case 4:
        return "Bgra8";
    case 5:
        return "R16Float";
    case 6:
        return "Rg16Float";
    case 7:
        return "Rgba16Float";
    case 8:
        return "R32Float";
    case 9:
        return "Rg32Float";
    case 10:
        return "Rgba32Float";
    case 100:
        return "Depth24Stencil8";
    case 101:
        return "Depth32Float";
    default:
        return "Unknown";
    }
}

bool easygpu_runtime_pixel_format(uint32_t format, GPU::Runtime::PixelFormat* out_format) {
    if (out_format == nullptr) {
        return false;
    }

    switch (format) {
    case 1:
        *out_format = GPU::Runtime::PixelFormat::R8;
        return true;
    case 2:
        *out_format = GPU::Runtime::PixelFormat::RG8;
        return true;
    case 3:
        *out_format = GPU::Runtime::PixelFormat::RGBA8;
        return true;
    case 5:
        *out_format = GPU::Runtime::PixelFormat::R16F;
        return true;
    case 6:
        *out_format = GPU::Runtime::PixelFormat::RG16F;
        return true;
    case 7:
        *out_format = GPU::Runtime::PixelFormat::RGBA16F;
        return true;
    case 8:
        *out_format = GPU::Runtime::PixelFormat::R32F;
        return true;
    case 9:
        *out_format = GPU::Runtime::PixelFormat::RG32F;
        return true;
    case 10:
        *out_format = GPU::Runtime::PixelFormat::RGBA32F;
        return true;
    default:
        return false;
    }
}

bool easygpu_backend_pixel_format(uint32_t format, GPU::Backend::PixelFormat* out_format) {
    if (out_format == nullptr) {
        return false;
    }

    switch (format) {
    case 100:
        *out_format = GPU::Backend::PixelFormat::D24S8;
        return true;
    case 101:
        *out_format = GPU::Backend::PixelFormat::D32F;
        return true;
    default: {
        GPU::Runtime::PixelFormat runtime_format = GPU::Runtime::PixelFormat::RGBA8;
        if (!easygpu_runtime_pixel_format(format, &runtime_format)) {
            return false;
        }
        *out_format = GPU::Runtime::ToBackendPixelFormat(runtime_format);
        return true;
    }
    }
}

bool validate_instruction_structure(const std::vector<IrInstruction>& instructions) {
    std::vector<uint8_t> block_stack;
    uint8_t pending_block_kind = kIrBlockKindGeneric;
    bool may_start_else = false;
    for (const auto& instruction : instructions) {
        if (instruction.opcode == 0 || instruction.opcode > kIrOpcodeSharedMemoryDeclaration ||
            instruction.operand_kind > kIrOperandKindSymbol) {
            return false;
        }

        if (instruction.opcode == kIrOpcodeIf) {
            pending_block_kind = kIrBlockKindIfTrue;
            may_start_else = false;
            continue;
        }

        // Validate legacy instruction block structure for compatibility payloads.
        // Canonical generated compute semantics are carried by section 7.
        if (instruction.opcode == kIrOpcodeBeginBlock) {
            block_stack.push_back(pending_block_kind);
            pending_block_kind = kIrBlockKindGeneric;
            may_start_else = false;
            continue;
        }

        if (instruction.opcode == kIrOpcodeEndBlock) {
            if (block_stack.empty()) {
                return false;
            }

            const auto ended_block = block_stack.back();
            block_stack.pop_back();
            may_start_else = ended_block == kIrBlockKindIfTrue;
            continue;
        }

        if (instruction.opcode == kIrOpcodeElse) {
            if (!may_start_else) {
                return false;
            }

            pending_block_kind = kIrBlockKindIfElse;
            may_start_else = false;
            continue;
        }

        pending_block_kind = kIrBlockKindGeneric;
        may_start_else = false;
    }

    return block_stack.empty();
}

bool parse_feather_ir(const std::vector<unsigned char>& ir, ParsedIr* parsed) {
    if (parsed == nullptr || ir.size() < 44 || std::memcmp(ir.data(), "FEIR", 4) != 0) {
        return false;
    }

    const auto major = read_u16(ir.data() + 4);
    const auto minor = read_u16(ir.data() + 6);
    const auto endian = ir[8];
    if (major != 1 || endian != 1) {
        return false;
    }

    // IR minor version 1 uses the reserved header slot at byte 10 as a section count.
    // Legacy compatibility payloads can still carry ASSIGN1 data, but section validation
    // keeps the typed Roslyn-to-EasyGPU bridge contract explicit.
    const auto section_count = read_u16(ir.data() + 10);
    parsed->shader_kind = ir[9];
    parsed->group_x = read_i32(ir.data() + 12);
    parsed->group_y = read_i32(ir.data() + 16);
    parsed->group_z = read_i32(ir.data() + 20);
    const auto resource_count = read_u32(ir.data() + 28);
    const auto instruction_count = read_u32(ir.data() + 36);
    const auto string_byte_length = read_u32(ir.data() + 40);

    size_t offset = 44;
    parsed->resources.clear();
    parsed->instructions.clear();
    parsed->elementwise_assignments.clear();
    parsed->expression_assignments.clear();
    parsed->expression_nodes.clear();
    parsed->strings.clear();
    parsed->resources.reserve(resource_count);
    parsed->instructions.reserve(instruction_count);

    for (uint32_t i = 0; i < resource_count; ++i) {
        size_t next = 0;
        if (!checked_add_size(offset, 15, &next) || next > ir.size()) {
            return false;
        }

        IrResource resource;
        resource.binding = read_u32(ir.data() + offset);
        resource.kind = ir[offset + 4];
        resource.access = ir[offset + 5];
        resource.name_string_id = read_u32(ir.data() + offset + 7);
        resource.element_type_string_id = read_u32(ir.data() + offset + 11);
        parsed->resources.push_back(resource);
        offset = next;
    }

    for (uint32_t i = 0; i < instruction_count; ++i) {
        size_t next = 0;
        if (!checked_add_size(offset, 8, &next) || next > ir.size()) {
            return false;
        }

        IrInstruction instruction;
        instruction.opcode = ir[offset];
        instruction.operand_kind = ir[offset + 1];
        instruction.operand_string_id = read_u32(ir.data() + offset + 4);
        parsed->instructions.push_back(instruction);
        offset = next;
    }

    if (!validate_instruction_structure(parsed->instructions)) {
        return false;
    }

    std::vector<uint32_t> section_kinds;
    std::vector<uint32_t> section_lengths;
    section_kinds.reserve(section_count);
    section_lengths.reserve(section_count);
    if (section_count > 0 && minor == 0) {
        return false;
    }

    for (uint32_t i = 0; i < section_count; ++i) {
        size_t next = 0;
        if (!checked_add_size(offset, kIrSectionRecordSize, &next) || next > ir.size()) {
            return false;
        }

        const auto kind = read_u32(ir.data() + offset);
        const auto byte_length = read_u32(ir.data() + offset + 4);
        if (kind != kIrSectionElementwiseAssignments && kind != kIrSectionElementwiseExpressionAssignments &&
            kind != kIrSectionControlFlowExpressions && kind != kIrSectionAdAnnotations &&
            kind != kIrSectionLocalVariables && kind != kIrSectionCompoundAssignments &&
            kind != 7 /* kIrSectionTypedShaderIr */) {
            return false;
        }

        const auto minimum_length = kind == kIrSectionElementwiseAssignments ? kIrAssignmentHeaderSize
                                  : kind == 7 /* kIrSectionTypedShaderIr */ ? kTypedIrHeaderSize
                                                                            : kIrExpressionAssignmentHeaderSize;
        if (byte_length < minimum_length) {
            return false;
        }

        section_kinds.push_back(kind);
        section_lengths.push_back(byte_length);
        offset = next;
    }

    for (uint32_t i = 0; i < section_count; ++i) {
        size_t next = 0;
        if (!checked_add_size(offset, section_lengths[i], &next) || next > ir.size()) {
            return false;
        }

        const auto* payload = ir.data() + offset;
        // Section 3: Control flow expressions — maps instruction indices to expression root nodes for if/for/while/do conditions.
        if (section_kinds[i] == kIrSectionControlFlowExpressions) {
            const auto record_count = read_u32(payload);
            const auto node_count = read_u32(payload + 4);
            const auto argument_index_count = read_u32(payload + 8);
            constexpr uint64_t kCfRecordSize = 12;
            constexpr uint64_t kCfNodeRecordSize = 40;
            size_t records_size = 0;
            size_t nodes_size = 0;
            size_t args_size = 0;
            if (!checked_add_size(0, static_cast<size_t>(record_count) * kCfRecordSize, &records_size) ||
                !checked_add_size(0, static_cast<size_t>(node_count) * kCfNodeRecordSize, &nodes_size) ||
                !checked_add_size(0, static_cast<size_t>(argument_index_count) * sizeof(uint32_t), &args_size)) {
                return false;
            }
            size_t expected = 12;
            if (!checked_add_size(expected, records_size, &expected) || !checked_add_size(expected, nodes_size, &expected) ||
                !checked_add_size(expected, args_size, &expected) || expected != section_lengths[i]) {
                return false;
            }
            const auto* records = payload + 12;
            const auto* nodes = records + records_size;
            const auto* args = nodes + nodes_size;
            for (uint32_t rec = 0; rec < record_count; ++rec) {
                const auto* record = records + (static_cast<uint64_t>(rec) * kCfRecordSize);
                const auto cf_instr = read_u32(record);
                const auto cf_role = record[4];
                const auto cf_root = read_u32(record + 8);
                if (cf_instr >= instruction_count || cf_role == 0 || cf_role > kCfRoleDoCondition ||
                    cf_root >= node_count) {
                    return false;
                }
                IrControlFlowExpression cf;
                cf.instruction_index = cf_instr;
                cf.role = cf_role;
                cf.root_node_index = cf_root;
                parsed->control_flow_expressions.push_back(cf);
            }
            for (uint32_t node = 0; node < node_count; ++node) {
                const auto* record = nodes + (static_cast<uint64_t>(node) * kCfNodeRecordSize);
                const auto node_kind = record[0];
                const auto operation = record[1];
                const auto left = read_u32(record + 20);
                const auto right = read_u32(record + 24);
                const auto symbol = read_u32(record + 28);
                const auto first_argument = read_u32(record + 32);
                const auto argument_count = read_u32(record + 36);
                if (node_kind == 0 || node_kind > kIrExpressionNodeKindMax || operation > 10 ||
                    (left != UINT32_MAX && left >= node_count) || (right != UINT32_MAX && right >= node_count) ||
                    (argument_count > 0 && (first_argument == UINT32_MAX || first_argument > argument_index_count ||
                     argument_count > argument_index_count - first_argument))) {
                    return false;
                }
                IrExpressionNode cf_node;
                cf_node.kind = node_kind;
                cf_node.operation = operation;
                cf_node.resource_binding = read_u32(record + 4);
                cf_node.index_string_id = read_u32(record + 8);
                cf_node.literal_string_id = read_u32(record + 12);
                cf_node.type_string_id = read_u32(record + 16);
                cf_node.left_node_index = left;
                cf_node.right_node_index = right;
                cf_node.symbol_string_id = symbol;
                cf_node.first_argument_index = first_argument;
                cf_node.argument_count = argument_count;
                parsed->control_flow_nodes.push_back(cf_node);
            }
            for (uint32_t arg = 0; arg < argument_index_count; ++arg) {
                const auto node_index = read_u32(args + (static_cast<uint64_t>(arg) * sizeof(uint32_t)));
                if (node_index >= node_count) return false;
                parsed->control_flow_argument_indices.push_back(node_index);
            }
            offset = next;
            continue;
        }
        if (section_kinds[i] == kIrSectionAdAnnotations) {
            if (section_lengths[i] < 8) return false;
            const auto maybe_version = read_u16(payload);
            if (maybe_version == 2) {
                if (section_lengths[i] < 12) return false;
                const auto param_count = read_u32(payload + 4);
                const auto loss_count = read_u32(payload + 8);
                constexpr uint64_t kAdRecordSize = 32;
                size_t records_size = 0;
                size_t expected = 12;
                if (!checked_add_size(0, static_cast<size_t>(param_count + loss_count) * kAdRecordSize, &records_size) ||
                    !checked_add_size(expected, records_size, &expected) ||
                    expected != section_lengths[i]) {
                    return false;
                }

                const auto* records = payload + 12;
                for (uint32_t record_index = 0; record_index < param_count + loss_count; ++record_index) {
                    const auto* record = records + (static_cast<uint64_t>(record_index) * kAdRecordSize);
                    IrAdAnnotation annotation;
                    annotation.role = read_u32(record);
                    annotation.binding = read_u32(record + 4);
                    annotation.name_string_id = read_u32(record + 8);
                    annotation.resource_name_string_id = read_u32(record + 12);
                    annotation.type_name_string_id = read_u32(record + 16);
                    annotation.index_string_id = read_u32(record + 20);
                    annotation.source_kind = read_u32(record + 24);
                    annotation.element_count = read_u32(record + 28);
                    if ((record_index < param_count && annotation.role != kIrAdRoleParameter) ||
                        (record_index >= param_count && annotation.role != kIrAdRoleLoss)) {
                        return false;
                    }
                    if (annotation.role == kIrAdRoleParameter) {
                        parsed->ad_parameter_bindings.push_back(annotation.binding);
                    } else if (annotation.role == kIrAdRoleLoss) {
                        parsed->ad_loss_bindings.push_back(annotation.binding);
                    } else {
                        return false;
                    }
                    parsed->ad_annotations.push_back(annotation);
                }
            } else {
                const auto param_count = read_u32(payload);
                const auto loss_count = read_u32(payload + 4);
                constexpr uint64_t kAdRecordSize = 4;
                size_t expected = 8;
                size_t params_size = 0;
                size_t losses_size = 0;
                if (!checked_add_size(0, static_cast<size_t>(param_count) * kAdRecordSize, &params_size) ||
                    !checked_add_size(0, static_cast<size_t>(loss_count) * kAdRecordSize, &losses_size) ||
                    !checked_add_size(expected, params_size, &expected) ||
                    !checked_add_size(expected, losses_size, &expected) ||
                    expected != section_lengths[i]) {
                    return false;
                }
                for (uint32_t p = 0; p < param_count; ++p) {
                    const auto binding = read_u32(payload + 8 + (static_cast<uint64_t>(p) * kAdRecordSize));
                    parsed->ad_parameter_bindings.push_back(binding);
                    IrAdAnnotation annotation;
                    annotation.role = kIrAdRoleParameter;
                    annotation.binding = binding;
                    parsed->ad_annotations.push_back(annotation);
                }
                for (uint32_t l = 0; l < loss_count; ++l) {
                    const auto binding = read_u32(payload + 8 + params_size + (static_cast<uint64_t>(l) * kAdRecordSize));
                    parsed->ad_loss_bindings.push_back(binding);
                    IrAdAnnotation annotation;
                    annotation.role = kIrAdRoleLoss;
                    annotation.binding = binding;
                    parsed->ad_annotations.push_back(annotation);
                }
            }
            offset = next;
            continue;
        }
        if (section_kinds[i] == kIrSectionLocalVariables) {
            if (section_lengths[i] < 4) return false;
            const auto decl_count = read_u32(payload);
            constexpr uint64_t kLocalVarRecordSize = 12;
            size_t records_size = 0;
            if (!checked_add_size(0, static_cast<size_t>(decl_count) * kLocalVarRecordSize, &records_size))
                return false;
            size_t expected = 4;
            if (!checked_add_size(expected, records_size, &expected) || expected != section_lengths[i])
                return false;
            for (uint32_t d = 0; d < decl_count; ++d) {
                const auto* rec = payload + 4 + (static_cast<uint64_t>(d) * kLocalVarRecordSize);
                IrLocalVariableDecl decl;
                decl.instruction_index = read_u32(rec);
                decl.name_string_id = read_u32(rec + 4);
                decl.glsl_text_string_id = read_u32(rec + 8);
                parsed->local_variable_decls.push_back(decl);
            }
            offset = next;
            continue;
        }
        if (section_kinds[i] == kIrSectionCompoundAssignments) {
            if (section_lengths[i] < 12) return false;
            const auto rec_count = read_u32(payload);
            const auto node_count = read_u32(payload + 4);
            const auto arg_count = read_u32(payload + 8);
            constexpr uint64_t kCaRecSize = 20;
            size_t recs = 0, nds = 0, ags = 0;
            if (!checked_add_size(0, static_cast<size_t>(rec_count) * kCaRecSize, &recs) ||
                !checked_add_size(0, static_cast<size_t>(node_count) * static_cast<size_t>(kIrExpressionNodeRecordWithArgumentsSize), &nds) ||
                !checked_add_size(0, static_cast<size_t>(arg_count) * sizeof(uint32_t), &ags))
                return false;
            size_t expected = 12;
            if (!checked_add_size(expected, recs, &expected) || !checked_add_size(expected, nds, &expected) ||
                !checked_add_size(expected, ags, &expected) || expected != section_lengths[i])
                return false;
            for (uint32_t r = 0; r < rec_count; ++r) {
                const auto* rec = payload + 12 + (static_cast<uint64_t>(r) * kCaRecSize);
                IrCompoundAssignment ca;
                ca.instruction_index = read_u32(rec);
                ca.destination_binding = read_u32(rec + 4);
                ca.index_string_id = read_u32(rec + 8);
                ca.operation = rec[12];
                ca.root_node_index = read_u32(rec + 16);
                parsed->compound_assignments.push_back(ca);
            }
            const auto* ca_nodes = payload + 12 + recs;
            for (uint32_t n = 0; n < node_count; ++n) {
                const auto* rec = ca_nodes + (static_cast<uint64_t>(n) * kIrExpressionNodeRecordWithArgumentsSize);
                IrExpressionNode node;
                node.kind = rec[0]; node.operation = rec[1];
                node.resource_binding = read_u32(rec + 4);
                node.index_string_id = read_u32(rec + 8);
                node.literal_string_id = read_u32(rec + 12);
                node.type_string_id = read_u32(rec + 16);
                node.left_node_index = read_u32(rec + 20);
                node.right_node_index = read_u32(rec + 24);
                node.symbol_string_id = read_u32(rec + 28);
                node.first_argument_index = read_u32(rec + 32);
                node.argument_count = read_u32(rec + 36);
                parsed->compound_assignment_nodes.push_back(node);
            }
            const auto* ca_args = ca_nodes + nds;
            for (uint32_t a = 0; a < arg_count; ++a)
                parsed->compound_assignment_args.push_back(read_u32(ca_args + (static_cast<uint64_t>(a) * sizeof(uint32_t))));
            offset = next;
            continue;
        }
        if (section_kinds[i] == kIrSectionElementwiseExpressionAssignments) {
            const auto assignment_count = read_u32(payload);
            const auto node_count = read_u32(payload + 4);
            size_t legacy_section_bytes = kIrExpressionAssignmentHeaderSize;
            size_t expression_section_bytes = kIrExpressionAssignmentHeaderWithArgumentsSize;
            size_t assignment_bytes = 0;
            size_t node_bytes = 0;
            size_t expression_node_bytes = 0;
            size_t argument_index_bytes = 0;
            if (!checked_add_size(0, static_cast<size_t>(assignment_count) * static_cast<size_t>(kIrExpressionAssignmentRecordSize), &assignment_bytes) ||
                !checked_add_size(0, static_cast<size_t>(node_count) * static_cast<size_t>(kIrExpressionNodeRecordSize), &node_bytes) ||
                !checked_add_size(legacy_section_bytes, assignment_bytes, &legacy_section_bytes) ||
                !checked_add_size(legacy_section_bytes, node_bytes, &legacy_section_bytes)) {
                return false;
            }

            const auto has_argument_table = legacy_section_bytes != section_lengths[i];
            uint32_t argument_index_count = 0;
            if (has_argument_table) {
                argument_index_count = read_u32(payload + 8);
                if (!checked_add_size(0, static_cast<size_t>(node_count) * static_cast<size_t>(kIrExpressionNodeRecordWithArgumentsSize), &expression_node_bytes) ||
                    !checked_add_size(0, static_cast<size_t>(argument_index_count) * sizeof(uint32_t), &argument_index_bytes) ||
                    !checked_add_size(expression_section_bytes, assignment_bytes, &expression_section_bytes) ||
                    !checked_add_size(expression_section_bytes, expression_node_bytes, &expression_section_bytes) ||
                    !checked_add_size(expression_section_bytes, argument_index_bytes, &expression_section_bytes) ||
                    expression_section_bytes != section_lengths[i]) {
                    return false;
                }
            }

            const auto header_size = has_argument_table ? kIrExpressionAssignmentHeaderWithArgumentsSize
                                                        : kIrExpressionAssignmentHeaderSize;
            const auto node_record_size = has_argument_table ? kIrExpressionNodeRecordWithArgumentsSize
                                                             : kIrExpressionNodeRecordSize;
            const auto* assignments = payload + header_size;
            const auto* nodes = assignments + assignment_bytes;
            for (uint32_t assignment = 0; assignment < assignment_count; ++assignment) {
                const auto* record = assignments + (static_cast<uint64_t>(assignment) * kIrExpressionAssignmentRecordSize);
                const auto instruction_index = read_u32(record);
                const auto root_node_index = read_u32(record + 12);
                if (instruction_index >= instruction_count || root_node_index >= node_count) {
                    return false;
                }

                IrExpressionAssignment parsed_assignment;
                parsed_assignment.instruction_index = instruction_index;
                parsed_assignment.destination_binding = read_u32(record + 4);
                parsed_assignment.index_string_id = read_u32(record + 8);
                parsed_assignment.root_node_index = root_node_index;
                parsed->expression_assignments.push_back(parsed_assignment);
            }

            for (uint32_t node = 0; node < node_count; ++node) {
                const auto* record = nodes + (static_cast<uint64_t>(node) * node_record_size);
                const auto node_kind = record[0];
                const auto operation = record[1];
                const auto left = read_u32(record + 20);
                const auto right = read_u32(record + 24);
                const auto symbol = has_argument_table ? read_u32(record + 28) : UINT32_MAX;
                const auto first_argument = has_argument_table ? read_u32(record + 32) : UINT32_MAX;
                const auto argument_count = has_argument_table ? read_u32(record + 36) : 0;
                if (node_kind == 0 || node_kind > kIrExpressionNodeKindMax || operation > 10 ||
                    (left != UINT32_MAX && left >= node_count) ||
                    (right != UINT32_MAX && right >= node_count) ||
                    (argument_count > 0 &&
                     (first_argument == UINT32_MAX || first_argument > argument_index_count ||
                      argument_count > argument_index_count - first_argument))) {
                    return false;
                }

                IrExpressionNode parsed_node;
                parsed_node.kind = node_kind;
                parsed_node.operation = operation;
                parsed_node.resource_binding = read_u32(record + 4);
                parsed_node.index_string_id = read_u32(record + 8);
                parsed_node.literal_string_id = read_u32(record + 12);
                parsed_node.type_string_id = read_u32(record + 16);
                parsed_node.left_node_index = left;
                parsed_node.right_node_index = right;
                parsed_node.symbol_string_id = symbol;
                parsed_node.first_argument_index = first_argument;
                parsed_node.argument_count = argument_count;
                parsed->expression_nodes.push_back(parsed_node);
            }

            const auto* argument_indices = nodes + (static_cast<uint64_t>(node_count) * node_record_size);
            for (uint32_t argument = 0; argument < argument_index_count; ++argument) {
                const auto node_index = read_u32(argument_indices + (static_cast<uint64_t>(argument) * sizeof(uint32_t)));
                if (node_index >= node_count) {
                    return false;
                }

                parsed->expression_argument_indices.push_back(node_index);
            }

            offset = next;
            continue;
        }
        if (section_kinds[i] == 7 /* kIrSectionTypedShaderIr */) {
            Feather::TypedIR::Module typed_module;
            if (!Feather::TypedIR::ParseSection(payload, section_lengths[i], &typed_module)) {
                return false;
            }

            adopt_typed_ir_module(typed_module, parsed);
            offset = next;
            continue;
        }

        const auto count = read_u32(payload);
        size_t record_bytes = 0;
        if (!checked_add_size(0, static_cast<size_t>(count) * static_cast<size_t>(kIrAssignmentRecordSize), &record_bytes) ||
            record_bytes + kIrAssignmentHeaderSize != section_lengths[i]) {
            return false;
        }

        for (uint32_t assignment = 0; assignment < count; ++assignment) {
            const auto* record = payload + kIrAssignmentHeaderSize + (static_cast<uint64_t>(assignment) * kIrAssignmentRecordSize);
            const auto instruction_index = read_u32(record);
            const auto destination_binding = read_u32(record + 4);
            const auto left_binding = read_u32(record + 8);
            const auto right_binding = read_u32(record + 12);
            const auto operation = record[16];
            const auto operand_kind = record[17];
            if (instruction_index >= instruction_count || operation == 0 || operation > 5 || operand_kind > 2) {
                return false;
            }

            IrElementwiseAssignment parsed_assignment;
            parsed_assignment.instruction_index = instruction_index;
            parsed_assignment.destination_binding = destination_binding;
            parsed_assignment.left_binding = left_binding;
            parsed_assignment.right_binding = right_binding;
            parsed_assignment.operation = operation;
            parsed_assignment.right_operand_kind = operand_kind;
            parsed_assignment.index_string_id = read_u32(record + 20);
            parsed_assignment.right_literal_string_id = read_u32(record + 24);
            parsed->elementwise_assignments.push_back(parsed_assignment);
        }

        offset = next;
    }

    size_t string_end = 0;
    if (!checked_add_size(offset, string_byte_length, &string_end) || string_end != ir.size() ||
        string_byte_length < 4) {
        return false;
    }

    const auto* string_data = ir.data() + offset;
    const auto string_count = read_u32(string_data);
    size_t string_offset = 4;
    parsed->strings.reserve(string_count);
    for (uint32_t i = 0; i < string_count; ++i) {
        size_t length_end = 0;
        if (!checked_add_size(string_offset, 4, &length_end) || length_end > string_byte_length) {
            return false;
        }

        const auto length = read_u32(string_data + string_offset);
        string_offset = length_end;
        size_t value_end = 0;
        if (!checked_add_size(string_offset, length, &value_end) || value_end > string_byte_length) {
            return false;
        }

        parsed->strings.emplace_back(reinterpret_cast<const char*>(string_data + string_offset), length);
        string_offset = value_end;
    }

    if (string_offset != string_byte_length) {
        return false;
    }

    return true;
}

// Evaluate a section 7 expression for CPU callable fallback dispatch.
// Returns false if the expression cannot be evaluated.
static bool evaluate_s7_expr(const ParsedIr& ir, uint32_t expr_index,
    const std::unordered_map<std::string, double>& param_bindings, double* result) {
    if (expr_index >= ir.s7_exprs.size() || result == nullptr) return false;
    const auto& e = ir.s7_exprs[expr_index];

    switch (e.kind) {
    case 1: { // Literal
        const auto* lit = e.name_id < ir.s7_strings.size() ? &ir.s7_strings[e.name_id] : nullptr;
        if (lit == nullptr) return false;
        // Parse float literal
        char* end = nullptr;
        *result = std::strtod(lit->c_str(), &end);
        return end != lit->c_str();
    }
    case 3: { // ParameterReference
        const auto* pname = e.name_id < ir.s7_strings.size() ? &ir.s7_strings[e.name_id] : nullptr;
        if (pname == nullptr) return false;
        auto it = param_bindings.find(*pname);
        if (it == param_bindings.end()) return false;
        *result = it->second;
        return true;
    }
    case 7: { // Binary
        const auto op = e.op; // ShaderBinaryOperator: 0=Add,1=Sub,2=Mul,3=Div
        double left = 0, right = 0;
        if (!evaluate_s7_expr(ir, e.a, param_bindings, &left) ||
            !evaluate_s7_expr(ir, e.b, param_bindings, &right))
            return false;
        switch (op) {
        case 0: *result = left + right; break; // Add
        case 1: *result = left - right; break; // Sub
        case 2: *result = left * right; break; // Mul
        case 3: if (right == 0.0) return false; *result = left / right; break; // Div
        default: return false;
        }
        return true;
    }
    case 14: { // CallableCall (nested)
        const auto* cname = e.name_id < ir.s7_strings.size() ? &ir.s7_strings[e.name_id] : nullptr;
        if (cname == nullptr) return false;
        auto cit = ir.s7_callables.find(*cname);
        if (cit == ir.s7_callables.end()) return false;
        if (cit->second.function_index >= ir.s7_functions.size()) return false;

        if (e.argument_count > 0 &&
            (e.first_argument == UINT32_MAX || e.first_argument > ir.s7_arguments.size() ||
             e.argument_count > ir.s7_arguments.size() - e.first_argument)) {
            return false;
        }

        std::vector<double> call_args;
        call_args.reserve(e.argument_count);
        for (uint32_t ai = 0; ai < e.argument_count; ++ai) {
            const auto arg_expr_idx = ir.s7_arguments[e.first_argument + ai];
            double aval = 0;
            if (!evaluate_s7_expr(ir, arg_expr_idx, param_bindings, &aval)) return false;
            call_args.push_back(aval);
        }

        const auto& callable_func = ir.s7_functions[cit->second.function_index];
        if (callable_func.parameter_count != call_args.size() ||
            (callable_func.parameter_count > 0 &&
             (callable_func.first_parameter == UINT32_MAX ||
              callable_func.first_parameter > ir.s7_parameters.size() ||
              callable_func.parameter_count > ir.s7_parameters.size() - callable_func.first_parameter))) {
            return false;
        }

        const auto body_block_idx = callable_func.body_statement_index;
        if (body_block_idx >= ir.s7_stmts.size()) return false;
        const auto& body_block = ir.s7_stmts[body_block_idx];
        if (body_block.kind != 1) return false;

        std::unordered_map<std::string, double> callable_bindings;
        for (uint32_t pi = 0; pi < callable_func.parameter_count; ++pi) {
            const auto& parameter = ir.s7_parameters[callable_func.first_parameter + pi];
            if (parameter.name_id >= ir.s7_strings.size()) return false;
            callable_bindings[ir.s7_strings[parameter.name_id]] = call_args[pi];
        }

        if (body_block.child_count > 0 &&
            (body_block.first_child == UINT32_MAX || body_block.first_child > ir.s7_children.size() ||
             body_block.child_count > ir.s7_children.size() - body_block.first_child)) {
            return false;
        }

        for (uint32_t ci = 0; ci < body_block.child_count; ++ci) {
            const auto child_stmt_idx = ir.s7_children[body_block.first_child + ci];
            if (child_stmt_idx >= ir.s7_stmts.size()) return false;
            const auto& child = ir.s7_stmts[child_stmt_idx];
            if (child.kind == 11) {
                const auto ret_expr_idx = child.a;
                if (ret_expr_idx == UINT32_MAX || ret_expr_idx >= ir.s7_exprs.size()) return false;
                return evaluate_s7_expr(ir, ret_expr_idx, callable_bindings, result);
            }
        }

        return false;
    }
    default:
        return false;
    }
}

const std::string* get_string(const ParsedIr& ir, uint32_t id) {
    return id < ir.strings.size() ? &ir.strings[id] : nullptr;
}

const IrResource* find_resource_by_name(const ParsedIr& ir, const std::string& name) {
    for (const auto& resource : ir.resources) {
        const auto* resource_name = get_string(ir, resource.name_string_id);
        if (resource_name != nullptr && *resource_name == name) {
            return &resource;
        }
    }

    return nullptr;
}

const IrResource* find_resource_by_binding(const ParsedIr& ir, uint32_t binding) {
    for (const auto& resource : ir.resources) {
        if (resource.binding == binding) {
            return &resource;
        }
    }

    return nullptr;
}

struct BufferUsageSummary {
    std::unordered_set<uint32_t> reads;
    std::unordered_set<uint32_t> writes;
};

BufferUsageSummary collect_ad_buffer_usage(const ParsedIr& ir) {
    BufferUsageSummary usage;

    auto mark_read_binding = [&](uint32_t binding) {
        const auto* resource = find_resource_by_binding(ir, binding);
        if (resource != nullptr && resource->kind == kIrResourceKindBuffer) {
            usage.reads.insert(binding);
        }
    };
    auto mark_write_binding = [&](uint32_t binding) {
        const auto* resource = find_resource_by_binding(ir, binding);
        if (resource != nullptr && resource->kind == kIrResourceKindBuffer) {
            usage.writes.insert(binding);
        }
    };
    auto mark_read_name = [&](const std::string& name) {
        const auto* resource = find_resource_by_name(ir, name);
        if (resource != nullptr && resource->kind == kIrResourceKindBuffer) {
            usage.reads.insert(resource->binding);
        }
    };
    auto mark_write_name = [&](const std::string& name) {
        const auto* resource = find_resource_by_name(ir, name);
        if (resource != nullptr && resource->kind == kIrResourceKindBuffer) {
            usage.writes.insert(resource->binding);
        }
    };

    std::function<void(uint32_t, const std::vector<IrExpressionNode>&, const std::vector<uint32_t>&)> collect_legacy_expr;
    collect_legacy_expr = [&](uint32_t node_index, const std::vector<IrExpressionNode>& nodes,
                              const std::vector<uint32_t>& args) {
        if (node_index >= nodes.size()) {
            return;
        }

        const auto& node = nodes[node_index];
        if (node.kind == 1) {
            mark_read_binding(node.resource_binding);
        }
        if (node.left_node_index != UINT32_MAX) {
            collect_legacy_expr(node.left_node_index, nodes, args);
        }
        if (node.right_node_index != UINT32_MAX) {
            collect_legacy_expr(node.right_node_index, nodes, args);
        }
        if (node.first_argument_index != UINT32_MAX && node.first_argument_index <= args.size() &&
            node.argument_count <= args.size() - node.first_argument_index) {
            for (uint32_t i = 0; i < node.argument_count; ++i) {
                collect_legacy_expr(args[node.first_argument_index + i], nodes, args);
            }
        }
    };

    for (const auto& assignment : ir.elementwise_assignments) {
        mark_write_binding(assignment.destination_binding);
        mark_read_binding(assignment.left_binding);
        if (assignment.right_binding != UINT32_MAX) {
            mark_read_binding(assignment.right_binding);
        }
    }
    for (const auto& assignment : ir.expression_assignments) {
        mark_write_binding(assignment.destination_binding);
        collect_legacy_expr(assignment.root_node_index, ir.expression_nodes, ir.expression_argument_indices);
    }
    for (const auto& assignment : ir.compound_assignments) {
        mark_write_binding(assignment.destination_binding);
        mark_read_binding(assignment.destination_binding);
        collect_legacy_expr(assignment.root_node_index, ir.compound_assignment_nodes, ir.compound_assignment_args);
    }
    for (const auto& expression : ir.control_flow_expressions) {
        collect_legacy_expr(expression.root_node_index, ir.control_flow_nodes, ir.control_flow_argument_indices);
    }

    if (!ir.has_section7) {
        return usage;
    }

    const auto& typed = ir.typed_module;
    auto typed_string = [&](uint32_t id) -> const std::string* {
        return id < typed.strings.size() ? &typed.strings[id] : nullptr;
    };

    std::function<void(uint32_t)> collect_typed_expr;
    std::function<void(uint32_t)> collect_lvalue_read;
    std::function<void(uint32_t)> collect_lvalue_write;
    std::function<void(uint32_t)> collect_statement;

    auto collect_typed_args = [&](const Feather::TypedIR::Expression& expr) {
        if (expr.first_argument == Feather::TypedIR::NoIndex ||
            expr.first_argument > typed.arguments.size() ||
            expr.argument_count > typed.arguments.size() - expr.first_argument) {
            return;
        }

        for (uint32_t i = 0; i < expr.argument_count; ++i) {
            collect_typed_expr(typed.arguments[expr.first_argument + i]);
        }
    };

    collect_typed_expr = [&](uint32_t expression_id) {
        if (expression_id >= typed.expressions.size()) {
            return;
        }

        const auto& expr = typed.expressions[expression_id];
        switch (expr.kind) {
        case kTypedExpressionResourceElement:
            if (const auto* name = typed_string(expr.name_id)) {
                mark_read_name(*name);
            }
            collect_typed_expr(expr.a);
            break;
        case kTypedExpressionUnary:
        case kTypedExpressionConversion:
        case kTypedExpressionSwizzle:
        case kTypedExpressionMemberAccess:
            collect_typed_expr(expr.a);
            break;
        case kTypedExpressionBinary:
        case kTypedExpressionComparison:
        case kTypedExpressionLogical:
        case kTypedExpressionIndexAccess:
        case kTypedExpressionMatrixColumn:
            collect_typed_expr(expr.a);
            collect_typed_expr(expr.b);
            break;
        case kTypedExpressionConditional:
            collect_typed_expr(expr.a);
            collect_typed_expr(expr.b);
            collect_typed_expr(expr.c);
            break;
        case kTypedExpressionConstructor:
        case kTypedExpressionIntrinsic:
        case kTypedExpressionCallableCall:
        case kTypedExpressionTextureSample:
            collect_typed_args(expr);
            break;
        case kTypedExpressionAtomic:
            collect_lvalue_read(expr.a);
            collect_lvalue_write(expr.a);
            collect_typed_args(expr);
            break;
        default:
            break;
        }
    };

    collect_lvalue_read = [&](uint32_t lvalue_id) {
        if (lvalue_id >= typed.lvalues.size()) {
            return;
        }

        const auto& lvalue = typed.lvalues[lvalue_id];
        switch (lvalue.kind) {
        case kTypedLValueResourceElement:
            if (const auto* name = typed_string(lvalue.name_id)) {
                mark_read_name(*name);
            }
            collect_typed_expr(lvalue.a);
            break;
        case kTypedLValueField:
        case kTypedLValueMemberAccess:
            collect_lvalue_read(lvalue.a);
            break;
        case kTypedLValueIndexAccess:
            collect_lvalue_read(lvalue.a);
            collect_typed_expr(lvalue.b);
            break;
        case kTypedLValueSwizzle:
        case kTypedLValueMatrixColumn:
            collect_typed_expr(lvalue.a);
            collect_typed_expr(lvalue.b);
            break;
        case kTypedLValueSharedMemoryElement:
            collect_typed_expr(lvalue.a);
            break;
        default:
            break;
        }
    };

    collect_lvalue_write = [&](uint32_t lvalue_id) {
        if (lvalue_id >= typed.lvalues.size()) {
            return;
        }

        const auto& lvalue = typed.lvalues[lvalue_id];
        switch (lvalue.kind) {
        case kTypedLValueResourceElement:
            if (const auto* name = typed_string(lvalue.name_id)) {
                mark_write_name(*name);
            }
            collect_typed_expr(lvalue.a);
            break;
        case kTypedLValueField:
        case kTypedLValueMemberAccess:
            collect_lvalue_write(lvalue.a);
            break;
        case kTypedLValueIndexAccess:
            collect_lvalue_write(lvalue.a);
            collect_typed_expr(lvalue.b);
            break;
        case kTypedLValueSwizzle:
        case kTypedLValueMatrixColumn:
            collect_typed_expr(lvalue.a);
            collect_typed_expr(lvalue.b);
            break;
        case kTypedLValueSharedMemoryElement:
            collect_typed_expr(lvalue.a);
            break;
        default:
            break;
        }
    };

    collect_statement = [&](uint32_t statement_id) {
        if (statement_id >= typed.statements.size()) {
            return;
        }

        const auto& statement = typed.statements[statement_id];
        switch (statement.kind) {
        case kTypedStatementBlock:
            if (statement.first_child != Feather::TypedIR::NoIndex &&
                statement.first_child <= typed.children.size() &&
                statement.child_count <= typed.children.size() - statement.first_child) {
                for (uint32_t i = 0; i < statement.child_count; ++i) {
                    collect_statement(typed.children[statement.first_child + i]);
                }
            }
            break;
        case kTypedStatementLocalDeclaration:
            collect_typed_expr(statement.a);
            break;
        case kTypedStatementAssignment:
            collect_lvalue_write(statement.a);
            collect_typed_expr(statement.b);
            break;
        case kTypedStatementCompoundAssignment:
            collect_lvalue_read(statement.a);
            collect_lvalue_write(statement.a);
            collect_typed_expr(statement.b);
            break;
        case kTypedStatementIf:
            collect_typed_expr(statement.a);
            collect_statement(statement.b);
            collect_statement(statement.c);
            break;
        case kTypedStatementFor:
            collect_statement(statement.a);
            collect_typed_expr(statement.b);
            collect_statement(statement.c);
            collect_statement(statement.op);
            break;
        case kTypedStatementWhile:
            collect_typed_expr(statement.a);
            collect_statement(statement.b);
            break;
        case kTypedStatementDoWhile:
            collect_statement(statement.a);
            collect_typed_expr(statement.b);
            break;
        case kTypedStatementReturn:
        case kTypedStatementExpression:
            collect_typed_expr(statement.a);
            break;
        case kTypedStatementIncrementDecrement:
            collect_lvalue_read(statement.a);
            collect_lvalue_write(statement.a);
            break;
        default:
            break;
        }
    };

    for (const auto& function : typed.functions) {
        collect_statement(function.body_statement_index);
    }

    return usage;
}

std::string trim_copy(const std::string& source) {
    const auto start = source.find_first_not_of(" \t\n");
    if (start == std::string::npos) {
        return {};
    }

    const auto end = source.find_last_not_of(" \t\n");
    return source.substr(start, end - start + 1);
}

bool parse_floating_literal(const std::string& source, double* value) {
    const auto text = trim_copy(source);
    if (text.empty()) {
        return false;
    }

    errno = 0;
    char* end = nullptr;
    const auto parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || errno == ERANGE) {
        return false;
    }

    while (*end != '\0') {
        if (*end != 'f' && *end != 'F' && *end != 'd' && *end != 'D' && *end != 'm' && *end != 'M' && *end != ' ') {
            return false;
        }
        ++end;
    }

    *value = parsed;
    return true;
}

std::vector<std::string> split_payload(const std::string& payload) {
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= payload.size()) {
        const auto separator = payload.find('|', start);
        if (separator == std::string::npos) {
            parts.push_back(payload.substr(start));
            break;
        }

        parts.push_back(payload.substr(start, separator - start));
        start = separator + 1;
    }

    return parts;
}

char operation_from_ir(uint8_t operation) {
    switch (operation) {
    case 2:
        return '+';
    case 3:
        return '-';
    case 4:
        return '*';
    case 5:
        return '/';
    default:
        return 0;
    }
}

bool convert_structured_assignment(const ParsedIr& ir, const IrElementwiseAssignment& source,
                                   FallbackAssignment* assignment) {
    if (assignment == nullptr) {
        return false;
    }

    if (find_resource_by_binding(ir, source.destination_binding) == nullptr ||
        find_resource_by_binding(ir, source.left_binding) == nullptr) {
        return false;
    }

    assignment->destination_binding = source.destination_binding;
    assignment->left_binding = source.left_binding;

    if (source.operation == 1) {
        assignment->kind = FallbackExpressionKind::Copy;
        return source.right_operand_kind == 0;
    }

    assignment->operation = operation_from_ir(source.operation);
    if (assignment->operation == 0) {
        return false;
    }

    if (source.right_operand_kind == 1) {
        if (find_resource_by_binding(ir, source.right_binding) == nullptr) {
            return false;
        }

        assignment->right_binding = source.right_binding;
        assignment->kind = FallbackExpressionKind::BufferBinaryBuffer;
        return true;
    }

    if (source.right_operand_kind == 2) {
        const auto* literal = get_string(ir, source.right_literal_string_id);
        if (literal == nullptr || !parse_floating_literal(*literal, &assignment->literal_value)) {
            return false;
        }

        assignment->kind = FallbackExpressionKind::BufferBinaryLiteral;
        return true;
    }

    return false;
}

bool parse_elementwise_assignment_payload(const ParsedIr& ir, const std::string& payload,
                                          FallbackAssignment* assignment) {
    if (assignment == nullptr) {
        return false;
    }

    const auto parts = split_payload(payload);
    if (parts.size() != 6 || parts[0] != "ASSIGN1" || parts[1].empty() || parts[2].empty() || parts[4].empty()) {
        return false;
    }

    const auto* destination = find_resource_by_name(ir, parts[1]);
    const auto* left = find_resource_by_name(ir, parts[4]);
    if (destination == nullptr || left == nullptr) {
        return false;
    }

    assignment->destination_binding = destination->binding;
    assignment->left_binding = left->binding;

    if (parts[3] == "copy") {
        assignment->kind = FallbackExpressionKind::Copy;
        return parts[5].empty();
    }

    assignment->operation = parts[3] == "add"   ? '+'
                            : parts[3] == "sub" ? '-'
                            : parts[3] == "mul" ? '*'
                            : parts[3] == "div" ? '/'
                                                : 0;
    if (assignment->operation == 0 || parts[5].empty()) {
        return false;
    }

    if (parse_floating_literal(parts[5], &assignment->literal_value)) {
        assignment->kind = FallbackExpressionKind::BufferBinaryLiteral;
        return true;
    }

    const auto* right = find_resource_by_name(ir, parts[5]);
    if (right == nullptr) {
        return false;
    }

    assignment->right_binding = right->binding;
    assignment->kind = FallbackExpressionKind::BufferBinaryBuffer;
    return true;
}

bool is_float_resource(const ParsedIr& ir, const IrResource& resource) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && (*type == "System.Single" || *type == "float");
}

bool is_int_resource(const ParsedIr& ir, const IrResource& resource) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && (*type == "System.Int32" || *type == "int");
}

bool is_uint_resource(const ParsedIr& ir, const IrResource& resource) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && (*type == "System.UInt32" || *type == "uint");
}

bool is_float_type(const ParsedIr& ir, uint32_t type_string_id) {
    const auto* type = get_string(ir, type_string_id);
    return type != nullptr && (*type == "System.Single" || *type == "float");
}

size_t float_vector_component_count(const std::string& type) {
    if (type == "Feather.Math.float2" || type == "global::Feather.Math.float2" || type == "float2") {
        return 2;
    }

    if (type == "Feather.Math.float3" || type == "global::Feather.Math.float3" || type == "float3") {
        return 3;
    }

    if (type == "Feather.Math.float4" || type == "global::Feather.Math.float4" || type == "float4") {
        return 4;
    }

    return 0;
}

bool is_float_vector_type_name(const std::string& type, size_t component_count) {
    return float_vector_component_count(type) == component_count;
}

bool is_float_vector_resource(const ParsedIr& ir, const IrResource& resource, size_t component_count) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && is_float_vector_type_name(*type, component_count);
}

bool is_int_vector_type_name(const std::string& type_name, size_t component_count) {
    const auto suffix = std::to_string(component_count);
    return type_name == "Feather.Math.int" + suffix ||
           type_name == "global::Feather.Math.int" + suffix ||
           type_name == "int" + suffix;
}

bool is_uint_vector_type_name(const std::string& type_name, size_t component_count) {
    const auto suffix = std::to_string(component_count);
    return type_name == "Feather.Math.uint" + suffix ||
           type_name == "global::Feather.Math.uint" + suffix ||
           type_name == "uint" + suffix;
}

bool is_int_vector_resource(const ParsedIr& ir, const IrResource& resource, size_t component_count) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && is_int_vector_type_name(*type, component_count);
}

bool is_uint_vector_resource(const ParsedIr& ir, const IrResource& resource, size_t component_count) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    return type != nullptr && is_uint_vector_type_name(*type, component_count);
}

bool is_float_vector_type(const ParsedIr& ir, uint32_t type_string_id, size_t component_count) {
    const auto* type = get_string(ir, type_string_id);
    return type != nullptr && is_float_vector_type_name(*type, component_count);
}

bool is_int_type(const ParsedIr& ir, uint32_t type_string_id) {
    const auto* type = get_string(ir, type_string_id);
    return type != nullptr && (*type == "System.Int32" || *type == "int");
}

size_t float_vector_buffer_stride(size_t component_count) {
    // Mirrors EasyGPU/include/Utility/Meta/Std430Layout.h::GetStd430SizeHelper for Vec2/Vec3/Vec4 arrays.
    switch (component_count) {
    case 2:
        return 8;
    case 3:
    case 4:
        return 16;
    default:
        return 0;
    }
}

size_t easygpu_buffer_element_stride(const ParsedIr& ir, const IrResource& resource) {
    if (is_float_resource(ir, resource) || is_int_resource(ir, resource) || is_uint_resource(ir, resource)) {
        return 4;
    }

    for (const auto component_count : {size_t{2}, size_t{3}, size_t{4}}) {
        if (is_float_vector_resource(ir, resource, component_count) ||
            is_int_vector_resource(ir, resource, component_count) ||
            is_uint_vector_resource(ir, resource, component_count)) {
            return float_vector_buffer_stride(component_count);
        }
    }

    const auto* type = get_string(ir, resource.element_type_string_id);
    if (type != nullptr && ir.has_section7) {
        for (const auto& structure : ir.typed_module.structs) {
            const auto* simple = structure.name_id < ir.typed_module.strings.size()
                                     ? &ir.typed_module.strings[structure.name_id]
                                     : nullptr;
            const auto* qualified = structure.fully_qualified_name_id < ir.typed_module.strings.size()
                                        ? &ir.typed_module.strings[structure.fully_qualified_name_id]
                                        : nullptr;
            const auto normalized_type = type->rfind("global::", 0) == 0 ? type->substr(8) : *type;
            const auto normalized_qualified = qualified != nullptr && qualified->rfind("global::", 0) == 0
                                                  ? qualified->substr(8)
                                                  : (qualified == nullptr ? std::string{} : *qualified);
            if ((simple != nullptr && *simple == *type) ||
                (qualified != nullptr && *qualified == *type) ||
                (!normalized_qualified.empty() && normalized_qualified == normalized_type)) {
                return structure.size_in_bytes;
            }
        }
    }

    return 0;
}

std::string easygpu_buffer_name(const IrResource& resource) {
    return "fe_" + std::to_string(resource.binding);
}

std::string string_or_empty(const ParsedIr& ir, uint32_t id) {
    const auto* value = get_string(ir, id);
    return value == nullptr ? std::string{} : *value;
}

void copy_fixed_c_string(char* destination, size_t destination_size, const std::string& value) {
    if (destination == nullptr || destination_size == 0) {
        return;
    }

    const auto count = std::min(destination_size - 1, value.size());
    std::memcpy(destination, value.data(), count);
    destination[count] = '\0';
}

std::string easygpu_push_constant_name(const IrResource& resource) {
    return "pc_" + std::to_string(resource.binding);
}

GPU::IR::Type easygpu_module_type(const std::string& type_name) {
    if (type_name == "System.Single" || type_name == "float") {
        return GPU::IR::Type::Float();
    }

    if (type_name == "System.Int32" || type_name == "int") {
        return GPU::IR::Type::Int();
    }

    if (type_name == "System.UInt32" || type_name == "uint") {
        return GPU::IR::Type::UInt();
    }

    if (type_name == "Feather.Math.int2" || type_name == "global::Feather.Math.int2" || type_name == "int2") {
        return GPU::IR::Type::Int2();
    }

    if (type_name == "Feather.Math.int3" || type_name == "global::Feather.Math.int3" || type_name == "int3") {
        return GPU::IR::Type::Int3();
    }

    if (type_name == "Feather.Math.int4" || type_name == "global::Feather.Math.int4" || type_name == "int4") {
        return GPU::IR::Type::Int4();
    }

    if (type_name == "Feather.Math.uint2" || type_name == "global::Feather.Math.uint2" || type_name == "uint2") {
        return GPU::IR::Type::UInt2();
    }

    if (type_name == "Feather.Math.uint3" || type_name == "global::Feather.Math.uint3" || type_name == "uint3") {
        return GPU::IR::Type::UInt3();
    }

    if (type_name == "Feather.Math.uint4" || type_name == "global::Feather.Math.uint4" || type_name == "uint4") {
        return GPU::IR::Type::UInt4();
    }

    // Byte-sized GpuStruct fields map to float in GLSL (no byte type).
    if (type_name == "System.Byte" || type_name == "byte") {
        return GPU::IR::Type::Float();
    }

    if (type_name == "Feather.Math.float2" || type_name == "global::Feather.Math.float2" || type_name == "float2") {
        return GPU::IR::Type::Float2();
    }

    if (type_name == "Feather.Math.float3" || type_name == "global::Feather.Math.float3" || type_name == "float3") {
        return GPU::IR::Type::Float3();
    }

    if (type_name == "Feather.Math.float4" || type_name == "global::Feather.Math.float4" || type_name == "float4") {
        return GPU::IR::Type::Float4();
    }

    if (type_name == "Feather.Math.float2x2" || type_name == "global::Feather.Math.float2x2" ||
        type_name == "float2x2") {
        return GPU::IR::Type::Float2x2();
    }

    if (type_name == "Feather.Math.float3x3" || type_name == "global::Feather.Math.float3x3" ||
        type_name == "float3x3") {
        return GPU::IR::Type::Float3x3();
    }

    if (type_name == "Feather.Math.float4x4" || type_name == "global::Feather.Math.float4x4" ||
        type_name == "float4x4") {
        return GPU::IR::Type::Float4x4();
    }

    // GpuStruct types with 4 byte fields (Rgba32 etc.) → vec4 in GLSL.
    if (type_name.find("Rgba32") != std::string::npos ||
        type_name.find("Rgba") != std::string::npos) {
        return GPU::IR::Type::Float4();
    }

    return {};
}

GPU::IR::ResourceAccess easygpu_resource_access(uint8_t access) {
    switch (access) {
    case 1:
        return GPU::IR::ResourceAccess::Read;
    case 2:
        return GPU::IR::ResourceAccess::Write;
    case 4: // Sampled texture
        return GPU::IR::ResourceAccess::Read;
    default:
        return GPU::IR::ResourceAccess::ReadWrite;
    }
}

bool easygpu_expression_binary_op(uint8_t operation, GPU::IR::BinaryOp* op) {
    if (op == nullptr) {
        return false;
    }

    switch (operation) {
    case 1:
        *op = GPU::IR::BinaryOp::Add;
        return true;
    case 2:
        *op = GPU::IR::BinaryOp::Sub;
        return true;
    case 3:
        *op = GPU::IR::BinaryOp::Mul;
        return true;
    case 4:
        *op = GPU::IR::BinaryOp::Div;
        return true;
    default:
        return false;
    }
}

bool easygpu_structured_binary_op(uint8_t operation, GPU::IR::BinaryOp* op) {
    if (op == nullptr) {
        return false;
    }

    switch (operation) {
    case 2:
        *op = GPU::IR::BinaryOp::Add;
        return true;
    case 3:
        *op = GPU::IR::BinaryOp::Sub;
        return true;
    case 4:
        *op = GPU::IR::BinaryOp::Mul;
        return true;
    case 5:
        *op = GPU::IR::BinaryOp::Div;
        return true;
    default:
        return false;
    }
}

bool easygpu_compare_op(uint8_t operation, GPU::IR::CompareOp* op) {
    if (op == nullptr) return false;
    switch (operation) {
    case 5: *op = GPU::IR::CompareOp::Equal; return true;
    case 6: *op = GPU::IR::CompareOp::NotEqual; return true;
    case 7: *op = GPU::IR::CompareOp::Greater; return true;
    case 8: *op = GPU::IR::CompareOp::Less; return true;
    case 9: *op = GPU::IR::CompareOp::GreaterEqual; return true;
    case 10: *op = GPU::IR::CompareOp::LessEqual; return true;
    default: return false;
    }
}

std::string easygpu_intrinsic_name_for_type(const std::string& type_name) {
    if (type_name == "Feather.Math.float2" || type_name == "float2") return "vec2";
    if (type_name == "Feather.Math.float3" || type_name == "float3") return "vec3";
    if (type_name == "Feather.Math.float4" || type_name == "float4") return "vec4";
    if (type_name == "Feather.Math.int2" || type_name == "int2") return "ivec2";
    if (type_name == "Feather.Math.int3" || type_name == "int3") return "ivec3";
    if (type_name == "Feather.Math.int4" || type_name == "int4") return "ivec4";
    return {};
}

std::string easygpu_ad_glsl_type_name(const std::string& type_name) {
    if (type_name == "System.Single" || type_name == "float") return "float";
    if (type_name == "Feather.Math.float2" || type_name == "global::Feather.Math.float2" || type_name == "float2") return "vec2";
    if (type_name == "Feather.Math.float3" || type_name == "global::Feather.Math.float3" || type_name == "float3") return "vec3";
    if (type_name == "Feather.Math.float4" || type_name == "global::Feather.Math.float4" || type_name == "float4") return "vec4";
    return {};
}

uint32_t ad_component_count_for_type(const std::string& type_name) {
    const auto glsl_type = easygpu_ad_glsl_type_name(type_name);
    if (glsl_type == "float") return 1;
    if (glsl_type == "vec2") return 2;
    if (glsl_type == "vec3") return 3;
    if (glsl_type == "vec4") return 4;
    return 0;
}

size_t ad_scalar_slot_count_for_type(const std::string& type_name) {
    const auto components = ad_component_count_for_type(type_name);
    return components == 0 ? 0 : static_cast<size_t>(components);
}

void release_ad_gradient_buffers(KernelState& kernel) {
    GPU::Runtime::AutoInitContext();
    release_ad_gradient_buffers_with_backend(kernel, GPU::Runtime::Context::GetBackend());
}

void release_pending_ad_gradient_buffers(std::vector<ADGradientState>& gradients) {
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        for (auto& gradient : gradients) {
            gradient.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
        }
        return;
    }

    std::unordered_set<GPU::Backend::BufferHandle> released;
    for (auto& gradient : gradients) {
        if (gradient.backend_buffer != GPU::Backend::INVALID_BUFFER_HANDLE &&
            released.insert(gradient.backend_buffer).second) {
            backend->DestroyBuffer(gradient.backend_buffer);
        }
        gradient.backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
    }
}

std::string easygpu_intrinsic_name(const std::string& symbol) {
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

    return {};
}

size_t push_constant_type_size(const ParsedIr& ir, const IrResource& resource) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    if (type == nullptr) {
        return 0;
    }

    if (*type == "System.Boolean" || *type == "bool") {
        return 4;
    }
    if (*type == "System.Single" || *type == "float" ||
        *type == "System.Int32" || *type == "int" ||
        *type == "System.UInt32" || *type == "uint") {
        return 4;
    }

    const auto normalized_type = type->rfind("global::", 0) == 0 ? type->substr(8) : *type;
    if (normalized_type == "Feather.Math.bool2") {
        return 8;
    }
    if (normalized_type == "Feather.Math.bool3") {
        return 12;
    }
    if (normalized_type == "Feather.Math.bool4") {
        return 16;
    }
    if (normalized_type == "Feather.Math.float2x2" || normalized_type == "float2x2") {
        return 16;
    }
    if (normalized_type == "Feather.Math.float3x3" || normalized_type == "float3x3") {
        return 48;
    }
    if (normalized_type == "Feather.Math.float4x4" || normalized_type == "float4x4") {
        return 64;
    }

    const auto float_vector_components = float_vector_component_count(*type);
    if (float_vector_components != 0) {
        return float_vector_components * sizeof(float);
    }

    return 0;
}

size_t push_constant_type_alignment(const ParsedIr& ir, const IrResource& resource) {
    const auto* type = get_string(ir, resource.element_type_string_id);
    if (type == nullptr) {
        return 0;
    }

    if (*type == "System.Boolean" || *type == "bool") {
        return 4;
    }
    if (*type == "System.Single" || *type == "float" ||
        *type == "System.Int32" || *type == "int" ||
        *type == "System.UInt32" || *type == "uint") {
        return 4;
    }

    const auto normalized_type = type->rfind("global::", 0) == 0 ? type->substr(8) : *type;
    if (normalized_type == "Feather.Math.bool2") {
        return 8;
    }
    if (normalized_type == "Feather.Math.bool3" || normalized_type == "Feather.Math.bool4" ||
        normalized_type == "Feather.Math.float3x3" || normalized_type == "float3x3" ||
        normalized_type == "Feather.Math.float4x4" || normalized_type == "float4x4") {
        return 16;
    }
    if (normalized_type == "Feather.Math.float2x2" || normalized_type == "float2x2") {
        return 8;
    }

    const auto float_vector_components = float_vector_component_count(*type);
    if (float_vector_components == 2) {
        return 8;
    }

    if (float_vector_components == 3 || float_vector_components == 4) {
        return 16;
    }

    return 0;
}

size_t align_offset(size_t offset, size_t alignment) {
    if (alignment <= 1) {
        return offset;
    }

    const auto remainder = offset % alignment;
    return remainder == 0 ? offset : offset + alignment - remainder;
}

bool find_push_constant_offset(const ParsedIr& ir, uint32_t binding, size_t* offset, size_t* size) {
    if (offset == nullptr || size == nullptr) {
        return false;
    }

    // Push constants are packed by generated C# binding code in resource-table order.
    // The native fallback mirrors that layout so expression nodes can read by binding.
    size_t current_offset = 0;
    for (const auto& resource : ir.resources) {
        if (resource.kind != kIrResourceKindPushConstant) {
            continue;
        }

        const auto resource_size = push_constant_type_size(ir, resource);
        const auto resource_alignment = push_constant_type_alignment(ir, resource);
        if (resource_size == 0 || resource_alignment == 0) {
            return false;
        }

        // Matches EasyGPU/source/Kernel/KernelBuildContext.cpp::RegisterUniform.
        current_offset = align_offset(current_offset, resource_alignment);
        if (resource.binding == binding) {
            *offset = current_offset;
            *size = resource_size;
            return true;
        }

        current_offset += resource_size;
    }

    return false;
}

double apply_binary_operation(double left, double right, char operation) {
    switch (operation) {
    case '+':
        return left + right;
    case '-':
        return left - right;
    case '*':
        return left * right;
    case '/':
        return right == 0.0 ? 0.0 : left / right;
    default:
        return left;
    }
}

template <size_t N>
FloatVectorValue<N> apply_float_vector_binary_operation(FloatVectorValue<N> left, FloatVectorValue<N> right,
                                                        char operation) {
    FloatVectorValue<N> result{};
    for (size_t i = 0; i < N; ++i) {
        result.components[i] = static_cast<float>(apply_binary_operation(left.components[i], right.components[i], operation));
    }

    return result;
}

template <size_t N> float dot_float_vector(FloatVectorValue<N> left, FloatVectorValue<N> right) {
    float result = 0.0f;
    for (size_t i = 0; i < N; ++i) {
        result += left.components[i] * right.components[i];
    }

    return result;
}

FloatVectorValue<3> cross_float3_vector(FloatVectorValue<3> left, FloatVectorValue<3> right) {
    FloatVectorValue<3> result{};
    result.components[0] = (left.components[1] * right.components[2]) - (left.components[2] * right.components[1]);
    result.components[1] = (left.components[2] * right.components[0]) - (left.components[0] * right.components[2]);
    result.components[2] = (left.components[0] * right.components[1]) - (left.components[1] * right.components[0]);
    return result;
}

char expression_operation_from_ir(uint8_t operation) {
    switch (operation) {
    case 1:
        return '+';
    case 2:
        return '-';
    case 3:
        return '*';
    case 4:
        return '/';
    default:
        return 0;
    }
}

bool try_parse_float_literal(const ParsedIr& ir, uint32_t literal_string_id, float* value) {
    if (value == nullptr) {
        return false;
    }

    const auto* literal = get_string(ir, literal_string_id);
    double parsed = 0.0;
    if (literal == nullptr || !parse_floating_literal(*literal, &parsed)) {
        return false;
    }

    *value = static_cast<float>(parsed);
    return true;
}

double apply_intrinsic_operation(const std::string& symbol, const std::vector<double>& arguments, bool* supported) {
    if (supported == nullptr) {
        return 0.0;
    }

    *supported = true;
    if (arguments.size() == 1) {
        const auto x = arguments[0];
        if (symbol == "global::Feather.Math.ShaderMath.Sin" || symbol == "global::Feather.Math.Hlsl.Sin") {
            return std::sin(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Cos" || symbol == "global::Feather.Math.Hlsl.Cos") {
            return std::cos(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Tan" || symbol == "global::Feather.Math.Hlsl.Tan") {
            return std::tan(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Exp" || symbol == "global::Feather.Math.Hlsl.Exp") {
            return std::exp(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Log" || symbol == "global::Feather.Math.Hlsl.Log") {
            return std::log(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Sqrt" || symbol == "global::Feather.Math.Hlsl.Sqrt") {
            return std::sqrt(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.InverseSqrt") {
            return 1.0 / std::sqrt(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Abs" || symbol == "global::Feather.Math.Hlsl.Abs") {
            return std::fabs(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Floor" || symbol == "global::Feather.Math.Hlsl.Floor") {
            return std::floor(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Ceil" || symbol == "global::Feather.Math.Hlsl.Ceil") {
            return std::ceil(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Round") {
            return std::round(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Fract" || symbol == "global::Feather.Math.Hlsl.Fract") {
            return x - std::floor(x);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Saturate") {
            return std::min(1.0, std::max(0.0, x));
        }
    }

    if (arguments.size() == 2) {
        const auto x = arguments[0];
        const auto y = arguments[1];
        if (symbol == "global::Feather.Math.ShaderMath.Pow" || symbol == "global::Feather.Math.Hlsl.Pow") {
            return std::pow(x, y);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Min") {
            return std::min(x, y);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Max") {
            return std::max(x, y);
        }
    }

    if (arguments.size() == 3) {
        const auto x = arguments[0];
        const auto y = arguments[1];
        const auto z = arguments[2];
        if (symbol == "global::Feather.Math.ShaderMath.Clamp" || symbol == "global::Feather.Math.Hlsl.Clamp") {
            return std::min(z, std::max(y, x));
        }
        if (symbol == "global::Feather.Math.ShaderMath.Lerp" || symbol == "global::Feather.Math.Hlsl.Lerp" ||
            symbol == "global::Feather.Math.ShaderMath.Mix" || symbol == "global::Feather.Math.Hlsl.Mix") {
            return x + ((y - x) * z);
        }
        if (symbol == "global::Feather.Math.ShaderMath.Smoothstep") {
            const auto t = std::min(1.0, std::max(0.0, (z - x) / (y - x)));
            return t * t * (3.0 - (2.0 * t));
        }
    }

    *supported = false;
    return 0.0;
}

bool is_dot_intrinsic_symbol(const std::string& symbol) {
    return symbol == "global::Feather.Math.ShaderMath.Dot" || symbol == "global::Feather.Math.Hlsl.Dot";
}

bool is_cross_intrinsic_symbol(const std::string& symbol) {
    return symbol == "global::Feather.Math.ShaderMath.Cross" || symbol == "global::Feather.Math.Hlsl.Cross";
}

template <size_t N>
bool evaluate_float_vector_expression_node(const ParsedIr& ir, const KernelState& kernel, uint32_t node_index,
                                           size_t element_index, FloatVectorValue<N>* value);

template <size_t N>
bool try_evaluate_dot_intrinsic(const ParsedIr& ir, const KernelState& kernel, const IrExpressionNode& node,
                                size_t element_index, float* value) {
    if (value == nullptr || node.argument_count != 2 ||
        node.first_argument_index == UINT32_MAX ||
        node.first_argument_index > ir.expression_argument_indices.size() ||
        node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
        return false;
    }

    FloatVectorValue<N> left{};
    FloatVectorValue<N> right{};
    const auto left_node_index = ir.expression_argument_indices[node.first_argument_index];
    const auto right_node_index = ir.expression_argument_indices[node.first_argument_index + 1];
    if (!evaluate_float_vector_expression_node(ir, kernel, left_node_index, element_index, &left) ||
        !evaluate_float_vector_expression_node(ir, kernel, right_node_index, element_index, &right)) {
        return false;
    }

    *value = dot_float_vector(left, right);
    return true;
}

bool try_evaluate_cross_intrinsic(const ParsedIr& ir, const KernelState& kernel, const IrExpressionNode& node,
                                  size_t element_index, FloatVectorValue<3>* value) {
    if (value == nullptr || node.argument_count != 2 ||
        node.first_argument_index == UINT32_MAX ||
        node.first_argument_index > ir.expression_argument_indices.size() ||
        node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
        return false;
    }

    FloatVectorValue<3> left{};
    FloatVectorValue<3> right{};
    const auto left_node_index = ir.expression_argument_indices[node.first_argument_index];
    const auto right_node_index = ir.expression_argument_indices[node.first_argument_index + 1];
    if (!evaluate_float_vector_expression_node(ir, kernel, left_node_index, element_index, &left) ||
        !evaluate_float_vector_expression_node(ir, kernel, right_node_index, element_index, &right)) {
        return false;
    }

    *value = cross_float3_vector(left, right);
    return true;
}

template <typename T>
bool evaluate_expression_node(const ParsedIr& ir, const KernelState& kernel, uint32_t node_index, size_t element_index,
                              T* value) {
    if (value == nullptr || node_index >= ir.expression_nodes.size()) {
        return false;
    }

    const auto& node = ir.expression_nodes[node_index];
    switch (node.kind) {
    case 1: {
        const auto* resource = find_resource_by_binding(ir, node.resource_binding);
        if (resource == nullptr || resource->kind != 1) {
            return false;
        }

        const auto bound = kernel.buffers.find(resource->binding);
        if (bound == kernel.buffers.end()) {
            return false;
        }

        const auto buffer = g_buffers.find(bound->second);
        if (buffer == g_buffers.end() || buffer->second.stride < sizeof(T) ||
            element_index >= buffer->second.bytes.size() / buffer->second.stride ||
            (element_index * buffer->second.stride) > buffer->second.bytes.size() - sizeof(T)) {
            return false;
        }

        std::memcpy(value, buffer->second.bytes.data() + (element_index * buffer->second.stride), sizeof(T));
        return true;
    }
    case 2: {
        const auto* literal = get_string(ir, node.literal_string_id);
        double parsed = 0.0;
        if (literal == nullptr || !parse_floating_literal(*literal, &parsed)) {
            return false;
        }

        *value = static_cast<T>(parsed);
        return true;
    }
    case 3: {
        const auto operation = expression_operation_from_ir(node.operation);
        T left{};
        T right{};
        if (operation == 0 ||
            !evaluate_expression_node(ir, kernel, node.left_node_index, element_index, &left) ||
            !evaluate_expression_node(ir, kernel, node.right_node_index, element_index, &right)) {
            return false;
        }

        *value = static_cast<T>(apply_binary_operation(static_cast<double>(left), static_cast<double>(right), operation));
        return true;
    }
    case 4: {
        const auto* symbol = get_string(ir, node.symbol_string_id);
        if (symbol == nullptr ||
            (node.argument_count > 0 &&
             (node.first_argument_index == UINT32_MAX ||
              node.first_argument_index > ir.expression_argument_indices.size() ||
              node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index))) {
            return false;
        }

        if (is_dot_intrinsic_symbol(*symbol)) {
            float dot = 0.0f;
            // Dot is the first supported scalar-result vector intrinsic; evaluate arguments as typed vectors
            // rather than forcing them through the scalar invocation path.
            if ((try_evaluate_dot_intrinsic<2>(ir, kernel, node, element_index, &dot) ||
                 try_evaluate_dot_intrinsic<3>(ir, kernel, node, element_index, &dot) ||
                 try_evaluate_dot_intrinsic<4>(ir, kernel, node, element_index, &dot))) {
                *value = static_cast<T>(dot);
                return true;
            }

            return false;
        }

        std::vector<double> arguments;
        arguments.reserve(node.argument_count);
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            T argument_value{};
            const auto argument_node_index = ir.expression_argument_indices[node.first_argument_index + i];
            if (!evaluate_expression_node(ir, kernel, argument_node_index, element_index, &argument_value)) {
                return false;
            }

            arguments.push_back(static_cast<double>(argument_value));
        }

        bool supported = false;
        const auto result = apply_intrinsic_operation(*symbol, arguments, &supported);
        if (!supported) {
            return false;
        }

        *value = static_cast<T>(result);
        return true;
    }
    case 5: {
        // Push constant (byte 5)
        size_t offset = 0;
        size_t size = 0;
        if (!find_push_constant_offset(ir, node.resource_binding, &offset, &size) ||
            offset + size > kernel.push_constants.size() || size != sizeof(T)) {
            return false;
        }
        T constant_value{};
        std::memcpy(&constant_value, kernel.push_constants.data() + offset, sizeof(T));
        *value = constant_value;
        return true;
    }
    case 6: {
        // Comparison node (byte 6): apply the comparison operator and return 1.0 or 0.0.
        T left{}, right{};
        if (!evaluate_expression_node(ir, kernel, node.left_node_index, element_index, &left) ||
            !evaluate_expression_node(ir, kernel, node.right_node_index, element_index, &right))
            return false;
        double r = 0.0;
        switch (node.operation) {
            case 5: r = (left == right) ? 1.0 : 0.0; break;
            case 6: r = (left != right) ? 1.0 : 0.0; break;
            case 7: r = (left > right) ? 1.0 : 0.0; break;
            case 8: r = (left < right) ? 1.0 : 0.0; break;
            case 9: r = (left >= right) ? 1.0 : 0.0; break;
            case 10: r = (left <= right) ? 1.0 : 0.0; break;
            default: return false;
        }
        *value = static_cast<T>(r);
        return true;
    }
    case 7: return false; // LocalVariable: not supported in scalar fallback
    case 9: {
        // Ternary: condition=left, whenTrue=right, whenFalse=arguments[0]
        if (node.left_node_index == UINT32_MAX || node.right_node_index == UINT32_MAX ||
            node.argument_count < 1 || node.first_argument_index == UINT32_MAX ||
            node.first_argument_index >= ir.expression_argument_indices.size() ||
            node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index)
            return false;
        T cond{}, whenTrue{}, whenFalse{};
        if (!evaluate_expression_node(ir, kernel, node.left_node_index, element_index, &cond) ||
            !evaluate_expression_node(ir, kernel, node.right_node_index, element_index, &whenTrue) ||
            !evaluate_expression_node(ir, kernel, ir.expression_argument_indices[node.first_argument_index],
                element_index, &whenFalse))
            return false;
        *value = cond ? whenTrue : whenFalse;
        return true;
    }
    case 10: {
        // Constructor node: evaluate arguments and combine for scalar result
        if (node.argument_count == 0 ||
            node.first_argument_index == UINT32_MAX ||
            node.first_argument_index > ir.expression_argument_indices.size() ||
            node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
            return false;
        }
        // For a scalar constructor, evaluate each argument and return the first.
        // Multi-component constructors (float2/3/4) are handled by the vector evaluator.
        if (node.argument_count == 1) {
            T arg{};
            if (!evaluate_expression_node(ir, kernel,
                    ir.expression_argument_indices[node.first_argument_index],
                    element_index, &arg)) {
                return false;
            }
            *value = arg;
            return true;
        }
        return false;
    }
    case 12:
    case 13: {
        // TextureSample / TextureSampleLevel CPU fallback.
        const auto* tex_res = find_resource_by_binding(ir, node.resource_binding);
        if (tex_res == nullptr || tex_res->kind != kIrResourceKindTexture2D) return false;
        const auto tex_it = kernel.textures.find(tex_res->binding);
        if (tex_it == kernel.textures.end()) return false;
        const auto tex = g_textures.find(tex_it->second);
        if (tex == g_textures.end()) return false;

        // Evaluate UV coordinates from the first argument expression.
        if (node.first_argument_index == UINT32_MAX || node.argument_count == 0) return false;
        const auto uv_node = ir.expression_argument_indices[node.first_argument_index];
        const auto& uv_expr = ir.expression_nodes[uv_node];
        float uv_x = 0, uv_y = 0;
        if (uv_expr.kind == kIrExpressionNodeKindConstructor && uv_expr.argument_count >= 2) {
            float fx = 0, fy = 0;
            if (!evaluate_expression_node(ir, kernel, ir.expression_argument_indices[uv_expr.first_argument_index],
                    element_index, &fx)) return false;
            if (!evaluate_expression_node(ir, kernel, ir.expression_argument_indices[uv_expr.first_argument_index + 1],
                    element_index, &fy)) return false;
            uv_x = fx; uv_y = fy;
        } else {
            return false; // UV must be a float2 constructor in the simple case
        }

        // LOD for SampleLevel
        float lod = 0.0f;
        if (node.kind == 13 && node.argument_count >= 2) {
            const auto lod_node = ir.expression_argument_indices[node.first_argument_index + 1];
            if (!evaluate_expression_node(ir, kernel, lod_node, element_index, &lod)) return false;
        }

        const auto tw = static_cast<float>(tex->second.width);
        const auto th = static_cast<float>(tex->second.height);
        const auto bpp = pixel_size(tex->second.pixel_format);
        if (bpp < 4 || tw <= 0 || th <= 0) return false;

        float tx = uv_x * tw - 0.5f;
        float ty = uv_y * th - 0.5f;
        int x0 = std::max(0, std::min(static_cast<int>(tx), static_cast<int>(tw) - 1));
        int y0 = std::max(0, std::min(static_cast<int>(ty), static_cast<int>(th) - 1));

        if (node.kind == 12 /* TextureSample — bilinear */) {
            int x1 = std::min(x0 + 1, static_cast<int>(tw) - 1);
            int y1 = std::min(y0 + 1, static_cast<int>(th) - 1);
            float fx = tx - std::floor(tx);
            float fy = ty - std::floor(ty);
            size_t stride = static_cast<size_t>(tw) * bpp;
            float v00 = 0, v10 = 0, v01 = 0, v11 = 0;
            size_t off00 = static_cast<size_t>(y0) * stride + static_cast<size_t>(x0) * bpp;
            size_t off10 = static_cast<size_t>(y0) * stride + static_cast<size_t>(x1) * bpp;
            size_t off01 = static_cast<size_t>(y1) * stride + static_cast<size_t>(x0) * bpp;
            size_t off11 = static_cast<size_t>(y1) * stride + static_cast<size_t>(x1) * bpp;
            if (off00 < tex->second.bytes.size()) v00 = static_cast<float>(tex->second.bytes[off00]) / 255.0f;
            if (off10 < tex->second.bytes.size()) v10 = static_cast<float>(tex->second.bytes[off10]) / 255.0f;
            if (off01 < tex->second.bytes.size()) v01 = static_cast<float>(tex->second.bytes[off01]) / 255.0f;
            if (off11 < tex->second.bytes.size()) v11 = static_cast<float>(tex->second.bytes[off11]) / 255.0f;
            *value = static_cast<T>((v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy) +
                                     (v01 * (1.0f - fx) + v11 * fx) * fy);
        } else {
            // SampleLevel — nearest
            size_t off = static_cast<size_t>(y0) * static_cast<size_t>(tw) * bpp + static_cast<size_t>(x0) * bpp;
            float val = off < tex->second.bytes.size() ? static_cast<float>(tex->second.bytes[off]) / 255.0f : 0.0f;
            *value = static_cast<T>(val);
        }
        return true;
    }
    case kIrExpressionNodeKindGpuStructField: {
        // GpuStruct field access: evaluate instance, extract byte component.
        if (node.argument_count < 1 || node.first_argument_index == UINT32_MAX) return false;
        const auto inst_node = ir.expression_argument_indices[node.first_argument_index];
        // Evaluate the instance as a float4 (component struct)
        float components[4] = {};
        const auto& inst_expr = ir.expression_nodes[inst_node];
        // The instance is typically a TextureSample node; evaluate it directly
        if (inst_expr.kind == kIrExpressionNodeKindTextureSample ||
            inst_expr.kind == kIrExpressionNodeKindTextureSampleLevel) {
            // For CPU fallback, TextureSample returns RGBA as 4 floats
            // Evaluate by looking up texture pixels
            // Simplified: return 0 for unsupported struct field evaluation
            *value = static_cast<T>(0.0f);
            return true;
        }
        // General case: evaluate the instance expression and extract component
        float fv = 0;
        if (!evaluate_expression_node(ir, kernel, inst_node, element_index, &fv))
            return false;
        *value = static_cast<T>(fv);
        return true;
    }
    default:
        return false;
    }
}

template <size_t N>
bool evaluate_float_vector_expression_node(const ParsedIr& ir, const KernelState& kernel, uint32_t node_index,
                                           size_t element_index, FloatVectorValue<N>* value) {
    if (value == nullptr || node_index >= ir.expression_nodes.size()) {
        return false;
    }

    const auto& node = ir.expression_nodes[node_index];
    switch (node.kind) {
    case 1: {
        const auto* resource = find_resource_by_binding(ir, node.resource_binding);
        if (resource == nullptr || resource->kind != kIrResourceKindBuffer || !is_float_vector_resource(ir, *resource, N)) {
            return false;
        }

        const auto bound = kernel.buffers.find(resource->binding);
        if (bound == kernel.buffers.end()) {
            return false;
        }

        const auto buffer = g_buffers.find(bound->second);
        if (buffer == g_buffers.end() || buffer->second.stride < sizeof(FloatVectorValue<N>) ||
            element_index >= buffer->second.bytes.size() / buffer->second.stride ||
            (element_index * buffer->second.stride) > buffer->second.bytes.size() - sizeof(FloatVectorValue<N>)) {
            return false;
        }

        std::memcpy(value, buffer->second.bytes.data() + (element_index * buffer->second.stride),
                    sizeof(FloatVectorValue<N>));
        return true;
    }
    case 2: {
        float scalar = 0.0f;
        if (!try_parse_float_literal(ir, node.literal_string_id, &scalar)) {
            return false;
        }

        // C# vector-scalar operators lower as binary vector expressions with scalar child nodes.
        for (size_t i = 0; i < N; ++i) {
            value->components[i] = scalar;
        }

        return true;
    }
    case 3: {
        const auto operation = expression_operation_from_ir(node.operation);
        FloatVectorValue<N> left{};
        FloatVectorValue<N> right{};
        if (operation == 0 ||
            !evaluate_float_vector_expression_node(ir, kernel, node.left_node_index, element_index, &left) ||
            !evaluate_float_vector_expression_node(ir, kernel, node.right_node_index, element_index, &right)) {
            return false;
        }

        *value = apply_float_vector_binary_operation(left, right, operation);
        return true;
    }
    case 4: {
        const auto* symbol = get_string(ir, node.symbol_string_id);
        if (symbol == nullptr) {
            return false;
        }

        if constexpr (N == 3) {
            if (is_cross_intrinsic_symbol(*symbol)) {
                // Cross is a vector-result intrinsic, so its arguments stay in the vector evaluator.
                return try_evaluate_cross_intrinsic(ir, kernel, node, element_index, value);
            }
        }

        return false;
    }
    case 5: {
        // Push constant (byte 5)
        size_t offset = 0;
        size_t size = 0;
        if (!find_push_constant_offset(ir, node.resource_binding, &offset, &size) ||
            offset + size > kernel.push_constants.size()) {
            return false;
        }

        if (size == sizeof(FloatVectorValue<N>)) {
            std::memcpy(value, kernel.push_constants.data() + offset, sizeof(FloatVectorValue<N>));
            return true;
        }

        if (size == sizeof(float)) {
            float scalar = 0.0f;
            std::memcpy(&scalar, kernel.push_constants.data() + offset, sizeof(float));
            // Match shader scalar-to-vector splatting for expressions such as input[i] * scale.Value.
            for (size_t i = 0; i < N; ++i) {
                value->components[i] = scalar;
            }

            return true;
        }

        return false;
    }
    case 6: return false; // Comparison: not supported in vector fallback
    case 7: return false; // LocalVariable: not supported in vector fallback
    case 9: return false; // Ternary: not supported in vector fallback
    case 10: {
        // Constructor node for float vectors: evaluate each argument as a scalar float
        // and pack them into the vector result.
        // e.g., new float3(1.0f, 2.0f, 3.0f) lowers as Constructor with 3 literal children.
        if (node.argument_count != N ||
            node.first_argument_index == UINT32_MAX ||
            node.first_argument_index > ir.expression_argument_indices.size() ||
            node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
            return false;
        }
        for (uint32_t c = 0; c < N; ++c) {
            float component = 0.0f;
            const auto arg_node_index = ir.expression_argument_indices[node.first_argument_index + c];
            if (!evaluate_expression_node(ir, kernel, arg_node_index, element_index, &component)) {
                return false;
            }
            value->components[c] = component;
        }
        return true;
    }
    default:
        return false;
    }
}

template <typename T>
FeResult execute_expression_assignment_typed(const KernelState& kernel, const ParsedIr& ir,
                                             const IrExpressionAssignment& assignment,
                                             BufferState& destination_buffer, size_t copied_elements) {
    for (size_t i = 0; i < copied_elements; ++i) {
        T value{};
        if (!evaluate_expression_node(ir, kernel, assignment.root_node_index, i, &value)) {
            return fail(FE_ERROR_UNSUPPORTED, "Kernel expression fallback contains an unsupported expression node.");
        }

        std::memcpy(destination_buffer.bytes.data() + (i * destination_buffer.stride), &value, sizeof(T));
    }

    return ok();
}

template <size_t N>
FeResult execute_float_vector_expression_assignment(const KernelState& kernel, const ParsedIr& ir,
                                                    const IrExpressionAssignment& assignment,
                                                    BufferState& destination_buffer, size_t copied_elements) {
    for (size_t i = 0; i < copied_elements; ++i) {
        FloatVectorValue<N> value{};
        if (!evaluate_float_vector_expression_node(ir, kernel, assignment.root_node_index, i, &value)) {
            return fail(FE_ERROR_UNSUPPORTED,
                        "Kernel float-vector expression fallback contains an unsupported expression node.");
        }

        std::memcpy(destination_buffer.bytes.data() + (i * destination_buffer.stride), &value,
                    sizeof(FloatVectorValue<N>));
    }

    return ok();
}

template <typename T>
void execute_literal_binary(BufferState& destination, const BufferState& source, size_t copied_elements, double literal,
                            char operation) {
    auto* destination_values = reinterpret_cast<T*>(destination.bytes.data());
    const auto* source_values = reinterpret_cast<const T*>(source.bytes.data());
    const auto right = static_cast<T>(literal);
    for (size_t i = 0; i < copied_elements; ++i) {
        destination_values[i] = static_cast<T>(apply_binary_operation(source_values[i], right, operation));
    }
}

template <typename T>
void execute_buffer_binary(BufferState& destination, const BufferState& left, const BufferState& right,
                           size_t copied_elements, char operation) {
    auto* destination_values = reinterpret_cast<T*>(destination.bytes.data());
    const auto* left_values = reinterpret_cast<const T*>(left.bytes.data());
    const auto* right_values = reinterpret_cast<const T*>(right.bytes.data());
    for (size_t i = 0; i < copied_elements; ++i) {
        destination_values[i] = static_cast<T>(apply_binary_operation(left_values[i], right_values[i], operation));
    }
}

FeResult execute_texture2d_copy(const KernelState& kernel, const ParsedIr& ir, uint32_t destination_binding,
                                uint32_t source_binding, uint32_t group_x, uint32_t group_y, uint32_t group_z) {
    const auto* destination = find_resource_by_binding(ir, destination_binding);
    const auto* source = find_resource_by_binding(ir, source_binding);
    if (destination == nullptr || source == nullptr || destination->kind != kIrResourceKindTexture2D ||
        source->kind != kIrResourceKindTexture2D) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel texture copy fallback requires two 2D texture resources.");
    }

    const auto destination_texture_binding = kernel.textures.find(destination->binding);
    const auto source_texture_binding = kernel.textures.find(source->binding);
    if (destination_texture_binding == kernel.textures.end() || source_texture_binding == kernel.textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel texture copy resources are not bound.");
    }

    auto destination_texture = g_textures.find(destination_texture_binding->second);
    auto source_texture = g_textures.find(source_texture_binding->second);
    if (destination_texture == g_textures.end() || source_texture == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel texture copy contains invalid texture handles.");
    }

    if (destination_texture->second.depth != 1 || source_texture->second.depth != 1 ||
        destination_texture->second.width != source_texture->second.width ||
        destination_texture->second.height != source_texture->second.height ||
        destination_texture->second.pixel_format != source_texture->second.pixel_format) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel texture copy fallback requires matching 2D texture dimensions and format.");
    }

    const auto bytes_per_pixel = pixel_size(destination_texture->second.pixel_format);
    const auto requested_elements = static_cast<size_t>(group_x) * static_cast<size_t>(group_y) *
                                    static_cast<size_t>(group_z) * static_cast<size_t>(ir.group_x) *
                                    static_cast<size_t>(ir.group_y) * static_cast<size_t>(ir.group_z);
    const auto available_pixels = std::min(destination_texture->second.bytes.size(), source_texture->second.bytes.size()) /
                                  bytes_per_pixel;
    const auto copied_pixels = std::min(requested_elements, available_pixels);
    std::memcpy(destination_texture->second.bytes.data(), source_texture->second.bytes.data(),
                copied_pixels * bytes_per_pixel);
    return ok();
}

FeResult execute_fallback_assignment(const KernelState& kernel, const ParsedIr& ir,
                                     const FallbackAssignment& assignment, uint32_t group_x, uint32_t group_y,
                                     uint32_t group_z) {
    const auto* destination = find_resource_by_binding(ir, assignment.destination_binding);
    const auto* left = find_resource_by_binding(ir, assignment.left_binding);
    const IrResource* right = nullptr;
    if (assignment.kind == FallbackExpressionKind::BufferBinaryBuffer) {
        right = find_resource_by_binding(ir, assignment.right_binding);
    }

    if (destination != nullptr && left != nullptr && assignment.kind == FallbackExpressionKind::Copy &&
        destination->kind == kIrResourceKindTexture2D && left->kind == kIrResourceKindTexture2D) {
        return execute_texture2d_copy(kernel, ir, assignment.destination_binding, assignment.left_binding, group_x,
                                      group_y, group_z);
    }

    if (destination == nullptr || left == nullptr || destination->kind != kIrResourceKindBuffer ||
        left->kind != kIrResourceKindBuffer ||
        (assignment.kind == FallbackExpressionKind::BufferBinaryBuffer &&
         (right == nullptr || right->kind != kIrResourceKindBuffer))) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Kernel dispatch fallback only supports elementwise buffer assignments or 2D texture copies.");
    }

    const auto destination_binding = kernel.buffers.find(destination->binding);
    const auto left_binding = kernel.buffers.find(left->binding);
    auto right_binding = kernel.buffers.end();
    if (right != nullptr) {
        right_binding = kernel.buffers.find(right->binding);
    }

    if (destination_binding == kernel.buffers.end() || left_binding == kernel.buffers.end() ||
        (right != nullptr && right_binding == kernel.buffers.end())) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel buffer assignment resources are not bound.");
    }

    auto destination_buffer = g_buffers.find(destination_binding->second);
    auto left_buffer = g_buffers.find(left_binding->second);
    auto right_buffer = right != nullptr ? g_buffers.find(right_binding->second) : g_buffers.end();
    if (destination_buffer == g_buffers.end() || left_buffer == g_buffers.end() ||
        (right != nullptr && right_buffer == g_buffers.end())) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel buffer assignment contains invalid buffer handles.");
    }

    const auto element_stride = destination_buffer->second.stride;
    if (element_stride == 0 || element_stride != left_buffer->second.stride ||
        (right != nullptr && element_stride != right_buffer->second.stride)) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel dispatch fallback requires matching non-zero buffer strides.");
    }

    const auto requested_elements = static_cast<size_t>(group_x) * static_cast<size_t>(group_y) *
                                    static_cast<size_t>(group_z) * static_cast<size_t>(ir.group_x) *
                                    static_cast<size_t>(ir.group_y) * static_cast<size_t>(ir.group_z);
    auto available_bytes = std::min(destination_buffer->second.bytes.size(), left_buffer->second.bytes.size());
    if (right != nullptr) {
        available_bytes = std::min(available_bytes, right_buffer->second.bytes.size());
    }

    const auto available_elements = available_bytes / element_stride;
    const auto copied_elements = std::min(requested_elements, available_elements);

    if (assignment.kind == FallbackExpressionKind::Copy) {
        std::memcpy(destination_buffer->second.bytes.data(), left_buffer->second.bytes.data(),
                    copied_elements * element_stride);
        return ok();
    }

    if (is_float_resource(ir, *destination) && is_float_resource(ir, *left) &&
        (right == nullptr || is_float_resource(ir, *right)) && element_stride == sizeof(float)) {
        if (assignment.kind == FallbackExpressionKind::BufferBinaryLiteral) {
            execute_literal_binary<float>(destination_buffer->second, left_buffer->second, copied_elements,
                                          assignment.literal_value, assignment.operation);
        } else {
            execute_buffer_binary<float>(destination_buffer->second, left_buffer->second, right_buffer->second,
                                         copied_elements, assignment.operation);
        }

        return ok();
    }

    if (is_int_resource(ir, *destination) && is_int_resource(ir, *left) &&
        (right == nullptr || is_int_resource(ir, *right)) && element_stride == sizeof(int32_t)) {
        if (assignment.kind == FallbackExpressionKind::BufferBinaryLiteral) {
            execute_literal_binary<int32_t>(destination_buffer->second, left_buffer->second, copied_elements,
                                            assignment.literal_value, assignment.operation);
        } else {
            execute_buffer_binary<int32_t>(destination_buffer->second, left_buffer->second, right_buffer->second,
                                           copied_elements, assignment.operation);
        }

        return ok();
    }

    return fail(FE_ERROR_UNSUPPORTED, "Kernel dispatch fallback arithmetic currently supports int and float buffer elements.");
}

FeResult execute_expression_assignment(const KernelState& kernel, const ParsedIr& ir,
                                       const IrExpressionAssignment& assignment, uint32_t group_x, uint32_t group_y,
                                       uint32_t group_z) {
    const auto* destination = find_resource_by_binding(ir, assignment.destination_binding);
    const auto* root = assignment.root_node_index < ir.expression_nodes.size()
                           ? &ir.expression_nodes[assignment.root_node_index]
                           : nullptr;
    if (root == nullptr) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel expression assignment root node is invalid.");
    }

    if (destination != nullptr && destination->kind == kIrResourceKindTexture2D && root->kind == 1) {
        return execute_texture2d_copy(kernel, ir, assignment.destination_binding, root->resource_binding, group_x,
                                      group_y, group_z);
    }

    if (destination == nullptr || destination->kind != kIrResourceKindBuffer) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel expression fallback only supports buffer destinations.");
    }

    const auto destination_binding = kernel.buffers.find(destination->binding);
    if (destination_binding == kernel.buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel expression destination buffer is not bound.");
    }

    auto destination_buffer = g_buffers.find(destination_binding->second);
    if (destination_buffer == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Kernel expression destination buffer handle is invalid.");
    }

    const auto element_stride = destination_buffer->second.stride;
    if (element_stride == 0) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel expression fallback requires a non-zero buffer stride.");
    }

    const auto requested_elements = static_cast<size_t>(group_x) * static_cast<size_t>(group_y) *
                                    static_cast<size_t>(group_z) * static_cast<size_t>(ir.group_x) *
                                    static_cast<size_t>(ir.group_y) * static_cast<size_t>(ir.group_z);
    const auto available_elements = destination_buffer->second.bytes.size() / element_stride;
    const auto copied_elements = std::min(requested_elements, available_elements);
    if (element_stride == sizeof(float) && is_float_resource(ir, *destination) &&
        (is_float_type(ir, root->type_string_id) || root->type_string_id == UINT32_MAX)) {
        return execute_expression_assignment_typed<float>(kernel, ir, assignment, destination_buffer->second,
                                                          copied_elements);
    }

    if (element_stride == sizeof(int32_t) && is_int_resource(ir, *destination) &&
        (is_int_type(ir, root->type_string_id) || root->type_string_id == UINT32_MAX)) {
        return execute_expression_assignment_typed<int32_t>(kernel, ir, assignment, destination_buffer->second,
                                                            copied_elements);
    }

    if (element_stride == float_vector_buffer_stride(2) && is_float_vector_resource(ir, *destination, 2) &&
        (is_float_vector_type(ir, root->type_string_id, 2) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<2>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    if (element_stride == float_vector_buffer_stride(3) && is_float_vector_resource(ir, *destination, 3) &&
        (is_float_vector_type(ir, root->type_string_id, 3) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<3>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    if (element_stride == float_vector_buffer_stride(4) && is_float_vector_resource(ir, *destination, 4) &&
        (is_float_vector_type(ir, root->type_string_id, 4) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<4>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    return fail(FE_ERROR_UNSUPPORTED,
                "Kernel expression fallback currently supports int, float, float2, float3, and float4 buffer elements.");
}

using ModuleResourceMap = std::unordered_map<uint32_t, GPU::IR::ResourceId>;

// Map Feather IR shader builtin kind to the corresponding ModuleBuilder thread/group/local ID value.
GPU::IR::ValueId build_shader_builtin(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder, uint8_t builtin_kind) {
    switch (builtin_kind) {
    case 1: return builder.ThreadIndexX();  // ThreadIds.X
    case 2: return builder.ThreadIndexY();  // ThreadIds.Y
    case 3: return builder.ThreadIndexZ();  // ThreadIds.Z
    default: return GPU::IR::InvalidValueId;
    }
}

GPU::IR::ValueId build_easygpu_module_index(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                            uint32_t index_string_id) {
    const auto* index = get_string(ir, index_string_id);
    if (index == nullptr || index->empty()) {
        return GPU::IR::InvalidValueId;
    }

    // Feather IR stores the semantic C# index symbol; the EasyGPU module records the lowered thread index.
    // Default: map to the compute thread index.
    // Local variables declared with ThreadIds as initializer act as thread ID aliases.
    // Future: distinguish local loop counters from thread ID proxies.
    return builder.ThreadIndexX();
}

GPU::IR::ValueId build_easygpu_module_resource_access(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                      const ModuleResourceMap& resources, uint32_t resource_binding,
                                                      uint32_t index_string_id) {
    const auto* resource = find_resource_by_binding(ir, resource_binding);
    if (resource == nullptr || resource->kind != kIrResourceKindBuffer) {
        return GPU::IR::InvalidValueId;
    }

    const auto mapped = resources.find(resource_binding);
    if (mapped == resources.end()) {
        return GPU::IR::InvalidValueId;
    }

    const auto index = build_easygpu_module_index(ir, builder, index_string_id);
    if (index == GPU::IR::InvalidValueId) {
        return GPU::IR::InvalidValueId;
    }
    return builder.ResourceElement(mapped->second, index);
}

GPU::IR::ValueId build_easygpu_module_push_constant_access(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                           const ModuleResourceMap& resources,
                                                           uint32_t resource_binding) {
    const auto* resource = find_resource_by_binding(ir, resource_binding);
    if (resource == nullptr || resource->kind != kIrResourceKindPushConstant) {
        return GPU::IR::InvalidValueId;
    }

    const auto mapped = resources.find(resource_binding);
    if (mapped == resources.end()) {
        return GPU::IR::InvalidValueId;
    }

    return builder.PushConstant(mapped->second);
}

std::string module_value_to_glsl(const GPU::IR::Module& module, GPU::IR::ValueId id,
                                 const std::unordered_map<uint32_t, std::string>& resource_names);

GPU::IR::ValueId build_easygpu_module_expression(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                 const ModuleResourceMap& resources, uint32_t node_index) {
    if (node_index >= ir.expression_nodes.size()) {
        return GPU::IR::InvalidValueId;
    }

    const auto& node = ir.expression_nodes[node_index];
    switch (node.kind) {
    case 1:
        return build_easygpu_module_resource_access(ir, builder, resources, node.resource_binding, node.index_string_id);
    case 2: {
        const auto* literal = get_string(ir, node.literal_string_id);
        const auto* type = get_string(ir, node.type_string_id);
        const auto module_type = type == nullptr ? GPU::IR::Type::Float() : easygpu_module_type(*type);
        if (literal == nullptr || !module_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        return builder.Literal(module_type, *literal);
    }
    case 3: {
        auto left = build_easygpu_module_expression(ir, builder, resources, node.left_node_index);
        auto right = build_easygpu_module_expression(ir, builder, resources, node.right_node_index);
        GPU::IR::BinaryOp op{};
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId ||
            !easygpu_expression_binary_op(node.operation, &op)) {
            return GPU::IR::InvalidValueId;
        }

        return builder.Binary(op, left, right);
    }
    case 4: {
        const auto* symbol = get_string(ir, node.symbol_string_id);
        if (symbol == nullptr || node.first_argument_index == UINT32_MAX ||
            node.first_argument_index > ir.expression_argument_indices.size() ||
            node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
            return GPU::IR::InvalidValueId;
        }

        auto intrinsic = easygpu_intrinsic_name(*symbol);
        if (intrinsic.empty()) {
            return GPU::IR::InvalidValueId;
        }

        const auto* type = get_string(ir, node.type_string_id);
        const auto result_type = type == nullptr ? GPU::IR::Type::Float() : easygpu_module_type(*type);
        if (!result_type.IsValid()) {
            return GPU::IR::InvalidValueId;
        }

        std::vector<GPU::IR::ValueId> arguments;
        arguments.reserve(node.argument_count);
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            auto argument = build_easygpu_module_expression(
                ir, builder, resources, ir.expression_argument_indices[node.first_argument_index + i]);
            if (argument == GPU::IR::InvalidValueId) {
                return GPU::IR::InvalidValueId;
            }

            arguments.push_back(argument);
        }
        return builder.Intrinsic(std::move(intrinsic), result_type, arguments);
    }
    case kIrExpressionNodeKindComparison: {
        auto left = build_easygpu_module_expression(ir, builder, resources, node.left_node_index);
        auto right = build_easygpu_module_expression(ir, builder, resources, node.right_node_index);
        GPU::IR::CompareOp op{};
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId ||
            !easygpu_compare_op(node.operation, &op)) {
            return GPU::IR::InvalidValueId;
        }
        return builder.Compare(op, left, right);
    }
    case kIrExpressionNodeKindLocalVariable: {
        const auto* name = get_string(ir, node.symbol_string_id);
        if (name == nullptr || name->empty()) return GPU::IR::InvalidValueId;
        const auto* type_str = get_string(ir, node.type_string_id);
        const auto var_type = type_str != nullptr ? easygpu_module_type(*type_str) : GPU::IR::Type::Int();
        return var_type.IsValid() ? builder.LocalVariable(var_type, *name)
                                  : builder.LocalVariable(GPU::IR::Type::Int(), *name);
    }
    case kIrExpressionNodeKindPushConstant:
        return build_easygpu_module_push_constant_access(ir, builder, resources, node.resource_binding);
    case kIrExpressionNodeKindShaderBuiltin:
        return build_shader_builtin(ir, builder, node.operation);
    case kIrExpressionNodeKindConstructor: {
        const auto* type_name = get_string(ir, node.type_string_id);
        if (type_name == nullptr) return GPU::IR::InvalidValueId;
        auto glsl_ctor = easygpu_intrinsic_name_for_type(*type_name);
        if (glsl_ctor.empty()) return GPU::IR::InvalidValueId;
        auto result_type = easygpu_module_type(*type_name);
        if (!result_type.IsValid()) return GPU::IR::InvalidValueId;
        if (node.first_argument_index == UINT32_MAX || node.argument_count == 0) return GPU::IR::InvalidValueId;
        std::vector<GPU::IR::ValueId> args;
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            auto arg = build_easygpu_module_expression(ir, builder, resources,
                ir.expression_argument_indices[node.first_argument_index + i]);
            if (arg == GPU::IR::InvalidValueId) return GPU::IR::InvalidValueId;
            args.push_back(arg);
        }
        return builder.Intrinsic(std::move(glsl_ctor), result_type, args);
    }
    case kIrExpressionNodeKindTernary: {
        auto cond = build_easygpu_module_expression(ir, builder, resources, node.left_node_index);
        auto tv = build_easygpu_module_expression(ir, builder, resources, node.right_node_index);
        if (node.first_argument_index == UINT32_MAX || node.argument_count < 1)
            return GPU::IR::InvalidValueId;
        auto fv = build_easygpu_module_expression(ir, builder, resources,
            ir.expression_argument_indices[node.first_argument_index]);
        if (cond == GPU::IR::InvalidValueId || tv == GPU::IR::InvalidValueId || fv == GPU::IR::InvalidValueId)
            return GPU::IR::InvalidValueId;
        return builder.Ternary(cond, tv, fv);
    }
    case kIrExpressionNodeKindCallableCall: {
        // Section 1-6 compatibility expression trees do not carry callable function
        // tables. Canonical callable calls are lowered from section 7 by the typed IR
        // lowerer, so reject this path instead of proving callables through legacy data.
        return GPU::IR::InvalidValueId;
    }
    case kIrExpressionNodeKindTextureSample:
    case kIrExpressionNodeKindTextureSampleLevel: {
        // Texture sample / sampleLevel: use the texture resource's GLSL name.
        const auto* type_name = get_string(ir, node.type_string_id);
        auto result_type = type_name != nullptr ? easygpu_module_type(*type_name) : GPU::IR::Type::Float4();
        if (!result_type.IsValid()) return GPU::IR::InvalidValueId;

        if (node.first_argument_index == UINT32_MAX || node.argument_count == 0)
            return GPU::IR::InvalidValueId;

        // Use the texture resource binding (not sampler) for the GLSL sampler2D name.
        auto tex_it = resources.find(node.resource_binding);
        if (tex_it == resources.end()) return GPU::IR::InvalidValueId;

        auto uv = build_easygpu_module_expression(ir, builder, resources,
            ir.expression_argument_indices[node.first_argument_index]);
        if (uv == GPU::IR::InvalidValueId) return GPU::IR::InvalidValueId;

        auto& mod = builder.GetModule();
        std::string tex_glsl;
        for (const auto& r : mod.resources) {
            if (r.id == tex_it->second) { tex_glsl = r.name; break; }
        }
        if (tex_glsl.empty()) return GPU::IR::InvalidValueId;

        std::string uv_glsl = module_value_to_glsl(mod, uv, {});
        if (node.kind == kIrExpressionNodeKindTextureSample) {
            return builder.Literal(result_type, "texture(" + tex_glsl + ", " + uv_glsl + ")");
        } else {
            if (node.argument_count < 2) return GPU::IR::InvalidValueId;
            auto lod = build_easygpu_module_expression(ir, builder, resources,
                ir.expression_argument_indices[node.first_argument_index + 1]);
            if (lod == GPU::IR::InvalidValueId) return GPU::IR::InvalidValueId;
            std::string lod_glsl = module_value_to_glsl(mod, lod, {});
            return builder.Literal(result_type, "textureLod(" + tex_glsl + ", " + uv_glsl + ", " + lod_glsl + ")");
        }
    }
    case kIrExpressionNodeKindGpuStructField: {
        // Emit GLSL swizzle: .x, .y, .z, .w based on field index.
        if (node.argument_count < 1 || node.first_argument_index == UINT32_MAX)
            return GPU::IR::InvalidValueId;
        auto inst = build_easygpu_module_expression(ir, builder, resources,
            ir.expression_argument_indices[node.first_argument_index]);
        if (inst == GPU::IR::InvalidValueId) return GPU::IR::InvalidValueId;
        auto& mod = builder.GetModule();
        std::string inst_glsl = module_value_to_glsl(mod, inst, {});
        static const char* swiz[] = {".x", ".y", ".z", ".w"};
        int idx = static_cast<int>(node.operation);
        if (idx < 0 || idx > 3) return GPU::IR::InvalidValueId;
        const auto* type_name = get_string(ir, node.type_string_id);
        auto result_type = type_name != nullptr ? easygpu_module_type(*type_name) : GPU::IR::Type::Float();
        return builder.Literal(result_type, inst_glsl + swiz[idx]);
    }
    default:
        return GPU::IR::InvalidValueId;
    }
}

// Build an expression node from the control flow expression pool instead of the assignment pool.
GPU::IR::ValueId build_easygpu_module_cf_expression(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                    const ModuleResourceMap& resources, uint32_t node_index) {
    if (node_index >= ir.control_flow_nodes.size()) return GPU::IR::InvalidValueId;

    const auto& node = ir.control_flow_nodes[node_index];
    switch (node.kind) {
    case 1:
        return build_easygpu_module_resource_access(ir, builder, resources, node.resource_binding, node.index_string_id);
    case 2: {
        const auto* literal = get_string(ir, node.literal_string_id);
        const auto* type = get_string(ir, node.type_string_id);
        const auto module_type = type == nullptr ? GPU::IR::Type::Float() : easygpu_module_type(*type);
        if (literal == nullptr || !module_type.IsValid()) return GPU::IR::InvalidValueId;
        return builder.Literal(module_type, *literal);
    }
    case 3: {
        auto left = build_easygpu_module_cf_expression(ir, builder, resources, node.left_node_index);
        auto right = build_easygpu_module_cf_expression(ir, builder, resources, node.right_node_index);
        GPU::IR::BinaryOp op{};
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId ||
            !easygpu_expression_binary_op(node.operation, &op)) return GPU::IR::InvalidValueId;
        return builder.Binary(op, left, right);
    }
    case kIrExpressionNodeKindComparison: {
        auto left = build_easygpu_module_cf_expression(ir, builder, resources, node.left_node_index);
        auto right = build_easygpu_module_cf_expression(ir, builder, resources, node.right_node_index);
        GPU::IR::CompareOp op{};
        if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId ||
            !easygpu_compare_op(node.operation, &op)) return GPU::IR::InvalidValueId;
        return builder.Compare(op, left, right);
    }
    case 4: {
        const auto* symbol = get_string(ir, node.symbol_string_id);
        if (symbol == nullptr || node.first_argument_index == UINT32_MAX ||
            node.first_argument_index > ir.control_flow_argument_indices.size() ||
            node.argument_count > ir.control_flow_argument_indices.size() - node.first_argument_index)
            return GPU::IR::InvalidValueId;
        auto intrinsic = easygpu_intrinsic_name(*symbol);
        if (intrinsic.empty()) return GPU::IR::InvalidValueId;
        const auto* type = get_string(ir, node.type_string_id);
        const auto result_type = type == nullptr ? GPU::IR::Type::Float() : easygpu_module_type(*type);
        if (!result_type.IsValid()) return GPU::IR::InvalidValueId;
        std::vector<GPU::IR::ValueId> arguments;
        arguments.reserve(node.argument_count);
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            auto argument = build_easygpu_module_cf_expression(
                ir, builder, resources, ir.control_flow_argument_indices[node.first_argument_index + i]);
            if (argument == GPU::IR::InvalidValueId) return GPU::IR::InvalidValueId;
            arguments.push_back(argument);
        }
        return builder.Intrinsic(std::move(intrinsic), result_type, arguments);
    }
    case kIrExpressionNodeKindLocalVariable: {
        const auto* name = get_string(ir, node.symbol_string_id);
        if (name == nullptr || name->empty()) return GPU::IR::InvalidValueId;
        const auto* type_str = get_string(ir, node.type_string_id);
        const auto var_type = type_str != nullptr ? easygpu_module_type(*type_str) : GPU::IR::Type::Int();
        return var_type.IsValid() ? builder.LocalVariable(var_type, *name)
                                  : builder.LocalVariable(GPU::IR::Type::Int(), *name);
    }
    case kIrExpressionNodeKindPushConstant:
        return build_easygpu_module_push_constant_access(ir, builder, resources, node.resource_binding);
    default:
        return GPU::IR::InvalidValueId;
    }
}

// Build an expression from the compound assignment expression pool.
GPU::IR::ValueId build_compound_expression(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                            const ModuleResourceMap& resources, uint32_t node_index) {
    if (node_index >= ir.compound_assignment_nodes.size()) return GPU::IR::InvalidValueId;
    const auto& node = ir.compound_assignment_nodes[node_index];
    switch (node.kind) {
    case 1: return build_easygpu_module_resource_access(ir, builder, resources, node.resource_binding, node.index_string_id);
    case 2: {
        const auto* lit = get_string(ir, node.literal_string_id);
        const auto* type = get_string(ir, node.type_string_id);
        const auto mod_type = type == nullptr ? GPU::IR::Type::Float() : easygpu_module_type(*type);
        return (lit != nullptr && mod_type.IsValid()) ? builder.Literal(mod_type, *lit) : GPU::IR::InvalidValueId;
    }
    case 3: {
        auto left = build_compound_expression(ir, builder, resources, node.left_node_index);
        auto right = build_compound_expression(ir, builder, resources, node.right_node_index);
        GPU::IR::BinaryOp op{};
        return (left != GPU::IR::InvalidValueId && right != GPU::IR::InvalidValueId && easygpu_expression_binary_op(node.operation, &op))
                   ? builder.Binary(op, left, right) : GPU::IR::InvalidValueId;
    }
    case 5: return build_easygpu_module_push_constant_access(ir, builder, resources, node.resource_binding);
    case 7: {
        const auto* name = get_string(ir, node.symbol_string_id);
        return (name != nullptr && !name->empty()) ? builder.LocalVariable(GPU::IR::Type::Float(), *name) : GPU::IR::InvalidValueId;
    }
    default: return GPU::IR::InvalidValueId;
    }
}

// Convert a ModuleBuilder ValueId to a GLSL expression string by walking the value record tree.
// Used to build proper conditions for if/for/while instead of hardcoded "true".
std::string module_value_to_glsl(const GPU::IR::Module& module, GPU::IR::ValueId id,
                                  const std::unordered_map<uint32_t, std::string>& resource_names) {
    if (id >= module.values.size()) return "true";
    const auto& v = module.values[id];
    switch (v.kind) {
    case GPU::IR::ValueRecord::Kind::ThreadIndexX: return "int(gl_GlobalInvocationID.x)";
    case GPU::IR::ValueRecord::Kind::ThreadIndexY: return "int(gl_GlobalInvocationID.y)";
    case GPU::IR::ValueRecord::Kind::ThreadIndexZ: return "int(gl_GlobalInvocationID.z)";
    case GPU::IR::ValueRecord::Kind::Literal: return v.literal;
    case GPU::IR::ValueRecord::Kind::LocalVar: return v.localName;
    case GPU::IR::ValueRecord::Kind::Binary: {
        auto left = module_value_to_glsl(module, v.left, resource_names);
        auto right = module_value_to_glsl(module, v.right, resource_names);
        const char* op = v.binaryOp == GPU::IR::BinaryOp::Add ? "+" :
                         v.binaryOp == GPU::IR::BinaryOp::Sub ? "-" :
                         v.binaryOp == GPU::IR::BinaryOp::Mul ? "*" : "/";
        return "(" + left + " " + op + " " + right + ")";
    }
    case GPU::IR::ValueRecord::Kind::Compare: {
        auto left = module_value_to_glsl(module, v.left, resource_names);
        auto right = module_value_to_glsl(module, v.right, resource_names);
        const char* op = v.compareOp == GPU::IR::CompareOp::Equal      ? "=="
                       : v.compareOp == GPU::IR::CompareOp::NotEqual    ? "!="
                       : v.compareOp == GPU::IR::CompareOp::Less        ? "<"
                       : v.compareOp == GPU::IR::CompareOp::LessEqual   ? "<="
                       : v.compareOp == GPU::IR::CompareOp::Greater     ? ">"
                       : v.compareOp == GPU::IR::CompareOp::GreaterEqual ? ">=" : "==";
        return "(" + left + " " + op + " " + right + ")";
    }
    case GPU::IR::ValueRecord::Kind::Intrinsic: {
        std::string args;
        for (size_t i = 0; i < v.arguments.size(); ++i) {
            if (i > 0) args += ", ";
            args += module_value_to_glsl(module, v.arguments[i], resource_names);
        }
        return v.intrinsic + "(" + args + ")";
    }
    case GPU::IR::ValueRecord::Kind::ResourceElement: {
        auto it = resource_names.find(v.resource);
        auto buf_name = it != resource_names.end() ? it->second : "unknown";
        auto idx = module_value_to_glsl(module, v.index, resource_names);
        return buf_name + "[" + idx + "]";
    }
    case GPU::IR::ValueRecord::Kind::PushConstant: {
        auto it = resource_names.find(v.resource);
        return it != resource_names.end() ? it->second : "unknown_pc";
    }
    default: return "true";
    }
}

// Lookup the root node index for a control flow condition expression.
uint32_t find_cf_condition_node(const ParsedIr& ir, uint32_t instruction_index, uint8_t role) {
    for (const auto& cf : ir.control_flow_expressions) {
        if (cf.instruction_index == instruction_index && cf.role == role) return cf.root_node_index;
    }
    return UINT32_MAX;
}

// Build a boolean condition value for ModuleBuilder control flow statements.
// Returns InvalidValueId if no expression is available for this instruction.
GPU::IR::ValueId build_easygpu_module_condition(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                 const ModuleResourceMap& resources, uint32_t instruction_index,
                                                 uint8_t role) {
    const auto root = find_cf_condition_node(ir, instruction_index, role);
    if (root == UINT32_MAX) return GPU::IR::InvalidValueId;
    return build_easygpu_module_cf_expression(ir, builder, resources, root);
}

bool register_easygpu_module_resources(const ParsedIr& ir, const KernelState& kernel,
                                       GPU::IR::ModuleBuilder& builder, ModuleResourceMap* resources) {
    if (resources == nullptr) {
        return false;
    }

    for (const auto& resource : ir.resources) {
        const auto* type = get_string(ir, resource.element_type_string_id);
        if (type == nullptr) {
            return false;
        }

        GPU::IR::ResourceId id = GPU::IR::InvalidResourceId;
        if (resource.kind == kIrResourceKindBuffer) {
            const auto module_type = easygpu_module_type(*type);
            if (!module_type.IsValid()) {
                return false;
            }
            id = builder.AddBuffer(
                resource.binding, module_type, easygpu_resource_access(resource.access), easygpu_buffer_name(resource));
        } else if (resource.kind == kIrResourceKindPushConstant) {
            const auto module_type = easygpu_module_type(*type);
            if (!module_type.IsValid()) {
                return false;
            }

            size_t offset = 0;
            size_t size = 0;
            if (!find_push_constant_offset(ir, resource.binding, &offset, &size)) {
                return false;
            }

            const auto alignment = push_constant_type_alignment(ir, resource);
            auto* data = offset + size <= kernel.push_constants.size()
                             ? const_cast<unsigned char*>(kernel.push_constants.data() + offset)
                             : nullptr;
            id = builder.AddPushConstant(
                resource.binding, module_type, easygpu_push_constant_name(resource), data, size, alignment);
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto is_texture3d = resource.kind == kIrResourceKindTexture3D;
            // Texture imageLoad/imageStore always return/accept vec4 for normalized formats.
            // Use Float4 as the shader element type regardless of the C# pixel type.
            const auto tex_module_type = GPU::IR::Type::Float4();

            // Look up the bound texture for format and dimensions, or use safe defaults for shader inspection.
            uint32_t tex_width = 1;
            uint32_t tex_height = 1;
            uint32_t tex_depth = 1;
            uint32_t tex_format = 3; // RGBA8
            bool tex_sampled = (resource.access == 4); // SampledTexture2D

            const auto bound = kernel.textures.find(resource.binding);
            if (bound != kernel.textures.end()) {
                const auto texture = g_textures.find(bound->second);
                if (texture != g_textures.end()) {
                    tex_width = texture->second.width;
                    tex_height = texture->second.height;
                    tex_depth = texture->second.depth;
                    tex_format = texture->second.pixel_format;
                }
            }

            GPU::Runtime::PixelFormat runtime_format = GPU::Runtime::PixelFormat::RGBA8;
            easygpu_runtime_pixel_format(tex_format, &runtime_format);

            if (is_texture3d) {
                id = builder.AddTexture3D(resource.binding, tex_module_type, easygpu_resource_access(resource.access),
                                          "te_" + std::to_string(resource.binding), runtime_format, tex_width,
                                          tex_height, tex_depth, tex_sampled);
            } else {
                id = builder.AddTexture2D(resource.binding, tex_module_type, easygpu_resource_access(resource.access),
                                          "te_" + std::to_string(resource.binding), runtime_format, tex_width,
                                          tex_height, tex_sampled);
            }
        } else if (resource.kind == 3 /* kIrResourceKindSampler */) {
            // Sampler resources: register as a placeholder; actual sampling uses
            // the sampled texture's combined sampler2D name in GLSL.
            id = static_cast<GPU::IR::ResourceId>(resource.binding + 1000);
        } else {
            return false;
        }

        resources->emplace(resource.binding, id);
    }

    return true;
}

bool build_typed_ir_lowering_inputs(const ParsedIr& ir, const KernelState& kernel,
                                    Feather::TypedIR::LoweringInputs* inputs) {
    if (inputs == nullptr) {
        return false;
    }

    inputs->shader_kind = ir.shader_kind;
    inputs->group_x = ir.group_x;
    inputs->group_y = ir.group_y;
    inputs->group_z = ir.group_z;
    inputs->bounds_check = kernel.bounds_check;
    inputs->logical_x = kernel.logical_x;
    inputs->logical_y = kernel.logical_y == 0 ? 1 : kernel.logical_y;
    inputs->logical_z = kernel.logical_z == 0 ? 1 : kernel.logical_z;
    inputs->logical_x_data = const_cast<int32_t*>(&kernel.logical_x);
    inputs->logical_y_data = const_cast<int32_t*>(&kernel.logical_y);
    inputs->logical_z_data = const_cast<int32_t*>(&kernel.logical_z);
    inputs->resources.clear();
    inputs->push_constants.clear();

    for (const auto& resource : ir.resources) {
        const auto* name = get_string(ir, resource.name_string_id);
        const auto* element_type = get_string(ir, resource.element_type_string_id);
        if (name == nullptr || element_type == nullptr) {
            return false;
        }

        Feather::TypedIR::ResourceInfo resource_info;
        resource_info.binding = resource.binding;
        resource_info.kind = resource.kind;
        resource_info.access = resource.access;
        resource_info.name = *name;
        resource_info.element_type = *element_type;

        if (resource.kind == kIrResourceKindPushConstant) {
            size_t offset = 0;
            size_t size = 0;
            if (!find_push_constant_offset(ir, resource.binding, &offset, &size)) {
                return false;
            }

            Feather::TypedIR::PushConstantInfo push_constant;
            push_constant.binding = resource.binding;
            push_constant.size = size;
            push_constant.alignment = push_constant_type_alignment(ir, resource);
            push_constant.data = offset + size <= kernel.push_constants.size()
                                     ? const_cast<unsigned char*>(kernel.push_constants.data() + offset)
                                     : nullptr;
            inputs->push_constants.push_back(push_constant);
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            resource_info.sampled = resource.access == 4;
            resource_info.width = 1;
            resource_info.height = 1;
            resource_info.depth = 1;
            resource_info.texture_format = GPU::Runtime::PixelFormat::RGBA8;

            const auto bound = kernel.textures.find(resource.binding);
            if (bound != kernel.textures.end()) {
                const auto texture = g_textures.find(bound->second);
                if (texture != g_textures.end()) {
                    resource_info.width = texture->second.width;
                    resource_info.height = texture->second.height;
                    resource_info.depth = texture->second.depth;
                    if (!easygpu_runtime_pixel_format(texture->second.pixel_format, &resource_info.texture_format)) {
                        return false;
                    }
                }
            }
        }

        inputs->resources.push_back(std::move(resource_info));
    }

    return true;
}

bool has_typed_section7_semantics(const KernelState& kernel) {
    ParsedIr ir;
    return parse_feather_ir(kernel.ir, &ir) && ir.has_section7;
}

std::unique_ptr<GPU::IR::Module> try_build_typed_easygpu_module(const KernelState& kernel,
                                                                bool enable_fused_multiply_add,
                                                                std::string* error = nullptr) {
    if (error != nullptr) {
        error->clear();
    }

    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir) || !ir.has_section7) {
        if (error != nullptr) {
            *error = "Kernel IR does not contain a valid section 7 typed IR payload.";
        }

        return nullptr;
    }

    Feather::TypedIR::LoweringInputs inputs;
    if (!build_typed_ir_lowering_inputs(ir, kernel, &inputs)) {
        if (error != nullptr) {
            *error = "Section 7 typed IR resources could not be matched to bound native resources.";
        }

        return nullptr;
    }
    inputs.enable_fused_multiply_add = enable_fused_multiply_add;

    auto module = Feather::TypedIR::TryLowerToEasyGpuModule(ir.typed_module, inputs, error);
    if (module == nullptr && error != nullptr && error->empty()) {
        *error = "Section 7 typed IR lowerer rejected the module before EasyGPU dispatch.";
    }

    return module;
}

bool build_easygpu_module_texture_element_access(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                 const ModuleResourceMap& resources, uint32_t resource_binding) {
    const auto mapped = resources.find(resource_binding);
    if (mapped == resources.end()) {
        return false;
    }

    // Texture copy uses the current thread index: x = ThreadIds.X, y = ThreadIds.Y
    const auto x = builder.ThreadIndexX();
    const auto y = builder.ThreadIndexY();
    if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId) {
        return false;
    }

    // Create the texture element access value.
    const auto element = builder.TextureElement(mapped->second, x, y);
    return element != GPU::IR::InvalidValueId;
}

bool build_easygpu_module_expression_assignment(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                const ModuleResourceMap& resources,
                                                const IrExpressionAssignment& assignment) {
    const auto* destination = find_resource_by_binding(ir, assignment.destination_binding);
    if (destination == nullptr) {
        return false;
    }

    // Texture-to-texture copy through imageLoad/imageStore.
    if (destination->kind == kIrResourceKindTexture2D) {
        if (assignment.root_node_index >= ir.expression_nodes.size()) {
            return false;
        }
        const auto& root = ir.expression_nodes[assignment.root_node_index];
        if (root.kind != 1) {
            return false;
        }

        const auto dst_mapped = resources.find(assignment.destination_binding);
        const auto src_mapped = resources.find(root.resource_binding);
        if (dst_mapped == resources.end() || src_mapped == resources.end()) {
            return false;
        }

        const auto x = builder.ThreadIndexX();
        const auto y = builder.ThreadIndexY();
        if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto dst = builder.TextureElement(dst_mapped->second, x, y);
        const auto src = builder.TextureElement(src_mapped->second, x, y);
        if (dst == GPU::IR::InvalidValueId || src == GPU::IR::InvalidValueId) {
            return false;
        }

        builder.Store(dst, src);
        return true;
    }

    if (destination->kind != kIrResourceKindBuffer) {
        return false;
    }

    auto left = build_easygpu_module_resource_access(
        ir, builder, resources, assignment.destination_binding, assignment.index_string_id);
    auto right = build_easygpu_module_expression(ir, builder, resources, assignment.root_node_index);
    if (left == GPU::IR::InvalidValueId || right == GPU::IR::InvalidValueId) {
        return false;
    }

    builder.Store(left, right);
    return true;
}

bool build_easygpu_module_structured_assignment(const ParsedIr& ir, GPU::IR::ModuleBuilder& builder,
                                                const ModuleResourceMap& resources,
                                                const IrElementwiseAssignment& assignment) {
    const auto* destination = find_resource_by_binding(ir, assignment.destination_binding);
    const auto* left_resource = find_resource_by_binding(ir, assignment.left_binding);
    if (destination == nullptr || left_resource == nullptr) {
        return false;
    }

    // Texture-to-texture copy (operation 1) through imageLoad/imageStore.
    if (destination->kind == kIrResourceKindTexture2D && left_resource->kind == kIrResourceKindTexture2D &&
        assignment.operation == 1) {
        const auto dst_mapped = resources.find(assignment.destination_binding);
        const auto src_mapped = resources.find(assignment.left_binding);
        if (dst_mapped == resources.end() || src_mapped == resources.end()) {
            return false;
        }

        const auto x = builder.ThreadIndexX();
        const auto y = builder.ThreadIndexY();
        if (x == GPU::IR::InvalidValueId || y == GPU::IR::InvalidValueId) {
            return false;
        }

        const auto dst = builder.TextureElement(dst_mapped->second, x, y);
        const auto src = builder.TextureElement(src_mapped->second, x, y);
        if (dst == GPU::IR::InvalidValueId || src == GPU::IR::InvalidValueId) {
            return false;
        }

        builder.Store(dst, src);
        return true;
    }

    if (destination->kind != kIrResourceKindBuffer || left_resource->kind != kIrResourceKindBuffer) {
        return false;
    }

    auto destination_node = build_easygpu_module_resource_access(
        ir, builder, resources, assignment.destination_binding, assignment.index_string_id);
    auto source_node = build_easygpu_module_resource_access(
        ir, builder, resources, assignment.left_binding, assignment.index_string_id);
    if (destination_node == GPU::IR::InvalidValueId || source_node == GPU::IR::InvalidValueId) {
        return false;
    }

    auto right = source_node;
    if (assignment.operation != 1) {
        GPU::IR::ValueId rhs = GPU::IR::InvalidValueId;
        if (assignment.right_operand_kind == 1) {
            rhs = build_easygpu_module_resource_access(ir, builder, resources, assignment.right_binding,
                                                       assignment.index_string_id);
        } else if (assignment.right_operand_kind == 2) {
            const auto* literal = get_string(ir, assignment.right_literal_string_id);
            const auto* type = get_string(ir, left_resource->element_type_string_id);
            const auto module_type = type == nullptr ? GPU::IR::Type{} : easygpu_module_type(*type);
            rhs = literal == nullptr || !module_type.IsValid()
                      ? GPU::IR::InvalidValueId
                      : builder.Literal(module_type, *literal);
        } else {
            return false;
        }

        GPU::IR::BinaryOp op{};
        if (rhs == GPU::IR::InvalidValueId || !easygpu_structured_binary_op(assignment.operation, &op)) {
            return false;
        }

        right = builder.Binary(op, right, rhs);
        if (right == GPU::IR::InvalidValueId) {
            return false;
        }
    }

    builder.Store(destination_node, right);
    return true;
}

std::unique_ptr<GPU::IR::Module> try_build_easygpu_module(const KernelState& kernel,
                                                          GPU::AD::GradientTape* gradientTape = nullptr) {
    std::string typed_error;
    const bool enable_fused_multiply_add = kEnableFusedMultiplyAdd && !kernel.auto_diff && gradientTape == nullptr;
    if (auto typed_module = try_build_typed_easygpu_module(kernel, enable_fused_multiply_add, &typed_error)) {
        return typed_module;
    }

    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir) || ir.shader_kind < 1 || ir.shader_kind > 3 || ir.group_x <= 0 || ir.group_y <= 0 ||
        ir.group_z <= 0) {
        return nullptr;
    }

    if (ir.has_section7) {
        if (!typed_error.empty()) {
            fail(FE_ERROR_UNSUPPORTED, "Section 7 typed IR could not be lowered to an EasyGPU module: " + typed_error);
        }

        return nullptr;
    }

    GPU::IR::ModuleBuilder builder;
    // If a GradientTape is provided, set it on the Builder for AD recording.
    // The tape is activated before module building and extracted after.
    if (gradientTape != nullptr) {
        // The tape will be activated in the ModuleLowerer during ScopedBind.
        // We store it here for the ModuleLowerer to use.
    }

    builder.BeginComputeKernel(static_cast<uint32_t>(ir.group_x), static_cast<uint32_t>(ir.group_y),
                               static_cast<uint32_t>(ir.group_z), static_cast<uint32_t>(ir.shader_kind) + 1);

	    ModuleResourceMap resources;
	    if (!register_easygpu_module_resources(ir, kernel, builder, &resources)) {
	        return nullptr;
	    }

    // Process FEIR instructions in order, building assignments and control flow.
    // Uses a two-pass approach: first determine control flow block structure,
    // then emit ModuleBuilder calls in the correct order.

    // Step 1: Build a mapping of instruction index -> block membership.
    // Block nesting stack: each entry tracks the current control flow type.
    struct BlockInfo {
        uint32_t id = 0;
        bool isIfBlock = false;
        bool isElse = false;
    };
    std::vector<BlockInfo> blockStack;
    std::vector<uint32_t> instrToBlock; // For each instruction, which block it belongs to (0 = main)

    // Track block assignment containers: for each block, which instruction indices it contains.
    std::vector<std::vector<uint32_t>> blockAssignments; // block index -> instruction indices
    blockAssignments.push_back({}); // Block 0 = main (no control flow)

    // Step 2: Scan instructions to build block hierarchy.
    for (uint32_t idx = 0; idx < static_cast<uint32_t>(ir.instructions.size()); idx++) {
        const auto& inst = ir.instructions[idx];

        if (inst.opcode == kIrOpcodeBeginBlock) {
            // Start collecting into a new block
            BlockInfo info;
            info.id = static_cast<uint32_t>(blockAssignments.size());
            blockAssignments.push_back({});
            blockStack.push_back(info);
            continue;
        }

        if (inst.opcode == kIrOpcodeEndBlock) {
            if (!blockStack.empty()) {
                blockStack.pop_back();
            }
            continue;
        }

        if (inst.opcode == kIrOpcodeIf || inst.opcode == kIrOpcodeFor ||
            inst.opcode == kIrOpcodeWhile || inst.opcode == kIrOpcodeDo ||
            inst.opcode == kIrOpcodeElse || inst.opcode == kIrOpcodeBreak ||
            inst.opcode == kIrOpcodeContinue || inst.opcode == kIrOpcodeReturn ||
            inst.opcode == kIrOpcodeInvocation || inst.opcode == kIrOpcodeResourceAccess ||
            inst.opcode == kIrOpcodeExpression || inst.opcode == kIrOpcodeLocalDeclaration) {
            continue;
        }

        if (inst.opcode == kIrOpcodeAssignment) {
            // Assign this instruction to the current block (or main block 0)
            uint32_t blockId = blockStack.empty() ? 0 : blockStack.back().id;
            if (blockId >= blockAssignments.size()) {
                blockAssignments.resize(blockId + 1);
            }
            blockAssignments[blockId].push_back(idx);
            continue;
        }

        // Barrier and shared memory instructions
        if (inst.opcode == kIrOpcodeWorkgroupBarrier || inst.opcode == kIrOpcodeMemoryBarrier ||
            inst.opcode == kIrOpcodeFullBarrier || inst.opcode == kIrOpcodeSharedMemoryDeclaration) {
            continue;
        }
    }

    // Build a resource-id → GLSL name map for condition expression lowering.
    std::unordered_map<uint32_t, std::string> resource_glsl_names;
    for (const auto& binding : builder.GetModule().resources) {
        resource_glsl_names[binding.id] = binding.name;
    }

    // Process instructions in order, emitting assignments and control flow.
    // Control flow structure uses RawGLSL but conditions are built through the typed
    // expression system (Section 3/5) instead of raw C# source text.
    struct CfFrame { uint8_t opcode; bool expect_else; };
    std::vector<CfFrame> cf_stack;

    for (uint32_t idx = 0; idx < static_cast<uint32_t>(ir.instructions.size()); idx++) {
        const auto& inst = ir.instructions[idx];

        switch (inst.opcode) {
        case kIrOpcodeAssignment: {
            // Check for compound assignment first (Section 6)
            bool had_compound = false;
            for (const auto& ca : ir.compound_assignments) {
                if (ca.instruction_index != idx) continue;
                auto dest = build_easygpu_module_resource_access(ir, builder, resources,
                    ca.destination_binding, ca.index_string_id);
                auto value = build_compound_expression(ir, builder, resources, ca.root_node_index);
                GPU::IR::BinaryOp op{};
                if (dest != GPU::IR::InvalidValueId && value != GPU::IR::InvalidValueId &&
                    easygpu_structured_binary_op(ca.operation, &op)) {
                    auto result = builder.Binary(op, dest, value);
                    if (result != GPU::IR::InvalidValueId)
                        builder.Store(dest, result);
                }
                had_compound = true;
                break;
            }
            if (had_compound) break;

            bool had_expression = false;
            for (const auto& assn : ir.expression_assignments) {
                if (assn.instruction_index == idx) {
                    build_easygpu_module_expression_assignment(ir, builder, resources, assn);
                    had_expression = true;
                }
            }
            if (!had_expression) {
                for (const auto& assn : ir.elementwise_assignments) {
                    if (assn.instruction_index == idx)
                        build_easygpu_module_structured_assignment(ir, builder, resources, assn);
                }
            }
            break;
        }
        case kIrOpcodeWorkgroupBarrier:
            builder.Barrier(GPU::IR::BarrierKind::Workgroup);
            break;
        case kIrOpcodeMemoryBarrier:
            builder.Barrier(GPU::IR::BarrierKind::Memory);
            break;
        case kIrOpcodeFullBarrier:
            builder.Barrier(GPU::IR::BarrierKind::Full);
            break;
        case kIrOpcodeSharedMemoryDeclaration:
            return nullptr;
        case kIrOpcodeLocalDeclaration:
            for (const auto& decl : ir.local_variable_decls) {
                if (decl.instruction_index != idx) continue;
                const auto* glsl_text = get_string(ir, decl.glsl_text_string_id);
                if (glsl_text == nullptr) break;
                builder.RawGLSL(*glsl_text + "\n");
                break;
            }
            break;
        case kIrOpcodeIf: {
            cf_stack.push_back({kIrOpcodeIf, true});
            auto cond = build_easygpu_module_condition(ir, builder, resources, idx, kCfRoleIfCondition);
            std::string cond_str = "true";
            if (cond != GPU::IR::InvalidValueId)
                cond_str = module_value_to_glsl(builder.GetModule(), cond, resource_glsl_names);
            builder.RawGLSL("if (" + cond_str + ") {\n");
            break;
        }
        case kIrOpcodeElse:
            if (!cf_stack.empty() && cf_stack.back().opcode == kIrOpcodeIf)
                cf_stack.back().expect_else = false;
            builder.RawGLSL("} else {\n");
            break;
        case kIrOpcodeEndBlock:
            if (!cf_stack.empty() && cf_stack.back().expect_else && cf_stack.back().opcode == kIrOpcodeIf) {
                // Then-body end: don't close — else is coming
                cf_stack.back().expect_else = false;
            } else if (!cf_stack.empty()) {
                cf_stack.pop_back();
                builder.RawGLSL("}\n");
            }
            break;
        case kIrOpcodeFor: {
            cf_stack.push_back({kIrOpcodeFor, false});
            auto cond = build_easygpu_module_condition(ir, builder, resources, idx, kCfRoleForCondition);
            std::string cond_str = module_value_to_glsl(builder.GetModule(),
                cond != GPU::IR::InvalidValueId ? cond : builder.Literal(GPU::IR::Type::Bool(), "true"),
                resource_glsl_names);
            // Look for for-step expression
            auto step_val = build_easygpu_module_condition(ir, builder, resources, idx, kCfRoleForStep);
            std::string step_str;
            if (step_val != GPU::IR::InvalidValueId)
                step_str = module_value_to_glsl(builder.GetModule(), step_val, resource_glsl_names);
            builder.RawGLSL("for (; " + cond_str + "; " + step_str + ") {\n");
            break;
        }
        case kIrOpcodeWhile: {
            cf_stack.push_back({kIrOpcodeWhile, false});
            auto cond = build_easygpu_module_condition(ir, builder, resources, idx, kCfRoleWhileCondition);
            std::string cond_str = "true";
            if (cond != GPU::IR::InvalidValueId)
                cond_str = module_value_to_glsl(builder.GetModule(), cond, resource_glsl_names);
            builder.RawGLSL("while (" + cond_str + ") {\n");
            break;
        }
        case kIrOpcodeDo:
            cf_stack.push_back({kIrOpcodeDo, false});
            builder.RawGLSL("do {\n");
            break;
        case kIrOpcodeBreak:
            builder.Break();
            break;
        case kIrOpcodeContinue:
            builder.Continue();
            break;
        case kIrOpcodeReturn:
            builder.Return();
            break;
        case kIrOpcodeBeginBlock:
        case kIrOpcodeInvocation:
        case kIrOpcodeTextureSample:
        case kIrOpcodeResourceAccess:
        case kIrOpcodeExpression:
        default:
            break;
        }
    }

    return std::make_unique<GPU::IR::Module>(builder.GetModule());
}

std::unique_ptr<GPU::Kernel::KernelBuildContext> try_build_easygpu_kernel_context(const KernelState& kernel, GPU::AD::GradientTape* gradientTape = nullptr) {
    auto module = try_build_easygpu_module(kernel, gradientTape);
    if (module == nullptr) {
        return nullptr;
    }

    auto context = GPU::IR::BuildKernelBuildContext(*module);
    if (context != nullptr) {
        context->SetOptimizationLevel(kShaderOptimizationLevel);
    }
    return context;
}

GPU::Backend::BufferMode easygpu_buffer_storage_mode(uint32_t mode) {
    switch (mode) {
    case 1:
        return GPU::Backend::BufferMode::Read;
    case 2:
        return GPU::Backend::BufferMode::Write;
    default:
        return GPU::Backend::BufferMode::ReadWrite;
    }
}

uint32_t easygpu_texture_usage_flags(const TextureState& texture) {
    using namespace GPU::Backend;

    uint32_t usage = TextureUsageTransferSrc | TextureUsageTransferDst;
    switch (texture.access) {
    case 1: // ReadOnlyTexture2D
        usage |= TextureUsageStorage | TextureUsageSampled;
        break;
    case 2: // WriteOnlyTexture2D
    case 3: // ReadWriteTexture2D
        usage |= TextureUsageStorage | TextureUsageSampled;
        break;
    case 4: // SampledTexture2D
        usage |= TextureUsageSampled;
        break;
    case 5: // RenderTarget
        usage |= TextureUsageColorAttachment | TextureUsageSampled | TextureUsageStorage;
        break;
    case 6: // DepthStencil
        usage |= TextureUsageDepthStencilAttachment;
        break;
    default:
        usage |= TextureUsageStorage | TextureUsageSampled;
        if (texture.depth == 1) {
            usage |= TextureUsageColorAttachment;
        }
        break;
    }

    return usage;
}

GPU::Backend::BufferHandle ensure_easygpu_buffer(BufferState& buffer, GPU::Backend::Backend& backend) {
    if (buffer.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
        GPU::Backend::BufferDesc desc;
        desc.sizeInBytes = buffer.bytes.size();
        desc.mode = easygpu_buffer_storage_mode(buffer.mode);
        desc.initialData = buffer.bytes.empty() ? nullptr : buffer.bytes.data();
        buffer.backend_buffer = backend.CreateBuffer(desc);
        if (buffer.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
            throw std::runtime_error("EasyGPU backend failed to create buffer.");
        }

        buffer.host_dirty = false;
        buffer.device_dirty = false;
        return buffer.backend_buffer;
    }

    if (buffer.host_dirty && !buffer.bytes.empty()) {
        backend.UploadBuffer(buffer.backend_buffer, 0, buffer.bytes.size(), buffer.bytes.data());
        buffer.host_dirty = false;
        buffer.device_dirty = false;
    }

    return buffer.backend_buffer;
}

void download_easygpu_buffer(BufferState& buffer, GPU::Backend::Backend& backend) {
    if (buffer.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE || !buffer.device_dirty || buffer.bytes.empty()) {
        return;
    }

    backend.DownloadBuffer(buffer.backend_buffer, 0, buffer.bytes.size(), buffer.bytes.data());
    buffer.device_dirty = false;
}

GPU::Backend::TextureHandle ensure_easygpu_texture(TextureState& texture, GPU::Backend::Backend& backend) {
    if (texture.backend_texture != GPU::Backend::INVALID_TEXTURE_HANDLE) {
        if (texture.host_dirty && !texture.bytes.empty()) {
            if (texture.pixel_format == 100) {
                throw std::runtime_error("Depth24Stencil8 textures cannot be uploaded from host memory on this backend.");
            }
            if (texture.depth > 1) {
                backend.UploadTexture3D(texture.backend_texture, 0, 0, 0, texture.width, texture.height,
                                        texture.depth, texture.bytes.data());
            } else {
                backend.UploadTexture(texture.backend_texture, 0, 0, texture.width, texture.height,
                                      texture.bytes.data());
            }
            texture.host_dirty = false;
            texture.device_dirty = false;
            texture.mipmaps_dirty = texture.mipmaps_requested && texture.mip_levels > 1;
        }
        if (texture.mipmaps_dirty && texture.mipmaps_requested && texture.mip_levels > 1) {
            if (texture.depth > 1) {
                throw std::runtime_error("EasyGPU mipmap generation currently supports 2D textures only.");
            }
            if (texture.pixel_format == 100 || texture.pixel_format == 101) {
                throw std::runtime_error("Depth textures do not support mipmap generation.");
            }
            backend.GenerateMipmaps(texture.backend_texture);
            texture.mipmaps_dirty = false;
        }
        return texture.backend_texture;
    }

    if (texture.bytes.empty()) {
        return GPU::Backend::INVALID_TEXTURE_HANDLE;
    }

    GPU::Backend::PixelFormat backend_format = GPU::Backend::PixelFormat::RGBA8;
    if (!easygpu_backend_pixel_format(texture.pixel_format, &backend_format)) {
        throw std::runtime_error("EasyGPU texture format is not supported.");
    }

    GPU::Backend::TextureDesc desc;
    desc.width = texture.width;
    desc.height = texture.height;
    desc.depth = texture.depth;
    desc.format = backend_format;
    desc.initialData = texture.pixel_format == 100 ? nullptr : texture.bytes.data();
    desc.mipLevels = texture.mip_levels;
    desc.usage = easygpu_texture_usage_flags(texture);

    texture.backend_texture = backend.CreateTexture(desc);
    if (texture.backend_texture == GPU::Backend::INVALID_TEXTURE_HANDLE) {
        throw std::runtime_error("EasyGPU backend failed to create texture.");
    }

    texture.host_dirty = false;
    texture.device_dirty = false;
    if (texture.mipmaps_requested && texture.mip_levels > 1) {
        if (texture.depth > 1) {
            throw std::runtime_error("EasyGPU mipmap generation currently supports 2D textures only.");
        }
        if (texture.pixel_format == 100 || texture.pixel_format == 101) {
            throw std::runtime_error("Depth textures do not support mipmap generation.");
        }
        backend.GenerateMipmaps(texture.backend_texture);
        texture.mipmaps_dirty = false;
    }
    return texture.backend_texture;
}

void download_easygpu_texture(TextureState& texture, GPU::Backend::Backend& backend) {
    if (texture.backend_texture == GPU::Backend::INVALID_TEXTURE_HANDLE || !texture.device_dirty || texture.bytes.empty()) {
        return;
    }

    if (texture.depth > 1) {
        trace_graphics_step("download texture3d");
        backend.DownloadTexture3D(texture.backend_texture, 0, 0, 0, texture.width, texture.height, texture.depth,
                                  texture.bytes.data());
    } else if (texture.pixel_format == 100) {
        throw std::runtime_error("Depth24Stencil8 textures cannot be downloaded to host memory on this backend.");
    } else {
        trace_graphics_step("download texture2d");
        backend.DownloadTexture(texture.backend_texture, 0, 0, texture.width, texture.height, texture.bytes.data());
    }
    texture.device_dirty = false;
}

void bind_easygpu_runtime_buffers(const KernelState& kernel, GPU::Kernel::KernelBuildContext& context,
                                  GPU::Backend::Backend& backend) {
    for (const auto& [binding, feather_buffer] : kernel.buffers) {
        auto buffer = g_buffers.find(feather_buffer);
        if (buffer == g_buffers.end()) {
            throw std::runtime_error("Kernel references an invalid Feather buffer.");
        }

        context.BindRuntimeBuffer(binding, ensure_easygpu_buffer(buffer->second, backend));
    }
}

void mark_easygpu_writable_buffers_dirty(const KernelState& kernel, const GPU::Kernel::KernelBuildContext& context) {
    for (const auto& info : context.GetBufferInfos()) {
        if (info.mode == GPU::Backend::BUFFER_MODE_READ_ONLY) {
            continue;
        }

        const auto binding = kernel.buffers.find(info.binding);
        if (binding == kernel.buffers.end()) {
            continue;
        }

        auto buffer = g_buffers.find(binding->second);
        if (buffer != g_buffers.end()) {
            buffer->second.device_dirty = true;
            buffer->second.host_dirty = false;
        }
    }
}

void bind_easygpu_runtime_textures(const KernelState& kernel, GPU::Kernel::KernelBuildContext& context,
                                   GPU::Backend::Backend& backend) {
    for (const auto& [binding, feather_texture] : kernel.textures) {
        auto texture = g_textures.find(feather_texture);
        if (texture == g_textures.end()) {
            throw std::runtime_error("Kernel references an invalid Feather texture.");
        }

        context.BindRuntimeTexture(binding, ensure_easygpu_texture(texture->second, backend));
    }
}

void mark_easygpu_writable_textures_dirty(const KernelState& kernel, const GPU::Kernel::KernelBuildContext& context) {
    for (const auto& info : context.GetTextureInfos()) {
        if (info.sampled) {
            continue;
        }

        const auto binding = kernel.textures.find(info.binding);
        if (binding == kernel.textures.end()) {
            continue;
        }

        auto texture = g_textures.find(binding->second);
        if (texture != g_textures.end()) {
            texture->second.device_dirty = true;
            texture->second.host_dirty = false;
            texture->second.mipmaps_dirty = texture->second.mipmaps_requested && texture->second.mip_levels > 1;
        }
    }
}

bool is_easygpu_texture_resource(const ParsedIr& ir, uint32_t binding) {
    const auto* resource = find_resource_by_binding(ir, binding);
    if (resource == nullptr || resource->kind != kIrResourceKindTexture2D) {
        return false;
    }

    // Texture resources are valid if they have a name and element type string.
    const auto* type = get_string(ir, resource->element_type_string_id);
    return type != nullptr && !type->empty();
}

bool is_easygpu_buffer_resource(const ParsedIr& ir, uint32_t binding) {
    const auto* resource = find_resource_by_binding(ir, binding);
    if (resource == nullptr || resource->kind != kIrResourceKindBuffer) {
        return false;
    }

    return easygpu_buffer_element_stride(ir, *resource) != 0;
}

bool is_easygpu_push_constant_resource(const ParsedIr& ir, uint32_t binding) {
    const auto* resource = find_resource_by_binding(ir, binding);
    if (resource == nullptr || resource->kind != kIrResourceKindPushConstant) {
        return false;
    }

    const auto* type = get_string(ir, resource->element_type_string_id);
    return type != nullptr && easygpu_module_type(*type).IsValid() &&
           push_constant_type_size(ir, *resource) != 0 &&
           push_constant_type_alignment(ir, *resource) != 0;
}

bool is_easygpu_expression_tree(const ParsedIr& ir, uint32_t node_index) {
    if (node_index >= ir.expression_nodes.size()) {
        return false;
    }

    const auto& node = ir.expression_nodes[node_index];
    switch (node.kind) {
    case 1:
        return is_easygpu_buffer_resource(ir, node.resource_binding) ||

               is_easygpu_texture_resource(ir, node.resource_binding);
    case 2:
        return true;
    case 3:
        return is_easygpu_expression_tree(ir, node.left_node_index) &&
               is_easygpu_expression_tree(ir, node.right_node_index);
    case 4: {
        const auto* symbol = get_string(ir, node.symbol_string_id);
        if (symbol == nullptr || easygpu_intrinsic_name(*symbol).empty() ||
            node.first_argument_index == UINT32_MAX ||
            node.first_argument_index > ir.expression_argument_indices.size() ||
            node.argument_count > ir.expression_argument_indices.size() - node.first_argument_index) {
            return false;
        }

        for (uint32_t i = 0; i < node.argument_count; ++i) {
            if (!is_easygpu_expression_tree(ir, ir.expression_argument_indices[node.first_argument_index + i])) {
                return false;
            }
        }
        return true;
    }
    case 5:
        return is_easygpu_expression_tree(ir, node.left_node_index) &&
               is_easygpu_expression_tree(ir, node.right_node_index);
    case 6:
        return is_easygpu_push_constant_resource(ir, node.resource_binding);
    case kIrExpressionNodeKindLocalVariable:
        return true;
    case kIrExpressionNodeKindShaderBuiltin:
        return true;
    case kIrExpressionNodeKindConstructor: {
        const auto* type_name = get_string(ir, node.type_string_id);
        if (type_name == nullptr) return false;
        if (easygpu_intrinsic_name_for_type(*type_name).empty()) return false;
        if (!easygpu_module_type(*type_name).IsValid()) return false;
        if (node.first_argument_index == UINT32_MAX || node.argument_count == 0) return false;
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            if (!is_easygpu_expression_tree(ir, ir.expression_argument_indices[node.first_argument_index + i])) {
                return false;
            }
        }
        return true;
    }
    case kIrExpressionNodeKindCallableCall: {
        // Section 1-6 compatibility expression trees do not carry callable function
        // tables. Generated section 7 kernels are handled by the typed EasyGPU path and
        // are not allowed to fall back here.
        return false;
    }
    case kIrExpressionNodeKindTextureSample:
    case kIrExpressionNodeKindGpuStructField: {
        if (node.argument_count < 1 || node.first_argument_index == UINT32_MAX) return false;
        return is_easygpu_expression_tree(ir, ir.expression_argument_indices[node.first_argument_index]);
    }
    case kIrExpressionNodeKindTextureSampleLevel: {
        // Verify texture and sampler resources exist and arguments are valid.
        const auto* r = find_resource_by_binding(ir, node.resource_binding);
        if (r == nullptr || r->kind != kIrResourceKindTexture2D) return false;
        const auto* sampler_name = get_string(ir, node.symbol_string_id);
        if (sampler_name == nullptr) return false;
        const auto* sr = find_resource_by_name(ir, *sampler_name);
        if (sr == nullptr) return false;
        if (node.first_argument_index == UINT32_MAX || node.argument_count == 0) return false;
        for (uint32_t i = 0; i < node.argument_count; ++i) {
            if (!is_easygpu_expression_tree(ir, ir.expression_argument_indices[node.first_argument_index + i]))
                return false;
        }
        return true;
    }
    case kIrExpressionNodeKindTernary:
        // Ternary in legacy expression-assignment sections remains compatibility-only.
        // Generated section 7 ternaries lower through the typed EasyGPU path.
        return false;
    default:
        return false;
    }
}

bool can_dispatch_easygpu_buffer_kernel(const KernelState& kernel) {
    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir) || ir.group_x <= 0 || ir.group_y <= 0 ||
        ir.group_z <= 0) {
        return false;
    }

    if (ir.shader_kind < 1 || ir.shader_kind > 3) {
        return false;
    }

    bool has_texture = false;
    for (const auto& resource : ir.resources) {
        if (resource.kind == kIrResourceKindTexture2D) {
            has_texture = true;
            break;
        }
    }

    // Legacy 2D/3D buffer-only kernels historically use CPU fallback because
    // old assignment sections do not encode full dimensional indexing. When the
    // test harness strips those sections away, section 7 owns the semantics and
    // can prove a real typed EasyGPU path instead.
    if (ir.shader_kind != 1 && !has_texture && !ir.has_section7) {
        return false;
    }

    for (const auto& resource : ir.resources) {
        if (resource.kind == kIrResourceKindBuffer) {
            const auto expected_stride = easygpu_buffer_element_stride(ir, resource);
            if (expected_stride == 0) {
                return false;
            }

            const auto bound = kernel.buffers.find(resource.binding);
            if (bound == kernel.buffers.end()) {
                return false;
            }

            const auto buffer = g_buffers.find(bound->second);
            if (buffer == g_buffers.end()) {
                return false;
            }

            if (buffer->second.stride != expected_stride) {
                return false;
            }

            continue;
        }

        if (resource.kind == kIrResourceKindPushConstant) {
            size_t offset = 0;
            size_t size = 0;
            const auto* type = get_string(ir, resource.element_type_string_id);
            if (!is_easygpu_push_constant_resource(ir, resource.binding)) {
                fail(FE_ERROR_UNSUPPORTED,
                     "EasyGPU typed dispatch does not support push constant binding " +
                         std::to_string(resource.binding) + " of type '" +
                         (type == nullptr ? std::string("<unknown>") : *type) + "'.");
                return false;
            }

            if (!find_push_constant_offset(ir, resource.binding, &offset, &size)) {
                fail(FE_ERROR_UNSUPPORTED,
                     "EasyGPU typed dispatch could not compute a push constant range for binding " +
                         std::to_string(resource.binding) + " of type '" +
                         (type == nullptr ? std::string("<unknown>") : *type) + "'.");
                return false;
            }

            if (offset + size > kernel.push_constants.size()) {
                fail(FE_ERROR_UNSUPPORTED,
                     "EasyGPU typed dispatch push constant binding " + std::to_string(resource.binding) +
                         " requires bytes [" + std::to_string(offset) + ", " +
                         std::to_string(offset + size) + ") but only " +
                         std::to_string(kernel.push_constants.size()) + " bytes were uploaded.");
                return false;
            }

            continue;
        }

        if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto bound = kernel.textures.find(resource.binding);
            if (bound == kernel.textures.end()) {
                return false;
            }

            const auto texture = g_textures.find(bound->second);
            if (texture == g_textures.end()) {
                return false;
            }

            if (texture->second.width == 0 || texture->second.height == 0 ||
                (resource.kind == kIrResourceKindTexture2D ? texture->second.depth != 1 : texture->second.depth == 0)) {
                return false;
            }

            GPU::Runtime::PixelFormat format;
            if (!easygpu_runtime_pixel_format(texture->second.pixel_format, &format)) {
                fail(FE_ERROR_UNSUPPORTED,
                     std::string("EasyGPU texture format ") + pixel_format_name(texture->second.pixel_format) +
                         " is not supported by the typed texture bridge.");
                return false;
            }

            continue;
        }

        if (resource.kind == kIrResourceKindSampler) {
            const auto bound = kernel.samplers.find(resource.binding);
            if (bound == kernel.samplers.end()) {
                return false;
            }

            if (bound->second != 0 && g_samplers.find(bound->second) == g_samplers.end()) {
                return false;
            }

            continue;
        }

        return false;
    }

    if (ir.has_section7) {
        return true;
    }

    for (const auto& assignment : ir.elementwise_assignments) {
	        if (is_easygpu_texture_resource(ir, assignment.destination_binding) &&
	            is_easygpu_texture_resource(ir, assignment.left_binding)) {
	            if (assignment.operation != 1 || assignment.right_operand_kind != 0) {
	                return false;
	            }
	            continue;
	        }

	        if (!is_easygpu_buffer_resource(ir, assignment.destination_binding) ||
	            !is_easygpu_buffer_resource(ir, assignment.left_binding) ||
	            (assignment.right_operand_kind == 1 &&
	             !is_easygpu_buffer_resource(ir, assignment.right_binding))) {
	            return false;
	        }
	    }


    for (const auto& assignment : ir.expression_assignments) {
        if (is_easygpu_texture_resource(ir, assignment.destination_binding)) {
            if (assignment.root_node_index >= ir.expression_nodes.size()) {
                return false;
            }
            const auto& root = ir.expression_nodes[assignment.root_node_index];
            if (root.kind != 1 || !is_easygpu_texture_resource(ir, root.resource_binding)) {
                return false;
            }
            continue;
        }

        if (!is_easygpu_buffer_resource(ir, assignment.destination_binding) ||
            !is_easygpu_expression_tree(ir, assignment.root_node_index)) {
            return false;
        }
    }

    return true; // Accept kernels with or without assignments
}

GPU::Backend::PipelineHandle create_easygpu_compute_pipeline(GPU::Kernel::KernelBuildContext& context,
                                                             GPU::Backend::Backend& backend) {
    const auto shader_source = context.GetCompleteCode();
    GPU::Backend::ShaderDesc shader_desc;
    shader_desc.type = GPU::Backend::ShaderType::Compute;
    shader_desc.sourceCode = shader_source;
    shader_desc.entryPoint = "main";
    shader_desc.optimizationLevel = context.GetOptimizationLevel();

    const auto shader = backend.CreateShader(shader_desc);
    if (shader == GPU::Backend::INVALID_SHADER_HANDLE) {
        throw std::runtime_error("EasyGPU backend failed to create compute shader.");
    }

    EasyGpuShaderGuard shader_guard(backend, shader);
    GPU::Backend::PipelineDesc pipeline_desc;
    pipeline_desc.computeShader = shader;
    pipeline_desc.workGroupSizeX = static_cast<uint32_t>(context.WorkSizeX);
    pipeline_desc.workGroupSizeY = static_cast<uint32_t>(context.WorkSizeY);
    pipeline_desc.workGroupSizeZ = static_cast<uint32_t>(context.WorkSizeZ);
    pipeline_desc.pushConstantSize = context.GetPushConstantSize();

    for (const auto& buffer_info : context.GetBufferInfos()) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = buffer_info.binding;
        entry.type = GPU::Backend::BindingType::Buffer;
        entry.readOnly = buffer_info.mode == GPU::Backend::BUFFER_MODE_READ_ONLY;
        pipeline_desc.resources.push_back(entry);
    }

    for (const auto& texture_info : context.GetTextureInfos()) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = texture_info.binding;
        entry.type = texture_info.sampled ? GPU::Backend::BindingType::Sampler : GPU::Backend::BindingType::Texture;
        entry.format = GPU::Runtime::ToBackendPixelFormat(texture_info.format);
        entry.readOnly = texture_info.sampled;
        pipeline_desc.resources.push_back(entry);
    }

    const auto pipeline = backend.CreatePipeline(pipeline_desc);
    if (pipeline == GPU::Backend::INVALID_PIPELINE_HANDLE) {
        throw std::runtime_error("EasyGPU backend failed to create compute pipeline.");
    }

    return pipeline;
}

bool try_dispatch_easygpu_buffer_kernel(FeKernelHandle kernel_handle, KernelState& kernel, uint32_t group_x,
                                        uint32_t group_y, uint32_t group_z, bool wait) {
    if (!can_dispatch_easygpu_buffer_kernel(kernel)) {
        return false;
    }

    GPU::Runtime::AutoInitContext();
    GPU::Runtime::Context::GetInstance().MakeCurrent();
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        return false;
    }

    GPU::Kernel::KernelBuildContext* context = nullptr;
    auto cache_it = g_compute_kernel_caches.find(kernel_handle);
    if (cache_it != g_compute_kernel_caches.end() &&
        cache_it->second.context != nullptr &&
        cache_it->second.context->HasCachedPipeline()) {
        context = cache_it->second.context.get();
    }

    std::unique_ptr<GPU::Kernel::KernelBuildContext> local_context;
    if (context == nullptr) {
        local_context = try_build_easygpu_kernel_context(kernel);
        if (local_context == nullptr) {
            return false;
        }

        context = local_context.get();
    }

    bind_easygpu_runtime_buffers(kernel, *context, *backend);
    bind_easygpu_runtime_textures(kernel, *context, *backend);

    if (!context->HasCachedPipeline()) {
        const auto pipeline = create_easygpu_compute_pipeline(*context, *backend);
        context->SetCachedPipeline(pipeline);
    }

    if (local_context != nullptr && local_context->GetTextureInfos().empty()) {
        auto& cache = g_compute_kernel_caches[kernel_handle];
        cache.context = std::move(local_context);
        context = cache.context.get();
    }

    const auto pipeline = context->GetCachedPipeline();
    backend->BindPipeline(pipeline);
    context->UploadUniformValues(pipeline);

    const auto& bindings = context->GetCachedBindings();
    if (!bindings.empty()) {
        backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    }

    backend->Dispatch(group_x, group_y, group_z);
    const auto barrier = context->GetRequiredBarrierType();
    if (barrier != GPU::Backend::BarrierType::None) {
        backend->MemoryBarrier(barrier);
    }

    if (wait) {
        backend->Finish();
    }

    mark_easygpu_writable_buffers_dirty(kernel, *context);
    mark_easygpu_writable_textures_dirty(kernel, *context);
    return true;
}


bool try_dispatch_easygpu_ad_kernel(KernelState& kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                     bool wait) {
    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir)) {
        fail(FE_ERROR_INVALID_ARGUMENT, "AD kernel IR could not be parsed.");
        return false;
    }
    if (ir.shader_kind != 1 || ir.group_y != 1 || ir.group_z != 1 || group_y != 1 || group_z != 1) {
        fail(FE_ERROR_UNSUPPORTED, "Feather AD currently supports 1D compute kernels only.");
        return false;
    }
    if (!can_dispatch_easygpu_buffer_kernel(kernel)) {
        if (g_last_error.empty()) {
            fail(FE_ERROR_UNSUPPORTED, "AD kernel is not supported by the EasyGPU typed dispatch path.");
        }
        return false;
    }
    std::string unsupported_control_flow;
    if (ir.has_section7 && typed_ir_contains_unsupported_ad_control_flow(ir.typed_module, &unsupported_control_flow)) {
        fail(FE_ERROR_UNSUPPORTED, "Feather AD " + unsupported_control_flow + ".");
        return false;
    }

    std::vector<IrAdAnnotation> parameters;
    std::vector<IrAdAnnotation> losses;
    for (const auto& annotation : ir.ad_annotations) {
        if (annotation.role == kIrAdRoleParameter) {
            parameters.push_back(annotation);
        } else if (annotation.role == kIrAdRoleLoss) {
            losses.push_back(annotation);
        }
    }
    if (parameters.empty()) {
        fail(FE_ERROR_UNSUPPORTED, "AD kernel does not contain any differentiable parameter annotations.");
        return false;
    }
    if (losses.size() != 1) {
        fail(FE_ERROR_UNSUPPORTED, "AD kernel must contain exactly one scalar loss annotation.");
        return false;
    }

    GPU::Runtime::AutoInitContext();
    GPU::Runtime::Context::GetInstance().MakeCurrent();
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for AD dispatch.");
        return false;
    }

    GPU::AD::GradientTape gradientTape;
    std::vector<ADGradientState> next_gradients;
    next_gradients.reserve(parameters.size());
    std::unordered_set<uint32_t> requested_gradient_bindings;

    for (const auto& parameter : parameters) {
        if (parameter.binding == kIrNoBinding || parameter.source_kind != kIrAdSourceKindBufferElement) {
            fail(FE_ERROR_UNSUPPORTED, "AD.Parameter must reference a differentiable buffer element.");
            return false;
        }
        const auto* resource = find_resource_by_binding(ir, parameter.binding);
        if (resource == nullptr || resource->kind != kIrResourceKindBuffer) {
            fail(FE_ERROR_UNSUPPORTED, "AD parameter annotation does not refer to a buffer resource.");
            return false;
        }
        const auto bound = kernel.buffers.find(resource->binding);
        if (bound == kernel.buffers.end()) {
            fail(FE_ERROR_INVALID_ARGUMENT, "AD parameter buffer is not bound.");
            return false;
        }
        const auto buffer = g_buffers.find(bound->second);
        if (buffer == g_buffers.end()) {
            fail(FE_ERROR_INVALID_HANDLE, "AD parameter buffer handle is invalid.");
            return false;
        }
        const auto resource_stride = easygpu_buffer_element_stride(ir, *resource);
        if (resource_stride == 0 || buffer->second.stride != resource_stride || buffer->second.bytes.empty()) {
            fail(FE_ERROR_UNSUPPORTED, "AD parameter buffer has an unsupported element stride.");
            return false;
        }
        const auto element_count = static_cast<uint32_t>(buffer->second.bytes.size() / buffer->second.stride);
        if (element_count == 0) {
            fail(FE_ERROR_INVALID_ARGUMENT, "AD parameter buffer must contain at least one element.");
            return false;
        }

        auto element_type = string_or_empty(ir, parameter.type_name_string_id);
        if (element_type.empty()) {
            element_type = string_or_empty(ir, resource->element_type_string_id);
        }
        const auto glsl_type = easygpu_ad_glsl_type_name(element_type);
        if (glsl_type.empty()) {
            fail(FE_ERROR_UNSUPPORTED, "AD parameter type is not supported by the native EasyGPU AD bridge.");
            return false;
        }
        const auto component_count = ad_component_count_for_type(element_type);
        if (component_count == 0) {
            fail(FE_ERROR_UNSUPPORTED, "AD parameter component count could not be determined.");
            return false;
        }

        const auto easygpu_name = easygpu_buffer_name(*resource);
        gradientTape.RegisterBufferParameter(easygpu_name, glsl_type, element_count);
        if (!requested_gradient_bindings.insert(resource->binding).second) {
            continue;
        }

        ADGradientState gradient;
        gradient.name = string_or_empty(ir, parameter.name_string_id);
        if (gradient.name.empty()) {
            gradient.name = string_or_empty(ir, resource->name_string_id);
        }
        if (gradient.name.empty()) {
            gradient.name = easygpu_name;
        }
        gradient.resource_name = string_or_empty(ir, parameter.resource_name_string_id);
        if (gradient.resource_name.empty()) {
            gradient.resource_name = string_or_empty(ir, resource->name_string_id);
        }
        gradient.element_type = element_type;
        gradient.easygpu_name = easygpu_name;
        gradient.source_binding = resource->binding;
        gradient.element_count = element_count;
        gradient.element_stride = static_cast<uint32_t>(resource_stride);
        gradient.component_count = component_count;
        next_gradients.push_back(std::move(gradient));
    }

    std::unordered_set<uint32_t> parameter_bindings;
    parameter_bindings.reserve(next_gradients.size());
    for (const auto& gradient : next_gradients) {
        parameter_bindings.insert(gradient.source_binding);
    }

    const auto buffer_usage = collect_ad_buffer_usage(ir);
    for (const auto& resource : ir.resources) {
        if (resource.kind != kIrResourceKindBuffer ||
            resource.access != 3 ||
            parameter_bindings.count(resource.binding) != 0 ||
            buffer_usage.reads.count(resource.binding) == 0 ||
            buffer_usage.writes.count(resource.binding) == 0) {
            continue;
        }
        const auto bound = kernel.buffers.find(resource.binding);
        if (bound == kernel.buffers.end()) {
            continue;
        }
        const auto buffer = g_buffers.find(bound->second);
        if (buffer == g_buffers.end() || buffer->second.stride == 0 || buffer->second.bytes.empty()) {
            continue;
        }
        auto element_type = string_or_empty(ir, resource.element_type_string_id);
        const auto glsl_type = easygpu_ad_glsl_type_name(element_type);
        if (glsl_type.empty()) {
            continue;
        }
        const auto component_count = ad_component_count_for_type(element_type);
        if (component_count == 0) {
            continue;
        }
        const auto element_count = static_cast<uint32_t>(buffer->second.bytes.size() / buffer->second.stride);
        if (element_count == 0) {
            continue;
        }

        gradientTape.RegisterBufferAdjointStorage(
            easygpu_buffer_name(resource),
            glsl_type,
            static_cast<size_t>(element_count) * static_cast<size_t>(component_count));
    }

    const auto& loss = losses.front();
    auto loss_name = string_or_empty(ir, loss.name_string_id);
    auto loss_type = string_or_empty(ir, loss.type_name_string_id);
    if (loss_name.empty() || (loss.source_kind != kIrAdSourceKindLocal && loss.source_kind != 0)) {
        fail(FE_ERROR_UNSUPPORTED, "AD loss annotation must identify a scalar local value.");
        return false;
    }
    if (easygpu_ad_glsl_type_name(loss_type) != "float") {
        fail(FE_ERROR_UNSUPPORTED, "AD loss annotation must have scalar float type.");
        return false;
    }
    gradientTape.MarkLoss(loss_name, "float");

    auto forwardModule = try_build_easygpu_module(kernel, &gradientTape);
    if (forwardModule == nullptr) {
        if (g_last_error.empty()) {
            fail(FE_ERROR_UNSUPPORTED, "AD forward module could not be built for EasyGPU.");
        }
        return false;
    }

    auto forwardContext = GPU::IR::BuildKernelBuildContext(*forwardModule, &gradientTape);
    if (forwardContext == nullptr) {
        fail(FE_ERROR_UNSUPPORTED, "AD forward build context could not be lowered with an active GradientTape.");
        return false;
    }
    forwardContext->SetOptimizationLevel(kShaderOptimizationLevel);
    if (gradientTape.Size() == 0) {
        fail(FE_ERROR_UNSUPPORTED, "AD forward lowering did not record any differentiable operations.");
        return false;
    }

    const auto forward_glsl = forwardContext->GetCompleteCode();
    GPU::AD::AdjointGenerator adjointGen;
    auto body = adjointGen.GenerateBody(gradientTape, true);
    if (body.lines.empty() && body.writebacks.empty() && body.bufferWritebacks.empty()) {
        fail(FE_ERROR_UNSUPPORTED, "EasyGPU AD did not generate a backward body for this kernel.");
        return false;
    }

    auto next_binding = static_cast<int>(forwardContext->GetNextBinding());
    for (const auto& buffer_info : forwardContext->GetBufferInfos()) {
        next_binding = std::max(next_binding, static_cast<int>(buffer_info.binding) + 1);
    }
    for (const auto& texture_info : forwardContext->GetTextureInfos()) {
        next_binding = std::max(next_binding, static_cast<int>(texture_info.binding) + 1);
    }
    std::vector<GPU::AD::GradBufGroup> grad_groups;
    grad_groups.reserve(next_gradients.size());
    std::unordered_set<std::string> parameter_adjoint_arrays;
    auto adj_base_name = [](const std::string& name) {
        const auto bracket = name.find('[');
        return bracket == std::string::npos ? name : name.substr(0, bracket);
    };
    for (const auto& writeback : body.writebacks) {
        parameter_adjoint_arrays.insert(adj_base_name(writeback.second));
    }
    for (const auto& writeback : body.bufferWritebacks) {
        parameter_adjoint_arrays.insert(adj_base_name(writeback.adjName));
    }
    const auto work_size_x = std::max(1, forwardContext->WorkSizeX);
    const auto dispatched_threads = static_cast<size_t>(group_x) * static_cast<size_t>(work_size_x);
    for (auto& gradient : next_gradients) {
        if (next_binding >= static_cast<int>(GPU::Backend::MAX_BUFFER_BINDINGS)) {
            fail(FE_ERROR_UNSUPPORTED, "AD gradient buffers exceed backend buffer binding limits.");
            return false;
        }
        gradient.gradient_binding = static_cast<uint32_t>(next_binding++);
        const auto scalar_slots = static_cast<size_t>(gradient.element_count) *
                                  static_cast<size_t>(std::max<uint32_t>(gradient.component_count, 1));
        if (scalar_slots == 0 || dispatched_threads == 0 ||
            scalar_slots > (SIZE_MAX / sizeof(float)) ||
            dispatched_threads > (SIZE_MAX / scalar_slots / sizeof(float))) {
            fail(FE_ERROR_OUT_OF_MEMORY, "AD gradient buffer size overflowed.");
            return false;
        }
        gradient.byte_size = dispatched_threads * scalar_slots * sizeof(float);

        GPU::AD::GradBufGroup group;
        group.baseName = gradient.easygpu_name;
        group.binding = static_cast<int>(gradient.gradient_binding);
        group.stride = static_cast<int>(scalar_slots);
        grad_groups.push_back(std::move(group));
    }

    int adj_pool_binding = -1;
    if (!body.declarations.empty()) {
        for (const auto& declaration : body.declarations) {
            if (declaration.second.find('[') != std::string::npos &&
                parameter_adjoint_arrays.find(adj_base_name(declaration.first)) == parameter_adjoint_arrays.end()) {
                adj_pool_binding = next_binding++;
                break;
            }
        }
    }
    if (adj_pool_binding >= static_cast<int>(GPU::Backend::MAX_BUFFER_BINDINGS)) {
        fail(FE_ERROR_UNSUPPORTED, "AD adjoint pool exceeds backend buffer binding limits.");
        return false;
    }

    std::string combined_glsl;
    try {
        combined_glsl = GPU::AD::MergeForwardBackward(
            forward_glsl,
            body,
            forwardContext->WorkSizeX,
            forwardContext->WorkSizeY,
            forwardContext->WorkSizeZ,
            grad_groups,
            adj_pool_binding);
    } catch (const std::exception& ex) {
        fail(FE_ERROR_UNSUPPORTED, std::string("EasyGPU AD merge failed: ") + ex.what());
        return false;
    }

    // Match EasyGPU::ADKernel1D: Feather dispatches ceil(logical/threadgroup) workgroups, so
    // padded lanes must not execute user code or write per-thread gradient slots.
    {
        const auto main_pos = combined_glsl.find("void main()");
        if (main_pos != std::string::npos) {
            const auto brace_pos = combined_glsl.find("{", main_pos);
            if (brace_pos != std::string::npos) {
                const auto logical_x = static_cast<uint32_t>(kernel.logical_x);
                combined_glsl.insert(
                    brace_pos + 1,
                    "\n    if (gl_GlobalInvocationID.x >= " + std::to_string(logical_x) + "u) return;\n");
            }
        }
    }

    release_ad_gradient_buffers(kernel);
    kernel.last_ad_backward_glsl = combined_glsl;

    std::vector<unsigned char> zero_bytes;
    for (auto& gradient : next_gradients) {
        GPU::Backend::BufferDesc desc;
        desc.sizeInBytes = gradient.byte_size;
        desc.mode = GPU::Backend::BufferMode::ReadWrite;
        zero_bytes.assign(gradient.byte_size, 0);
        desc.initialData = zero_bytes.empty() ? nullptr : zero_bytes.data();
        gradient.backend_buffer = backend->CreateBuffer(desc);
        if (gradient.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
            release_pending_ad_gradient_buffers(next_gradients);
            fail(FE_ERROR_OUT_OF_MEMORY, "EasyGPU backend failed to allocate AD gradient buffer.");
            return false;
        }
    }

    size_t adj_pool_float_count = 0;
    for (const auto& declaration : body.declarations) {
        if (parameter_adjoint_arrays.find(adj_base_name(declaration.first)) != parameter_adjoint_arrays.end()) {
            continue;
        }
        const auto bracket = declaration.second.find('[');
        if (bracket == std::string::npos) {
            continue;
        }
        const auto close = declaration.second.find(']', bracket);
        if (close == std::string::npos || close <= bracket + 1) {
            continue;
        }
        try {
            adj_pool_float_count += static_cast<size_t>(
                std::stoull(declaration.second.substr(bracket + 1, close - bracket - 1)));
        } catch (...) {
            adj_pool_float_count += dispatched_threads;
        }
    }
    if (adj_pool_binding >= 0 && adj_pool_float_count > 0) {
        GPU::Backend::BufferDesc desc;
        desc.sizeInBytes = adj_pool_float_count * sizeof(float);
        desc.mode = GPU::Backend::BufferMode::ReadWrite;
        std::vector<unsigned char> zeros(desc.sizeInBytes, 0);
        desc.initialData = zeros.data();
        kernel.ad_adj_pool = backend->CreateBuffer(desc);
        kernel.ad_adj_pool_size = desc.sizeInBytes;
        if (kernel.ad_adj_pool == GPU::Backend::INVALID_BUFFER_HANDLE) {
            release_pending_ad_gradient_buffers(next_gradients);
            release_ad_gradient_buffers(kernel);
            fail(FE_ERROR_OUT_OF_MEMORY, "EasyGPU backend failed to allocate AD adjoint pool.");
            return false;
        }
    }

    GPU::Backend::ShaderDesc shader_desc;
    shader_desc.type = GPU::Backend::ShaderType::Compute;
    shader_desc.sourceCode = combined_glsl;
    shader_desc.entryPoint = "main";
    shader_desc.optimizationLevel = forwardContext->GetOptimizationLevel();

    GPU::Backend::ShaderHandle shader = GPU::Backend::INVALID_SHADER_HANDLE;
    try {
        shader = backend->CreateShader(shader_desc);
    } catch (const std::exception& ex) {
        release_pending_ad_gradient_buffers(next_gradients);
        release_ad_gradient_buffers(kernel);
        fail(FE_ERROR_SHADER_COMPILE_FAILED,
             std::string("EasyGPU backend failed to compile merged AD backward shader: ") + ex.what() +
                 "\nMerged AD GLSL excerpt:\n" + glsl_error_excerpt(combined_glsl, ex.what()));
        return false;
    }
    if (shader == GPU::Backend::INVALID_SHADER_HANDLE) {
        release_pending_ad_gradient_buffers(next_gradients);
        release_ad_gradient_buffers(kernel);
        fail(FE_ERROR_SHADER_COMPILE_FAILED,
             "EasyGPU backend failed to compile merged AD backward shader.\nMerged AD GLSL excerpt:\n" +
                 glsl_excerpt(combined_glsl, 1, 120));
        return false;
    }
    EasyGpuShaderGuard shader_guard(*backend, shader);

    GPU::Backend::PipelineDesc pipeline_desc;
    pipeline_desc.computeShader = shader;
    pipeline_desc.workGroupSizeX = static_cast<uint32_t>(forwardContext->WorkSizeX);
    pipeline_desc.workGroupSizeY = static_cast<uint32_t>(forwardContext->WorkSizeY);
    pipeline_desc.workGroupSizeZ = static_cast<uint32_t>(forwardContext->WorkSizeZ);
    pipeline_desc.pushConstantSize = forwardContext->GetPushConstantSize();

    for (const auto& buffer_info : forwardContext->GetBufferInfos()) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = buffer_info.binding;
        entry.type = GPU::Backend::BindingType::Buffer;
        entry.readOnly = buffer_info.mode == GPU::Backend::BUFFER_MODE_READ_ONLY;
        pipeline_desc.resources.push_back(entry);
    }
    for (const auto& texture_info : forwardContext->GetTextureInfos()) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = texture_info.binding;
        entry.type = texture_info.sampled ? GPU::Backend::BindingType::Sampler : GPU::Backend::BindingType::Texture;
        entry.format = GPU::Runtime::ToBackendPixelFormat(texture_info.format);
        entry.readOnly = texture_info.sampled;
        pipeline_desc.resources.push_back(entry);
    }
    for (const auto& gradient : next_gradients) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = gradient.gradient_binding;
        entry.type = GPU::Backend::BindingType::Buffer;
        entry.readOnly = false;
        pipeline_desc.resources.push_back(entry);
    }
    if (kernel.ad_adj_pool != GPU::Backend::INVALID_BUFFER_HANDLE && adj_pool_binding >= 0) {
        GPU::Backend::ResourceLayoutEntry entry;
        entry.binding = static_cast<uint32_t>(adj_pool_binding);
        entry.type = GPU::Backend::BindingType::Buffer;
        entry.readOnly = false;
        pipeline_desc.resources.push_back(entry);
    }

    const auto pipeline = backend->CreatePipeline(pipeline_desc);
    if (pipeline == GPU::Backend::INVALID_PIPELINE_HANDLE) {
        release_pending_ad_gradient_buffers(next_gradients);
        release_ad_gradient_buffers(kernel);
        fail(FE_ERROR_SHADER_COMPILE_FAILED, "EasyGPU backend failed to create merged AD backward pipeline.");
        return false;
    }

    bind_easygpu_runtime_buffers(kernel, *forwardContext, *backend);
    bind_easygpu_runtime_textures(kernel, *forwardContext, *backend);
    auto bindings = forwardContext->GetCachedBindings();
    for (const auto& gradient : next_gradients) {
        GPU::Backend::ResourceBinding binding;
        binding.binding = gradient.gradient_binding;
        binding.type = GPU::Backend::BindingType::Buffer;
        binding.buffer = gradient.backend_buffer;
        binding.readOnly = false;
        bindings.push_back(binding);
    }
    if (kernel.ad_adj_pool != GPU::Backend::INVALID_BUFFER_HANDLE && adj_pool_binding >= 0) {
        GPU::Backend::ResourceBinding binding;
        binding.binding = static_cast<uint32_t>(adj_pool_binding);
        binding.type = GPU::Backend::BindingType::Buffer;
        binding.buffer = kernel.ad_adj_pool;
        binding.readOnly = false;
        bindings.push_back(binding);
    }

    backend->BindPipeline(pipeline);
    forwardContext->UploadUniformValues(pipeline);
    if (!bindings.empty()) {
        backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    }

    backend->Dispatch(group_x, group_y, group_z);
    backend->MemoryBarrier(GPU::Backend::BarrierType::All);
    if (wait) {
        backend->Finish();
    }

    backend->DestroyPipeline(pipeline);
    kernel.ad_gradients = std::move(next_gradients);
    mark_easygpu_writable_buffers_dirty(kernel, *forwardContext);
    mark_easygpu_writable_textures_dirty(kernel, *forwardContext);
    return true;
}

FeResult build_easygpu_kernel_source(const KernelState& kernel, std::string* source) {
    if (source == nullptr) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Output source pointer must not be null.");
    }

    auto module = try_build_easygpu_module(kernel);
    if (module == nullptr) {
        ParsedIr parsed;
        if (parse_feather_ir(kernel.ir, &parsed) && parsed.has_section7) {
            return g_last_error.empty()
                       ? fail(FE_ERROR_UNSUPPORTED, "Section 7 typed IR could not be lowered to an EasyGPU module.")
                       : FE_ERROR_UNSUPPORTED;
        }

        return fail(FE_ERROR_UNSUPPORTED, "Kernel IR is not yet supported by the EasyGPU IR module bridge.");
    }

    auto context = GPU::IR::BuildKernelBuildContext(*module);
    if (context == nullptr) {
        return fail(FE_ERROR_UNSUPPORTED, "EasyGPU module could not be lowered to a kernel build context.");
    }

    *source = context->GetCompleteCode();
    return ok();
}

bool try_build_easygpu_kernel_source(const KernelState& kernel, std::string* source) {
    return build_easygpu_kernel_source(kernel, source) == FE_OK;
}

bool try_build_easygpu_optimized_kernel_source(const KernelState& kernel, std::string* source) {
    if (source == nullptr) {
        return false;
    }

    auto context = try_build_easygpu_kernel_context(kernel);
    if (context == nullptr) {
        return false;
    }

    GPU::Runtime::AutoInitContext();
    GPU::Runtime::ContextGuard guard(GPU::Runtime::Context::GetInstance());
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        throw std::runtime_error("EasyGPU backend is not available.");
    }

    GPU::Backend::ShaderDesc shader_desc;
    shader_desc.type = GPU::Backend::ShaderType::Compute;
    shader_desc.sourceCode = context->GetCompleteCode();
    shader_desc.entryPoint = "main";
    shader_desc.optimizationLevel = context->GetOptimizationLevel();
    *source = backend->GetOptimizedGLSL(shader_desc);
    return !source->empty();
}

bool dispatch_ad_gradient_reduce_to_buffer(ADGradientState& gradient, BufferState& destination, uint64_t destination_offset,
                                           uint64_t destination_size, GPU::Backend::Backend& backend,
                                           std::string* error) {
    if (gradient.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
        if (error != nullptr) {
            *error = "AD gradient buffer is not available.";
        }
        return false;
    }

    const auto component_count = std::max<uint32_t>(gradient.component_count, 1);
    const auto scalar_slots = static_cast<uint64_t>(gradient.element_count) * static_cast<uint64_t>(component_count);
    if (scalar_slots == 0 || gradient.byte_size % sizeof(float) != 0) {
        if (error != nullptr) {
            *error = "AD gradient has an invalid scalar layout.";
        }
        return false;
    }

    const auto total_scalars = static_cast<uint64_t>(gradient.byte_size / sizeof(float));
    if (total_scalars % scalar_slots != 0) {
        if (error != nullptr) {
            *error = "AD gradient storage cannot be reduced to the requested element layout.";
        }
        return false;
    }

    const auto dispatch_count = total_scalars / scalar_slots;
    const auto expected_size = scalar_slots * sizeof(float);
    if (destination_offset % sizeof(float) != 0) {
        if (error != nullptr) {
            *error = "Destination buffer offset must be float-aligned.";
        }
        return false;
    }

    if (scalar_slots > UINT32_MAX || dispatch_count > UINT32_MAX) {
        if (error != nullptr) {
            *error = "AD gradient reduction exceeds shader loop limits.";
        }
        return false;
    }

    if (destination_size != expected_size) {
        if (error != nullptr) {
            *error = "Destination buffer range size does not match the reduced AD gradient size.";
        }
        return false;
    }

    if (destination_offset > destination.bytes.size() || destination_size > destination.bytes.size() - destination_offset) {
        if (error != nullptr) {
            *error = "Destination buffer range exceeds buffer size.";
        }
        return false;
    }

    auto destination_buffer = ensure_easygpu_buffer(destination, backend);
    constexpr uint32_t work_group_size = 64;
    const auto source_binding = 0u;
    const auto destination_binding = 1u;

    std::ostringstream glsl;
    glsl << "#version 450\n";
    glsl << "layout(local_size_x = " << work_group_size << ", local_size_y = 1, local_size_z = 1) in;\n";
    glsl << "layout(std430, binding = " << source_binding << ") readonly buffer SourceGradient { float source_data[]; };\n";
    glsl << "layout(std430, binding = " << destination_binding << ") buffer DestinationGradient { float destination_data[]; };\n";
    glsl << "void main() {\n";
    glsl << "  uint slot = gl_GlobalInvocationID.x;\n";
    glsl << "  if (slot >= " << scalar_slots << "u) { return; }\n";
    glsl << "  float sum = 0.0;\n";
    glsl << "  for (uint dispatch_index = 0u; dispatch_index < " << dispatch_count << "u; ++dispatch_index) {\n";
    glsl << "    sum += source_data[(dispatch_index * " << scalar_slots << "u) + slot];\n";
    glsl << "  }\n";
    glsl << "  destination_data[" << (destination_offset / sizeof(float)) << "u + slot] = sum;\n";
    glsl << "}\n";

    GPU::Backend::ShaderDesc shader_desc;
    shader_desc.type = GPU::Backend::ShaderType::Compute;
    shader_desc.sourceCode = glsl.str();
    shader_desc.entryPoint = "main";
    shader_desc.optimizationLevel = kShaderOptimizationLevel;

    const auto shader = backend.CreateShader(shader_desc);
    if (shader == GPU::Backend::INVALID_SHADER_HANDLE) {
        if (error != nullptr) {
            *error = "EasyGPU backend failed to compile AD gradient reduction shader.";
        }
        return false;
    }
    EasyGpuShaderGuard shader_guard(backend, shader);

    GPU::Backend::PipelineDesc pipeline_desc;
    pipeline_desc.computeShader = shader;
    pipeline_desc.workGroupSizeX = work_group_size;
    pipeline_desc.workGroupSizeY = 1;
    pipeline_desc.workGroupSizeZ = 1;
    pipeline_desc.resources.push_back(GPU::Backend::ResourceLayoutEntry{
        source_binding,
        GPU::Backend::BindingType::Buffer,
        GPU::Backend::PixelFormat::RGBA8,
        true});
    pipeline_desc.resources.push_back(GPU::Backend::ResourceLayoutEntry{
        destination_binding,
        GPU::Backend::BindingType::Buffer,
        GPU::Backend::PixelFormat::RGBA8,
        false});

    const auto pipeline = backend.CreatePipeline(pipeline_desc);
    if (pipeline == GPU::Backend::INVALID_PIPELINE_HANDLE) {
        if (error != nullptr) {
            *error = "EasyGPU backend failed to create AD gradient reduction pipeline.";
        }
        return false;
    }

    GPU::Backend::ResourceBinding bindings[2]{};
    bindings[0].binding = source_binding;
    bindings[0].type = GPU::Backend::BindingType::Buffer;
    bindings[0].buffer = gradient.backend_buffer;
    bindings[0].readOnly = true;
    bindings[1].binding = destination_binding;
    bindings[1].type = GPU::Backend::BindingType::Buffer;
    bindings[1].buffer = destination_buffer;
    bindings[1].readOnly = false;

    backend.BindPipeline(pipeline);
    backend.BindResources(bindings, 2);
    backend.Dispatch(static_cast<uint32_t>((scalar_slots + work_group_size - 1) / work_group_size), 1, 1);
    backend.MemoryBarrier(GPU::Backend::BarrierType::All);
    backend.Finish();
    backend.DestroyPipeline(pipeline);

    destination.device_dirty = true;
    destination.host_dirty = false;
    return true;
}

FeResult dispatch_simple_buffer_assignment(const KernelState& kernel, uint32_t group_x, uint32_t group_y,
                                           uint32_t group_z) {
    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir) || ir.shader_kind < 1 || ir.shader_kind > 3 || ir.group_x <= 0 ||
        ir.group_y <= 0 || ir.group_z <= 0) {
        return fail(FE_ERROR_UNSUPPORTED, "Kernel dispatch fallback requires valid compute Feather IR.");
    }

    // Execute the typed section records in source order. Simple copies appear in both section kinds, so
    // expression records own their instruction index and structured records fill only the remaining gaps.
    bool executed = false;
    std::set<uint32_t> expression_instruction_indices;
    for (const auto& expression : ir.expression_assignments) {
        expression_instruction_indices.insert(expression.instruction_index);
    }

    struct PlannedAssignment {
        uint32_t instruction_index = 0;
        const IrExpressionAssignment* expression = nullptr;
        FallbackAssignment structured;
    };

    std::vector<PlannedAssignment> planned;
    planned.reserve(ir.expression_assignments.size() + ir.elementwise_assignments.size());
    for (const auto& expression : ir.expression_assignments) {
        planned.push_back(PlannedAssignment{expression.instruction_index, &expression, {}});
    }

    for (const auto& structured : ir.elementwise_assignments) {
        if (expression_instruction_indices.find(structured.instruction_index) != expression_instruction_indices.end()) {
            continue;
        }

        FallbackAssignment assignment;
        if (convert_structured_assignment(ir, structured, &assignment)) {
            planned.push_back(PlannedAssignment{structured.instruction_index, nullptr, assignment});
        }
    }

    std::sort(planned.begin(), planned.end(), [](const auto& left, const auto& right) {
        return left.instruction_index < right.instruction_index;
    });

    for (const auto& assignment : planned) {
        const auto result = assignment.expression != nullptr
                                ? execute_expression_assignment(kernel, ir, *assignment.expression, group_x, group_y,
                                                                group_z)
                                : execute_fallback_assignment(kernel, ir, assignment.structured, group_x, group_y,
                                                              group_z);
        if (result != FE_OK) {
            return result;
        }

        executed = true;
    }

    if (executed) {
        return ok();
    }

    for (const auto& instruction : ir.instructions) {
        if (instruction.opcode != 2 || instruction.operand_kind != kIrOperandKindElementwiseAssignment) {
            continue;
        }

        const auto* payload = get_string(ir, instruction.operand_string_id);
        FallbackAssignment assignment;
        if (payload != nullptr && parse_elementwise_assignment_payload(ir, *payload, &assignment)) {
            return execute_fallback_assignment(kernel, ir, assignment, group_x, group_y, group_z);
        }
    }

    return fail(FE_ERROR_UNSUPPORTED, "Kernel dispatch fallback did not find a supported assignment.");
}

FeResult clear_graphics_target(FeGraphicsPipelineHandle pipeline, FeTextureHandle color_target) {
    if (g_pipelines.find(pipeline) == g_pipelines.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid graphics pipeline handle.");
    }

    auto target = g_textures.find(color_target);
    if (target == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid color target handle.");
    }

    // Draw fallback validates the graphics command path and mutates the target deterministically.
    // Real rasterization remains owned by the EasyGPU graphics backend bridge.
    std::fill(target->second.bytes.begin(), target->second.bytes.end(), 0);
    return ok();
}

struct GraphicsVertex2D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float w = 1.0f;
};

struct GraphicsColor {
    uint8_t r = 255;
    uint8_t g = 255;
    uint8_t b = 255;
    uint8_t a = 255;
};

float read_f32_unaligned(const unsigned char* data) {
    float value = 0.0f;
    std::memcpy(&value, data, sizeof(float));
    return value;
}

uint32_t read_u32_unaligned(const unsigned char* data) {
    uint32_t value = 0;
    std::memcpy(&value, data, sizeof(uint32_t));
    return value;
}

uint16_t read_u16_unaligned(const unsigned char* data) {
    uint16_t value = 0;
    std::memcpy(&value, data, sizeof(uint16_t));
    return value;
}

GraphicsColor graphics_color_from_push_constants(const GraphicsPipelineState& pipeline) {
    if (pipeline.push_constants.size() >= sizeof(float) * 4) {
        const auto* data = pipeline.push_constants.data();
        const auto clamp_to_byte = [](float value) {
            const auto scaled = std::clamp(value, 0.0f, 1.0f) * 255.0f;
            return static_cast<uint8_t>(std::lround(scaled));
        };
        return GraphicsColor{
            clamp_to_byte(read_f32_unaligned(data)),
            clamp_to_byte(read_f32_unaligned(data + 4)),
            clamp_to_byte(read_f32_unaligned(data + 8)),
            clamp_to_byte(read_f32_unaligned(data + 12))};
    }

    return GraphicsColor{64, 180, 255, 255};
}

GraphicsColor graphics_color_from_sampled_texture(const GraphicsPipelineState& pipeline) {
    for (const auto& [binding, texture_handle] : pipeline.textures) {
        (void)binding;
        const auto texture_it = g_textures.find(texture_handle);
        if (texture_it == g_textures.end()) {
            continue;
        }
        const auto& texture = texture_it->second;
        if (texture.depth != 1 || texture.width == 0 || texture.height == 0 || texture.bytes.empty()) {
            continue;
        }
        if (texture.pixel_format == 3 && texture.bytes.size() >= 4) {
            return GraphicsColor{texture.bytes[0], texture.bytes[1], texture.bytes[2], texture.bytes[3]};
        }
        if (texture.pixel_format == 4 && texture.bytes.size() >= 4) {
            return GraphicsColor{texture.bytes[2], texture.bytes[1], texture.bytes[0], texture.bytes[3]};
        }
    }

    return graphics_color_from_push_constants(pipeline);
}

bool read_graphics_vertex(const BufferState& buffer, uint32_t stride, uint32_t index, GraphicsVertex2D* out_vertex) {
    if (out_vertex == nullptr || stride < sizeof(float) * 2) {
        return false;
    }

    const auto offset = static_cast<size_t>(index) * stride;
    if (offset + sizeof(float) * 2 > buffer.bytes.size()) {
        return false;
    }

    const auto* data = buffer.bytes.data() + offset;
    out_vertex->x = read_f32_unaligned(data);
    out_vertex->y = read_f32_unaligned(data + 4);
    out_vertex->z = offset + 12 <= buffer.bytes.size() ? read_f32_unaligned(data + 8) : 0.0f;
    out_vertex->w = offset + 16 <= buffer.bytes.size() ? read_f32_unaligned(data + 12) : 1.0f;
    if (std::abs(out_vertex->w) < 0.000001f) {
        out_vertex->w = 1.0f;
    }
    return true;
}

float edge_function(float ax, float ay, float bx, float by, float cx, float cy) {
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

void write_graphics_pixel(TextureState& target, uint32_t x, uint32_t y, GraphicsColor color) {
    const auto pixel = pixel_size(target.pixel_format);
    const auto offset = (static_cast<size_t>(y) * target.width + x) * pixel;
    if (offset + pixel > target.bytes.size()) {
        return;
    }

    if (target.pixel_format == 3) {
        target.bytes[offset + 0] = color.r;
        target.bytes[offset + 1] = color.g;
        target.bytes[offset + 2] = color.b;
        target.bytes[offset + 3] = color.a;
    } else if (target.pixel_format == 4) {
        target.bytes[offset + 0] = color.b;
        target.bytes[offset + 1] = color.g;
        target.bytes[offset + 2] = color.r;
        target.bytes[offset + 3] = color.a;
    }
}

FeResult rasterize_graphics_triangle(TextureState& target, const std::array<GraphicsVertex2D, 3>& vertices,
                                     GraphicsColor color) {
    if (target.width == 0 || target.height == 0 || target.depth != 1) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw requires a valid 2D color target.");
    }
    if (target.pixel_format != 3 && target.pixel_format != 4) {
        return fail(FE_ERROR_UNSUPPORTED, "Graphics rasterization currently supports Rgba8 and Bgra8 color targets.");
    }

    struct ScreenVertex {
        float x = 0.0f;
        float y = 0.0f;
    };

    std::array<ScreenVertex, 3> screen{};
    for (size_t i = 0; i < vertices.size(); ++i) {
        const auto inv_w = 1.0f / vertices[i].w;
        const auto ndc_x = vertices[i].x * inv_w;
        const auto ndc_y = vertices[i].y * inv_w;
        screen[i].x = (ndc_x * 0.5f + 0.5f) * static_cast<float>(target.width - 1);
        screen[i].y = (1.0f - (ndc_y * 0.5f + 0.5f)) * static_cast<float>(target.height - 1);
    }

    const auto min_xf = std::floor(std::min({screen[0].x, screen[1].x, screen[2].x}));
    const auto max_xf = std::ceil(std::max({screen[0].x, screen[1].x, screen[2].x}));
    const auto min_yf = std::floor(std::min({screen[0].y, screen[1].y, screen[2].y}));
    const auto max_yf = std::ceil(std::max({screen[0].y, screen[1].y, screen[2].y}));
    const auto min_x = static_cast<uint32_t>(std::clamp(min_xf, 0.0f, static_cast<float>(target.width - 1)));
    const auto max_x = static_cast<uint32_t>(std::clamp(max_xf, 0.0f, static_cast<float>(target.width - 1)));
    const auto min_y = static_cast<uint32_t>(std::clamp(min_yf, 0.0f, static_cast<float>(target.height - 1)));
    const auto max_y = static_cast<uint32_t>(std::clamp(max_yf, 0.0f, static_cast<float>(target.height - 1)));

    const auto area = edge_function(screen[0].x, screen[0].y, screen[1].x, screen[1].y, screen[2].x, screen[2].y);
    if (std::abs(area) < 0.000001f) {
        const auto cx = target.width / 2;
        const auto cy = target.height / 2;
        write_graphics_pixel(target, cx, cy, color);
        return ok();
    }

    bool wrote_pixel = false;
    for (uint32_t y = min_y; y <= max_y; ++y) {
        for (uint32_t x = min_x; x <= max_x; ++x) {
            const auto px = static_cast<float>(x) + 0.5f;
            const auto py = static_cast<float>(y) + 0.5f;
            const auto w0 = edge_function(screen[1].x, screen[1].y, screen[2].x, screen[2].y, px, py);
            const auto w1 = edge_function(screen[2].x, screen[2].y, screen[0].x, screen[0].y, px, py);
            const auto w2 = edge_function(screen[0].x, screen[0].y, screen[1].x, screen[1].y, px, py);
            const auto has_neg = w0 < 0.0f || w1 < 0.0f || w2 < 0.0f;
            const auto has_pos = w0 > 0.0f || w1 > 0.0f || w2 > 0.0f;
            if (has_neg && has_pos) {
                continue;
            }

            write_graphics_pixel(target, x, y, color);
            wrote_pixel = true;
        }
    }

    if (!wrote_pixel) {
        write_graphics_pixel(target, target.width / 2, target.height / 2, color);
    }
    return ok();
}

bool map_graphics_topology(uint32_t topology, GPU::Backend::PrimitiveTopology* out_topology) {
    if (out_topology == nullptr) {
        return false;
    }

    switch (topology) {
    case 0:
        *out_topology = GPU::Backend::PrimitiveTopology::TriangleList;
        return true;
    case 1:
        *out_topology = GPU::Backend::PrimitiveTopology::TriangleStrip;
        return true;
    case 2:
        *out_topology = GPU::Backend::PrimitiveTopology::LineList;
        return true;
    case 3:
        *out_topology = GPU::Backend::PrimitiveTopology::LineStrip;
        return true;
    case 4:
        *out_topology = GPU::Backend::PrimitiveTopology::PointList;
        return true;
    case 5:
        *out_topology = GPU::Backend::PrimitiveTopology::TriangleFan;
        return true;
    default:
        return false;
    }
}

bool map_graphics_sample_count(uint32_t sample_count, GPU::Backend::SampleCount* out_sample_count) {
    if (out_sample_count == nullptr) {
        return false;
    }

    switch (sample_count == 0 ? 1 : sample_count) {
    case 1:
        *out_sample_count = GPU::Backend::SampleCount::X1;
        return true;
    case 2:
        *out_sample_count = GPU::Backend::SampleCount::X2;
        return true;
    case 4:
        *out_sample_count = GPU::Backend::SampleCount::X4;
        return true;
    case 8:
        *out_sample_count = GPU::Backend::SampleCount::X8;
        return true;
    case 16:
        *out_sample_count = GPU::Backend::SampleCount::X16;
        return true;
    default:
        return false;
    }
}

bool map_sampler_desc(const FeSamplerDesc& source, GPU::Backend::SamplerDesc* out) {
    if (out == nullptr || source.min_filter > 1 || source.mag_filter > 1 ||
        source.mipmap_mode > 1 ||
        source.address_u > 3 || source.address_v > 3 || source.address_w > 3 ||
        source.compare_op > 7 || source.border_color > 5 ||
        !std::isfinite(source.mip_lod_bias) || !std::isfinite(source.min_lod) ||
        !std::isfinite(source.max_lod) || !std::isfinite(source.max_anisotropy) ||
        source.min_lod < 0.0f || source.max_lod < source.min_lod ||
        source.max_anisotropy < 1.0f) {
        return false;
    }

    out->minFilter = static_cast<GPU::Backend::SamplerFilter>(source.min_filter);
    out->magFilter = static_cast<GPU::Backend::SamplerFilter>(source.mag_filter);
    out->mipmapMode = static_cast<GPU::Backend::SamplerMipmapMode>(source.mipmap_mode);
    out->addressU = static_cast<GPU::Backend::SamplerAddressMode>(source.address_u);
    out->addressV = static_cast<GPU::Backend::SamplerAddressMode>(source.address_v);
    out->addressW = static_cast<GPU::Backend::SamplerAddressMode>(source.address_w);
    out->mipLodBias = source.mip_lod_bias;
    out->minLod = source.min_lod;
    out->maxLod = source.max_lod;
    out->anisotropyEnable = source.anisotropy_enable != 0;
    out->maxAnisotropy = source.max_anisotropy;
    out->compareEnable = source.compare_enable != 0;
    out->compareOp = static_cast<GPU::Backend::CompareOp>(source.compare_op);
    out->borderColor = static_cast<GPU::Backend::SamplerBorderColor>(source.border_color);
    return true;
}

bool map_graphics_compare_op(uint32_t value, GPU::Backend::CompareOp* out) {
    if (out == nullptr || value > 7) {
        return false;
    }
    *out = static_cast<GPU::Backend::CompareOp>(value);
    return true;
}

bool map_graphics_stencil_op(uint32_t value, GPU::Backend::StencilOp* out) {
    if (out == nullptr || value > 7) {
        return false;
    }
    *out = static_cast<GPU::Backend::StencilOp>(value);
    return true;
}

bool map_graphics_stencil_face(const FeGraphicsStencilFaceDesc& source, GPU::Backend::StencilFaceState* out) {
    return out != nullptr &&
           map_graphics_stencil_op(source.fail_op, &out->failOp) &&
           map_graphics_stencil_op(source.pass_op, &out->passOp) &&
           map_graphics_stencil_op(source.depth_fail_op, &out->depthFailOp) &&
           map_graphics_compare_op(source.compare_op, &out->compareOp);
}

bool map_graphics_blend_factor(uint32_t value, GPU::Backend::BlendFactor* out) {
    if (out == nullptr || value > 9) {
        return false;
    }
    *out = static_cast<GPU::Backend::BlendFactor>(value);
    return true;
}

bool map_graphics_blend_op(uint32_t value, GPU::Backend::BlendOp* out) {
    if (out == nullptr || value > 4) {
        return false;
    }
    *out = static_cast<GPU::Backend::BlendOp>(value);
    return true;
}

bool map_graphics_color_blend_attachment(const FeGraphicsColorBlendAttachmentDesc& source,
                                         GPU::Backend::ColorAttachmentBlendState* out) {
    if (out == nullptr || (source.write_mask & ~15u) != 0) {
        return false;
    }

    GPU::Backend::BlendFactor src_color;
    GPU::Backend::BlendFactor dst_color;
    GPU::Backend::BlendFactor src_alpha;
    GPU::Backend::BlendFactor dst_alpha;
    GPU::Backend::BlendOp color_op;
    GPU::Backend::BlendOp alpha_op;
    if (!map_graphics_blend_factor(source.src_color, &src_color) ||
        !map_graphics_blend_factor(source.dst_color, &dst_color) ||
        !map_graphics_blend_factor(source.src_alpha, &src_alpha) ||
        !map_graphics_blend_factor(source.dst_alpha, &dst_alpha) ||
        !map_graphics_blend_op(source.color_op, &color_op) ||
        !map_graphics_blend_op(source.alpha_op, &alpha_op)) {
        return false;
    }

    out->blendEnable = source.blend_enable != 0;
    out->srcColorBlendFactor = src_color;
    out->dstColorBlendFactor = dst_color;
    out->colorBlendOp = color_op;
    out->srcAlphaBlendFactor = src_alpha;
    out->dstAlphaBlendFactor = dst_alpha;
    out->alphaBlendOp = alpha_op;
    out->colorWriteMask = source.write_mask;
    return true;
}

bool map_graphics_raster_state(const GraphicsPipelineState& pipeline,
                               GPU::Backend::CullMode* cull_mode,
                               GPU::Backend::FrontFace* front_face,
                               GPU::Backend::PolygonMode* polygon_mode) {
    if (cull_mode == nullptr || front_face == nullptr || polygon_mode == nullptr ||
        pipeline.cull_mode > 3 || pipeline.front_face > 1 || pipeline.polygon_mode > 2) {
        return false;
    }

    *cull_mode = static_cast<GPU::Backend::CullMode>(pipeline.cull_mode);
    *front_face = static_cast<GPU::Backend::FrontFace>(pipeline.front_face);
    *polygon_mode = static_cast<GPU::Backend::PolygonMode>(pipeline.polygon_mode);
    return true;
}

struct GraphicsVaryingField {
    std::string name;
    std::string glsl_type;
    uint32_t type_id = UINT32_MAX;
    uint32_t location = UINT32_MAX;
    bool position = false;
};

struct GraphicsVaryingLayout {
    bool is_float4 = false;
    uint32_t type_id = UINT32_MAX;
    uint32_t position_field_type_id = UINT32_MAX;
    std::string position_field_name;
    std::vector<GraphicsVaryingField> fields;
};

struct GraphicsFragmentOutputField {
    std::string name;
    std::string glsl_type;
    uint32_t type_id = UINT32_MAX;
    uint32_t location = UINT32_MAX;
};

struct GraphicsFragmentOutputLayout {
    bool is_float4 = false;
    uint32_t type_id = UINT32_MAX;
    std::vector<GraphicsFragmentOutputField> fields;
};

enum class GraphicsStage {
    Vertex,
    Fragment
};

enum class GraphicsDepthLoadOp : uint32_t {
    Default = 0,
    Load = 1,
    Clear = 2
};

enum class GraphicsColorLoadOp : uint32_t {
    Default = 0,
    Load = 1,
    Clear = 2,
    DontCare = 3
};

struct GraphicsLoweringContext {
    GraphicsStage stage = GraphicsStage::Fragment;
    const ParsedIr& ir;
    const Feather::TypedIR::Module& module;
    const GraphicsPipelineState& pipeline;
    const std::vector<GraphicsPushConstantLayoutEntry>& push_constants;
    const GraphicsVaryingLayout& varyings;
    const GraphicsResourceLayout& resources;
    std::string parameter_name;
    std::optional<uint32_t> sampled_texture_binding;
    std::unordered_map<std::string, uint32_t> locals;
    std::set<uint32_t> used_buffer_bindings;
};

struct GraphicsFragmentLoweringContext {
    const ParsedIr& ir;
    const Feather::TypedIR::Module& module;
    const GraphicsPipelineState& pipeline;
    const std::vector<GraphicsPushConstantLayoutEntry>& push_constants;
    std::string fragment_parameter_name;
    std::optional<uint32_t> sampled_texture_binding;
    std::unordered_map<std::string, uint32_t> locals;
};

bool same_graphics_resource(const GraphicsPushConstantLayoutEntry& entry, uint32_t binding, const std::string& name) {
    return entry.binding == binding && entry.name == name;
}

const IrResource* find_graphics_resource_by_binding_and_name(const ParsedIr& ir, uint32_t binding,
                                                             const std::string& name);

bool append_graphics_push_constants(const ParsedIr& ir, std::vector<GraphicsPushConstantLayoutEntry>* entries) {
    if (entries == nullptr) {
        return false;
    }

    for (const auto& resource : ir.resources) {
        if (resource.kind != kIrResourceKindPushConstant) {
            continue;
        }

        const auto name = string_or_empty(ir, resource.name_string_id);
        if (name.empty()) {
            return false;
        }

        const auto duplicate = std::any_of(entries->begin(), entries->end(), [&](const auto& entry) {
            return same_graphics_resource(entry, resource.binding, name);
        });
        if (duplicate) {
            continue;
        }

        const auto size = push_constant_type_size(ir, resource);
        if (size == 0) {
            return false;
        }

        GraphicsPushConstantLayoutEntry entry;
        entry.binding = resource.binding;
        entry.name = name;
        entry.size = size;
        entries->push_back(std::move(entry));
    }

    return true;
}

bool build_graphics_push_constant_layout(const ParsedIr& vertex_ir, const ParsedIr& fragment_ir,
                                         std::vector<GraphicsPushConstantLayoutEntry>* entries) {
    if (entries == nullptr) {
        return false;
    }

    entries->clear();
    if (!append_graphics_push_constants(vertex_ir, entries) ||
        !append_graphics_push_constants(fragment_ir, entries)) {
        return false;
    }

    std::stable_sort(entries->begin(), entries->end(), [](const auto& left, const auto& right) {
        return left.binding < right.binding;
    });

    size_t offset = 0;
    for (auto& entry : *entries) {
        const ParsedIr* resource_ir = &vertex_ir;
        const auto* resource = find_graphics_resource_by_binding_and_name(vertex_ir, entry.binding, entry.name);
        if (resource == nullptr) {
            resource_ir = &fragment_ir;
            resource = find_graphics_resource_by_binding_and_name(fragment_ir, entry.binding, entry.name);
        }

        if (resource == nullptr) {
            return false;
        }

        const auto alignment = push_constant_type_alignment(*resource_ir, *resource);
        if (alignment == 0) {
            return false;
        }

        offset = align_offset(offset, alignment);
        entry.offset = offset;
        offset += entry.size;
    }

    return true;
}

const std::string* typed_ir_string(const Feather::TypedIR::Module& module, uint32_t id) {
    return id < module.strings.size() ? &module.strings[id] : nullptr;
}

std::string sanitize_graphics_glsl_identifier(std::string_view value) {
    std::string result;
    result.reserve(value.size() + 1);
    for (const auto ch : value) {
        const auto uch = static_cast<unsigned char>(ch);
        result.push_back((std::isalnum(uch) || ch == '_') ? static_cast<char>(ch) : '_');
    }

    if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front()))) {
        result.insert(result.begin(), '_');
    }

    return result;
}

std::string graphics_glsl_type_name(const Feather::TypedIR::Module& module, uint32_t type_id) {
    if (type_id >= module.types.size()) {
        return {};
    }

    const auto& type = module.types[type_id];
    switch (type.kind) {
    case 1: // primitive
        switch (type.a) {
        case 0:
            return "bool";
        case 1:
            return "int";
        case 2:
            return "uint";
        case 3:
            return "float";
        default:
            return {};
        }
    case 2: { // vector
        if (type.b < 2 || type.b > 4 || type.a >= module.types.size()) {
            return {};
        }

        const auto element = graphics_glsl_type_name(module, type.a);
        if (element == "float") {
            return "vec" + std::to_string(type.b);
        }
        if (element == "int") {
            return "ivec" + std::to_string(type.b);
        }
        if (element == "uint") {
            return "uvec" + std::to_string(type.b);
        }
        if (element == "bool") {
            return "bvec" + std::to_string(type.b);
        }
        return {};
    }
    case 3: { // matrix
        if (type.a >= module.types.size() || type.b < 2 || type.b > 4 || type.c < 2 || type.c > 4) {
            return {};
        }

        const auto element = graphics_glsl_type_name(module, type.a);
        if (element != "float") {
            return {};
        }

        return type.b == type.c
            ? "mat" + std::to_string(type.b)
            : "mat" + std::to_string(type.c) + "x" + std::to_string(type.b);
    }
    case 4: { // struct
        if (type.a >= module.structs.size()) {
            return {};
        }

        const auto& structure = module.structs[type.a];
        const auto* name = typed_ir_string(module, structure.name_id);
        return name == nullptr ? std::string{} : ("fe_" + sanitize_graphics_glsl_identifier(*name));
    }
    default:
        return {};
    }
}

std::string graphics_glsl_type_name_from_record(const Feather::TypedIR::Module& module,
                                                const Feather::TypedIR::Type& type) {
    for (uint32_t i = 0; i < module.types.size(); ++i) {
        const auto& candidate = module.types[i];
        if (candidate.kind == type.kind && candidate.a == type.a && candidate.b == type.b &&
            candidate.c == type.c && candidate.d == type.d) {
            return graphics_glsl_type_name(module, i);
        }
    }

    return {};
}

bool graphics_find_struct_field(const Feather::TypedIR::Module& module, uint32_t struct_index,
                                const std::string& name, Feather::TypedIR::StructField* out_field = nullptr) {
    if (struct_index >= module.structs.size()) {
        return false;
    }

    const auto& structure = module.structs[struct_index];
    if (structure.field_count == 0) {
        return false;
    }
    if (structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) {
        return false;
    }

    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        const auto* field_name = typed_ir_string(module, field.name_id);
        if (field_name != nullptr && *field_name == name) {
            if (out_field != nullptr) {
                *out_field = field;
            }
            return true;
        }
    }

    return false;
}

std::optional<uint32_t> graphics_struct_index_for_type(const Feather::TypedIR::Module& module, uint32_t type_id) {
    if (type_id >= module.types.size()) {
        return std::nullopt;
    }

    const auto& type = module.types[type_id];
    if (type.kind != 4 || type.a >= module.structs.size()) {
        return std::nullopt;
    }

    return type.a;
}

bool build_graphics_varying_layout(const Feather::TypedIR::Module& module, uint32_t type_id,
                                   GraphicsVaryingLayout* layout) {
    if (layout == nullptr || type_id >= module.types.size()) {
        return false;
    }

    layout->is_float4 = false;
    layout->type_id = type_id;
    layout->position_field_type_id = UINT32_MAX;
    layout->position_field_name.clear();
    layout->fields.clear();

    const auto type_name = graphics_glsl_type_name(module, type_id);
    if (type_name == "vec4") {
        layout->is_float4 = true;
        layout->position_field_type_id = type_id;
        layout->fields.push_back(GraphicsVaryingField{"", "vec4", type_id, 0, true});
        return true;
    }

    const auto struct_index = graphics_struct_index_for_type(module, type_id);
    if (!struct_index.has_value()) {
        return false;
    }

    const auto& structure = module.structs[*struct_index];
    if (structure.field_count == 0 ||
        structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) {
        return false;
    }

    uint32_t location = 0;
    bool found_position = false;
    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        const auto* field_name = typed_ir_string(module, field.name_id);
        const auto field_type_name = graphics_glsl_type_name(module, field.type_id);
        if (field_name == nullptr || field_name->empty() || field_type_name.empty()) {
            return false;
        }

        const auto is_position = (field.flags & kTypedStructFieldFlagPosition) != 0;
        if (is_position && field_type_name != "vec4") {
            return false;
        }
        if (!is_position) {
            layout->fields.push_back(GraphicsVaryingField{*field_name, field_type_name, field.type_id, location++, false});
        } else {
            layout->position_field_type_id = field.type_id;
            layout->position_field_name = *field_name;
            found_position = true;
        }
    }

    return found_position;
}

bool build_graphics_fragment_output_layout(const Feather::TypedIR::Module& module,
                                           uint32_t type_id,
                                           uint32_t color_attachment_count,
                                           GraphicsFragmentOutputLayout* layout) {
    if (layout == nullptr || type_id >= module.types.size() ||
        color_attachment_count == 0 ||
        color_attachment_count > GPU::Backend::MAX_COLOR_ATTACHMENTS) {
        return false;
    }

    layout->is_float4 = false;
    layout->type_id = type_id;
    layout->fields.clear();

    const auto type_name = graphics_glsl_type_name(module, type_id);
    if (type_name == "vec4") {
        if (color_attachment_count != 1) {
            return false;
        }

        layout->is_float4 = true;
        layout->fields.push_back(GraphicsFragmentOutputField{"", "vec4", type_id, 0});
        return true;
    }

    const auto struct_index = graphics_struct_index_for_type(module, type_id);
    if (!struct_index.has_value()) {
        return false;
    }

    const auto& structure = module.structs[*struct_index];
    if (structure.field_count == 0 ||
        structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) {
        return false;
    }

    std::set<uint32_t> locations;
    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        if ((field.flags & kTypedStructFieldFlagPosition) != 0) {
            return false;
        }
        if ((field.flags & kTypedStructFieldFlagColor) == 0) {
            return false;
        }

        const auto location = field.flags >> kTypedStructFieldColorIndexShift;
        if (location >= color_attachment_count || !locations.insert(location).second) {
            return false;
        }

        const auto* field_name = typed_ir_string(module, field.name_id);
        const auto field_type_name = graphics_glsl_type_name(module, field.type_id);
        if (field_name == nullptr || field_name->empty() || field_type_name != "vec4") {
            return false;
        }

        layout->fields.push_back(GraphicsFragmentOutputField{*field_name, field_type_name, field.type_id, location});
    }

    if (locations.size() != color_attachment_count) {
        return false;
    }

    std::sort(layout->fields.begin(), layout->fields.end(), [](const auto& left, const auto& right) {
        return left.location < right.location;
    });
    return true;
}

std::string graphics_float_literal(float value) {
    if (!std::isfinite(value)) {
        return "0.0";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(8) << value;
    return stream.str();
}

std::string normalize_graphics_literal(std::string value) {
    value = trim_copy(value);
    while (!value.empty() && (value.back() == 'f' || value.back() == 'F' ||
                              value.back() == 'd' || value.back() == 'D')) {
        value.pop_back();
    }

    return value.empty() ? "0" : value;
}

std::string graphics_swizzle_components(std::string value) {
    for (auto& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        switch (ch) {
        case 'r':
        case 's':
            ch = 'x';
            break;
        case 'g':
        case 't':
            ch = 'y';
            break;
        case 'b':
        case 'p':
            ch = 'z';
            break;
        case 'a':
        case 'q':
            ch = 'w';
            break;
        default:
            break;
        }
    }

    return value;
}

bool feather_graphics_trace_enabled();

void trace_graphics_expression_failure(const GraphicsLoweringContext& context, const char* stage, uint32_t expr_id,
                                       const char* reason) {
    if (!feather_graphics_trace_enabled()) {
        return;
    }

    std::cerr << "[feather graphics] " << stage << " expression lowering failed";
    if (expr_id < context.module.expressions.size()) {
        const auto& expression = context.module.expressions[expr_id];
        std::cerr << ": id=" << expr_id << " kind=" << static_cast<int>(expression.kind)
                  << " type=" << expression.type_id << " op=" << expression.op;
        if (expression.name_id < context.module.strings.size()) {
            std::cerr << " name=" << context.module.strings[expression.name_id];
        }
    } else {
        std::cerr << ": id=" << expr_id;
    }
    std::cerr << " reason=" << reason << "\n";
}

bool is_fragment_parameter_reference(const GraphicsFragmentLoweringContext& context, uint32_t expr_id) {
    if (expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    if (expression.kind != 3 || expression.name_id >= context.module.strings.size()) {
        return false;
    }

    return context.module.strings[expression.name_id] == context.fragment_parameter_name;
}

bool try_graphics_resource_name(const GraphicsFragmentLoweringContext& context, uint32_t expr_id, std::string* name) {
    if (name == nullptr || expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    if ((expression.kind == 2 || expression.kind == 3) && expression.name_id < context.module.strings.size()) {
        *name = context.module.strings[expression.name_id];
        return true;
    }

    if (expression.kind == 4 && expression.name_id < context.module.strings.size()) {
        *name = context.module.strings[expression.name_id];
        return true;
    }

    return false;
}

const IrResource* find_graphics_resource_by_binding_and_name(const ParsedIr& ir, uint32_t binding,
                                                             const std::string& name) {
    for (const auto& resource : ir.resources) {
        const auto resource_name = string_or_empty(ir, resource.name_string_id);
        if (resource.binding == binding && resource_name == name) {
            return &resource;
        }
    }

    return nullptr;
}

bool try_graphics_push_constant_literal(const GraphicsFragmentLoweringContext& context,
                                        const Feather::TypedIR::Expression& expression,
                                        std::string* glsl) {
    if (glsl == nullptr || expression.name_id >= context.module.strings.size()) {
        return false;
    }

    const auto& name = context.module.strings[expression.name_id];
    const auto it = std::find_if(context.push_constants.begin(), context.push_constants.end(), [&](const auto& entry) {
        return same_graphics_resource(entry, expression.op, name);
    });
    if (it == context.push_constants.end() ||
        it->offset > context.pipeline.push_constants.size() ||
        it->size > context.pipeline.push_constants.size() - it->offset) {
        return false;
    }

    const auto* data = context.pipeline.push_constants.data() + it->offset;
    switch (it->size) {
    case sizeof(float): {
        *glsl = graphics_float_literal(read_f32_unaligned(data));
        return true;
    }
    case sizeof(float) * 2:
    case sizeof(float) * 3:
    case sizeof(float) * 4: {
        const auto components = it->size / sizeof(float);
        std::ostringstream stream;
        stream << "vec" << components << "(";
        for (size_t i = 0; i < components; ++i) {
            if (i != 0) {
                stream << ", ";
            }
            stream << graphics_float_literal(read_f32_unaligned(data + i * sizeof(float)));
        }
        stream << ")";
        *glsl = stream.str();
        return true;
    }
    default:
        return false;
    }
}

bool lower_graphics_fragment_expression(GraphicsFragmentLoweringContext& context, uint32_t expr_id,
                                        std::string* glsl);

bool is_graphics_sample_expression(const GraphicsFragmentLoweringContext& context, uint32_t expr_id) {
    if (expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    if (expression.kind == 23) {
        return true;
    }
    if ((expression.kind == 11 || expression.kind == 15 || expression.kind == 16) &&
        expression.a != UINT32_MAX) {
        return is_graphics_sample_expression(context, expression.a);
    }

    return false;
}

bool lower_graphics_texture_sample(GraphicsFragmentLoweringContext& context,
                                   const Feather::TypedIR::Expression& expression,
                                   std::string* glsl) {
    if (glsl == nullptr || expression.argument_count < 3 ||
        expression.first_argument == UINT32_MAX ||
        expression.first_argument > context.module.arguments.size() ||
        expression.argument_count > context.module.arguments.size() - expression.first_argument) {
        return false;
    }

    const auto texture_expr_id = context.module.arguments[expression.first_argument];
    const auto uv_expr_id = context.module.arguments[expression.first_argument + 2];
    std::string texture_name;
    if (!try_graphics_resource_name(context, texture_expr_id, &texture_name)) {
        return false;
    }

    const IrResource* texture_resource = nullptr;
    for (const auto& resource : context.ir.resources) {
        const auto resource_name = string_or_empty(context.ir, resource.name_string_id);
        if (resource_name == texture_name && resource.kind == kIrResourceKindTexture2D) {
            texture_resource = &resource;
            break;
        }
    }
    if (texture_resource == nullptr) {
        return false;
    }

    std::string uv;
    if (!lower_graphics_fragment_expression(context, uv_expr_id, &uv)) {
        return false;
    }

    context.sampled_texture_binding = texture_resource->binding;
    std::ostringstream stream;
    stream << "texture(u_texture_" << texture_resource->binding
           << ", " << uv << ")";
    *glsl = stream.str();
    return true;
}

bool lower_graphics_fragment_expression(GraphicsFragmentLoweringContext& context, uint32_t expr_id,
                                        std::string* glsl) {
    if (glsl == nullptr || expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    switch (expression.kind) {
    case 1: {
        const auto* literal = typed_ir_string(context.module, expression.name_id);
        if (literal == nullptr) {
            return false;
        }
        *glsl = normalize_graphics_literal(*literal);
        return true;
    }
    case 2:
    case 3: {
        const auto* name = typed_ir_string(context.module, expression.name_id);
        if (name == nullptr) {
            return false;
        }

        if (*name == context.fragment_parameter_name) {
            *glsl = "v_color";
            return true;
        }

        if (expression.kind == 2) {
            const auto local = context.locals.find(*name);
            if (local != context.locals.end()) {
                return lower_graphics_fragment_expression(context, local->second, glsl);
            }
        }

        return false;
    }
    case 6: {
        std::string operand;
        if (!lower_graphics_fragment_expression(context, expression.a, &operand)) {
            return false;
        }
        const auto op = expression.op == 0 ? "-" : expression.op == 1 ? "!" : "~";
        *glsl = std::string("(") + op + operand + ")";
        return true;
    }
    case 7: {
        std::string left;
        std::string right;
        if (!lower_graphics_fragment_expression(context, expression.a, &left) ||
            !lower_graphics_fragment_expression(context, expression.b, &right)) {
            return false;
        }
        const auto* op = expression.op == 0 ? "+"
                       : expression.op == 1 ? "-"
                       : expression.op == 2 ? "*"
                       : expression.op == 3 ? "/"
                       : expression.op == 4 ? "%"
                       : nullptr;
        if (op == nullptr) {
            return false;
        }
        *glsl = "(" + left + " " + op + " " + right + ")";
        return true;
    }
    case 11:
        return lower_graphics_fragment_expression(context, expression.a, glsl);
    case 12: {
        const auto type_name = graphics_glsl_type_name(context.module, expression.type_id);
        if (type_name.empty() || expression.argument_count == 0 ||
            expression.first_argument == UINT32_MAX ||
            expression.first_argument > context.module.arguments.size() ||
            expression.argument_count > context.module.arguments.size() - expression.first_argument) {
            return false;
        }

        std::ostringstream stream;
        stream << type_name << "(";
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            std::string argument;
            if (!lower_graphics_fragment_expression(context, context.module.arguments[expression.first_argument + i], &argument)) {
                return false;
            }
            if (i != 0) {
                stream << ", ";
            }
            stream << argument;
        }
        stream << ")";
        *glsl = stream.str();
        return true;
    }
    case 15: {
        const auto* swizzle = typed_ir_string(context.module, expression.name_id);
        if (swizzle == nullptr) {
            return false;
        }

        const auto components = graphics_swizzle_components(*swizzle);
        if (is_fragment_parameter_reference(context, expression.a) &&
            (components == "xy" || components == "rg")) {
            *glsl = "v_uv";
            return true;
        }

        std::string value;
        if (!lower_graphics_fragment_expression(context, expression.a, &value)) {
            return false;
        }
        *glsl = "(" + value + ")." + components;
        return true;
    }
    case 16: {
        const auto* member = typed_ir_string(context.module, expression.name_id);
        if (member == nullptr || member->empty()) {
            return false;
        }

        std::string value;
        if (!lower_graphics_fragment_expression(context, expression.a, &value)) {
            return false;
        }
        if (*member == "X" || *member == "R") {
            *glsl = "(" + value + ").r";
            return true;
        }
        if (*member == "Y" || *member == "G") {
            *glsl = "(" + value + ").g";
            return true;
        }
        if (*member == "Z" || *member == "B") {
            *glsl = "(" + value + ").b";
            return true;
        }
        if (*member == "W" || *member == "A") {
            *glsl = "(" + value + ").a";
            return true;
        }
        if (*member == "XY" || *member == "RG") {
            *glsl = "(" + value + ").rg";
            return true;
        }
        if (*member == "ZW" || *member == "BA") {
            *glsl = "(" + value + ").ba";
            return true;
        }
        if (*member == "XYZ" || *member == "RGB") {
            *glsl = "(" + value + ").rgb";
            return true;
        }
        if (*member == "RGBA") {
            *glsl = value;
            return true;
        }

        if (is_graphics_sample_expression(context, expression.a)) {
            if (*member == "R") {
                *glsl = "(" + value + ").r";
                return true;
            }
            if (*member == "G") {
                *glsl = "(" + value + ").g";
                return true;
            }
            if (*member == "B") {
                *glsl = "(" + value + ").b";
                return true;
            }
            if (*member == "A") {
                *glsl = "(" + value + ").a";
                return true;
            }
        }

        *glsl = "(" + value + ")." + graphics_swizzle_components(*member);
        return true;
    }
    case 19:
        return try_graphics_push_constant_literal(context, expression, glsl);
    case 23:
        return lower_graphics_texture_sample(context, expression, glsl);
    default:
        return false;
    }
}

bool find_graphics_fragment_return_expression(const GraphicsFragmentLoweringContext& context, uint32_t* expr_id) {
    if (expr_id == nullptr ||
        context.module.entry_function >= context.module.functions.size()) {
        return false;
    }

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.kind != 4 || function.body_statement_index >= context.module.statements.size()) {
        return false;
    }

    if (function.parameter_count > 0) {
        if (function.first_parameter == UINT32_MAX ||
            function.first_parameter > context.module.parameters.size() ||
            function.parameter_count > context.module.parameters.size() - function.first_parameter) {
            return false;
        }
    }

    const auto& body = context.module.statements[function.body_statement_index];
    if (body.kind != 1 || body.child_count == 0 ||
        body.first_child == UINT32_MAX ||
        body.first_child > context.module.children.size() ||
        body.child_count > context.module.children.size() - body.first_child) {
        return false;
    }

    for (uint32_t i = 0; i < body.child_count; ++i) {
        const auto statement_id = context.module.children[body.first_child + i];
        if (statement_id >= context.module.statements.size()) {
            return false;
        }

        const auto& statement = context.module.statements[statement_id];
        if (statement.kind != 11) {
            continue;
        }
        if (statement.a == UINT32_MAX || statement.a >= context.module.expressions.size()) {
            return false;
        }

        *expr_id = statement.a;
        return true;
    }

    return false;
}

bool collect_graphics_fragment_locals(GraphicsFragmentLoweringContext* context) {
    if (context == nullptr ||
        context->module.entry_function >= context->module.functions.size()) {
        return false;
    }

    const auto& function = context->module.functions[context->module.entry_function];
    if (function.body_statement_index >= context->module.statements.size()) {
        return false;
    }

    const auto& body = context->module.statements[function.body_statement_index];
    if (body.kind != 1) {
        return false;
    }
    if (body.child_count == 0) {
        return true;
    }
    if (body.first_child == UINT32_MAX ||
        body.first_child > context->module.children.size() ||
        body.child_count > context->module.children.size() - body.first_child) {
        return false;
    }

    for (uint32_t i = 0; i < body.child_count; ++i) {
        const auto statement_id = context->module.children[body.first_child + i];
        if (statement_id >= context->module.statements.size()) {
            return false;
        }

        const auto& statement = context->module.statements[statement_id];
        if (statement.kind == 11) {
            break;
        }
        if (statement.kind != 2) {
            continue;
        }
        if (statement.a == UINT32_MAX || statement.a >= context->module.expressions.size() ||
            statement.name_id >= context->module.strings.size()) {
            return false;
        }

        context->locals[context->module.strings[statement.name_id]] = statement.a;
    }

    return true;
}

bool try_build_graphics_fragment_glsl(const GraphicsPipelineState& pipeline, std::string* source,
                                      std::optional<uint32_t>* sampled_texture_binding) {
    if (source == nullptr || sampled_texture_binding == nullptr) {
        return false;
    }

    ParsedIr vertex_ir;
    ParsedIr fragment_ir;
    if (!parse_feather_ir(pipeline.vertex_ir, &vertex_ir) ||
        !parse_feather_ir(pipeline.fragment_ir, &fragment_ir) ||
        !fragment_ir.has_section7 ||
        fragment_ir.typed_module.entry_function >= fragment_ir.typed_module.functions.size()) {
        return false;
    }

    std::vector<GraphicsPushConstantLayoutEntry> push_constants;
    if (!build_graphics_push_constant_layout(vertex_ir, fragment_ir, &push_constants)) {
        return false;
    }

    GraphicsFragmentLoweringContext context{
        fragment_ir,
        fragment_ir.typed_module,
        pipeline,
        push_constants,
        "input",
        std::nullopt,
        {}};

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.parameter_count > 0 && function.first_parameter < context.module.parameters.size()) {
        const auto& parameter = context.module.parameters[function.first_parameter];
        if (parameter.name_id < context.module.strings.size()) {
            context.fragment_parameter_name = context.module.strings[parameter.name_id];
        }
    }

    if (!collect_graphics_fragment_locals(&context)) {
        return false;
    }

    uint32_t return_expr_id = UINT32_MAX;
    if (!find_graphics_fragment_return_expression(context, &return_expr_id)) {
        return false;
    }

    const auto return_type = graphics_glsl_type_name(context.module, context.module.expressions[return_expr_id].type_id);
    if (return_type != "vec4") {
        return false;
    }

    std::string return_expression;
    if (!lower_graphics_fragment_expression(context, return_expr_id, &return_expression)) {
        return false;
    }

    std::ostringstream glsl;
    glsl << "#version 450\n";
    glsl << "layout(location = 0) in vec4 v_color;\n";
    glsl << "layout(location = 1) in vec2 v_uv;\n";
    glsl << "layout(location = 0) out vec4 out_color;\n";
    if (context.sampled_texture_binding.has_value()) {
        glsl << "layout(set = 0, binding = " << *context.sampled_texture_binding
             << ") uniform sampler2D u_texture_" << *context.sampled_texture_binding << ";\n";
    }
    glsl << "void main() {\n";
    glsl << "    out_color = " << return_expression << ";\n";
    glsl << "}\n";

    *sampled_texture_binding = context.sampled_texture_binding;
    *source = glsl.str();
    return true;
}

std::optional<uint32_t> graphics_backend_binding(const GraphicsResourceLayout& layout, uint8_t kind,
                                                 uint32_t source_binding) {
    for (const auto& entry : layout.entries) {
        if (entry.kind == kind && entry.source_binding == source_binding) {
            return entry.backend_binding;
        }
    }

    return std::nullopt;
}

const IrResource* find_graphics_sampler_for_texture_sample(const ParsedIr& ir, const IrResource& texture_resource) {
    for (const auto& node : ir.expression_nodes) {
        if ((node.kind != kIrExpressionNodeKindTextureSample &&
             node.kind != kIrExpressionNodeKindTextureSampleLevel) ||
            node.resource_binding != texture_resource.binding) {
            continue;
        }

        const auto* sampler_name = get_string(ir, node.symbol_string_id);
        if (sampler_name == nullptr || sampler_name->empty()) {
            continue;
        }

        const auto* sampler_resource = find_resource_by_name(ir, *sampler_name);
        if (sampler_resource != nullptr && sampler_resource->kind == kIrResourceKindSampler) {
            return sampler_resource;
        }
    }

    return nullptr;
}

bool append_graphics_resource_layout_from_ir(const ParsedIr& ir, GraphicsResourceLayout* layout) {
    if (layout == nullptr) {
        return false;
    }

    std::set<uint32_t> used_bindings;
    for (const auto& entry : layout->entries) {
        used_bindings.insert(entry.backend_binding);
    }

    for (const auto& resource : ir.resources) {
        if (resource.kind == kIrResourceKindPushConstant || resource.kind == kIrResourceKindSampler) {
            continue;
        }
        if (resource.kind != kIrResourceKindBuffer && resource.kind != kIrResourceKindTexture2D) {
            return false;
        }

        const auto duplicate = std::any_of(layout->entries.begin(), layout->entries.end(), [&](const auto& entry) {
            return entry.kind == resource.kind && entry.source_binding == resource.binding;
        });
        if (duplicate) {
            continue;
        }

        uint32_t backend_binding = 0;
        while (used_bindings.count(backend_binding) != 0) {
            ++backend_binding;
        }
        if (backend_binding >= GPU::Backend::MAX_BUFFER_BINDINGS ||
            backend_binding >= GPU::Backend::MAX_TEXTURE_BINDINGS) {
            return false;
        }

        uint32_t sampler_binding = UINT32_MAX;
        if (resource.kind == kIrResourceKindTexture2D) {
            if (const auto* sampler = find_graphics_sampler_for_texture_sample(ir, resource); sampler != nullptr) {
                sampler_binding = sampler->binding;
            }
        }

        layout->entries.push_back(GraphicsResourceBindingEntry{
            resource.binding,
            backend_binding,
            resource.kind,
            resource.access,
            sampler_binding});
        used_bindings.insert(backend_binding);
    }

    return true;
}

bool build_graphics_resource_layout(const ParsedIr& vertex_ir, const ParsedIr& fragment_ir,
                                    GraphicsResourceLayout* layout) {
    if (layout == nullptr) {
        return false;
    }

    layout->entries.clear();
    return append_graphics_resource_layout_from_ir(vertex_ir, layout) &&
           append_graphics_resource_layout_from_ir(fragment_ir, layout);
}

bool graphics_find_type_by_name(const Feather::TypedIR::Module& module, const std::string& type_name,
                                uint32_t* type_id) {
    if (type_id == nullptr) {
        return false;
    }

    const auto normalized_type = type_name.rfind("global::", 0) == 0 ? type_name.substr(8) : type_name;
    for (uint32_t i = 0; i < module.types.size(); ++i) {
        const auto& type = module.types[i];
        if (type.kind == 4 && type.a < module.structs.size()) {
            const auto& structure = module.structs[type.a];
            const auto* simple = typed_ir_string(module, structure.name_id);
            const auto* qualified = typed_ir_string(module, structure.fully_qualified_name_id);
            const auto normalized_qualified = qualified != nullptr && qualified->rfind("global::", 0) == 0
                                                  ? qualified->substr(8)
                                                  : (qualified == nullptr ? std::string{} : *qualified);
            if ((simple != nullptr && *simple == type_name) ||
                (qualified != nullptr && *qualified == type_name) ||
                (!normalized_qualified.empty() && normalized_qualified == normalized_type)) {
                *type_id = i;
                return true;
            }
            continue;
        }

        const auto glsl_type = graphics_glsl_type_name(module, i);
        if ((type_name == "float" && glsl_type == "float") ||
            (type_name == "System.Single" && glsl_type == "float") ||
            (type_name == "int" && glsl_type == "int") ||
            (type_name == "System.Int32" && glsl_type == "int") ||
            (type_name == "uint" && glsl_type == "uint") ||
            (type_name == "System.UInt32" && glsl_type == "uint") ||
            (type_name == "Feather.Math.float2" && glsl_type == "vec2") ||
            (type_name == "global::Feather.Math.float2" && glsl_type == "vec2") ||
            (type_name == "float2" && glsl_type == "vec2") ||
            (type_name == "Feather.Math.float3" && glsl_type == "vec3") ||
            (type_name == "global::Feather.Math.float3" && glsl_type == "vec3") ||
            (type_name == "float3" && glsl_type == "vec3") ||
            (type_name == "Feather.Math.float4" && glsl_type == "vec4") ||
            (type_name == "global::Feather.Math.float4" && glsl_type == "vec4") ||
            (type_name == "float4" && glsl_type == "vec4") ||
            (type_name == "Feather.Math.float2x2" && glsl_type == "mat2") ||
            (type_name == "global::Feather.Math.float2x2" && glsl_type == "mat2") ||
            (type_name == "float2x2" && glsl_type == "mat2") ||
            (type_name == "Feather.Math.float3x3" && glsl_type == "mat3") ||
            (type_name == "global::Feather.Math.float3x3" && glsl_type == "mat3") ||
            (type_name == "float3x3" && glsl_type == "mat3") ||
            (type_name == "Feather.Math.float4x4" && glsl_type == "mat4") ||
            (type_name == "global::Feather.Math.float4x4" && glsl_type == "mat4") ||
            (type_name == "float4x4" && glsl_type == "mat4")) {
            *type_id = i;
            return true;
        }
    }

    return false;
}

bool graphics_find_resource_element_type_id(const GraphicsLoweringContext& context, const IrResource& resource,
                                            uint32_t* type_id) {
    if (type_id == nullptr) {
        return false;
    }

    const auto resource_name = string_or_empty(context.ir, resource.name_string_id);
    for (const auto& expression : context.module.expressions) {
        if (expression.kind != 5 || expression.name_id >= context.module.strings.size()) {
            continue;
        }
        if (context.module.strings[expression.name_id] == resource_name &&
            expression.type_id < context.module.types.size()) {
            *type_id = expression.type_id;
            return true;
        }
    }

    const auto type_name = string_or_empty(context.ir, resource.element_type_string_id);
    return !type_name.empty() && graphics_find_type_by_name(context.module, type_name, type_id);
}

std::string graphics_struct_field_glsl_name(const Feather::TypedIR::Module& module,
                                            const Feather::TypedIR::StructField& field) {
    const auto* name = typed_ir_string(module, field.name_id);
    return name == nullptr ? std::string{} : sanitize_graphics_glsl_identifier(*name);
}

bool append_graphics_struct_declaration(const Feather::TypedIR::Module& module, uint32_t type_id,
                                        std::set<uint32_t>* declared, std::ostringstream* glsl) {
    if (declared == nullptr || glsl == nullptr || type_id >= module.types.size()) {
        return false;
    }

    const auto& type = module.types[type_id];
    if (type.kind == 2 || type.kind == 3 || type.kind == 1 || type.kind == 7) {
        return true;
    }
    if (type.kind != 4 || type.a >= module.structs.size()) {
        return false;
    }
    if (declared->count(type.a) != 0) {
        return true;
    }

    const auto& structure = module.structs[type.a];
    if (structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) {
        return false;
    }

    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        if (!append_graphics_struct_declaration(module, field.type_id, declared, glsl)) {
            return false;
        }
    }

    const auto struct_name = graphics_glsl_type_name(module, type_id);
    if (struct_name.empty()) {
        return false;
    }

    *glsl << "struct " << struct_name << " {\n";
    for (uint32_t i = 0; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        const auto field_type = graphics_glsl_type_name(module, field.type_id);
        const auto field_name = graphics_struct_field_glsl_name(module, field);
        if (field_type.empty() || field_name.empty()) {
            return false;
        }

        *glsl << "    " << field_type << " " << field_name << ";\n";
    }
    *glsl << "};\n";

    declared->insert(type.a);
    return true;
}

bool append_graphics_resource_declarations(GraphicsLoweringContext& context, std::set<uint32_t>* declared_structs,
                                           std::ostringstream* glsl) {
    if (declared_structs == nullptr || glsl == nullptr) {
        return false;
    }

    for (const auto& resource : context.ir.resources) {
        if (resource.kind == kIrResourceKindSampler || resource.kind == kIrResourceKindPushConstant) {
            continue;
        }

        const auto backend_binding = graphics_backend_binding(context.resources, resource.kind, resource.binding);
        if (!backend_binding.has_value()) {
            return false;
        }

        if (resource.kind == kIrResourceKindBuffer) {
            uint32_t element_type_id = UINT32_MAX;
            if (!graphics_find_resource_element_type_id(context, resource, &element_type_id)) {
                return false;
            }
            if (!append_graphics_struct_declaration(context.module, element_type_id, declared_structs, glsl)) {
                return false;
            }

            const auto element_type = graphics_glsl_type_name(context.module, element_type_id);
            if (element_type.empty()) {
                return false;
            }

            *glsl << "layout(std430, set = 0, binding = " << *backend_binding << ") readonly buffer fe_buffer_block_"
                  << *backend_binding << " {\n";
            *glsl << "    " << element_type << " fe_buffer_" << *backend_binding << "[];\n";
            *glsl << "};\n";
            continue;
        }

        if (resource.kind == kIrResourceKindTexture2D) {
            *glsl << "layout(set = 0, binding = " << *backend_binding << ") uniform sampler2D u_texture_"
                  << *backend_binding << ";\n";
            continue;
        }

        return false;
    }

    return true;
}

std::string graphics_varying_variable_name(const GraphicsVaryingField& field) {
    return "v_" + sanitize_graphics_glsl_identifier(field.name);
}

bool append_graphics_varying_declarations(const GraphicsVaryingLayout& varyings, GraphicsStage stage,
                                          std::ostringstream* glsl) {
    if (glsl == nullptr) {
        return false;
    }

    const auto qualifier = stage == GraphicsStage::Vertex ? "out" : "in";
    if (varyings.is_float4) {
        *glsl << "layout(location = 0) " << qualifier << " vec4 v_fe_color;\n";
        *glsl << "layout(location = 1) " << qualifier << " vec2 v_fe_uv;\n";
        return true;
    }

    for (const auto& field : varyings.fields) {
        if (field.position || field.glsl_type.empty()) {
            continue;
        }

        *glsl << "layout(location = " << field.location << ") " << qualifier << " " << field.glsl_type
              << " " << graphics_varying_variable_name(field) << ";\n";
    }

    return true;
}

std::string graphics_push_constant_variable_name(const GraphicsPushConstantLayoutEntry& entry) {
    return "pc_" + std::to_string(entry.binding) + "_" + sanitize_graphics_glsl_identifier(entry.name);
}

bool append_graphics_push_constant_declarations(const GraphicsLoweringContext& context,
                                                std::set<uint32_t>* declared_structs,
                                                std::ostringstream* glsl) {
    if (declared_structs == nullptr || glsl == nullptr || context.push_constants.empty()) {
        return true;
    }

    std::ostringstream body;
    bool has_entries = false;
    for (const auto& entry : context.push_constants) {
        const auto* resource = find_graphics_resource_by_binding_and_name(context.ir, entry.binding, entry.name);
        if (resource == nullptr) {
            continue;
        }

        uint32_t type_id = UINT32_MAX;
        const auto type_name = string_or_empty(context.ir, resource->element_type_string_id);
        if (type_name.empty() || !graphics_find_type_by_name(context.module, type_name, &type_id) ||
            !append_graphics_struct_declaration(context.module, type_id, declared_structs, glsl)) {
            return false;
        }

        const auto glsl_type = graphics_glsl_type_name(context.module, type_id);
        if (glsl_type.empty()) {
            return false;
        }

        body << "    layout(offset = " << entry.offset << ") " << glsl_type << " "
             << graphics_push_constant_variable_name(entry) << ";\n";
        has_entries = true;
    }
    if (!has_entries) {
        return true;
    }

    *glsl << "layout(push_constant) uniform fe_push_constants {\n";
    *glsl << body.str();
    *glsl << "} fe_pc;\n";
    return true;
}

bool is_graphics_parameter_reference(const GraphicsLoweringContext& context, uint32_t expr_id) {
    if (expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    return expression.kind == 3 &&
           expression.name_id < context.module.strings.size() &&
           context.module.strings[expression.name_id] == context.parameter_name;
}

bool try_graphics_stage_resource_name(const GraphicsLoweringContext& context, uint32_t expr_id,
                                      std::string* name) {
    if (name == nullptr || expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    if ((expression.kind == 2 || expression.kind == 3) && expression.name_id < context.module.strings.size()) {
        *name = context.module.strings[expression.name_id];
        return true;
    }

    if (expression.kind == 4 && expression.name_id < context.module.strings.size()) {
        *name = context.module.strings[expression.name_id];
        return true;
    }

    return false;
}

const IrResource* find_graphics_resource_by_name_and_kind(const ParsedIr& ir, const std::string& name,
                                                          uint8_t kind) {
    for (const auto& resource : ir.resources) {
        if (resource.kind == kind && string_or_empty(ir, resource.name_string_id) == name) {
            return &resource;
        }
    }

    return nullptr;
}

bool lower_graphics_stage_expression(GraphicsLoweringContext& context, uint32_t expr_id,
                                     std::string* glsl);

bool is_graphics_stage_sample_expression(const GraphicsLoweringContext& context, uint32_t expr_id) {
    if (expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    if (expression.kind == 23) {
        return true;
    }
    if ((expression.kind == 11 || expression.kind == 15 || expression.kind == 16) &&
        expression.a != UINT32_MAX) {
        return is_graphics_stage_sample_expression(context, expression.a);
    }

    return false;
}

bool lower_graphics_texture_sample_expression(GraphicsLoweringContext& context,
                                              const Feather::TypedIR::Expression& expression,
                                              std::string* glsl) {
    if (glsl == nullptr || expression.argument_count < 3 ||
        expression.first_argument == UINT32_MAX ||
        expression.first_argument > context.module.arguments.size() ||
        expression.argument_count > context.module.arguments.size() - expression.first_argument) {
        return false;
    }

    const auto texture_expr_id = context.module.arguments[expression.first_argument];
    const auto uv_expr_id = context.module.arguments[expression.first_argument + 2];
    std::string texture_name;
    if (!try_graphics_stage_resource_name(context, texture_expr_id, &texture_name)) {
        trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                          texture_expr_id, "texture sample resource name lowering failed");
        return false;
    }

    const auto* texture_resource = find_graphics_resource_by_name_and_kind(
        context.ir, texture_name, kIrResourceKindTexture2D);
    if (texture_resource == nullptr) {
        trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                          texture_expr_id, "texture sample resource lookup failed");
        return false;
    }

    const auto backend_binding = graphics_backend_binding(
        context.resources, kIrResourceKindTexture2D, texture_resource->binding);
    if (!backend_binding.has_value()) {
        trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                          texture_expr_id, "texture sample backend binding lookup failed");
        return false;
    }

    std::string uv;
    if (!lower_graphics_stage_expression(context, uv_expr_id, &uv)) {
        trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                          uv_expr_id, "texture sample uv lowering failed");
        return false;
    }

    context.sampled_texture_binding = texture_resource->binding;
    std::ostringstream stream;
    if (expression.op == 0) {
        stream << "texture(u_texture_" << *backend_binding << ", " << uv;
    } else if (expression.op == 1) {
        stream << "textureLod(u_texture_" << *backend_binding << ", " << uv;
        std::string lod;
        if (expression.argument_count < 4 ||
            !lower_graphics_stage_expression(context, context.module.arguments[expression.first_argument + 3], &lod)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              context.module.arguments[expression.first_argument + 3],
                                              "texture sample lod lowering failed");
            return false;
        }
        stream << ", " << lod;
    } else if (expression.op == 2) {
        stream << "textureGrad(u_texture_" << *backend_binding << ", " << uv;
        std::string ddx;
        std::string ddy;
        if (expression.argument_count < 5 ||
            !lower_graphics_stage_expression(context, context.module.arguments[expression.first_argument + 3], &ddx) ||
            !lower_graphics_stage_expression(context, context.module.arguments[expression.first_argument + 4], &ddy)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              texture_expr_id,
                                              "texture sample explicit-gradient lowering failed");
            return false;
        }
        stream << ", " << ddx << ", " << ddy;
    } else {
        trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                          texture_expr_id,
                                          "unsupported texture sample operation");
        return false;
    }
    stream << ")";
    *glsl = stream.str();
    return true;
}

bool graphics_member_access_from_fragment_parameter(const GraphicsLoweringContext& context,
                                                    const std::string& member,
                                                    std::string* glsl) {
    if (glsl == nullptr || context.stage != GraphicsStage::Fragment) {
        return false;
    }

    if (context.varyings.is_float4) {
        const auto components = graphics_swizzle_components(member);
        if (components == "xy") {
            *glsl = "v_fe_uv";
            return true;
        }
        if (components == "xyzw") {
            *glsl = "v_fe_color";
            return true;
        }
        *glsl = "(v_fe_color)." + components;
        return true;
    }

    if (member == context.varyings.position_field_name) {
        *glsl = "gl_FragCoord";
        return true;
    }

    for (const auto& field : context.varyings.fields) {
        if (field.name == member) {
            *glsl = graphics_varying_variable_name(field);
            return true;
        }
    }

    return false;
}

const char* graphics_binary_operator(uint32_t op) {
    switch (op) {
    case 0:
        return "+";
    case 1:
        return "-";
    case 2:
        return "*";
    case 3:
        return "/";
    case 4:
        return "%";
    case 5:
        return "&";
    case 6:
        return "|";
    case 7:
        return "^";
    case 8:
        return "<<";
    case 9:
        return ">>";
    default:
        return nullptr;
    }
}

const char* graphics_compare_operator(uint32_t op) {
    switch (op) {
    case 0:
        return "==";
    case 1:
        return "!=";
    case 2:
        return "<";
    case 3:
        return "<=";
    case 4:
        return ">";
    case 5:
        return ">=";
    default:
        return nullptr;
    }
}

std::string graphics_intrinsic_name(std::string symbol) {
    if (symbol == "global::Feather.Math.ShaderMath.Normalize" ||
        symbol == "global::Feather.Math.Hlsl.Normalize") {
        return "normalize";
    }
    if (symbol == "global::Feather.Math.ShaderMath.Length" ||
        symbol == "global::Feather.Math.Hlsl.Length") {
        return "length";
    }
    if (symbol == "global::Feather.Math.ShaderMath.InverseSqrt") {
        return "inversesqrt";
    }
    if (symbol == "global::Feather.Math.ShaderMath.Fract" ||
        symbol == "global::Feather.Math.Hlsl.Fract") {
        return "fract";
    }
    if (symbol == "global::Feather.Math.ShaderMath.Ddx") {
        return "dFdx";
    }
    if (symbol == "global::Feather.Math.ShaderMath.Ddy") {
        return "dFdy";
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

    return easygpu_intrinsic_name(symbol);
}

bool lower_graphics_intrinsic_expression(GraphicsLoweringContext& context,
                                         const Feather::TypedIR::Expression& expression,
                                         std::string* glsl) {
    if (glsl == nullptr || expression.name_id >= context.module.strings.size() ||
        expression.first_argument == UINT32_MAX ||
        expression.first_argument > context.module.arguments.size() ||
        expression.argument_count > context.module.arguments.size() - expression.first_argument) {
        return false;
    }

    const auto& symbol = context.module.strings[expression.name_id];
    std::vector<std::string> arguments;
    arguments.reserve(expression.argument_count);
    for (uint32_t i = 0; i < expression.argument_count; ++i) {
        std::string argument;
        if (!lower_graphics_stage_expression(context, context.module.arguments[expression.first_argument + i], &argument)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              context.module.arguments[expression.first_argument + i],
                                              "intrinsic argument lowering failed");
            return false;
        }
        arguments.push_back(std::move(argument));
    }

    if (symbol == "global::Feather.Math.ShaderMath.Mul" ||
        symbol == "global::Feather.Math.Hlsl.Mul") {
        if (arguments.size() != 2) {
            return false;
        }
        *glsl = "(" + arguments[0] + " * " + arguments[1] + ")";
        return true;
    }

    if (symbol == "global::Feather.Math.ShaderMath.Saturate") {
        if (arguments.size() != 1) {
            return false;
        }
        *glsl = "clamp(" + arguments[0] + ", 0.0, 1.0)";
        return true;
    }

    const auto intrinsic = graphics_intrinsic_name(symbol);
    if (intrinsic.empty()) {
        return false;
    }

    std::ostringstream stream;
    stream << intrinsic << "(";
    for (size_t i = 0; i < arguments.size(); ++i) {
        if (i != 0) {
            stream << ", ";
        }
        stream << arguments[i];
    }
    stream << ")";
    *glsl = stream.str();
    return true;
}

bool lower_graphics_stage_expression(GraphicsLoweringContext& context, uint32_t expr_id,
                                     std::string* glsl) {
    if (glsl == nullptr || expr_id >= context.module.expressions.size()) {
        return false;
    }

    const auto& expression = context.module.expressions[expr_id];
    switch (expression.kind) {
    case 1: {
        const auto* literal = typed_ir_string(context.module, expression.name_id);
        if (literal == nullptr) {
            return false;
        }
        *glsl = normalize_graphics_literal(*literal);
        return true;
    }
    case 2:
    case 3: {
        const auto* name = typed_ir_string(context.module, expression.name_id);
        if (name == nullptr) {
            return false;
        }

        if (context.stage == GraphicsStage::Fragment && *name == context.parameter_name) {
            if (context.varyings.is_float4) {
                *glsl = "v_fe_color";
                return true;
            }
            trace_graphics_expression_failure(context, "fragment", expr_id,
                                              "struct varying parameter needs a field/member access");
            return false;
        }

        if (expression.kind == 2) {
            const auto local = context.locals.find(*name);
            if (local != context.locals.end()) {
                return lower_graphics_stage_expression(context, local->second, glsl);
            }
        }

        *glsl = sanitize_graphics_glsl_identifier(*name);
        return true;
    }
    case 4: {
        const auto* member = typed_ir_string(context.module, expression.name_id);
        if (member == nullptr || member->empty()) {
            return false;
        }
        if (is_graphics_parameter_reference(context, expression.a) &&
            graphics_member_access_from_fragment_parameter(context, *member, glsl)) {
            return true;
        }

        std::string instance;
        if (!lower_graphics_stage_expression(context, expression.a, &instance)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "field instance lowering failed");
            return false;
        }
        *glsl = "(" + instance + ")." + sanitize_graphics_glsl_identifier(*member);
        return true;
    }
    case 5: {
        const auto* resource_name = typed_ir_string(context.module, expression.name_id);
        if (resource_name == nullptr || resource_name->empty()) {
            return false;
        }
        const auto* resource = find_graphics_resource_by_name_and_kind(
            context.ir, *resource_name, kIrResourceKindBuffer);
        if (resource == nullptr) {
            return false;
        }
        const auto backend_binding = graphics_backend_binding(
            context.resources, kIrResourceKindBuffer, resource->binding);
        if (!backend_binding.has_value()) {
            return false;
        }

        std::string index;
        if (!lower_graphics_stage_expression(context, expression.a, &index)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "resource index lowering failed");
            return false;
        }

        context.used_buffer_bindings.insert(resource->binding);
        *glsl = "fe_buffer_" + std::to_string(*backend_binding) + "[" + index + "]";
        return true;
    }
    case 6: {
        std::string operand;
        if (!lower_graphics_stage_expression(context, expression.a, &operand)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "unary operand lowering failed");
            return false;
        }
        const auto op = expression.op == 0 ? "-" : expression.op == 1 ? "!" : "~";
        *glsl = std::string("(") + op + operand + ")";
        return true;
    }
    case 7: {
        std::string left;
        std::string right;
        const auto* op = graphics_binary_operator(expression.op);
        if (op == nullptr) {
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.a, &left)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "binary left operand lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &right)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "binary right operand lowering failed");
            return false;
        }
        *glsl = "(" + left + " " + op + " " + right + ")";
        return true;
    }
    case 8: {
        std::string left;
        std::string right;
        const auto* op = graphics_compare_operator(expression.op);
        if (op == nullptr) {
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.a, &left)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "comparison left operand lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &right)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "comparison right operand lowering failed");
            return false;
        }
        *glsl = "(" + left + " " + op + " " + right + ")";
        return true;
    }
    case 9: {
        std::string left;
        std::string right;
        if (!lower_graphics_stage_expression(context, expression.a, &left)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "logical left operand lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &right)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "logical right operand lowering failed");
            return false;
        }
        *glsl = "(" + left + (expression.op == 0 ? " && " : " || ") + right + ")";
        return true;
    }
    case 10: {
        std::string condition;
        std::string when_true;
        std::string when_false;
        if (!lower_graphics_stage_expression(context, expression.a, &condition)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "conditional condition lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &when_true)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "conditional true branch lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.c, &when_false)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.c, "conditional false branch lowering failed");
            return false;
        }
        *glsl = "(" + condition + " ? " + when_true + " : " + when_false + ")";
        return true;
    }
    case 11: {
        std::string operand;
        if (!lower_graphics_stage_expression(context, expression.a, &operand)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "conversion operand lowering failed");
            return false;
        }
        const auto target_type = graphics_glsl_type_name(context.module, expression.type_id);
        const auto source_type = expression.a < context.module.expressions.size()
                                     ? graphics_glsl_type_name(context.module, context.module.expressions[expression.a].type_id)
                                     : std::string{};
        if (target_type.empty()) {
            return false;
        }
        *glsl = target_type == source_type ? operand : (target_type + "(" + operand + ")");
        return true;
    }
    case 12: {
        const auto type_name = graphics_glsl_type_name(context.module, expression.type_id);
        if (type_name.empty() ||
            expression.first_argument == UINT32_MAX ||
            expression.first_argument > context.module.arguments.size() ||
            expression.argument_count > context.module.arguments.size() - expression.first_argument) {
            return false;
        }

        std::ostringstream stream;
        stream << type_name << "(";
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            std::string argument;
            if (!lower_graphics_stage_expression(context, context.module.arguments[expression.first_argument + i], &argument)) {
                trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                                  context.module.arguments[expression.first_argument + i],
                                                  "constructor argument lowering failed");
                return false;
            }
            if (i != 0) {
                stream << ", ";
            }
            stream << argument;
        }
        stream << ")";
        *glsl = stream.str();
        return true;
    }
    case 13:
        return lower_graphics_intrinsic_expression(context, expression, glsl);
    case 14: {
        if (expression.name_id >= context.module.strings.size() ||
            (expression.argument_count == 0 && expression.first_argument != UINT32_MAX) ||
            (expression.argument_count > 0 &&
             (expression.first_argument == UINT32_MAX ||
              expression.first_argument > context.module.arguments.size() ||
              expression.argument_count > context.module.arguments.size() - expression.first_argument))) {
            return false;
        }

        std::ostringstream stream;
        stream << sanitize_graphics_glsl_identifier(context.module.strings[expression.name_id]) << "(";
        for (uint32_t i = 0; i < expression.argument_count; ++i) {
            std::string argument;
            const auto argument_expr_id = context.module.arguments[expression.first_argument + i];
            if (!lower_graphics_stage_expression(context, argument_expr_id, &argument)) {
                trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                                  argument_expr_id,
                                                  "callable argument lowering failed");
                return false;
            }
            if (i != 0) {
                stream << ", ";
            }
            stream << argument;
        }
        stream << ")";
        *glsl = stream.str();
        return true;
    }
    case 15: {
        const auto* swizzle = typed_ir_string(context.module, expression.name_id);
        if (swizzle == nullptr) {
            return false;
        }
        const auto components = graphics_swizzle_components(*swizzle);
        if (is_graphics_parameter_reference(context, expression.a) &&
            graphics_member_access_from_fragment_parameter(context, components, glsl)) {
            return true;
        }

        std::string value;
        if (!lower_graphics_stage_expression(context, expression.a, &value)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "swizzle value lowering failed");
            return false;
        }
        *glsl = "(" + value + ")." + components;
        return true;
    }
    case 16: {
        const auto* member = typed_ir_string(context.module, expression.name_id);
        if (member == nullptr || member->empty()) {
            return false;
        }
        if (is_graphics_parameter_reference(context, expression.a) &&
            graphics_member_access_from_fragment_parameter(context, *member, glsl)) {
            return true;
        }

        std::string value;
        if (!lower_graphics_stage_expression(context, expression.a, &value)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "member value lowering failed");
            return false;
        }
        if (*member == "X" || *member == "R") {
            *glsl = "(" + value + ").r";
            return true;
        }
        if (*member == "Y" || *member == "G") {
            *glsl = "(" + value + ").g";
            return true;
        }
        if (*member == "Z" || *member == "B") {
            *glsl = "(" + value + ").b";
            return true;
        }
        if (*member == "W" || *member == "A") {
            *glsl = "(" + value + ").a";
            return true;
        }
        if (*member == "XY" || *member == "RG") {
            *glsl = "(" + value + ").rg";
            return true;
        }
        if (*member == "ZW" || *member == "BA") {
            *glsl = "(" + value + ").ba";
            return true;
        }
        if (*member == "XYZ" || *member == "RGB") {
            *glsl = "(" + value + ").rgb";
            return true;
        }
        if (*member == "RGBA") {
            *glsl = value;
            return true;
        }
        if (is_graphics_stage_sample_expression(context, expression.a)) {
            *glsl = "(" + value + ")." + graphics_swizzle_components(*member);
            return true;
        }

        *glsl = "(" + value + ")." + sanitize_graphics_glsl_identifier(*member);
        return true;
    }
    case 17: {
        std::string value;
        std::string index;
        if (!lower_graphics_stage_expression(context, expression.a, &value)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "index value lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &index)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "index expression lowering failed");
            return false;
        }
        *glsl = "(" + value + ")[" + index + "]";
        return true;
    }
    case 18:
        switch (expression.op) {
        case 16:
            *glsl = "int(gl_VertexIndex)";
            return true;
        case 17:
            *glsl = "int(gl_InstanceIndex)";
            return true;
        case 18:
            *glsl = "gl_FragCoord";
            return true;
        case 19:
            *glsl = "gl_FragCoord.y";
            return true;
        case 20:
            *glsl = "gl_FragCoord.z";
            return true;
        case 21:
            *glsl = "gl_FragCoord.w";
            return true;
        default:
            return false;
        }
    case 19: {
        if (expression.name_id >= context.module.strings.size()) {
            return false;
        }
        const auto& name = context.module.strings[expression.name_id];
        const auto it = std::find_if(context.push_constants.begin(), context.push_constants.end(), [&](const auto& entry) {
            return same_graphics_resource(entry, expression.op, name);
        });
        if (it == context.push_constants.end()) {
            return false;
        }

        *glsl = "fe_pc." + graphics_push_constant_variable_name(*it);
        return true;
    }
    case 20: {
        std::string matrix;
        std::string column;
        if (!lower_graphics_stage_expression(context, expression.a, &matrix)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.a, "matrix value lowering failed");
            return false;
        }
        if (!lower_graphics_stage_expression(context, expression.b, &column)) {
            trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                              expression.b, "matrix column lowering failed");
            return false;
        }
        *glsl = "(" + matrix + ")[" + column + "]";
        return true;
    }
    case 23:
        return lower_graphics_texture_sample_expression(context, expression, glsl);
    default:
        return false;
    }
}

bool collect_graphics_stage_locals(GraphicsLoweringContext* context) {
    if (context == nullptr || context->module.entry_function >= context->module.functions.size()) {
        return false;
    }

    const auto& function = context->module.functions[context->module.entry_function];
    if (function.body_statement_index >= context->module.statements.size()) {
        return false;
    }

    const auto& body = context->module.statements[function.body_statement_index];
    if (body.kind != 1) {
        return false;
    }
    if (body.child_count == 0) {
        return true;
    }
    if (body.first_child == UINT32_MAX ||
        body.first_child > context->module.children.size() ||
        body.child_count > context->module.children.size() - body.first_child) {
        return false;
    }

    for (uint32_t i = 0; i < body.child_count; ++i) {
        const auto statement_id = context->module.children[body.first_child + i];
        if (statement_id >= context->module.statements.size()) {
            return false;
        }

        const auto& statement = context->module.statements[statement_id];
        if (statement.kind == 11) {
            break;
        }
        if (statement.kind != 2) {
            continue;
        }
        if (statement.a == UINT32_MAX || statement.a >= context->module.expressions.size() ||
            statement.name_id >= context->module.strings.size()) {
            return false;
        }

        context->locals[context->module.strings[statement.name_id]] = statement.a;
    }

    return true;
}

bool find_graphics_stage_return_expression(const GraphicsLoweringContext& context, uint32_t expected_function_kind,
                                           uint32_t* expr_id) {
    if (expr_id == nullptr || context.module.entry_function >= context.module.functions.size()) {
        return false;
    }

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.kind != expected_function_kind || function.body_statement_index >= context.module.statements.size()) {
        return false;
    }

    const auto& body = context.module.statements[function.body_statement_index];
    if (body.kind != 1 || body.child_count == 0 ||
        body.first_child == UINT32_MAX ||
        body.first_child > context.module.children.size() ||
        body.child_count > context.module.children.size() - body.first_child) {
        return false;
    }

    for (uint32_t i = 0; i < body.child_count; ++i) {
        const auto statement_id = context.module.children[body.first_child + i];
        if (statement_id >= context.module.statements.size()) {
            return false;
        }

        const auto& statement = context.module.statements[statement_id];
        if (statement.kind != 11) {
            continue;
        }
        if (statement.a == UINT32_MAX || statement.a >= context.module.expressions.size()) {
            return false;
        }

        *expr_id = statement.a;
        return true;
    }

    return false;
}

size_t graphics_push_constant_total_size(const std::vector<GraphicsPushConstantLayoutEntry>& push_constants) {
    size_t total = 0;
    for (const auto& entry : push_constants) {
        total = std::max(total, entry.offset + entry.size);
    }

    return total;
}

bool append_graphics_shader_prelude(GraphicsLoweringContext& context,
                                    std::ostringstream* glsl,
                                    uint32_t extra_struct_type_id = UINT32_MAX) {
    if (glsl == nullptr) {
        return false;
    }

    std::set<uint32_t> declared_structs;
    *glsl << "#version 450\n";
    for (uint32_t type_id = 0; type_id < context.module.types.size(); ++type_id) {
        if (context.module.types[type_id].kind == 4 &&
            !append_graphics_struct_declaration(context.module, type_id, &declared_structs, glsl)) {
            return false;
        }
    }
    if (context.stage == GraphicsStage::Vertex &&
        !append_graphics_struct_declaration(context.module, context.varyings.type_id, &declared_structs, glsl)) {
        return false;
    }
    if (extra_struct_type_id != UINT32_MAX &&
        !append_graphics_struct_declaration(context.module, extra_struct_type_id, &declared_structs, glsl)) {
        return false;
    }
    if (!append_graphics_resource_declarations(context, &declared_structs, glsl) ||
        !append_graphics_push_constant_declarations(context, &declared_structs, glsl) ||
        !append_graphics_varying_declarations(context.varyings, context.stage, glsl)) {
        return false;
    }

    return true;
}

std::string graphics_indent(int depth) {
    return std::string(static_cast<size_t>(std::max(depth, 0)) * 4, ' ');
}

const char* graphics_compound_operator(uint32_t op) {
    switch (op) {
    case 0:
        return "+=";
    case 1:
        return "-=";
    case 2:
        return "*=";
    case 3:
        return "/=";
    case 4:
        return "%=";
    case 5:
        return "&=";
    case 6:
        return "|=";
    case 7:
        return "^=";
    case 8:
        return "<<=";
    case 9:
        return ">>=";
    default:
        return nullptr;
    }
}

bool lower_graphics_stage_lvalue(GraphicsLoweringContext& context, uint32_t lvalue_id,
                                 std::string* glsl) {
    if (glsl == nullptr || lvalue_id >= context.module.lvalues.size()) {
        return false;
    }

    const auto& lvalue = context.module.lvalues[lvalue_id];
    switch (lvalue.kind) {
    case 1:
    case 2: {
        const auto* name = typed_ir_string(context.module, lvalue.name_id);
        if (name == nullptr || name->empty()) {
            return false;
        }
        *glsl = sanitize_graphics_glsl_identifier(*name);
        return true;
    }
    case 3:
    case 6: {
        const auto* member = typed_ir_string(context.module, lvalue.name_id);
        if (member == nullptr || member->empty()) {
            return false;
        }
        if (lvalue.a == UINT32_MAX) {
            *glsl = sanitize_graphics_glsl_identifier(*member);
            return true;
        }

        std::string instance;
        if (!lower_graphics_stage_lvalue(context, lvalue.a, &instance)) {
            return false;
        }
        *glsl = "(" + instance + ")." + sanitize_graphics_glsl_identifier(*member);
        return true;
    }
    case 4: {
        const auto* resource_name = typed_ir_string(context.module, lvalue.name_id);
        if (resource_name == nullptr || resource_name->empty()) {
            return false;
        }
        const auto* resource = find_graphics_resource_by_name_and_kind(
            context.ir, *resource_name, kIrResourceKindBuffer);
        if (resource == nullptr) {
            return false;
        }
        const auto backend_binding = graphics_backend_binding(
            context.resources, kIrResourceKindBuffer, resource->binding);
        if (!backend_binding.has_value()) {
            return false;
        }

        std::string index;
        if (!lower_graphics_stage_expression(context, lvalue.a, &index)) {
            return false;
        }

        context.used_buffer_bindings.insert(resource->binding);
        *glsl = "fe_buffer_" + std::to_string(*backend_binding) + "[" + index + "]";
        return true;
    }
    case 5: {
        const auto* swizzle = typed_ir_string(context.module, lvalue.name_id);
        if (swizzle == nullptr || swizzle->empty()) {
            return false;
        }

        std::string value;
        if (!lower_graphics_stage_expression(context, lvalue.a, &value)) {
            return false;
        }
        *glsl = "(" + value + ")." + graphics_swizzle_components(*swizzle);
        return true;
    }
    case 7: {
        std::string value;
        std::string index;
        if (!lower_graphics_stage_lvalue(context, lvalue.a, &value) ||
            !lower_graphics_stage_expression(context, lvalue.b, &index)) {
            return false;
        }
        *glsl = "(" + value + ")[" + index + "]";
        return true;
    }
    case 8: {
        std::string matrix;
        std::string column;
        if (!lower_graphics_stage_expression(context, lvalue.a, &matrix) ||
            !lower_graphics_stage_expression(context, lvalue.b, &column)) {
            return false;
        }
        *glsl = "(" + matrix + ")[" + column + "]";
        return true;
    }
    default:
        return false;
    }
}

using GraphicsReturnEmitter = std::function<bool(GraphicsLoweringContext&, uint32_t, int, std::ostringstream*)>;

bool append_graphics_stage_statement(GraphicsLoweringContext& context,
                                     uint32_t statement_id,
                                     int indent,
                                     const GraphicsReturnEmitter& return_emitter,
                                     std::ostringstream* glsl);

bool append_graphics_stage_statement_clause(GraphicsLoweringContext& context,
                                            uint32_t statement_id,
                                            std::string* glsl) {
    if (glsl == nullptr || statement_id >= context.module.statements.size()) {
        return false;
    }

    const auto& statement = context.module.statements[statement_id];
    switch (statement.kind) {
    case 1: {
        if (statement.child_count == 0) {
            *glsl = "";
            return statement.first_child == UINT32_MAX;
        }
        if (statement.child_count != 1 ||
            statement.first_child == UINT32_MAX ||
            statement.first_child >= context.module.children.size()) {
            return false;
        }
        return append_graphics_stage_statement_clause(
            context,
            context.module.children[statement.first_child],
            glsl);
    }
    case 2: {
        if (statement.name_id >= context.module.strings.size() || statement.op >= context.module.types.size()) {
            return false;
        }
        const auto type_name = graphics_glsl_type_name(context.module, statement.op);
        if (type_name.empty()) {
            return false;
        }
        std::ostringstream stream;
        stream << type_name << " " << sanitize_graphics_glsl_identifier(context.module.strings[statement.name_id]);
        if (statement.a != UINT32_MAX) {
            std::string initializer;
            if (!lower_graphics_stage_expression(context, statement.a, &initializer)) {
                return false;
            }
            stream << " = " << initializer;
        }
        *glsl = stream.str();
        return true;
    }
    case 3: {
        std::string target;
        std::string value;
        if (!lower_graphics_stage_lvalue(context, statement.a, &target) ||
            !lower_graphics_stage_expression(context, statement.b, &value)) {
            return false;
        }
        *glsl = target + " = " + value;
        return true;
    }
    case 4: {
        const auto* op = graphics_compound_operator(statement.op);
        std::string target;
        std::string value;
        if (op == nullptr ||
            !lower_graphics_stage_lvalue(context, statement.a, &target) ||
            !lower_graphics_stage_expression(context, statement.b, &value)) {
            return false;
        }
        *glsl = target + " " + op + " " + value;
        return true;
    }
    case 12: {
        return lower_graphics_stage_expression(context, statement.a, glsl);
    }
    case 14: {
        std::string target;
        if (!lower_graphics_stage_lvalue(context, statement.a, &target)) {
            return false;
        }
        const auto op = (statement.op & 1u) ? "++" : "--";
        *glsl = (statement.op & 2u) ? std::string(op) + target : target + op;
        return true;
    }
    default:
        return false;
    }
}

bool append_graphics_stage_statement_list(GraphicsLoweringContext& context,
                                          uint32_t statement_id,
                                          int indent,
                                          const GraphicsReturnEmitter& return_emitter,
                                          std::ostringstream* glsl) {
    if (glsl == nullptr || statement_id >= context.module.statements.size()) {
        return false;
    }

    const auto& statement = context.module.statements[statement_id];
    if (statement.kind != 1) {
        return append_graphics_stage_statement(context, statement_id, indent, return_emitter, glsl);
    }
    if (statement.child_count == 0) {
        return statement.first_child == UINT32_MAX;
    }
    if (statement.first_child == UINT32_MAX ||
        statement.first_child > context.module.children.size() ||
        statement.child_count > context.module.children.size() - statement.first_child) {
        return false;
    }

    for (uint32_t i = 0; i < statement.child_count; ++i) {
        if (!append_graphics_stage_statement(
                context,
                context.module.children[statement.first_child + i],
                indent,
                return_emitter,
                glsl)) {
            return false;
        }
    }

    return true;
}

bool append_graphics_stage_statement(GraphicsLoweringContext& context,
                                     uint32_t statement_id,
                                     int indent,
                                     const GraphicsReturnEmitter& return_emitter,
                                     std::ostringstream* glsl) {
    if (glsl == nullptr || statement_id >= context.module.statements.size()) {
        return false;
    }

    const auto& statement = context.module.statements[statement_id];
    const auto pad = graphics_indent(indent);
    switch (statement.kind) {
    case 1:
        *glsl << pad << "{\n";
        if (!append_graphics_stage_statement_list(context, statement_id, indent + 1, return_emitter, glsl)) {
            return false;
        }
        *glsl << pad << "}\n";
        return true;
    case 2: {
        if (statement.name_id >= context.module.strings.size() || statement.op >= context.module.types.size()) {
            return false;
        }
        const auto type_name = graphics_glsl_type_name(context.module, statement.op);
        if (type_name.empty()) {
            return false;
        }
        *glsl << pad << type_name << " " << sanitize_graphics_glsl_identifier(context.module.strings[statement.name_id]);
        if (statement.a != UINT32_MAX) {
            std::string initializer;
            if (!lower_graphics_stage_expression(context, statement.a, &initializer)) {
                trace_graphics_expression_failure(context, context.stage == GraphicsStage::Vertex ? "vertex" : "fragment",
                                                  statement.a, "local initializer lowering failed");
                return false;
            }
            *glsl << " = " << initializer;
        }
        *glsl << ";\n";
        return true;
    }
    case 3: {
        std::string target;
        std::string value;
        if (!lower_graphics_stage_lvalue(context, statement.a, &target) ||
            !lower_graphics_stage_expression(context, statement.b, &value)) {
            return false;
        }
        *glsl << pad << target << " = " << value << ";\n";
        return true;
    }
    case 4: {
        const auto* op = graphics_compound_operator(statement.op);
        std::string target;
        std::string value;
        if (op == nullptr ||
            !lower_graphics_stage_lvalue(context, statement.a, &target) ||
            !lower_graphics_stage_expression(context, statement.b, &value)) {
            return false;
        }
        *glsl << pad << target << " " << op << " " << value << ";\n";
        return true;
    }
    case 5: {
        std::string condition;
        if (!lower_graphics_stage_expression(context, statement.a, &condition) ||
            statement.b >= context.module.statements.size()) {
            return false;
        }
        *glsl << pad << "if (" << condition << ") {\n";
        if (!append_graphics_stage_statement_list(context, statement.b, indent + 1, return_emitter, glsl)) {
            return false;
        }
        *glsl << pad << "}";
        if (statement.c != UINT32_MAX) {
            if (statement.c >= context.module.statements.size()) {
                return false;
            }
            *glsl << " else {\n";
            if (!append_graphics_stage_statement_list(context, statement.c, indent + 1, return_emitter, glsl)) {
                return false;
            }
            *glsl << pad << "}";
        }
        *glsl << "\n";
        return true;
    }
    case 6: {
        if (statement.op >= context.module.statements.size()) {
            return false;
        }

        std::string initializer;
        std::string condition;
        std::string step;
        if (statement.a != UINT32_MAX &&
            !append_graphics_stage_statement_clause(context, statement.a, &initializer)) {
            return false;
        }
        if (statement.b != UINT32_MAX &&
            !lower_graphics_stage_expression(context, statement.b, &condition)) {
            return false;
        }
        if (statement.c != UINT32_MAX &&
            !append_graphics_stage_statement_clause(context, statement.c, &step)) {
            return false;
        }

        *glsl << pad << "for (" << initializer << "; " << condition << "; " << step << ") {\n";
        if (!append_graphics_stage_statement_list(context, statement.op, indent + 1, return_emitter, glsl)) {
            return false;
        }
        *glsl << pad << "}\n";
        return true;
    }
    case 7: {
        std::string condition;
        if (!lower_graphics_stage_expression(context, statement.a, &condition) ||
            statement.b >= context.module.statements.size()) {
            return false;
        }
        *glsl << pad << "while (" << condition << ") {\n";
        if (!append_graphics_stage_statement_list(context, statement.b, indent + 1, return_emitter, glsl)) {
            return false;
        }
        *glsl << pad << "}\n";
        return true;
    }
    case 8: {
        if (statement.a >= context.module.statements.size()) {
            return false;
        }
        std::string condition;
        if (!lower_graphics_stage_expression(context, statement.b, &condition)) {
            return false;
        }
        *glsl << pad << "do {\n";
        if (!append_graphics_stage_statement_list(context, statement.a, indent + 1, return_emitter, glsl)) {
            return false;
        }
        *glsl << pad << "} while (" << condition << ");\n";
        return true;
    }
    case 9:
        *glsl << pad << "break;\n";
        return true;
    case 10:
        *glsl << pad << "continue;\n";
        return true;
    case 11:
        if (return_emitter) {
            if (statement.a == UINT32_MAX) {
                return false;
            }
            return return_emitter(context, statement.a, indent, glsl);
        }
        if (statement.a == UINT32_MAX) {
            *glsl << pad << "return;\n";
            return true;
        } else {
            std::string value;
            if (!lower_graphics_stage_expression(context, statement.a, &value)) {
                return false;
            }
            *glsl << pad << "return " << value << ";\n";
            return true;
        }
    case 12: {
        std::string value;
        if (!lower_graphics_stage_expression(context, statement.a, &value)) {
            return false;
        }
        *glsl << pad << value << ";\n";
        return true;
    }
    case 14: {
        std::string target;
        if (!lower_graphics_stage_lvalue(context, statement.a, &target)) {
            return false;
        }
        *glsl << pad << (statement.op & 1u ? "++" : "--") << target << ";\n";
        return true;
    }
    default:
        return false;
    }
}

bool append_graphics_callable_declarations(GraphicsLoweringContext& context,
                                           std::ostringstream* glsl) {
    if (glsl == nullptr) {
        return false;
    }

    for (const auto& function : context.module.functions) {
        if (function.kind != 5) {
            continue;
        }
        if (function.mangled_name_id >= context.module.strings.size() ||
            function.return_type_id >= context.module.types.size()) {
            return false;
        }

        const auto return_type = graphics_glsl_type_name(context.module, function.return_type_id);
        if (return_type.empty()) {
            return false;
        }

        *glsl << return_type << " "
              << sanitize_graphics_glsl_identifier(context.module.strings[function.mangled_name_id]) << "(";
        if (function.parameter_count > 0) {
            if (function.first_parameter == UINT32_MAX ||
                function.first_parameter > context.module.parameters.size() ||
                function.parameter_count > context.module.parameters.size() - function.first_parameter) {
                return false;
            }
            for (uint32_t i = 0; i < function.parameter_count; ++i) {
                const auto& parameter = context.module.parameters[function.first_parameter + i];
                if (parameter.name_id >= context.module.strings.size()) {
                    return false;
                }
                const auto parameter_type = graphics_glsl_type_name(context.module, parameter.type_id);
                if (parameter_type.empty()) {
                    return false;
                }
                if (i != 0) {
                    *glsl << ", ";
                }
                if (parameter.direction != 0) {
                    *glsl << "inout ";
                }
                *glsl << parameter_type << " "
                      << sanitize_graphics_glsl_identifier(context.module.strings[parameter.name_id]);
            }
        }
        *glsl << ");\n";
    }

    if (!context.module.functions.empty()) {
        *glsl << "\n";
    }
    return true;
}

bool append_graphics_callable_definitions(GraphicsLoweringContext& context,
                                          std::ostringstream* glsl) {
    if (glsl == nullptr) {
        return false;
    }

    for (const auto& function : context.module.functions) {
        if (function.kind != 5) {
            continue;
        }
        if (function.mangled_name_id >= context.module.strings.size() ||
            function.return_type_id >= context.module.types.size() ||
            function.body_statement_index >= context.module.statements.size()) {
            return false;
        }

        const auto return_type = graphics_glsl_type_name(context.module, function.return_type_id);
        if (return_type.empty()) {
            return false;
        }

        *glsl << return_type << " "
              << sanitize_graphics_glsl_identifier(context.module.strings[function.mangled_name_id]) << "(";
        if (function.parameter_count > 0) {
            if (function.first_parameter == UINT32_MAX ||
                function.first_parameter > context.module.parameters.size() ||
                function.parameter_count > context.module.parameters.size() - function.first_parameter) {
                return false;
            }
            for (uint32_t i = 0; i < function.parameter_count; ++i) {
                const auto& parameter = context.module.parameters[function.first_parameter + i];
                if (parameter.name_id >= context.module.strings.size()) {
                    return false;
                }
                const auto parameter_type = graphics_glsl_type_name(context.module, parameter.type_id);
                if (parameter_type.empty()) {
                    return false;
                }
                if (i != 0) {
                    *glsl << ", ";
                }
                if (parameter.direction != 0) {
                    *glsl << "inout ";
                }
                *glsl << parameter_type << " "
                      << sanitize_graphics_glsl_identifier(context.module.strings[parameter.name_id]);
            }
        }
        *glsl << ") {\n";

        auto callable_context = context;
        callable_context.parameter_name.clear();
        callable_context.locals.clear();
        if (!append_graphics_stage_statement_list(
                callable_context,
                function.body_statement_index,
                1,
                {},
                glsl)) {
            return false;
        }
        *glsl << "}\n\n";
    }

    return true;
}

bool append_graphics_vertex_main(GraphicsLoweringContext& context, std::ostringstream* glsl) {
    if (glsl == nullptr) {
        return false;
    }

    if (context.module.entry_function >= context.module.functions.size()) {
        return false;
    }

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.body_statement_index >= context.module.statements.size()) {
        return false;
    }

    GraphicsReturnEmitter emit_vertex_return = [](GraphicsLoweringContext& context,
                                                  uint32_t expr_id,
                                                  int indent,
                                                  std::ostringstream* glsl) {
        std::string return_expression;
        if (!lower_graphics_stage_expression(context, expr_id, &return_expression)) {
            trace_graphics_expression_failure(context, "vertex", expr_id, "return expression lowering failed");
            return false;
        }

        const auto pad = graphics_indent(indent);
        const auto result_name = "fe_result_" + std::to_string(expr_id);
        if (context.varyings.is_float4) {
            *glsl << pad << "vec4 " << result_name << " = " << return_expression << ";\n";
            *glsl << pad << "gl_Position = " << result_name << ";\n";
            *glsl << pad << "v_fe_color = " << result_name << ";\n";
            *glsl << pad << "v_fe_uv = " << result_name << ".xy * 0.5 + vec2(0.5);\n";
            *glsl << pad << "return;\n";
            return true;
        }

        const auto result_type = graphics_glsl_type_name(context.module, context.varyings.type_id);
        if (result_type.empty() || context.varyings.position_field_name.empty()) {
            return false;
        }

        *glsl << pad << result_type << " " << result_name << " = " << return_expression << ";\n";
        *glsl << pad << "gl_Position = " << result_name << "."
              << sanitize_graphics_glsl_identifier(context.varyings.position_field_name) << ";\n";
        for (const auto& field : context.varyings.fields) {
            if (field.position) {
                continue;
            }
            *glsl << pad << graphics_varying_variable_name(field) << " = " << result_name << "."
                  << sanitize_graphics_glsl_identifier(field.name) << ";\n";
        }
        *glsl << pad << "return;\n";
        return true;
    };

    *glsl << "void main() {\n";
    if (!append_graphics_stage_statement_list(
            context,
            function.body_statement_index,
            1,
            emit_vertex_return,
            glsl)) {
        return false;
    }
    *glsl << "}\n";
    return true;
}

bool append_graphics_fragment_output_declarations(const GraphicsFragmentOutputLayout& outputs,
                                                  std::ostringstream* glsl) {
    if (glsl == nullptr || outputs.fields.empty()) {
        return false;
    }

    for (const auto& field : outputs.fields) {
        if (field.glsl_type.empty()) {
            return false;
        }

        *glsl << "layout(location = " << field.location << ") out "
              << field.glsl_type << " out_color_" << field.location << ";\n";
    }
    return true;
}

bool append_graphics_fragment_main(GraphicsLoweringContext& context,
                                   const GraphicsFragmentOutputLayout& outputs,
                                   std::ostringstream* glsl) {
    if (glsl == nullptr || outputs.fields.empty()) {
        return false;
    }

    if (context.module.entry_function >= context.module.functions.size()) {
        return false;
    }

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.body_statement_index >= context.module.statements.size()) {
        return false;
    }

    GraphicsReturnEmitter emit_fragment_return = [&outputs](GraphicsLoweringContext& context,
                                                            uint32_t expr_id,
                                                            int indent,
                                                            std::ostringstream* glsl) {
        std::string return_expression;
        if (!lower_graphics_stage_expression(context, expr_id, &return_expression)) {
            trace_graphics_expression_failure(context, "fragment", expr_id, "return expression lowering failed");
            return false;
        }

        const auto pad = graphics_indent(indent);
        if (outputs.is_float4) {
            *glsl << pad << "out_color_0 = " << return_expression << ";\n";
            *glsl << pad << "return;\n";
            return true;
        }

        const auto result_type = graphics_glsl_type_name(context.module, outputs.type_id);
        if (result_type.empty()) {
            return false;
        }

        const auto result_name = "fe_result_" + std::to_string(expr_id);
        *glsl << pad << result_type << " " << result_name << " = " << return_expression << ";\n";
        for (const auto& field : outputs.fields) {
            if (field.name.empty()) {
                return false;
            }

            *glsl << pad << "out_color_" << field.location << " = " << result_name << "."
                  << sanitize_graphics_glsl_identifier(field.name) << ";\n";
        }
        *glsl << pad << "return;\n";
        return true;
    };

    *glsl << "void main() {\n";
    if (!append_graphics_stage_statement_list(
            context,
            function.body_statement_index,
            1,
            emit_fragment_return,
            glsl)) {
        return false;
    }
    *glsl << "}\n";
    return true;
}

bool build_graphics_vertex_glsl(const ParsedIr& vertex_ir, const GraphicsPipelineState& pipeline,
                                const std::vector<GraphicsPushConstantLayoutEntry>& push_constants,
                                const GraphicsVaryingLayout& varyings,
                                const GraphicsResourceLayout& resources,
                                std::string* source) {
    if (source == nullptr || !vertex_ir.has_section7 ||
        vertex_ir.typed_module.entry_function >= vertex_ir.typed_module.functions.size()) {
        return false;
    }

    GraphicsLoweringContext context{
        GraphicsStage::Vertex,
        vertex_ir,
        vertex_ir.typed_module,
        pipeline,
        push_constants,
        varyings,
        resources,
        {},
        std::nullopt,
        {},
        {}};

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.kind != 3 || function.body_statement_index >= context.module.statements.size()) {
        return false;
    }

    std::ostringstream glsl;
    if (!append_graphics_shader_prelude(context, &glsl) ||
        !append_graphics_callable_declarations(context, &glsl) ||
        !append_graphics_callable_definitions(context, &glsl) ||
        !append_graphics_vertex_main(context, &glsl)) {
        return false;
    }

    *source = glsl.str();
    return true;
}

bool build_graphics_fragment_glsl(const ParsedIr& fragment_ir, const GraphicsPipelineState& pipeline,
                                  const std::vector<GraphicsPushConstantLayoutEntry>& push_constants,
                                  const GraphicsVaryingLayout& varyings,
                                  const GraphicsResourceLayout& resources,
                                  std::string* source) {
    if (source == nullptr || !fragment_ir.has_section7 ||
        fragment_ir.typed_module.entry_function >= fragment_ir.typed_module.functions.size()) {
        return false;
    }

    GraphicsLoweringContext context{
        GraphicsStage::Fragment,
        fragment_ir,
        fragment_ir.typed_module,
        pipeline,
        push_constants,
        varyings,
        resources,
        "input",
        std::nullopt,
        {},
        {}};

    const auto& function = context.module.functions[context.module.entry_function];
    if (function.kind != 4 || function.body_statement_index >= context.module.statements.size()) {
        return false;
    }
    if (function.parameter_count > 0 && function.first_parameter < context.module.parameters.size()) {
        const auto& parameter = context.module.parameters[function.first_parameter];
        if (parameter.name_id < context.module.strings.size()) {
            context.parameter_name = context.module.strings[parameter.name_id];
        }
    }

    GraphicsFragmentOutputLayout outputs;
    if (!build_graphics_fragment_output_layout(
            context.module,
            function.return_type_id,
            pipeline.color_attachment_count,
            &outputs)) {
        if (feather_graphics_trace_enabled()) {
            std::cerr << "[feather graphics] fragment output type is not a supported color output shape\n";
        }
        return false;
    }

    std::ostringstream glsl;
    if (!append_graphics_shader_prelude(context, &glsl, outputs.is_float4 ? UINT32_MAX : outputs.type_id)) {
        if (feather_graphics_trace_enabled()) {
            std::cerr << "[feather graphics] fragment prelude lowering failed\n";
        }
        return false;
    }
    if (!append_graphics_fragment_output_declarations(outputs, &glsl) ||
        !append_graphics_callable_declarations(context, &glsl) ||
        !append_graphics_callable_definitions(context, &glsl) ||
        !append_graphics_fragment_main(context, outputs, &glsl)) {
        return false;
    }

    *source = glsl.str();
    return true;
}

bool try_build_graphics_pipeline_glsl(const GraphicsPipelineState& pipeline,
                                      std::string* vertex_source,
                                      std::string* fragment_source,
                                      GraphicsResourceLayout* resource_layout,
                                      std::vector<GraphicsPushConstantLayoutEntry>* push_constants,
                                      std::string* failure_reason = nullptr) {
    auto trace_failure = [&](const char* reason) {
        if (failure_reason != nullptr) {
            *failure_reason = reason;
        }
        const auto* trace = std::getenv("FEATHER_GRAPHICS_TRACE");
        if (trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0) {
            std::cerr << "[feather graphics] lowering failed: " << reason << "\n";
        }
        return false;
    };

    if (vertex_source == nullptr || fragment_source == nullptr ||
        resource_layout == nullptr || push_constants == nullptr) {
        return trace_failure("null output pointer");
    }

    ParsedIr vertex_ir;
    ParsedIr fragment_ir;
    if (!parse_feather_ir(pipeline.vertex_ir, &vertex_ir) ||
        !parse_feather_ir(pipeline.fragment_ir, &fragment_ir) ||
        !vertex_ir.has_section7 ||
        !fragment_ir.has_section7 ||
        vertex_ir.typed_module.entry_function >= vertex_ir.typed_module.functions.size()) {
        return trace_failure("invalid typed IR sections");
    }

    const auto& vertex_function = vertex_ir.typed_module.functions[vertex_ir.typed_module.entry_function];
    if (vertex_function.kind != 3 || vertex_function.return_type_id >= vertex_ir.typed_module.types.size()) {
        return trace_failure("invalid vertex entry function");
    }

    GraphicsVaryingLayout varyings;
    if (!build_graphics_varying_layout(vertex_ir.typed_module, vertex_function.return_type_id, &varyings)) {
        return trace_failure("varying layout unsupported");
    }
    if (!build_graphics_resource_layout(vertex_ir, fragment_ir, resource_layout)) {
        return trace_failure("resource layout unsupported");
    }
    if (!build_graphics_push_constant_layout(vertex_ir, fragment_ir, push_constants)) {
        return trace_failure("push constant layout unsupported");
    }

    if (!build_graphics_vertex_glsl(vertex_ir, pipeline, *push_constants, varyings, *resource_layout, vertex_source)) {
        return trace_failure("vertex stage lowering failed");
    }
    if (!build_graphics_fragment_glsl(fragment_ir, pipeline, *push_constants, varyings, *resource_layout, fragment_source)) {
        return trace_failure("fragment stage lowering failed");
    }
    return true;
}

void dump_graphics_glsl_if_requested(const GraphicsPipelineState& pipeline,
                                     const std::string& vertex_source,
                                     const std::string& fragment_source) {
    const auto* dump = std::getenv("FEATHER_GRAPHICS_DUMP_GLSL");
    if (dump == nullptr || dump[0] == '\0' || std::strcmp(dump, "0") == 0) {
        return;
    }

    std::cerr << "=== Feather generated vertex GLSL: " << pipeline.debug_name << " ===\n"
              << vertex_source
              << "\n=== Feather generated fragment GLSL: " << pipeline.debug_name << " ===\n"
              << fragment_source
              << "\n";
}

bool feather_graphics_trace_enabled() {
    const auto* trace = std::getenv("FEATHER_GRAPHICS_TRACE");
    return trace != nullptr && trace[0] != '\0' && std::strcmp(trace, "0") != 0;
}

void trace_graphics_step(const char* step) {
    if (!feather_graphics_trace_enabled()) {
        return;
    }

    std::cerr << "[feather graphics] " << step << "\n";
}

std::string generic_graphics_vertex_glsl() {
    return R"GLSL(#version 450
layout(location = 0) in vec4 in_position;
layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

void main() {
    gl_Position = in_position;
    v_color = vec4(clamp(in_position.xyz * 0.5 + vec3(0.5), vec3(0.0), vec3(1.0)), in_position.w);
    v_uv = in_position.xy * 0.5 + vec2(0.5);
}
)GLSL";
}

std::string generic_graphics_fragment_glsl(bool sampled_texture, GraphicsColor color) {
    std::ostringstream glsl;
    glsl << "#version 450\n";
    glsl << "layout(location = 1) in vec2 v_uv;\n";
    glsl << "layout(location = 0) out vec4 out_color;\n";
    if (sampled_texture) {
        glsl << "layout(set = 0, binding = 0) uniform sampler2D u_texture;\n";
    }
    glsl << "void main() {\n";
    if (sampled_texture) {
        glsl << "    out_color = texture(u_texture, clamp(v_uv, vec2(0.0), vec2(1.0)));\n";
    } else {
        glsl << std::fixed << std::setprecision(8);
        glsl << "    out_color = vec4(" << (static_cast<float>(color.r) / 255.0f) << ", "
             << (static_cast<float>(color.g) / 255.0f) << ", " << (static_cast<float>(color.b) / 255.0f)
             << ", " << (static_cast<float>(color.a) / 255.0f) << ");\n";
    }
    glsl << "}\n";
    return glsl.str();
}

std::string graphics_resource_layout_key(const GraphicsResourceLayout& layout,
                                         const std::vector<GPU::Backend::PixelFormat>& sampled_texture_formats) {
    std::ostringstream key;
    key << "resources=" << layout.entries.size();
    for (const auto& resource : layout.entries) {
        key << "|b" << resource.backend_binding
            << ":s" << resource.source_binding
            << ":k" << static_cast<uint32_t>(resource.kind)
            << ":a" << static_cast<uint32_t>(resource.access)
            << ":sam" << resource.sampler_binding;
        if (resource.kind == kIrResourceKindTexture2D && resource.backend_binding < sampled_texture_formats.size()) {
            key << ":fmt" << static_cast<uint32_t>(sampled_texture_formats[resource.backend_binding]);
        }
    }
    return key.str();
}

std::string graphics_pipeline_variant_key(const GraphicsPipelineState& pipeline,
                                          const std::vector<GPU::Backend::PixelFormat>& color_formats,
                                          GPU::Backend::PixelFormat depth_format,
                                          bool has_depth_attachment,
                                          const GraphicsResourceLayout& layout,
                                          const std::vector<GPU::Backend::PixelFormat>& sampled_texture_formats,
                                          uint32_t push_constant_size) {
    std::ostringstream key;
    key << "topo=" << pipeline.topology
        << "|samples=" << pipeline.sample_count
        << "|colors=" << color_formats.size();
    for (auto format : color_formats) {
        key << ":" << static_cast<uint32_t>(format);
    }
    key << "|depth=" << (has_depth_attachment ? static_cast<uint32_t>(depth_format) : UINT32_MAX)
        << "|depthTest=" << pipeline.depth_test
        << "|depthWrite=" << pipeline.depth_write
        << "|depthCompare=" << pipeline.depth_compare
        << "|stencil=" << pipeline.stencil_test
        << "|stencilMasks=" << pipeline.stencil_read_mask << "," << pipeline.stencil_write_mask << "," << pipeline.stencil_reference
        << "|stencilFront=" << pipeline.stencil_front.fail_op << "," << pipeline.stencil_front.pass_op << ","
        << pipeline.stencil_front.depth_fail_op << "," << pipeline.stencil_front.compare_op
        << "|stencilBack=" << pipeline.stencil_back.fail_op << "," << pipeline.stencil_back.pass_op << ","
        << pipeline.stencil_back.depth_fail_op << "," << pipeline.stencil_back.compare_op
        << "|blend=" << pipeline.blend_enable << "," << pipeline.blend_src_color << "," << pipeline.blend_dst_color << ","
        << pipeline.blend_color_op << "," << pipeline.blend_src_alpha << "," << pipeline.blend_dst_alpha << ","
        << pipeline.blend_alpha_op << "," << pipeline.blend_write_mask
        << "|blendAttachments=" << pipeline.color_blend_attachment_count;
    for (uint32_t i = 0; i < pipeline.color_blend_attachment_count; ++i) {
        const auto& blend = pipeline.color_blend_attachments[i];
        key << ":" << blend.blend_enable << "," << blend.src_color << "," << blend.dst_color << ","
            << blend.color_op << "," << blend.src_alpha << "," << blend.dst_alpha << ","
            << blend.alpha_op << "," << blend.write_mask;
    }
    key
        << "|raster=" << pipeline.cull_mode << "," << pipeline.front_face << "," << pipeline.polygon_mode << ","
        << pipeline.depth_clamp
        << "|push=" << push_constant_size
        << "|" << graphics_resource_layout_key(layout, sampled_texture_formats);
    return key.str();
}

FeResult get_or_create_graphics_pipeline_variant(GraphicsPipelineState& pipeline,
                                                 GPU::Backend::Backend& backend,
                                                 GPU::Backend::PrimitiveTopology topology,
                                                 GPU::Backend::SampleCount sample_count,
                                                 const std::vector<TextureState*>& color_targets,
                                                 const TextureState* depth,
                                                 const std::string& vertex_shader_source,
                                                 const std::string& fragment_shader_source,
                                                 const GraphicsResourceLayout& resource_layout,
                                                 uint32_t push_constant_size,
                                                 GPU::Backend::PipelineHandle* out_pipeline) {
    if (out_pipeline == nullptr) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline variant output is required.");
    }

    std::vector<GPU::Backend::PixelFormat> color_formats;
    color_formats.reserve(color_targets.size());
    for (const auto* color_target : color_targets) {
        GPU::Backend::PixelFormat format = GPU::Backend::PixelFormat::RGBA8;
        if (color_target == nullptr || !easygpu_backend_pixel_format(color_target->pixel_format, &format)) {
            return fail(FE_ERROR_UNSUPPORTED, "Color target format is not supported by EasyGPU graphics.");
        }
        color_formats.push_back(format);
    }

    GPU::Backend::PixelFormat depth_format = GPU::Backend::PixelFormat::D32F;
    const bool has_depth_attachment = depth != nullptr;
    if (has_depth_attachment && !easygpu_backend_pixel_format(depth->pixel_format, &depth_format)) {
        return fail(FE_ERROR_UNSUPPORTED, "Depth target format is not supported by EasyGPU graphics.");
    }

    std::vector<GPU::Backend::PixelFormat> sampled_texture_formats(GPU::Backend::MAX_TEXTURE_BINDINGS,
                                                                    GPU::Backend::PixelFormat::RGBA8);
    for (const auto& resource : resource_layout.entries) {
        if (resource.kind != kIrResourceKindTexture2D) {
            continue;
        }

        const auto bound_texture = pipeline.textures.find(resource.source_binding);
        if (bound_texture == pipeline.textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Generated graphics shader references an unbound sampled texture.");
        }

        const auto texture_it = g_textures.find(bound_texture->second);
        if (texture_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid sampled texture.");
        }

        if (texture_it->second.depth != 1 ||
            !easygpu_backend_pixel_format(texture_it->second.pixel_format, &sampled_texture_formats[resource.backend_binding])) {
            return fail(FE_ERROR_UNSUPPORTED, "Generated graphics shader sampled texture format is not supported.");
        }
    }

    const auto key = graphics_pipeline_variant_key(
        pipeline,
        color_formats,
        depth_format,
        has_depth_attachment,
        resource_layout,
        sampled_texture_formats,
        push_constant_size);
    for (const auto& entry : pipeline.backend_cache) {
        if (entry.key == key) {
            *out_pipeline = entry.pipeline;
            trace_graphics_step("reuse graphics pipeline");
            return ok();
        }
    }

    GPU::Backend::ShaderDesc vertex_shader_desc;
    vertex_shader_desc.type = GPU::Backend::ShaderType::Vertex;
    vertex_shader_desc.sourceCode = vertex_shader_source;
    vertex_shader_desc.entryPoint = "main";
    vertex_shader_desc.optimizationLevel = kShaderOptimizationLevel;
    trace_graphics_step("create vertex shader");
    const auto vertex_shader = backend.CreateShader(vertex_shader_desc);
    if (vertex_shader == GPU::Backend::INVALID_SHADER_HANDLE) {
        return fail(FE_ERROR_SHADER_COMPILE_FAILED, "EasyGPU backend failed to compile generated vertex shader.");
    }

    GPU::Backend::ShaderDesc fragment_shader_desc;
    fragment_shader_desc.type = GPU::Backend::ShaderType::Fragment;
    fragment_shader_desc.sourceCode = fragment_shader_source;
    fragment_shader_desc.entryPoint = "main";
    fragment_shader_desc.optimizationLevel = kShaderOptimizationLevel;
    trace_graphics_step("create fragment shader");
    const auto fragment_shader = backend.CreateShader(fragment_shader_desc);
    if (fragment_shader == GPU::Backend::INVALID_SHADER_HANDLE) {
        backend.DestroyShader(vertex_shader);
        return fail(FE_ERROR_SHADER_COMPILE_FAILED, "EasyGPU backend failed to compile generated fragment shader.");
    }

    GPU::Backend::GraphicsPipelineDesc pipeline_desc;
    pipeline_desc.vertexShader = vertex_shader;
    pipeline_desc.fragmentShader = fragment_shader;
    pipeline_desc.topology = topology;
    pipeline_desc.colorAttachmentFormats = color_formats;
    pipeline_desc.colorAttachmentFormat = color_formats.front();
    pipeline_desc.depthAttachmentFormat = depth_format;
    pipeline_desc.sampleCount = sample_count;
    pipeline_desc.depthTestEnable = depth != nullptr && pipeline.depth_test != 0;
    pipeline_desc.depthWriteEnable = depth != nullptr && pipeline.depth_write != 0;
    if (!map_graphics_compare_op(pipeline.depth_compare, &pipeline_desc.depthCompareOp) ||
        !map_graphics_stencil_face(pipeline.stencil_front, &pipeline_desc.stencilFront) ||
        !map_graphics_stencil_face(pipeline.stencil_back, &pipeline_desc.stencilBack) ||
        !map_graphics_blend_factor(pipeline.blend_src_color, &pipeline_desc.blendSrcColor) ||
        !map_graphics_blend_factor(pipeline.blend_dst_color, &pipeline_desc.blendDstColor) ||
        !map_graphics_blend_op(pipeline.blend_color_op, &pipeline_desc.blendColorOp) ||
        !map_graphics_blend_factor(pipeline.blend_src_alpha, &pipeline_desc.blendSrcAlpha) ||
        !map_graphics_blend_factor(pipeline.blend_dst_alpha, &pipeline_desc.blendDstAlpha) ||
        !map_graphics_blend_op(pipeline.blend_alpha_op, &pipeline_desc.blendAlphaOp) ||
        !map_graphics_raster_state(pipeline, &pipeline_desc.cullMode, &pipeline_desc.frontFace, &pipeline_desc.polygonMode)) {
        backend.DestroyShader(fragment_shader);
        backend.DestroyShader(vertex_shader);
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline descriptor contains an unsupported state enum value.");
    }
    pipeline_desc.stencilTestEnable = depth != nullptr && pipeline.stencil_test != 0;
    pipeline_desc.stencilReadMask = pipeline.stencil_read_mask;
    pipeline_desc.stencilWriteMask = pipeline.stencil_write_mask;
    pipeline_desc.stencilReference = pipeline.stencil_reference;
    pipeline_desc.blendEnable = pipeline.blend_enable != 0;
    pipeline_desc.colorWriteMask = pipeline.blend_write_mask;
    pipeline_desc.colorBlendAttachments.clear();
    pipeline_desc.colorBlendAttachments.reserve(pipeline.color_blend_attachment_count);
    for (uint32_t i = 0; i < pipeline.color_blend_attachment_count; ++i) {
        GPU::Backend::ColorAttachmentBlendState blend_state;
        if (!map_graphics_color_blend_attachment(pipeline.color_blend_attachments[i], &blend_state)) {
            backend.DestroyShader(fragment_shader);
            backend.DestroyShader(vertex_shader);
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics color blend attachment descriptor contains an unsupported value.");
        }
        pipeline_desc.colorBlendAttachments.push_back(blend_state);
    }
    pipeline_desc.depthClampEnable = pipeline.depth_clamp != 0;
    pipeline_desc.pushConstantSize = push_constant_size;

    for (const auto& resource : resource_layout.entries) {
        if (resource.kind == kIrResourceKindBuffer) {
            GPU::Backend::ResourceLayoutEntry entry;
            entry.binding = resource.backend_binding;
            entry.type = GPU::Backend::BindingType::Buffer;
            entry.format = GPU::Backend::PixelFormat::RGBA8;
            entry.readOnly = true;
            entry.stageFlags = GPU::Backend::ResourceStageVertex;
            pipeline_desc.resources.push_back(entry);
            continue;
        }

        if (resource.kind == kIrResourceKindTexture2D) {
            GPU::Backend::ResourceLayoutEntry entry;
            entry.binding = resource.backend_binding;
            entry.type = GPU::Backend::BindingType::Sampler;
            entry.format = sampled_texture_formats[resource.backend_binding];
            entry.readOnly = true;
            entry.stageFlags = GPU::Backend::ResourceStageFragment;
            pipeline_desc.resources.push_back(entry);
        }
    }

    trace_graphics_step("create graphics pipeline");
    GPU::Backend::PipelineHandle backend_pipeline = GPU::Backend::INVALID_PIPELINE_HANDLE;
    try {
        backend_pipeline = backend.CreateGraphicsPipeline(pipeline_desc);
    } catch (const std::exception& ex) {
        backend.DestroyShader(fragment_shader);
        backend.DestroyShader(vertex_shader);
        const std::string message = ex.what();
        if (message.find("fillModeNonSolid") != std::string::npos ||
            message.find("depthClamp") != std::string::npos) {
            return fail(FE_ERROR_UNSUPPORTED, message);
        }
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend failed to create graphics pipeline: " + message);
    }
    if (backend_pipeline == GPU::Backend::INVALID_PIPELINE_HANDLE) {
        backend.DestroyShader(fragment_shader);
        backend.DestroyShader(vertex_shader);
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend failed to create graphics pipeline.");
    }

    GraphicsPipelineCacheEntry entry;
    entry.key = key;
    entry.vertex_shader = vertex_shader;
    entry.fragment_shader = fragment_shader;
    entry.pipeline = backend_pipeline;
    entry.push_constant_size = push_constant_size;
    pipeline.backend_cache.push_back(std::move(entry));
    *out_pipeline = backend_pipeline;
    return ok();
}

FeResult validate_graphics_sampler_binding(const GraphicsPipelineState& pipeline,
                                           const GraphicsResourceBindingEntry& texture_resource,
                                           GPU::Backend::SamplerDesc* out_sampler) {
    if (out_sampler == nullptr) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Sampler descriptor output is required.");
    }

    FeSamplerHandle sampler_handle = 0;
    if (texture_resource.sampler_binding != UINT32_MAX) {
        const auto bound_sampler = pipeline.samplers.find(texture_resource.sampler_binding);
        if (bound_sampler == pipeline.samplers.end() || bound_sampler->second == 0) {
            return fail(FE_ERROR_INVALID_HANDLE,
                        "Generated graphics shader references an unbound sampler.");
        }
        sampler_handle = bound_sampler->second;
    } else if (pipeline.samplers.size() == 1) {
        sampler_handle = pipeline.samplers.begin()->second;
    } else {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Graphics texture sampling requires an explicit SamplerState argument.");
    }

    if (sampler_handle == 0) {
        return fail(FE_ERROR_INVALID_HANDLE,
                    "Generated graphics shader references an unbound sampler.");
    }

    const auto sampler_it = g_samplers.find(sampler_handle);
    if (sampler_it == g_samplers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE,
                    "Graphics draw references an invalid sampler.");
    }

    if (!map_sampler_desc(sampler_it->second.desc, out_sampler)) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Graphics draw references an unsupported sampler descriptor.");
    }

    return ok();
}

FeResult draw_graphics_pipeline_easygpu(GraphicsPipelineState& pipeline, const FeGraphicsDrawDesc& draw) {
    trace_graphics_step("draw begin");
    GPU::Backend::PrimitiveTopology topology;
    if (!map_graphics_topology(pipeline.topology, &topology)) {
        return fail(FE_ERROR_UNSUPPORTED, "Unsupported graphics primitive topology.");
    }

    GPU::Backend::SampleCount sample_count;
    if (!map_graphics_sample_count(pipeline.sample_count, &sample_count)) {
        return fail(FE_ERROR_UNSUPPORTED, "Unsupported graphics sample count.");
    }

    const uint32_t minimum_count =
        topology == GPU::Backend::PrimitiveTopology::PointList
            ? 1u
            : (topology == GPU::Backend::PrimitiveTopology::LineList ||
                       topology == GPU::Backend::PrimitiveTopology::LineStrip
                   ? 2u
                   : 3u);
    if (draw.count < minimum_count) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw count is too small for the selected topology.");
    }
    const uint32_t instance_count = draw.instance_count == 0 ? 1u : draw.instance_count;
    if (draw.color_targets == nullptr || draw.color_target_count == 0 ||
        draw.color_target_count > GPU::Backend::MAX_COLOR_ATTACHMENTS) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw requires one to eight color targets.");
    }
    if (draw.color_target_count != pipeline.color_attachment_count) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw color target count must match the pipeline descriptor.");
    }
    if ((draw.viewport_enabled != 0 && (draw.viewport_width == 0 || draw.viewport_height == 0)) ||
        (draw.scissor_enabled != 0 && (draw.scissor_width == 0 || draw.scissor_height == 0))) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw viewport and scissor rectangles must have positive dimensions.");
    }

    std::vector<TextureState*> targets;
    targets.reserve(draw.color_target_count);
    for (uint32_t i = 0; i < draw.color_target_count; ++i) {
        auto target_it = g_textures.find(draw.color_targets[i]);
        if (target_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid color target handle.");
        }
        targets.push_back(&target_it->second);
    }

    auto& target = *targets.front();
    if (target.depth != 1 || target.width == 0 || target.height == 0) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw requires a valid 2D color target.");
    }
    for (const auto* color_target : targets) {
        if (color_target->depth != 1 ||
            color_target->width != target.width ||
            color_target->height != target.height) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics MRT color target dimensions must match.");
        }
    }

    TextureState* depth = nullptr;
    if (draw.depth_target != 0) {
        auto depth_it = g_textures.find(draw.depth_target);
        if (depth_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid depth target handle.");
        }
        if (depth_it->second.width != target.width || depth_it->second.height != target.height ||
            depth_it->second.depth != 1) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Depth target dimensions must match the color target.");
        }
        if (pipeline.stencil_test != 0) {
            if (depth_it->second.pixel_format != 100) {
                return fail(FE_ERROR_UNSUPPORTED, "Graphics stencil state requires a Depth24Stencil8 depth target.");
            }
        } else if (depth_it->second.pixel_format != 101 && depth_it->second.pixel_format != 100) {
            return fail(FE_ERROR_UNSUPPORTED, "EasyGPU graphics bridge currently supports Depth32Float or Depth24Stencil8 depth targets.");
        }
        depth = &depth_it->second;
    }

    FeBufferHandle vertex_handle = pipeline.vertex_buffer;
    if (vertex_handle == 0 && !pipeline.buffers.empty()) {
        vertex_handle = pipeline.buffers.begin()->second;
    }
    auto vertex_it = g_buffers.find(vertex_handle);
    if (vertex_it == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw requires a bound vertex buffer.");
    }

    uint32_t stride = pipeline.vertex_stride;
    if (stride == 0) {
        stride = vertex_it->second.stride != 0 ? vertex_it->second.stride : static_cast<uint32_t>(sizeof(float) * 4);
    }
    if (stride < sizeof(float) * 4) {
        return fail(FE_ERROR_UNSUPPORTED, "EasyGPU graphics bridge currently requires float4 vertex positions.");
    }

    BufferState* index_buffer = nullptr;
    bool index_buffer_is_uint16 = false;
    if (draw.indexed != 0) {
        if (draw.index_buffer == 0) {
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw requires an explicit index buffer.");
        }

        auto index_it = g_buffers.find(draw.index_buffer);
        if (index_it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw references an invalid index buffer.");
        }
        const auto index_stride = index_it->second.stride != 0 ? index_it->second.stride : sizeof(uint32_t);
        if (index_stride != sizeof(uint32_t) && index_stride != sizeof(uint16_t)) {
            return fail(FE_ERROR_UNSUPPORTED, "EasyGPU graphics bridge currently requires uint or ushort index buffers.");
        }
        const auto required_index_elements = static_cast<uint64_t>(draw.first_index) + draw.count;
        if (required_index_elements > std::numeric_limits<size_t>::max() / index_stride ||
            index_it->second.bytes.size() < static_cast<size_t>(required_index_elements) * index_stride) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Index buffer is too small for the requested indexed draw.");
        }
        index_buffer_is_uint16 = index_stride == sizeof(uint16_t);
        index_buffer = &index_it->second;
    }

    GPU::Runtime::AutoInitContext();
    GPU::Runtime::ContextGuard guard(GPU::Runtime::Context::GetInstance());
    auto* backend = GPU::Runtime::Context::GetBackend();
    if (backend == nullptr) {
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for graphics draw.");
    }
    const auto caps = backend->GetCaps();
    if (!caps.supportsGraphics) {
        return fail(FE_ERROR_UNSUPPORTED, "Active EasyGPU backend does not support graphics pipelines.");
    }

    if (!pipeline.graphics_lowered) {
        std::string graphics_lowering_failure;
        if (!try_build_graphics_pipeline_glsl(
                pipeline,
                &pipeline.vertex_shader_source,
                &pipeline.fragment_shader_source,
                &pipeline.resource_layout,
                &pipeline.push_constant_layout,
                &graphics_lowering_failure)) {
            return fail(FE_ERROR_UNSUPPORTED,
                        "Generated graphics pipeline could not be lowered to typed EasyGPU GLSL: " +
                        graphics_lowering_failure + ".");
        }
        pipeline.graphics_lowered = true;
        dump_graphics_glsl_if_requested(pipeline, pipeline.vertex_shader_source, pipeline.fragment_shader_source);
        trace_graphics_step("glsl lowered");
    }

    const auto vertex_buffer = ensure_easygpu_buffer(vertex_it->second, *backend);
    std::vector<GPU::Backend::TextureHandle> target_textures;
    target_textures.reserve(targets.size());
    for (auto* color_target : targets) {
        const auto target_texture = ensure_easygpu_texture(*color_target, *backend);
        if (target_texture == GPU::Backend::INVALID_TEXTURE_HANDLE) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU color target texture could not be created.");
        }
        target_textures.push_back(target_texture);
    }
    GPU::Backend::TextureHandle depth_texture = GPU::Backend::INVALID_TEXTURE_HANDLE;
    if (depth != nullptr) {
        depth_texture = ensure_easygpu_texture(*depth, *backend);
        if (depth_texture == GPU::Backend::INVALID_TEXTURE_HANDLE) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU depth target texture could not be created.");
        }
    }

    GPU::Backend::BufferHandle index_backend_buffer = GPU::Backend::INVALID_BUFFER_HANDLE;
    std::unique_ptr<EasyGpuBufferGuard> expanded_index_buffer_guard;
    if (draw.indexed != 0 && index_buffer != nullptr) {
        if (index_buffer_is_uint16) {
            std::vector<uint32_t> expanded(draw.count);
            const auto first_index_byte_offset = static_cast<size_t>(draw.first_index) * sizeof(uint16_t);
            for (uint32_t i = 0; i < draw.count; ++i) {
                expanded[i] = read_u16_unaligned(index_buffer->bytes.data() + first_index_byte_offset +
                                                  static_cast<size_t>(i) * sizeof(uint16_t));
            }

            GPU::Backend::BufferDesc desc;
            desc.sizeInBytes = expanded.size() * sizeof(uint32_t);
            desc.mode = GPU::Backend::BufferMode::Read;
            desc.initialData = expanded.data();
            index_backend_buffer = backend->CreateBuffer(desc);
            if (index_backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend failed to create expanded index buffer.");
            }
            expanded_index_buffer_guard = std::make_unique<EasyGpuBufferGuard>(*backend, index_backend_buffer);
        } else {
            index_backend_buffer = ensure_easygpu_buffer(*index_buffer, *backend);
        }
    }

    GPU::Backend::PipelineHandle backend_pipeline = GPU::Backend::INVALID_PIPELINE_HANDLE;
    const uint32_t push_constant_size = static_cast<uint32_t>(graphics_push_constant_total_size(pipeline.push_constant_layout));
    const auto pipeline_result = get_or_create_graphics_pipeline_variant(
        pipeline,
        *backend,
        topology,
        sample_count,
        targets,
        depth,
        pipeline.vertex_shader_source,
        pipeline.fragment_shader_source,
        pipeline.resource_layout,
        push_constant_size,
        &backend_pipeline);
    if (pipeline_result != FE_OK) {
        return pipeline_result;
    }

    const bool has_depth_attachment = depth_texture != GPU::Backend::INVALID_TEXTURE_HANDLE;
    const auto depth_load_op = static_cast<GraphicsDepthLoadOp>(draw.depth_load_op);
    bool clear_depth = false;
    switch (depth_load_op) {
    case GraphicsDepthLoadOp::Default:
        clear_depth = has_depth_attachment;
        break;
    case GraphicsDepthLoadOp::Load:
        if (draw.clear_depth != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT,
                        "GraphicsDrawDesc cannot specify ClearDepth when DepthLoadOp is Load.");
        }
        clear_depth = false;
        break;
    case GraphicsDepthLoadOp::Clear:
        clear_depth = has_depth_attachment;
        break;
    default:
        return fail(FE_ERROR_INVALID_ARGUMENT, "GraphicsDrawDesc depth load op contains an unsupported value.");
    }
    if (draw.clear_depth != 0) {
        clear_depth = has_depth_attachment;
    }
    if (sample_count != GPU::Backend::SampleCount::X1 && has_depth_attachment && !clear_depth) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "MSAA depth load is not supported because EasyGPU uses transient multisampled depth attachments; clear depth for MSAA draws.");
    }

    const auto color_load_op = static_cast<GraphicsColorLoadOp>(draw.color_load_op);
    GPU::Backend::AttachmentLoadOp backend_color_load_op = GPU::Backend::AttachmentLoadOp::Default;
    bool clear_color = false;
    switch (color_load_op) {
    case GraphicsColorLoadOp::Default:
        clear_color = draw.clear_color != 0 || sample_count != GPU::Backend::SampleCount::X1;
        backend_color_load_op = GPU::Backend::AttachmentLoadOp::Default;
        break;
    case GraphicsColorLoadOp::Load:
        if (draw.clear_color != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT,
                        "GraphicsDrawDesc cannot specify ClearColor when ColorLoadOp is Load.");
        }
        clear_color = false;
        backend_color_load_op = GPU::Backend::AttachmentLoadOp::Load;
        break;
    case GraphicsColorLoadOp::Clear:
        clear_color = true;
        backend_color_load_op = GPU::Backend::AttachmentLoadOp::Clear;
        break;
    case GraphicsColorLoadOp::DontCare:
        if (draw.clear_color != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT,
                        "GraphicsDrawDesc cannot specify ClearColor when ColorLoadOp is DontCare.");
        }
        clear_color = false;
        backend_color_load_op = GPU::Backend::AttachmentLoadOp::DontCare;
        break;
    default:
        return fail(FE_ERROR_INVALID_ARGUMENT, "GraphicsDrawDesc color load op contains an unsupported value.");
    }
    GPU::Backend::RenderPassBeginDesc render_pass;
    render_pass.colorAttachment = target_textures.front();
    render_pass.colorAttachments = target_textures;
    render_pass.depthAttachment = depth_texture;
    render_pass.sampleCount = sample_count;
    render_pass.clearColorFlag = clear_color;
    render_pass.colorLoadOp = backend_color_load_op;
    render_pass.clearColor[0] = draw.clear_color != 0 ? draw.clear_color_r : 0.0f;
    render_pass.clearColor[1] = draw.clear_color != 0 ? draw.clear_color_g : 0.0f;
    render_pass.clearColor[2] = draw.clear_color != 0 ? draw.clear_color_b : 0.0f;
    render_pass.clearColor[3] = draw.clear_color != 0 ? draw.clear_color_a : 1.0f;
    render_pass.clearDepthFlag = has_depth_attachment && clear_depth;
    render_pass.clearDepth = draw.clear_depth != 0
                                 ? std::clamp(draw.clear_depth_value, 0.0f, 1.0f)
                                 : 1.0f;

    std::vector<GPU::Backend::ResourceBinding> bindings;
    for (const auto& resource : pipeline.resource_layout.entries) {
        if (resource.kind == kIrResourceKindBuffer) {
            const auto bound_buffer = pipeline.buffers.find(resource.source_binding);
            if (bound_buffer == pipeline.buffers.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Generated graphics shader references an unbound buffer.");
            }

            const auto buffer_it = g_buffers.find(bound_buffer->second);
            if (buffer_it == g_buffers.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid buffer resource.");
            }

            GPU::Backend::ResourceBinding binding;
            binding.binding = resource.backend_binding;
            binding.type = GPU::Backend::BindingType::Buffer;
            binding.buffer = ensure_easygpu_buffer(buffer_it->second, *backend);
            binding.readOnly = true;
            bindings.push_back(binding);
            continue;
        }

        if (resource.kind == kIrResourceKindTexture2D) {
            GPU::Backend::SamplerDesc sampler_desc;
            const auto sampler_validation = validate_graphics_sampler_binding(pipeline, resource, &sampler_desc);
            if (sampler_validation != FE_OK) {
                return sampler_validation;
            }

            const auto bound_texture = pipeline.textures.find(resource.source_binding);
            if (bound_texture == pipeline.textures.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Generated graphics shader references an unbound sampled texture.");
            }

            auto texture_it = g_textures.find(bound_texture->second);
            if (texture_it == g_textures.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid sampled texture.");
            }

            GPU::Backend::PixelFormat texture_format = GPU::Backend::PixelFormat::RGBA8;
            if (!easygpu_backend_pixel_format(texture_it->second.pixel_format, &texture_format)) {
                return fail(FE_ERROR_UNSUPPORTED, "Generated graphics shader sampled texture format is not supported.");
            }

            const auto sampled_backend_texture = ensure_easygpu_texture(texture_it->second, *backend);
            if (sampled_backend_texture == GPU::Backend::INVALID_TEXTURE_HANDLE) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU sampled texture could not be created.");
            }

            GPU::Backend::ResourceBinding binding;
            binding.binding = resource.backend_binding;
            binding.type = GPU::Backend::BindingType::Sampler;
            binding.texture = sampled_backend_texture;
            binding.format = texture_format;
            binding.readOnly = true;
            binding.sampler = sampler_desc;
            bindings.push_back(binding);
        }
    }

    backend->BindPipeline(backend_pipeline);
    backend->BindVertexBuffer(vertex_buffer, stride);

    if (!bindings.empty()) {
        trace_graphics_step("bind resources");
        backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    }

    if (push_constant_size != 0) {
        if (pipeline.push_constants.size() < push_constant_size) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Generated graphics pipeline push constants are not fully bound.");
        }
        trace_graphics_step("set push constants");
        backend->SetUniformData(backend_pipeline, pipeline.push_constants.data(), push_constant_size);
    }

    trace_graphics_step("begin rendering");
    backend->BeginRendering(render_pass);
    const uint32_t viewport_x = draw.viewport_enabled ? draw.viewport_x : 0u;
    const uint32_t viewport_y = draw.viewport_enabled ? draw.viewport_y : 0u;
    const uint32_t viewport_width = draw.viewport_enabled ? draw.viewport_width : target.width;
    const uint32_t viewport_height = draw.viewport_enabled ? draw.viewport_height : target.height;
    const uint32_t scissor_x = draw.scissor_enabled ? draw.scissor_x : 0u;
    const uint32_t scissor_y = draw.scissor_enabled ? draw.scissor_y : 0u;
    const uint32_t scissor_width = draw.scissor_enabled ? draw.scissor_width : target.width;
    const uint32_t scissor_height = draw.scissor_enabled ? draw.scissor_height : target.height;
    backend->SetViewport(viewport_x, viewport_y, viewport_width, viewport_height);
    backend->SetScissor(scissor_x, scissor_y, scissor_width, scissor_height);
    if (!bindings.empty()) {
        trace_graphics_step("bind resources in render pass");
        backend->BindResources(bindings.data(), static_cast<uint32_t>(bindings.size()));
    }

    if (draw.indexed != 0) {
        trace_graphics_step("draw indexed");
        backend->BindIndexBuffer(index_backend_buffer);
        const uint32_t first_index = index_buffer_is_uint16 ? 0u : draw.first_index;
        backend->DrawIndexed(draw.count, instance_count, first_index, draw.vertex_offset, draw.first_instance);
    } else {
        trace_graphics_step("draw");
        backend->Draw(draw.count, instance_count, draw.first_vertex, draw.first_instance);
    }
    trace_graphics_step("end rendering");
    backend->EndRendering();
    if (draw.wait != 0) {
        trace_graphics_step("finish");
        backend->Finish();
    }

    for (auto* color_target : targets) {
        color_target->device_dirty = true;
        color_target->host_dirty = false;
        color_target->mipmaps_dirty = color_target->mipmaps_requested && color_target->mip_levels > 1;
    }
    if (depth != nullptr) {
        depth->device_dirty = true;
        depth->host_dirty = false;
        depth->mipmaps_dirty = depth->mipmaps_requested && depth->mip_levels > 1;
    }
    return ok();
}

FeResult rasterize_graphics_pipeline(GraphicsPipelineState& pipeline, FeTextureHandle color_target,
                                     FeTextureHandle depth_target, FeBufferHandle index_buffer_handle,
                                     uint32_t count, bool indexed) {
    if (pipeline.topology != 0) {
        return fail(FE_ERROR_UNSUPPORTED, "Graphics rasterization currently supports TriangleList topology.");
    }
    if (pipeline.sample_count != 1) {
        return fail(FE_ERROR_UNSUPPORTED, "Graphics rasterization currently supports SampleCount.X1.");
    }
    if (count < 3) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw requires at least three vertices or indices.");
    }

    auto target_it = g_textures.find(color_target);
    if (target_it == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid color target handle.");
    }
    if (depth_target != 0) {
        const auto depth_it = g_textures.find(depth_target);
        if (depth_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid depth target handle.");
        }
        if (depth_it->second.width != target_it->second.width || depth_it->second.height != target_it->second.height) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Depth target dimensions must match the color target.");
        }
    }

    FeBufferHandle vertex_handle = pipeline.vertex_buffer;
    if (vertex_handle == 0 && !pipeline.buffers.empty()) {
        vertex_handle = pipeline.buffers.begin()->second;
    }
    const auto vertex_it = g_buffers.find(vertex_handle);
    if (vertex_it == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw requires a bound vertex buffer.");
    }

    uint32_t stride = pipeline.vertex_stride;
    if (stride == 0) {
        stride = vertex_it->second.stride != 0 ? vertex_it->second.stride : static_cast<uint32_t>(sizeof(float) * 4);
    }
    if (stride < sizeof(float) * 2) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw requires a vertex stride large enough for float2 positions.");
    }

    std::array<uint32_t, 3> indices{0, 1, 2};
    if (indexed) {
        if (index_buffer_handle == 0) {
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw requires an explicit index buffer.");
        }

        const auto index_it = g_buffers.find(index_buffer_handle);
        if (index_it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw references an invalid index buffer.");
        }
        if (index_it->second.bytes.size() >= sizeof(uint32_t) * 3) {
            indices = {
                read_u32_unaligned(index_it->second.bytes.data()),
                read_u32_unaligned(index_it->second.bytes.data() + 4),
                read_u32_unaligned(index_it->second.bytes.data() + 8)};
        } else if (index_it->second.bytes.size() >= sizeof(uint16_t) * 3) {
            indices = {
                static_cast<uint32_t>(read_u16_unaligned(index_it->second.bytes.data())),
                static_cast<uint32_t>(read_u16_unaligned(index_it->second.bytes.data() + 2)),
                static_cast<uint32_t>(read_u16_unaligned(index_it->second.bytes.data() + 4))};
        } else {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Index buffer is too small for an indexed triangle draw.");
        }
    }

    std::array<GraphicsVertex2D, 3> vertices{};
    for (size_t i = 0; i < vertices.size(); ++i) {
        if (!read_graphics_vertex(vertex_it->second, stride, indices[i], &vertices[i])) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Vertex buffer is too small for the requested triangle draw.");
        }
    }

    auto color = graphics_color_from_sampled_texture(pipeline);
    const auto result = rasterize_graphics_triangle(target_it->second, vertices, color);
    if (result != FE_OK) {
        return result;
    }
    target_it->second.host_dirty = true;
    target_it->second.device_dirty = false;
    return ok();
}

bool profiler_enabled_locked() {
    return g_profiler_enabled;
}

void record_profiler_event_locked(const std::string& name, double elapsed_ms, uint32_t group_x, uint32_t group_y,
                                  uint32_t group_z) {
    if (!g_profiler_enabled) {
        return;
    }

    const auto safe_elapsed_ms = elapsed_ms < 0.0 ? 0.0 : elapsed_ms;
    g_profiler_records.push_back(ProfilerRecord{name, safe_elapsed_ms, group_x, group_y, group_z});

    auto& stats = g_profiler_stats[name];
    if (stats.count == 0) {
        stats.min_time_ms = safe_elapsed_ms;
        stats.max_time_ms = safe_elapsed_ms;
    } else {
        stats.min_time_ms = std::min(stats.min_time_ms, safe_elapsed_ms);
        stats.max_time_ms = std::max(stats.max_time_ms, safe_elapsed_ms);
    }

    stats.count += 1;
    stats.total_time_ms += safe_elapsed_ms;
}

double profiler_total_time_locked() {
    double total = 0.0;
    for (const auto& item : g_profiler_stats) {
        total += item.second.total_time_ms;
    }

    return total;
}

std::vector<std::pair<std::string, ProfilerStats>> profiler_sorted_stats_locked() {
    std::vector<std::pair<std::string, ProfilerStats>> stats(g_profiler_stats.begin(), g_profiler_stats.end());
    std::sort(stats.begin(), stats.end(), [](const auto& left, const auto& right) {
        if (left.second.total_time_ms == right.second.total_time_ms) {
            return left.first < right.first;
        }

        return left.second.total_time_ms > right.second.total_time_ms;
    });

    return stats;
}

std::string format_profiler_report_locked() {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(3);

    if (!g_profiler_enabled) {
        stream << "[FeatherProfiler] Profiling is disabled. Call GpuProfiler.SetEnabled(true) to enable.\n";
        return stream.str();
    }

    if (g_profiler_records.empty()) {
        stream << "[FeatherProfiler] No GPU commands recorded.\n";
        return stream.str();
    }

    // Keep the C ABI string report aggregate-oriented so callers do not need record arrays across the boundary.
    const auto total_time_ms = profiler_total_time_locked();
    stream << "Feather GPU Profiling Results\n";
    stream << "Name\tCount\tMin(ms)\tAvg(ms)\tMax(ms)\tTotal(ms)\tPercent\n";
    for (const auto& item : profiler_sorted_stats_locked()) {
        const auto average = item.second.count == 0 ? 0.0 : item.second.total_time_ms / item.second.count;
        const auto percent = total_time_ms <= 0.0 ? 0.0 : item.second.total_time_ms / total_time_ms * 100.0;
        stream << item.first << '\t' << item.second.count << '\t' << item.second.min_time_ms << '\t' << average << '\t'
               << item.second.max_time_ms << '\t' << item.second.total_time_ms << '\t' << std::setprecision(1)
               << percent << std::setprecision(3) << "%\n";
    }

    stream << "TOTAL\t" << g_profiler_records.size() << "\t\t\t\t" << total_time_ms << "\t100.0%\n";
    return stream.str();
}

FeResult write_string(const std::string& value, char* buffer, size_t buffer_size, size_t* out_required_size) {
    if (out_required_size != nullptr) {
        *out_required_size = value.size();
    }

    if (buffer == nullptr || buffer_size == 0) {
        return FE_OK;
    }

    const size_t copied = std::min(buffer_size - 1, value.size());
    std::memcpy(buffer, value.data(), copied);
    buffer[copied] = '\0';
    return FE_OK;
}

#if FEATHER_BUILD_WINDOW
FeWindowEvent to_fe_window_event(const GPU::Window::WindowEvent& source) {
    FeWindowEvent target{};
    std::visit(
        [&](const auto& event) {
            using EventT = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<EventT, GPU::Window::WindowResizeEvent>) {
                target.kind = kWindowEventResize;
                target.width = event.width;
                target.height = event.height;
            } else if constexpr (std::is_same_v<EventT, GPU::Window::WindowCloseEvent>) {
                target.kind = kWindowEventClose;
            } else if constexpr (std::is_same_v<EventT, GPU::Window::KeyEvent>) {
                target.kind = kWindowEventKey;
                target.key = static_cast<uint32_t>(static_cast<int32_t>(event.key));
                target.pressed = event.pressed ? 1u : 0u;
                target.modifiers = static_cast<uint32_t>(event.modifiers);
            } else if constexpr (std::is_same_v<EventT, GPU::Window::CharInputEvent>) {
                target.kind = kWindowEventCharInput;
                target.codepoint = event.codepoint;
            } else if constexpr (std::is_same_v<EventT, GPU::Window::MouseButtonEvent>) {
                target.kind = kWindowEventMouseButton;
                target.mouse_button = static_cast<uint32_t>(event.button);
                target.pressed = event.pressed ? 1u : 0u;
                target.x = event.x;
                target.y = event.y;
                target.modifiers = static_cast<uint32_t>(event.modifiers);
            } else if constexpr (std::is_same_v<EventT, GPU::Window::MouseMoveEvent>) {
                target.kind = kWindowEventMouseMove;
                target.x = event.x;
                target.y = event.y;
                target.dx = event.dx;
                target.dy = event.dy;
            } else if constexpr (std::is_same_v<EventT, GPU::Window::MouseScrollEvent>) {
                target.kind = kWindowEventMouseScroll;
                target.scroll_x = event.dx;
                target.scroll_y = event.dy;
            } else if constexpr (std::is_same_v<EventT, GPU::Window::WindowFocusEvent>) {
                target.kind = kWindowEventFocus;
                target.pressed = event.focused ? 1u : 0u;
            }
        },
        source);
    return target;
}

FeResult present_texture_cpu_locked(GPU::Window::TexturePresenter& presenter, TextureState& texture) {
    if (texture.depth != 1 || texture.width == 0 || texture.height == 0) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Texture presenter requires a valid 2D texture.");
    }
    if (texture.pixel_format != 3 && texture.pixel_format != 4 && texture.pixel_format != 10) {
        return fail(FE_ERROR_UNSUPPORTED, "Texture presenter currently supports Rgba8, Bgra8, and Rgba32Float textures.");
    }

    if (texture.device_dirty) {
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                        "EasyGPU backend is unavailable for texture presentation readback.");
        }

        download_easygpu_texture(texture, *backend);
    }

    const auto pixel_count = static_cast<size_t>(texture.width) * texture.height;
    const auto minimum_bytes = pixel_count * pixel_size(texture.pixel_format);
    if (texture.bytes.size() < minimum_bytes) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Texture storage is smaller than its declared dimensions.");
    }

    if (texture.pixel_format == 3) {
        presenter.Present(reinterpret_cast<const uint32_t*>(texture.bytes.data()), texture.width, texture.height);
        return ok();
    }

    std::vector<uint32_t> rgba(pixel_count);
    if (texture.pixel_format == 10) {
        const auto* floats = reinterpret_cast<const float*>(texture.bytes.data());
        const auto to_byte = [](float value) -> uint32_t {
            const auto clamped = std::min(1.0f, std::max(0.0f, value));
            return static_cast<uint32_t>(clamped * 255.0f + 0.5f);
        };
        for (size_t i = 0; i < rgba.size(); ++i) {
            const auto r = to_byte(floats[i * 4 + 0]);
            const auto g = to_byte(floats[i * 4 + 1]);
            const auto b = to_byte(floats[i * 4 + 2]);
            const auto a = to_byte(floats[i * 4 + 3]);
            rgba[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
        presenter.Present(rgba.data(), texture.width, texture.height);
        return ok();
    }

    const auto* bgra = texture.bytes.data();
    for (size_t i = 0; i < rgba.size(); ++i) {
        const auto b = static_cast<uint32_t>(bgra[i * 4 + 0]);
        const auto g = static_cast<uint32_t>(bgra[i * 4 + 1]);
        const auto r = static_cast<uint32_t>(bgra[i * 4 + 2]);
        const auto a = static_cast<uint32_t>(bgra[i * 4 + 3]);
        rgba[i] = r | (g << 8) | (b << 16) | (a << 24);
    }
    presenter.Present(rgba.data(), texture.width, texture.height);
    return ok();
}
#endif

template <typename Func> FeResult protect(Func&& func) {
    try {
        return func();
    } catch (const std::bad_alloc&) {
        return fail(FE_ERROR_OUT_OF_MEMORY, "Native allocation failed.");
    } catch (const std::exception& ex) {
        const std::string message = ex.what();
        if (message.find("backend") != std::string::npos || message.find("Backend") != std::string::npos ||
            message.find("Vulkan") != std::string::npos || message.find("OpenGL") != std::string::npos ||
            message.find("GPU context") != std::string::npos || message.find("Context not initialized") != std::string::npos) {
            const auto decorated = "EasyGPU backend unavailable: " + message;
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, decorated.c_str());
        }

        if (message.find("shader") != std::string::npos || message.find("Shader") != std::string::npos ||
            message.find("SPIR") != std::string::npos || message.find("GLSL") != std::string::npos ||
            message.find("pipeline") != std::string::npos || message.find("Pipeline") != std::string::npos) {
            const auto decorated = "EasyGPU shader compilation failed: " + message;
            return fail(FE_ERROR_SHADER_COMPILE_FAILED, decorated.c_str());
        }

        return fail(FE_ERROR_UNKNOWN, ex.what());
    } catch (...) {
        return fail(FE_ERROR_UNKNOWN, "Unknown native exception.");
    }
}

} // namespace

extern "C" {

FE_API FeResult fe_context_get_default(FeContextHandle* out_context) {
    return protect([&] {
        if (out_context == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_context must not be null.");
        }
        *out_context = kDefaultContext;
        return ok();
    });
}

FE_API FeResult fe_context_initialize(FeContextHandle context) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }

        (void)require_backend();
        return ok();
    });
}

FE_API FeResult fe_context_shutdown(FeContextHandle context) {
    return context == kDefaultContext || context == 0 ? ok() : fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
}

FE_API FeResult fe_runtime_flush_caches(void) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend != nullptr) {
            backend->FlushPipelineCache();
        }
        return ok();
    });
}

FE_API FeResult fe_runtime_shutdown(void) {
    return protect([&] {
        const bool was_shutting_down = g_runtime_shutting_down.exchange(true, std::memory_order_acq_rel);
        if (was_shutting_down) {
            return ok();
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        destroy_backend_resources_for_shutdown();
        return ok();
    });
}

FE_API FeResult fe_runtime_process_exit(void) {
    return protect([&] {
        const bool was_shutting_down = g_runtime_shutting_down.exchange(true, std::memory_order_acq_rel);
        if (was_shutting_down) {
            return ok();
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        abandon_native_resources_for_process_exit();
        return ok();
    });
}

FE_API FeResult fe_context_get_backend_type(FeContextHandle context, uint32_t* out_backend) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (out_backend == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_backend must not be null.");
        }
        auto& runtime_context = GPU::Runtime::Context::GetInstance();
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable.");
        }

        *out_backend = map_backend_type(runtime_context.GetBackendType());
        if (*out_backend == 0) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend type is unavailable.");
        }

        return ok();
    });
}

FE_API FeResult fe_context_get_caps(FeContextHandle context, FeBackendCaps* out_caps) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (out_caps == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_caps must not be null.");
        }
        auto& runtime_context = GPU::Runtime::Context::GetInstance();
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable.");
        }

        const auto caps = backend->GetCaps();
        int max_x = 0;
        int max_y = 0;
        int max_z = 0;
        runtime_context.GetMaxWorkGroupSize(max_x, max_y, max_z);
#if FEATHER_BUILD_WINDOW
        constexpr uint32_t supports_window = 1u;
#else
        constexpr uint32_t supports_window = 0u;
#endif
        *out_caps = FeBackendCaps{
            map_backend_type(runtime_context.GetBackendType()),
            static_cast<uint32_t>(std::max(max_x, 0)),
            static_cast<uint32_t>(std::max(max_y, 0)),
            static_cast<uint32_t>(std::max(max_z, 0)),
            caps.supportsGraphics ? 1u : 0u,
            0u,
            0u,
            supports_window,
            caps.supportsDepthClamp ? 1u : 0u,
            caps.supportsNonFillPolygonMode ? 1u : 0u
        };
        if (out_caps->backend_type == 0) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend type is unavailable.");
        }

        return ok();
    });
}

FE_API FeResult fe_get_last_error(char* buffer, size_t buffer_size, size_t* out_required_size) {
    return write_string(g_last_error, buffer, buffer_size, out_required_size);
}

FE_API FeResult fe_window_create(const FeWindowDesc* desc, FeWindowHandle* out_window) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (desc == nullptr || out_window == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window descriptor and output handle are required.");
        }
        if (desc->width == 0 || desc->height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window dimensions must be positive.");
        }

        auto* backend = require_backend();
        if (backend == nullptr || backend->GetType() != GPU::Backend::BackendType::Vulkan) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                        "Feather window support requires the EasyGPU Vulkan backend in this build.");
        }

        GPU::Window::WindowConfig config;
        config.width = desc->width;
        config.height = desc->height;
        config.title = desc->title == nullptr ? "Feather" : desc->title;
        config.resizable = desc->resizable != 0;
        config.visible = desc->visible != 0;
        config.vsync = desc->vsync != 0;
        config.highDPI = desc->high_dpi != 0;
        config.centerOnCreate = desc->center_on_create != 0;

        WindowState state;
        state.window = std::make_unique<GPU::Window::AppWindow>(config);

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_windows.emplace(handle, std::move(state));
        *out_window = handle;
        return ok();
#else
        (void)desc;
        (void)out_window;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_destroy(FeWindowHandle window) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (window == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        for (auto it = g_texture_presenters.begin(); it != g_texture_presenters.end();) {
            if (it->second.window_handle == window) {
                it = g_texture_presenters.erase(it);
            } else {
                ++it;
            }
        }
        return g_windows.erase(window) == 1 ? ok() : fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
#else
        (void)window;
        return ok();
#endif
    });
}

FE_API FeResult fe_window_is_open(FeWindowHandle window, bool* out_is_open) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_is_open == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window open-state output pointer must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        *out_is_open = it->second.window->IsOpen();
        return ok();
#else
        (void)window;
        (void)out_is_open;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_close(FeWindowHandle window) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->Close();
        return ok();
#else
        (void)window;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_poll_events(FeWindowHandle window) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->PollEvents();
        return ok();
#else
        (void)window;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_wait_events(FeWindowHandle window) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->WaitEvents();
        return ok();
#else
        (void)window;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_poll_event(FeWindowHandle window, FeWindowEvent* out_event, bool* out_has_event) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_event == nullptr || out_has_event == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window event output pointers must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        GPU::Window::WindowEvent event;
        if (!it->second.window->PollEvent(event)) {
            *out_has_event = false;
            *out_event = FeWindowEvent{};
            return ok();
        }
        *out_has_event = true;
        *out_event = to_fe_window_event(event);
        return ok();
#else
        (void)window;
        (void)out_event;
        (void)out_has_event;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_get_size(FeWindowHandle window, uint32_t* out_width, uint32_t* out_height) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_width == nullptr || out_height == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window size output pointers must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        *out_width = it->second.window->Width();
        *out_height = it->second.window->Height();
        return ok();
#else
        (void)window;
        (void)out_width;
        (void)out_height;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_set_title(FeWindowHandle window, const char* title) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (title == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Window title must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->SetTitle(title);
        return ok();
#else
        (void)window;
        (void)title;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_set_vsync(FeWindowHandle window, bool enabled) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->SetVSync(enabled);
        return ok();
#else
        (void)window;
        (void)enabled;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_is_key_down(FeWindowHandle window, uint32_t key, bool* out_is_down) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_is_down == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Key-state output pointer must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        *out_is_down = it->second.window->IsKeyDown(static_cast<GPU::Window::Key>(static_cast<int32_t>(key)));
        return ok();
#else
        (void)window;
        (void)key;
        (void)out_is_down;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_is_mouse_down(FeWindowHandle window, uint32_t mouse_button, bool* out_is_down) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_is_down == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Mouse-state output pointer must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        *out_is_down = it->second.window->IsMouseDown(static_cast<GPU::Window::MouseButton>(mouse_button));
        return ok();
#else
        (void)window;
        (void)mouse_button;
        (void)out_is_down;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_get_mouse_position(FeWindowHandle window, int32_t* out_x, int32_t* out_y) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_x == nullptr || out_y == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Mouse-position output pointers must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        const auto [x, y] = it->second.window->MousePosition();
        *out_x = x;
        *out_y = y;
        return ok();
#else
        (void)window;
        (void)out_x;
        (void)out_y;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_get_mouse_scroll(FeWindowHandle window, float* out_x, float* out_y) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_x == nullptr || out_y == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Mouse-scroll output pointers must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        const auto [x, y] = it->second.window->MouseScroll();
        *out_x = x;
        *out_y = y;
        return ok();
#else
        (void)window;
        (void)out_x;
        (void)out_y;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_window_present_pixels(FeWindowHandle window, const uint32_t* pixels, uint32_t width,
                                         uint32_t height) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (pixels == nullptr || width == 0 || height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Pixel presentation requires non-null pixels and dimensions.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_windows.find(window);
        if (it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
        it->second.window->Present(pixels, width, height);
        return ok();
#else
        (void)window;
        (void)pixels;
        (void)width;
        (void)height;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_texture_presenter_create(FeWindowHandle window, FeTexturePresenterHandle* out_presenter) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (out_presenter == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture presenter output handle is required.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto window_it = g_windows.find(window);
        if (window_it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }

        TexturePresenterState state;
        state.window_handle = window;
        state.presenter = std::make_unique<GPU::Window::TexturePresenter>(*window_it->second.window);
        const auto handle = next_handle();
        g_texture_presenters.emplace(handle, std::move(state));
        *out_presenter = handle;
        return ok();
#else
        (void)window;
        (void)out_presenter;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_texture_presenter_destroy(FeTexturePresenterHandle presenter) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (presenter == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_texture_presenters.erase(presenter) == 1
                   ? ok()
                   : fail(FE_ERROR_INVALID_HANDLE, "Invalid texture presenter handle.");
#else
        (void)presenter;
        return ok();
#endif
    });
}

FE_API FeResult fe_texture_presenter_present_texture(FeTexturePresenterHandle presenter, FeTextureHandle texture,
                                                     uint32_t mode) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto presenter_it = g_texture_presenters.find(presenter);
        if (presenter_it == g_texture_presenters.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture presenter handle.");
        }
        const auto texture_it = g_textures.find(texture);
        if (texture_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        if (mode != 1) {
            GPU::Runtime::AutoInitContext();
            GPU::Runtime::Context::GetInstance().MakeCurrent();
            auto* backend = GPU::Runtime::Context::GetBackend();
            if (backend != nullptr) {
                try {
                    const auto backend_texture = ensure_easygpu_texture(texture_it->second, *backend);
                    if (backend_texture != GPU::Backend::INVALID_TEXTURE_HANDLE) {
                        if (presenter_it->second.presenter->PresentTextureHandle(backend_texture)) {
                            return ok();
                        }
                        if (mode == 2) {
                            return fail(FE_ERROR_UNSUPPORTED,
                                        "Direct texture presentation is unavailable for this window/backend.");
                        }
                    }
                } catch (...) {
                    if (mode == 2) {
                        throw;
                    }
                }
            } else if (mode == 2) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            "EasyGPU backend is unavailable for direct texture presentation.");
            }
        }

        return present_texture_cpu_locked(*presenter_it->second.presenter, texture_it->second);
#else
        (void)presenter;
        (void)texture;
        (void)mode;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_texture_presenter_present_pixels(FeTexturePresenterHandle presenter, const uint32_t* pixels,
                                                    uint32_t width, uint32_t height) {
    return protect([&] {
#if FEATHER_BUILD_WINDOW
        if (pixels == nullptr || width == 0 || height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Pixel presentation requires non-null pixels and dimensions.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto presenter_it = g_texture_presenters.find(presenter);
        if (presenter_it == g_texture_presenters.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture presenter handle.");
        }
        presenter_it->second.presenter->Present(pixels, width, height);
        return ok();
#else
        (void)presenter;
        (void)pixels;
        (void)width;
        (void)height;
        return fail(FE_ERROR_UNSUPPORTED, "Feather native library was built without window support.");
#endif
    });
}

FE_API FeResult fe_buffer_create(FeContextHandle context, const FeBufferDesc* desc, const void* initial_data,
                                 FeBufferHandle* out_buffer) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_buffer == nullptr || desc->size_in_bytes == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Buffer descriptor and output handle are required.");
        }

        BufferState state;
        state.bytes.resize(static_cast<size_t>(desc->size_in_bytes));
        state.mode = desc->mode;
        state.stride = desc->element_stride;
        if (initial_data != nullptr) {
            std::memcpy(state.bytes.data(), initial_data, state.bytes.size());
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_buffers.emplace(handle, std::move(state));
        *out_buffer = handle;
        return ok();
    });
}

FE_API FeResult fe_buffer_destroy(FeBufferHandle buffer) {
    return protect([&] {
        if (buffer == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_buffers.find(buffer);
        if (it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid buffer handle.");
        }

        if (it->second.backend_buffer != GPU::Backend::INVALID_BUFFER_HANDLE) {
            if (auto* backend = GPU::Runtime::Context::GetBackend(); backend != nullptr) {
                backend->DestroyBuffer(it->second.backend_buffer);
            }
        }

        g_buffers.erase(it);
        return ok();
    });
}

FE_API FeResult fe_buffer_upload(FeBufferHandle buffer, uint64_t offset, uint64_t size, const void* data) {
    return protect([&] {
        if (data == nullptr && size != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Upload data must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_buffers.find(buffer);
        if (it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid buffer handle.");
        }
        if (offset + size > it->second.bytes.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Upload range exceeds buffer size.");
        }
        std::memcpy(it->second.bytes.data() + offset, data, static_cast<size_t>(size));
        it->second.host_dirty = true;
        it->second.device_dirty = false;
        return ok();
    });
}

FE_API FeResult fe_buffer_download(FeBufferHandle buffer, uint64_t offset, uint64_t size, void* out_data) {
    return protect([&] {
        if (out_data == nullptr && size != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Download output must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_buffers.find(buffer);
        if (it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid buffer handle.");
        }
        if (offset + size > it->second.bytes.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Download range exceeds buffer size.");
        }
        if (it->second.device_dirty) {
            auto* backend = GPU::Runtime::Context::GetBackend();
            if (backend == nullptr) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for buffer download.");
            }

            download_easygpu_buffer(it->second, *backend);
        }

        std::memcpy(out_data, it->second.bytes.data() + offset, static_cast<size_t>(size));
        return ok();
    });
}

FE_API FeResult fe_buffer_map(FeBufferHandle buffer, uint32_t, void** out_ptr) {
    return protect([&] {
        if (out_ptr == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_ptr must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_buffers.find(buffer);
        if (it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid buffer handle.");
        }
        if (it->second.device_dirty) {
            auto* backend = GPU::Runtime::Context::GetBackend();
            if (backend == nullptr) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for buffer mapping.");
            }

            download_easygpu_buffer(it->second, *backend);
        }

        *out_ptr = it->second.bytes.data();
        it->second.host_dirty = true;
        return ok();
    });
}

FE_API FeResult fe_buffer_unmap(FeBufferHandle buffer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_buffers.find(buffer) == g_buffers.end() ? fail(FE_ERROR_INVALID_HANDLE, "Invalid buffer handle.") : ok();
}

FE_API FeResult fe_texture2d_create(FeContextHandle context, const FeTexture2DDesc* desc, const void* initial_data,
                                    FeTextureHandle* out_texture) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_texture == nullptr || desc->width == 0 || desc->height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture descriptor and output handle are required.");
        }
        TextureState state;
        state.width = desc->width;
        state.height = desc->height;
        state.depth = 1;
        state.mip_levels = desc->mip_levels;
        state.pixel_format = desc->pixel_format;
        state.access = desc->access;
        state.bytes.resize(static_cast<size_t>(desc->width) * desc->height * pixel_size(desc->pixel_format));
        if (initial_data != nullptr) {
            std::memcpy(state.bytes.data(), initial_data, state.bytes.size());
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_textures.emplace(handle, std::move(state));
        *out_texture = handle;
        return ok();
    });
}

FE_API FeResult fe_texture3d_create(FeContextHandle context, const FeTexture3DDesc* desc, const void* initial_data,
                                    FeTextureHandle* out_texture) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_texture == nullptr || desc->width == 0 || desc->height == 0 || desc->depth == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture descriptor and output handle are required.");
        }

        TextureState state;
        state.width = desc->width;
        state.height = desc->height;
        state.depth = desc->depth;
        state.mip_levels = desc->mip_levels;
        state.pixel_format = desc->pixel_format;
        state.access = desc->access;
        state.bytes.resize(static_cast<size_t>(desc->width) * desc->height * desc->depth *
                           pixel_size(desc->pixel_format));
        if (initial_data != nullptr) {
            std::memcpy(state.bytes.data(), initial_data, state.bytes.size());
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_textures.emplace(handle, std::move(state));
        *out_texture = handle;
        return ok();
    });
}

FE_API FeResult fe_texture_destroy(FeTextureHandle texture) {
    return protect([&] {
        if (texture == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        // Destroy the backend GPU texture if one was created.
        if (it->second.backend_texture != GPU::Backend::INVALID_TEXTURE_HANDLE) {
            if (auto* backend = GPU::Runtime::Context::GetBackend(); backend != nullptr) {
                backend->DestroyTexture(it->second.backend_texture);
            }
        }

        g_textures.erase(it);
        return ok();
    });
}

FE_API FeResult fe_texture2d_upload(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                    const void* data) {
    return protect([&] {
        if (data == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture upload data must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }
        const auto pixel = pixel_size(it->second.pixel_format);
        if (it->second.depth != 1 || x + width > it->second.width || y + height > it->second.height) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture upload range exceeds texture dimensions.");
        }
        const auto* src = static_cast<const unsigned char*>(data);
        for (uint32_t row = 0; row < height; ++row) {
            const auto dst_offset = (static_cast<size_t>(y + row) * it->second.width + x) * pixel;
            std::memcpy(it->second.bytes.data() + dst_offset, src + static_cast<size_t>(row) * width * pixel,
                        static_cast<size_t>(width) * pixel);
        }
        it->second.host_dirty = true;
        it->second.device_dirty = false;
        it->second.mipmaps_dirty = it->second.mipmaps_requested && it->second.mip_levels > 1;
        return ok();
    });
}

FE_API FeResult fe_texture2d_download(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t width, uint32_t height,
                                      void* out_data) {
    return protect([&] {
        if (out_data == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture download output must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        // If the device has newer data, download it first.
        if (it->second.device_dirty) {
            auto* backend = GPU::Runtime::Context::GetBackend();
            if (backend != nullptr) {
                download_easygpu_texture(it->second, *backend);
            }
        }
        const auto pixel = pixel_size(it->second.pixel_format);
        if (it->second.depth != 1 || x + width > it->second.width || y + height > it->second.height) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture download range exceeds texture dimensions.");
        }
        auto* dst = static_cast<unsigned char*>(out_data);
        for (uint32_t row = 0; row < height; ++row) {
            const auto src_offset = (static_cast<size_t>(y + row) * it->second.width + x) * pixel;
            std::memcpy(dst + static_cast<size_t>(row) * width * pixel, it->second.bytes.data() + src_offset,
                        static_cast<size_t>(width) * pixel);
        }
        return ok();
    });
}

FE_API FeResult fe_texture3d_upload(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
                                    uint32_t height, uint32_t depth, const void* data) {
    return protect([&] {
        if (data == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture upload data must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        const auto pixel = pixel_size(it->second.pixel_format);
        if (x + width > it->second.width || y + height > it->second.height || z + depth > it->second.depth) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture upload range exceeds texture dimensions.");
        }

        const auto* src = static_cast<const unsigned char*>(data);
        for (uint32_t slice = 0; slice < depth; ++slice) {
            for (uint32_t row = 0; row < height; ++row) {
                const auto dst_offset =
                    (((static_cast<size_t>(z + slice) * it->second.height + (y + row)) * it->second.width) + x) * pixel;
                const auto src_offset = ((static_cast<size_t>(slice) * height + row) * width) * pixel;
                std::memcpy(it->second.bytes.data() + dst_offset, src + src_offset, static_cast<size_t>(width) * pixel);
            }
        }

        it->second.host_dirty = true;
        it->second.device_dirty = false;
        it->second.mipmaps_dirty = it->second.mipmaps_requested && it->second.mip_levels > 1;
        return ok();
    });
}

FE_API FeResult fe_texture3d_download(FeTextureHandle texture, uint32_t x, uint32_t y, uint32_t z, uint32_t width,
                                      uint32_t height, uint32_t depth, void* out_data) {
    return protect([&] {
        if (out_data == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture download output must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        if (it->second.device_dirty) {
            auto* backend = GPU::Runtime::Context::GetBackend();
            if (backend != nullptr) {
                download_easygpu_texture(it->second, *backend);
            }
        }

        const auto pixel = pixel_size(it->second.pixel_format);
        if (x + width > it->second.width || y + height > it->second.height || z + depth > it->second.depth) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture download range exceeds texture dimensions.");
        }

        auto* dst = static_cast<unsigned char*>(out_data);
        for (uint32_t slice = 0; slice < depth; ++slice) {
            for (uint32_t row = 0; row < height; ++row) {
                const auto src_offset =
                    (((static_cast<size_t>(z + slice) * it->second.height + (y + row)) * it->second.width) + x) * pixel;
                const auto dst_offset = ((static_cast<size_t>(slice) * height + row) * width) * pixel;
                std::memcpy(dst + dst_offset, it->second.bytes.data() + src_offset, static_cast<size_t>(width) * pixel);
            }
        }

        return ok();
    });
}

FE_API FeResult fe_texture_generate_mipmaps(FeTextureHandle texture) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_textures.find(texture);
        if (it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture handle.");
        }

        auto& state = it->second;
        if (state.mip_levels <= 1) {
            state.mipmaps_requested = false;
            state.mipmaps_dirty = false;
            return ok();
        }

        if (state.depth != 1) {
            return fail(FE_ERROR_UNSUPPORTED, "Mipmap generation currently supports 2D textures only.");
        }
        if (state.pixel_format == 101) {
            return fail(FE_ERROR_UNSUPPORTED, "Depth textures do not support mipmap generation.");
        }

        state.mipmaps_requested = true;
        state.mipmaps_dirty = true;

        GPU::Runtime::AutoInitContext();
        GPU::Runtime::Context::GetInstance().MakeCurrent();
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for mipmap generation.");
        }

        const auto backend_texture = ensure_easygpu_texture(state, *backend);
        if (backend_texture == GPU::Backend::INVALID_TEXTURE_HANDLE) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU texture could not be created for mipmap generation.");
        }

        return ok();
    });
}

FE_API FeResult fe_sampler_create(FeContextHandle context, const FeSamplerDesc* desc, FeSamplerHandle* out_sampler) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_sampler == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Sampler descriptor and output handle are required.");
        }
        GPU::Backend::SamplerDesc mapped_sampler;
        if (!map_sampler_desc(*desc, &mapped_sampler)) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Sampler descriptor contains an unsupported enum value.");
        }

        SamplerState state;
        state.desc = *desc;
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_samplers.emplace(handle, state);
        *out_sampler = handle;
        return ok();
    });
}

FE_API FeResult fe_sampler_destroy(FeSamplerHandle sampler) {
    return protect([&] {
        if (sampler == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_samplers.erase(sampler) == 1 ? ok() : fail(FE_ERROR_INVALID_HANDLE, "Invalid sampler handle.");
    });
}

FE_API FeResult fe_kernel_create_from_ir(FeContextHandle context, const FeKernelCreateDesc* desc,
                                         FeKernelHandle* out_kernel) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_kernel == nullptr || desc->ir_data == nullptr || desc->ir_size == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel IR and output handle are required.");
        }
        const auto validation = fe_ir_validate(desc->ir_data, desc->ir_size);
        if (validation != FE_OK) {
            return fail(validation, "Kernel IR failed Feather IR validation.");
        }
        KernelState state;
        const auto* bytes = static_cast<const unsigned char*>(desc->ir_data);
        state.ir.assign(bytes, bytes + desc->ir_size);
        state.debug_name = copy_debug_name(desc->debug_name, "Kernel");
        state.auto_diff = desc->auto_diff;
        state.bounds_check = desc->bounds_check;
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_kernels.emplace(handle, std::move(state));
        *out_kernel = handle;
        return ok();
    });
}

FE_API FeResult fe_kernel_destroy(FeKernelHandle kernel) {
    return protect([&] {
        if (kernel == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        erase_compute_kernel_cache(kernel);
        release_ad_gradient_buffers(it->second);
        g_kernels.erase(it);
        return ok();
    });
}

FE_API FeResult fe_kernel_bind_buffer(FeKernelHandle kernel, uint32_t binding, FeBufferHandle buffer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    if (it == g_kernels.end() || g_buffers.find(buffer) == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or buffer handle.");
    }
    it->second.buffers[binding] = buffer;
    return ok();
}

FE_API FeResult fe_kernel_bind_texture(FeKernelHandle kernel, uint32_t binding, FeTextureHandle texture) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    if (it == g_kernels.end() || g_textures.find(texture) == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or texture handle.");
    }
    it->second.textures[binding] = texture;
    return ok();
}

FE_API FeResult fe_kernel_bind_sampler(FeKernelHandle kernel, uint32_t binding, FeSamplerHandle sampler) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    if (it == g_kernels.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
    }
    if (sampler != 0 && g_samplers.find(sampler) == g_samplers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid sampler handle.");
    }
    it->second.samplers[binding] = sampler;
    return ok();
}

FE_API FeResult fe_kernel_set_push_constants(FeKernelHandle kernel, const void* data, uint64_t size) {
    return protect([&] {
        if (data == nullptr && size != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Push constant data must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        auto& kernel_state = it->second;
        if (size > kernel_state.push_constants.capacity()) {
            erase_compute_kernel_cache(kernel);
        }

        kernel_state.push_constants.resize(static_cast<size_t>(size));
        if (size != 0) {
            std::memcpy(kernel_state.push_constants.data(), data, static_cast<size_t>(size));
        }
        return ok();
    });
}

FE_API FeResult fe_kernel_dispatch(FeKernelHandle kernel, uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                   uint32_t logical_x, uint32_t logical_y, uint32_t logical_z, bool wait) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        if (group_x == 0 || group_y == 0 || group_z == 0 ||
            logical_x == 0 || logical_y == 0 || logical_z == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel dispatch group counts and logical sizes must be positive.");
        }

        if (logical_x > static_cast<uint32_t>(INT32_MAX) ||
            logical_y > static_cast<uint32_t>(INT32_MAX) ||
            logical_z > static_cast<uint32_t>(INT32_MAX)) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel logical dispatch sizes must fit in a signed 32-bit shader int.");
        }

        it->second.logical_x = static_cast<int32_t>(logical_x);
        it->second.logical_y = static_cast<int32_t>(logical_y);
        it->second.logical_z = static_cast<int32_t>(logical_z);

        const auto should_profile = profiler_enabled_locked();
        const auto start = std::chrono::steady_clock::now();
        FeResult result = FE_OK;

        // Use the AD dispatch path for AutoDiff kernels, which sets up the GradienTape
        // and generates the backward pass after the forward dispatch.
        it->second.last_dispatch_path = FE_DISPATCH_PATH_NONE;
        if (it->second.auto_diff) {
            if (try_dispatch_easygpu_ad_kernel(it->second, group_x, group_y, group_z, wait)) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_TYPED_EASYGPU;
            } else {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                result = g_last_result == FE_OK ? fail(FE_ERROR_UNSUPPORTED, "AD dispatch failed.") : g_last_result;
            }
        } else if (try_dispatch_easygpu_buffer_kernel(kernel, it->second, group_x, group_y, group_z, wait)) {
            it->second.last_dispatch_path = FE_DISPATCH_PATH_TYPED_EASYGPU;
        } else if (has_typed_section7_semantics(it->second)) {
            it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
            result = g_last_error.empty()
                         ? fail(FE_ERROR_UNSUPPORTED, "Typed EasyGPU dispatch rejected the section 7 kernel.")
                         : FE_ERROR_UNSUPPORTED;
        } else {
            result = dispatch_simple_buffer_assignment(it->second, group_x, group_y, group_z);
            it->second.last_dispatch_path =
                result == FE_OK ? FE_DISPATCH_PATH_CPU_REFERENCE_FALLBACK : FE_DISPATCH_PATH_REJECTED;
        }

        if (should_profile && result == FE_OK) {
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            record_profiler_event_locked(it->second.debug_name, elapsed, logical_x, logical_y, logical_z);
        }

        return result;
    });
}

FE_API FeResult fe_kernel_get_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size, size_t* out_required_size) {
    return protect([&] {
        KernelState state;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const auto it = g_kernels.find(kernel);
            if (it == g_kernels.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
            }

            state = it->second;
        }

        std::string source;
        const auto result = build_easygpu_kernel_source(state, &source);
        if (result != FE_OK) {
            return result;
        }

        return write_string(source, buffer, buffer_size, out_required_size);
    });
}

FE_API FeResult fe_kernel_get_optimized_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size,
                                             size_t* out_required_size) {
    return protect([&] {
        KernelState state;
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            const auto it = g_kernels.find(kernel);
            if (it == g_kernels.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
            }

            state = it->second;
        }

        std::string source;
        if (!try_build_easygpu_optimized_kernel_source(state, &source)) {
            return fail(FE_ERROR_UNSUPPORTED,
                        "Kernel IR is not supported by EasyGPU optimized GLSL inspection on this backend.");
        }

        return write_string(source, buffer, buffer_size, out_required_size);
    });
}

FE_API FeResult fe_kernel_get_last_dispatch_path(FeKernelHandle kernel, uint32_t* out_path) {
    return protect([&] {
        if (out_path == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_path must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }

        *out_path = static_cast<uint32_t>(it->second.last_dispatch_path);
        return ok();
    });
}

FE_API FeResult fe_kernel_get_ad_gradient_count(FeKernelHandle kernel, uint32_t* out_count) {
    return protect([&] {
        if (out_count == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_count must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }

        *out_count = static_cast<uint32_t>(it->second.ad_gradients.size());
        return ok();
    });
}

FE_API FeResult fe_kernel_get_ad_gradient_info(FeKernelHandle kernel, uint32_t index, FeADGradientInfo* out_info) {
    return protect([&] {
        if (out_info == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_info must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        if (index >= it->second.ad_gradients.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient index is out of range.");
        }

        const auto& gradient = it->second.ad_gradients[index];
        std::memset(out_info, 0, sizeof(*out_info));
        copy_fixed_c_string(out_info->name, sizeof(out_info->name), gradient.name);
        copy_fixed_c_string(out_info->resource_name, sizeof(out_info->resource_name), gradient.resource_name);
        copy_fixed_c_string(out_info->element_type, sizeof(out_info->element_type), gradient.element_type);
        copy_fixed_c_string(out_info->easygpu_name, sizeof(out_info->easygpu_name), gradient.easygpu_name);
        out_info->source_binding = gradient.source_binding;
        out_info->gradient_binding = gradient.gradient_binding;
        out_info->element_count = gradient.element_count;
        out_info->element_stride = gradient.element_stride;
        out_info->byte_size = static_cast<uint64_t>(gradient.byte_size);
        out_info->component_count = gradient.component_count;
        return ok();
    });
}

FE_API FeResult fe_kernel_read_ad_gradient(FeKernelHandle kernel, uint32_t index, uint64_t offset, uint64_t size,
                                           void* out_data) {
    return protect([&] {
        if (out_data == nullptr && size != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient output buffer must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        if (index >= it->second.ad_gradients.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient index is out of range.");
        }

        auto& gradient = it->second.ad_gradients[index];
        if (offset > gradient.byte_size || size > gradient.byte_size - offset) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient read range exceeds gradient buffer size.");
        }
        if (size == 0) {
            return ok();
        }
        if (gradient.backend_buffer == GPU::Backend::INVALID_BUFFER_HANDLE) {
            return fail(FE_ERROR_INVALID_HANDLE, "AD gradient buffer is not available.");
        }

        GPU::Runtime::AutoInitContext();
        GPU::Runtime::Context::GetInstance().MakeCurrent();
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for AD gradient readback.");
        }

        backend->DownloadBuffer(
            gradient.backend_buffer,
            static_cast<size_t>(offset),
            static_cast<size_t>(size),
            out_data);
        return ok();
    });
}

FE_API FeResult fe_kernel_reduce_ad_gradient_to_buffer(FeKernelHandle kernel, uint32_t index, FeBufferHandle destination,
                                                       uint64_t destination_offset, uint64_t destination_size) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        auto kernel_it = g_kernels.find(kernel);
        if (kernel_it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }
        if (index >= kernel_it->second.ad_gradients.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient index is out of range.");
        }

        auto destination_it = g_buffers.find(destination);
        if (destination_it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid destination buffer handle.");
        }

        if (destination_size == 0) {
            return ok();
        }

        GPU::Runtime::AutoInitContext();
        GPU::Runtime::Context::GetInstance().MakeCurrent();
        auto* backend = GPU::Runtime::Context::GetBackend();
        if (backend == nullptr) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "EasyGPU backend is unavailable for AD gradient reduction.");
        }

        std::string error;
        if (!dispatch_ad_gradient_reduce_to_buffer(
                kernel_it->second.ad_gradients[index],
                destination_it->second,
                destination_offset,
                destination_size,
                *backend,
                &error)) {
            return fail(FE_ERROR_UNSUPPORTED, error.empty() ? "AD gradient could not be reduced to the destination buffer." : error.c_str());
        }

        return ok();
    });
}

FE_API FeResult fe_kernel_get_ad_backward_glsl(FeKernelHandle kernel, char* buffer, size_t buffer_size,
                                               size_t* out_required_size) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_kernels.find(kernel);
        if (it == g_kernels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel handle.");
        }

        return write_string(it->second.last_ad_backward_glsl, buffer, buffer_size, out_required_size);
    });
}

FE_API FeResult fe_graphics_pipeline_create_from_ir(FeContextHandle context, const FeGraphicsPipelineCreateDesc* desc,
                                                    FeGraphicsPipelineHandle* out_pipeline) {
    return protect([&] {
        if (context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        if (desc == nullptr || out_pipeline == nullptr || desc->ir_data == nullptr || desc->ir_size == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline IR and output handle are required.");
        }
        const auto validation = fe_ir_validate(desc->ir_data, desc->ir_size);
        if (validation != FE_OK) {
            return fail(validation, "Graphics pipeline IR failed Feather IR validation.");
        }

        const auto* vertex_ir_data = desc->vertex_ir_data != nullptr ? desc->vertex_ir_data : desc->ir_data;
        const auto vertex_ir_size = desc->vertex_ir_data != nullptr ? desc->vertex_ir_size : desc->ir_size;
        const auto* fragment_ir_data = desc->fragment_ir_data != nullptr ? desc->fragment_ir_data : desc->ir_data;
        const auto fragment_ir_size = desc->fragment_ir_data != nullptr ? desc->fragment_ir_size : desc->ir_size;
        if (vertex_ir_data == nullptr || vertex_ir_size == 0 || fragment_ir_data == nullptr || fragment_ir_size == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline vertex and fragment IR are required.");
        }

        const auto vertex_validation = fe_ir_validate(vertex_ir_data, vertex_ir_size);
        if (vertex_validation != FE_OK) {
            return fail(vertex_validation, "Graphics pipeline vertex IR failed Feather IR validation.");
        }
        const auto fragment_validation = fe_ir_validate(fragment_ir_data, fragment_ir_size);
        if (fragment_validation != FE_OK) {
            return fail(fragment_validation, "Graphics pipeline fragment IR failed Feather IR validation.");
        }
        if (desc->color_attachment_count == 0 || desc->color_attachment_count > GPU::Backend::MAX_COLOR_ATTACHMENTS) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color attachment count must be between 1 and 8.");
        }
        if (desc->color_blend_attachment_count != 0 &&
            desc->color_blend_attachment_count != desc->color_attachment_count) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color blend attachment count must match color attachment count.");
        }
        GPU::Backend::CompareOp unused_compare;
        GPU::Backend::StencilFaceState unused_face;
        GPU::Backend::BlendFactor unused_factor;
        GPU::Backend::BlendOp unused_blend_op;
        GPU::Backend::CullMode unused_cull;
        GPU::Backend::FrontFace unused_front_face;
        GPU::Backend::PolygonMode unused_polygon_mode;
        GPU::Backend::ColorAttachmentBlendState unused_attachment_blend;
        GraphicsPipelineState validation_state;
        validation_state.cull_mode = desc->cull_mode;
        validation_state.front_face = desc->front_face;
        validation_state.polygon_mode = desc->polygon_mode;
        if (!map_graphics_compare_op(desc->depth_compare, &unused_compare) ||
            !map_graphics_stencil_face(desc->stencil_front, &unused_face) ||
            !map_graphics_stencil_face(desc->stencil_back, &unused_face) ||
            !map_graphics_blend_factor(desc->blend_src_color, &unused_factor) ||
            !map_graphics_blend_factor(desc->blend_dst_color, &unused_factor) ||
            !map_graphics_blend_factor(desc->blend_src_alpha, &unused_factor) ||
            !map_graphics_blend_factor(desc->blend_dst_alpha, &unused_factor) ||
            !map_graphics_blend_op(desc->blend_color_op, &unused_blend_op) ||
            !map_graphics_blend_op(desc->blend_alpha_op, &unused_blend_op) ||
            !map_graphics_raster_state(validation_state, &unused_cull, &unused_front_face, &unused_polygon_mode)) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline descriptor contains an unsupported state enum value.");
        }
        if ((desc->blend_write_mask & ~15u) != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline blend write mask contains unsupported bits.");
        }
        for (uint32_t i = 0; i < desc->color_blend_attachment_count; ++i) {
            if (!map_graphics_color_blend_attachment(desc->color_blend_attachments[i], &unused_attachment_blend)) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color blend attachment descriptor contains an unsupported value.");
            }
        }

        GraphicsPipelineState state;
        const auto* bytes = static_cast<const unsigned char*>(desc->ir_data);
        state.ir.assign(bytes, bytes + desc->ir_size);
        const auto* vertex_bytes = static_cast<const unsigned char*>(vertex_ir_data);
        state.vertex_ir.assign(vertex_bytes, vertex_bytes + vertex_ir_size);
        const auto* fragment_bytes = static_cast<const unsigned char*>(fragment_ir_data);
        state.fragment_ir.assign(fragment_bytes, fragment_bytes + fragment_ir_size);
        state.debug_name = copy_debug_name(desc->debug_name, "GraphicsPipeline");
        state.topology = desc->topology;
        state.sample_count = desc->sample_count == 0 ? 1 : desc->sample_count;
        state.color_attachment_count = desc->color_attachment_count == 0 ? 1 : desc->color_attachment_count;
        state.depth_test = desc->depth_test;
        state.depth_write = desc->depth_write;
        state.depth_compare = desc->depth_compare;
        state.stencil_test = desc->stencil_test;
        state.stencil_front = desc->stencil_front;
        state.stencil_back = desc->stencil_back;
        state.stencil_read_mask = desc->stencil_read_mask;
        state.stencil_write_mask = desc->stencil_write_mask;
        state.stencil_reference = desc->stencil_reference;
        state.blend_enable = desc->blend_enable;
        state.blend_src_color = desc->blend_src_color;
        state.blend_dst_color = desc->blend_dst_color;
        state.blend_color_op = desc->blend_color_op;
        state.blend_src_alpha = desc->blend_src_alpha;
        state.blend_dst_alpha = desc->blend_dst_alpha;
        state.blend_alpha_op = desc->blend_alpha_op;
        state.blend_write_mask = desc->blend_write_mask;
        state.color_blend_attachment_count =
            desc->color_blend_attachment_count == 0 ? state.color_attachment_count : desc->color_blend_attachment_count;
        for (uint32_t i = 0; i < state.color_blend_attachment_count; ++i) {
            state.color_blend_attachments[i] = desc->color_blend_attachment_count == 0
                                                   ? FeGraphicsColorBlendAttachmentDesc{
                                                         desc->blend_enable,
                                                         desc->blend_src_color,
                                                         desc->blend_dst_color,
                                                         desc->blend_color_op,
                                                         desc->blend_src_alpha,
                                                         desc->blend_dst_alpha,
                                                         desc->blend_alpha_op,
                                                         desc->blend_write_mask}
                                                   : desc->color_blend_attachments[i];
        }
        state.cull_mode = desc->cull_mode;
        state.front_face = desc->front_face;
        state.polygon_mode = desc->polygon_mode;
        state.depth_clamp = desc->depth_clamp;
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto handle = next_handle();
        g_pipelines.emplace(handle, std::move(state));
        *out_pipeline = handle;
        return ok();
    });
}

FE_API FeResult fe_graphics_pipeline_destroy(FeGraphicsPipelineHandle pipeline) {
    return protect([&] {
        if (pipeline == 0) {
            return ok();
        }
        if (g_runtime_shutting_down.load(std::memory_order_acquire)) {
            return ok();
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(pipeline);
        if (it == g_pipelines.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid graphics pipeline handle.");
        }
        destroy_graphics_pipeline_cache(it->second);
        g_pipelines.erase(it);
        return ok();
    });
}

FE_API FeResult fe_graphics_pipeline_set_vertex_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer,
                                                       uint32_t stride) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    if (it == g_pipelines.end() || g_buffers.find(buffer) == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or buffer handle.");
    }
    it->second.vertex_buffer = buffer;
    it->second.vertex_stride = stride;
    return ok();
}

FE_API FeResult fe_graphics_pipeline_set_index_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer) {
    (void)pipeline;
    (void)buffer;
    return fail(FE_ERROR_UNSUPPORTED,
                "Persistent graphics index-buffer binding is not supported. Pass the index buffer to DrawIndexed for each draw.");
}

FE_API FeResult fe_graphics_pipeline_bind_buffer(FeGraphicsPipelineHandle pipeline, uint32_t binding, FeBufferHandle buffer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    if (it == g_pipelines.end() || g_buffers.find(buffer) == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or buffer handle.");
    }
    it->second.buffers[binding] = buffer;
    return ok();
}

FE_API FeResult fe_graphics_pipeline_bind_texture(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                  FeTextureHandle texture) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    if (it == g_pipelines.end() || g_textures.find(texture) == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or texture handle.");
    }
    it->second.textures[binding] = texture;
    return ok();
}

FE_API FeResult fe_graphics_pipeline_bind_sampler(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                  FeSamplerHandle sampler) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    if (it == g_pipelines.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline handle.");
    }
    if (sampler != 0 && g_samplers.find(sampler) == g_samplers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid sampler handle.");
    }
    it->second.samplers[binding] = sampler;
    return ok();
}

FE_API FeResult fe_graphics_pipeline_set_push_constants(FeGraphicsPipelineHandle pipeline, const void* data,
                                                        uint64_t size) {
    return protect([&] {
        if (data == nullptr && size != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Push constant data must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(pipeline);
        if (it == g_pipelines.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid graphics pipeline handle.");
        }
        const auto* bytes = static_cast<const unsigned char*>(data);
        it->second.push_constants.assign(bytes, bytes + size);
        return ok();
    });
}

FE_API FeResult fe_graphics_pipeline_draw(FeGraphicsPipelineHandle pipeline, FeTextureHandle color_target,
                                          FeTextureHandle depth_target, uint32_t vertex_count, bool wait) {
    FeGraphicsDrawDesc desc{};
    desc.color_targets = &color_target;
    desc.color_target_count = 1;
    desc.depth_target = depth_target;
    desc.count = vertex_count;
    desc.index_buffer = 0;
    desc.indexed = 0;
    desc.wait = wait ? 1u : 0u;
    desc.clear_depth = depth_target != 0 ? 1u : 0u;
    desc.clear_depth_value = 1.0f;
    desc.depth_load_op = static_cast<uint32_t>(GraphicsDepthLoadOp::Clear);
    return fe_graphics_pipeline_draw_ex(pipeline, &desc);
}

FE_API FeResult fe_graphics_pipeline_draw_ex(FeGraphicsPipelineHandle pipeline, const FeGraphicsDrawDesc* desc) {
    return protect([&] {
        if (desc == nullptr || desc->color_targets == nullptr || desc->color_target_count == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw descriptor requires at least one color target.");
        }
        if (desc->color_target_count > GPU::Backend::MAX_COLOR_ATTACHMENTS) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw color target count exceeds EasyGPU limits.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto it = g_pipelines.find(pipeline);
        if (it == g_pipelines.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid graphics pipeline handle.");
        }
        if (desc->color_target_count != it->second.color_attachment_count) {
            it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw color target count must match the pipeline descriptor.");
        }

        for (uint32_t i = 0; i < desc->color_target_count; ++i) {
            if (desc->color_targets[i] == 0 || g_textures.find(desc->color_targets[i]) == g_textures.end()) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid color target.");
            }
        }
        if (desc->depth_target != 0 && g_textures.find(desc->depth_target) == g_textures.end()) {
            it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid depth target.");
        }
        if (desc->indexed != 0 && (desc->index_buffer == 0 || g_buffers.find(desc->index_buffer) == g_buffers.end())) {
            it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
            return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw requires a valid explicit index buffer.");
        }

        const auto should_profile = profiler_enabled_locked();
        const auto start = std::chrono::steady_clock::now();
        auto result = draw_graphics_pipeline_easygpu(it->second, *desc);
        auto dispatch_path = result == FE_OK ? FE_DISPATCH_PATH_TYPED_EASYGPU : FE_DISPATCH_PATH_REJECTED;

        it->second.last_dispatch_path = dispatch_path;
        if (should_profile && result == FE_OK) {
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            record_profiler_event_locked(it->second.debug_name, elapsed, 1, 1, 1);
        }

        return result;
    });
}

FE_API FeResult fe_graphics_pipeline_draw_indexed(FeGraphicsPipelineHandle pipeline, FeTextureHandle color_target,
                                                  FeTextureHandle depth_target, FeBufferHandle index_buffer,
                                                  uint32_t index_count, bool wait) {
    FeGraphicsDrawDesc desc{};
    desc.color_targets = &color_target;
    desc.color_target_count = 1;
    desc.depth_target = depth_target;
    desc.count = index_count;
    desc.index_buffer = index_buffer;
    desc.indexed = 1;
    desc.wait = wait ? 1u : 0u;
    desc.clear_depth = depth_target != 0 ? 1u : 0u;
    desc.clear_depth_value = 1.0f;
    desc.depth_load_op = static_cast<uint32_t>(GraphicsDepthLoadOp::Clear);
    return fe_graphics_pipeline_draw_ex(pipeline, &desc);
}

FE_API FeResult fe_graphics_pipeline_get_last_dispatch_path(FeGraphicsPipelineHandle pipeline, uint32_t* out_path) {
    return protect([&] {
        if (out_path == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_path must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        const auto it = g_pipelines.find(pipeline);
        if (it == g_pipelines.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid graphics pipeline handle.");
        }

        *out_path = static_cast<uint32_t>(it->second.last_dispatch_path);
        return ok();
    });
}

FE_API FeResult fe_profiler_set_enabled(bool enabled) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_profiler_enabled = enabled;
        return ok();
    });
}

FE_API FeResult fe_profiler_is_enabled(bool* out_enabled) {
    return protect([&] {
        if (out_enabled == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Profiler enabled output pointer must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        *out_enabled = g_profiler_enabled;
        return ok();
    });
}

FE_API FeResult fe_profiler_clear(void) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_profiler_records.clear();
        g_profiler_stats.clear();
        return ok();
    });
}

FE_API FeResult fe_profiler_get_total_time(double* out_total_time_ms) {
    return protect([&] {
        if (out_total_time_ms == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Profiler total-time output pointer must not be null.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        *out_total_time_ms = profiler_total_time_locked();
        return ok();
    });
}

FE_API FeResult fe_profiler_query(const char* name, FeProfilerQueryResult* out_result) {
    return protect([&] {
        if (name == nullptr || out_result == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Profiler query name and output result are required.");
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        *out_result = FeProfilerQueryResult{};

        const auto it = g_profiler_stats.find(name);
        if (it == g_profiler_stats.end()) {
            return ok();
        }

        out_result->count = it->second.count;
        out_result->min_time_ms = it->second.min_time_ms;
        out_result->max_time_ms = it->second.max_time_ms;
        out_result->total_time_ms = it->second.total_time_ms;
        out_result->average_time_ms =
            it->second.count == 0 ? 0.0 : it->second.total_time_ms / static_cast<double>(it->second.count);
        return ok();
    });
}

FE_API FeResult fe_profiler_get_formatted(char* buffer, size_t buffer_size, size_t* out_required_size) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        return write_string(format_profiler_report_locked(), buffer, buffer_size, out_required_size);
    });
}

} // extern "C"
