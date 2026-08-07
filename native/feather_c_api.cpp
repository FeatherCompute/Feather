#include "feather_c_api.h"
#include "feather_typed_ir.h"
#include "feather_typed_ir_lowerer.h"

#include "feather_luisa_backend.h"

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

#if FEATHER_BUILD_WINDOW
#include "feather_window_host.h"
#endif

namespace {

void trace_graphics_step(const char* step) {
    if (std::getenv("FEATHER_GRAPHICS_TRACE") != nullptr)
        std::cerr << "[feather graphics] " << step << '\n';
}

constexpr FeContextHandle kDefaultContext = 1;
constexpr uint32_t kMaximumColorAttachments = 8u;
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
constexpr uint8_t kIrResourceKindAccel = 7;
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
    FeContextHandle context = kDefaultContext;
    std::vector<unsigned char> bytes;
    uint32_t mode = 0;
    uint32_t stride = 0;
    bool host_dirty = false;
    bool luisa_uploaded = false;
    uint64_t content_revision = 1u;
};

template <size_t N> struct FloatVectorValue {
    float components[N]{};
};

struct TextureState {
    FeContextHandle context = kDefaultContext;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    uint32_t mip_levels = 1;
    uint32_t pixel_format = 0;
    uint32_t access = 0;
    std::vector<unsigned char> bytes;
    bool host_dirty = false;
    bool mipmaps_requested = false;
    bool mipmaps_dirty = false;
    bool luisa_dirty = false;
    bool luisa_uploaded = false;
    uint64_t content_revision = 1u;
};

struct SamplerState {
    FeContextHandle context = kDefaultContext;
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
    uint32_t stage_flags = 0;
};

struct GraphicsResourceLayout {
    std::vector<GraphicsResourceBindingEntry> entries;
};

struct ADGradientState {
    std::string name;
    std::string resource_name;
    std::string element_type;
    std::string native_name;
    uint32_t source_binding = 0;
    uint32_t gradient_binding = 0;
    uint32_t element_count = 0;
    uint32_t element_stride = 0;
    uint32_t component_count = 0;
    size_t byte_size = 0;
    std::vector<unsigned char> host_bytes;
};

struct KernelState {
    FeContextHandle context = kDefaultContext;
    std::vector<unsigned char> ir;
    std::vector<unsigned char> push_constants;
    std::unordered_map<uint32_t, FeBufferHandle> buffers;
    std::unordered_map<uint32_t, FeTextureHandle> textures;
    std::unordered_map<uint32_t, FeSamplerHandle> samplers;
    std::unordered_map<uint32_t, FeAccelHandle> accels;
    std::string debug_name;
    bool auto_diff = false;
    bool bounds_check = false;
    int32_t logical_x = 0;
    int32_t logical_y = 0;
    int32_t logical_z = 0;
    std::vector<unsigned char> backward_ir;
    std::vector<ADGradientState> ad_gradients;
    FeDispatchPath last_dispatch_path = FE_DISPATCH_PATH_NONE;
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
    FeContextHandle context = kDefaultContext;
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
    std::array<FeGraphicsColorBlendAttachmentDesc, kMaximumColorAttachments> color_blend_attachments{};
    uint32_t cull_mode = 0;
    uint32_t front_face = 0;
    uint32_t polygon_mode = 0;
    uint32_t depth_clamp = 0;
    std::unordered_map<uint32_t, FeBufferHandle> buffers;
    std::unordered_map<uint32_t, FeTextureHandle> textures;
    std::unordered_map<uint32_t, FeSamplerHandle> samplers;
    std::string debug_name;
    FeDispatchPath last_dispatch_path = FE_DISPATCH_PATH_NONE;
    std::vector<GraphicsPushConstantLayoutEntry> push_constant_layout;
    uint64_t compute_raster_index_cache_key = 0u;
    uint32_t compute_raster_maximum_vertex = 0u;
    std::vector<uint32_t> compute_raster_indices;
};

#if FEATHER_BUILD_WINDOW
enum : uint32_t {
    kPresentationFrameFree = 0u,
    kPresentationFramePending = 1u,
    kPresentationFrameReady = 2u,
};

struct AsyncPresentationFrame {
    std::vector<unsigned char> bytes;
    std::atomic<uint32_t> state{kPresentationFrameFree};
    uint64_t revision = 0u;
};

struct AsyncTexturePresentation {
    std::array<std::shared_ptr<AsyncPresentationFrame>, 3u> frames{
        std::make_shared<AsyncPresentationFrame>(),
        std::make_shared<AsyncPresentationFrame>(),
        std::make_shared<AsyncPresentationFrame>()};
    uint64_t last_scheduled_revision = 0u;
    bool has_presented_frame = false;
};

struct WindowState {
    std::unique_ptr<Feather::WindowHost> window;
    std::unordered_set<FeContextHandle> native_contexts;
};

struct TexturePresenterState {
    FeWindowHandle window_handle = 0;
    std::unordered_map<FeTextureHandle, std::shared_ptr<AsyncTexturePresentation>> async_textures;
    std::unordered_set<FeContextHandle> native_contexts;
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

struct ContextDeviceState {
    FeDeviceInfo device{};
};

struct StreamState {
    FeContextHandle context = 0u;
};

struct FenceState {
    FeContextHandle context = 0u;
    FeStreamHandle stream = 0u;
};

std::mutex g_mutex;
std::atomic<uint64_t> g_next_handle{100};
std::atomic<bool> g_runtime_shutting_down{false};
std::unordered_map<FeBufferHandle, BufferState> g_buffers;
struct AccelState {
    FeContextHandle context = kDefaultContext;
    uint64_t accel_key = 0u;
};
std::unordered_map<FeAccelHandle, AccelState> g_accels;
std::unordered_map<FeTextureHandle, TextureState> g_textures;
std::unordered_map<FeSamplerHandle, SamplerState> g_samplers;
std::unordered_map<FeKernelHandle, KernelState> g_kernels;
std::unordered_map<FeGraphicsPipelineHandle, GraphicsPipelineState> g_pipelines;
std::unordered_map<FeContextHandle, ContextDeviceState> g_contexts;
std::unordered_map<FeStreamHandle, StreamState> g_streams;
std::unordered_map<FeFenceHandle, FenceState> g_fences;
#if FEATHER_BUILD_WINDOW
std::unordered_map<FeWindowHandle, WindowState> g_windows;
std::unordered_map<FeTexturePresenterHandle, TexturePresenterState> g_texture_presenters;
#endif
bool g_profiler_enabled = false;
std::vector<ProfilerRecord> g_profiler_records;
std::unordered_map<std::string, ProfilerStats> g_profiler_stats;
thread_local std::string g_last_error;
thread_local FeResult g_last_result = FE_OK;

bool context_exists_locked(FeContextHandle context) {
    return context == kDefaultContext || g_contexts.find(context) != g_contexts.end();
}

void release_ad_gradient_buffers(KernelState& kernel) {
    kernel.ad_gradients.clear();
}

void destroy_backend_resources_for_shutdown() {
    Feather::Luisa::Shutdown();

#if FEATHER_BUILD_WINDOW
    g_texture_presenters.clear();
    g_windows.clear();
#endif

    g_pipelines.clear();
    g_kernels.clear();

    g_textures.clear();
    g_buffers.clear();
    g_accels.clear();
    g_samplers.clear();
    g_fences.clear();
    g_streams.clear();
    g_contexts.clear();
    g_profiler_records.clear();
    g_profiler_stats.clear();
}

void abandon_native_resources_for_process_exit() {
    Feather::Luisa::Abandon();
#if FEATHER_BUILD_WINDOW
    g_texture_presenters.clear();

    for (auto& [handle, window] : g_windows) {
        (void)handle;
        (void)window.window.release();
    }
    g_windows.clear();
#endif

    g_pipelines.clear();
    g_kernels.clear();
    g_textures.clear();
    g_buffers.clear();
    g_accels.clear();

    g_samplers.clear();
    g_fences.clear();
    g_streams.clear();
    g_contexts.clear();
    g_profiler_records.clear();
    g_profiler_stats.clear();
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

bool luisa_pixel_format(uint32_t format) {
    switch (format) {
    case 1u:
    case 2u:
    case 3u:
    case 5u:
    case 6u:
    case 7u:
    case 8u:
    case 9u:
    case 10u:
        return true;
    default:
        return false;
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
    // keeps the typed Roslyn-to-native bridge contract explicit.
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

size_t vector_buffer_stride(size_t component_count) {
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

std::string native_buffer_name(const IrResource& resource) {
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

std::string configured_luisa_runtime_directory() {
    const auto* configured = std::getenv("FEATHER_LUISA_RUNTIME_DIR");
    return configured != nullptr && configured[0] != '\0'
               ? std::string{configured}
               : Feather::Luisa::RuntimeDirectory();
}

FeDeviceInfo make_device_info(const Feather::Luisa::DeviceInfo& source) {
    FeDeviceInfo result{};
    copy_fixed_c_string(result.backend_name, sizeof(result.backend_name), source.backend_name);
    copy_fixed_c_string(result.device_name, sizeof(result.device_name), source.device_name);
    result.device_index = source.device_index;
    result.is_default = source.backend_name == Feather::Luisa::DefaultBackendName &&
                                source.device_index == 0u
                            ? 1u
                            : 0u;
    result.compute_warp_size = source.compute_warp_size;
    result.bindless_capacity_sufficient = FE_CAPABILITY_UNKNOWN;
    result.subgroup = FE_CAPABILITY_UNKNOWN;
    result.quad = FE_CAPABILITY_UNKNOWN;
    return result;
}

bool configure_luisa_dispatch_locked(FeContextHandle context,
                                     Feather::Luisa::DispatchInputs* dispatch,
                                     std::string* error) {
    if (dispatch == nullptr) return false;
    dispatch->context_key = context;
    dispatch->runtime_directory = configured_luisa_runtime_directory();
    if (context == kDefaultContext) {
        const auto* configured_backend = std::getenv("FEATHER_LUISA_BACKEND");
        dispatch->backend_name = configured_backend != nullptr && configured_backend[0] != '\0'
                                     ? configured_backend
                                     : std::string{Feather::Luisa::DefaultBackendName};
        if (dispatch->backend_name == "vulkan") dispatch->backend_name = "vk";
        dispatch->device_index = UINT32_MAX;
        return true;
    }
    const auto found = g_contexts.find(context);
    if (found == g_contexts.end()) {
        if (error != nullptr) *error = "Invalid context handle.";
        return false;
    }
    dispatch->backend_name = found->second.device.backend_name;
    dispatch->device_index = found->second.device.device_index;
    return true;
}

bool synchronize_luisa_context_locked(FeContextHandle context, std::string* error = nullptr) {
    return Feather::Luisa::Synchronize(context, error);
}

bool configured_luisa_presentation_backend_locked() {
    Feather::Luisa::DispatchInputs dispatch;
    std::string error;
    if (!configure_luisa_dispatch_locked(kDefaultContext, &dispatch, &error)) return false;
    return dispatch.backend_name == "metal" || dispatch.backend_name == "vk" ||
           dispatch.backend_name == "dx";
}

uint32_t ad_component_count_for_type(const std::string& type_name) {
    if (type_name == "System.Single" || type_name == "float") return 1u;
    if (type_name == "Feather.Math.float2" || type_name == "global::Feather.Math.float2" || type_name == "float2") return 2u;
    if (type_name == "Feather.Math.float3" || type_name == "global::Feather.Math.float3" || type_name == "float3") return 3u;
    if (type_name == "Feather.Math.float4" || type_name == "global::Feather.Math.float4" || type_name == "float4") return 4u;
    return 0;
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

    // The integer vectors, which the managed layout has always packed the same way as their float
    // counterparts (GpuValueLayout pairs int2 with float2, int3 with float3, int4 with float4).
    // Omitting them here meant push_constant_type_size returned zero, and push-constant resource
    // rejected the binding, and a kernel taking a Uniform<int3> failed the dispatch gate with
    // "does not support push constant binding" -- despite the generator, the shader model validator
    // and the GLSL lowering all accepting it. A grid size is the natural int3 uniform, so this was
    // reachable by any 3D kernel that needed to know its own extent.
    for (size_t components = 2; components <= 4; ++components) {
        if (is_int_vector_type_name(*type, components) || is_uint_vector_type_name(*type, components)) {
            return components * sizeof(int32_t);
        }
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

    // Integer vectors align like their float counterparts, for the reason given in
    // push_constant_type_size: the managed layout treats the two as the same shape, so a three
    // component vector occupies twelve bytes on a sixteen byte boundary either way.
    if (is_int_vector_type_name(*type, 2) || is_uint_vector_type_name(*type, 2)) {
        return 8;
    }

    for (size_t components = 3; components <= 4; ++components) {
        if (is_int_vector_type_name(*type, components) || is_uint_vector_type_name(*type, components)) {
            return 16;
        }
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

        // Keep uniform offsets aligned to their declared native layout.
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

    if (element_stride == vector_buffer_stride(2) && is_float_vector_resource(ir, *destination, 2) &&
        (is_float_vector_type(ir, root->type_string_id, 2) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<2>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    if (element_stride == vector_buffer_stride(3) && is_float_vector_resource(ir, *destination, 3) &&
        (is_float_vector_type(ir, root->type_string_id, 3) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<3>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    if (element_stride == vector_buffer_stride(4) && is_float_vector_resource(ir, *destination, 4) &&
        (is_float_vector_type(ir, root->type_string_id, 4) || root->type_string_id == UINT32_MAX)) {
        return execute_float_vector_expression_assignment<4>(kernel, ir, assignment, destination_buffer->second,
                                                             copied_elements);
    }

    return fail(FE_ERROR_UNSUPPORTED,
                "Kernel expression fallback currently supports int, float, float2, float3, and float4 buffer elements.");
}

bool build_typed_ir_lowering_inputs(const ParsedIr& ir, const KernelState& kernel,
                                    Feather::TypedIR::LoweringInputs* inputs,
                                    bool allow_unbound_samplers = false) {
    if (inputs == nullptr) return false;
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
        if (name == nullptr || element_type == nullptr) return false;

        Feather::TypedIR::ResourceInfo resource_info;
        resource_info.binding = resource.binding;
        resource_info.kind = resource.kind;
        resource_info.access = resource.access;
        resource_info.name = *name;
        resource_info.element_type = *element_type;

        if (resource.kind == kIrResourceKindPushConstant) {
            size_t offset = 0u;
            size_t size = 0u;
            if (!find_push_constant_offset(ir, resource.binding, &offset, &size)) return false;
            inputs->push_constants.push_back(Feather::TypedIR::PushConstantInfo{
                resource.binding,
                offset + size <= kernel.push_constants.size()
                    ? const_cast<unsigned char*>(kernel.push_constants.data() + offset)
                    : nullptr,
                size,
                push_constant_type_alignment(ir, resource)});
        } else if (resource.kind == kIrResourceKindBuffer) {
            const auto bound = kernel.buffers.find(resource.binding);
            if (bound == kernel.buffers.end()) return false;
            const auto buffer = g_buffers.find(bound->second);
            if (buffer == g_buffers.end() || buffer->second.stride == 0u ||
                buffer->second.bytes.size() % buffer->second.stride != 0u ||
                buffer->second.bytes.size() / buffer->second.stride > UINT32_MAX) return false;
            resource_info.element_count =
                static_cast<uint32_t>(buffer->second.bytes.size() / buffer->second.stride);
        } else if (resource.kind == kIrResourceKindSampler) {
            const auto bound = kernel.samplers.find(resource.binding);
            if (bound == kernel.samplers.end()) {
                if (!allow_unbound_samplers) return false;
            } else {
                const auto sampler = g_samplers.find(bound->second);
                if (sampler == g_samplers.end()) return false;
                const auto& desc = sampler->second.desc;
                resource_info.sampler_min_filter = desc.min_filter;
                resource_info.sampler_mag_filter = desc.mag_filter;
                resource_info.sampler_mipmap_mode = desc.mipmap_mode;
                resource_info.sampler_address_u = desc.address_u;
                resource_info.sampler_address_v = desc.address_v;
                resource_info.sampler_address_w = desc.address_w;
                resource_info.sampler_anisotropy = desc.anisotropy_enable != 0u;
            }
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            resource_info.sampled = resource.access == 4u;
            const auto bound = kernel.textures.find(resource.binding);
            if (bound != kernel.textures.end()) {
                const auto texture = g_textures.find(bound->second);
                if (texture != g_textures.end()) {
                    resource_info.width = texture->second.width;
                    resource_info.height = texture->second.height;
                    resource_info.depth = texture->second.depth;
                    resource_info.mip_levels = texture->second.mip_levels;
                    resource_info.texture_format = texture->second.pixel_format;
                }
            }
        }
        inputs->resources.push_back(std::move(resource_info));
    }
    return true;
}

FeResult dispatch_luisa_kernel(FeKernelHandle kernel_handle, KernelState& kernel, uint32_t group_x, uint32_t group_y,
                               uint32_t group_z, uint32_t logical_x, uint32_t logical_y, uint32_t logical_z, bool wait,
                               uint64_t stream_key = 0u, uint64_t fence_key = 0u) {
#if !FEATHER_HAS_LUISA
    (void)kernel_handle;
    (void)kernel;
    (void)group_x;
    (void)group_y;
    (void)group_z;
    (void)logical_x;
    (void)logical_y;
    (void)logical_z;
    (void)wait;
    return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Feather was built without the LuisaCompute Vulkan backend.");
#else
    // Luisa's resident stream and resource cache are per native context. Keep
    // command recording and cache mutation serialized when managed callers
    // dispatch from multiple worker threads.
    static std::mutex luisa_dispatch_mutex;
    std::scoped_lock dispatch_lock{luisa_dispatch_mutex};
    ParsedIr ir;
    if (!parse_feather_ir(kernel.ir, &ir) || !ir.has_section7) {
        return fail(FE_ERROR_UNSUPPORTED, "Luisa dispatch requires a valid section 7 typed IR payload.");
    }

    Feather::TypedIR::LoweringInputs lowering;
    if (!build_typed_ir_lowering_inputs(ir, kernel, &lowering)) {
        return fail(FE_ERROR_INVALID_ARGUMENT,
                    "Section 7 typed IR resources could not be matched to bound native resources.");
    }
    // Compute kernels share their compiled LC shader across dispatches. Uniform
    // values are command data, not shader identity, so expose them as dynamic
    // push-constant buffers; otherwise the first dispatch's values would be
    // folded into the cached shader and later NN/parameterized dispatches would
    // silently reuse stale constants.
    lowering.dynamic_push_constants = true;

    std::optional<Feather::Luisa::AdInputs> ad_inputs;
    std::vector<ADGradientState> next_gradients;
    if (kernel.auto_diff) {
        std::string unsupported_control_flow;
        if (typed_ir_contains_unsupported_ad_control_flow(ir.typed_module, &unsupported_control_flow)) {
            return fail(FE_ERROR_UNSUPPORTED, "Feather AD " + unsupported_control_flow + ".");
        }
        // XIR reverse mode accepts only fixed-trip loops. Specialize AD shaders
        // for their uniform values so loop bounds remain compile-time constants.
        lowering.dynamic_push_constants = false;
        std::vector<IrAdAnnotation> parameters;
        std::vector<IrAdAnnotation> losses;
        for (const auto& annotation : ir.ad_annotations) {
            if (annotation.role == kIrAdRoleParameter) parameters.push_back(annotation);
            else if (annotation.role == kIrAdRoleLoss) losses.push_back(annotation);
        }
        if (parameters.empty() || losses.size() != 1u)
            return fail(FE_ERROR_UNSUPPORTED, "Luisa AD requires at least one parameter and exactly one loss annotation.");
        auto loss_name = string_or_empty(ir, losses.front().name_string_id);
        if (loss_name.empty() || (losses.front().source_kind != kIrAdSourceKindLocal && losses.front().source_kind != 0u))
            return fail(FE_ERROR_UNSUPPORTED, "Luisa AD loss must identify a scalar float local.");
        ad_inputs.emplace();
        ad_inputs->loss_name = std::move(loss_name);
        next_gradients.reserve(parameters.size());
        std::unordered_set<uint32_t> seen;
        for (const auto& parameter : parameters) {
            if (parameter.binding == kIrNoBinding || parameter.source_kind != kIrAdSourceKindBufferElement ||
                !seen.insert(parameter.binding).second) continue;
            const auto* resource = find_resource_by_binding(ir, parameter.binding);
            const auto bound = kernel.buffers.find(parameter.binding);
            if (resource == nullptr || resource->kind != kIrResourceKindBuffer || bound == kernel.buffers.end())
                return fail(FE_ERROR_UNSUPPORTED, "Luisa AD parameter must identify a bound buffer element.");
            auto buffer = g_buffers.find(bound->second);
            if (buffer == g_buffers.end() || buffer->second.stride == 0u || buffer->second.bytes.empty())
                return fail(FE_ERROR_INVALID_HANDLE, "Luisa AD parameter buffer is invalid or empty.");
            auto element_type = string_or_empty(ir, parameter.type_name_string_id);
            if (element_type.empty()) element_type = string_or_empty(ir, resource->element_type_string_id);
            const auto component_count = ad_component_count_for_type(element_type);
            if (component_count == 0u)
                return fail(FE_ERROR_UNSUPPORTED, "Luisa AD supports float scalar and vector parameters.");
            const auto element_count = static_cast<uint32_t>(buffer->second.bytes.size() / buffer->second.stride);
            ad_inputs->parameters.push_back({parameter.binding, element_count});
            ADGradientState gradient;
            gradient.name = string_or_empty(ir, parameter.name_string_id);
            if (gradient.name.empty()) gradient.name = string_or_empty(ir, resource->name_string_id);
            gradient.resource_name = string_or_empty(ir, parameter.resource_name_string_id);
            if (gradient.resource_name.empty()) gradient.resource_name = string_or_empty(ir, resource->name_string_id);
            gradient.element_type = element_type;
            gradient.native_name = native_buffer_name(*resource);
            gradient.source_binding = parameter.binding;
            gradient.element_count = element_count;
            gradient.element_stride = static_cast<uint32_t>(buffer->second.stride);
            gradient.component_count = component_count;
            next_gradients.push_back(std::move(gradient));
        }
        if (next_gradients.empty())
            return fail(FE_ERROR_UNSUPPORTED, "Luisa AD did not resolve any unique parameter buffers.");
    }

    std::vector<Feather::Luisa::HostBufferBinding> bindings;
    bindings.reserve(lowering.resources.size());
    std::vector<Feather::Luisa::HostTextureBinding> texture_bindings;
    texture_bindings.reserve(lowering.resources.size());
    std::vector<Feather::Luisa::HostAccelBinding> accel_bindings;
    accel_bindings.reserve(lowering.resources.size());
    for (const auto& resource : lowering.resources) {
        if (resource.kind == kIrResourceKindPushConstant) {
            continue;
        }
        if (resource.kind == kIrResourceKindSampler) {
            continue;
        }
        if (resource.kind == kIrResourceKindAccel) {
            const auto bound = kernel.accels.find(resource.binding);
            if (bound == kernel.accels.end())
                return fail(FE_ERROR_INVALID_ARGUMENT, "Luisa dispatch is missing a required accel binding.");
            auto accel = g_accels.find(bound->second);
            if (accel == g_accels.end())
                return fail(FE_ERROR_INVALID_HANDLE, "Luisa dispatch references an invalid Feather accel.");
            accel_bindings.push_back(Feather::Luisa::HostAccelBinding{.binding = resource.binding,
                                                                      .accel_key = accel->second.accel_key});
            continue;
        }
        if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto bound = kernel.textures.find(resource.binding);
            if (bound == kernel.textures.end())
                return fail(FE_ERROR_INVALID_ARGUMENT, "Luisa dispatch is missing a required texture binding.");
            auto texture = g_textures.find(bound->second);
            if (texture == g_textures.end())
                return fail(FE_ERROR_INVALID_HANDLE, "Luisa dispatch references an invalid Feather texture.");
            texture_bindings.push_back(Feather::Luisa::HostTextureBinding{.binding = resource.binding,
                                                                          .kind = resource.kind,
                                                                          .access = resource.access,
                                                                          .width = texture->second.width,
                                                                          .height = texture->second.height,
                                                                          .depth = texture->second.depth,
                                                                          .mip_levels = texture->second.mip_levels,
                                                                          .pixel_format = texture->second.pixel_format,
                                                                          .bytes = &texture->second.bytes,
                                                                          .resident_key = bound->second,
                                                                          .upload = !texture->second.luisa_uploaded});
            continue;
        }
        if (resource.kind != kIrResourceKindBuffer) {
            return fail(FE_ERROR_UNSUPPORTED, "The Luisa buffer path received an unsupported resource kind.");
        }
        const auto bound = kernel.buffers.find(resource.binding);
        if (bound == kernel.buffers.end()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Luisa dispatch is missing a required buffer binding.");
        }
        auto buffer = g_buffers.find(bound->second);
        if (buffer == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Luisa dispatch references an invalid Feather buffer.");
        }
        bindings.push_back(Feather::Luisa::HostBufferBinding{.binding = resource.binding,
                                                             .access = resource.access,
                                                             .stride = buffer->second.stride,
                                                             .bytes = &buffer->second.bytes,
                                                             .resident_key = bound->second,
                                                             .upload = !buffer->second.luisa_uploaded});
    }

    Feather::Luisa::DispatchInputs context_dispatch{};
    std::string context_error;
    if (!configure_luisa_dispatch_locked(kernel.context, &context_dispatch, &context_error)) {
        return fail(FE_ERROR_INVALID_HANDLE, context_error);
    }
    const auto& backend_name = context_dispatch.backend_name;
    if (backend_name != "vk" && backend_name != "metal" &&
        backend_name != "cuda" && backend_name != "hip") {
        return fail(FE_ERROR_INVALID_ARGUMENT,
                    "FEATHER_LUISA_BACKEND must be one of: vk, metal, cuda, hip.");
    }
#if !FEATHER_HAS_LUISA_METAL
    if (backend_name == "metal") {
        return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                    "Luisa Metal backend was not built for this Feather runtime.");
    }
#endif
#if !FEATHER_HAS_LUISA_CUDA
    if (backend_name == "cuda") {
        return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                    "Luisa CUDA backend was not built; configure FEATHER_LUISA_ENABLE_CUDA with CUDA Toolkit 12.1+.");
    }
#endif
#if !FEATHER_HAS_LUISA_HIP
    if (backend_name == "hip") {
        return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                    "Luisa HIP backend was not built; configure FEATHER_LUISA_ENABLE_HIP with ROCm/HIP and hiprtc.");
    }
#endif
    uint64_t shader_cache_key = 1469598103934665603ull;
    const auto mix_cache_key = [&](uint64_t value) {
        shader_cache_key ^= value;
        shader_cache_key *= 1099511628211ull;
    };
    mix_cache_key(kernel_handle);
    mix_cache_key(kernel.auto_diff ? 1u : 0u);
    // AD gradient buffers are sized from the logical dispatch extent. Include
    // it in the identity so a resized dispatch gets fresh device storage.
    mix_cache_key(logical_x);
    mix_cache_key(logical_y);
    mix_cache_key(logical_z);
    if (kernel.auto_diff) {
        for (const auto& push : lowering.push_constants) {
            mix_cache_key(push.binding);
            mix_cache_key(push.size);
            const auto* bytes = static_cast<const unsigned char*>(push.data);
            for (size_t i = 0u; bytes != nullptr && i < push.size; ++i)
                mix_cache_key(bytes[i]);
        }
    }
    for (const auto& resource : lowering.resources) {
        mix_cache_key(resource.kind);
        mix_cache_key(resource.binding);
        if (resource.kind == kIrResourceKindBuffer) {
            const auto bound = kernel.buffers.find(resource.binding);
            if (bound != kernel.buffers.end()) {
                mix_cache_key(bound->second);
                const auto buffer = g_buffers.find(bound->second);
                if (buffer != g_buffers.end()) {
                    mix_cache_key(buffer->second.bytes.size());
                    mix_cache_key(buffer->second.stride);
                }
            }
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto bound = kernel.textures.find(resource.binding);
            if (bound != kernel.textures.end()) {
                mix_cache_key(bound->second);
                const auto texture = g_textures.find(bound->second);
                if (texture != g_textures.end()) {
                    mix_cache_key(texture->second.width);
                    mix_cache_key(texture->second.height);
                    mix_cache_key(texture->second.depth);
                    mix_cache_key(texture->second.mip_levels);
                    mix_cache_key(texture->second.pixel_format);
                }
            }
        }
    }
    Feather::Luisa::DispatchInputs dispatch{.group_x = group_x,
                                            .group_y = group_y,
                                            .group_z = group_z,
                                            .logical_x = logical_x,
                                            .logical_y = logical_y,
                                            .logical_z = logical_z,
                                            .shader_cache_key = shader_cache_key,
                                            .backend_name = backend_name,
                                            .runtime_directory = context_dispatch.runtime_directory,
                                            .synchronize = wait,
                                            .context_key = context_dispatch.context_key,
                                            .device_index = context_dispatch.device_index,
                                            .stream_key = stream_key,
                                            .fence_key = fence_key,
                                            .retain_fence = stream_key != 0u};
    std::vector<Feather::Luisa::AdGradientBinding> gradient_bindings;
    gradient_bindings.reserve(next_gradients.size());
    for (auto& gradient : next_gradients) {
        gradient_bindings.push_back({gradient.source_binding, gradient.element_count,
                                     gradient.component_count, &gradient.host_bytes});
    }
    std::string error;
    if (!Feather::Luisa::Dispatch(ir.typed_module, lowering, bindings, texture_bindings, dispatch,
                                  ad_inputs ? &*ad_inputs : nullptr, gradient_bindings,
                                  accel_bindings, &error)) {
        return fail(FE_ERROR_UNSUPPORTED, error.empty() ? "Luisa dispatch failed." : std::move(error));
    }

    if (ad_inputs) {
        release_ad_gradient_buffers(kernel);
        for (auto& gradient : next_gradients) gradient.byte_size = gradient.host_bytes.size();
        kernel.ad_gradients = std::move(next_gradients);
    }

    for (const auto& [binding, handle] : kernel.buffers) {
        (void)binding;
        if (auto buffer = g_buffers.find(handle); buffer != g_buffers.end()) {
            buffer->second.luisa_uploaded = true;
        }
    }
    for (const auto& [binding, handle] : kernel.textures) {
        (void)binding;
        if (auto texture = g_textures.find(handle); texture != g_textures.end()) {
            texture->second.luisa_uploaded = true;
        }
    }

    for (const auto& resource : lowering.resources) {
        if (resource.kind != kIrResourceKindBuffer) continue;
        if (resource.access == 1) continue;
        const auto bound = kernel.buffers.find(resource.binding);
        auto buffer = g_buffers.find(bound->second);
        buffer->second.host_dirty = true;
        buffer->second.luisa_uploaded = true;
        ++buffer->second.content_revision;
    }
    for (const auto& resource : lowering.resources) {
        if (resource.kind != kIrResourceKindTexture2D && resource.kind != kIrResourceKindTexture3D) continue;
        if (resource.access == 1 || resource.access == 4) continue;
        const auto bound = kernel.textures.find(resource.binding);
        auto texture = g_textures.find(bound->second);
        texture->second.host_dirty = true;
        texture->second.luisa_uploaded = true;
    }
    return ok();
#endif
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

bool validate_sampler_desc(const FeSamplerDesc& source) {
    if (source.min_filter > 1 || source.mag_filter > 1 ||
        source.mipmap_mode > 1 ||
        source.address_u > 3 || source.address_v > 3 || source.address_w > 3 ||
        source.compare_op > 7 || source.border_color > 5 ||
        !std::isfinite(source.mip_lod_bias) || !std::isfinite(source.min_lod) ||
        !std::isfinite(source.max_lod) || !std::isfinite(source.max_anisotropy) ||
        source.min_lod < 0.0f || source.max_lod < source.min_lod ||
        source.max_anisotropy < 1.0f) {
        return false;
    }
    return true;
}

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

bool same_graphics_resource(const GraphicsPushConstantLayoutEntry& entry, uint32_t binding, const std::string& name) {
    return entry.binding == binding && entry.name == name;
}

const IrResource* find_graphics_resource_by_binding_and_name(const ParsedIr& ir, uint32_t binding,
                                                             const std::string& name) {
    for (const auto& resource : ir.resources) {
        if (resource.binding == binding && string_or_empty(ir, resource.name_string_id) == name) {
            return &resource;
        }
    }
    return nullptr;
}

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

FeBufferHandle infer_graphics_vertex_buffer(const GraphicsPipelineState& pipeline) {
    if (pipeline.vertex_buffer != 0) {
        return pipeline.vertex_buffer;
    }

    FeBufferHandle lowest = 0;
    uint32_t lowest_binding = 0;
    for (const auto& [binding, buffer] : pipeline.buffers) {
        if (lowest == 0 || binding < lowest_binding) {
            lowest = buffer;
            lowest_binding = binding;
        }
    }
    return lowest;
}

#if FEATHER_HAS_LUISA
std::string luisa_graphics_type_name(const Feather::TypedIR::Module& module, uint32_t type_id) {
    if (type_id >= module.types.size()) return {};
    const auto& type = module.types[type_id];
    if (type.kind == 1u) {
        if (type.a == 0u) return "bool";
        if (type.a == 1u) return "int";
        if (type.a == 2u) return "uint";
        if (type.a == 3u) return "float";
        return {};
    }
    if (type.kind == 2u && type.b >= 2u && type.b <= 4u) {
        auto element = luisa_graphics_type_name(module, type.a);
        return element.empty() ? std::string{} : element + std::to_string(type.b);
    }
    if (type.kind == 4u && type.a < module.structs.size()) {
        const auto& structure = module.structs[type.a];
        const auto* qualified = typed_ir_string(module, structure.fully_qualified_name_id);
        const auto* simple = typed_ir_string(module, structure.name_id);
        return qualified != nullptr && !qualified->empty() ? *qualified
             : simple != nullptr ? *simple : std::string{};
    }
    return {};
}

bool luisa_graphics_interpolatable_type(const Feather::TypedIR::Module& module, uint32_t type_id) {
    if (type_id >= module.types.size()) return false;
    const auto& type = module.types[type_id];
    if (type.kind == 1u) return type.a == 3u && type.b == 32u;
    if (type.kind == 2u) return type.b >= 2u && type.b <= 4u &&
                                luisa_graphics_interpolatable_type(module, type.a);
    if (type.kind == 3u) return type.b >= 2u && type.b <= 4u;
    if (type.kind != 4u || type.a >= module.structs.size()) return false;
    const auto& structure = module.structs[type.a];
    if (structure.first_field == UINT32_MAX || structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) return false;
    for (uint32_t i = 0u; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        if (field.offset % sizeof(float) != 0u || field.size_in_bytes % sizeof(float) != 0u ||
            !luisa_graphics_interpolatable_type(module, field.type_id)) return false;
    }
    return true;
}

bool luisa_graphics_varying_layout(const Feather::TypedIR::Module& module, uint32_t type_id,
                                   uint32_t* stride, std::string* type_name) {
    if (stride == nullptr || type_name == nullptr || type_id >= module.types.size()) return false;
    *type_name = luisa_graphics_type_name(module, type_id);
    const auto& type = module.types[type_id];
    if (type.kind == 2u && *type_name == "float4") {
        *stride = sizeof(float) * 4u;
        return true;
    }
    if (type.kind != 4u || type.a >= module.structs.size()) return false;
    const auto& structure = module.structs[type.a];
    if (structure.size_in_bytes == 0u || structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) return false;
    const auto position = std::find_if(
        module.struct_fields.begin() + structure.first_field,
        module.struct_fields.begin() + structure.first_field + structure.field_count,
        [](const auto& field) { return (field.flags & kTypedStructFieldFlagPosition) != 0u; });
    if (position == module.struct_fields.begin() + structure.first_field + structure.field_count ||
        position->offset != 0u || luisa_graphics_type_name(module, position->type_id) != "float4") return false;
    *stride = structure.size_in_bytes;
    return !type_name->empty() && structure.size_in_bytes % sizeof(float) == 0u &&
           luisa_graphics_interpolatable_type(module, type_id);
}

bool luisa_graphics_fragment_output_fields(const Feather::TypedIR::Module& module, uint32_t type_id,
                                           uint32_t color_target_count,
                                           std::vector<uint32_t>* return_fields) {
    if (return_fields == nullptr || color_target_count == 0u || type_id >= module.types.size()) return false;
    return_fields->clear();
    if (luisa_graphics_type_name(module, type_id) == "float4") {
        if (color_target_count != 1u) return false;
        return_fields->push_back(Feather::TypedIR::NoIndex);
        return true;
    }
    const auto& type = module.types[type_id];
    if (type.kind != 4u || type.a >= module.structs.size()) return false;
    const auto& structure = module.structs[type.a];
    if (structure.field_count != color_target_count || structure.first_field == UINT32_MAX ||
        structure.first_field > module.struct_fields.size() ||
        structure.field_count > module.struct_fields.size() - structure.first_field) return false;
    return_fields->assign(color_target_count, Feather::TypedIR::NoIndex);
    for (uint32_t i = 0u; i < structure.field_count; ++i) {
        const auto& field = module.struct_fields[structure.first_field + i];
        if ((field.flags & kTypedStructFieldFlagColor) == 0u ||
            (field.flags & kTypedStructFieldFlagPosition) != 0u ||
            luisa_graphics_type_name(module, field.type_id) != "float4") return false;
        const auto location = field.flags >> kTypedStructFieldColorIndexShift;
        if (location >= color_target_count || (*return_fields)[location] != Feather::TypedIR::NoIndex) return false;
        (*return_fields)[location] = i;
    }
    return std::none_of(return_fields->begin(), return_fields->end(),
                        [](uint32_t field) { return field == Feather::TypedIR::NoIndex; });
}

bool bind_luisa_graphics_push_constants(const ParsedIr& parsed, const GraphicsPipelineState& pipeline,
                                        Feather::TypedIR::LoweringInputs* lowering) {
    if (lowering == nullptr) return false;
    lowering->push_constant_storage.clear();
    lowering->push_constant_storage.reserve(lowering->push_constants.size());
    for (auto& push : lowering->push_constants) {
        const auto resource = std::find_if(parsed.resources.begin(), parsed.resources.end(), [&](const auto& candidate) {
            return candidate.kind == kIrResourceKindPushConstant && candidate.binding == push.binding;
        });
        if (resource == parsed.resources.end()) return false;
        const auto* name = get_string(parsed, resource->name_string_id);
        if (name == nullptr) return false;
        const auto layout = std::find_if(
            pipeline.push_constant_layout.begin(), pipeline.push_constant_layout.end(), [&](const auto& candidate) {
                return candidate.binding == push.binding && candidate.name == *name;
            });
        if (layout == pipeline.push_constant_layout.end() || layout->size != push.size ||
            layout->offset > pipeline.push_constants.size() ||
            push.size > pipeline.push_constants.size() - layout->offset) return false;
        const auto alignment = std::max<size_t>(push.alignment, 1u);
        const auto padded_size = align_offset(push.size, alignment);
        auto& storage = lowering->push_constant_storage.emplace_back(padded_size, 0u);
        std::memcpy(storage.data(), pipeline.push_constants.data() + layout->offset, push.size);
        push.data = storage.data();
        push.size = storage.size();
    }
    return true;
}

uint64_t luisa_graphics_shader_cache_key(
    const std::vector<unsigned char>& ir, const Feather::TypedIR::LoweringInputs& lowering,
    const std::vector<Feather::Luisa::HostBufferBinding>& buffers,
    const std::vector<Feather::Luisa::HostTextureBinding>& textures, uint64_t stage_tag) {
    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(stage_tag);
    for (auto byte : ir) mix(byte);
    mix(lowering.logical_x);
    mix(lowering.logical_y);
    mix(lowering.logical_z);
    mix(lowering.graphics_vertex_count);
    mix(lowering.graphics_first_instance);
    mix(lowering.graphics_sample_count);
    mix(lowering.graphics_sample_index);
    for (const auto& resource : lowering.resources) {
        mix(resource.binding);
        mix(resource.kind);
        mix(resource.access);
        mix(resource.sampler_min_filter);
        mix(resource.sampler_mag_filter);
        mix(resource.sampler_mipmap_mode);
        mix(resource.sampler_address_u);
        mix(resource.sampler_address_v);
    }
    for (const auto& target : lowering.graphics_color_targets) {
        mix(target.binding);
        mix(target.return_field);
        mix(target.blend.enable);
        mix(target.blend.src_color);
        mix(target.blend.dst_color);
        mix(target.blend.color_op);
        mix(target.blend.src_alpha);
        mix(target.blend.dst_alpha);
        mix(target.blend.alpha_op);
        mix(target.blend.write_mask);
    }
    for (const auto& push : lowering.push_constants) {
        mix(push.binding);
        mix(push.size);
        mix(push.alignment);
        if (!lowering.dynamic_push_constants) {
            const auto* bytes = static_cast<const unsigned char*>(push.data);
            for (size_t i = 0u; bytes != nullptr && i < push.size; ++i) mix(bytes[i]);
        }
    }
    for (const auto& buffer : buffers) {
        mix(buffer.binding);
        mix(buffer.access);
        mix(buffer.stride);
        mix(buffer.bytes == nullptr ? 0u : buffer.bytes->size());
        mix(buffer.resident_key);
    }
    for (const auto& texture : textures) {
        mix(texture.binding);
        mix(texture.access);
        mix(texture.width);
        mix(texture.height);
        mix(texture.depth);
        mix(texture.mip_levels);
        mix(texture.pixel_format);
        mix(texture.resident_key);
    }
    return key == 0u ? 1u : key;
}

uint64_t luisa_graphics_execution_cache_key(const Feather::TypedIR::LoweringInputs& lowering) {
    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    for (const auto& push : lowering.push_constants) {
        mix(push.binding);
        mix(push.size);
        const auto* bytes = static_cast<const unsigned char*>(push.data);
        for (size_t i = 0u; bytes != nullptr && i < push.size; ++i) mix(bytes[i]);
    }
    return key == 0u ? 1u : key;
}

FeResult dispatch_graphics_vertex_stage(const GraphicsPipelineState& pipeline, uint32_t vertex_count,
                                        uint32_t instance_count, uint32_t first_instance,
                                        std::vector<unsigned char>* output, uint32_t* output_stride,
                                        uint64_t* output_resident_key, bool* output_reused) {
    if (output == nullptr || output_stride == nullptr || output_resident_key == nullptr ||
        output_reused == nullptr) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster vertex stage output is missing.");
    }
    ParsedIr parsed;
    if (!parse_feather_ir(pipeline.vertex_ir, &parsed) || !parsed.has_section7 ||
        parsed.typed_module.entry_function >= parsed.typed_module.functions.size()) {
        return fail(FE_ERROR_UNSUPPORTED, "Compute raster vertex stage requires typed FEIR.");
    }
    const auto& entry = parsed.typed_module.functions[parsed.typed_module.entry_function];
    std::string output_type;
    if (entry.kind != 3u || !luisa_graphics_varying_layout(
                                parsed.typed_module, entry.return_type_id, output_stride, &output_type)) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Compute raster vertex stage requires float4 or a position-first varying struct.");
    }

    KernelState adapter;
    adapter.ir = pipeline.vertex_ir;
    adapter.push_constants = pipeline.push_constants;
    adapter.buffers = pipeline.buffers;
    adapter.textures = pipeline.textures;
    adapter.samplers = pipeline.samplers;
    const auto invocation_count = static_cast<uint64_t>(vertex_count) * instance_count;
    if (invocation_count == 0u || invocation_count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster vertex invocation count is out of range.");
    }
    adapter.logical_x = static_cast<int32_t>(invocation_count);
    adapter.logical_y = 1;
    adapter.logical_z = 1;
    Feather::TypedIR::LoweringInputs lowering;
    if (!build_typed_ir_lowering_inputs(parsed, adapter, &lowering, true) ||
        !bind_luisa_graphics_push_constants(parsed, pipeline, &lowering)) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster could not bind vertex-stage FEIR resources.");
    }
    lowering.dynamic_push_constants = true;
    uint32_t output_binding = 0u;
    for (const auto& resource : lowering.resources) {
        if (resource.binding == UINT32_MAX) {
            return fail(FE_ERROR_UNSUPPORTED, "Compute raster cannot allocate a vertex-stage output binding.");
        }
        output_binding = std::max(output_binding, resource.binding + 1u);
    }
    lowering.stage_output_binding = output_binding;
    lowering.bounds_check = true;
    lowering.logical_x = static_cast<int32_t>(invocation_count);
    lowering.logical_y = 1;
    lowering.logical_z = 1;
    lowering.group_x = 1;
    lowering.group_y = 1;
    lowering.group_z = 1;
    lowering.graphics_vertex_count = vertex_count;
    lowering.graphics_first_instance = first_instance;
    lowering.resources.push_back(Feather::TypedIR::ResourceInfo{
        output_binding, kIrResourceKindBuffer, 2u, "__feather_vertex_output", output_type});

    std::vector<Feather::Luisa::HostBufferBinding> buffers;
    std::vector<Feather::Luisa::HostTextureBinding> textures;
    buffers.reserve(lowering.resources.size());
    for (const auto& resource : lowering.resources) {
        if (resource.binding == output_binding) continue;
        if (resource.kind == kIrResourceKindPushConstant || resource.kind == kIrResourceKindSampler) continue;
        if (resource.kind == kIrResourceKindBuffer) {
            const auto bound = pipeline.buffers.find(resource.binding);
            const auto native = bound == pipeline.buffers.end() ? g_buffers.end() : g_buffers.find(bound->second);
            if (native == g_buffers.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Compute raster vertex stage is missing a buffer.");
            }
            buffers.push_back({resource.binding, resource.access, native->second.stride, &native->second.bytes,
                               bound->second, !native->second.luisa_uploaded, false});
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto bound = pipeline.textures.find(resource.binding);
            const auto native = bound == pipeline.textures.end() ? g_textures.end() : g_textures.find(bound->second);
            if (native == g_textures.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Compute raster vertex stage is missing a texture.");
            }
            textures.push_back({resource.binding, resource.kind, resource.access,
                                native->second.width, native->second.height, native->second.depth,
                                native->second.mip_levels, native->second.pixel_format, &native->second.bytes,
                                bound->second, !native->second.luisa_uploaded, false,
                                native->second.mipmaps_requested});
        }
    }
    output->assign(static_cast<size_t>(invocation_count) * *output_stride, 0u);
    buffers.push_back({output_binding, 2u, *output_stride, output});

    const auto base_cache_key = luisa_graphics_shader_cache_key(
        pipeline.vertex_ir, lowering, buffers, textures, 0x766572746578ull);
    *output_resident_key = base_cache_key ^ 0x7665727465786f75ull;
    if (*output_resident_key == 0u) *output_resident_key = 1u;
    buffers.back().resident_key = *output_resident_key;
    buffers.back().upload = false;
    buffers.back().download = false;
    const auto shader_cache_key = luisa_graphics_shader_cache_key(
        pipeline.vertex_ir, lowering, buffers, textures, 0x766572746578ull);
    Feather::Luisa::DispatchInputs dispatch{
        1u, 1u, 1u, static_cast<uint32_t>(invocation_count), 1u, 1u};
    std::string context_error;
    if (!configure_luisa_dispatch_locked(pipeline.context, &dispatch, &context_error)) {
        return fail(FE_ERROR_INVALID_HANDLE, context_error);
    }
    dispatch.shader_cache_key = shader_cache_key;
    dispatch.execution_cache_key = luisa_graphics_execution_cache_key(lowering);
    dispatch.reuse_if_inputs_clean = true;
    dispatch.execution_skipped = output_reused;
    const auto* profile_stages = std::getenv("FEATHER_RASTER_PROFILE_STAGES");
    dispatch.synchronize = profile_stages != nullptr && profile_stages[0] != '\0' &&
                           std::strcmp(profile_stages, "0") != 0;
    if (!dispatch.synchronize) dispatch.fence_key = next_handle();
    std::string error;
    if (!Feather::Luisa::Dispatch(parsed.typed_module, lowering, buffers, textures, dispatch, nullptr, {}, {}, &error)) {
        return fail(FE_ERROR_UNSUPPORTED, error.empty() ? "Compute raster vertex FEIR dispatch failed." : error);
    }
    for (const auto& binding : textures) {
        if (binding.resident_key == 0u) continue;
        if (const auto texture = g_textures.find(binding.resident_key); texture != g_textures.end()) {
            texture->second.luisa_uploaded = true;
        }
    }
    for (const auto& binding : buffers) {
        if (binding.resident_key == 0u || !binding.upload) continue;
        if (const auto buffer = g_buffers.find(binding.resident_key); buffer != g_buffers.end()) {
            buffer->second.luisa_uploaded = true;
        }
    }
    return ok();
}

FeResult dispatch_graphics_fragment_stage(const GraphicsPipelineState& pipeline,
                                          const std::vector<TextureState*>& targets,
                                          std::vector<unsigned char>* varyings,
                                          std::vector<unsigned char>* coverage,
                                          uint64_t varying_resident_key, uint64_t coverage_resident_key,
                                          const FeTextureHandle* target_handles, uint32_t sample_count,
                                          const std::array<float, 4u>& clear_color,
                                          bool preserve_color,
                                          std::vector<uint64_t>* fused_callable_keys = nullptr) {
    const auto prepare_fused = fused_callable_keys != nullptr;
    if (targets.empty() || (!prepare_fused && (varyings == nullptr || coverage == nullptr))) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster fragment inputs are missing.");
    }
    ParsedIr parsed;
    if (!parse_feather_ir(pipeline.fragment_ir, &parsed) || !parsed.has_section7 ||
        parsed.typed_module.entry_function >= parsed.typed_module.functions.size()) {
        return fail(FE_ERROR_UNSUPPORTED, "Compute raster fragment stage requires typed FEIR.");
    }
    const auto& entry = parsed.typed_module.functions[parsed.typed_module.entry_function];
    uint32_t varying_stride = 0u;
    std::string varying_type;
    std::vector<uint32_t> return_fields;
    if (entry.kind != 4u || entry.parameter_count != 1u || entry.first_parameter == UINT32_MAX ||
        entry.first_parameter >= parsed.typed_module.parameters.size() ||
        !luisa_graphics_fragment_output_fields(parsed.typed_module, entry.return_type_id,
                                               static_cast<uint32_t>(targets.size()), &return_fields) ||
        !luisa_graphics_varying_layout(parsed.typed_module,
                                       parsed.typed_module.parameters[entry.first_parameter].type_id,
                                       &varying_stride, &varying_type)) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Compute raster fragment stage requires float varyings and dense float4 color outputs.");
    }
    if (prepare_fused && (targets.size() != 1u || return_fields[0] != UINT32_MAX)) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "Fused compute raster currently requires one float4 color output.");
    }
    const auto& first_target = *targets.front();
    const auto pixel_count = static_cast<size_t>(first_target.width) * first_target.height;
    if ((sample_count != 1u && sample_count != 4u) ||
        (!prepare_fused &&
         (varyings->size() != pixel_count * sample_count * varying_stride ||
          coverage->size() != pixel_count * sample_count * sizeof(float)))) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster fragment buffers have inconsistent dimensions.");
    }
    KernelState adapter;
    adapter.ir = pipeline.fragment_ir;
    adapter.push_constants = pipeline.push_constants;
    adapter.buffers = pipeline.buffers;
    adapter.textures = pipeline.textures;
    adapter.samplers = pipeline.samplers;
    adapter.logical_x = static_cast<int32_t>(first_target.width);
    adapter.logical_y = static_cast<int32_t>(first_target.height);
    adapter.logical_z = 1;
    Feather::TypedIR::LoweringInputs lowering;
    if (!build_typed_ir_lowering_inputs(parsed, adapter, &lowering) ||
        !bind_luisa_graphics_push_constants(parsed, pipeline, &lowering)) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster could not bind fragment-stage FEIR resources.");
    }
    uint32_t synthetic_binding = 0u;
    for (const auto& resource : lowering.resources) {
        if (resource.binding == UINT32_MAX) {
            return fail(FE_ERROR_UNSUPPORTED, "Compute raster cannot allocate fragment-stage bindings.");
        }
        synthetic_binding = std::max(synthetic_binding, resource.binding + 1u);
    }
    auto input_binding = UINT32_MAX;
    auto coverage_binding = UINT32_MAX;
    if (!prepare_fused) {
        input_binding = synthetic_binding++;
        coverage_binding = synthetic_binding++;
        lowering.stage_input_binding = input_binding;
        lowering.stage_coverage_binding = coverage_binding;
    }
    lowering.bounds_check = true;
    lowering.logical_x = static_cast<int32_t>(first_target.width);
    lowering.logical_y = static_cast<int32_t>(first_target.height);
    lowering.logical_z = 1;
    lowering.group_x = 1;
    lowering.group_y = 1;
    lowering.group_z = 1;
    if (!prepare_fused) {
        lowering.resources.push_back(Feather::TypedIR::ResourceInfo{
            input_binding, kIrResourceKindBuffer, 1u, "__feather_fragment_input", varying_type});
        lowering.resources.push_back(Feather::TypedIR::ResourceInfo{
            coverage_binding, kIrResourceKindBuffer, 1u, "__feather_fragment_coverage", "float"});
    }
    std::vector<uint32_t> output_bindings;
    output_bindings.reserve(targets.size());
    if (!prepare_fused) for (uint32_t i = 0u; i < targets.size(); ++i) {
        const auto output_binding = synthetic_binding++;
        output_bindings.push_back(output_binding);
        Feather::TypedIR::ResourceInfo output_resource;
        output_resource.binding = output_binding;
        output_resource.kind = kIrResourceKindTexture2D;
        output_resource.access = 3u;
        output_resource.name = "__feather_fragment_output_" + std::to_string(i);
        output_resource.element_type = "float4";
        output_resource.width = targets[i]->width;
        output_resource.height = targets[i]->height;
        if (!luisa_pixel_format(targets[i]->pixel_format)) {
            return fail(FE_ERROR_UNSUPPORTED, "Compute raster fragment target format is unsupported.");
        }
        output_resource.texture_format = targets[i]->pixel_format;
        lowering.resources.push_back(std::move(output_resource));
        const auto& blend = pipeline.color_blend_attachments[i];
        lowering.graphics_color_targets.push_back(Feather::TypedIR::GraphicsColorTargetInfo{
            output_binding, return_fields[i], Feather::TypedIR::GraphicsBlendInfo{
                blend.blend_enable != 0u, blend.src_color, blend.dst_color, blend.color_op,
                blend.src_alpha, blend.dst_alpha, blend.alpha_op, blend.write_mask}});
    }
    if (!prepare_fused) lowering.stage_output_binding = output_bindings.front();

    std::vector<Feather::Luisa::HostBufferBinding> buffers;
    std::vector<Feather::Luisa::HostTextureBinding> textures;
    for (const auto& resource : lowering.resources) {
        if (resource.binding == input_binding || resource.binding == coverage_binding ||
            std::find(output_bindings.begin(), output_bindings.end(), resource.binding) != output_bindings.end() ||
            resource.kind == kIrResourceKindPushConstant ||
            resource.kind == kIrResourceKindSampler) continue;
        if (resource.kind == kIrResourceKindBuffer) {
            const auto bound = pipeline.buffers.find(resource.binding);
            const auto native = bound == pipeline.buffers.end() ? g_buffers.end() : g_buffers.find(bound->second);
            if (native == g_buffers.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Compute raster fragment stage is missing a buffer.");
            }
            buffers.push_back({resource.binding, resource.access, native->second.stride, &native->second.bytes,
                               bound->second, !native->second.luisa_uploaded, false});
        } else if (resource.kind == kIrResourceKindTexture2D || resource.kind == kIrResourceKindTexture3D) {
            const auto bound = pipeline.textures.find(resource.binding);
            const auto native = bound == pipeline.textures.end() ? g_textures.end() : g_textures.find(bound->second);
            if (native == g_textures.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Compute raster fragment stage is missing a texture.");
            }
            textures.push_back({resource.binding, resource.kind, resource.access,
                                native->second.width, native->second.height, native->second.depth,
                                native->second.mip_levels, native->second.pixel_format, &native->second.bytes,
                                bound->second, !native->second.luisa_uploaded, false,
                                native->second.mipmaps_requested});
        }
    }
    if (!prepare_fused) {
        buffers.push_back({input_binding, 1u, varying_stride, varyings,
                           varying_resident_key, false, false});
        buffers.push_back({coverage_binding, 1u, sizeof(float), coverage,
                           coverage_resident_key, false, false});
    }
    const auto output_texture_offset = textures.size();
    const auto sample_texture_key = [](uint64_t target, uint32_t sample) {
        auto key = target ^ (0x6d73616173616d70ull +
                             static_cast<uint64_t>(sample + 1u) * 0x9e3779b97f4a7c15ull);
        return key == 0u ? static_cast<uint64_t>(sample + 1u) : key;
    };
    if (!prepare_fused) for (uint32_t i = 0u; i < targets.size(); ++i) {
        textures.push_back({output_bindings[i], kIrResourceKindTexture2D, 3u,
                            targets[i]->width, targets[i]->height, targets[i]->depth,
                            targets[i]->mip_levels, targets[i]->pixel_format, &targets[i]->bytes,
                            sample_count == 1u ? target_handles[i]
                                               : sample_texture_key(target_handles[i], 0u),
                            sample_count == 1u ? !targets[i]->luisa_uploaded : false,
                            false, false});
    }

    Feather::Luisa::DispatchInputs context_dispatch{};
    std::string context_error;
    if (!configure_luisa_dispatch_locked(pipeline.context, &context_dispatch, &context_error)) {
        return fail(FE_ERROR_INVALID_HANDLE, context_error);
    }
    if (prepare_fused) fused_callable_keys->clear();
    if (sample_count > 1u && !prepare_fused) {
        for (uint32_t target_index = 0u; target_index < targets.size(); ++target_index) {
            std::array<uint64_t, 4u> sample_keys{};
            for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                sample_keys[sample] = sample_texture_key(target_handles[target_index], sample);
            }
            Feather::Luisa::HostTextureBinding target_binding{
                output_bindings[target_index], kIrResourceKindTexture2D, 3u,
                targets[target_index]->width, targets[target_index]->height,
                targets[target_index]->depth, targets[target_index]->mip_levels,
                targets[target_index]->pixel_format, &targets[target_index]->bytes,
                target_handles[target_index], !targets[target_index]->luisa_uploaded, false, false};
            std::string error;
            const auto initialized = preserve_color
                ? Feather::Luisa::LoadMultisampleTexture(
                      pipeline.context, sample_keys, target_binding, &error)
                : Feather::Luisa::ClearMultisampleTexture(
                      pipeline.context, sample_keys, target_binding, clear_color, &error);
            if (!initialized) {
                return fail(FE_ERROR_UNSUPPORTED,
                            error.empty() ? "Compute raster multisample initialization failed." : error);
            }
        }
    }
    const auto fragment_variant_count = prepare_fused ? 1u : sample_count;
    for (uint32_t sample = 0u; sample < fragment_variant_count; ++sample) {
        if (!prepare_fused && sample_count > 1u) {
            for (uint32_t i = 0u; i < targets.size(); ++i) {
                auto& output = textures[output_texture_offset + i];
                output.resident_key = sample_texture_key(target_handles[i], sample);
                output.upload = false;
            }
        }
        lowering.graphics_sample_count = sample_count;
        lowering.graphics_sample_index = sample;
        const auto shader_cache_key = luisa_graphics_shader_cache_key(
            pipeline.fragment_ir, lowering, buffers, textures,
            prepare_fused ? 0x6675736564667261ull : 0x667261676d656e74ull);
        Feather::Luisa::DispatchInputs dispatch{
            1u, 1u, 1u, first_target.width, first_target.height, 1u};
        dispatch.backend_name = context_dispatch.backend_name;
        dispatch.runtime_directory = context_dispatch.runtime_directory;
        dispatch.context_key = context_dispatch.context_key;
        dispatch.device_index = context_dispatch.device_index;
        dispatch.shader_cache_key = shader_cache_key;
        dispatch.synchronize = sample_count == 1u;
        if (!dispatch.synchronize && !prepare_fused) dispatch.fence_key = next_handle();
        std::string error;
        const auto dispatched = prepare_fused
            ? Feather::Luisa::PrepareGraphicsFragment(
                  parsed.typed_module, lowering, buffers, textures,
                  dispatch, shader_cache_key, &error)
            : Feather::Luisa::Dispatch(
                  parsed.typed_module, lowering, buffers, textures,
                  dispatch, nullptr, {}, {}, &error);
        if (!dispatched) {
            return fail(FE_ERROR_UNSUPPORTED,
                        error.empty()
                            ? (prepare_fused
                                   ? "Compute raster fused fragment preparation failed."
                                   : "Compute raster fragment FEIR dispatch failed.")
                            : error);
        }
        if (prepare_fused) fused_callable_keys->push_back(shader_cache_key);
        for (size_t i = 0u; i < output_texture_offset; ++i) textures[i].upload = false;
    }
    for (const auto& binding : pipeline.textures) {
        if (const auto texture = g_textures.find(binding.second); texture != g_textures.end()) {
            texture->second.luisa_uploaded = true;
        }
    }
    for (const auto& binding : pipeline.buffers) {
        if (const auto buffer = g_buffers.find(binding.second); buffer != g_buffers.end()) {
            buffer->second.luisa_uploaded = true;
        }
    }
    if (prepare_fused) return ok();
    if (sample_count > 1u) {
        for (uint32_t target_index = 0u; target_index < targets.size(); ++target_index) {
            std::array<uint64_t, 4u> sample_keys{};
            for (uint32_t sample = 0u; sample < sample_count; ++sample) {
                sample_keys[sample] = sample_texture_key(target_handles[target_index], sample);
            }
            Feather::Luisa::HostTextureBinding target_binding{
                output_bindings[target_index], kIrResourceKindTexture2D, 3u,
                targets[target_index]->width, targets[target_index]->height,
                targets[target_index]->depth, targets[target_index]->mip_levels,
                targets[target_index]->pixel_format, &targets[target_index]->bytes,
                target_handles[target_index], false, false, false};
            std::string error;
            if (!Feather::Luisa::ResolveMultisampleTexture(
                    pipeline.context, sample_keys, target_binding,
                    target_index + 1u == targets.size(), &error)) {
                return fail(FE_ERROR_UNSUPPORTED,
                            error.empty() ? "Compute raster multisample resolve failed." : error);
            }
        }
    }
    for (auto* target : targets) target->luisa_uploaded = true;
    return ok();
}

FeResult finish_fused_graphics_fragment_stage(
    const std::vector<TextureState*>& targets,
    const FeTextureHandle* target_handles,
    uint32_t sample_count) {
    static_cast<void>(target_handles);
    static_cast<void>(sample_count);
    for (auto* target : targets) target->luisa_uploaded = true;
    return ok();
}

bool clear_compute_raster_color(TextureState& target, float red, float green, float blue, float alpha) {
    const auto pixel_count = static_cast<size_t>(target.width) * target.height;
    if (target.pixel_format == 10u) {
        const std::array<float, 4u> color{red, green, blue, alpha};
        for (size_t i = 0u; i < pixel_count; ++i) {
            std::memcpy(target.bytes.data() + i * sizeof(color), color.data(), sizeof(color));
        }
        target.luisa_dirty = false;
        target.luisa_uploaded = false;
        return true;
    }
    if (target.pixel_format != 3u && target.pixel_format != 4u) return false;
    const auto pack = [](float value) {
        return static_cast<unsigned char>(std::clamp(value, 0.0f, 1.0f) * 255.0f + 0.5f);
    };
    const std::array<unsigned char, 4u> rgba{pack(red), pack(green), pack(blue), pack(alpha)};
    const std::array<unsigned char, 4u> bgra{rgba[2], rgba[1], rgba[0], rgba[3]};
    const auto& color = target.pixel_format == 4u ? bgra : rgba;
    for (size_t i = 0u; i < pixel_count; ++i) {
        std::memcpy(target.bytes.data() + i * color.size(), color.data(), color.size());
    }
    target.luisa_dirty = false;
    target.luisa_uploaded = false;
    return true;
}

std::pair<uint64_t, uint64_t> compute_raster_scratch_keys(
    const GraphicsPipelineState& pipeline, const FeGraphicsDrawDesc& draw,
    uint32_t varying_stride, uint32_t width, uint32_t height) {
    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(0x726173746572ull);
    for (auto byte : pipeline.fragment_ir) mix(byte);
    mix(varying_stride);
    mix(width);
    mix(height);
    mix(draw.count);
    mix(draw.depth_target);
    mix(pipeline.sample_count);
    for (uint32_t i = 0u; i < draw.color_target_count; ++i) mix(draw.color_targets[i]);
    const auto varying_key = key == 0u ? 1u : key;
    mix(0x636f766572616765ull);
    return {varying_key, key == 0u ? 2u : key};
}

uint64_t compute_raster_geometry_cache_key(
    uint64_t vertex_resident_key, FeBufferHandle index_buffer, uint64_t index_revision,
    const Feather::Luisa::RasterDispatchInputs& raster, uint32_t target_width,
    uint32_t target_height, const FeGraphicsDrawDesc& draw) {
    uint64_t key = 1469598103934665603ull;
    const auto mix = [&](uint64_t value) {
        key ^= value;
        key *= 1099511628211ull;
    };
    mix(0x67656f6d65747279ull);
    mix(vertex_resident_key);
    mix(index_buffer);
    mix(index_revision);
    mix(draw.indexed);
    mix(draw.first_vertex);
    mix(draw.first_index);
    mix(static_cast<uint32_t>(draw.vertex_offset));
    mix(draw.first_instance);
    mix(draw.instance_count);
    mix(raster.vertex_count);
    mix(raster.vertices_per_instance);
    mix(raster.vertex_domain);
    mix(raster.viewport_x);
    mix(raster.viewport_y);
    mix(raster.viewport_width);
    mix(raster.viewport_height);
    mix(raster.scissor_x);
    mix(raster.scissor_y);
    mix(raster.scissor_width);
    mix(raster.scissor_height);
    mix(raster.cull_mode);
    mix(raster.front_face);
    mix(raster.depth_clamp);
    mix(target_width);
    mix(target_height);
    return key == 0u ? 1u : key;
}

FeResult draw_graphics_pipeline_compute_raster(GraphicsPipelineState& pipeline, const FeGraphicsDrawDesc& draw) {
    trace_graphics_step("compute draw begin");
    if (pipeline.topology != 0u || draw.color_target_count != pipeline.color_attachment_count ||
        (pipeline.sample_count != 1u && pipeline.sample_count != 4u)) {
        return fail(FE_ERROR_UNSUPPORTED,
                    "compute raster supports matching color targets and 1x/4x triangle lists");
    }
    if (draw.count < 3u || draw.count % 3u != 0u) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster triangle lists require a multiple of three vertices.");
    }
    ParsedIr vertex_ir;
    ParsedIr fragment_ir;
    if (!parse_feather_ir(pipeline.vertex_ir, &vertex_ir) || !parse_feather_ir(pipeline.fragment_ir, &fragment_ir) ||
        !build_graphics_push_constant_layout(vertex_ir, fragment_ir, &pipeline.push_constant_layout)) {
        return fail(FE_ERROR_UNSUPPORTED, "Compute raster could not build the graphics push-constant layout.");
    }
    const auto instance_count = draw.instance_count == 0u ? 1u : draw.instance_count;
    const auto vertex_handle = infer_graphics_vertex_buffer(pipeline);
    const auto vertex_it = g_buffers.find(vertex_handle);
    std::vector<TextureState*> color_targets;
    color_targets.reserve(draw.color_target_count);
    for (uint32_t i = 0u; i < draw.color_target_count; ++i) {
        const auto target_it = g_textures.find(draw.color_targets[i]);
        if (target_it == g_textures.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Compute raster draw requires valid color resources.");
        }
        color_targets.push_back(&target_it->second);
    }
    if (vertex_it == g_buffers.end() || color_targets.empty()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Compute raster draw requires valid vertex and color resources.");
    }
    auto& target = *color_targets.front();
    if (std::any_of(color_targets.begin() + 1u, color_targets.end(), [&](const TextureState* candidate) {
            return candidate->width != target.width || candidate->height != target.height ||
                   candidate->depth != target.depth;
        })) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster color targets must have matching dimensions.");
    }
    if ((draw.viewport_enabled != 0u && (draw.viewport_width == 0u || draw.viewport_height == 0u)) ||
        (draw.scissor_enabled != 0u && (draw.scissor_width == 0u || draw.scissor_height == 0u))) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster viewport and scissor must have positive dimensions.");
    }
    if (draw.color_load_op > static_cast<uint32_t>(GraphicsColorLoadOp::DontCare) ||
        draw.depth_load_op > static_cast<uint32_t>(GraphicsDepthLoadOp::Clear)) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster attachment load op is invalid.");
    }
    if (draw.color_load_op == static_cast<uint32_t>(GraphicsColorLoadOp::Load) && draw.clear_color != 0u) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "GraphicsDrawDesc cannot specify ClearColor when ColorLoadOp is Load.");
    }
    if (draw.depth_load_op == static_cast<uint32_t>(GraphicsDepthLoadOp::Load) && draw.clear_depth != 0u) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "GraphicsDrawDesc cannot specify ClearDepth when DepthLoadOp is Load.");
    }
    const auto stride = pipeline.vertex_stride != 0u
                            ? pipeline.vertex_stride
                            : (vertex_it->second.stride != 0u ? vertex_it->second.stride : sizeof(float) * 4u);
    if (stride == 0u || vertex_it->second.bytes.size() % stride != 0u) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster vertex buffer has an invalid stride.");
    }
    const auto vertex_capacity = vertex_it->second.bytes.size() / stride;
    uint64_t raster_index_cache_key = 1469598103934665603ull;
    const auto mix_index_key = [&](uint64_t value) {
        raster_index_cache_key ^= value;
        raster_index_cache_key *= 1099511628211ull;
    };
    mix_index_key(draw.indexed);
    mix_index_key(draw.count);
    mix_index_key(draw.first_vertex);
    mix_index_key(draw.first_index);
    mix_index_key(static_cast<uint32_t>(draw.vertex_offset));
    mix_index_key(draw.index_buffer);
    mix_index_key(vertex_capacity);
    uint64_t index_revision = 0u;
    const BufferState* index_state = nullptr;
    uint32_t index_stride = 0u;
    if (draw.indexed != 0u) {
        const auto index_it = g_buffers.find(draw.index_buffer);
        if (draw.index_buffer == 0u || index_it == g_buffers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Compute raster indexed draw requires a valid index buffer.");
        }
        index_state = &index_it->second;
        index_stride = index_state->stride != 0u
                           ? index_state->stride
                           : static_cast<uint32_t>(sizeof(uint32_t));
        index_revision = index_it->second.content_revision;
        mix_index_key(index_revision);
        mix_index_key(index_stride);
        if (index_stride != sizeof(uint16_t) && index_stride != sizeof(uint32_t)) {
            return fail(FE_ERROR_UNSUPPORTED, "Compute raster indices must be ushort or uint.");
        }
        const auto required_indices = static_cast<uint64_t>(draw.first_index) + draw.count;
        if (required_indices > std::numeric_limits<size_t>::max() / index_stride ||
            index_it->second.bytes.size() < static_cast<size_t>(required_indices) * index_stride) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster index buffer is too small.");
        }
    } else {
        const auto required_vertices = static_cast<uint64_t>(draw.first_vertex) + draw.count;
        if (required_vertices > vertex_capacity) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster vertex buffer is too small.");
        }
    }
    if (raster_index_cache_key == 0u) raster_index_cache_key = 1u;
    if (pipeline.compute_raster_index_cache_key != raster_index_cache_key ||
        pipeline.compute_raster_indices.size() != draw.count) {
        pipeline.compute_raster_indices.resize(draw.count);
        pipeline.compute_raster_maximum_vertex = 0u;
        if (draw.indexed != 0u) {
            for (uint32_t i = 0u; i < draw.count; ++i) {
                const auto offset = static_cast<size_t>(draw.first_index + i) * index_stride;
                const auto index = index_stride == sizeof(uint16_t)
                                       ? static_cast<uint32_t>(
                                             read_u16_unaligned(index_state->bytes.data() + offset))
                                       : read_u32_unaligned(index_state->bytes.data() + offset);
                const auto shifted = static_cast<int64_t>(index) + draw.vertex_offset;
                if (shifted < 0 || static_cast<uint64_t>(shifted) >= vertex_capacity) {
                    return fail(FE_ERROR_INVALID_ARGUMENT,
                                "Compute raster index references a vertex out of range.");
                }
                pipeline.compute_raster_indices[i] = static_cast<uint32_t>(shifted);
                pipeline.compute_raster_maximum_vertex = std::max(
                    pipeline.compute_raster_maximum_vertex, pipeline.compute_raster_indices[i]);
            }
        } else {
            for (uint32_t i = 0u; i < draw.count; ++i) {
                pipeline.compute_raster_indices[i] = draw.first_vertex + i;
            }
            pipeline.compute_raster_maximum_vertex = pipeline.compute_raster_indices.back();
        }
        pipeline.compute_raster_index_cache_key = raster_index_cache_key;
    }
    const auto& raster_indices = pipeline.compute_raster_indices;
    const auto maximum_vertex = pipeline.compute_raster_maximum_vertex;
    Feather::Luisa::DispatchInputs dispatch{
        1u, 1u, 1u, target.width, target.height, 1u};
    std::string context_error;
    if (!configure_luisa_dispatch_locked(pipeline.context, &dispatch, &context_error)) {
        return fail(FE_ERROR_INVALID_HANDLE, context_error);
    }
    std::vector<unsigned char> all_transformed_vertices;
    uint32_t transformed_stride = 0u;
    uint64_t vertex_resident_key = 0u;
    bool vertex_reused = false;
    const auto vertex_domain = maximum_vertex + 1u;
    trace_graphics_step("compute setup complete");
    const auto vertex_result = dispatch_graphics_vertex_stage(
        pipeline, vertex_domain, instance_count, draw.first_instance,
        &all_transformed_vertices, &transformed_stride, &vertex_resident_key, &vertex_reused);
    if (vertex_result != FE_OK) return vertex_result;
    trace_graphics_step("compute vertex complete");
    const auto raster_vertex_count = static_cast<uint64_t>(draw.count) * instance_count;
    if (raster_vertex_count > std::numeric_limits<uint32_t>::max()) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster assembled vertex count is out of range.");
    }
    trace_graphics_step("compute assembly complete");
    Feather::Luisa::HostBufferBinding vertex_binding{
        0u, 1u, transformed_stride, &all_transformed_vertices,
        vertex_resident_key, false, false};
    Feather::Luisa::HostTextureBinding target_binding{
        0u, 2u, 3u, target.width, target.height, target.depth,
        target.mip_levels, target.pixel_format, &target.bytes,
        draw.color_targets[0], false, false, false};
    const auto [varying_resident_key, coverage_resident_key] = compute_raster_scratch_keys(
        pipeline, draw, transformed_stride, target.width, target.height);
    Feather::Luisa::HostTextureBinding depth_binding{};
    Feather::Luisa::HostTextureBinding* depth_binding_ptr = nullptr;
    auto depth_it = g_textures.end();
    if (draw.depth_target != 0u) {
        depth_it = g_textures.find(draw.depth_target);
        const auto required_depth_format = pipeline.stencil_test != 0u ? 100u : 101u;
        if (depth_it == g_textures.end() || depth_it->second.pixel_format != required_depth_format ||
            depth_it->second.width != target.width || depth_it->second.height != target.height) {
            return fail(FE_ERROR_UNSUPPORTED,
                        pipeline.stencil_test != 0u
                            ? "Compute raster stencil requires a matching Depth24Stencil8 target."
                            : "Compute raster depth requires a matching Depth32Float target.");
        }
        depth_binding = Feather::Luisa::HostTextureBinding{
            1u, 2u, 3u, depth_it->second.width, depth_it->second.height, depth_it->second.depth,
            depth_it->second.mip_levels, depth_it->second.pixel_format, &depth_it->second.bytes};
        depth_binding_ptr = &depth_binding;
    } else if (pipeline.depth_test != 0u || pipeline.depth_write != 0u) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Compute raster depth state requires a depth target.");
    }
    const auto clear_color = draw.clear_color != 0u ||
                             draw.color_load_op == static_cast<uint32_t>(GraphicsColorLoadOp::Clear);
    const auto load_color = draw.color_load_op == static_cast<uint32_t>(GraphicsColorLoadOp::Load);
    const auto clear_depth = depth_binding_ptr != nullptr &&
                             (draw.clear_depth != 0u ||
                              draw.depth_load_op == static_cast<uint32_t>(GraphicsDepthLoadOp::Clear) ||
                              draw.depth_load_op == static_cast<uint32_t>(GraphicsDepthLoadOp::Default));
    Feather::Luisa::RasterDispatchInputs raster{
        static_cast<uint32_t>(raster_vertex_count),
        draw.viewport_enabled != 0u ? draw.viewport_x : 0u,
        draw.viewport_enabled != 0u ? draw.viewport_y : 0u,
        draw.viewport_enabled != 0u ? draw.viewport_width : target.width,
        draw.viewport_enabled != 0u ? draw.viewport_height : target.height,
        draw.scissor_enabled != 0u ? draw.scissor_x : 0u,
        draw.scissor_enabled != 0u ? draw.scissor_y : 0u,
        draw.scissor_enabled != 0u ? draw.scissor_width : target.width,
        draw.scissor_enabled != 0u ? draw.scissor_height : target.height,
        pipeline.cull_mode,
        pipeline.front_face,
        pipeline.polygon_mode,
        pipeline.depth_test,
        pipeline.depth_write,
        pipeline.depth_compare,
        pipeline.depth_clamp,
        pipeline.stencil_test,
        pipeline.stencil_front.fail_op,
        pipeline.stencil_front.pass_op,
        pipeline.stencil_front.depth_fail_op,
        pipeline.stencil_front.compare_op,
        pipeline.stencil_back.fail_op,
        pipeline.stencil_back.pass_op,
        pipeline.stencil_back.depth_fail_op,
        pipeline.stencil_back.compare_op,
        pipeline.stencil_read_mask,
        pipeline.stencil_write_mask,
        pipeline.stencil_reference,
        clear_depth ? 1u : 0u,
        clear_depth && pipeline.stencil_test != 0u ? 1u : 0u,
        draw.clear_depth != 0u ? std::clamp(draw.clear_depth_value, 0.0f, 1.0f) : 1.0f,
        clear_color ? 1u : 0u,
        draw.clear_color != 0u ? draw.clear_color_r : 0.0f,
        draw.clear_color != 0u ? draw.clear_color_g : 0.0f,
        draw.clear_color != 0u ? draw.clear_color_b : 0.0f,
        draw.clear_color != 0u ? draw.clear_color_a : 1.0f,
        load_color ? 1u : 0u,
        draw.count,
        vertex_domain,
        pipeline.sample_count,
        std::all_of(
            pipeline.color_blend_attachments.begin(),
            pipeline.color_blend_attachments.begin() + pipeline.color_attachment_count,
            [](const auto& blend) { return blend.blend_enable == 0u && blend.write_mask == 0x0fu; })
            ? 1u
            : 0u};
    const auto geometry_cache_key = compute_raster_geometry_cache_key(
        vertex_resident_key, draw.index_buffer, index_revision, raster,
        target.width, target.height, draw);
    if (clear_color) {
        for (auto* color_target : color_targets) {
            if (!clear_compute_raster_color(
                    *color_target, raster.clear_color_r, raster.clear_color_g,
                    raster.clear_color_b, raster.clear_color_a)) {
                return fail(FE_ERROR_UNSUPPORTED, "Compute raster color clear format is unsupported.");
            }
        }
    }
    std::vector<unsigned char> fragment_varyings;
    std::vector<unsigned char> fragment_coverage;
    const auto fuse_fragment = raster.vertex_count >= 512u && depth_binding_ptr != nullptr &&
                               pipeline.sample_count == 4u && draw.color_target_count == 1u &&
                               raster.polygon_mode == 0u && raster.depth_test != 0u &&
                               raster.depth_write != 0u && raster.depth_compare == 1u &&
                               raster.stencil_test == 0u && raster.opaque_fragment != 0u;
    std::vector<uint64_t> fragment_callable_keys;
    std::array<uint64_t, 4u> fragment_target_keys{};
    if (fuse_fragment) {
        for (uint32_t sample = 0u; sample < fragment_target_keys.size(); ++sample) {
            auto key = draw.color_targets[0] ^
                       (0x6d73616173616d70ull +
                        static_cast<uint64_t>(sample + 1u) * 0x9e3779b97f4a7c15ull);
            fragment_target_keys[sample] = key == 0u ? static_cast<uint64_t>(sample + 1u) : key;
        }
        const auto prepare_result = dispatch_graphics_fragment_stage(
            pipeline, color_targets, nullptr, nullptr,
            varying_resident_key, coverage_resident_key,
            draw.color_targets, pipeline.sample_count,
            {raster.clear_color_r, raster.clear_color_g,
             raster.clear_color_b, raster.clear_color_a},
            load_color,
            &fragment_callable_keys);
        if (prepare_result != FE_OK) return prepare_result;
        trace_graphics_step("compute fragment prepared");
    }
    std::string error;
    if (!Feather::Luisa::DispatchVerticalRaster(
            vertex_binding, target_binding, raster_indices, depth_binding_ptr, raster, dispatch,
            varying_resident_key, coverage_resident_key,
            geometry_cache_key, !vertex_reused,
            fragment_callable_keys,
            fuse_fragment ? std::span<const uint64_t>{fragment_target_keys} : std::span<const uint64_t>{},
            &fragment_varyings, &fragment_coverage, &error)) {
        return fail(FE_ERROR_UNSUPPORTED, error.empty() ? "Compute raster dispatch failed." : error);
    }
    trace_graphics_step("compute raster complete");
    const auto fragment_result = fuse_fragment
        ? finish_fused_graphics_fragment_stage(
              color_targets, draw.color_targets, pipeline.sample_count)
        : dispatch_graphics_fragment_stage(
              pipeline, color_targets, &fragment_varyings, &fragment_coverage,
              varying_resident_key, coverage_resident_key, draw.color_targets, pipeline.sample_count,
              {raster.clear_color_r, raster.clear_color_g,
               raster.clear_color_b, raster.clear_color_a},
              load_color);
    if (fragment_result != FE_OK) return fragment_result;
    trace_graphics_step("compute fragment complete");
    for (auto* color_target : color_targets) {
        color_target->host_dirty = false;
        color_target->mipmaps_dirty = false;
        color_target->luisa_dirty = true;
        color_target->luisa_uploaded = true;
        ++color_target->content_revision;
    }
    if (depth_it != g_textures.end()) {
        depth_it->second.host_dirty = true;
        depth_it->second.mipmaps_dirty = false;
    }
    return ok();
}
#endif

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
FeWindowEvent to_fe_window_event(const Feather::Window::Event& source) {
    FeWindowEvent target{};
    std::visit(
        [&](const auto& event) {
            using EventT = std::decay_t<decltype(event)>;
            if constexpr (std::is_same_v<EventT, Feather::Window::ResizeEvent>) {
                target.kind = kWindowEventResize;
                target.width = event.width;
                target.height = event.height;
            } else if constexpr (std::is_same_v<EventT, Feather::Window::CloseEvent>) {
                target.kind = kWindowEventClose;
            } else if constexpr (std::is_same_v<EventT, Feather::Window::KeyEvent>) {
                target.kind = kWindowEventKey;
                target.key = static_cast<uint32_t>(static_cast<int32_t>(event.key));
                target.pressed = event.pressed ? 1u : 0u;
                target.modifiers = static_cast<uint32_t>(event.modifiers);
            } else if constexpr (std::is_same_v<EventT, Feather::Window::CharInputEvent>) {
                target.kind = kWindowEventCharInput;
                target.codepoint = event.codepoint;
            } else if constexpr (std::is_same_v<EventT, Feather::Window::MouseButtonEvent>) {
                target.kind = kWindowEventMouseButton;
                target.mouse_button = static_cast<uint32_t>(event.button);
                target.pressed = event.pressed ? 1u : 0u;
                target.x = event.x;
                target.y = event.y;
                target.modifiers = static_cast<uint32_t>(event.modifiers);
            } else if constexpr (std::is_same_v<EventT, Feather::Window::MouseMoveEvent>) {
                target.kind = kWindowEventMouseMove;
                target.x = event.x;
                target.y = event.y;
                target.dx = event.dx;
                target.dy = event.dy;
            } else if constexpr (std::is_same_v<EventT, Feather::Window::MouseScrollEvent>) {
                target.kind = kWindowEventMouseScroll;
                target.scroll_x = event.dx;
                target.scroll_y = event.dy;
            } else if constexpr (std::is_same_v<EventT, Feather::Window::FocusEvent>) {
                target.kind = kWindowEventFocus;
                target.pressed = event.focused ? 1u : 0u;
            }
        },
        source);
    return target;
}

FeResult prepare_texture_pixels_locked(TextureState& texture, std::vector<uint32_t>* converted,
                                       const uint32_t** pixels) {
    if (texture.depth != 1 || texture.width == 0 || texture.height == 0) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Texture presenter requires a valid 2D texture.");
    }
    if (texture.pixel_format != 3 && texture.pixel_format != 4 && texture.pixel_format != 10) {
        return fail(FE_ERROR_UNSUPPORTED, "Texture presenter currently supports Rgba8, Bgra8, and Rgba32Float textures.");
    }

    const auto pixel_count = static_cast<size_t>(texture.width) * texture.height;
    const auto minimum_bytes = pixel_count * pixel_size(texture.pixel_format);
    if (texture.bytes.size() < minimum_bytes) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Texture storage is smaller than its declared dimensions.");
    }

    if (texture.pixel_format == 3) {
        *pixels = reinterpret_cast<const uint32_t*>(texture.bytes.data());
        return ok();
    }

    converted->resize(pixel_count);
    if (texture.pixel_format == 10) {
        const auto* floats = reinterpret_cast<const float*>(texture.bytes.data());
        const auto to_byte = [](float value) -> uint32_t {
            const auto clamped = std::min(1.0f, std::max(0.0f, value));
            return static_cast<uint32_t>(clamped * 255.0f + 0.5f);
        };
        for (size_t i = 0; i < converted->size(); ++i) {
            const auto r = to_byte(floats[i * 4 + 0]);
            const auto g = to_byte(floats[i * 4 + 1]);
            const auto b = to_byte(floats[i * 4 + 2]);
            const auto a = to_byte(floats[i * 4 + 3]);
            (*converted)[i] = r | (g << 8) | (b << 16) | (a << 24);
        }
        *pixels = converted->data();
        return ok();
    }

    const auto* bgra = texture.bytes.data();
    for (size_t i = 0; i < converted->size(); ++i) {
        const auto b = static_cast<uint32_t>(bgra[i * 4 + 0]);
        const auto g = static_cast<uint32_t>(bgra[i * 4 + 1]);
        const auto r = static_cast<uint32_t>(bgra[i * 4 + 2]);
        const auto a = static_cast<uint32_t>(bgra[i * 4 + 3]);
        (*converted)[i] = r | (g << 8) | (b << 16) | (a << 24);
    }
    *pixels = converted->data();
    return ok();
}

void trace_present_submission(const char* path, std::chrono::steady_clock::time_point start,
                              Feather::Luisa::NativeTextureHandleKind handle_kind =
                                  Feather::Luisa::NativeTextureHandleKind::Unknown) {
    const auto* profile = std::getenv("FEATHER_PRESENT_PROFILE");
    if (profile == nullptr || profile[0] == '\0' || std::strcmp(profile, "0") == 0) return;
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    struct Samples {
        std::array<double, 120u> values{};
        size_t count = 0u;
    };
    static thread_local Samples resident_samples;
    static thread_local Samples staging_samples;
    auto& samples = std::strcmp(path, "resident-swapchain") == 0 ? resident_samples : staging_samples;
    samples.values[samples.count++] = elapsed;
    if (samples.count != samples.values.size()) return;
    auto ordered = samples.values;
    std::sort(ordered.begin(), ordered.end());
    const auto* native_kind = handle_kind == Feather::Luisa::NativeTextureHandleKind::MetalTexture ? "metal" :
                              handle_kind == Feather::Luisa::NativeTextureHandleKind::VulkanImage ? "vulkan" :
                              handle_kind == Feather::Luisa::NativeTextureHandleKind::Direct3D12Resource ? "d3d12" :
                                                                                                          "none";
    std::cerr << "[feather present] path=" << path << " native=" << native_kind
              << " calls=120 median_ms=" << std::fixed << std::setprecision(3)
              << ordered[ordered.size() / 2u] << " p95_ms=" << ordered[114u] << '\n';
    samples.count = 0u;
}

FeResult present_host_pixels_locked(FeContextHandle context, FeTexturePresenterHandle presenter,
                                    Feather::WindowHost& window, const uint32_t* pixels,
                                    uint32_t width, uint32_t height) {
    Feather::Luisa::DispatchInputs dispatch;
    std::string error;
    if (!configure_luisa_dispatch_locked(context, &dispatch, &error)) {
        return fail(FE_ERROR_INVALID_HANDLE, error.empty() ? "Invalid presentation context." : error);
    }
    const auto start = std::chrono::steady_clock::now();
    if (!Feather::Luisa::PresentHostTexture(
            dispatch.context_key, dispatch.runtime_directory, dispatch.backend_name,
            dispatch.device_index, presenter, pixels,
            static_cast<size_t>(width) * height * sizeof(uint32_t),
            window.NativeDisplay(), window.NativeWindow(), width, height, window.VSync(), &error)) {
        return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                    error.empty() ? "Luisa host-staged presentation failed." : error);
    }
    trace_graphics_step("presenter queued Luisa host fallback");
    trace_present_submission("host-staging", start);
    return ok();
}

void destroy_native_presenter_locked(FeTexturePresenterHandle handle, TexturePresenterState& presenter) {
    for (const auto context : presenter.native_contexts) {
        std::string ignored;
        (void)Feather::Luisa::DestroyPresenter(context, handle, &ignored);
    }
    presenter.native_contexts.clear();
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
            const auto decorated = "GPU backend unavailable: " + message;
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, decorated.c_str());
        }

        if (message.find("shader") != std::string::npos || message.find("Shader") != std::string::npos ||
            message.find("SPIR") != std::string::npos || message.find("GLSL") != std::string::npos ||
            message.find("pipeline") != std::string::npos || message.find("Pipeline") != std::string::npos) {
            const auto decorated = "GPU shader compilation failed: " + message;
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

FE_API FeResult fe_runtime_get_device_count(uint32_t* out_count) {
    return protect([&] {
        if (out_count == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_count must not be null.");
        }
#if FEATHER_HAS_LUISA
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto devices = Feather::Luisa::EnumerateDevices(configured_luisa_runtime_directory());
        if (devices.size() > std::numeric_limits<uint32_t>::max()) {
            return fail(FE_ERROR_UNKNOWN, "Luisa reported too many devices.");
        }
        *out_count = static_cast<uint32_t>(devices.size());
        return ok();
#else
        *out_count = 0u;
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_runtime_get_device_info(uint32_t ordinal, FeDeviceInfo* out_info) {
    return protect([&] {
        if (out_info == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_info must not be null.");
        }
#if FEATHER_HAS_LUISA
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto devices = Feather::Luisa::EnumerateDevices(configured_luisa_runtime_directory());
        if (ordinal >= devices.size()) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Device ordinal is out of range.");
        }
        *out_info = make_device_info(devices[ordinal]);
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_context_create(const char* backend_name, uint32_t device_index,
                                  FeContextHandle* out_context) {
    return protect([&] {
        if (backend_name == nullptr || backend_name[0] == '\0' || out_context == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "backend_name and out_context must not be null.");
        }
#if FEATHER_HAS_LUISA
        auto normalized_backend = std::string{backend_name};
        if (normalized_backend == "vulkan") normalized_backend = "vk";
        if (normalized_backend != "vk" && normalized_backend != "metal" &&
            normalized_backend != "cuda" && normalized_backend != "hip") {
            return fail(FE_ERROR_INVALID_ARGUMENT,
                        "backend_name must be one of: vk, metal, cuda, hip.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        Feather::Luisa::DeviceInfo selected{};
        std::string error;
        if (!Feather::Luisa::ValidateDevice(configured_luisa_runtime_directory(), normalized_backend,
                                           device_index, &selected, &error)) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                        error.empty() ? "Luisa device creation failed." : error.c_str());
        }
        const auto handle = next_handle();
        g_contexts.emplace(handle, ContextDeviceState{make_device_info(selected)});
        *out_context = handle;
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_context_initialize(FeContextHandle context) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        Feather::Luisa::DispatchInputs dispatch;
        std::string error;
        if (!configure_luisa_dispatch_locked(context, &dispatch, &error)) {
            return fail(FE_ERROR_INVALID_HANDLE, error);
        }
        Feather::Luisa::DeviceInfo selected;
        const auto device_index = dispatch.device_index == UINT32_MAX ? 0u : dispatch.device_index;
        if (!Feather::Luisa::ValidateDevice(dispatch.runtime_directory, dispatch.backend_name,
                                            device_index, &selected, &error)) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                        error.empty() ? "Luisa device initialization failed." : error);
        }
        return ok();
    });
}

FE_API FeResult fe_context_shutdown(FeContextHandle context) {
    if (context != kDefaultContext && context != 0) {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_contexts.find(context) == g_contexts.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        Feather::Luisa::Shutdown(context);
        std::erase_if(g_fences, [context](const auto& entry) { return entry.second.context == context; });
        std::erase_if(g_streams, [context](const auto& entry) { return entry.second.context == context; });
        for (auto it = g_pipelines.begin(); it != g_pipelines.end();) {
            if (it->second.context == context) {
                it = g_pipelines.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = g_kernels.begin(); it != g_kernels.end();) {
            if (it->second.context == context) {
                it->second.ad_gradients.clear();
                it = g_kernels.erase(it);
            } else {
                ++it;
            }
        }
        std::erase_if(g_textures, [context](const auto& entry) { return entry.second.context == context; });
        std::erase_if(g_buffers, [context](const auto& entry) { return entry.second.context == context; });
        std::erase_if(g_samplers, [context](const auto& entry) { return entry.second.context == context; });
        g_contexts.erase(context);
        return ok();
    }
    try {
        Feather::Luisa::Shutdown();
    } catch (...) {
        return fail(FE_ERROR_UNKNOWN, "Luisa runtime shutdown failed.");
    }
    return ok();
}

FE_API FeResult fe_context_get_device_info(FeContextHandle context, FeDeviceInfo* out_info) {
    return protect([&] {
        if (out_info == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_info must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        if (context != kDefaultContext) {
            const auto found = g_contexts.find(context);
            if (found == g_contexts.end()) {
                return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
            }
            *out_info = found->second.device;
            return ok();
        }
#if FEATHER_HAS_LUISA
        const auto devices = Feather::Luisa::EnumerateDevices(configured_luisa_runtime_directory());
        const auto found = std::find_if(devices.begin(), devices.end(), [](const auto& device) {
            return device.backend_name == Feather::Luisa::DefaultBackendName && device.device_index == 0u;
        });
        if (found == devices.end()) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, "The default Luisa device is unavailable.");
        }
        *out_info = make_device_info(*found);
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_runtime_flush_caches(void) {
    return ok();
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

FE_API FeResult fe_stream_create(FeContextHandle context, FeStreamHandle* out_stream) {
    return protect([&] {
        if (out_stream == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_stream must not be null.");
        }
#if FEATHER_HAS_LUISA
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        Feather::Luisa::DispatchInputs configured{};
        std::string error;
        if (!configure_luisa_dispatch_locked(context, &configured, &error)) {
            return fail(FE_ERROR_INVALID_HANDLE, error);
        }
        const auto handle = next_handle();
        if (!Feather::Luisa::CreateStream(configured.context_key, configured.runtime_directory, configured.backend_name,
                                          configured.device_index, handle, &error)) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, error.empty() ? "Luisa stream creation failed." : error);
        }
        g_streams.emplace(handle, StreamState{context});
        *out_stream = handle;
        return ok();
#else
        (void)context;
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_stream_destroy(FeStreamHandle stream) {
    return protect([&] {
        if (stream == 0u || g_runtime_shutting_down.load(std::memory_order_acquire))
            return ok();
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_streams.find(stream);
        if (found == g_streams.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid stream handle.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::DestroyStream(found->second.context, stream, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa stream destruction failed." : error);
        }
#endif
        g_streams.erase(found);
        return ok();
    });
}

FE_API FeResult fe_stream_synchronize(FeStreamHandle stream) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_streams.find(stream);
        if (found == g_streams.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid stream handle.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::SynchronizeStream(found->second.context, stream, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa stream synchronization failed." : error);
        }
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_stream_wait_fence(FeStreamHandle stream, FeFenceHandle fence) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto stream_state = g_streams.find(stream);
        const auto fence_state = g_fences.find(fence);
        if (stream_state == g_streams.end() || fence_state == g_fences.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid stream or fence handle.");
        }
        if (stream_state->second.context != fence_state->second.context) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot wait on a fence from a different GPU context.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::WaitFence(stream_state->second.context, stream, fence, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa stream wait failed." : error);
        }
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_fence_destroy(FeFenceHandle fence) {
    return protect([&] {
        if (fence == 0u || g_runtime_shutting_down.load(std::memory_order_acquire))
            return ok();
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_fences.find(fence);
        if (found == g_fences.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid fence handle.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::DestroyFence(found->second.context, fence, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa fence destruction failed." : error);
        }
#endif
        g_fences.erase(found);
        return ok();
    });
}

FE_API FeResult fe_fence_is_completed(FeFenceHandle fence, bool* out_completed) {
    return protect([&] {
        if (out_completed == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_completed must not be null.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_fences.find(fence);
        if (found == g_fences.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid fence handle.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::IsFenceCompleted(found->second.context, fence, out_completed, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa fence query failed." : error);
        }
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
    });
}

FE_API FeResult fe_fence_wait(FeFenceHandle fence) {
    return protect([&] {
        std::lock_guard<std::mutex> lock(g_mutex);
        const auto found = g_fences.find(fence);
        if (found == g_fences.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid fence handle.");
        }
#if FEATHER_HAS_LUISA
        std::string error;
        if (!Feather::Luisa::SynchronizeFence(found->second.context, fence, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa fence wait failed." : error);
        }
        return ok();
#else
        return fail(FE_ERROR_BACKEND_UNAVAILABLE, "Luisa runtime support was not built.");
#endif
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

        Feather::Window::Config config;
        config.width = desc->width;
        config.height = desc->height;
        config.title = desc->title == nullptr ? "Feather" : desc->title;
        config.resizable = desc->resizable != 0;
        config.visible = desc->visible != 0;
        config.vsync = desc->vsync != 0;
        config.high_dpi = desc->high_dpi != 0;
        config.center_on_create = desc->center_on_create != 0;

        WindowState state;
        state.window = std::make_unique<Feather::WindowHost>(config);

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
#if FEATHER_HAS_LUISA
                destroy_native_presenter_locked(it->first, it->second);
#endif
                it = g_texture_presenters.erase(it);
            } else {
                ++it;
            }
        }
        const auto found = g_windows.find(window);
        if (found == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid window handle.");
        }
#if FEATHER_HAS_LUISA
        for (const auto context : found->second.native_contexts) {
            std::string ignored;
            (void)Feather::Luisa::DestroyPresenter(context, window, &ignored);
        }
#endif
        g_windows.erase(found);
        return ok();
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
        Feather::Window::Event event;
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
        *out_is_down = it->second.window->IsKeyDown(static_cast<Feather::Window::Key>(static_cast<int32_t>(key)));
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
        *out_is_down = it->second.window->IsMouseDown(static_cast<Feather::Window::MouseButton>(mouse_button));
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
        const auto result = present_host_pixels_locked(
            kDefaultContext, window, *it->second.window, pixels, width, height);
        if (result == FE_OK) it->second.native_contexts.emplace(kDefaultContext);
        return result;
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
        const auto found = g_texture_presenters.find(presenter);
        if (found == g_texture_presenters.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid texture presenter handle.");
        }
#if FEATHER_HAS_LUISA
        destroy_native_presenter_locked(presenter, found->second);
#endif
        g_texture_presenters.erase(found);
        return ok();
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
        const auto window_it = g_windows.find(presenter_it->second.window_handle);
        if (window_it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Texture presenter window is no longer available.");
        }
        auto& texture_state = texture_it->second;

        if (mode != 1u && texture_state.luisa_uploaded) {
            Feather::Luisa::NativeTextureInfo native_texture;
            std::string error;
            const auto start = std::chrono::steady_clock::now();
            if (Feather::Luisa::PresentResidentTexture(
                    texture_state.context, presenter, texture,
                    window_it->second.window->NativeDisplay(), window_it->second.window->NativeWindow(),
                    texture_state.width, texture_state.height, window_it->second.window->VSync(),
                    &native_texture, &error)) {
                presenter_it->second.native_contexts.emplace(texture_state.context);
                trace_graphics_step("presenter queued Luisa resident texture");
                trace_present_submission("resident-swapchain", start, native_texture.kind);
                return ok();
            }
            if (mode == 2u) {
                return fail(FE_ERROR_UNSUPPORTED,
                            error.empty() ? "Native texture presentation is unavailable." : error);
            }
        }

        // When the selected backend cannot create a swapchain for the native window,
        // retain the bounded asynchronous readback ring and upload through Luisa.
        auto& async = presenter_it->second.async_textures[texture];
        if (async == nullptr) async = std::make_shared<AsyncTexturePresentation>();
        std::shared_ptr<AsyncPresentationFrame> newest;
        for (const auto& frame : async->frames) {
            if (frame->state.load(std::memory_order_acquire) == kPresentationFrameReady &&
                (newest == nullptr || frame->revision > newest->revision)) {
                newest = frame;
            }
        }
        if (newest != nullptr) {
            const auto presented_revision = newest->revision;
            texture_state.bytes.swap(newest->bytes);
            for (const auto& frame : async->frames) {
                if (frame->state.load(std::memory_order_acquire) == kPresentationFrameReady &&
                    frame->revision <= presented_revision) {
                    frame->state.store(kPresentationFrameFree, std::memory_order_release);
                }
            }
            texture_state.host_dirty = true;
            texture_state.luisa_dirty = presented_revision != texture_state.content_revision;
            texture_state.mipmaps_dirty = texture_state.mipmaps_requested && texture_state.mip_levels > 1u;
            async->has_presented_frame = true;
            trace_graphics_step("presenter consumed async Luisa frame");
        }

        if (texture_state.luisa_dirty && !async->has_presented_frame) {
            std::string error;
            if (!Feather::Luisa::DownloadResidentTexture(
                    texture_state.context, texture, texture_state.bytes.data(), texture_state.bytes.size(), &error)) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            error.empty() ? "Luisa texture presentation download failed." : error);
            }
            texture_state.luisa_dirty = false;
            texture_state.host_dirty = true;
            texture_state.mipmaps_dirty = texture_state.mipmaps_requested && texture_state.mip_levels > 1u;
            async->last_scheduled_revision = texture_state.content_revision;
            async->has_presented_frame = true;
            trace_graphics_step("presenter synchronized first Luisa frame");
        } else if (texture_state.luisa_dirty &&
                   texture_state.content_revision > async->last_scheduled_revision) {
            const auto frame = std::find_if(async->frames.begin(), async->frames.end(), [](const auto& candidate) {
                return candidate->state.load(std::memory_order_acquire) == kPresentationFrameFree;
            });
            if (frame != async->frames.end()) {
                (*frame)->bytes.resize(texture_state.bytes.size());
                (*frame)->revision = texture_state.content_revision;
                (*frame)->state.store(kPresentationFramePending, std::memory_order_release);
                std::string error;
                if (!Feather::Luisa::DownloadResidentTextureAsync(
                        texture_state.context, texture, (*frame)->bytes.data(), (*frame)->bytes.size(),
                        [frame = *frame] {
                            frame->state.store(kPresentationFrameReady, std::memory_order_release);
                        },
                        &error)) {
                    (*frame)->state.store(kPresentationFrameFree, std::memory_order_release);
                    return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                                error.empty() ? "Luisa texture presentation queue failed." : error);
                }
                async->last_scheduled_revision = texture_state.content_revision;
                trace_graphics_step("presenter queued async Luisa frame");
            }
        }
        std::vector<uint32_t> converted;
        const uint32_t* pixels = nullptr;
        const auto prepared = prepare_texture_pixels_locked(texture_state, &converted, &pixels);
        if (prepared != FE_OK) return prepared;
        const auto result = present_host_pixels_locked(
            texture_state.context, presenter, *window_it->second.window,
            pixels, texture_state.width, texture_state.height);
        if (result == FE_OK) presenter_it->second.native_contexts.emplace(texture_state.context);
        return result;
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
        const auto window_it = g_windows.find(presenter_it->second.window_handle);
        if (window_it == g_windows.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Texture presenter window is no longer available.");
        }
        const auto result = present_host_pixels_locked(
            kDefaultContext, presenter, *window_it->second.window, pixels, width, height);
        if (result == FE_OK) presenter_it->second.native_contexts.emplace(kDefaultContext);
        return result;
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
        if (desc == nullptr || out_buffer == nullptr || desc->size_in_bytes == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Buffer descriptor and output handle are required.");
        }

        BufferState state;
        state.context = context;
        state.bytes.resize(static_cast<size_t>(desc->size_in_bytes));
        state.mode = desc->mode;
        state.stride = desc->element_stride;
        if (initial_data != nullptr) {
            std::memcpy(state.bytes.data(), initial_data, state.bytes.size());
        }

        std::lock_guard<std::mutex> lock(g_mutex);
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        const auto handle = next_handle();
        g_buffers.emplace(handle, std::move(state));
        *out_buffer = handle;
        return ok();
    });
}

FE_API FeResult fe_accel_create(FeContextHandle context, uint32_t mesh_count,
                                 const FeAccelMeshDesc* meshes, FeAccelHandle* out_accel) {
    return protect([&] {
        if (meshes == nullptr || out_accel == nullptr || mesh_count == 0u) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Accel meshes and output handle are required.");
        }
        std::vector<Feather::Luisa::AccelMeshDesc> descs;
        descs.reserve(mesh_count);
        for (uint32_t i = 0u; i < mesh_count; i++) {
            const auto vertex = g_buffers.find(meshes[i].vertex_buffer);
            const auto index = g_buffers.find(meshes[i].index_buffer);
            if (vertex == g_buffers.end() || index == g_buffers.end()) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Accel mesh references an unknown buffer.");
            }
            if (vertex->second.stride != sizeof(float)) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Accel vertex buffer must be flat float data.");
            }
            if (vertex->second.bytes.size() % (sizeof(float) * 3u) != 0u) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Accel vertex buffer must hold float3 triplets.");
            }
            if (index->second.stride != sizeof(uint32_t)) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Accel index buffer must be uint (stride 4).");
            }
            const auto vertex_count = vertex->second.bytes.size() / vertex->second.stride;
            const auto index_count = index->second.bytes.size() / index->second.stride;
            if (vertex_count == 0u || index_count == 0u || index_count % 3u != 0u) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Accel mesh buffers have invalid sizes.");
            }
            Feather::Luisa::AccelMeshDesc desc;
            desc.vertex_count = static_cast<uint32_t>(vertex_count);
            desc.vertices = reinterpret_cast<const float*>(vertex->second.bytes.data());
            desc.index_count = static_cast<uint32_t>(index_count);
            desc.indices = reinterpret_cast<const uint32_t*>(index->second.bytes.data());
            descs.emplace_back(std::move(desc));
        }
        uint64_t accel_key = 0u;
        std::string error;
        const auto context_found = g_contexts.find(context);
        if (context_found == g_contexts.end() && context != kDefaultContext) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        const auto* device_info = context_found == g_contexts.end()
                                      ? nullptr
                                      : &context_found->second.device;
        const auto backend_name = device_info != nullptr
                                      ? device_info->backend_name
                                      : std::string{Feather::Luisa::DefaultBackendName};
        const auto device_index = device_info != nullptr ? device_info->device_index : 0u;
        if (!Feather::Luisa::CreateAccel(context, configured_luisa_runtime_directory(),
                                         backend_name, device_index, descs,
                                         &accel_key, &error)) {
            return fail(FE_ERROR_BACKEND_UNAVAILABLE, error.c_str());
        }
        AccelState state;
        state.context = context;
        state.accel_key = accel_key;
        const auto handle = next_handle();
        g_accels.emplace(handle, state);
        *out_accel = handle;
        return FE_OK;
    });
}

FE_API FeResult fe_accel_destroy(FeContextHandle context, FeAccelHandle accel) {
    return protect([&] {
        const auto found = g_accels.find(accel);
        if (found == g_accels.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid accel handle.");
        }
        std::string error;
        if (!Feather::Luisa::DestroyAccel(context, found->second.accel_key, &error)) {
            return fail(FE_ERROR_INVALID_HANDLE, error.c_str());
        }
        g_accels.erase(found);
        return FE_OK;
    });
}

FE_API FeResult fe_accel_destroy_raw(FeContextHandle context, uint64_t accel) {
    return fe_accel_destroy(context, static_cast<FeAccelHandle>(accel));
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
        std::string error;
        if (!synchronize_luisa_context_locked(it->second.context, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa resource synchronization failed." : error);
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
        std::string error;
        if (!synchronize_luisa_context_locked(it->second.context, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa resource synchronization failed." : error);
        }
        std::memcpy(it->second.bytes.data() + offset, data, static_cast<size_t>(size));
        it->second.host_dirty = true;
        it->second.luisa_uploaded = false;
        ++it->second.content_revision;
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
        std::string error;
        if (!synchronize_luisa_context_locked(it->second.context, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa resource synchronization failed." : error);
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
        std::string error;
        if (!synchronize_luisa_context_locked(it->second.context, &error)) {
            return fail(FE_ERROR_UNKNOWN, error.empty() ? "Luisa resource synchronization failed." : error);
        }
        *out_ptr = it->second.bytes.data();
        it->second.host_dirty = true;
        it->second.luisa_uploaded = false;
        ++it->second.content_revision;
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
        if (desc == nullptr || out_texture == nullptr || desc->width == 0 || desc->height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Texture descriptor and output handle are required.");
        }
        trace_graphics_step("create texture2d");
        TextureState state;
        state.context = context;
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
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
        const auto handle = next_handle();
        g_textures.emplace(handle, std::move(state));
        *out_texture = handle;
        return ok();
    });
}

FE_API FeResult fe_texture3d_create(FeContextHandle context, const FeTexture3DDesc* desc, const void* initial_data,
                                    FeTextureHandle* out_texture) {
    return protect([&] {
        if (desc == nullptr || out_texture == nullptr || desc->width == 0 || desc->height == 0 || desc->depth == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "3D texture descriptor and output handle are required.");
        }

        TextureState state;
        state.context = context;
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
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
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
        std::string synchronization_error;
        if (!synchronize_luisa_context_locked(it->second.context, &synchronization_error)) {
            return fail(FE_ERROR_UNKNOWN, synchronization_error.empty() ? "Luisa resource synchronization failed."
                                                                        : synchronization_error);
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
        std::string synchronization_error;
        if (!synchronize_luisa_context_locked(it->second.context, &synchronization_error)) {
            return fail(FE_ERROR_UNKNOWN, synchronization_error.empty() ? "Luisa resource synchronization failed."
                                                                        : synchronization_error);
        }
        if (it->second.luisa_dirty) {
            std::string error;
            if (!Feather::Luisa::DownloadResidentTexture(
                    it->second.context, texture, it->second.bytes.data(), it->second.bytes.size(), &error)) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            error.empty() ? "Luisa texture upload could not preserve resident data." : error);
            }
            it->second.luisa_dirty = false;
        }
        trace_graphics_step("upload texture2d");
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
        it->second.luisa_uploaded = false;
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
        std::string synchronization_error;
        if (!synchronize_luisa_context_locked(it->second.context, &synchronization_error)) {
            return fail(FE_ERROR_UNKNOWN, synchronization_error.empty() ? "Luisa resource synchronization failed."
                                                                        : synchronization_error);
        }
        if (it->second.luisa_dirty) {
            std::string error;
            if (!Feather::Luisa::DownloadResidentTexture(
                    it->second.context, texture, it->second.bytes.data(), it->second.bytes.size(), &error)) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            error.empty() ? "Luisa texture download failed." : error);
            }
            it->second.luisa_dirty = false;
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
        std::string synchronization_error;
        if (!synchronize_luisa_context_locked(it->second.context, &synchronization_error)) {
            return fail(FE_ERROR_UNKNOWN, synchronization_error.empty() ? "Luisa resource synchronization failed."
                                                                        : synchronization_error);
        }
        if (it->second.luisa_dirty) {
            std::string error;
            if (!Feather::Luisa::DownloadResidentTexture(
                    it->second.context, texture, it->second.bytes.data(), it->second.bytes.size(), &error)) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            error.empty() ? "Luisa 3D texture upload could not preserve resident data." : error);
            }
            it->second.luisa_dirty = false;
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
        it->second.luisa_uploaded = false;
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
        std::string synchronization_error;
        if (!synchronize_luisa_context_locked(it->second.context, &synchronization_error)) {
            return fail(FE_ERROR_UNKNOWN, synchronization_error.empty() ? "Luisa resource synchronization failed."
                                                                        : synchronization_error);
        }
        if (it->second.luisa_dirty) {
            std::string error;
            if (!Feather::Luisa::DownloadResidentTexture(
                    it->second.context, texture, it->second.bytes.data(), it->second.bytes.size(), &error)) {
                return fail(FE_ERROR_BACKEND_UNAVAILABLE,
                            error.empty() ? "Luisa 3D texture download failed." : error);
            }
            it->second.luisa_dirty = false;
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
        state.luisa_uploaded = false;
        return ok();
    });
}

FE_API FeResult fe_bilinear_upscale_rgba8(const uint8_t* source, uint32_t source_width, uint32_t source_height,
                                          uint8_t* destination, uint32_t width, uint32_t height) {
    return protect([&] {
        if (source == nullptr || destination == nullptr || source_width == 0 || source_height == 0 ||
            width == 0 || height == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Bilinear RGBA8 images require data and positive dimensions.");
        }

        struct HorizontalTap {
            uint32_t x0;
            uint32_t x1;
            uint32_t weight;
        };

        std::vector<HorizontalTap> horizontal(width);
        for (uint32_t x = 0; x < width; ++x) {
            const float coordinate = ((static_cast<float>(x) + 0.5f) * source_width / width) - 0.5f;
            const int base = static_cast<int>(std::floor(coordinate));
            horizontal[x] = {
                static_cast<uint32_t>(std::clamp(base, 0, static_cast<int>(source_width) - 1)),
                static_cast<uint32_t>(std::clamp(base + 1, 0, static_cast<int>(source_width) - 1)),
                static_cast<uint32_t>(std::clamp(
                    static_cast<int>(std::lround((coordinate - std::floor(coordinate)) * 256.0f)), 0, 256))};
        }

        for (uint32_t y = 0; y < height; ++y) {
            const float coordinate = ((static_cast<float>(y) + 0.5f) * source_height / height) - 0.5f;
            const int base = static_cast<int>(std::floor(coordinate));
            const auto y0 = static_cast<uint32_t>(std::clamp(base, 0, static_cast<int>(source_height) - 1));
            const auto y1 = static_cast<uint32_t>(std::clamp(base + 1, 0, static_cast<int>(source_height) - 1));
            const auto wy = static_cast<uint32_t>(std::clamp(
                static_cast<int>(std::lround((coordinate - std::floor(coordinate)) * 256.0f)), 0, 256));
            const auto inverse_y = 256u - wy;
            for (uint32_t x = 0; x < width; ++x) {
                const auto tap = horizontal[x];
                const auto inverse_x = 256u - tap.weight;
                const size_t offsets[4] = {
                    (static_cast<size_t>(y0) * source_width + tap.x0) * 4,
                    (static_cast<size_t>(y0) * source_width + tap.x1) * 4,
                    (static_cast<size_t>(y1) * source_width + tap.x0) * 4,
                    (static_cast<size_t>(y1) * source_width + tap.x1) * 4};
                const auto destination_offset = (static_cast<size_t>(y) * width + x) * 4;
                for (size_t channel = 0; channel < 4; ++channel) {
                    const auto lower = source[offsets[0] + channel] * inverse_x +
                                       source[offsets[1] + channel] * tap.weight;
                    const auto upper = source[offsets[2] + channel] * inverse_x +
                                       source[offsets[3] + channel] * tap.weight;
                    destination[destination_offset + channel] = static_cast<uint8_t>(
                        ((lower * inverse_y) + (upper * wy) + 32768u) >> 16);
                }
            }
        }

        return ok();
    });
}

FE_API FeResult fe_sampler_create(FeContextHandle context, const FeSamplerDesc* desc, FeSamplerHandle* out_sampler) {
    return protect([&] {
        if (desc == nullptr || out_sampler == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Sampler descriptor and output handle are required.");
        }
        if (!validate_sampler_desc(*desc)) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Sampler descriptor contains an unsupported enum value.");
        }

        SamplerState state;
        state.context = context;
        state.desc = *desc;
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
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
        if (desc == nullptr || out_kernel == nullptr || desc->ir_data == nullptr || desc->ir_size == 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel IR and output handle are required.");
        }
        const auto validation = fe_ir_validate(desc->ir_data, desc->ir_size);
        if (validation != FE_OK) {
            return fail(validation, "Kernel IR failed Feather IR validation.");
        }
        KernelState state;
        state.context = context;
        const auto* bytes = static_cast<const unsigned char*>(desc->ir_data);
        state.ir.assign(bytes, bytes + desc->ir_size);
        state.debug_name = copy_debug_name(desc->debug_name, "Kernel");
        state.auto_diff = desc->auto_diff;
        state.bounds_check = desc->bounds_check;
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
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
        release_ad_gradient_buffers(it->second);
        g_kernels.erase(it);
        return ok();
    });
}

FE_API FeResult fe_kernel_bind_buffer(FeKernelHandle kernel, uint32_t binding, FeBufferHandle buffer) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    const auto resource = g_buffers.find(buffer);
    if (it == g_kernels.end() || resource == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or buffer handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a buffer from a different GPU context.");
    }
    it->second.buffers[binding] = buffer;
    return ok();
}

FE_API FeResult fe_kernel_bind_accel(FeKernelHandle kernel, uint32_t binding, FeAccelHandle accel) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    const auto resource = g_accels.find(accel);
    if (it == g_kernels.end() || resource == g_accels.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or accel handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind an accel from a different GPU context.");
    }
    it->second.accels[binding] = accel;
    return ok();
}

FE_API FeResult fe_kernel_bind_texture(FeKernelHandle kernel, uint32_t binding, FeTextureHandle texture) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_kernels.find(kernel);
    const auto resource = g_textures.find(texture);
    if (it == g_kernels.end() || resource == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or texture handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a texture from a different GPU context.");
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
    if (sampler != 0) {
        const auto resource = g_samplers.find(sampler);
        if (resource == g_samplers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid sampler handle.");
        }
        if (it->second.context != resource->second.context) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a sampler from a different GPU context.");
        }
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

        it->second.last_dispatch_path = FE_DISPATCH_PATH_NONE;
        const auto fence_key = wait ? 0u : next_handle();
        result = dispatch_luisa_kernel(kernel, it->second, group_x, group_y, group_z, logical_x, logical_y,
                                       logical_z, wait, 0u, fence_key);
        it->second.last_dispatch_path = result == FE_OK ? FE_DISPATCH_PATH_LUISA : FE_DISPATCH_PATH_REJECTED;

        if (should_profile && result == FE_OK) {
            const auto elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            record_profiler_event_locked(it->second.debug_name, elapsed, logical_x, logical_y, logical_z);
        }

        return result;
    });
}

FE_API FeResult fe_kernel_dispatch_stream(FeKernelHandle kernel, FeStreamHandle stream, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t logical_x, uint32_t logical_y,
                                          uint32_t logical_z, FeFenceHandle* out_fence) {
    return protect([&] {
        if (out_fence == nullptr) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "out_fence must not be null.");
        }
        if (group_x == 0u || group_y == 0u || group_z == 0u || logical_x == 0u || logical_y == 0u || logical_z == 0u) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Kernel dispatch group counts and logical sizes must be positive.");
        }
        std::lock_guard<std::mutex> lock(g_mutex);
        auto kernel_state = g_kernels.find(kernel);
        const auto stream_state = g_streams.find(stream);
        if (kernel_state == g_kernels.end() || stream_state == g_streams.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid kernel or stream handle.");
        }
        if (kernel_state->second.context != stream_state->second.context) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "GPU kernel and stream must have the same owner context.");
        }
        if (kernel_state->second.auto_diff) {
            return fail(
                FE_ERROR_UNSUPPORTED,
                "Asynchronous autodiff dispatch is not supported because gradient retrieval is host-synchronous.");
        }
        kernel_state->second.logical_x = static_cast<int32_t>(logical_x);
        kernel_state->second.logical_y = static_cast<int32_t>(logical_y);
        kernel_state->second.logical_z = static_cast<int32_t>(logical_z);
        const auto fence = next_handle();
        const auto result = dispatch_luisa_kernel(kernel, kernel_state->second, group_x, group_y, group_z, logical_x,
                                                  logical_y, logical_z, false, stream, fence);
        kernel_state->second.last_dispatch_path = result == FE_OK ? FE_DISPATCH_PATH_LUISA : FE_DISPATCH_PATH_REJECTED;
        if (result != FE_OK)
            return result;
        g_fences.emplace(fence, FenceState{stream_state->second.context, stream});
        *out_fence = fence;
        return ok();
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
        copy_fixed_c_string(out_info->native_name, sizeof(out_info->native_name), gradient.native_name);
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

        const auto& gradient = it->second.ad_gradients[index];
        if (offset > gradient.byte_size || size > gradient.byte_size - offset) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "AD gradient read range exceeds gradient buffer size.");
        }
        if (size == 0) {
            return ok();
        }
        if (gradient.host_bytes.size() < gradient.byte_size) {
            return fail(FE_ERROR_INVALID_HANDLE, "AD gradient buffer is not available.");
        }
        std::memcpy(out_data, gradient.host_bytes.data() + static_cast<size_t>(offset), static_cast<size_t>(size));
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
        if (kernel_it->second.context != destination_it->second.context) {
            return fail(FE_ERROR_INVALID_ARGUMENT,
                        "Cannot reduce a gradient into a buffer from a different GPU context.");
        }

        if (destination_size == 0) return ok();
        const auto& gradient = kernel_it->second.ad_gradients[index];
        Feather::Luisa::DispatchInputs dispatch{};
        std::string error;
        if (!configure_luisa_dispatch_locked(kernel_it->second.context, &dispatch, &error))
            return fail(FE_ERROR_INVALID_HANDLE, error);
        if (!Feather::Luisa::ReduceAdGradient(
                dispatch.context_key, dispatch.runtime_directory, dispatch.backend_name,
                dispatch.device_index, gradient.host_bytes, gradient.element_count,
                std::max(gradient.component_count, 1u), destination,
                &destination_it->second.bytes, destination_offset, destination_size,
                !destination_it->second.luisa_uploaded, &error)) {
            return fail(FE_ERROR_UNSUPPORTED,
                        error.empty() ? "AD gradient could not be reduced to the destination buffer." : error);
        }
        destination_it->second.host_dirty = false;
        destination_it->second.luisa_uploaded = true;
        ++destination_it->second.content_revision;
        return ok();
    });
}

FE_API FeResult fe_graphics_pipeline_create_from_ir(FeContextHandle context, const FeGraphicsPipelineCreateDesc* desc,
                                                    FeGraphicsPipelineHandle* out_pipeline) {
    return protect([&] {
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
        if (desc->color_attachment_count == 0 || desc->color_attachment_count > kMaximumColorAttachments) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color attachment count must be between 1 and 8.");
        }
        if (desc->color_blend_attachment_count != 0 &&
            desc->color_blend_attachment_count != desc->color_attachment_count) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color blend attachment count must match color attachment count.");
        }
        const auto valid_stencil_face = [](const FeGraphicsStencilFaceDesc& face) noexcept {
            return face.fail_op <= 7u && face.pass_op <= 7u &&
                   face.depth_fail_op <= 7u && face.compare_op <= 7u;
        };
        const auto valid_blend_attachment = [](const FeGraphicsColorBlendAttachmentDesc& blend) noexcept {
            return blend.src_color <= 9u && blend.dst_color <= 9u &&
                   blend.src_alpha <= 9u && blend.dst_alpha <= 9u &&
                   blend.color_op <= 4u && blend.alpha_op <= 4u &&
                   (blend.write_mask & ~15u) == 0u;
        };
        if (desc->depth_compare > 7u || !valid_stencil_face(desc->stencil_front) ||
            !valid_stencil_face(desc->stencil_back) || desc->blend_src_color > 9u ||
            desc->blend_dst_color > 9u || desc->blend_src_alpha > 9u ||
            desc->blend_dst_alpha > 9u || desc->blend_color_op > 4u ||
            desc->blend_alpha_op > 4u || desc->cull_mode > 3u ||
            desc->front_face > 1u || desc->polygon_mode > 2u) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline descriptor contains an unsupported state enum value.");
        }
        if ((desc->blend_write_mask & ~15u) != 0) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline blend write mask contains unsupported bits.");
        }
        for (uint32_t i = 0; i < desc->color_blend_attachment_count; ++i) {
            if (!valid_blend_attachment(desc->color_blend_attachments[i])) {
                return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics pipeline color blend attachment descriptor contains an unsupported value.");
            }
        }

        GraphicsPipelineState state;
        state.context = context;
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
        if (!context_exists_locked(context)) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid context handle.");
        }
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
        g_pipelines.erase(it);
        return ok();
    });
}

FE_API FeResult fe_graphics_pipeline_set_vertex_buffer(FeGraphicsPipelineHandle pipeline, FeBufferHandle buffer,
                                                       uint32_t stride) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    const auto resource = g_buffers.find(buffer);
    if (it == g_pipelines.end() || resource == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or buffer handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a vertex buffer from a different GPU context.");
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
    const auto resource = g_buffers.find(buffer);
    if (it == g_pipelines.end() || resource == g_buffers.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or buffer handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a buffer from a different GPU context.");
    }
    it->second.buffers[binding] = buffer;
    return ok();
}

FE_API FeResult fe_graphics_pipeline_bind_texture(FeGraphicsPipelineHandle pipeline, uint32_t binding,
                                                  FeTextureHandle texture) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_pipelines.find(pipeline);
    const auto resource = g_textures.find(texture);
    if (it == g_pipelines.end() || resource == g_textures.end()) {
        return fail(FE_ERROR_INVALID_HANDLE, "Invalid pipeline or texture handle.");
    }
    if (it->second.context != resource->second.context) {
        return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a texture from a different GPU context.");
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
    if (sampler != 0) {
        const auto resource = g_samplers.find(sampler);
        if (resource == g_samplers.end()) {
            return fail(FE_ERROR_INVALID_HANDLE, "Invalid sampler handle.");
        }
        if (it->second.context != resource->second.context) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Cannot bind a sampler from a different GPU context.");
        }
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
        if (desc->color_target_count > kMaximumColorAttachments) {
            return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics draw color target count exceeds Feather limits.");
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
            const auto target = g_textures.find(desc->color_targets[i]);
            if (desc->color_targets[i] == 0 || target == g_textures.end()) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid color target.");
            }
            if (target->second.context != it->second.context) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics color target belongs to a different GPU context.");
            }
        }
        if (desc->depth_target != 0) {
            const auto target = g_textures.find(desc->depth_target);
            if (target == g_textures.end()) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics draw references an invalid depth target.");
            }
            if (target->second.context != it->second.context) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics depth target belongs to a different GPU context.");
            }
        }
        if (desc->indexed != 0) {
            const auto index = g_buffers.find(desc->index_buffer);
            if (desc->index_buffer == 0 || index == g_buffers.end()) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_HANDLE, "Graphics indexed draw requires a valid explicit index buffer.");
            }
            if (index->second.context != it->second.context) {
                it->second.last_dispatch_path = FE_DISPATCH_PATH_REJECTED;
                return fail(FE_ERROR_INVALID_ARGUMENT, "Graphics index buffer belongs to a different GPU context.");
            }
        }

        const auto should_profile = profiler_enabled_locked();
        const auto start = std::chrono::steady_clock::now();
#if FEATHER_HAS_LUISA
        const auto result = draw_graphics_pipeline_compute_raster(it->second, *desc);
#endif
#if !FEATHER_HAS_LUISA
        const auto result = fail(FE_ERROR_BACKEND_UNAVAILABLE, "Feather was built without LuisaCompute.");
#endif
        it->second.last_dispatch_path = result == FE_OK ? FE_DISPATCH_PATH_LUISA : FE_DISPATCH_PATH_REJECTED;
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
