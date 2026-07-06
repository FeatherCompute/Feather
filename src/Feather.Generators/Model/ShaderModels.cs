using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Feather.Generators.Model;

internal enum ShaderKind
{
    Compute1D,
    Compute2D,
    Compute3D,
    Vertex,
    Fragment
}

internal sealed record ShaderModel(
    StructDeclarationSyntax Syntax,
    INamedTypeSymbol Symbol,
    ShaderKind Kind,
    string Namespace,
    string Name,
    string FullyQualifiedMetadataName,
    MethodDeclarationSyntax? EntryPointSyntax,
    IMethodSymbol? EntryPointSymbol,
    ThreadGroupModel ThreadGroup,
    bool BoundsCheck,
    bool AutoDiff,
    EquatableArray<ResourceModel> Resources,
    EquatableArray<LoweredShaderInstructionModel> LoweredInstructions,
    EquatableArray<CallableMethodModel> Callables = default,
    byte[]? TypedIrSection = null,
    EquatableArray<TypedIrDiagnosticModel> TypedIrDiagnostics = default,
    EquatableArray<ShaderBodyDiagnosticModel> BodyDiagnostics = default);

internal sealed record TypedIrDiagnosticModel(Location? Location, string Message);

internal sealed record ShaderBodyDiagnosticModel(
    DiagnosticDescriptor Descriptor,
    Location? Location,
    EquatableArray<string> Arguments);

/// <summary>A [Callable] or [ShaderFunction] method imported into a shader module.</summary>
internal sealed record CallableMethodModel(
    MethodDeclarationSyntax Syntax,
    IMethodSymbol Symbol,
    string Name,
    string MangledName,
    string ReturnTypeName,
    bool IsStatic,
    EquatableArray<CallableParameterModel> Parameters);

internal sealed record CallableParameterModel(
    string Name,
    string TypeName,
    bool IsRef);

internal sealed record ThreadGroupModel(int X, int Y, int Z);

internal sealed record ResourceModel(
    uint Binding,
    string Name,
    string TypeName,
    string ElementTypeName,
    ResourceKindModel Kind,
    ResourceAccessModel Access,
    bool IsUniformWrapper = false,
    bool IsPushConstantSupported = true,
    bool IsComputeStorageElementSupported = true);

internal sealed record GraphicsPipelineModel(
    ShaderModel VertexShader,
    ShaderModel FragmentShader,
    string VaryingsTypeName,
    string VertexInterfaceName,
    string FragmentInterfaceName);

internal enum ResourceKindModel
{
    Buffer,
    Texture2D,
    Texture3D,
    Sampler,
    Uniform,
    PushConstant
}

internal enum ResourceAccessModel
{
    Read,
    Write,
    ReadWrite,
    Sample
}

internal sealed record LoweredShaderInstructionModel(
    int SyntaxStart,
    LoweredShaderInstructionKind Kind,
    string Payload,
    LoweredElementwiseAssignmentModel? ElementwiseAssignment = null,
    LoweredElementwiseExpressionAssignmentModel? ElementwiseExpressionAssignment = null,
    LoweredControlFlowConditionModel? ControlFlowCondition = null,
    LoweredAdAnnotationModel? AdAnnotation = null,
    LoweredLocalDeclarationModel? LocalDeclaration = null,
    LoweredLocalAssignmentModel? LocalAssignment = null,
    LoweredCompoundAssignmentModel? CompoundAssignment = null);

internal sealed record LoweredElementwiseAssignmentModel(
    string DestinationResourceName,
    string IndexName,
    LoweredElementwiseAssignmentOperation Operation,
    string LeftOperand,
    LoweredElementwiseAssignmentOperandKind RightOperandKind,
    string RightOperand);

internal enum LoweredElementwiseAssignmentOperation
{
    Copy,
    Add,
    Subtract,
    Multiply,
    Divide
}

internal enum LoweredElementwiseAssignmentOperandKind
{
    None,
    Resource,
    Literal
}

internal sealed record LoweredElementwiseExpressionAssignmentModel(
    string DestinationResourceName,
    string IndexName,
    LoweredElementwiseExpressionNodeModel Expression);

internal sealed record LoweredElementwiseExpressionNodeModel(
    LoweredElementwiseExpressionNodeKind Kind,
    LoweredElementwiseExpressionOperation Operation,
    string ResourceName,
    string IndexName,
    string Literal,
    string SymbolName,
    string TypeName,
    LoweredElementwiseExpressionNodeModel? Left = null,
    LoweredElementwiseExpressionNodeModel? Right = null,
    EquatableArray<LoweredElementwiseExpressionNodeModel> Arguments = default,
    LoweredShaderBuiltinKind BuiltinKind = default);

internal enum LoweredElementwiseExpressionNodeKind
{
    Resource,
    Literal,
    Binary,
    Invocation,
    Comparison,
    PushConstant,
    LocalVariable,
    ShaderBuiltin,
    Ternary,
    Constructor,
    CallableCall,
    TextureSample,
    TextureSampleLevel,
    GpuStructField
}

/// <summary>Identifies a built-in shader variable (gl_GlobalInvocationID, etc.).</summary>
internal enum LoweredShaderBuiltinKind : byte
{
    ThreadIndexX = 1,
    ThreadIndexY = 2,
    ThreadIndexZ = 3,
    LocalIndexX = 4,
    LocalIndexY = 5,
    LocalIndexZ = 6,
    GroupIdX = 7,
    GroupIdY = 8,
    GroupIdZ = 9,
    DispatchSizeX = 10,
    DispatchSizeY = 11,
    DispatchSizeZ = 12,
    GroupSizeX = 13,
    GroupSizeY = 14,
    GroupSizeZ = 15,
    VertexIndex = 16,
    InstanceIndex = 17,
    FragmentCoordX = 18,
    FragmentCoordY = 19,
    FragmentCoordZ = 20,
    FragmentCoordW = 21
}

internal enum LoweredElementwiseExpressionOperation
{
    None,
    Add,
    Subtract,
    Multiply,
    Divide,
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual
}

internal enum LoweredControlFlowRole
{
    IfCondition,
    ForInit,
    ForCondition,
    ForStep,
    WhileCondition,
    DoCondition,
    ForInitDeclaration
}

internal sealed record LoweredControlFlowConditionModel(
    int SyntaxStart,
    LoweredControlFlowRole Role,
    LoweredElementwiseExpressionNodeModel Expression);

internal enum LoweredShaderInstructionKind
{
    ElementwiseAssignment,
    WorkgroupBarrier,
    MemoryBarrier,
    FullBarrier,
    AtomicAdd,
    AtomicSub,
    AtomicMin,
    AtomicMax,
    AtomicAnd,
    AtomicOr,
    AtomicXor,
    AtomicExchange,
    AtomicCompareExchange,
    KnownSymbolInvocation,
    ResourceAccess,
    TextureSample,
    AdParameter,
    AdLoss,
    LocalDeclaration,
    LocalAssignment,
    CompoundAssignment
}

internal sealed record LoweredCompoundAssignmentModel(
    string ResourceName,
    string IndexName,
    LoweredElementwiseAssignmentOperation Operation,
    LoweredElementwiseExpressionNodeModel Value);

internal enum LoweredAdAnnotationRole
{
    Parameter,
    Loss
}

internal sealed record LoweredAdAnnotationModel(
    LoweredAdAnnotationRole Role,
    string Name,
    string ResourceName,
    string TypeName,
    string IndexName,
    LoweredAdSourceKind SourceKind);

internal enum LoweredAdSourceKind
{
    Unknown,
    BufferElement,
    Local
}

internal sealed record LoweredLocalDeclarationModel(
    string VariableName,
    string GlslTypeName,
    LoweredElementwiseExpressionNodeModel? Initializer);

internal sealed record LoweredLocalAssignmentModel(
    string VariableName,
    LoweredElementwiseExpressionNodeModel Value);

internal readonly struct EquatableArray<T> : IEquatable<EquatableArray<T>>
    where T : IEquatable<T>
{
    private readonly T[]? items;

    public EquatableArray(IEnumerable<T> items)
    {
        this.items = items.ToArray();
    }

    public IReadOnlyList<T> Items => items ?? Array.Empty<T>();

    public bool Equals(EquatableArray<T> other) => Items.SequenceEqual(other.Items);

    public override bool Equals(object? obj) => obj is EquatableArray<T> other && Equals(other);

    public override int GetHashCode()
    {
        var hash = 17;
        foreach (var item in Items)
        {
            hash = unchecked((hash * 31) + item.GetHashCode());
        }

        return hash;
    }
}
