using Microsoft.CodeAnalysis;

namespace Feather.Generators.Diagnostics;

internal static class FeatherDiagnostics
{
    private const string Category = "Feather";
    private const string Docs = "https://feather.dev/docs/diagnostics/";

    public static readonly DiagnosticDescriptor ShaderTypeShape = Create(
        "FE0001",
        "Shader type must be a readonly partial struct",
        "Shader type '{0}' must be a readonly partial struct");

    public static readonly DiagnosticDescriptor ShaderInterface = Create(
        "FE0002",
        "Shader type must implement a supported shader interface",
        "Shader type '{0}' must implement IKernel1D, IKernel2D, IKernel3D, IVertexShader<T>, or IFragmentShader<T>");

    public static readonly DiagnosticDescriptor ComputeExecute = Create(
        "FE0003",
        "Shader entry point is invalid",
        "Shader '{0}' must declare exactly one compatible entry point: either one [Entry] method or the legacy Execute method");

    public static readonly DiagnosticDescriptor UnsupportedConstructorParameter = Create(
        "FE0004",
        "Shader constructor parameter type is not supported",
        "Constructor parameter '{0}' has unsupported shader type '{1}'");

    public static readonly DiagnosticDescriptor UnsupportedCapturedField = Create(
        "FE0005",
        "Captured field type is not supported",
        "Captured field '{0}' has unsupported shader type '{1}'");

    public static readonly DiagnosticDescriptor UnsupportedStatement = Create(
        "FE0006",
        "Unsupported statement in shader body",
        "Statement '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedExpression = Create(
        "FE0007",
        "Unsupported expression in shader body",
        "Expression '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedCall = Create(
        "FE0008",
        "Unsupported method call in shader body",
        "Method call '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedControlFlow = Create(
        "FE0009",
        "Unsupported control flow",
        "Control flow '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedGenericUsage = Create(
        "FE0010",
        "Unsupported generic usage",
        "Generic usage '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedAllocation = Create(
        "FE0011",
        "Unsupported allocation in shader body",
        "Reference type allocation is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedExceptionHandling = Create(
        "FE0012",
        "Unsupported exception handling in shader body",
        "Exception handling is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedAsync = Create(
        "FE0013",
        "Unsupported async/await in shader body",
        "Async and await are not supported in Feather shader code");

    public static readonly DiagnosticDescriptor UnsupportedVirtualCall = Create(
        "FE0014",
        "Unsupported virtual/interface call in shader body",
        "Virtual or interface call '{0}' is not supported in Feather shader code");

    public static readonly DiagnosticDescriptor RecursiveShaderFunction = Create(
        "FE0015",
        "Recursive shader function is not supported",
        "Recursive shader function call '{0}' is not supported");

    public static readonly DiagnosticDescriptor ResourceAccessViolation = Create(
        "FE0016",
        "Resource access violates declared access mode",
        "Resource access for '{0}' violates its declared access mode");

    public static readonly DiagnosticDescriptor BufferIndexInvalid = Create(
        "FE0017",
        "Buffer index type must be int-compatible",
        "Buffer index for '{0}' must be int-compatible");

    public static readonly DiagnosticDescriptor TextureIndexInvalid = Create(
        "FE0018",
        "Texture index type must be int2 or int3",
        "Texture index for '{0}' must be int2 or int3");

    public static readonly DiagnosticDescriptor StructLayoutUnsupported = Create(
        "FE0019",
        "Struct layout is not std430-compatible",
        "Struct layout for '{0}' is not std430-compatible: {1}");

    public static readonly DiagnosticDescriptor MatrixLayoutUnsupported = Create(
        "FE0020",
        "Matrix layout requires explicit Feather matrix type",
        "Matrix field '{0}' must use an explicit Feather matrix type");

    public static readonly DiagnosticDescriptor AutoDiffMarkerUnsupported = Create(
        "FE0021",
        "Automatic differentiation marker is not supported here",
        "AD marker '{0}' is not supported: {1}");

    public static readonly DiagnosticDescriptor AutoDiffSourceUnsupported = Create(
        "FE0022",
        "Automatic differentiation source is not supported",
        "AD marker '{0}' source is not supported: {1}");

    public static readonly DiagnosticDescriptor GraphicsVaryingUnsupported = Create(
        "FE0023",
        "Graphics shader varying type must be unmanaged and [GpuStruct]",
        "Graphics shader varying type '{0}' must be unmanaged and [GpuStruct] with a [Position] float4 member");

    public static readonly DiagnosticDescriptor FragmentOutputUnsupported = Create(
        "FE0024",
        "Fragment shader output type is not supported",
        "Fragment shader output type '{0}' is not supported");

    public static readonly DiagnosticDescriptor ThreadGroupSizeInvalid = Create(
        "FE0025",
        "ThreadGroupSize is invalid for kernel dimension",
        "ThreadGroupSize for '{0}' must contain positive dimensions compatible with the kernel dimension");

    public static readonly DiagnosticDescriptor UnsupportedElementwiseIntrinsic = Create(
        "FE0026",
        "Elementwise expression intrinsic is not supported",
        "Intrinsic call '{0}' is not supported in typed elementwise expression lowering");

    public static readonly DiagnosticDescriptor TypedIrLoweringFailed = Create(
        "FE0027",
        "Typed shader IR lowering failed",
        "Typed shader IR lowering failed for '{0}': {1}");

    public static readonly DiagnosticDescriptor TopLevelLocalUnsupported = Create(
        "FE0028",
        "Top-level local is not supported in shader code",
        "Top-level local '{0}' cannot be referenced from Feather shader code; move it to a static const or static readonly field on a named type");

    public static readonly DiagnosticDescriptor[] All =
    [
        ShaderTypeShape,
        ShaderInterface,
        ComputeExecute,
        UnsupportedConstructorParameter,
        UnsupportedCapturedField,
        UnsupportedStatement,
        UnsupportedExpression,
        UnsupportedCall,
        UnsupportedControlFlow,
        UnsupportedGenericUsage,
        UnsupportedAllocation,
        UnsupportedExceptionHandling,
        UnsupportedAsync,
        UnsupportedVirtualCall,
        RecursiveShaderFunction,
        ResourceAccessViolation,
        BufferIndexInvalid,
        TextureIndexInvalid,
        StructLayoutUnsupported,
        MatrixLayoutUnsupported,
        AutoDiffMarkerUnsupported,
        AutoDiffSourceUnsupported,
        GraphicsVaryingUnsupported,
        FragmentOutputUnsupported,
        ThreadGroupSizeInvalid,
        UnsupportedElementwiseIntrinsic,
        TypedIrLoweringFailed,
        TopLevelLocalUnsupported
    ];

    private static DiagnosticDescriptor Create(string id, string title, string message)
        => new(id, title, message, Category, DiagnosticSeverity.Error, isEnabledByDefault: true, helpLinkUri: Docs + id.ToLowerInvariant());
}
