namespace Feather.Diagnostics;

public static class DiagnosticDescriptors
{
    public const string DocumentationBaseUrl = "https://feather.dev/docs/diagnostics/";

    public static readonly IReadOnlyList<FeatherDiagnosticInfo> All =
    [
        new("FE0001", "Shader type must be a readonly partial struct."),
        new("FE0002", "Shader type must implement a supported shader interface."),
        new("FE0003", "Shader entry point is invalid."),
        new("FE0004", "Shader constructor parameter type is not supported."),
        new("FE0005", "Captured field type is not supported."),
        new("FE0006", "Unsupported statement in shader body."),
        new("FE0007", "Unsupported expression in shader body."),
        new("FE0008", "Unsupported method call in shader body."),
        new("FE0009", "Unsupported control flow."),
        new("FE0010", "Unsupported generic usage."),
        new("FE0011", "Unsupported allocation in shader body."),
        new("FE0012", "Unsupported exception handling in shader body."),
        new("FE0013", "Unsupported async/await in shader body."),
        new("FE0014", "Unsupported virtual/interface call in shader body."),
        new("FE0015", "Recursive shader function is not supported."),
        new("FE0016", "Resource access violates declared access mode."),
        new("FE0017", "Buffer index type must be int-compatible."),
        new("FE0018", "Texture index type must be int2 or int3."),
        new("FE0019", "Struct layout is not std430-compatible."),
        new("FE0020", "Matrix layout requires explicit Feather matrix type."),
        new("FE0021", "AD parameter type is not differentiable."),
        new("FE0022", "AD loss must be scalar float."),
        new("FE0023", "Graphics shader varying type must be unmanaged and [GpuStruct]."),
        new("FE0024", "Fragment shader output type is not supported."),
        new("FE0025", "ThreadGroupSize is invalid for kernel dimension.")
    ];
}

public sealed record FeatherDiagnosticInfo(string Id, string Message)
{
    public string HelpLink => DiagnosticDescriptors.DocumentationBaseUrl + Id.ToLowerInvariant();
}
