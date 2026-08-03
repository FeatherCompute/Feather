using System.Text.RegularExpressions;

namespace Feather.Blender.RenderHost;

/// <summary>A stable, machine-readable RenderHost problem carried in the JSON event stream.</summary>
internal sealed record RenderHostDiagnostic(
    string ErrorCode,
    string Severity,
    string Message,
    string PassGuid,
    string NodeGuid,
    string SourcePath,
    string Action,
    IReadOnlyDictionary<string, string> Context)
{
    private static readonly Regex CodePrefix = new(
        @"^(?<code>[A-Z][A-Z0-9_]+)(?:\s+at\b|:)",
        RegexOptions.CultureInvariant);

    public static RenderHostDiagnostic ForMaterial(
        SceneMaterialMetadata metadata,
        string message,
        string nodeGuid = "")
    {
        var code = ExtractCode(message, "UNSUPPORTED_MATERIAL");
        return new RenderHostDiagnostic(
            code,
            "ERROR",
            message,
            "",
            nodeGuid,
            "",
            "Open the material in the Shader Editor and inspect the highlighted node.",
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["stage"] = "material",
                ["materialId"] = metadata.MaterialId,
                ["materialName"] = metadata.Name,
                ["nodeTree"] = metadata.NodeTree ?? ""
            });
    }

    public static RenderHostDiagnostic FromException(Exception exception)
    {
        var execution = exception as RenderHostExecutionException;
        var cause = execution?.InnerException ?? exception;
        var message = cause.Message;
        var code = cause switch
        {
            OutOfMemoryException => "HOST_OUT_OF_MEMORY",
            MaterialExpressionException material => material.ErrorCode,
            KeyNotFoundException => "RESOURCE_BINDING_ERROR",
            FileNotFoundException => "RESOURCE_BINDING_ERROR",
            _ when message.Contains("shader", StringComparison.OrdinalIgnoreCase) &&
                   message.Contains("compil", StringComparison.OrdinalIgnoreCase)
                => "SHADER_COMPILE_ERROR",
            _ when message.Contains("texture", StringComparison.OrdinalIgnoreCase) &&
                   message.Contains("missing", StringComparison.OrdinalIgnoreCase)
                => "RESOURCE_BINDING_ERROR",
            InvalidDataException => "HOST_PROTOCOL_ERROR",
            _ => "SHADER_RUNTIME_ERROR"
        };
        var action = code switch
        {
            "HOST_OUT_OF_MEMORY" => "Reduce texture or render resolution and retry.",
            "SHADER_COMPILE_ERROR" => "Open the pass source, fix the compiler error, and rebuild.",
            "RESOURCE_BINDING_ERROR" => "Check graph links and material texture resources.",
            _ => "Open Feather Diagnostics and inspect the failing graph or material node."
        };
        return new RenderHostDiagnostic(
            code,
            "ERROR",
            message,
            execution?.PassGuid ?? "",
            cause is MaterialExpressionException materialException
                ? materialException.NodeGuid
                : execution?.NodeGuid ?? "",
            execution?.SourcePath ?? "",
            action,
            new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["stage"] = "host",
                ["exceptionType"] = cause.GetType().Name
            });
    }

    public static string ExtractCode(string message, string fallback)
    {
        var match = CodePrefix.Match(message ?? "");
        return match.Success ? match.Groups["code"].Value : fallback;
    }
}

internal sealed class RenderHostExecutionException : Exception
{
    public RenderHostExecutionException(
        Exception innerException,
        string passGuid,
        string nodeGuid,
        string sourcePath = "")
        : base(innerException.Message, innerException)
    {
        PassGuid = passGuid;
        NodeGuid = nodeGuid;
        SourcePath = sourcePath;
    }

    public string PassGuid { get; }
    public string NodeGuid { get; }
    public string SourcePath { get; }
}

internal sealed class MaterialExpressionException : Exception
{
    public MaterialExpressionException(string message, string nodeGuid = "")
        : base(message)
    {
        ErrorCode = RenderHostDiagnostic.ExtractCode(message, "MATERIAL_EXPRESSION_UNSUPPORTED");
        NodeGuid = nodeGuid;
    }

    public string ErrorCode { get; }
    public string NodeGuid { get; }
}
