using System.Text.Json;
using Feather.Math;
using Feather.RenderGraph;

namespace Feather.Blender.RenderHost;

internal sealed class RenderRequest
{
    public const int CurrentSchemaVersion = 1;
    public const int MaximumDimension = 32768;
    public const long MaximumFramePayloadSize = 256L * 1024 * 1024;

    public int SchemaVersion { get; init; }
    public ulong RequestId { get; init; }
    public string GenerationId { get; init; } = "";
    public string ViewId { get; init; } = "";
    public string ScenePath { get; init; } = "";
    public string GraphPath { get; init; } = "";
    public string? ManifestPath { get; init; }
    public string OutputPath { get; init; } = "";
    public int Width { get; init; }
    public int Height { get; init; }
    public string MatrixLayout { get; init; } = "row-major";
    public string ClipSpace { get; init; } = "blender-opengl";
    public float[] ViewProjection { get; init; } = [];

    // Optional camera eye in world space. Added without a schema bump: older hosts ignore the extra
    // field, and this host defaults the eye when it is absent, so old and new add-ons interoperate.
    public float[]? CameraPosition { get; init; }

    // Why this frame is wanted, so a pass can spend less on a viewport preview than on a saved
    // render. It rides on the request rather than the graph because both kinds of render read the
    // same graph document: the difference is which code path in the add-on asked, not how the View
    // is configured.
    //
    // Also added without a schema bump, and absent means interactive -- an add-on old enough to omit
    // it drove the viewport through a host with no separate final path to distinguish.
    public string? Purpose { get; init; }

    public static ResolvedRenderRequest Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var requestPath = Path.GetFullPath(path);
        RenderRequest request;
        try
        {
            using var stream = new FileStream(requestPath, FileMode.Open, FileAccess.Read, FileShare.Read | FileShare.Delete);
            request = JsonSerializer.Deserialize<RenderRequest>(stream, ProtocolJson.Options)
                ?? throw new InvalidDataException("Render request JSON contains null.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"Render request JSON is invalid: {exception.Message}", exception);
        }

        request.Validate();
        var baseDirectory = Path.GetDirectoryName(requestPath)
            ?? throw new InvalidDataException("Render request has no parent directory.");
        return new ResolvedRenderRequest(
            request.RequestId,
            request.GenerationId,
            request.ViewId,
            ResolvePath(baseDirectory, request.ScenePath),
            ResolvePath(baseDirectory, request.GraphPath),
            string.IsNullOrWhiteSpace(request.ManifestPath)
                ? null
                : ResolvePath(baseDirectory, request.ManifestPath),
            ResolvePath(baseDirectory, request.OutputPath),
            request.Width,
            request.Height,
            MatrixProtocol.ConvertViewProjection(request.ViewProjection, request.ClipSpace),
            MatrixProtocol.FromRowMajor(request.ViewProjection).Inverse(),
            MatrixProtocol.ResolveCameraPosition(request.CameraPosition, request.ViewProjection),
            ResolvePurpose(request.Purpose));
    }

    /// <summary>
    /// Maps the request's purpose to the value passes read. An absent field is interactive; an
    /// unrecognised one is rejected rather than defaulted, because silently treating a future
    /// purpose as interactive would quietly render a saved frame at preview quality.
    /// </summary>
    private static RenderPurpose ResolvePurpose(string? purpose) => purpose switch
    {
        null or "" => RenderPurpose.Interactive,
        "INTERACTIVE" => RenderPurpose.Interactive,
        "FINAL" => RenderPurpose.Final,
        _ => throw new InvalidDataException($"Unsupported render request purpose: '{purpose}'.")
    };

    private void Validate()
    {
        if (SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported render request schema version: {SchemaVersion}.");
        }

        if (!Guid.TryParse(GenerationId, out _))
        {
            throw new InvalidDataException("Render request generationId must be a GUID.");
        }
        RequirePath(ScenePath, nameof(ScenePath));
        RequirePath(GraphPath, nameof(GraphPath));
        RequirePath(OutputPath, nameof(OutputPath));
        if (string.IsNullOrWhiteSpace(ViewId))
        {
            throw new InvalidDataException("Render request viewId is required.");
        }
        if (Width is < 1 or > MaximumDimension || Height is < 1 or > MaximumDimension)
        {
            throw new InvalidDataException($"Render dimensions must be between 1 and {MaximumDimension} pixels.");
        }

        var payloadSize = checked((long)Width * Height * 4);
        if (payloadSize > MaximumFramePayloadSize)
        {
            throw new InvalidDataException($"Render payload exceeds the {MaximumFramePayloadSize} byte frame protocol limit.");
        }

        if (!string.Equals(MatrixLayout, "row-major", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"Unsupported request matrix layout: '{MatrixLayout}'.");
        }

        if (ViewProjection is null || ViewProjection.Length != 16 || ViewProjection.Any(value => !float.IsFinite(value)))
        {
            throw new InvalidDataException("viewProjection must contain 16 finite row-major values.");
        }

        if (ClipSpace is not ("blender-opengl" or "vulkan"))
        {
            throw new InvalidDataException($"Unsupported request clip space: '{ClipSpace}'.");
        }

        if (CameraPosition is not null
            && (CameraPosition.Length != 3 || CameraPosition.Any(value => !float.IsFinite(value))))
        {
            throw new InvalidDataException("cameraPosition, when present, must contain 3 finite values.");
        }
    }

    private static void RequirePath(string value, string name)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidDataException($"Render request {name} is required.");
        }
    }

    private static string ResolvePath(string baseDirectory, string value)
        => Path.GetFullPath(Path.IsPathRooted(value) ? value : Path.Combine(baseDirectory, value));
}

internal sealed record ResolvedRenderRequest(
    ulong RequestId,
    string GenerationId,
    string ViewId,
    string ScenePath,
    string GraphPath,
    string? ManifestPath,
    string OutputPath,
    int Width,
    int Height,
    float4x4 ViewProjection,
    float4x4 InverseViewProjection,
    float3 CameraPosition,
    RenderPurpose Purpose);

internal static class MatrixProtocol
{
    public static float4x4 FromRowMajor(IReadOnlyList<float> values)
    {
        if (values.Count != 16)
        {
            throw new InvalidDataException("A 4x4 matrix must contain 16 values.");
        }

        return new float4x4(
            new float4(values[0], values[4], values[8], values[12]),
            new float4(values[1], values[5], values[9], values[13]),
            new float4(values[2], values[6], values[10], values[14]),
            new float4(values[3], values[7], values[11], values[15]));
    }

    /// <summary>
    /// Resolves the camera eye in world space. When the request omits <c>cameraPosition</c>, the eye is
    /// recovered by unprojecting the screen-centre near-plane point through the inverse raw
    /// view-projection, which keeps ray origins valid for older add-ons that do not send it.
    /// </summary>
    public static float3 ResolveCameraPosition(IReadOnlyList<float>? position, IReadOnlyList<float> viewProjection)
    {
        if (position is { Count: 3 })
        {
            return new float3(position[0], position[1], position[2]);
        }

        var inverse = FromRowMajor(viewProjection).Inverse();
        var near = inverse * new float4(0.0f, 0.0f, -1.0f, 1.0f);
        if (near.W == 0.0f || !float.IsFinite(near.W))
        {
            return new float3(0.0f, 0.0f, 0.0f);
        }

        return new float3(near.X / near.W, near.Y / near.W, near.Z / near.W);
    }

    public static float4x4 ConvertViewProjection(IReadOnlyList<float> values, string clipSpace)
    {
        var matrix = FromRowMajor(values);
        if (clipSpace == "vulkan")
        {
            return matrix;
        }

        // Blender exposes an OpenGL-style matrix (Y up, depth -W..W). EasyGPU's
        // Vulkan viewport is Y down with depth 0..W.
        var blenderToVulkan = new float4x4(
            new float4(1.0f, 0.0f, 0.0f, 0.0f),
            new float4(0.0f, -1.0f, 0.0f, 0.0f),
            new float4(0.0f, 0.0f, 0.5f, 0.0f),
            new float4(0.0f, 0.0f, 0.5f, 1.0f));
        return blenderToVulkan * matrix;
    }
}
