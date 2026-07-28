using Feather.Math;
using Feather.Resources;

namespace Feather.RenderGraph;

/// <summary>
/// Canonical evaluated scene vertex supplied to public Feather raster passes.
/// Positions and normals are in world space.
/// </summary>
public struct SceneVertex
{
    public float3 Position;
    public float3 Normal;
}

/// <summary>
/// Read-only indexed scene geometry owned by a render host.
/// </summary>
public sealed class SceneGeometry
{
    public SceneGeometry(ReadOnlyMemory<SceneVertex> vertices, ReadOnlyMemory<uint> indices)
    {
        if (indices.Length % 3 != 0)
        {
            throw new ArgumentException("Scene indices must contain complete triangles.", nameof(indices));
        }

        Vertices = vertices;
        Indices = indices;
    }

    public ReadOnlyMemory<SceneVertex> Vertices { get; }

    public ReadOnlyMemory<uint> Indices { get; }
}

/// <summary>
/// Camera data resolved by a render host for the current render request.
/// </summary>
public readonly record struct RenderCamera(float4x4 ViewProjection);

/// <summary>
/// Host-side storage for the RGBA8 viewport output contract.
/// </summary>
public readonly record struct Rgba8(byte R, byte G, byte B, byte A);

/// <summary>
/// Host-independent services used by <see cref="RenderContext"/>. Render-host authors implement
/// this interface; project passes normally consume it only through <see cref="RenderContext"/>.
/// </summary>
public interface IRenderContextBackend
{
    int Width { get; }

    int Height { get; }

    SampleCount SampleCount { get; }

    SceneGeometry GetSceneGeometry(SceneGeometryHandle handle);

    RenderCamera GetCamera(CameraHandle handle);

    void SetColorOutput(
        TextureHandle handle,
        Rgba8[] pixels,
        DispatchPath dispatchPath);
}

/// <summary>
/// Public execution context supplied to a render-graph pass. The host owns scene resources and
/// the pass owns the Feather GPU work it records, including buffers, textures, and pipelines.
/// </summary>
public sealed class RenderContext
{
    private readonly IRenderContextBackend? backend;

    /// <summary>
    /// Creates an unbound context for compatibility with code that only stores the contract.
    /// Resource operations on an unbound context throw <see cref="InvalidOperationException"/>.
    /// </summary>
    public RenderContext()
    {
    }

    /// <summary>
    /// Creates a context backed by host-provided resources.
    /// </summary>
    public RenderContext(IRenderContextBackend backend)
    {
        this.backend = backend ?? throw new ArgumentNullException(nameof(backend));
        if (backend.Width <= 0 || backend.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(backend), "Render dimensions must be positive.");
        }
        if (backend.SampleCount is not (
                global::Feather.SampleCount.X1 or
                global::Feather.SampleCount.X2 or
                global::Feather.SampleCount.X4 or
                global::Feather.SampleCount.X8 or
                global::Feather.SampleCount.X16))
        {
            throw new ArgumentOutOfRangeException(nameof(backend), "Render sample count is unsupported.");
        }
    }

    public int Width => Backend.Width;

    public int Height => Backend.Height;

    public SampleCount SampleCount => Backend.SampleCount;

    public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
        => Backend.GetSceneGeometry(handle);

    public RenderCamera GetCamera(CameraHandle handle)
        => Backend.GetCamera(handle);

    /// <summary>
    /// Synchronously reads an RGBA8 Feather texture and publishes it as a graph output.
    /// </summary>
    public void SetColorOutput<TValue>(
        TextureHandle handle,
        GpuTexture2D<Rgba8, TValue> texture,
        DispatchPath dispatchPath = DispatchPath.None)
        where TValue : unmanaged
    {
        ArgumentNullException.ThrowIfNull(texture);
        if (texture.Width != Width || texture.Height != Height)
        {
            throw new ArgumentException(
                $"Color output is {texture.Width}x{texture.Height}; expected {Width}x{Height}.",
                nameof(texture));
        }
        if (texture.Format != PixelFormat.Rgba8)
        {
            throw new ArgumentException("Color output must use PixelFormat.Rgba8.", nameof(texture));
        }

        var pixels = new Rgba8[checked(Width * Height)];
        texture.Read(pixels);
        Backend.SetColorOutput(handle, pixels, dispatchPath);
    }

    /// <summary>
    /// Publishes a tightly packed CPU RGBA8 image. This also supports software rendering passes.
    /// </summary>
    public void SetColorOutput(
        TextureHandle handle,
        ReadOnlySpan<Rgba8> pixels,
        DispatchPath dispatchPath = DispatchPath.None)
    {
        if (pixels.Length != checked(Width * Height))
        {
            throw new ArgumentException(
                $"Color output contains {pixels.Length} pixels; expected {Width * Height}.",
                nameof(pixels));
        }
        Backend.SetColorOutput(handle, pixels.ToArray(), dispatchPath);
    }

    private IRenderContextBackend Backend
        => backend ?? throw new InvalidOperationException("The render context is not bound to a render host.");
}
