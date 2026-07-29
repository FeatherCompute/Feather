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
    public float2 UV;
}

/// <summary>
/// Read-only indexed scene geometry owned by a render host.
/// </summary>
public sealed class SceneGeometry
{
    public SceneGeometry(ReadOnlyMemory<SceneVertex> vertices, ReadOnlyMemory<uint> indices)
        : this(vertices, indices, ReadOnlyMemory<SceneSubmesh>.Empty)
    {
    }

    public SceneGeometry(
        ReadOnlyMemory<SceneVertex> vertices,
        ReadOnlyMemory<uint> indices,
        ReadOnlyMemory<SceneSubmesh> submeshes)
    {
        if (indices.Length % 3 != 0)
        {
            throw new ArgumentException("Scene indices must contain complete triangles.", nameof(indices));
        }
        foreach (var submesh in submeshes.Span)
        {
            if (submesh.FirstIndex < 0 || submesh.IndexCount < 0 || submesh.MaterialIndex < 0 ||
                submesh.FirstIndex % 3 != 0 || submesh.IndexCount % 3 != 0 ||
                submesh.FirstIndex > indices.Length - submesh.IndexCount)
            {
                throw new ArgumentException("Scene submeshes must contain valid triangle index ranges.", nameof(submeshes));
            }
        }

        Vertices = vertices;
        Indices = indices;
        Submeshes = submeshes;
    }

    public ReadOnlyMemory<SceneVertex> Vertices { get; }

    public ReadOnlyMemory<uint> Indices { get; }

    public ReadOnlyMemory<SceneSubmesh> Submeshes { get; }
}

/// <summary>
/// Camera data resolved by a render host for the current render request.
/// </summary>
public readonly record struct RenderCamera(float4x4 ViewProjection);

/// <summary>
/// Host-side storage for the RGBA8 viewport output contract.
/// </summary>
[GpuStruct]
public readonly partial record struct Rgba8(byte R, byte G, byte B, byte A);

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

    SceneMaterialTable GetMaterials(MaterialTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene materials.");

    SceneTextureTable GetTextures(TextureTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene textures.");

    SceneLightTable GetLights(LightTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene lights.");

    RenderTime GetTime(TimeHandle handle)
        => throw new NotSupportedException("This render host does not expose render time.");

    ReadOnlyMemory<Rgba8> GetColorInput(TextureHandle handle);

    ReadOnlyMemory<T> GetBufferInput<T>(BufferHandle<T> handle)
        where T : unmanaged
        => throw new NotSupportedException("This render host does not expose graph buffer inputs.");

    void SetColorOutput(
        TextureHandle handle,
        Rgba8[] pixels,
        DispatchPath dispatchPath);

    void SetBufferOutput<T>(
        BufferHandle<T> handle,
        T[] values,
        DispatchPath dispatchPath)
        where T : unmanaged
        => throw new NotSupportedException("This render host does not accept graph buffer outputs.");

    /// <summary>
    /// Receives the synchronous GPU readback duration for host diagnostics.
    /// </summary>
    void ReportGpuReadback(TimeSpan elapsed)
    {
    }
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

    public SceneMaterialTable GetMaterials(MaterialTableHandle handle)
        => Backend.GetMaterials(handle);

    public SceneTextureTable GetTextures(TextureTableHandle handle)
        => Backend.GetTextures(handle);

    public SceneLightTable GetLights(LightTableHandle handle)
        => Backend.GetLights(handle);

    public RenderTime GetTime(TimeHandle handle)
        => Backend.GetTime(handle);

    /// <summary>
    /// Resolves an RGBA8 texture produced by an upstream graph pass.
    /// The returned memory remains owned by the render host and is valid only for this execution.
    /// </summary>
    public ReadOnlyMemory<Rgba8> GetColorInput(TextureHandle handle)
        => Backend.GetColorInput(handle);

    /// <summary>
    /// Resolves a typed buffer produced by an upstream graph pass. The returned memory remains
    /// owned by the render host and is valid only for this execution.
    /// </summary>
    public ReadOnlyMemory<T> GetBufferInput<T>(BufferHandle<T> handle)
        where T : unmanaged
        => Backend.GetBufferInput(handle);

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
        var readback = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            texture.Read(pixels);
        }
        finally
        {
            readback.Stop();
            Backend.ReportGpuReadback(readback.Elapsed);
        }
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

    /// <summary>
    /// Synchronously reads a Feather buffer and publishes its logical elements as a graph output.
    /// </summary>
    public void SetBufferOutput<T>(
        BufferHandle<T> handle,
        GpuBuffer<T> buffer,
        DispatchPath dispatchPath = DispatchPath.None)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(buffer);
        var values = new T[buffer.Length];
        var readback = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            buffer.Read(values);
        }
        finally
        {
            readback.Stop();
            Backend.ReportGpuReadback(readback.Elapsed);
        }
        Backend.SetBufferOutput(handle, values, dispatchPath);
    }

    /// <summary>
    /// Publishes a typed CPU array as a graph buffer output.
    /// </summary>
    public void SetBufferOutput<T>(
        BufferHandle<T> handle,
        ReadOnlySpan<T> values,
        DispatchPath dispatchPath = DispatchPath.None)
        where T : unmanaged
        => Backend.SetBufferOutput(handle, values.ToArray(), dispatchPath);

    private IRenderContextBackend Backend
        => backend ?? throw new InvalidOperationException("The render context is not bound to a render host.");
}
