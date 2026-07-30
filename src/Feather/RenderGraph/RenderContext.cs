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
/// <param name="ViewProjection">
/// The clip-space view-projection a rasterizer feeds to <c>gl_Position</c>. On the Blender path this
/// carries the OpenGL-to-Vulkan viewport fixup (Y flip, depth 0..1), so it is intentionally NOT the
/// mutual inverse of <paramref name="InverseViewProjection"/>.
/// </param>
/// <param name="InverseViewProjection">
/// The inverse of the raw camera view-projection, before any rasterizer viewport fixup. A compute
/// pass unprojects clip-space NDC through this to build world-space rays; keeping it in the raw space
/// decouples ray reconstruction from rasterization-only clip conventions.
/// </param>
/// <param name="WorldPosition">The camera eye in world space, used for eye-distance shading.</param>
public readonly record struct RenderCamera(
    float4x4 ViewProjection,
    float4x4 InverseViewProjection,
    float3 WorldPosition)
{
    /// <summary>
    /// Back-compat constructor for callers that only have a view-projection: the inverse is derived
    /// and the eye defaults to the origin. Keeps existing <c>new RenderCamera(vp)</c> call sites working.
    /// </summary>
    public RenderCamera(float4x4 viewProjection)
        : this(viewProjection, viewProjection.Inverse(), new float3(0.0f, 0.0f, 0.0f))
    {
    }
}

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

    /// <summary>
    /// Resolves a host-owned graph texture, allocating it on first use. The host owns the
    /// returned texture so it outlives the pass and downstream passes can read it without a
    /// round trip through system memory; a pass must not dispose it.
    /// </summary>
    GpuTexture2D<TPixel, TValue> GetOrCreateGraphTexture<TPixel, TValue>(
        TextureHandle handle,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => throw new NotSupportedException("This render host does not expose GPU-resident graph textures.");

    /// <summary>
    /// Resolves a GPU-resident texture published by an upstream graph pass.
    /// </summary>
    IGpuTexture2D GetTextureInput(TextureHandle handle)
        => throw new NotSupportedException("This render host does not expose GPU-resident graph textures.");

    /// <summary>
    /// Publishes a GPU-resident texture as a graph output without reading it back. Any pixel
    /// format is accepted; conversion to the display format happens once, when the host takes
    /// the final frame.
    /// </summary>
    void SetTextureOutput(
        TextureHandle handle,
        IGpuTexture2D texture,
        DispatchPath dispatchPath)
        => throw new NotSupportedException("This render host does not accept GPU-resident graph textures.");

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

    /// <summary>
    /// Resolves a host-owned graph texture sized to the current render, allocating it on first
    /// use. Use this for pass outputs and for scratch that has to survive between passes.
    /// </summary>
    /// <remarks>
    /// The host owns the texture, so it must not be disposed by the pass and must not be wrapped
    /// in a <c>using</c> declaration. Textures obtained from <c>GPU.Create*</c> remain
    /// pass-private and do have to be disposed.
    /// </remarks>
    public GpuTexture2D<TPixel, TValue> GetOrCreateTexture<TPixel, TValue>(
        TextureHandle handle,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => Backend.GetOrCreateGraphTexture<TPixel, TValue>(handle, Width, Height, format);

    /// <summary>
    /// Resolves a GPU-resident texture published by an upstream graph pass, keeping the data on
    /// the GPU instead of reading it back.
    /// </summary>
    public IGpuTexture2D GetTextureInput(TextureHandle handle)
        => Backend.GetTextureInput(handle);

    /// <summary>
    /// Publishes a GPU-resident texture as a graph output. Unlike <see cref="SetColorOutput"/>
    /// this performs no readback and accepts any pixel format, so simulation state can stay in a
    /// float target across passes and frames.
    /// </summary>
    public void SetTextureOutput(
        TextureHandle handle,
        IGpuTexture2D texture,
        DispatchPath dispatchPath = DispatchPath.None)
    {
        ArgumentNullException.ThrowIfNull(texture);
        if (texture.Width != Width || texture.Height != Height)
        {
            throw new ArgumentException(
                $"Texture output is {texture.Width}x{texture.Height}; expected {Width}x{Height}.",
                nameof(texture));
        }
        Backend.SetTextureOutput(handle, texture, dispatchPath);
    }

    private IRenderContextBackend Backend
        => backend ?? throw new InvalidOperationException("The render context is not bound to a render host.");
}
