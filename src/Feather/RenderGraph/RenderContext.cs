using Feather.Math;
using Feather.NN;
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
/// One named scene object: its placement in the world and the geometry belonging to it alone.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="Geometry"/> carries world-space positions, matching <see cref="SceneGeometry"/> from the
/// scene as a whole, so a pass can draw an object without applying anything first.
/// <see cref="ModelMatrix"/> is supplied on top of that for the passes that need the placement itself
/// -- a simulation whose domain is the object's local box, or an effect that wants object space back
/// out of world space. Handing over both is deliberate: one is derivable from the other, but that is
/// exactly the matrix bookkeeping a pass author should not be repeating.
/// </para>
/// <para>
/// A host asked for an object the scene does not contain reports it through <see cref="Exists"/>
/// rather than throwing. A named object legitimately disappears while a graph is being edited --
/// renamed, hidden, deleted -- and refusing the whole frame would break the graph at exactly the
/// moment the user is changing it.
/// </para>
/// </remarks>
public sealed class SceneObject
{
    public SceneObject(string name, float4x4 modelMatrix, SceneGeometry geometry, bool exists = true)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
        ModelMatrix = modelMatrix;
        Geometry = geometry ?? throw new ArgumentNullException(nameof(geometry));
        Exists = exists;
    }

    /// <summary>The host-side name this object was selected by.</summary>
    public string Name { get; }

    /// <summary>Whether the scene actually contains an object with that name.</summary>
    public bool Exists { get; }

    /// <summary>The object-to-world transform, column-major.</summary>
    public float4x4 ModelMatrix { get; }

    /// <summary>
    /// This object's geometry alone, in world space. Empty when <see cref="Exists"/> is false.
    /// </summary>
    /// <remarks>
    /// Narrowed through the indices, not the vertices: <see cref="SceneGeometry.Indices"/> covers only
    /// this object's triangles while <see cref="SceneGeometry.Vertices"/> may remain the whole scene's
    /// buffer, which is what lets a host hand over an object without copying anything. Walk the object
    /// through its indices; treating the vertex span as this object's own would read its neighbours.
    /// </remarks>
    public SceneGeometry Geometry { get; }
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

    /// <summary>
    /// Returns this camera with the Y and Z world axes exchanged.
    /// </summary>
    /// <remarks>
    /// Converts between a Z-up world and a Y-up one. Which of those a camera arrives in is a property
    /// of the host, so the decision to call this belongs to whoever knows that; the swap itself is
    /// plain geometry and is its own inverse. Only <see cref="InverseViewProjection"/> and
    /// <see cref="WorldPosition"/> move, because those are what a compute pass builds rays from --
    /// <see cref="ViewProjection"/> is rasterizer-only and already documented as not their mutual
    /// inverse.
    /// </remarks>
    public RenderCamera SwapUpAxis()
        => new(
            ViewProjection,
            SwapYZ * InverseViewProjection,
            new float3(WorldPosition.X, WorldPosition.Z, WorldPosition.Y));

    /// <summary>
    /// Exchanges the Y and Z axes. Columns, because <see cref="float4x4"/> is column-major.
    /// </summary>
    private static float4x4 SwapYZ { get; } = new(
        new float4(1.0f, 0.0f, 0.0f, 0.0f),
        new float4(0.0f, 0.0f, 1.0f, 0.0f),
        new float4(0.0f, 1.0f, 0.0f, 0.0f),
        new float4(0.0f, 0.0f, 0.0f, 1.0f));
}

/// <summary>
/// Which world axis a consumer of a camera treats as up.
/// </summary>
/// <remarks>
/// Exists so the choice can be made in a render graph rather than inside an effect. A procedural
/// shader is conventionally written Y-up while several hosts model the world Z-up, and porting every
/// shader is the wrong trade when the conversion is one matrix.
/// </remarks>
public enum CameraUpAxis
{
    /// <summary>Y is up, the convention procedural shaders are conventionally written in.</summary>
    Y,

    /// <summary>Z is up, matching Blender's world axes.</summary>
    Z
}

/// <summary>
/// Host-side storage for the RGBA8 viewport output contract.
/// </summary>
[GpuStruct]
public readonly partial record struct Rgba8(byte R, byte G, byte B, byte A);

/// <summary>
/// Why the host is asking for this frame.
/// </summary>
/// <remarks>
/// <para>
/// A pass usually wants to do less work for an interactive frame than for the one that gets saved:
/// fewer samples, fewer march steps, a coarser simulation. Without this it cannot tell the two apart,
/// so every author has to pick one quality level and live with it being either too slow to navigate
/// or too rough to keep.
/// </para>
/// <para>
/// Deliberately coarse. The values name the situation, not the settings that produced it, so a pass
/// that branches on <see cref="Interactive"/> keeps working when the host gains new execution modes.
/// Read the concrete budgets from <see cref="RenderContext.SampleCount"/> and the pass's own
/// parameters.
/// </para>
/// </remarks>
public enum RenderPurpose
{
    /// <summary>
    /// A frame drawn into the viewport while the user is working. Latency matters more than quality:
    /// another one is coming as soon as the camera moves.
    /// </summary>
    Interactive = 0,

    /// <summary>
    /// The frame the user asked for and will keep, from F12 or from a render job. Quality matters
    /// more than latency, and nothing is going to supersede it.
    /// </summary>
    Final = 1
}

/// <summary>
/// Host-independent services used by <see cref="RenderContext"/>. Render-host authors implement
/// this interface; project passes normally consume it only through <see cref="RenderContext"/>.
/// </summary>
public interface IRenderContextBackend
{
    int Width { get; }

    int Height { get; }

    SampleCount SampleCount { get; }

    /// <summary>
    /// Why this frame is being rendered. Defaults to <see cref="RenderPurpose.Interactive"/> so that
    /// a host predating this member keeps compiling, and so that a pass which trims work for
    /// interactive frames stays responsive against one rather than silently rendering at full
    /// quality on every viewport redraw.
    /// </summary>
    RenderPurpose Purpose => RenderPurpose.Interactive;

    /// <summary>Whether the current View is executing progressive iterations.</summary>
    bool IsProgressive => false;

    /// <summary>Zero-based progressive iteration index before the current pass executes.</summary>
    long Iteration => 0;

    /// <summary>Samples committed by completed iterations before the current pass executes.</summary>
    long AccumulatedSamples => 0;

    /// <summary>Samples the scheduler assigns to this iteration.</summary>
    int SamplesPerIteration => 1;

    /// <summary>Whether temporal state was reset while preparing this iteration.</summary>
    bool HistoryReset => false;

    /// <summary>Monotonic reset generation for the current View.</summary>
    long ResetCount => 0;

    /// <summary>The absolute root of the project currently being rendered.</summary>
    string ProjectRoot
        => throw new NotSupportedException("This render host does not expose a project root.");

    /// <summary>Returns host-cached inference weights for a project-relative checkpoint.</summary>
    InferenceWeights GetOrLoadWeights(string projectRelativePath)
        => throw new NotSupportedException("This render host does not expose inference weights.");

    /// <summary>
    /// Returns an assembly-generation-owned resource associated with the current scene and graph.
    /// The host disposes cached values when either identity changes or the pass assembly unloads.
    /// </summary>
    T GetOrCreateSceneResource<T>(string identity, Func<T> factory)
        where T : class, IDisposable
        => throw new NotSupportedException("This render host does not expose retained scene resources.");

    /// <summary>
    /// Returns a resource owned by the current pass-assembly generation. The resource survives graph
    /// and View switches and is disposed when that assembly generation unloads. Callers must not
    /// dispose returned resources. An identity must always refer to the same resource type within one
    /// generation; use a pass-qualified, versioned identity such as
    /// <c>"MyPass.Pipelines.v1"</c>.
    /// </summary>
    /// <param name="identity">The stable, assembly-wide identity of the retained resource.</param>
    /// <param name="factory">Creates the resource when <paramref name="identity"/> is first requested.</param>
    /// <typeparam name="T">A disposable resource owner retained by the render host.</typeparam>
    /// <returns>The existing resource for <paramref name="identity"/>, or the newly created value.</returns>
    /// <exception cref="InvalidOperationException">
    /// The same identity was already used with another resource type, or the factory returned null.
    /// </exception>
    T GetOrCreateAssemblyResource<T>(string identity, Func<T> factory)
        where T : class, IDisposable
        => throw new NotSupportedException("This render host does not expose retained assembly resources.");

    /// <summary>
    /// Resolves a host-owned texture private to the current pass node and View. The allocation survives
    /// camera-only executions and is released when the graph or pass assembly changes.
    /// </summary>
    GpuTexture2D<TPixel, TValue> GetOrCreatePassTexture<TPixel, TValue>(
        string identity,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => throw new NotSupportedException("This render host does not expose retained pass textures.");

    GpuBuffer<T> GetDataBuffer<T>(DataHandle handle, string resourceGuid)
        where T : unmanaged
        => throw new NotSupportedException("This render host does not expose Data Manager buffers.");

    GpuTexture2D<TPixel, TValue> GetDataTexture2D<TPixel, TValue>(
        DataHandle handle,
        string resourceGuid,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => throw new NotSupportedException("This render host does not expose Data Manager textures.");

    GpuTexture2D<TPixel, TValue> GetDataTexture2D<TPixel, TValue>(
        DataHandle handle,
        string resourceGuid,
        PixelFormat format,
        DataFrameVersion version)
        where TPixel : unmanaged
        where TValue : unmanaged
        => version == DataFrameVersion.Current
            ? GetDataTexture2D<TPixel, TValue>(handle, resourceGuid, format)
            : throw new NotSupportedException(
                "This render host does not expose frame-versioned Data Manager textures.");

    SceneGeometry GetSceneGeometry(SceneGeometryHandle handle);

    RenderScene GetScene(SceneHandle handle)
        => throw new NotSupportedException("This render host does not expose evaluated scenes.");

    RenderCamera GetCamera(CameraHandle handle);

    SceneMaterialTable GetMaterials(MaterialTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene materials.");

    SceneTextureTable GetTextures(TextureTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene textures.");

    SceneLightTable GetLights(LightTableHandle handle)
        => throw new NotSupportedException("This render host does not expose scene lights.");

    RenderTime GetTime(TimeHandle handle)
        => throw new NotSupportedException("This render host does not expose render time.");

    SceneObject GetSceneObject(SceneObjectHandle handle)
        => throw new NotSupportedException("This render host does not expose individual scene objects.");

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
    /// Resolves a host-owned render target associated with a graph texture output. The host owns
    /// the returned target so a raster pass must not dispose it.
    /// </summary>
    GpuTexture2D<TPixel, TValue> GetOrCreateGraphRenderTarget<TPixel, TValue>(
        TextureHandle handle,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => throw new NotSupportedException("This render host does not expose reusable render targets.");

    /// <summary>
    /// Resolves the host-owned depth target associated with a graph texture output. The host owns
    /// the returned target so a raster pass must not dispose it.
    /// </summary>
    GpuTexture2D<float, float> GetOrCreateGraphDepthTarget(
        TextureHandle handle,
        int width,
        int height)
        => throw new NotSupportedException("This render host does not expose reusable depth targets.");

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

    /// <summary>
    /// Why the host wants this frame. Branch on this to spend less on a frame the user is navigating
    /// through than on the one they are going to keep.
    /// </summary>
    public RenderPurpose Purpose => Backend.Purpose;

    /// <summary>
    /// Whether this frame is a viewport preview. Shorthand for
    /// <c>Purpose == RenderPurpose.Interactive</c>, which is the test nearly every pass wants.
    /// </summary>
    public bool IsInteractive => Backend.Purpose == RenderPurpose.Interactive;

    /// <summary>Whether the current View is executing progressive iterations.</summary>
    public bool IsProgressive => Backend.IsProgressive;

    /// <summary>Zero-based scheduler iteration before this execution.</summary>
    public long Iteration => Backend.Iteration;

    /// <summary>Samples committed before this execution.</summary>
    public long AccumulatedSamples => Backend.AccumulatedSamples;

    /// <summary>Samples assigned to this scheduler iteration.</summary>
    public int SamplesPerIteration => Backend.SamplesPerIteration;

    /// <summary>Whether the scheduler reset temporal state before this execution.</summary>
    public bool HistoryReset => Backend.HistoryReset;

    /// <summary>Monotonic reset generation for the current View.</summary>
    public long ResetCount => Backend.ResetCount;

    /// <summary>Gets the absolute root of the project currently being rendered.</summary>
    public string ProjectRoot => Backend.ProjectRoot;

    /// <summary>
    /// Resolves a project-relative checkpoint through the host-owned inference cache. The host owns
    /// the returned weights; a pass must not dispose them.
    /// </summary>
    public InferenceWeights GetOrLoadWeights(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        return Backend.GetOrLoadWeights(checkpointPath);
    }

    /// <summary>
    /// Returns a host-owned resource retained while the scene, graph, and pass build identities match.
    /// The caller must not dispose the returned value.
    /// </summary>
    public T GetOrCreateSceneResource<T>(string identity, Func<T> factory)
        where T : class, IDisposable
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identity);
        ArgumentNullException.ThrowIfNull(factory);
        return Backend.GetOrCreateSceneResource(identity, factory);
    }

    /// <summary>
    /// Returns a host-owned resource retained for the current pass-assembly generation. Use this for
    /// compiled shader pipelines and similar heavy objects whose identity does not depend on the
    /// active graph, View, or scene. The value survives graph and View switches, and the host disposes
    /// it when the assembly generation unloads during reload or shutdown. The caller must not dispose
    /// the returned value. An identity must always refer to the same resource type within one
    /// generation; use a pass-qualified, versioned identity such as
    /// <c>"MyPass.Pipelines.v1"</c>.
    /// </summary>
    /// <param name="identity">The stable, assembly-wide identity of the retained resource.</param>
    /// <param name="factory">Creates the resource when <paramref name="identity"/> is first requested.</param>
    /// <typeparam name="T">A disposable resource owner retained by the render host.</typeparam>
    /// <returns>The existing resource for <paramref name="identity"/>, or the newly created value.</returns>
    /// <exception cref="ArgumentException"><paramref name="identity"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentNullException"><paramref name="factory"/> is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// The same identity was already used with another resource type, or the factory returned null.
    /// </exception>
    public T GetOrCreateAssemblyResource<T>(string identity, Func<T> factory)
        where T : class, IDisposable
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identity);
        ArgumentNullException.ThrowIfNull(factory);
        return Backend.GetOrCreateAssemblyResource(identity, factory);
    }

    /// <summary>
    /// Resolves a host-owned texture private to this pass node and View. Use this for temporal state or
    /// reusable scratch whose lifetime must span pass instances without becoming a graph socket.
    /// </summary>
    public GpuTexture2D<TPixel, TValue> GetOrCreatePassTexture<TPixel, TValue>(
        string identity,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(identity);
        if (width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width));
        }
        if (height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(height));
        }
        return Backend.GetOrCreatePassTexture<TPixel, TValue>(identity, width, height, format);
    }

    /// <summary>
    /// Resolves one typed buffer declared inside a Data Type. The render host validates the stable
    /// resource identity, access mode, element count, exact Type layout, and initializer before it
    /// returns the host-owned allocation. The caller must not dispose the buffer.
    /// </summary>
    public GpuBuffer<T> GetDataBuffer<T>(DataHandle handle, string resourceGuid)
        where T : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resourceGuid);
        return Backend.GetDataBuffer<T>(handle, resourceGuid);
    }

    /// <summary>
    /// Resolves one 2D texture declared inside a Data Type. Width, height, lifetime, and update
    /// policy come from the Data manifest rather than the current viewport or the pass source.
    /// </summary>
    public GpuTexture2D<TPixel, TValue> GetDataTexture2D<TPixel, TValue>(
        DataHandle handle,
        string resourceGuid,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resourceGuid);
        return Backend.GetDataTexture2D<TPixel, TValue>(handle, resourceGuid, format);
    }

    /// <summary>
    /// Resolves Current or Next for one frame-versioned logical Data texture. A successful graph
    /// frame atomically promotes Next to Current; an aborted frame leaves Current unchanged.
    /// </summary>
    public GpuTexture2D<TPixel, TValue> GetDataTexture2D<TPixel, TValue>(
        DataHandle handle,
        string resourceGuid,
        PixelFormat format,
        DataFrameVersion version)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(resourceGuid);
        if (!Enum.IsDefined(version))
        {
            throw new ArgumentOutOfRangeException(nameof(version));
        }
        return Backend.GetDataTexture2D<TPixel, TValue>(
            handle,
            resourceGuid,
            format,
            version);
    }

    public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
        => Backend.GetSceneGeometry(handle);

    /// <summary>
    /// Resolves the immutable heterogeneous Scene selected by the graph. Representation lowering is
    /// not imposed by Studio: a renderer may execute several native geometry domains directly, and
    /// an explicit adapter pass remains optional.
    /// </summary>
    public RenderScene GetScene(SceneHandle handle)
        => Backend.GetScene(handle);

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
    /// Resolves one scene object the graph selected by name.
    /// </summary>
    public SceneObject GetSceneObject(SceneObjectHandle handle)
        => Backend.GetSceneObject(handle);

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
    /// Resolves a host-owned render target associated with <paramref name="handle" />, allocating
    /// it on first use. The target follows the render size and must not be disposed by the pass.
    /// </summary>
    public GpuTexture2D<TPixel, TValue> GetOrCreateRenderTarget<TPixel, TValue>(
        TextureHandle handle,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => Backend.GetOrCreateGraphRenderTarget<TPixel, TValue>(handle, Width, Height, format);

    /// <summary>
    /// Resolves the host-owned 32-bit depth target associated with <paramref name="handle" />,
    /// allocating it on first use. The target follows the render size and must not be disposed by
    /// the pass.
    /// </summary>
    public GpuTexture2D<float, float> GetOrCreateDepthTarget(TextureHandle handle)
        => Backend.GetOrCreateGraphDepthTarget(handle, Width, Height);

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
