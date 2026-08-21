using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather.Graphics;

public sealed class GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> : IDisposable, ILoadedGraphicsShaderInspection
    where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
    where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
    where TVaryings : unmanaged
{
    private bool disposed;
    private readonly GpuContext context;
    private readonly object dispatchGate = new();

    private GpuGraphicsPipeline(GpuContext context, FeGraphicsPipelineHandle handle, GraphicsPipelineDesc desc)
    {
        this.context = context;
        Handle = handle;
        Desc = desc;
    }

    internal FeGraphicsPipelineHandle Handle { get; }
    public GraphicsPipelineDesc Desc { get; }

    /// <summary>
    /// Gets the native route used by this graphics pipeline's most recent draw.
    /// </summary>
    public DispatchPath LastDispatchPath
    {
        get
        {
            using var operation = context.EnterOperation();
            lock (dispatchGate)
            {
                ThrowIfDisposed();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_get_last_dispatch_path(Handle, out var path));
                return (DispatchPath)path;
            }
        }
    }

    internal static GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> Create(GpuContext context, GraphicsPipelineDesc desc)
    {
        using var operation = context.EnterOperation();
        var normalized = NormalizeDesc(desc);
        var ir = TVertexShader.IR;
        var vertexIr = TVertexShader.VertexIR;
        var fragmentIr = TVertexShader.FragmentIR;
        Span<FeGraphicsColorBlendAttachmentDesc> blendAttachments = stackalloc FeGraphicsColorBlendAttachmentDesc[8];
        for (var i = 0; i < blendAttachments.Length; i++)
        {
            blendAttachments[i] = ToNative(normalized.Blend);
        }
        if (normalized.BlendAttachments is not null)
        {
            for (var i = 0; i < normalized.BlendAttachments.Count; i++)
            {
                blendAttachments[i] = ToNative(normalized.BlendAttachments[i]);
            }
        }

        unsafe
        {
            fixed (byte* irPtr = ir)
            fixed (byte* vertexIrPtr = vertexIr)
            fixed (byte* fragmentIrPtr = fragmentIr)
            {
                var createDesc = new FeGraphicsPipelineCreateDesc(
                    (IntPtr)irPtr,
                    (ulong)ir.Length,
                    (IntPtr)vertexIrPtr,
                    (ulong)vertexIr.Length,
                    (IntPtr)fragmentIrPtr,
                    (ulong)fragmentIr.Length,
                    (uint)normalized.Topology,
                    (uint)normalized.SampleCount,
                    normalized.ColorAttachmentCount,
                    normalized.DepthStencil.DepthTest ? 1u : 0u,
                    normalized.DepthStencil.DepthWrite ? 1u : 0u,
                    (uint)normalized.DepthStencil.DepthCompare,
                    normalized.DepthStencil.StencilTest ? 1u : 0u,
                    ToNative(normalized.DepthStencil.Front),
                    ToNative(normalized.DepthStencil.Back),
                    normalized.DepthStencil.StencilReadMask,
                    normalized.DepthStencil.StencilWriteMask,
                    normalized.DepthStencil.StencilReference,
                    normalized.Blend.Enabled ? 1u : 0u,
                    (uint)normalized.Blend.SrcColor,
                    (uint)normalized.Blend.DstColor,
                    (uint)normalized.Blend.ColorOp,
                    (uint)normalized.Blend.SrcAlpha,
                    (uint)normalized.Blend.DstAlpha,
                    (uint)normalized.Blend.AlphaOp,
                    (uint)normalized.Blend.WriteMask,
                    normalized.ColorAttachmentCount,
                    blendAttachments[0],
                    blendAttachments[1],
                    blendAttachments[2],
                    blendAttachments[3],
                    blendAttachments[4],
                    blendAttachments[5],
                    blendAttachments[6],
                    blendAttachments[7],
                    (uint)normalized.Raster.CullMode,
                    (uint)normalized.Raster.FrontFace,
                    (uint)normalized.Raster.PolygonMode,
                    normalized.Raster.DepthClamp ? 1u : 0u,
                    normalized.DebugName ?? typeof(TVertexShader).Name + "+" + typeof(TFragmentShader).Name);
                NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_create_from_ir(context.Handle, in createDesc, out var handle));
                var pipeline = new GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>(context, handle, normalized);
                try
                {
                    context.RegisterGraphicsPipeline(pipeline);
                    return pipeline;
                }
                catch
                {
                    pipeline.Dispose();
                    throw;
                }
            }
        }
    }

    private static GraphicsPipelineDesc NormalizeDesc(GraphicsPipelineDesc desc)
    {
        var colorAttachmentCount = desc.ColorAttachmentCount == 0 ? 1u : desc.ColorAttachmentCount;
        if (colorAttachmentCount > 8)
        {
            throw new ArgumentOutOfRangeException(nameof(desc), "Graphics pipelines support between one and eight color attachments.");
        }
        if (desc.BlendAttachments is not null && desc.BlendAttachments.Count != colorAttachmentCount)
        {
            throw new ArgumentException("BlendAttachments count must match ColorAttachmentCount.", nameof(desc));
        }

        var depthStencil = desc.DepthStencil == default
            ? DepthStencilState.Default
            : desc.DepthStencil;
        if (desc.DepthTest || desc.DepthWrite)
        {
            depthStencil = depthStencil with
            {
                DepthTest = depthStencil.DepthTest || desc.DepthTest,
                DepthWrite = depthStencil.DepthWrite || desc.DepthWrite
            };
        }

        return desc with
        {
            Topology = desc.Topology,
            SampleCount = desc.SampleCount == 0 ? SampleCount.X1 : desc.SampleCount,
            DepthStencil = depthStencil,
            Blend = desc.Blend == default ? BlendState.Opaque : desc.Blend,
            BlendAttachments = desc.BlendAttachments,
            Raster = desc.Raster == default ? RasterState.Default : desc.Raster,
            ColorAttachmentCount = colorAttachmentCount
        };
    }

    private static FeGraphicsStencilFaceDesc ToNative(StencilFaceState state)
        => new((uint)state.FailOp, (uint)state.PassOp, (uint)state.DepthFailOp, (uint)state.Compare);

    private static FeGraphicsColorBlendAttachmentDesc ToNative(BlendState state)
        => new(
            state.Enabled ? 1u : 0u,
            (uint)state.SrcColor,
            (uint)state.DstColor,
            (uint)state.ColorOp,
            (uint)state.SrcAlpha,
            (uint)state.DstAlpha,
            (uint)state.AlphaOp,
            (uint)state.WriteMask);

    public void Draw<TPixel, TValue>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        uint vertexCount,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
        => Draw(vertexShader, fragmentShader, target, vertexCount, default, wait);

    public void Draw<TPixel, TValue>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        uint vertexCount,
        GraphicsDrawDesc drawDesc,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ThrowIfDisposed();
        DrawCore(vertexShader, fragmentShader, [target], null, vertexCount, drawDesc, wait);
    }

    /// <summary>
    /// Issues a non-indexed draw against the supplied color and depth targets.
    /// </summary>
    public void Draw<TPixel, TValue, TDepthPixel, TDepthValue>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        GpuTexture2D<TDepthPixel, TDepthValue> depthTarget,
        uint vertexCount,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
        where TDepthPixel : unmanaged
        where TDepthValue : unmanaged
        => Draw(vertexShader, fragmentShader, target, depthTarget, vertexCount, default, wait);

    public void Draw<TPixel, TValue, TDepthPixel, TDepthValue>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        GpuTexture2D<TDepthPixel, TDepthValue> depthTarget,
        uint vertexCount,
        GraphicsDrawDesc drawDesc,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
        where TDepthPixel : unmanaged
        where TDepthValue : unmanaged
    {
        ThrowIfDisposed();
        DrawCore(vertexShader, fragmentShader, [target], depthTarget, vertexCount, drawDesc, wait);
    }

    public void Draw(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        uint vertexCount,
        GraphicsDrawDesc drawDesc = default,
        bool wait = true)
    {
        ThrowIfDisposed();
        DrawCore(vertexShader, fragmentShader, targets, null, vertexCount, drawDesc, wait);
    }

    public void Draw(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D depthTarget,
        uint vertexCount,
        GraphicsDrawDesc drawDesc = default,
        bool wait = true)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(depthTarget);
        DrawCore(vertexShader, fragmentShader, targets, depthTarget, vertexCount, drawDesc, wait);
    }

    private void DrawCore(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D? depthTarget,
        uint vertexCount,
        GraphicsDrawDesc drawDesc,
        bool wait)
    {
        using var operation = context.EnterOperation();
        lock (context.QueueGate)
        {
            var effectiveWait = wait && context.ActiveTimestampRecorder is null;
            var leases = ExecuteDraw(
                vertexShader,
                fragmentShader,
                targets,
                depthTarget,
                vertexCount,
                FeBufferHandle.Null,
                indexed: false,
                drawDesc,
                effectiveWait);
            CompleteImmediateDraw(leases, effectiveWait);
        }
    }

    /// <summary>
    /// Issues an indexed draw after binding the supplied index buffer.
    /// </summary>
    public void DrawIndexed<TPixel, TValue, TIndex>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        GpuBuffer<TIndex> indices,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
        where TIndex : unmanaged
    {
        ThrowIfDisposed();
        var indexBuffer = GetValidatedIndexBuffer(indices);
        DrawIndexedCore(vertexShader, fragmentShader, [target], null, indexBuffer, (uint)indices.Length, default, wait);
    }

    /// <summary>
    /// Issues an indexed draw against the supplied color and depth targets after binding the supplied index buffer.
    /// </summary>
    public void DrawIndexed<TPixel, TValue, TDepthPixel, TDepthValue, TIndex>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        GpuTexture2D<TPixel, TValue> target,
        GpuTexture2D<TDepthPixel, TDepthValue> depthTarget,
        GpuBuffer<TIndex> indices,
        bool wait = true)
        where TPixel : unmanaged
        where TValue : unmanaged
        where TDepthPixel : unmanaged
        where TDepthValue : unmanaged
        where TIndex : unmanaged
    {
        ThrowIfDisposed();
        var indexBuffer = GetValidatedIndexBuffer(indices);
        DrawIndexedCore(vertexShader, fragmentShader, [target], depthTarget, indexBuffer, (uint)indices.Length, default, wait);
    }

    public void DrawIndexed<TIndex>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc = default,
        bool wait = true)
        where TIndex : unmanaged
    {
        ThrowIfDisposed();
        var indexBuffer = GetValidatedIndexBuffer(indices);
        DrawIndexedCore(vertexShader, fragmentShader, targets, null, indexBuffer, (uint)indices.Length, drawDesc, wait);
    }

    public void DrawIndexed<TIndex>(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D depthTarget,
        GpuBuffer<TIndex> indices,
        GraphicsDrawDesc drawDesc = default,
        bool wait = true)
        where TIndex : unmanaged
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(depthTarget);
        var indexBuffer = GetValidatedIndexBuffer(indices);
        DrawIndexedCore(vertexShader, fragmentShader, targets, depthTarget, indexBuffer, (uint)indices.Length, drawDesc, wait);
    }

    private FeBufferHandle GetValidatedIndexBuffer<TIndex>(GpuBuffer<TIndex> indices)
        where TIndex : unmanaged
    {
        ArgumentNullException.ThrowIfNull(indices);
        if (!ReferenceEquals(indices.Context, context))
        {
            throw new ArgumentException("The index buffer belongs to a different GPU context.", nameof(indices));
        }
        return indices.GetNativeHandle();
    }

    private void DrawIndexedCore(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D? depthTarget,
        FeBufferHandle indexBuffer,
        uint indexCount,
        GraphicsDrawDesc drawDesc,
        bool wait)
    {
        using var operation = context.EnterOperation();
        lock (context.QueueGate)
        {
            var effectiveWait = wait && context.ActiveTimestampRecorder is null;
            var leases = ExecuteDraw(
                vertexShader,
                fragmentShader,
                targets,
                depthTarget,
                indexCount,
                indexBuffer,
                indexed: true,
                drawDesc,
                effectiveWait);
            CompleteImmediateDraw(leases, effectiveWait);
        }
    }

    internal GpuContext Context => context;

    internal List<IDisposable> DrawForQueue(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D? depthTarget,
        uint count,
        FeBufferHandle indexBuffer,
        bool indexed,
        GraphicsDrawDesc drawDesc)
        => ExecuteDraw(vertexShader, fragmentShader, targets, depthTarget, count, indexBuffer, indexed, drawDesc, wait: false);

    private List<IDisposable> ExecuteDraw(
        TVertexShader vertexShader,
        TFragmentShader fragmentShader,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D? depthTarget,
        uint count,
        FeBufferHandle indexBuffer,
        bool indexed,
        GraphicsDrawDesc drawDesc,
        bool wait)
    {
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            using var command = new GpuGraphicsCommand(Handle);
            TVertexShader.BindVertex(in vertexShader, command);
            TFragmentShader.BindFragment(in fragmentShader, command);
            DrawNative(command, targets, depthTarget, count, indexBuffer, indexed, drawDesc, wait);
            return command.DetachLeases();
        }
    }

    private void CompleteImmediateDraw(List<IDisposable> leases, bool wait)
    {
        if (!wait)
        {
            context.TrackSubmission(leases);
            return;
        }

        foreach (var lease in leases)
        {
            lease.Dispose();
        }
        context.CompleteSubmittedWork();
    }

    private unsafe void DrawNative(
        GpuGraphicsCommand command,
        ReadOnlySpan<IGpuTexture2D> targets,
        IGpuTexture2D? depthTarget,
        uint count,
        FeBufferHandle indexBuffer,
        bool indexed,
        GraphicsDrawDesc drawDesc,
        bool wait)
    {
        if (targets.IsEmpty)
        {
            throw new ArgumentException("At least one color target is required.", nameof(targets));
        }

        Span<ulong> colorHandles = targets.Length <= 8
            ? stackalloc ulong[targets.Length]
            : new ulong[targets.Length];
        for (var i = 0; i < targets.Length; i++)
        {
            if (targets[i] is not IGpuTexture2DNative native)
            {
                throw new ArgumentException("Graphics draw targets must be Feather GPU textures.", nameof(targets));
            }
            if (!ReferenceEquals(native.Context, context))
            {
                throw new ArgumentException("Graphics draw targets must belong to this pipeline's context.", nameof(targets));
            }

            var targetHandle = native.NativeHandle;
            command.Retain(targetHandle);
            colorHandles[i] = targetHandle.RawValue;
        }

        ulong depthHandle = 0;
        if (depthTarget is not null)
        {
            if (depthTarget is not IGpuTexture2DNative nativeDepth)
            {
                throw new ArgumentException("Graphics depth target must be a Feather GPU texture.", nameof(depthTarget));
            }
            if (!ReferenceEquals(nativeDepth.Context, context))
            {
                throw new ArgumentException("Graphics depth target must belong to this pipeline's context.", nameof(depthTarget));
            }

            var nativeDepthHandle = nativeDepth.NativeHandle;
            command.Retain(nativeDepthHandle);
            depthHandle = nativeDepthHandle.RawValue;
        }

        if (indexed)
        {
            command.Retain(indexBuffer);
        }

        fixed (ulong* colorPtr = colorHandles)
        {
            var viewport = drawDesc.Viewport;
            var scissor = drawDesc.Scissor;
            var clearColor = drawDesc.ClearColor.GetValueOrDefault();
            var nativeDraw = new FeGraphicsDrawDesc(
                (IntPtr)colorPtr,
                (uint)colorHandles.Length,
                depthHandle,
                count,
                indexBuffer.RawValue,
                indexed ? 1u : 0u,
                wait ? 1u : 0u,
                viewport.HasValue ? 1u : 0u,
                viewport.GetValueOrDefault().X,
                viewport.GetValueOrDefault().Y,
                viewport.GetValueOrDefault().Width,
                viewport.GetValueOrDefault().Height,
                scissor.HasValue ? 1u : 0u,
                scissor.GetValueOrDefault().X,
                scissor.GetValueOrDefault().Y,
                scissor.GetValueOrDefault().Width,
                scissor.GetValueOrDefault().Height,
                drawDesc.ClearDepth.HasValue ? 1u : 0u,
                drawDesc.ClearDepth.GetValueOrDefault(),
                (uint)drawDesc.DepthLoadOp,
                drawDesc.ClearColor.HasValue ? 1u : 0u,
                clearColor.X,
                clearColor.Y,
                clearColor.Z,
                clearColor.W,
                (uint)drawDesc.ColorLoadOp,
                drawDesc.InstanceCount,
                drawDesc.FirstVertex,
                drawDesc.FirstIndex,
                drawDesc.VertexOffset,
                drawDesc.FirstInstance);
            var recorder = context.ActiveTimestampRecorder;
            using (recorder?.IncludeCommandIntervals == true
                ? recorder.BeginCommand(
                    GpuTimestampIntervalKind.Draw,
                    Desc.DebugName ?? typeof(TVertexShader).Name + "+" + typeof(TFragmentShader).Name)
                : null)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_draw_ex(Handle, in nativeDraw));
            }
        }
    }

    public string GetVertexGLSL()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_vertex_glsl(Handle, buffer, length, out required));

    public string GetFragmentGLSL()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_fragment_glsl(Handle, buffer, length, out required));

    public string GetOptimizedVertexGLSL()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_optimized_vertex_glsl(Handle, buffer, length, out required));

    public string GetOptimizedFragmentGLSL()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_optimized_fragment_glsl(Handle, buffer, length, out required));

    public string GetOptimizedVertexIR()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_optimized_vertex_ir(Handle, buffer, length, out required));

    public string GetOptimizedFragmentIR()
        => GetInspectionString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_graphics_pipeline_get_optimized_fragment_ir(Handle, buffer, length, out required));

    IReadOnlyList<LoadedShaderSource> ILoadedGraphicsShaderInspection.InspectLoadedShaders()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return
            [
                new LoadedShaderSource(
                    typeof(TVertexShader),
                    ShaderStage.Vertex,
                    false,
                    GetNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_vertex_glsl(Handle, buffer, length, out required)),
                    GetOptionalNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_optimized_vertex_glsl(Handle, buffer, length, out required)),
                    GetOptionalNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_optimized_vertex_ir(Handle, buffer, length, out required))),
                new LoadedShaderSource(
                    typeof(TFragmentShader),
                    ShaderStage.Fragment,
                    false,
                    GetNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_fragment_glsl(Handle, buffer, length, out required)),
                    GetOptionalNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_optimized_fragment_glsl(Handle, buffer, length, out required)),
                    GetOptionalNativeString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_graphics_pipeline_get_optimized_fragment_ir(Handle, buffer, length, out required)))
            ];
        }
    }

    private string GetInspectionString(NativeStringCall.Getter getter)
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return GetNativeString(getter);
        }
    }

    private static string GetNativeString(NativeStringCall.Getter getter)
        => NativeStringCall.GetString(getter);

    private static string GetOptionalNativeString(NativeStringCall.Getter getter)
    {
        try
        {
            return GetNativeString(getter);
        }
        catch (FeatherNativeException exception) when (exception.Result == FeResult.ErrorUnsupported)
        {
            return string.Empty;
        }
    }

    public void Dispose()
    {
        lock (dispatchGate)
        {
            if (disposed)
            {
                return;
            }

            context.UnregisterGraphicsPipeline(this);
            Handle.Dispose();
            disposed = true;
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}

public readonly record struct GraphicsPipelineDesc
{
    public PrimitiveTopology Topology { get; init; } = PrimitiveTopology.TriangleList;
    public SampleCount SampleCount { get; init; } = SampleCount.X1;
    public bool DepthTest { get; init; }
    public bool DepthWrite { get; init; }
    public BlendState Blend { get; init; } = BlendState.Opaque;
    public IReadOnlyList<BlendState>? BlendAttachments { get; init; }
    public DepthStencilState DepthStencil { get; init; } = DepthStencilState.Default;
    public RasterState Raster { get; init; } = RasterState.Default;
    public uint ColorAttachmentCount { get; init; } = 1;
    public string? DebugName { get; init; }

    public GraphicsPipelineDesc()
    {
    }
}

[Flags]
public enum ColorWriteMask : uint
{
    None = 0,
    Red = 1,
    Green = 2,
    Blue = 4,
    Alpha = 8,
    Rgb = Red | Green | Blue,
    All = Red | Green | Blue | Alpha
}

public enum BlendFactor : uint
{
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha
}

public enum BlendOp : uint
{
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max
}

public readonly record struct BlendState
{
    public static BlendState Opaque => new()
    {
        Enabled = false,
        SrcColor = BlendFactor.One,
        DstColor = BlendFactor.Zero,
        ColorOp = BlendOp.Add,
        SrcAlpha = BlendFactor.One,
        DstAlpha = BlendFactor.Zero,
        AlphaOp = BlendOp.Add,
        WriteMask = ColorWriteMask.All
    };

    public static BlendState AlphaBlend => new()
    {
        Enabled = true,
        SrcColor = BlendFactor.SrcAlpha,
        DstColor = BlendFactor.OneMinusSrcAlpha,
        ColorOp = BlendOp.Add,
        SrcAlpha = BlendFactor.One,
        DstAlpha = BlendFactor.OneMinusSrcAlpha,
        AlphaOp = BlendOp.Add,
        WriteMask = ColorWriteMask.All
    };

    public static BlendState Additive => new()
    {
        Enabled = true,
        SrcColor = BlendFactor.One,
        DstColor = BlendFactor.One,
        ColorOp = BlendOp.Add,
        SrcAlpha = BlendFactor.One,
        DstAlpha = BlendFactor.One,
        AlphaOp = BlendOp.Add,
        WriteMask = ColorWriteMask.All
    };

    public bool Enabled { get; init; }
    public BlendFactor SrcColor { get; init; }
    public BlendFactor DstColor { get; init; }
    public BlendOp ColorOp { get; init; }
    public BlendFactor SrcAlpha { get; init; }
    public BlendFactor DstAlpha { get; init; }
    public BlendOp AlphaOp { get; init; }
    public ColorWriteMask WriteMask { get; init; }
}

public enum CompareOp : uint
{
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always
}

public enum StencilOp : uint
{
    Keep,
    Zero,
    Replace,
    IncrementAndClamp,
    DecrementAndClamp,
    Invert,
    IncrementAndWrap,
    DecrementAndWrap
}

public readonly record struct StencilFaceState
{
    public static StencilFaceState KeepAlways => new()
    {
        FailOp = StencilOp.Keep,
        PassOp = StencilOp.Keep,
        DepthFailOp = StencilOp.Keep,
        Compare = CompareOp.Always
    };

    public StencilOp FailOp { get; init; }
    public StencilOp PassOp { get; init; }
    public StencilOp DepthFailOp { get; init; }
    public CompareOp Compare { get; init; }
}

public readonly record struct DepthStencilState
{
    public static DepthStencilState Default => new()
    {
        DepthTest = false,
        DepthWrite = false,
        DepthCompare = CompareOp.Less,
        StencilTest = false,
        Front = StencilFaceState.KeepAlways,
        Back = StencilFaceState.KeepAlways,
        StencilReadMask = uint.MaxValue,
        StencilWriteMask = uint.MaxValue,
        StencilReference = 0
    };

    public bool DepthTest { get; init; }
    public bool DepthWrite { get; init; }
    public CompareOp DepthCompare { get; init; }
    public bool StencilTest { get; init; }
    public StencilFaceState Front { get; init; }
    public StencilFaceState Back { get; init; }
    public uint StencilReadMask { get; init; }
    public uint StencilWriteMask { get; init; }
    public uint StencilReference { get; init; }
}

public enum CullMode : uint
{
    None,
    Front,
    Back,
    FrontAndBack
}

public enum FrontFace : uint
{
    CounterClockwise,
    Clockwise
}

public enum PolygonMode : uint
{
    Fill,
    Line,
    Point
}

public readonly record struct RasterState
{
    public static RasterState Default => new()
    {
        CullMode = CullMode.None,
        FrontFace = FrontFace.CounterClockwise,
        PolygonMode = PolygonMode.Fill,
        DepthClamp = false
    };

    public CullMode CullMode { get; init; }
    public FrontFace FrontFace { get; init; }
    public PolygonMode PolygonMode { get; init; }
    public bool DepthClamp { get; init; }
}

public readonly record struct GraphicsRect(uint X, uint Y, uint Width, uint Height);

public enum GraphicsDepthLoadOp : uint
{
    Default,
    Load,
    Clear
}

public enum GraphicsColorLoadOp : uint
{
    Default,
    Load,
    Clear,
    DontCare
}

public readonly record struct GraphicsDrawDesc
{
    public GraphicsRect? Viewport { get; init; }
    public GraphicsRect? Scissor { get; init; }
    public GraphicsColorLoadOp ColorLoadOp { get; init; }
    public float4? ClearColor { get; init; }
    public GraphicsDepthLoadOp DepthLoadOp { get; init; }
    public float? ClearDepth { get; init; }
    public uint InstanceCount { get; init; }
    public uint FirstVertex { get; init; }
    public uint FirstIndex { get; init; }
    public int VertexOffset { get; init; }
    public uint FirstInstance { get; init; }
}

public sealed class GpuGraphicsCommand : IDisposable
{
    private byte[]? pushConstants;
    private List<IDisposable>? leases = [];

    internal GpuGraphicsCommand(FeGraphicsPipelineHandle handle)
    {
        Handle = handle;
        Retain(handle);
    }

    internal FeGraphicsPipelineHandle Handle { get; }

    /// <summary>
    /// Sets the vertex buffer consumed by the generated graphics pipeline.
    /// </summary>
    /// <param name="buffer">The vertex buffer.</param>
    /// <param name="stride">The vertex stride in bytes.</param>
    public void SetVertexBuffer<T>(GpuBuffer<T> buffer, uint stride)
        where T : unmanaged
    {
        ArgumentNullException.ThrowIfNull(buffer);
        Retain(buffer.Handle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_set_vertex_buffer(Handle, buffer.Handle, stride));
    }

    /// <summary>
    /// Binds a native buffer handle to a generated graphics resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="buffer">The native buffer handle.</param>
    public void BindBuffer(uint binding, IGpuBufferBinding buffer)
    {
        var native = buffer as INativeBufferBinding
            ?? throw new ArgumentException("Buffer binding was not created by Feather.", nameof(buffer));
        Retain(native.NativeBufferHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_bind_buffer(Handle, binding, native.NativeBufferHandle));
    }

    /// <summary>
    /// Binds a native texture handle to a generated graphics resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="texture">The native texture handle.</param>
    public void BindTexture(uint binding, IGpuTextureBinding texture)
    {
        var native = texture as INativeTextureBinding
            ?? throw new ArgumentException("Texture binding was not created by Feather.", nameof(texture));
        Retain(native.NativeTextureHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_bind_texture(Handle, binding, native.NativeTextureHandle));
    }

    /// <summary>
    /// Binds a native sampler handle to a generated graphics resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="sampler">The native sampler handle.</param>
    public void BindSampler(uint binding, IGpuSamplerBinding sampler)
    {
        var native = sampler as INativeSamplerBinding
            ?? throw new ArgumentException("Sampler binding was not created by Feather.", nameof(sampler));
        Retain(native.NativeSamplerHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_bind_sampler(Handle, binding, native.NativeSamplerHandle));
    }

    /// <summary>
    /// Uploads the complete push-constant byte block for the current generated graphics pipeline.
    /// </summary>
    /// <param name="data">The packed push-constant bytes.</param>
    public unsafe void SetPushConstants(ReadOnlySpan<byte> data)
    {
        fixed (byte* ptr = data)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_set_push_constants(Handle, (IntPtr)ptr, (ulong)data.Length));
        }
    }

    /// <summary>
    /// Writes a generated shader stage's push-constant range while preserving any ranges written by earlier stages.
    /// </summary>
    /// <param name="offset">The byte offset of the range to replace.</param>
    /// <param name="data">The packed range bytes.</param>
    /// <param name="totalSize">The total push-constant byte size declared by the generated pipeline.</param>
    public void SetPushConstantRange(uint offset, ReadOnlySpan<byte> data, uint totalSize)
    {
        if (offset > totalSize || data.Length > totalSize - offset)
        {
            throw new ArgumentOutOfRangeException(nameof(data), "Push-constant range exceeds the generated pipeline layout.");
        }

        // Graphics stages bind independently; keep one backing block so the second stage does not erase the first stage's range.
        pushConstants ??= new byte[totalSize];
        if (pushConstants.Length != totalSize)
        {
            pushConstants = new byte[totalSize];
        }

        data.CopyTo(pushConstants.AsSpan((int)offset, data.Length));
        SetPushConstants(pushConstants);
    }

    internal void Retain(FeSafeHandle handle)
        => (leases ?? throw new ObjectDisposedException(nameof(GpuGraphicsCommand))).Add(new NativeHandleLease(handle));

    internal List<IDisposable> DetachLeases()
    {
        var detached = leases ?? throw new ObjectDisposedException(nameof(GpuGraphicsCommand));
        leases = null;
        return detached;
    }

    public void Dispose()
    {
        if (leases is null)
        {
            return;
        }
        foreach (var lease in leases)
        {
            lease.Dispose();
        }
        leases = null;
    }
}
