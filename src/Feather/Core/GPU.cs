using Feather.AD;
using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

namespace Feather;

public static class GPU
{
    private static readonly Lazy<GpuContext> DefaultContext = new(GpuContext.GetDefault);

    public static GpuContext Context => DefaultContext.Value;

    public static GpuBuffer<T> CreateBuffer<T>(int count, BufferAccess access = BufferAccess.ReadWrite)
        where T : unmanaged
        => GpuBuffer<T>.Create(Context, count, access);

    public static GpuBuffer<T> CreateBuffer<T>(ReadOnlySpan<T> data, BufferAccess access = BufferAccess.ReadWrite)
        where T : unmanaged
        => GpuBuffer<T>.Create(Context, data, access);

    /// <summary>
    /// Creates an index buffer suitable for graphics indexed draws.
    /// </summary>
    /// <param name="data">The index values to upload.</param>
    /// <typeparam name="T">The unmanaged index value type.</typeparam>
    /// <returns>A GPU buffer containing the uploaded indices.</returns>
    public static GpuBuffer<T> CreateIndexBuffer<T>(ReadOnlySpan<T> data)
        where T : unmanaged
        => CreateBuffer(data, BufferAccess.ReadOnly);

    public static ReadOnlyBuffer<T> CreateReadOnlyBuffer<T>(ReadOnlySpan<T> data)
        where T : unmanaged
        => CreateBuffer(data, BufferAccess.ReadOnly).AsReadOnly();

    public static WriteOnlyBuffer<T> CreateWriteOnlyBuffer<T>(int count)
        where T : unmanaged
        => CreateBuffer<T>(count, BufferAccess.WriteOnly).AsWriteOnly();

    public static ReadWriteBuffer<T> CreateReadWriteBuffer<T>(int count)
        where T : unmanaged
        => CreateBuffer<T>(count, BufferAccess.ReadWrite).AsReadWrite();

    public static ReadWriteBuffer<T> CreateReadWriteBuffer<T>(ReadOnlySpan<T> data)
        where T : unmanaged
        => CreateBuffer(data, BufferAccess.ReadWrite).AsReadWrite();

    /// <summary>
    /// Creates a two-dimensional texture with one mip level.
    /// </summary>
    /// <param name="width">The base-level width in pixels.</param>
    /// <param name="height">The base-level height in pixels.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <typeparam name="TPixel">The host pixel storage type.</typeparam>
    /// <typeparam name="TValue">The shader-facing pixel value type.</typeparam>
    /// <returns>The created texture.</returns>
    public static GpuTexture2D<TPixel, TValue> CreateTexture2D<TPixel, TValue>(
        int width,
        int height,
        PixelFormat format,
        TextureAccess access = TextureAccess.ReadWrite)
        where TPixel : unmanaged
        where TValue : unmanaged
        => GpuTexture2D<TPixel, TValue>.Create(Context, width, height, format, access);

    /// <summary>
    /// Creates a two-dimensional texture with an explicit mip-level count.
    /// </summary>
    /// <param name="width">The base-level width in pixels.</param>
    /// <param name="height">The base-level height in pixels.</param>
    /// <param name="mipLevels">The number of mip levels to allocate.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <typeparam name="TPixel">The host pixel storage type.</typeparam>
    /// <typeparam name="TValue">The shader-facing pixel value type.</typeparam>
    /// <returns>The created texture.</returns>
    public static GpuTexture2D<TPixel, TValue> CreateTexture2D<TPixel, TValue>(
        int width,
        int height,
        int mipLevels,
        PixelFormat format,
        TextureAccess access = TextureAccess.ReadWrite)
        where TPixel : unmanaged
        where TValue : unmanaged
        => GpuTexture2D<TPixel, TValue>.Create(Context, width, height, mipLevels, format, access);

    /// <summary>
    /// Creates a three-dimensional texture with one mip level.
    /// </summary>
    /// <param name="width">The base-level width in voxels.</param>
    /// <param name="height">The base-level height in voxels.</param>
    /// <param name="depth">The base-level depth in voxels.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <typeparam name="TPixel">The host voxel storage type.</typeparam>
    /// <typeparam name="TValue">The shader-facing voxel value type.</typeparam>
    /// <returns>The created texture.</returns>
    public static GpuTexture3D<TPixel, TValue> CreateTexture3D<TPixel, TValue>(
        int width,
        int height,
        int depth,
        PixelFormat format,
        TextureAccess access = TextureAccess.ReadWrite)
        where TPixel : unmanaged
        where TValue : unmanaged
        => GpuTexture3D<TPixel, TValue>.Create(Context, width, height, depth, format, access);

    /// <summary>
    /// Creates a three-dimensional texture with an explicit mip-level count.
    /// </summary>
    /// <param name="width">The base-level width in voxels.</param>
    /// <param name="height">The base-level height in voxels.</param>
    /// <param name="depth">The base-level depth in voxels.</param>
    /// <param name="mipLevels">The number of mip levels to allocate.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <typeparam name="TPixel">The host voxel storage type.</typeparam>
    /// <typeparam name="TValue">The shader-facing voxel value type.</typeparam>
    /// <returns>The created texture.</returns>
    public static GpuTexture3D<TPixel, TValue> CreateTexture3D<TPixel, TValue>(
        int width,
        int height,
        int depth,
        int mipLevels,
        PixelFormat format,
        TextureAccess access = TextureAccess.ReadWrite)
        where TPixel : unmanaged
        where TValue : unmanaged
        => GpuTexture3D<TPixel, TValue>.Create(Context, width, height, depth, mipLevels, format, access);

    /// <summary>
    /// Creates a render-target texture for graphics draw output.
    /// </summary>
    public static GpuTexture2D<TPixel, TValue> CreateRenderTexture2D<TPixel, TValue>(
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => CreateTexture2D<TPixel, TValue>(width, height, format, TextureAccess.RenderTarget);

    /// <summary>
    /// Creates a 32-bit floating-point depth texture for graphics pipelines.
    /// </summary>
    public static GpuTexture2D<float, float> CreateDepthTexture2D(int width, int height)
        => CreateTexture2D<float, float>(width, height, PixelFormat.Depth32Float, TextureAccess.DepthStencil);

    /// <summary>
    /// Creates a packed depth/stencil texture for graphics pipelines that use stencil state.
    /// </summary>
    public static GpuTexture2D<uint, uint> CreateDepthStencilTexture2D(int width, int height)
        => CreateTexture2D<uint, uint>(width, height, PixelFormat.Depth24Stencil8, TextureAccess.DepthStencil);

    public static GpuTexture2D<TPixel, TValue> LoadReadWriteTexture2D<TPixel, TValue>(string path)
        where TPixel : unmanaged
        where TValue : unmanaged
        => TextureImageCodec.LoadReadWrite<TPixel, TValue>(Context, path);

    /// <summary>
    /// Loads an image as a sampled texture using Feather's dependency-free TGA baseline loader.
    /// </summary>
    public static GpuTexture2D<TPixel, TValue> LoadSampledTexture2D<TPixel, TValue>(string path)
        where TPixel : unmanaged
        where TValue : unmanaged
        => TextureImageCodec.LoadSampled<TPixel, TValue>(Context, path);

    public static SamplerState CreateSampler(SamplerDesc desc)
        => SamplerState.Create(Context, desc);

    public static GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings> CreateGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>(
        GraphicsPipelineDesc desc = default)
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        => GpuGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>.Create(Context, desc);

    /// <summary>
    /// Creates a managed AD kernel wrapper for a generated one-dimensional kernel.
    /// </summary>
    public static GpuADKernel<TKernel> CreateADKernel<TKernel>(TKernel kernel)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => new(kernel);

    public static void Dispatch<TKernel>(TKernel kernel, int x, bool wait = true)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(x, 1, 1), wait);

    /// <summary>
    /// Dispatches a generated one-dimensional kernel and returns the native route used for the dispatch.
    /// </summary>
    public static DispatchPath DispatchAndGetPath<TKernel>(TKernel kernel, int x, bool wait = true)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(x, 1, 1), wait);

    public static void Dispatch<TKernel>(TKernel kernel, int2 size, bool wait = true)
        where TKernel : struct, IKernel2D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(size.X, size.Y, 1), wait);

    /// <summary>
    /// Dispatches a generated two-dimensional kernel and returns the native route used for the dispatch.
    /// </summary>
    public static DispatchPath DispatchAndGetPath<TKernel>(TKernel kernel, int2 size, bool wait = true)
        where TKernel : struct, IKernel2D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(size.X, size.Y, 1), wait);

    public static void Dispatch<TKernel>(TKernel kernel, int3 size, bool wait = true)
        where TKernel : struct, IKernel3D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(size.X, size.Y, size.Z), wait);

    /// <summary>
    /// Dispatches a generated three-dimensional kernel and returns the native route used for the dispatch.
    /// </summary>
    public static DispatchPath DispatchAndGetPath<TKernel>(TKernel kernel, int3 size, bool wait = true)
        where TKernel : struct, IKernel3D, IGeneratedKernel<TKernel>
        => CachedKernelDispatcher<TKernel>.Dispatch(kernel, new GpuDispatchSize(size.X, size.Y, size.Z), wait);

    private static class CachedKernelDispatcher<TKernel>
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        private static readonly object Gate = new();
        private static GpuKernel? cachedKernel;

        public static DispatchPath Dispatch(TKernel kernel, GpuDispatchSize size, bool wait)
        {
            if (GpuKernel.IrTransformForTesting is not null)
            {
                using var uncachedKernel = GpuKernel.Create<TKernel>(Context);
                GpuKernel.Dispatch(Context, uncachedKernel, kernel, size, wait);
                return uncachedKernel.LastDispatchPath;
            }

            lock (Gate)
            {
                cachedKernel ??= GpuKernel.Create<TKernel>(Context);
                GpuKernel.Dispatch(Context, cachedKernel, kernel, size, wait);
                return cachedKernel.LastDispatchPath;
            }
        }
    }
}

public readonly record struct GpuDispatchSize(int X, int Y, int Z);
