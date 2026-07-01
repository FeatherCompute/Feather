using Feather.Math;
using Feather.Native;

namespace Feather.Resources;

/// <summary>
/// Represents a three-dimensional GPU texture.
/// </summary>
/// <typeparam name="TPixel">The host-side pixel or voxel storage type.</typeparam>
/// <typeparam name="TValue">The shader-facing texture value type.</typeparam>
public sealed class GpuTexture3D<TPixel, TValue> : IDisposable
    where TPixel : unmanaged
    where TValue : unmanaged
{
    private bool disposed;

    private GpuTexture3D(GpuContext context, FeTextureHandle handle, int width, int height, int depth, int mipLevels, PixelFormat format, TextureAccess access)
    {
        Context = context;
        Handle = handle;
        Width = width;
        Height = height;
        Depth = depth;
        MipLevels = mipLevels;
        Format = format;
        Access = access;
    }

    internal GpuContext Context { get; }
    internal FeTextureHandle Handle { get; }

    /// <summary>
    /// Gets the width of the base texture level in voxels.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Gets the height of the base texture level in voxels.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Gets the depth of the base texture level in voxels.
    /// </summary>
    public int Depth { get; }

    /// <summary>
    /// Gets the three-dimensional size of the base texture level.
    /// </summary>
    public int3 Size => new(Width, Height, Depth);

    /// <summary>
    /// Gets the number of mip levels allocated for this texture.
    /// </summary>
    public int MipLevels { get; }

    /// <summary>
    /// Gets the texture pixel format.
    /// </summary>
    public PixelFormat Format { get; }

    /// <summary>
    /// Gets the host and shader access mode for this texture.
    /// </summary>
    public TextureAccess Access { get; }

    /// <summary>
    /// Creates a three-dimensional texture with one mip level.
    /// </summary>
    public static GpuTexture3D<TPixel, TValue> Create(GpuContext context, int width, int height, int depth, PixelFormat format, TextureAccess access)
        => Create(context, width, height, depth, mipLevels: 1, format, access);

    /// <summary>
    /// Creates a three-dimensional texture with an explicit mip-level count.
    /// </summary>
    public static GpuTexture3D<TPixel, TValue> Create(GpuContext context, int width, int height, int depth, int mipLevels, PixelFormat format, TextureAccess access)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(depth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(mipLevels);

        var maxMipLevels = CalculateFullMipLevelCount(width, height, depth);
        if (mipLevels > maxMipLevels)
        {
            throw new ArgumentOutOfRangeException(nameof(mipLevels), $"A {width}x{height}x{depth} texture supports at most {maxMipLevels} mip levels.");
        }

        var desc = new FeTexture3DDesc((uint)width, (uint)height, (uint)depth, (uint)mipLevels, (uint)format, (uint)access);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_texture3d_create(context.Handle, in desc, IntPtr.Zero, out var handle));
        return new GpuTexture3D<TPixel, TValue>(context, handle, width, height, depth, mipLevels, format, access);
    }

    /// <summary>
    /// Creates a shader-facing read-only texture view over this texture.
    /// </summary>
    public ReadOnlyTexture3D<TValue> AsReadOnly() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing write-only texture view over this texture.
    /// </summary>
    public WriteOnlyTexture3D<TValue> AsWriteOnly() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing read-write texture view over this texture.
    /// </summary>
    public ReadWriteTexture3D<TValue> AsReadWrite() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing normalized read-write texture view over this texture.
    /// </summary>
    public ReadWriteNormalizedTexture3D<TValue> AsReadWriteNormalized() => new(Handle, Size);

    /// <summary>
    /// Uploads voxels into the base texture level.
    /// </summary>
    /// <param name="voxels">The source voxels. The span must contain at least <c>Width * Height * Depth</c> elements.</param>
    public unsafe void Upload(ReadOnlySpan<TPixel> voxels)
    {
        ThrowIfDisposed();
        if (voxels.Length < Width * Height * Depth)
        {
            throw new ArgumentException("Voxel span is shorter than texture volume.", nameof(voxels));
        }

        fixed (TPixel* ptr = voxels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture3d_upload(Handle, 0, 0, 0, (uint)Width, (uint)Height, (uint)Depth, (IntPtr)ptr));
        }
    }

    /// <summary>
    /// Reads voxels from the base texture level.
    /// </summary>
    /// <param name="voxels">The destination voxels. The span must contain at least <c>Width * Height * Depth</c> elements.</param>
    public unsafe void Read(Span<TPixel> voxels)
    {
        ThrowIfDisposed();
        if (voxels.Length < Width * Height * Depth)
        {
            throw new ArgumentException("Voxel span is shorter than texture volume.", nameof(voxels));
        }

        fixed (TPixel* ptr = voxels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture3d_download(Handle, 0, 0, 0, (uint)Width, (uint)Height, (uint)Depth, (IntPtr)ptr));
        }
    }

    /// <summary>
    /// Requests native mipmap generation for the allocated mip chain.
    /// </summary>
    public void GenerateMipmaps()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_texture_generate_mipmaps(Handle));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Handle.Dispose();
        disposed = true;
    }

    private static int CalculateFullMipLevelCount(int width, int height, int depth)
    {
        var levels = 1;
        var size = System.Math.Max(System.Math.Max(width, height), depth);
        while (size > 1)
        {
            size /= 2;
            levels++;
        }

        return levels;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}

/// <summary>
/// Shader-facing read-only three-dimensional texture view.
/// </summary>
public readonly struct ReadOnlyTexture3D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadOnlyTexture3D(FeTextureHandle handle, int3 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }

    /// <summary>
    /// Gets the base-level texture dimensions.
    /// </summary>
    public int3 Size { get; }

    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;

    /// <summary>
    /// Gets the shader value at a three-dimensional coordinate.
    /// </summary>
    public T this[int3 xyz] => ShaderRuntimeMarker<T>.Value;
}

/// <summary>
/// Shader-facing write-only three-dimensional texture view.
/// </summary>
public readonly struct WriteOnlyTexture3D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal WriteOnlyTexture3D(FeTextureHandle handle, int3 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }

    /// <summary>
    /// Gets the base-level texture dimensions.
    /// </summary>
    public int3 Size { get; }

    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;

    /// <summary>
    /// Sets the shader value at a three-dimensional coordinate.
    /// </summary>
    public T this[int3 xyz]
    {
        set => _ = value;
    }
}

/// <summary>
/// Shader-facing read-write three-dimensional texture view.
/// </summary>
public readonly struct ReadWriteTexture3D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadWriteTexture3D(FeTextureHandle handle, int3 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }

    /// <summary>
    /// Gets the base-level texture dimensions.
    /// </summary>
    public int3 Size { get; }

    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;

    /// <summary>
    /// Gets or sets the shader value at a three-dimensional coordinate.
    /// </summary>
    public T this[int3 xyz]
    {
        get => ShaderRuntimeMarker<T>.Value;
        set => _ = value;
    }
}

/// <summary>
/// Shader-facing normalized read-write three-dimensional texture view.
/// </summary>
public readonly struct ReadWriteNormalizedTexture3D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadWriteNormalizedTexture3D(FeTextureHandle handle, int3 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }

    /// <summary>
    /// Gets the base-level texture dimensions.
    /// </summary>
    public int3 Size { get; }

    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;

    /// <summary>
    /// Gets or sets the shader value at a three-dimensional coordinate.
    /// </summary>
    public T this[int3 xyz]
    {
        get => ShaderRuntimeMarker<T>.Value;
        set => _ = value;
    }
}
