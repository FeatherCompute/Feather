using Feather.Math;
using Feather.Native;

namespace Feather.Resources;

public interface IGpuTexture2D
{
    int Width { get; }
    int Height { get; }
    PixelFormat Format { get; }
    TextureAccess Access { get; }
}

internal interface IGpuTexture2DNative : IGpuTexture2D
{
    FeTextureHandle NativeHandle { get; }
}

public sealed class GpuTexture2D<TPixel, TValue> : IDisposable
    , IGpuTexture2D
    , IGpuTexture2DNative
    where TPixel : unmanaged
    where TValue : unmanaged
{
    private bool disposed;

    private GpuTexture2D(GpuContext context, FeTextureHandle handle, int width, int height, int mipLevels, PixelFormat format, TextureAccess access)
    {
        Context = context;
        Handle = handle;
        Width = width;
        Height = height;
        MipLevels = mipLevels;
        Format = format;
        Access = access;
    }

    internal GpuContext Context { get; }
    internal FeTextureHandle Handle { get; }
    FeTextureHandle IGpuTexture2DNative.NativeHandle => GetNativeHandle();

    internal FeTextureHandle GetNativeHandle()
    {
        ThrowIfDisposed();
        return Handle;
    }

    /// <summary>
    /// Gets the width of the base texture level in pixels.
    /// </summary>
    public int Width { get; }

    /// <summary>
    /// Gets the height of the base texture level in pixels.
    /// </summary>
    public int Height { get; }

    /// <summary>
    /// Gets the two-dimensional size of the base texture level.
    /// </summary>
    public int2 Size => new(Width, Height);

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
    /// Creates a two-dimensional texture with one mip level.
    /// </summary>
    /// <param name="context">The owning GPU context.</param>
    /// <param name="width">The base-level width in pixels.</param>
    /// <param name="height">The base-level height in pixels.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <returns>The created texture.</returns>
    public static GpuTexture2D<TPixel, TValue> Create(GpuContext context, int width, int height, PixelFormat format, TextureAccess access)
        => Create(context, width, height, mipLevels: 1, format, access);

    /// <summary>
    /// Creates a two-dimensional texture with an explicit mip-level count.
    /// </summary>
    /// <param name="context">The owning GPU context.</param>
    /// <param name="width">The base-level width in pixels.</param>
    /// <param name="height">The base-level height in pixels.</param>
    /// <param name="mipLevels">The number of mip levels to allocate.</param>
    /// <param name="format">The texture pixel format.</param>
    /// <param name="access">The texture access mode.</param>
    /// <returns>The created texture.</returns>
    public static GpuTexture2D<TPixel, TValue> Create(GpuContext context, int width, int height, int mipLevels, PixelFormat format, TextureAccess access)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(mipLevels);
        var maxMipLevels = CalculateFullMipLevelCount(width, height);
        if (mipLevels > maxMipLevels)
        {
            throw new ArgumentOutOfRangeException(nameof(mipLevels), $"A {width}x{height} texture supports at most {maxMipLevels} mip levels.");
        }

        var desc = new FeTexture2DDesc((uint)width, (uint)height, (uint)mipLevels, (uint)format, (uint)access);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_create(context.Handle, in desc, IntPtr.Zero, out var handle));
        return new GpuTexture2D<TPixel, TValue>(context, handle, width, height, mipLevels, format, access);
    }

    /// <summary>
    /// Creates a shader-facing read-only texture view over this texture.
    /// </summary>
    public ReadOnlyTexture2D<TValue> AsReadOnly() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing write-only texture view over this texture.
    /// </summary>
    public WriteOnlyTexture2D<TValue> AsWriteOnly() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing read-write texture view over this texture.
    /// </summary>
    public ReadWriteTexture2D<TValue> AsReadWrite() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing normalized read-write texture view over this texture.
    /// </summary>
    public ReadWriteNormalizedTexture2D<TValue> AsReadWriteNormalized() => new(Handle, Size);

    /// <summary>
    /// Creates a shader-facing sampled texture view over this texture.
    /// </summary>
    public SampledTexture2D<TValue> AsSampled() => new(Handle, Size);

    /// <summary>
    /// Uploads pixels into the base texture level.
    /// </summary>
    /// <param name="pixels">The source pixels. The span must contain at least <c>Width * Height</c> elements.</param>
    public unsafe void Upload(ReadOnlySpan<TPixel> pixels)
    {
        ThrowIfDisposed();
        if (pixels.Length < Width * Height)
        {
            throw new ArgumentException("Pixel span is shorter than texture area.", nameof(pixels));
        }

        fixed (TPixel* ptr = pixels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_upload(Handle, 0, 0, (uint)Width, (uint)Height, (IntPtr)ptr));
        }
    }

    /// <summary>
    /// Reads pixels from the base texture level.
    /// </summary>
    /// <param name="pixels">The destination pixels. The span must contain at least <c>Width * Height</c> elements.</param>
    public unsafe void Read(Span<TPixel> pixels)
    {
        ThrowIfDisposed();
        if (pixels.Length < Width * Height)
        {
            throw new ArgumentException("Pixel span is shorter than texture area.", nameof(pixels));
        }

        fixed (TPixel* ptr = pixels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture2d_download(Handle, 0, 0, (uint)Width, (uint)Height, (IntPtr)ptr));
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

    /// <summary>
    /// Saves the base texture level as an uncompressed TGA image.
    /// </summary>
    /// <param name="path">The path to write.</param>
    public void Save(string path)
    {
        ThrowIfDisposed();
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        var pixels = new TPixel[checked(Width * Height)];
        Read(pixels);
        TextureImageCodec.SaveTga(path, pixels, Width, Height, Format);
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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }

    private static int CalculateFullMipLevelCount(int width, int height)
    {
        var levels = 1;
        var size = System.Math.Max(width, height);
        while (size > 1)
        {
            size /= 2;
            levels++;
        }

        return levels;
    }
}

internal static class TextureImageCodec
{
    public static GpuTexture2D<TPixel, TValue> LoadReadWrite<TPixel, TValue>(GpuContext context, string path)
        where TPixel : unmanaged
        where TValue : unmanaged
        => Load<TPixel, TValue>(context, path, TextureAccess.ReadWrite);

    public static GpuTexture2D<TPixel, TValue> LoadSampled<TPixel, TValue>(GpuContext context, string path)
        where TPixel : unmanaged
        where TValue : unmanaged
        => Load<TPixel, TValue>(context, path, TextureAccess.Sampled);

    private static GpuTexture2D<TPixel, TValue> Load<TPixel, TValue>(GpuContext context, string path, TextureAccess access)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        var image = LoadTga<TPixel>(path);
        var texture = GpuTexture2D<TPixel, TValue>.Create(context, image.Width, image.Height, image.Format, access);
        var pixels = image.Pixels;
        texture.Upload(pixels);
        return texture;
    }

    public static void SaveTga<TPixel>(string path, ReadOnlySpan<TPixel> pixels, int width, int height, PixelFormat format)
        where TPixel : unmanaged
    {
        var pixelSize = BytePixelSize(format);
        if (pixelSize != System.Runtime.CompilerServices.Unsafe.SizeOf<TPixel>())
        {
            throw new NotSupportedException($"Pixel type {typeof(TPixel).FullName} does not match {format}.");
        }

        if (pixels.Length < width * height)
        {
            throw new ArgumentException("Pixel span is shorter than texture area.", nameof(pixels));
        }

        // TGA is used as the dependency-free baseline image format for byte-addressable textures.
        using var stream = File.Create(path);
        Span<byte> header = stackalloc byte[18];
        header[2] = format == PixelFormat.R8 ? (byte)3 : (byte)2;
        header[12] = (byte)width;
        header[13] = (byte)(width >> 8);
        header[14] = (byte)height;
        header[15] = (byte)(height >> 8);
        header[16] = (byte)(pixelSize * 8);
        header[17] = 0x20;
        if (format is PixelFormat.Rgba8 or PixelFormat.Bgra8)
        {
            header[17] |= 8;
        }

        stream.Write(header);

        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(pixels);
        var encoded = new byte[checked(width * height * pixelSize)];
        for (var i = 0; i < width * height; i++)
        {
            var offset = i * pixelSize;
            // TGA stores true-color bytes in BGR/BGRA order; Feather texture bytes stay in declared GPU order.
            switch (format)
            {
                case PixelFormat.R8:
                    encoded[offset] = bytes[offset];
                    break;
                case PixelFormat.Rg8:
                    encoded[offset] = bytes[offset + 1];
                    encoded[offset + 1] = bytes[offset];
                    break;
                case PixelFormat.Rgba8:
                    encoded[offset] = bytes[offset + 2];
                    encoded[offset + 1] = bytes[offset + 1];
                    encoded[offset + 2] = bytes[offset];
                    encoded[offset + 3] = bytes[offset + 3];
                    break;
                case PixelFormat.Bgra8:
                    encoded[offset] = bytes[offset];
                    encoded[offset + 1] = bytes[offset + 1];
                    encoded[offset + 2] = bytes[offset + 2];
                    encoded[offset + 3] = bytes[offset + 3];
                    break;
                default:
                    throw UnsupportedFormat(format);
            }
        }

        stream.Write(encoded);
    }

    private static TextureImage<TPixel> LoadTga<TPixel>(string path)
        where TPixel : unmanaged
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 18)
        {
            throw new InvalidDataException("TGA file is shorter than the 18-byte header.");
        }

        var idLength = bytes[0];
        var colorMapType = bytes[1];
        var imageType = bytes[2];
        if (colorMapType != 0 || imageType is not (2 or 3))
        {
            throw new NotSupportedException("Texture image loading supports uncompressed true-color and grayscale TGA files.");
        }

        var width = bytes[12] | (bytes[13] << 8);
        var height = bytes[14] | (bytes[15] << 8);
        var bitsPerPixel = bytes[16];
        var descriptor = bytes[17];
        if (width <= 0 || height <= 0)
        {
            throw new InvalidDataException("TGA dimensions must be positive.");
        }

        var format = (imageType, bitsPerPixel) switch
        {
            (3, 8) => PixelFormat.R8,
            (2, 24) => PixelFormat.Rgba8,
            (2, 32) => PixelFormat.Rgba8,
            _ => throw new NotSupportedException($"Unsupported TGA pixel depth {bitsPerPixel}.")
        };
        var pixelSize = bitsPerPixel / 8;
        if (format == PixelFormat.Rgba8 && System.Runtime.CompilerServices.Unsafe.SizeOf<TPixel>() != 4)
        {
            throw new NotSupportedException($"Pixel type {typeof(TPixel).FullName} must be 4 bytes for TGA color images.");
        }

        if (format == PixelFormat.R8 && System.Runtime.CompilerServices.Unsafe.SizeOf<TPixel>() != 1)
        {
            throw new NotSupportedException($"Pixel type {typeof(TPixel).FullName} must be 1 byte for TGA grayscale images.");
        }

        var dataOffset = 18 + idLength;
        var required = checked(width * height * pixelSize);
        if (bytes.Length - dataOffset < required)
        {
            throw new InvalidDataException("TGA file does not contain enough pixel data.");
        }

        var pixels = new TPixel[checked(width * height)];
        var pixelBytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(pixels.AsSpan());
        var topOrigin = (descriptor & 0x20) != 0;

        // TGA stores color pixels as BGR/BGRA. Feather exposes loaded color images as Rgba8.
        for (var y = 0; y < height; y++)
        {
            var srcY = topOrigin ? y : height - 1 - y;
            for (var x = 0; x < width; x++)
            {
                var src = dataOffset + (((srcY * width) + x) * pixelSize);
                var dst = ((y * width) + x) * BytePixelSize(format);
                if (format == PixelFormat.R8)
                {
                    pixelBytes[dst] = bytes[src];
                    continue;
                }

                pixelBytes[dst] = bytes[src + 2];
                pixelBytes[dst + 1] = bytes[src + 1];
                pixelBytes[dst + 2] = bytes[src];
                pixelBytes[dst + 3] = pixelSize == 4 ? bytes[src + 3] : byte.MaxValue;
            }
        }

        return new TextureImage<TPixel>(width, height, format, pixels);
    }

    private static int BytePixelSize(PixelFormat format)
        => format switch
        {
            PixelFormat.R8 => 1,
            PixelFormat.Rg8 => 2,
            PixelFormat.Rgba8 or PixelFormat.Bgra8 => 4,
            _ => throw UnsupportedFormat(format)
        };

    private static NotSupportedException UnsupportedFormat(PixelFormat format)
        => new($"Texture image IO supports uncompressed TGA for R8, Rg8, Rgba8, and Bgra8 textures. {format} is not supported.");

    private readonly record struct TextureImage<TPixel>(int Width, int Height, PixelFormat Format, TPixel[] Pixels)
        where TPixel : unmanaged;
}

public readonly struct ReadOnlyTexture2D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadOnlyTexture2D(FeTextureHandle handle, int2 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }
    public int2 Size { get; }
    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;
    public T this[int2 xy] => ShaderRuntimeMarker<T>.Value;
}

public readonly struct WriteOnlyTexture2D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal WriteOnlyTexture2D(FeTextureHandle handle, int2 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }
    public int2 Size { get; }
    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;
    public T this[int2 xy]
    {
        set => _ = value;
    }
}

public readonly struct ReadWriteTexture2D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadWriteTexture2D(FeTextureHandle handle, int2 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }
    public int2 Size { get; }
    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;
    public T this[int2 xy]
    {
        get => ShaderRuntimeMarker<T>.Value;
        set => _ = value;
    }
}

public readonly struct ReadWriteNormalizedTexture2D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal ReadWriteNormalizedTexture2D(FeTextureHandle handle, int2 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }
    public int2 Size { get; }
    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;
    public T this[int2 xy]
    {
        get => ShaderRuntimeMarker<T>.Value;
        set => _ = value;
    }
}

public readonly struct SampledTexture2D<T> : IGpuTextureBinding
    where T : unmanaged
{
    internal SampledTexture2D(FeTextureHandle handle, int2 size)
    {
        Handle = handle;
        Size = size;
    }

    internal FeTextureHandle Handle { get; }
    public int2 Size { get; }
    FeTextureHandle IGpuTextureBinding.NativeTextureHandle => Handle;

    public T Sample(SamplerState sampler, float2 uv) => ShaderRuntimeMarker<T>.Value;

    public T SampleLevel(SamplerState sampler, float2 uv, float lod) => ShaderRuntimeMarker<T>.Value;

    public T SampleGrad(SamplerState sampler, float2 uv, float2 ddx, float2 ddy) => ShaderRuntimeMarker<T>.Value;
}
