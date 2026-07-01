namespace Feather.Windowing;

public sealed class GpuPixelBuffer : IDisposable
{
    private uint[] pixels;
    private bool disposed;

    public GpuPixelBuffer(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        Width = width;
        Height = height;
        pixels = new uint[checked(width * height)];
    }

    public int Width { get; private set; }
    public int Height { get; private set; }

    public Span<uint> Pixels
    {
        get
        {
            ThrowIfDisposed();
            return pixels;
        }
    }

    public void Resize(int width, int height)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (width == Width && height == Height)
        {
            return;
        }

        Width = width;
        Height = height;
        pixels = new uint[checked(width * height)];
    }

    public void Clear(uint rgba)
    {
        ThrowIfDisposed();
        pixels.AsSpan().Fill(rgba);
    }

    public void SetPixel(int x, int y, uint rgba)
    {
        ThrowIfDisposed();
        CheckCoordinates(x, y);
        pixels[(y * Width) + x] = rgba;
    }

    public uint GetPixel(int x, int y)
    {
        ThrowIfDisposed();
        CheckCoordinates(x, y);
        return pixels[(y * Width) + x];
    }

    public void Dispose()
    {
        disposed = true;
        pixels = [];
    }

    private void CheckCoordinates(int x, int y)
    {
        if ((uint)x >= (uint)Width)
        {
            throw new ArgumentOutOfRangeException(nameof(x));
        }
        if ((uint)y >= (uint)Height)
        {
            throw new ArgumentOutOfRangeException(nameof(y));
        }
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}
