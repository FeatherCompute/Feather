using Feather.Native;
using Feather.Resources;

namespace Feather.Windowing;

public sealed class GpuTexturePresenter : IDisposable
{
    private readonly GpuWindow window;
    private bool disposed;

    internal GpuTexturePresenter(GpuWindow window, FeTexturePresenterHandle handle)
    {
        this.window = window;
        Handle = handle;
    }

    internal FeTexturePresenterHandle Handle { get; }

    public void Present<TPixel, TValue>(
        GpuTexture2D<TPixel, TValue> texture,
        PresentMode mode = PresentMode.Auto)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(texture);
        _ = window.IsOpen;
        NativeMethods.ThrowIfFailed(NativeMethods.fe_texture_presenter_present_texture(Handle, texture.GetNativeHandle(), (uint)mode));
    }

    public unsafe void Present(GpuPixelBuffer pixels)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pixels);
        _ = window.IsOpen;
        fixed (uint* ptr = pixels.Pixels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_texture_presenter_present_pixels(Handle, (IntPtr)ptr, (uint)pixels.Width, (uint)pixels.Height));
        }
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Handle.Dispose();
        window.RemovePresenter(this);
        disposed = true;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}
