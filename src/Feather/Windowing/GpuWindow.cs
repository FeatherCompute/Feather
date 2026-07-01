using Feather.Math;
using Feather.Native;

namespace Feather.Windowing;

public sealed class GpuWindow : IDisposable
{
    private readonly List<GpuTexturePresenter> presenters = [];
    private bool disposed;

    private GpuWindow(FeWindowHandle handle)
    {
        Handle = handle;
    }

    internal FeWindowHandle Handle { get; }

    public static GpuWindow Create(GpuWindowOptions options)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.Height);
        ArgumentNullException.ThrowIfNull(options.Title);

        var desc = new FeWindowDesc
        {
            Width = (uint)options.Width,
            Height = (uint)options.Height,
            Title = options.Title,
            Resizable = options.Resizable ? 1u : 0u,
            Visible = options.Visible ? 1u : 0u,
            VSync = options.VSync ? 1u : 0u,
            HighDpi = options.HighDpi ? 1u : 0u,
            CenterOnCreate = options.CenterOnCreate ? 1u : 0u
        };

        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_create(in desc, out var handle));
        return new GpuWindow(handle);
    }

    public bool IsOpen
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_window_is_open(Handle, out var isOpen));
            return isOpen;
        }
    }

    public int Width
    {
        get
        {
            GetSize(out var width, out _);
            return width;
        }
    }

    public int Height
    {
        get
        {
            GetSize(out _, out var height);
            return height;
        }
    }

    public float Aspect => Height == 0 ? 0.0f : (float)Width / Height;

    public int2 MousePosition
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_window_get_mouse_position(Handle, out var x, out var y));
            return new int2(x, y);
        }
    }

    public float2 MouseScroll
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_window_get_mouse_scroll(Handle, out var x, out var y));
            return new float2(x, y);
        }
    }

    public void Close()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_close(Handle));
    }

    public void PollEvents()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_poll_events(Handle));
    }

    public void WaitEvents()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_wait_events(Handle));
    }

    public bool TryPollEvent(out WindowEvent windowEvent)
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_poll_event(Handle, out var nativeEvent, out var hasEvent));
        if (!hasEvent)
        {
            windowEvent = null!;
            return false;
        }

        windowEvent = ConvertEvent(nativeEvent);
        return true;
    }

    public bool IsKeyDown(WindowKey key)
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_is_key_down(Handle, unchecked((uint)(int)key), out var isDown));
        return isDown;
    }

    public bool IsMouseDown(MouseButton button)
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_is_mouse_down(Handle, (uint)button, out var isDown));
        return isDown;
    }

    public void SetTitle(string title)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(title);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_set_title(Handle, title));
    }

    public void SetVSync(bool enabled)
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_set_vsync(Handle, enabled));
    }

    public unsafe void Present(GpuPixelBuffer pixels)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(pixels);
        fixed (uint* ptr = pixels.Pixels)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_window_present_pixels(Handle, (IntPtr)ptr, (uint)pixels.Width, (uint)pixels.Height));
        }
    }

    public GpuTexturePresenter CreateTexturePresenter()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_texture_presenter_create(Handle, out var presenter));
        var managedPresenter = new GpuTexturePresenter(this, presenter);
        presenters.Add(managedPresenter);
        return managedPresenter;
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        foreach (var presenter in presenters.ToArray())
        {
            presenter.Dispose();
        }
        presenters.Clear();
        Handle.Dispose();
        disposed = true;
    }

    internal void RemovePresenter(GpuTexturePresenter presenter)
    {
        if (!disposed)
        {
            presenters.Remove(presenter);
        }
    }

    private void GetSize(out int width, out int height)
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_window_get_size(Handle, out var nativeWidth, out var nativeHeight));
        width = checked((int)nativeWidth);
        height = checked((int)nativeHeight);
    }

    private static WindowEvent ConvertEvent(FeWindowEvent nativeEvent)
        => (WindowEventKind)nativeEvent.Kind switch
        {
            WindowEventKind.Resize => new WindowResizeEvent(checked((int)nativeEvent.Width), checked((int)nativeEvent.Height)),
            WindowEventKind.Close => new WindowCloseEvent(),
            WindowEventKind.Key => new WindowKeyEvent((WindowKey)unchecked((int)nativeEvent.Key), nativeEvent.Pressed != 0, (KeyModifiers)nativeEvent.Modifiers),
            WindowEventKind.CharInput => new WindowCharInputEvent(nativeEvent.Codepoint),
            WindowEventKind.MouseButton => new WindowMouseButtonEvent((MouseButton)nativeEvent.MouseButton, nativeEvent.Pressed != 0, nativeEvent.X, nativeEvent.Y, (KeyModifiers)nativeEvent.Modifiers),
            WindowEventKind.MouseMove => new WindowMouseMoveEvent(nativeEvent.X, nativeEvent.Y, nativeEvent.DeltaX, nativeEvent.DeltaY),
            WindowEventKind.MouseScroll => new WindowMouseScrollEvent(nativeEvent.ScrollX, nativeEvent.ScrollY),
            WindowEventKind.Focus => new WindowFocusEvent(nativeEvent.Pressed != 0),
            _ => throw new NotSupportedException($"Unknown native window event kind {nativeEvent.Kind}.")
        };

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}
