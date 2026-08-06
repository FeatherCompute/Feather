using Feather.Native;

namespace Feather;

public sealed class GpuContext : IDisposable
{
    private readonly object gate = new();
    private readonly Lazy<GpuDeviceInfo> device;
    private bool disposed;

    internal GpuContext(FeContextHandle handle, GpuDeviceInfo device)
        : this(handle, () => device)
    {
    }

    private GpuContext(FeContextHandle handle, Func<GpuDeviceInfo> deviceFactory)
    {
        Handle = handle;
        device = new Lazy<GpuDeviceInfo>(deviceFactory);
    }

    internal FeContextHandle Handle { get; }

    /// <summary>
    /// Gets the Luisa device selected for this context.
    /// </summary>
    public GpuDeviceInfo Device
    {
        get
        {
            ThrowIfDisposed();
            return device.Value;
        }
    }

    /// <summary>
    /// Gets the Luisa backend selected for this context.
    /// </summary>
    public GpuBackend Backend => Device.Backend;

    /// <summary>Creates an independent compute stream owned by this context.</summary>
    public GpuStream CreateStream()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_stream_create(Handle, out var stream));
        return new GpuStream(this, stream);
    }

    public BackendType BackendType
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_backend_type(Handle, out var backend));
            return (BackendType)backend;
        }
    }

    /// <summary>
    /// Gets the active EasyGPU backend capabilities reported by the native runtime.
    /// </summary>
    public BackendCaps Caps
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_caps(Handle, out var caps));
            return new BackendCaps(
                (BackendType)caps.BackendType,
                caps.MaxWorkGroupSizeX,
                caps.MaxWorkGroupSizeY,
                caps.MaxWorkGroupSizeZ,
                caps.SupportsGraphics != 0,
                caps.SupportsAD != 0,
                caps.SupportsNN != 0,
                caps.SupportsDepthClamp != 0,
                caps.SupportsNonFillPolygonMode != 0);
        }
    }

    public static GpuContext GetDefault()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var handle));
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(handle));
        return new GpuContext(handle, () => GpuRuntime.GetContextDevice(handle));
    }

    public void Dispose()
    {
        lock (gate)
        {
            if (disposed)
            {
                return;
            }

            _ = NativeMethods.fe_context_shutdown(Handle);
            Handle.SetHandleAsInvalid();
            disposed = true;
        }
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }

    internal bool HasSameNativeContext(GpuContext other)
    {
        ArgumentNullException.ThrowIfNull(other);
        ThrowIfDisposed();
        other.ThrowIfDisposed();
        return Handle.RawValue == other.Handle.RawValue;
    }
}

public enum BackendType : uint
{
    Unavailable,
    OpenGL,
    Vulkan
}

/// <summary>
/// Describes native EasyGPU backend limits and feature flags.
/// </summary>
public readonly record struct BackendCaps(
    BackendType BackendType,
    uint MaxWorkGroupSizeX,
    uint MaxWorkGroupSizeY,
    uint MaxWorkGroupSizeZ,
    bool SupportsGraphics,
    bool SupportsAD,
    bool SupportsNN,
    bool SupportsDepthClamp,
    bool SupportsNonFillPolygonMode);
