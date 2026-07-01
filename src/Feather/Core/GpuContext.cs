using Feather.Native;

namespace Feather;

public sealed class GpuContext : IDisposable
{
    private readonly object gate = new();
    private bool disposed;

    internal GpuContext(FeContextHandle handle)
    {
        Handle = handle;
    }

    internal FeContextHandle Handle { get; }

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
        return new GpuContext(handle);
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

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
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
