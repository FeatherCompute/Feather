using Microsoft.Win32.SafeHandles;

namespace Feather.Native;

public abstract class FeSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    protected FeSafeHandle() : base(true)
    {
    }

    protected FeSafeHandle(IntPtr invalidHandle) : base(true)
    {
        SetHandle(invalidHandle);
    }

    public ulong RawValue => (ulong)handle;
}

public sealed class FeContextHandle : FeSafeHandle
{
    protected override bool ReleaseHandle()
    {
        // ReleaseHandle runs while the SafeHandle is already being disposed/finalized, so use the raw native handle
        // instead of passing this SafeHandle back through P/Invoke and triggering a second DangerousAddRef.
        return IsInvalid || NativeMethods.fe_context_shutdown_raw(handle).Succeeded();
    }
}

public sealed class FeBufferHandle : FeSafeHandle
{
    public static FeBufferHandle Null { get; } = new();

    protected override bool ReleaseHandle() => IsInvalid || NativeMethods.fe_buffer_destroy(handle).Succeeded();
}

public sealed class FeTextureHandle : FeSafeHandle
{
    public static FeTextureHandle Null { get; } = new();

    protected override bool ReleaseHandle() => IsInvalid || NativeMethods.fe_texture_destroy(handle).Succeeded();
}

public sealed class FeSamplerHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => NativeMethods.fe_sampler_destroy(handle).Succeeded();
}

public sealed class FeKernelHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => NativeMethods.fe_kernel_destroy(handle).Succeeded();
}

public sealed class FeGraphicsPipelineHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => NativeMethods.fe_graphics_pipeline_destroy(handle).Succeeded();
}

public sealed class FeADKernelHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => true;
}

public sealed class FeTensorHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => true;
}

public sealed class FeWindowHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => IsInvalid || NativeMethods.fe_window_destroy(handle).Succeeded();
}

public sealed class FeTexturePresenterHandle : FeSafeHandle
{
    protected override bool ReleaseHandle() => IsInvalid || NativeMethods.fe_texture_presenter_destroy(handle).Succeeded();
}
