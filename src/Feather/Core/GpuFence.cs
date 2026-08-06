using Feather.Native;

namespace Feather;

/// <summary>Represents completion of work submitted to a <see cref="GpuStream"/>.</summary>
public sealed class GpuFence : IDisposable
{
    private bool disposed;

    internal GpuFence(GpuContext context, FeFenceHandle handle)
    {
        Context = context;
        Handle = handle;
    }

    internal FeFenceHandle Handle { get; }
    public GpuContext Context { get; }

    public bool IsCompleted
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_is_completed(Handle, out var completed));
            return completed;
        }
    }

    public void Wait()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_fence_wait(Handle));
    }

    public async ValueTask WaitAsync(CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        await Task.Run(Wait, cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (disposed) return;
        Handle.Dispose();
        disposed = true;
    }

    internal void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        Context.ThrowIfDisposed();
    }
}
