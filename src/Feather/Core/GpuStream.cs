using Feather.Interop;
using Feather.Math;
using Feather.Native;

namespace Feather;

/// <summary>Owns an ordered, asynchronous Luisa compute submission queue.</summary>
public sealed class GpuStream : IDisposable
{
    private bool disposed;

    internal GpuStream(GpuContext context, FeStreamHandle handle)
    {
        Context = context;
        Handle = handle;
    }

    internal FeStreamHandle Handle { get; }
    public GpuContext Context { get; }

    public GpuFence Dispatch<TKernel>(TKernel kernel, int x)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(x, 1, 1));

    public GpuFence Dispatch<TKernel>(TKernel kernel, int2 size)
        where TKernel : struct, IKernel2D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(size.X, size.Y, 1));

    public GpuFence Dispatch<TKernel>(TKernel kernel, int3 size)
        where TKernel : struct, IKernel3D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(size.X, size.Y, size.Z));

    public GpuFence Dispatch<TKernel>(GpuKernel compiledKernel, TKernel kernel, GpuDispatchSize size)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ThrowIfDisposed();
        return GpuKernel.Dispatch(this, compiledKernel, kernel, size);
    }

    public void Wait(GpuFence fence)
    {
        ArgumentNullException.ThrowIfNull(fence);
        ThrowIfDisposed();
        fence.ThrowIfDisposed();
        if (!Context.HasSameNativeContext(fence.Context))
        {
            throw new ArgumentException("Cannot wait on a fence from a different GPU context.", nameof(fence));
        }
        NativeMethods.ThrowIfFailed(NativeMethods.fe_stream_wait_fence(Handle, fence.Handle));
    }

    public void Synchronize()
    {
        ThrowIfDisposed();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_stream_synchronize(Handle));
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

    private GpuFence DispatchCore<TKernel>(TKernel kernel, GpuDispatchSize size)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ThrowIfDisposed();
        using var compiled = GpuKernel.Create<TKernel>(Context);
        return GpuKernel.Dispatch(this, compiled, kernel, size);
    }
}
