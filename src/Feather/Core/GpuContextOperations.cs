using Feather.Interop;
using Feather.Math;
using Feather.Resources;

namespace Feather;

/// <summary>
/// Provides resource creation and dispatch operations bound to one explicit GPU context.
/// The facade does not change the process-default <see cref="GPU.Context"/>.
/// </summary>
public sealed class GpuContextOperations
{
    internal GpuContextOperations(GpuContext context)
    {
        Context = context;
    }

    /// <summary>Gets the context that owns resources and kernels created through this facade.</summary>
    public GpuContext Context { get; }

    public GpuBuffer<T> CreateBuffer<T>(int count, BufferAccess access = BufferAccess.ReadWrite)
        where T : unmanaged
    {
        Context.ThrowIfDisposed();
        return GpuBuffer<T>.Create(Context, count, access);
    }

    public GpuBuffer<T> CreateBuffer<T>(ReadOnlySpan<T> data, BufferAccess access = BufferAccess.ReadWrite)
        where T : unmanaged
    {
        Context.ThrowIfDisposed();
        return GpuBuffer<T>.Create(Context, data, access);
    }

    public DispatchPath Dispatch<TKernel>(TKernel kernel, int x, bool wait = true)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(x, 1, 1), wait);

    public DispatchPath Dispatch<TKernel>(TKernel kernel, int2 size, bool wait = true)
        where TKernel : struct, IKernel2D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(size.X, size.Y, 1), wait);

    public DispatchPath Dispatch<TKernel>(TKernel kernel, int3 size, bool wait = true)
        where TKernel : struct, IKernel3D, IGeneratedKernel<TKernel>
        => DispatchCore(kernel, new GpuDispatchSize(size.X, size.Y, size.Z), wait);

    private DispatchPath DispatchCore<TKernel>(TKernel kernel, GpuDispatchSize size, bool wait)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        Context.ThrowIfDisposed();
        using var compiled = GpuKernel.Create<TKernel>(Context);
        GpuKernel.Dispatch(Context, compiled, kernel, size, wait);
        return compiled.LastDispatchPath;
    }
}
