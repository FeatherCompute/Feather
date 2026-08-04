using Feather.Interop;
using Feather.Math;
using Feather.Native;

namespace Feather;

public sealed class GpuKernel : IDisposable
{
    private bool disposed;
    private readonly Type kernelType;
    internal delegate byte[] IrTransform(ReadOnlySpan<byte> ir);

    // Test-only hook used to validate native behavior against transformed generated IR without
    // adding public APIs for raw native kernel creation.
    internal static IrTransform? IrTransformForTesting;

    private GpuKernel(FeKernelHandle handle, KernelDescriptor descriptor, Type kernelType)
    {
        Handle = handle;
        Descriptor = descriptor;
        this.kernelType = kernelType;
    }

    internal FeKernelHandle Handle { get; }
    public KernelDescriptor Descriptor { get; }

    /// <summary>
    /// Gets the native route used by this kernel's most recent dispatch.
    /// </summary>
    public DispatchPath LastDispatchPath
    {
        get
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_last_dispatch_path(Handle, out var path));
            return (DispatchPath)path;
        }
    }

    public static GpuKernel Create<TKernel>(GpuContext context)
        where TKernel : struct, IGeneratedKernel<TKernel>
        => Create<TKernel>(context, TKernel.Descriptor.AutoDiff);

    internal static GpuKernel Create<TKernel>(GpuContext context, bool autoDiff)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var descriptor = TKernel.Descriptor;
        var transformedIr = IrTransformForTesting?.Invoke(TKernel.IR);
        var ir = transformedIr is null ? TKernel.IR : transformedIr.AsSpan();
        unsafe
        {
            fixed (byte* irPtr = ir)
            {
                var createDesc = new FeKernelCreateDesc(
                    (IntPtr)irPtr,
                    (ulong)ir.Length,
                    descriptor.DebugName,
                    autoDiff,
                    descriptor.BoundsCheck);
                NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_create_from_ir(context.Handle, in createDesc, out var handle));
                return new GpuKernel(handle, descriptor, typeof(TKernel));
            }
        }
    }

    public static void Dispatch<TKernel>(GpuContext context, TKernel kernel, GpuDispatchSize size, bool wait)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var gpuKernel = Create<TKernel>(context);
        Dispatch(context, gpuKernel, kernel, size, wait);
    }

    /// <summary>
    /// Dispatches a generated kernel through an existing compiled GPU kernel. The caller retains
    /// ownership of <paramref name="gpuKernel"/> and may reuse it for later dispatches of the same
    /// <typeparamref name="TKernel"/> type.
    /// </summary>
    /// <param name="context">The GPU context that created <paramref name="gpuKernel"/>.</param>
    /// <param name="gpuKernel">A live kernel created with <see cref="Create{TKernel}(GpuContext)"/>.</param>
    /// <param name="kernel">The generated kernel value whose resources and uniforms will be bound.</param>
    /// <param name="size">The logical dispatch extent.</param>
    /// <param name="wait">Whether to wait for the submitted GPU work to complete.</param>
    /// <typeparam name="TKernel">The generated kernel type used to create <paramref name="gpuKernel"/>.</typeparam>
    /// <exception cref="ArgumentException">
    /// <paramref name="gpuKernel"/> was created for a different generated kernel type.
    /// </exception>
    public static void Dispatch<TKernel>(
        GpuContext context,
        GpuKernel gpuKernel,
        TKernel kernel,
        GpuDispatchSize size,
        bool wait)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(gpuKernel);
        gpuKernel.ThrowIfDisposed();
        if (gpuKernel.kernelType != typeof(TKernel))
        {
            throw new ArgumentException(
                $"GPU kernel was created for '{gpuKernel.kernelType.FullName}', not "
                + $"'{typeof(TKernel).FullName}'.",
                nameof(gpuKernel));
        }
        var command = new GpuKernelCommand(gpuKernel.Handle);
        TKernel.Bind(in kernel, command);
        var groups = ComputeGroups(size, TKernel.Descriptor.ThreadGroupSize);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_dispatch(
            gpuKernel.Handle,
            (uint)groups.X,
            (uint)groups.Y,
            (uint)groups.Z,
            (uint)size.X,
            (uint)size.Y,
            (uint)size.Z,
            wait));
    }

    /// <summary>
    /// Creates a generated compute kernel for an explicitly selected execution backend.
    /// </summary>
    public static GpuKernel Create<TKernel>(GpuContext context, GpuExecutionBackend backend)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        if (!Enum.IsDefined(backend))
        {
            throw new ArgumentOutOfRangeException(nameof(backend));
        }

        var kernel = Create<TKernel>(context);
        try
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_set_execution_backend(
                kernel.Handle,
                (FeExecutionBackend)backend));
            return kernel;
        }
        catch
        {
            kernel.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Builds this generated kernel through the EasyGPU IR module bridge and returns the resulting GLSL source.
    /// </summary>
    /// <returns>The GLSL source produced by EasyGPU for this kernel.</returns>
    public string GetGLSL()
    {
        ThrowIfDisposed();
        return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_glsl(Handle, buffer, length, out required));
    }

    /// <summary>
    /// Builds this generated kernel through the EasyGPU IR module bridge and returns the backend-optimized GLSL inspection dump.
    /// </summary>
    /// <returns>The optimized GLSL produced by the active EasyGPU backend.</returns>
    public string GetOptimizedGLSL()
    {
        ThrowIfDisposed();
        return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimized_glsl(Handle, buffer, length, out required));
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Handle.Dispose();
        disposed = true;
    }

    private static int3 ComputeGroups(GpuDispatchSize dispatch, int3 group)
        => new(
            DivRoundUp(dispatch.X, group.X),
            DivRoundUp(dispatch.Y, group.Y),
            DivRoundUp(dispatch.Z, group.Z));

    private static int DivRoundUp(int value, int divisor)
        => divisor <= 0 ? throw new ArgumentOutOfRangeException(nameof(divisor)) : (value + divisor - 1) / divisor;

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}

/// <summary>
/// Provides resource and push-constant binding operations for a generated compute kernel dispatch.
/// </summary>
public sealed class GpuKernelCommand
{
    internal GpuKernelCommand(FeKernelHandle handle)
    {
        Handle = handle;
    }

    internal FeKernelHandle Handle { get; }

    /// <summary>
    /// Binds a native buffer handle to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="buffer">The native buffer handle.</param>
    public void BindBuffer(uint binding, Native.FeBufferHandle buffer)
        => NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_buffer(Handle, binding, buffer));

    /// <summary>
    /// Binds a native texture handle to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="texture">The native texture handle.</param>
    public void BindTexture(uint binding, Native.FeTextureHandle texture)
        => NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_texture(Handle, binding, texture));

    /// <summary>
    /// Binds a native sampler handle to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="sampler">The native sampler handle.</param>
    public void BindSampler(uint binding, Native.FeSamplerHandle sampler)
        => NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_sampler(Handle, binding, sampler));

    /// <summary>
    /// Uploads the complete push-constant byte block for the current generated kernel.
    /// </summary>
    /// <param name="data">The packed push-constant bytes.</param>
    public unsafe void SetPushConstants(ReadOnlySpan<byte> data)
    {
        fixed (byte* ptr = data)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_set_push_constants(Handle, (IntPtr)ptr, (ulong)data.Length));
        }
    }
}
