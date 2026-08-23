using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

public sealed class GpuKernel : IDisposable
{
    private bool disposed;
    private readonly Type kernelType;
    private readonly GpuContext context;
    private readonly object dispatchGate = new();
    private IGpuDiagnosticCapture? diagnosticCapture;
    internal delegate byte[] IrTransform(ReadOnlySpan<byte> ir);

    // Test-only hook used to validate native behavior against transformed generated IR without
    // adding public APIs for raw native kernel creation.
    internal static IrTransform? IrTransformForTesting;
    // Holds the post-submit window open so tests can verify disposal waits for lease tracking.
    internal static Action? DispatchSubmittedForTesting;

    private GpuKernel(GpuContext context, FeKernelHandle handle, KernelDescriptor descriptor, Type kernelType)
    {
        this.context = context;
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
            using var operation = context.EnterOperation();
            lock (dispatchGate)
            {
                ThrowIfDisposed();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_last_dispatch_path(Handle, out var path));
                return (DispatchPath)path;
            }
        }
    }

    /// <summary>
    /// Gets the number of backend pipelines actually compiled for this kernel handle.
    /// </summary>
    public ulong CompilationCount
    {
        get
        {
            using var operation = context.EnterOperation();
            lock (dispatchGate)
            {
                ThrowIfDisposed();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_compile_count(Handle, out var count));
                return count;
            }
        }
    }

    public static GpuKernel Create<TKernel>(GpuContext context)
        where TKernel : struct, IGeneratedKernel<TKernel>
        => Create<TKernel>(context, TKernel.Descriptor.AutoDiff);

    internal static GpuKernel Create<TKernel>(GpuContext context, bool autoDiff)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(context);
        using var operation = context.EnterOperation();
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
                return new GpuKernel(context, handle, descriptor, typeof(TKernel));
            }
        }
    }

    internal static GpuKernel CreateExecutionHeat<TKernel>(
        GpuContext context,
        GpuExecutionHeatCapture capture,
        bool autoDiff)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(capture);
        if (autoDiff)
        {
            throw new NotSupportedException(
                "Execution-heat diagnostics do not yet support differentiable kernels.");
        }

        var kernel = Create<TKernel>(context, autoDiff: false);
        try
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_configure_diagnostics(
                kernel.Handle,
                (uint)FeKernelDiagnosticMode.ExecutionHeat));
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_diagnostic_layout(
                kernel.Handle,
                out var layout));
            capture.AttachKernel(kernel, layout);
            kernel.diagnosticCapture = capture;
            return kernel;
        }
        catch
        {
            kernel.Dispose();
            throw;
        }
    }

    internal static GpuKernel CreateLineValue<TKernel>(
        GpuContext context,
        GpuLineValueCapture capture,
        bool autoDiff)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(capture);
        if (autoDiff)
            throw new NotSupportedException(
                "Line-value diagnostics do not yet support differentiable kernels.");

        var kernel = Create<TKernel>(context, autoDiff: false);
        try
        {
            var config = new FeKernelDiagnosticConfigV2(
                GpuLineValueCapture.AbiVersion,
                FeKernelDiagnosticMode.LineValue,
                capture.SourceSiteIndex,
                checked((uint)capture.SelectedInvocation.X),
                checked((uint)capture.SelectedInvocation.Y),
                checked((uint)capture.SelectedInvocation.Z),
                recordCapacity: 1,
                flags: 0);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_configure_diagnostics_v2(
                kernel.Handle,
                in config));
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_diagnostic_layout_v2(
                kernel.Handle,
                out var layout));
            capture.AttachKernel(kernel, layout);
            kernel.diagnosticCapture = capture;
            return kernel;
        }
        catch
        {
            kernel.Dispose();
            throw;
        }
    }

    internal static GpuKernel CreateUbsan<TKernel>(
        GpuContext context,
        GpuUbsanCapture capture,
        bool autoDiff)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(capture);
        if (autoDiff)
            throw new NotSupportedException(
                "GPU UBSan diagnostics do not yet support differentiable kernels.");

        var kernel = Create<TKernel>(context, autoDiff: false);
        try
        {
            var config = new FeKernelDiagnosticConfigV3(
                GpuUbsanCapture.AbiVersion,
                FeKernelDiagnosticMode.Ubsan,
                checked((uint)capture.RecordCapacity),
                (uint)capture.EnabledChecks,
                sourceSiteIndex: uint.MaxValue,
                selectedX: 0,
                selectedY: 0,
                selectedZ: 0);
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_configure_diagnostics_v3(
                kernel.Handle,
                in config));
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_diagnostic_layout_v3(
                kernel.Handle,
                out var layout));
            capture.AttachKernel(kernel, layout);
            kernel.diagnosticCapture = capture;
            return kernel;
        }
        catch
        {
            kernel.Dispose();
            throw;
        }
    }

    public static void Dispatch<TKernel>(GpuContext context, TKernel kernel, GpuDispatchSize size, bool wait)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var gpuKernel = context.GetOrCreateKernel<TKernel>();
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
        using var operation = context.EnterOperation();
        lock (context.QueueGate)
        {
            var effectiveWait = wait && context.ActiveTimestampRecorder is null;
            var leases = DispatchCore(context, gpuKernel, kernel, size, effectiveWait);
            if (effectiveWait)
            {
                DisposeLeases(leases);
                context.CompleteSubmittedWork();
            }
            else
            {
                context.TrackSubmission(leases);
            }
        }
    }

    internal static List<IDisposable> DispatchForQueue<TKernel>(
        GpuContext context,
        GpuKernel gpuKernel,
        TKernel kernel,
        GpuDispatchSize size)
        where TKernel : struct, IGeneratedKernel<TKernel>
        => DispatchCore(context, gpuKernel, kernel, size, wait: false);

    private static List<IDisposable> DispatchCore<TKernel>(
        GpuContext context,
        GpuKernel gpuKernel,
        TKernel kernel,
        GpuDispatchSize size,
        bool wait)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        lock (gpuKernel.dispatchGate)
        {
            gpuKernel.ThrowIfDisposed();
            if (!ReferenceEquals(gpuKernel.context, context))
            {
                throw new ArgumentException("GPU kernel belongs to a different context.", nameof(gpuKernel));
            }
            if (gpuKernel.kernelType != typeof(TKernel))
            {
                throw new ArgumentException(
                    $"GPU kernel was created for '{gpuKernel.kernelType.FullName}', not "
                    + $"'{typeof(TKernel).FullName}'.",
                    nameof(gpuKernel));
            }

            using var command = new GpuKernelCommand(gpuKernel.Handle);
            TKernel.Bind(in kernel, command);
            gpuKernel.diagnosticCapture?.Bind(
                gpuKernel,
                command,
                size,
                gpuKernel.Descriptor.ThreadGroupSize);
            var groups = ComputeGroups(size, gpuKernel.Descriptor.ThreadGroupSize);
            var recorder = context.ActiveTimestampRecorder;
            using (recorder?.IncludeCommandIntervals == true
                ? recorder.BeginCommand(GpuTimestampIntervalKind.Dispatch, gpuKernel.Descriptor.DebugName)
                : null)
            {
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
            DispatchSubmittedForTesting?.Invoke();
            return command.DetachLeases();
        }
    }

    private static void DisposeLeases(List<IDisposable> leases)
    {
        foreach (var lease in leases)
        {
            lease.Dispose();
        }
    }

    public void Compile()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_compile(Handle));
        }
    }

    /// <summary>
    /// Builds this generated kernel through the EasyGPU IR module bridge and returns the resulting GLSL source.
    /// </summary>
    /// <returns>The GLSL source produced by EasyGPU for this kernel.</returns>
    public string GetGLSL()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_glsl(Handle, buffer, length, out required));
        }
    }

    /// <summary>
    /// Builds this generated kernel through the EasyGPU IR module bridge and returns the backend-optimized GLSL inspection dump.
    /// </summary>
    /// <returns>The optimized GLSL produced by the active EasyGPU backend.</returns>
    public string GetOptimizedGLSL()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimized_glsl(Handle, buffer, length, out required));
        }
    }

    /// <summary>
    /// Builds this generated kernel through the active backend and returns its optimized target IR.
    /// </summary>
    /// <returns>Backend-specific optimized target IR, such as SPIR-V assembly on Vulkan.</returns>
    public string GetOptimizedIR()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimized_ir(Handle, buffer, length, out required));
        }
    }

    /// <summary>
    /// Builds this generated kernel through the active backend and returns its structured optimization report.
    /// </summary>
    /// <returns>A versioned backend-owned JSON report.</returns>
    public string GetOptimizationReport()
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimization_report(Handle, buffer, length, out required));
        }
    }

    internal LoadedShaderSource InspectLoadedShader(bool autoDiff)
    {
        using var operation = context.EnterOperation();
        lock (dispatchGate)
        {
            ThrowIfDisposed();
            var targetBinary = GetOptionalBinary(
                (IntPtr buffer, UIntPtr length, out UIntPtr required, out FeShaderBinaryFormat format) =>
                    NativeMethods.fe_kernel_get_shader_binary(Handle, buffer, length, out required, out format));
            return new LoadedShaderSource(
                kernelType,
                ShaderStage.Compute,
                autoDiff,
                NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_glsl(Handle, buffer, length, out required)),
                GetOptionalString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimized_glsl(Handle, buffer, length, out required)),
                GetOptionalString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimized_ir(Handle, buffer, length, out required)),
                GetOptionalString((IntPtr buffer, UIntPtr length, out UIntPtr required) => NativeMethods.fe_kernel_get_optimization_report(Handle, buffer, length, out required)),
                targetBinary.Format,
                targetBinary.Bytes);
        }
    }

    private static string GetOptionalString(NativeStringCall.Getter getter)
    {
        try
        {
            return NativeStringCall.GetString(getter);
        }
        catch (FeatherNativeException exception) when (exception.Result == FeResult.ErrorUnsupported)
        {
            return string.Empty;
        }
    }

    private static (ShaderBinaryFormat Format, ReadOnlyMemory<byte> Bytes) GetOptionalBinary(NativeByteCall.Getter getter)
    {
        try
        {
            var binary = NativeByteCall.GetBytes(getter);
            return binary.Format switch
            {
                FeShaderBinaryFormat.SpirV when binary.Bytes.Length > 0 =>
                    (ShaderBinaryFormat.SpirV, binary.Bytes),
                FeShaderBinaryFormat.Unavailable when binary.Bytes.Length == 0 =>
                    (ShaderBinaryFormat.Unavailable, ReadOnlyMemory<byte>.Empty),
                _ => throw new InvalidOperationException("Native shader binary format and payload are inconsistent.")
            };
        }
        catch (FeatherNativeException exception) when (exception.Result == FeResult.ErrorUnsupported)
        {
            return (ShaderBinaryFormat.Unavailable, ReadOnlyMemory<byte>.Empty);
        }
    }

    public void Dispose()
    {
        lock (dispatchGate)
        {
            if (disposed)
            {
                return;
            }

            Handle.Dispose();
            disposed = true;
        }
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
public sealed class GpuKernelCommand : IDisposable
{
    private List<IDisposable>? leases = [];

    internal GpuKernelCommand(FeKernelHandle handle)
    {
        Handle = handle;
        leases.Add(new NativeHandleLease(handle));
    }

    internal FeKernelHandle Handle { get; }

    /// <summary>
    /// Binds a shader-facing buffer to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="buffer">The buffer binding.</param>
    public void BindBuffer(uint binding, IGpuBufferBinding buffer)
    {
        var native = buffer as INativeBufferBinding
            ?? throw new ArgumentException("Buffer binding was not created by Feather.", nameof(buffer));
        Retain(native.NativeBufferHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_buffer(Handle, binding, native.NativeBufferHandle));
    }

    /// <summary>
    /// Binds a native texture handle to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="texture">The texture binding.</param>
    public void BindTexture(uint binding, IGpuTextureBinding texture)
    {
        var native = texture as INativeTextureBinding
            ?? throw new ArgumentException("Texture binding was not created by Feather.", nameof(texture));
        Retain(native.NativeTextureHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_texture(Handle, binding, native.NativeTextureHandle));
    }

    /// <summary>
    /// Binds a native sampler handle to a generated kernel resource slot.
    /// </summary>
    /// <param name="binding">The shader binding index.</param>
    /// <param name="sampler">The sampler binding.</param>
    public void BindSampler(uint binding, IGpuSamplerBinding sampler)
    {
        var native = sampler as INativeSamplerBinding
            ?? throw new ArgumentException("Sampler binding was not created by Feather.", nameof(sampler));
        Retain(native.NativeSamplerHandle);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_sampler(Handle, binding, native.NativeSamplerHandle));
    }

    internal void BindDiagnosticBuffer(GpuBuffer<uint> buffer)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        Retain(buffer.GetNativeHandle());
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_bind_diagnostic_buffer(
            Handle,
            buffer.GetNativeHandle()));
    }

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

    internal List<IDisposable> DetachLeases()
    {
        var detached = leases ?? throw new ObjectDisposedException(nameof(GpuKernelCommand));
        leases = null;
        return detached;
    }

    public void Dispose()
    {
        if (leases is null)
        {
            return;
        }
        foreach (var lease in leases)
        {
            lease.Dispose();
        }
        leases = null;
    }

    private void Retain(FeSafeHandle handle)
        => (leases ?? throw new ObjectDisposedException(nameof(GpuKernelCommand))).Add(new NativeHandleLease(handle));
}

internal sealed class NativeHandleLease : IDisposable
{
    private FeSafeHandle? handle;

    public NativeHandleLease(FeSafeHandle handle)
    {
        ArgumentNullException.ThrowIfNull(handle);
        var success = false;
        handle.DangerousAddRef(ref success);
        if (!success)
        {
            throw new ObjectDisposedException(handle.GetType().Name);
        }
        this.handle = handle;
    }

    public void Dispose()
    {
        var retained = Interlocked.Exchange(ref handle, null);
        retained?.DangerousRelease();
    }
}
