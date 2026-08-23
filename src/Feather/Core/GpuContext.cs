using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;
using System.Reflection;

namespace Feather;

public sealed class GpuContext : IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<(Type KernelType, bool AutoDiff), GpuKernel> kernels = [];
    private readonly List<WeakReference<ILoadedGraphicsShaderInspection>> graphicsPipelines = [];
    private readonly List<IDisposable> pendingSubmissions = [];
    private readonly List<WeakReference<ReadbackOperation>> readbackOperations = [];
    private GpuTimestampRecorder? activeTimestampRecorder;
    private IGpuDiagnosticCapture? activeDiagnosticCapture;
    private int activeOperations;
    private bool disposing;
    private bool disposed;

    internal GpuContext(FeContextHandle handle)
    {
        Handle = handle;
    }

    internal FeContextHandle Handle { get; }
    internal object QueueGate { get; } = new();

    internal GpuTimestampRecorder? ActiveTimestampRecorder
    {
        get
        {
            if (!Monitor.IsEntered(QueueGate))
            {
                throw new InvalidOperationException("GPU timestamp recorder access requires the queue lock.");
            }
            return activeTimestampRecorder;
        }
        set
        {
            if (!Monitor.IsEntered(QueueGate))
            {
                throw new InvalidOperationException("GPU timestamp recorder access requires the queue lock.");
            }
            activeTimestampRecorder = value;
        }
    }

    internal bool IsDisposed
    {
        get
        {
            lock (gate)
            {
                return disposed;
            }
        }
    }

    internal bool IsDisposing
    {
        get
        {
            lock (gate)
            {
                return disposing;
            }
        }
    }

    public GpuQueue Queue { get; private set; } = null!;

    public BackendType BackendType
    {
        get
        {
            using var operation = EnterOperation();
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
            using var operation = EnterOperation();
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

    /// <summary>
    /// Gets the live adapter, driver, backend, and image-limit identity reported by EasyGPU.
    /// </summary>
    public BackendDeviceInfo DeviceInfo
    {
        get
        {
            using var operation = EnterOperation();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_device_info(Handle, out var info));
            return CreateDeviceInfo(info);
        }
    }

    private static unsafe BackendDeviceInfo CreateDeviceInfo(FeBackendDeviceInfo info)
    {
        return new BackendDeviceInfo(
            info.NativeAbiVersion,
            info.MaxTextureDimension2D,
            info.SupportsTimestampQueries != 0,
            FixedUtf8(info.AdapterName, 256),
            FixedUtf8(info.DriverVersion, 128),
            FixedUtf8(info.BackendVersion, 64),
            new BackendSubgroupInfo(
                ReportedSize: info.Reserved >> 16,
                SupportsComputeStage: (info.Reserved & (1u << 0)) != 0,
                SupportsBasic: (info.Reserved & (1u << 1)) != 0,
                SupportsVote: (info.Reserved & (1u << 2)) != 0,
                SupportsBallot: (info.Reserved & (1u << 3)) != 0));

        static string FixedUtf8(byte* value, int capacity)
        {
            var length = 0;
            while (length < capacity && value[length] != 0)
            {
                length++;
            }

            return System.Text.Encoding.UTF8.GetString(value, length);
        }
    }

    /// <summary>
    /// Gets a point-in-time snapshot of backend synchronization and readback operations.
    /// </summary>
    public BackendOperationCounters OperationCounters
    {
        get
        {
            using var operation = EnterOperation();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_operation_counters(Handle, out var counters));
            return new BackendOperationCounters(
                counters.FinishCalls,
                counters.DeviceWaitIdleCalls,
                counters.GlobalDrainCalls,
                counters.BlockingSubmissionWaitCalls,
                counters.BlockingTextureDownloadCalls,
                counters.AsyncTextureReadbackCalls);
        }
    }

    /// <summary>
    /// Gets exact native handle/cache counts retained by the active backend. A backend that
    /// cannot expose these counts returns a snapshot with <see cref="BackendResourceCounters.TrackingSupported"/>
    /// set to <see langword="false"/> rather than reporting placeholder zeros as evidence.
    /// </summary>
    public BackendResourceCounters ResourceCounters
    {
        get
        {
            using var operation = EnterOperation();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_resource_counters(Handle, out var counters));
            return new BackendResourceCounters(
                counters.TrackingSupported != 0,
                counters.LiveBufferHandles,
                counters.LiveTextureHandles,
                counters.LivePipelineHandles,
                counters.LiveShaderHandles,
                counters.LiveSubmissionHandles,
                counters.CachedDescriptorSets,
                counters.DescriptorPools,
                counters.CachedSamplers,
                counters.CachedSubmissionResources,
                counters.LiveMsaaAttachments);
        }
    }

    /// <summary>
    /// Gets cumulative backend shader frontend/optimizer cache counters. These are exact
    /// backend observations, not inferred from a Studio descriptor key or pass duration.
    /// </summary>
    public BackendShaderCacheCounters ShaderCacheCounters
    {
        get
        {
            using var operation = EnterOperation();
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_shader_cache_counters(Handle, out var counters));
            return new BackendShaderCacheCounters(
                counters.TrackingSupported != 0,
                counters.MemoryCacheHits,
                counters.DiskCacheHits,
                counters.DiskCacheMisses,
                counters.FrontendCompilations,
                counters.DiskCacheWriteFailures,
                counters.LastFrontendMilliseconds,
                counters.LastOptimizationMilliseconds,
                counters.LastMemoryCacheHit != 0,
                counters.LastDiskCacheHit != 0);
        }
    }

    public static GpuContext GetDefault()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_get_default(out var handle));
        NativeMethods.ThrowIfFailed(NativeMethods.fe_context_initialize(handle));
        var context = new GpuContext(handle);
        context.Queue = new GpuQueue(context);
        return context;
    }

    public void Compile<TKernel>()
        where TKernel : struct, Interop.IGeneratedKernel<TKernel>
        => GetOrCreateKernel<TKernel>().Compile();

    public void WaitIdle()
    {
        using var operation = EnterOperation();
        lock (QueueGate)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_context_wait_idle(Handle));
            CompleteSubmittedWork();
        }
    }

    public void Dispose()
    {
        var releaseDefault = false;
        lock (gate)
        {
            while (disposing && !disposed)
            {
                Monitor.Wait(gate);
            }
            if (disposed)
            {
                return;
            }

            disposing = true;
            while (activeOperations != 0)
            {
                Monitor.Wait(gate);
            }

            try
            {
                CancelReadbacksForShutdownLocked();
                NativeMethods.ThrowIfFailed(NativeMethods.fe_context_wait_idle(Handle));
                DisposePendingSubmissionsLocked();
                activeDiagnosticCapture?.DisposeForContextShutdown();
                activeDiagnosticCapture = null;
                foreach (var kernel in kernels.Values)
                {
                    kernel.Dispose();
                }
                kernels.Clear();

                _ = NativeMethods.fe_context_shutdown(Handle);
                Handle.SetHandleAsInvalid();
                disposed = true;
                releaseDefault = true;
            }
            finally
            {
                disposing = false;
                Monitor.PulseAll(gate);
            }
        }

        if (releaseDefault)
        {
            GPU.ReleaseDefaultContext(this);
        }
    }

    internal GpuKernel GetOrCreateKernel<TKernel>(bool? autoDiff = null)
        where TKernel : struct, Interop.IGeneratedKernel<TKernel>
    {
        lock (gate)
        {
            ThrowIfDisposed();
            var resolvedAutoDiff = autoDiff ?? TKernel.Descriptor.AutoDiff;
            if (activeDiagnosticCapture is { } capture &&
                capture.TryGetOrCreateKernel<TKernel>(resolvedAutoDiff, out var diagnosticKernel))
            {
                return diagnosticKernel;
            }
            var key = (typeof(TKernel), resolvedAutoDiff);
            if (!kernels.TryGetValue(key, out var kernel))
            {
                kernel = GpuKernel.Create<TKernel>(this, resolvedAutoDiff);
                kernels.Add(key, kernel);
            }

            return kernel;
        }
    }

    /// <summary>
    /// Arms one explicitly bounded execution-heat diagnostic variant for the named generated
    /// compute shader type. Ordinary cached kernels remain untouched; only matching dispatches
    /// recorded while the returned scope is active use the instrumented variant.
    /// </summary>
    public GpuExecutionHeatCapture BeginExecutionHeatCapture(string shaderTypeName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
            {
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            }
            var capture = new GpuExecutionHeatCapture(this, shaderTypeName);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    /// <summary>
    /// Arms one selected-invocation typed line-value variant. Only the named generated compute
    /// shader is substituted and only while the returned capture scope remains active.
    /// </summary>
    public GpuLineValueCapture BeginLineValueCapture(
        string shaderTypeName,
        uint sourceSiteIndex,
        int targetDispatchIndex,
        int3 selectedInvocation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            var capture = new GpuLineValueCapture(
                this,
                shaderTypeName,
                sourceSiteIndex,
                targetDispatchIndex,
                selectedInvocation);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    /// <summary>
    /// Arms one bounded compiler-instrumented GPU UBSan variant for an exact matching compute
    /// dispatch. Ordinary cached kernels and non-diagnostic dispatches remain untouched.
    /// </summary>
    public GpuUbsanCapture BeginUbsanCapture(
        string shaderTypeName,
        int targetDispatchIndex,
        int recordCapacity = 256,
        GpuUbsanChecks enabledChecks = GpuUbsanChecks.All)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            var capture = new GpuUbsanCapture(
                this,
                shaderTypeName,
                targetDispatchIndex,
                recordCapacity,
                enabledChecks);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    /// <summary>
    /// Arms one bounded user-authored Print/Assert stream and dispatch-wide assertion mask for an
    /// exact matching compute dispatch. Ordinary cached kernels remain uninstrumented.
    /// </summary>
    public GpuPrintAssertCapture BeginPrintAssertCapture(
        string shaderTypeName,
        int targetDispatchIndex,
        GpuDispatchSize logicalSize,
        int recordCapacity = 256,
        GpuPrintAssertFilterMode filterMode = GpuPrintAssertFilterMode.AllInvocations,
        int3 selectedInvocation = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            var capture = new GpuPrintAssertCapture(
                this,
                shaderTypeName,
                targetDispatchIndex,
                logicalSize,
                recordCapacity,
                filterMode,
                selectedInvocation);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    /// <summary>
    /// Arms one profile-only subgroup predicate capture at a retained, converged top-level compute
    /// branch. Ordinary cached kernels and non-diagnostic dispatches remain uninstrumented.
    /// </summary>
    public GpuBranchDivergenceCapture BeginBranchDivergenceCapture(
        string shaderTypeName,
        uint sourceSiteIndex,
        int targetDispatchIndex,
        int recordCapacity = 256)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        BackendSubgroupInfo subgroups = DeviceInfo.Subgroups;
        if (!subgroups.SupportsBranchDivergence)
        {
            throw new NotSupportedException(
                "Branch-divergence capture requires compute basic/vote/ballot subgroup support.");
        }
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            var capture = new GpuBranchDivergenceCapture(
                this,
                shaderTypeName,
                sourceSiteIndex,
                targetDispatchIndex,
                recordCapacity,
                subgroups);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    /// <summary>
    /// Arms one bounded compiler-instrumented event trace for an exact selected compute
    /// invocation and matching dispatch. Ordinary cached kernels remain uninstrumented.
    /// </summary>
    public GpuComputeTraceCapture BeginComputeTraceCapture(
        string shaderTypeName,
        int targetDispatchIndex,
        int3 selectedInvocation,
        int recordCapacity = 1_024)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(shaderTypeName);
        lock (gate)
        {
            ThrowIfDisposed();
            if (activeDiagnosticCapture is not null)
                throw new InvalidOperationException(
                    "A GPU diagnostic capture is already active on this context.");
            var capture = new GpuComputeTraceCapture(
                this,
                shaderTypeName,
                targetDispatchIndex,
                selectedInvocation,
                recordCapacity);
            activeDiagnosticCapture = capture;
            return capture;
        }
    }

    internal void EndDiagnosticCapture(IGpuDiagnosticCapture capture)
    {
        lock (gate)
        {
            if (ReferenceEquals(activeDiagnosticCapture, capture))
            {
                activeDiagnosticCapture = null;
            }
        }
    }

    /// <summary>
    /// Releases cached generated kernels whose managed kernel types belong to one assembly.
    /// Render hosts call this after the assembly generation's GPU submissions complete and
    /// before unloading a collectible AssemblyLoadContext; otherwise the context cache would
    /// retain the generated Type objects and pin that load context indefinitely.
    /// </summary>
    /// <param name="assembly">The retiring generated-pass assembly.</param>
    /// <returns>The number of cached kernel variants released.</returns>
    public int ReleaseCachedKernels(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        using var operation = EnterOperation();
        List<GpuKernel> released;
        lock (gate)
        {
            var keys = kernels.Keys
                .Where(key => ReferenceEquals(key.KernelType.Assembly, assembly))
                .ToArray();
            released = new List<GpuKernel>(keys.Length);
            foreach (var key in keys)
            {
                released.Add(kernels[key]);
                kernels.Remove(key);
            }
        }

        foreach (var kernel in released)
        {
            kernel.Dispose();
        }
        return released.Count;
    }

    internal IReadOnlyList<LoadedShaderSource> GetLoadedShaders(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        using var operation = EnterOperation();

        (GpuKernel Kernel, bool AutoDiff)[] computeShaders;
        ILoadedGraphicsShaderInspection[] graphicsShaders;
        lock (gate)
        {
            computeShaders = kernels
                .Where(entry => ReferenceEquals(entry.Key.KernelType.Assembly, assembly))
                .Select(static entry => (entry.Value, entry.Key.AutoDiff))
                .ToArray();
            PruneGraphicsPipelinesLocked(remove: null);
            graphicsShaders = graphicsPipelines
                .Select(static weak => weak.TryGetTarget(out var pipeline) ? pipeline : null)
                .OfType<ILoadedGraphicsShaderInspection>()
                .ToArray();
        }

        var loaded = new List<LoadedShaderSource>(computeShaders.Length + graphicsShaders.Length * 2);
        foreach (var (kernel, autoDiff) in computeShaders)
        {
            try
            {
                loaded.Add(kernel.InspectLoadedShader(autoDiff));
            }
            catch (ObjectDisposedException)
            {
                // The cache can retire concurrently with a point-in-time inspection snapshot.
            }
        }

        foreach (var pipeline in graphicsShaders)
        {
            try
            {
                loaded.AddRange(pipeline.InspectLoadedShaders().Where(shader => ReferenceEquals(shader.SourceType.Assembly, assembly)));
            }
            catch (ObjectDisposedException)
            {
                // A weakly registered pipeline can be disposed after the registry snapshot.
            }
            catch (FeatherNativeException exception) when (exception.Result == FeResult.ErrorUnsupported)
            {
                // A pipeline has no backend shader until its first typed draw completes.
            }
        }

        return loaded
            .OrderBy(static shader => shader.SourceType.FullName, StringComparer.Ordinal)
            .ThenBy(static shader => shader.Stage)
            .ThenBy(static shader => shader.AutoDiff)
            .ToArray();
    }

    internal void RegisterGraphicsPipeline(ILoadedGraphicsShaderInspection pipeline)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed || disposing, this);
            PruneGraphicsPipelinesLocked(pipeline);
            graphicsPipelines.Add(new WeakReference<ILoadedGraphicsShaderInspection>(pipeline));
        }
    }

    internal void UnregisterGraphicsPipeline(ILoadedGraphicsShaderInspection pipeline)
    {
        lock (gate)
        {
            PruneGraphicsPipelinesLocked(pipeline);
        }
    }

    internal void TrackSubmission(IEnumerable<IDisposable> leases)
    {
        lock (gate)
        {
            pendingSubmissions.AddRange(leases);
        }
    }

    internal void RegisterReadback(ReadbackOperation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed || disposing, this);
            PruneReadbacksLocked(operation);
            readbackOperations.Add(new WeakReference<ReadbackOperation>(operation));
        }
    }

    internal void UnregisterReadback(ReadbackOperation operation)
    {
        lock (gate)
        {
            PruneReadbacksLocked(operation);
        }
    }

    internal void TransferSubmittedWorkTo(List<IDisposable> destination)
    {
        lock (gate)
        {
            destination.AddRange(pendingSubmissions);
            pendingSubmissions.Clear();
        }
    }

    internal void CompleteSubmittedWork()
    {
        lock (gate)
        {
            DisposePendingSubmissionsLocked();
        }
    }

    internal void ThrowIfDisposed()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed || disposing, this);
        }
    }

    internal OperationLease EnterOperation()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed || disposing, this);
            activeOperations++;
            return new OperationLease(this);
        }
    }

    private void DisposePendingSubmissionsLocked()
    {
        foreach (var lease in pendingSubmissions)
        {
            lease.Dispose();
        }
        pendingSubmissions.Clear();
    }

    private void CancelReadbacksForShutdownLocked()
    {
        var operations = new List<ReadbackOperation>(readbackOperations.Count);
        foreach (var weak in readbackOperations)
        {
            if (weak.TryGetTarget(out var operation))
            {
                operations.Add(operation);
            }
        }
        readbackOperations.Clear();

        foreach (var operation in operations)
        {
            operation.CancelForContextShutdown();
        }
    }

    private void PruneReadbacksLocked(ReadbackOperation? remove)
    {
        for (var index = readbackOperations.Count - 1; index >= 0; index--)
        {
            if (!readbackOperations[index].TryGetTarget(out var operation) || ReferenceEquals(operation, remove))
            {
                readbackOperations.RemoveAt(index);
            }
        }
    }

    private void PruneGraphicsPipelinesLocked(ILoadedGraphicsShaderInspection? remove)
    {
        for (var index = graphicsPipelines.Count - 1; index >= 0; index--)
        {
            if (!graphicsPipelines[index].TryGetTarget(out var pipeline) || ReferenceEquals(pipeline, remove))
            {
                graphicsPipelines.RemoveAt(index);
            }
        }
    }

    private void ExitOperation()
    {
        lock (gate)
        {
            activeOperations--;
            if (activeOperations == 0)
            {
                Monitor.PulseAll(gate);
            }
        }
    }

    internal sealed class OperationLease : IDisposable
    {
        private GpuContext? context;

        internal OperationLease(GpuContext context)
        {
            this.context = context;
        }

        public void Dispose()
        {
            Interlocked.Exchange(ref context, null)?.ExitOperation();
        }
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

/// <summary>
/// Stable identity and image limits for the initialized native backend.
/// </summary>
public readonly record struct BackendDeviceInfo(
    uint NativeAbiVersion,
    uint MaxTextureDimension2D,
    bool SupportsTimestampQueries,
    string AdapterName,
    string DriverVersion,
    string BackendVersion,
    BackendSubgroupInfo Subgroups = default);

/// <summary>Explicit compute subgroup capability retained from the initialized backend.</summary>
public readonly record struct BackendSubgroupInfo(
    uint ReportedSize,
    bool SupportsComputeStage,
    bool SupportsBasic,
    bool SupportsVote,
    bool SupportsBallot)
{
    /// <summary>Whether branch-divergence ballot instrumentation can run on this backend.</summary>
    public bool SupportsBranchDivergence =>
        ReportedSize > 0 && SupportsComputeStage && SupportsBasic && SupportsVote && SupportsBallot;
}

/// <summary>
/// Backend counters used to prove that asynchronous paths avoid global drains and blocking downloads.
/// </summary>
public readonly record struct BackendOperationCounters(
    ulong FinishCalls,
    ulong DeviceWaitIdleCalls,
    ulong GlobalDrainCalls,
    ulong BlockingSubmissionWaitCalls,
    ulong BlockingTextureDownloadCalls,
    ulong AsyncTextureReadbackCalls);

/// <summary>
/// Exact native handle/cache counts retained by a backend at one point in time.
/// </summary>
public readonly record struct BackendResourceCounters(
    bool TrackingSupported,
    ulong LiveBufferHandles,
    ulong LiveTextureHandles,
    ulong LivePipelineHandles,
    ulong LiveShaderHandles,
    ulong LiveSubmissionHandles,
    ulong CachedDescriptorSets,
    ulong DescriptorPools,
    ulong CachedSamplers,
    ulong CachedSubmissionResources,
    ulong LiveMsaaAttachments);

/// <summary>
/// Exact cumulative shader compilation/cache counters reported by the native backend.
/// </summary>
public readonly record struct BackendShaderCacheCounters(
    bool TrackingSupported,
    ulong MemoryCacheHits,
    ulong DiskCacheHits,
    ulong DiskCacheMisses,
    ulong FrontendCompilations,
    ulong DiskCacheWriteFailures,
    double LastFrontendMilliseconds,
    double LastOptimizationMilliseconds,
    bool LastMemoryCacheHit,
    bool LastDiskCacheHit);
