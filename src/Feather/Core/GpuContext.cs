using Feather.Native;
using Feather.Resources;
using System.Reflection;

namespace Feather;

public sealed class GpuContext : IDisposable
{
    private readonly object gate = new();
    private readonly Dictionary<(Type KernelType, bool AutoDiff), GpuKernel> kernels = [];
    private readonly List<IDisposable> pendingSubmissions = [];
    private readonly List<WeakReference<ReadbackOperation>> readbackOperations = [];
    private int activeOperations;
    private bool disposing;
    private bool disposed;

    internal GpuContext(FeContextHandle handle)
    {
        Handle = handle;
    }

    internal FeContextHandle Handle { get; }
    internal object QueueGate { get; } = new();

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
            FixedUtf8(info.BackendVersion, 64));

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
    string BackendVersion);

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
