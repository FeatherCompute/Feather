using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>One non-zero source-site counter from an execution-heat diagnostic dispatch.</summary>
public readonly record struct GpuExecutionHeatSite(uint SiteIndex, uint HitCount);

/// <summary>
/// Immutable geometry for one generated compute dispatch observed by an execution-heat capture.
/// Logical extents exclude padded lanes; workgroup counts are the exact native dispatch group
/// counts derived from the generated kernel's thread-group size.
/// </summary>
public readonly record struct GpuExecutionHeatDispatch(
    int DispatchIndex,
    int LogicalSizeX,
    int LogicalSizeY,
    int LogicalSizeZ,
    int ThreadGroupSizeX,
    int ThreadGroupSizeY,
    int ThreadGroupSizeZ,
    int WorkgroupCountX,
    int WorkgroupCountY,
    int WorkgroupCountZ);

/// <summary>
/// Immutable result of an instrumented compute capture. Counts are execution frequency, not
/// nanoseconds, hardware PC samples, or an estimate of source cost.
/// </summary>
public sealed record GpuExecutionHeatResult(
    string ShaderTypeName,
    int MatchedDispatchCount,
    int DispatchRecordCapacity,
    int DroppedDispatchRecordCount,
    IReadOnlyList<GpuExecutionHeatDispatch> Dispatches,
    int SiteCapacity,
    IReadOnlyList<GpuExecutionHeatSite> Sites);

/// <summary>
/// Context-scoped diagnostic shader substitution. The caller must complete the queue submission
/// that contains every matching dispatch before calling <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuExecutionHeatCapture : IDisposable
{
    public const uint AbiVersion = 1;
    public const int MaximumSites = 65_536;
    public const int MaximumDispatchRecords = 4_096;

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private GpuKernel? kernel;
    private GpuBuffer<uint>? counters;
    private readonly List<GpuExecutionHeatDispatch> dispatches = [];
    private int matchedDispatchCount;
    private int droppedDispatchRecordCount;
    private int siteCount;
    private bool completed;
    private bool disposed;

    internal GpuExecutionHeatCapture(GpuContext context, string shaderTypeName)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        if (this.shaderTypeName.Length == 0)
        {
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        }
    }

    public string ShaderTypeName => shaderTypeName;

    public bool Matched
    {
        get { lock (gate) { return matchedDispatchCount > 0; } }
    }

    public int SiteCount
    {
        get { lock (gate) { return siteCount; } }
    }

    /// <summary>
    /// Allocates and queues initialization of the exact diagnostic counter layout before command
    /// timestamp intervals begin. The returned fence must complete before a matching dispatch.
    /// This is an explicit-capture operation; ordinary kernels never call it or allocate the
    /// diagnostic buffer. The count must come from the matching generated source-map contract.
    /// </summary>
    public GpuFence PrepareCounterLayout(int expectedSiteCount)
    {
        if (expectedSiteCount is < 1 or > MaximumSites)
            throw new ArgumentOutOfRangeException(nameof(expectedSiteCount));

        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || kernel is not null || counters is not null)
            {
                throw new InvalidOperationException(
                    "The execution-heat counter layout can only be prepared once before dispatch.");
            }
        }

        GpuBuffer<uint>? prepared = GpuBuffer<uint>.Create(
            context,
            expectedSiteCount,
            BufferAccess.ReadWrite);
        using var zeroSource = GpuBuffer<uint>.Create(
            context,
            new uint[expectedSiteCount],
            BufferAccess.ReadOnly);
        using var commands = context.Queue.CreateCommandList();
        commands.CopyBuffer(zeroSource, prepared);
        commands.Close();
        GpuFence? preparationFence = null;
        try
        {
            preparationFence = context.Queue.Submit(commands);
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (completed || kernel is not null || counters is not null)
                {
                    throw new InvalidOperationException(
                        "The execution-heat counter layout changed while it was being prepared.");
                }
                counters = prepared;
                prepared = null;
                siteCount = expectedSiteCount;
            }
            GpuFence result = preparationFence;
            preparationFence = null;
            return result;
        }
        finally
        {
            preparationFence?.Dispose();
            prepared?.Dispose();
        }
    }

    internal bool TryGetOrCreateKernel<TKernel>(bool autoDiff, out GpuKernel diagnosticKernel)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        string candidate = NormalizeTypeName(typeof(TKernel).FullName ?? typeof(TKernel).Name);
        if (!string.Equals(candidate, shaderTypeName, StringComparison.Ordinal))
        {
            diagnosticKernel = null!;
            return false;
        }

        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
            {
                throw new InvalidOperationException("The execution-heat capture is already complete.");
            }
            kernel ??= GpuKernel.CreateExecutionHeat<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayout layout)
    {
        lock (gate)
        {
            if (kernel is not null)
            {
                throw new InvalidOperationException("The execution-heat capture already owns a kernel variant.");
            }
            if (layout.AbiVersion != AbiVersion ||
                layout.Mode != (uint)FeKernelDiagnosticMode.ExecutionHeat ||
                layout.CounterStrideBytes != sizeof(uint) ||
                layout.SiteCount is 0 or > MaximumSites)
            {
                throw new InvalidDataException("The native execution-heat buffer ABI is unsupported.");
            }

            int resolvedSiteCount = checked((int)layout.SiteCount);
            if (counters is null)
            {
                var initial = new uint[resolvedSiteCount];
                counters = GpuBuffer<uint>.Create(context, initial, BufferAccess.ReadWrite);
                siteCount = initial.Length;
            }
            else if (siteCount != resolvedSiteCount)
            {
                throw new InvalidDataException(
                    "The prepared execution-heat counter layout does not match the generated kernel.");
            }
            kernel = attachedKernel;
        }
    }

    internal void Bind(
        GpuKernel dispatchKernel,
        GpuKernelCommand command,
        GpuDispatchSize logicalSize,
        int3 threadGroupSize)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || !ReferenceEquals(kernel, dispatchKernel) || counters is null)
            {
                throw new InvalidOperationException("Execution-heat capture binding is not active.");
            }
            command.BindDiagnosticBuffer(counters);
            int dispatchIndex = matchedDispatchCount;
            matchedDispatchCount = checked(matchedDispatchCount + 1);
            if (dispatches.Count < MaximumDispatchRecords)
            {
                dispatches.Add(new GpuExecutionHeatDispatch(
                    dispatchIndex,
                    logicalSize.X,
                    logicalSize.Y,
                    logicalSize.Z,
                    threadGroupSize.X,
                    threadGroupSize.Y,
                    threadGroupSize.Z,
                    DivRoundUp(logicalSize.X, threadGroupSize.X),
                    DivRoundUp(logicalSize.Y, threadGroupSize.Y),
                    DivRoundUp(logicalSize.Z, threadGroupSize.Z)));
            }
            else
            {
                droppedDispatchRecordCount = checked(droppedDispatchRecordCount + 1);
            }
        }
    }

    /// <summary>
    /// Freezes the substitution and reads the completed device counters. The owning queue fence
    /// must already be complete; this explicit diagnostic read is never used by the steady-state
    /// Preview path.
    /// </summary>
    public GpuExecutionHeatResult CompleteAndRead()
    {
        GpuBuffer<uint> captureCounters;
        int dispatches;
        int droppedDispatches;
        GpuExecutionHeatDispatch[] dispatchRecords;
        int capacity;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
            {
                throw new InvalidOperationException("The execution-heat capture was already completed.");
            }
            if (counters is null || kernel is null || matchedDispatchCount == 0)
            {
                throw new InvalidOperationException(
                    "No matching generated compute shader was dispatched during the capture.");
            }
            completed = true;
            captureCounters = counters;
            dispatches = matchedDispatchCount;
            droppedDispatches = droppedDispatchRecordCount;
            dispatchRecords = [.. this.dispatches];
            capacity = siteCount;
        }

        context.EndExecutionHeatCapture(this);
        uint[] values = captureCounters.ToArray();
        var sites = values
            .Select(static (hits, index) => new GpuExecutionHeatSite(checked((uint)index), hits))
            .Where(static site => site.HitCount != 0)
            .ToArray();
        return new GpuExecutionHeatResult(
            shaderTypeName,
            dispatches,
            MaximumDispatchRecords,
            droppedDispatches,
            dispatchRecords,
            capacity,
            sites);
    }

    public void Dispose()
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedCounters;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            ownedKernel = kernel;
            ownedCounters = counters;
            kernel = null;
            counters = null;
        }
        context.EndExecutionHeatCapture(this);
        ownedKernel?.Dispose();
        ownedCounters?.Dispose();
    }

    internal void DisposeForContextShutdown()
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedCounters;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            ownedKernel = kernel;
            ownedCounters = counters;
            kernel = null;
            counters = null;
        }
        ownedKernel?.Dispose();
        ownedCounters?.Dispose();
    }

    private static string NormalizeTypeName(string value)
    {
        string normalized = value.Trim();
        if (normalized.StartsWith("global::", StringComparison.Ordinal))
        {
            normalized = normalized["global::".Length..];
        }
        return normalized.Replace('+', '.');
    }

    private static int DivRoundUp(int value, int divisor)
        => divisor <= 0
            ? throw new ArgumentOutOfRangeException(nameof(divisor))
            : checked((value + divisor - 1) / divisor);
}
