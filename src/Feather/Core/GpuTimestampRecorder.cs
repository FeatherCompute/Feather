using Feather.Native;

namespace Feather;

/// <summary>The hardware timestamp interval represented by one profiled queue submission.</summary>
public enum GpuTimestampIntervalKind
{
    Graph,
    Pass,
    Draw,
    Dispatch
}

/// <summary>Why a requested GPU timestamp interval has no native result slot.</summary>
public enum GpuTimestampUnavailableReason
{
    TimestampQueriesUnsupported,
    QueryPoolExhausted
}

/// <summary>
/// Controls the intervals recorded around one queue submission. Pass intervals are always
/// explicit; draw and dispatch intervals are added only when requested.
/// </summary>
public sealed record GpuProfilingOptions(
    bool IncludeCommandIntervals = false,
    string GraphCorrelationId = "graph",
    string? GraphLabel = null);

/// <summary>Stable metadata for one requested GPU timestamp interval.</summary>
public sealed record GpuTimestampIntervalDescriptor(
    int Index,
    GpuTimestampIntervalKind Kind,
    string CorrelationId,
    int? ParentIndex,
    int? CommandOrdinal,
    string? Label,
    int? ResultIndex,
    GpuTimestampUnavailableReason? UnavailableReason);

/// <summary>
/// Records nested pass scopes and, when enabled, automatic draw/dispatch scopes into the same
/// command stream. The owning <see cref="GpuQueue.SubmitProfiled{T}"/> call seals and submits it.
/// </summary>
public sealed class GpuTimestampRecorder
{
    private readonly GpuContext context;
    private readonly bool timestampQueriesSupported;
    private readonly List<Entry> entries = [];
    private readonly List<int> openEntries = [];
    private readonly List<uint> nativeIntervals = [];
    private bool sealedRecorder;
    private int nextCommandOrdinal;

    internal GpuTimestampRecorder(GpuContext context, GpuProfilingOptions options, bool timestampQueriesSupported)
    {
        this.context = context;
        this.timestampQueriesSupported = timestampQueriesSupported;
        IncludeCommandIntervals = options.IncludeCommandIntervals;
        _ = BeginInterval(
            GpuTimestampIntervalKind.Graph,
            ValidateCorrelationId(options.GraphCorrelationId, nameof(options.GraphCorrelationId)),
            options.GraphLabel,
            commandOrdinal: null);
    }

    internal bool IncludeCommandIntervals { get; }

    /// <summary>Begins one pass interval nested under the graph interval.</summary>
    public IDisposable BeginPass(string correlationId, string? label = null)
        => BeginInterval(
            GpuTimestampIntervalKind.Pass,
            ValidateCorrelationId(correlationId, nameof(correlationId)),
            label,
            commandOrdinal: null);

    internal IDisposable BeginCommand(GpuTimestampIntervalKind kind, string? label)
    {
        if (kind is not (GpuTimestampIntervalKind.Draw or GpuTimestampIntervalKind.Dispatch))
        {
            throw new ArgumentOutOfRangeException(nameof(kind));
        }
        var ordinal = checked(nextCommandOrdinal++);
        return BeginInterval(kind, "command:" + ordinal, label, ordinal);
    }

    internal (IReadOnlyList<GpuTimestampIntervalDescriptor> Descriptors, uint[] NativeIntervals) Seal()
    {
        if (!sealedRecorder)
        {
            while (openEntries.Count != 0)
            {
                EndInterval(openEntries[^1]);
            }
            sealedRecorder = true;
        }

        return (
            entries.Select(static entry => entry.ToDescriptor()).ToArray(),
            nativeIntervals.ToArray());
    }

    private IDisposable BeginInterval(
        GpuTimestampIntervalKind kind,
        string correlationId,
        string? label,
        int? commandOrdinal)
    {
        ObjectDisposedException.ThrowIf(sealedRecorder, this);
        if (label is { Length: > 512 })
        {
            throw new ArgumentOutOfRangeException(nameof(label), "GPU timestamp labels are limited to 512 characters.");
        }

        NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_begin_timestamp_interval(context.Handle, out var token));
        int? resultIndex = null;
        GpuTimestampUnavailableReason? unavailableReason = null;
        if (token == 0)
        {
            unavailableReason = timestampQueriesSupported
                ? GpuTimestampUnavailableReason.QueryPoolExhausted
                : GpuTimestampUnavailableReason.TimestampQueriesUnsupported;
        }
        else
        {
            resultIndex = nativeIntervals.Count;
            nativeIntervals.Add(token);
        }

        var index = entries.Count;
        entries.Add(new Entry(
            index,
            kind,
            correlationId,
            openEntries.Count == 0 ? null : openEntries[^1],
            commandOrdinal,
            label,
            resultIndex,
            unavailableReason,
            token));
        openEntries.Add(index);
        return new Scope(this, index);
    }

    private void EndInterval(int index)
    {
        if (index < 0 || index >= entries.Count || entries[index].Ended)
        {
            return;
        }
        if (openEntries.Count == 0 || openEntries[^1] != index)
        {
            throw new InvalidOperationException("GPU timestamp scopes must be disposed in last-in-first-out order.");
        }

        var entry = entries[index];
        if (entry.NativeInterval != 0)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_queue_end_timestamp_interval(
                context.Handle,
                entry.NativeInterval));
        }
        entry.Ended = true;
        openEntries.RemoveAt(openEntries.Count - 1);
    }

    private static string ValidateCorrelationId(string correlationId, string parameterName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(correlationId, parameterName);
        if (correlationId.Length > 256)
        {
            throw new ArgumentOutOfRangeException(parameterName, "GPU timestamp correlation IDs are limited to 256 characters.");
        }
        return correlationId;
    }

    private sealed class Entry(
        int index,
        GpuTimestampIntervalKind kind,
        string correlationId,
        int? parentIndex,
        int? commandOrdinal,
        string? label,
        int? resultIndex,
        GpuTimestampUnavailableReason? unavailableReason,
        uint nativeInterval)
    {
        internal uint NativeInterval { get; } = nativeInterval;
        internal bool Ended { get; set; }

        internal GpuTimestampIntervalDescriptor ToDescriptor()
            => new(index, kind, correlationId, parentIndex, commandOrdinal, label, resultIndex, unavailableReason);
    }

    private sealed class Scope(GpuTimestampRecorder owner, int index) : IDisposable
    {
        private GpuTimestampRecorder? owner = owner;

        public void Dispose()
        {
            var current = owner;
            if (current is null)
            {
                return;
            }
            current.EndInterval(index);
            owner = null;
        }
    }
}

/// <summary>Result, fence, queue identity, and interval map for one profiled submission.</summary>
public readonly record struct GpuProfiledSubmission<T>(
    T Result,
    GpuFence Fence,
    string QueueId,
    IReadOnlyList<GpuTimestampIntervalDescriptor> Intervals);
