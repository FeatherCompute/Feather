using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>Semantic event kinds retained by a selected compute-invocation trace.</summary>
public enum GpuComputeTraceEventKind : uint
{
    FunctionEnter = 1,
    Statement = 2,
    Value = 3,
    BranchPredicate = 4,
    FunctionExit = 5,
    InvocationEnd = 6,
}

/// <summary>
/// One fixed-size trace event. Function and symbol identifiers are exact identifiers from the
/// captured typed FEIR module; source mapping must use the same compiled artifact identity.
/// </summary>
public sealed record GpuComputeTraceEvent(
    uint Sequence,
    uint SourceSiteIndex,
    GpuComputeTraceEventKind Kind,
    uint CallDepth,
    uint FunctionId,
    uint? SymbolId,
    GpuLineValueType? ValueType,
    int ComponentCount,
    IReadOnlyList<uint> RawComponents,
    int3 GlobalInvocation,
    uint Flags);

/// <summary>Immutable selected-invocation trace for one exact matching compute dispatch.</summary>
public sealed record GpuComputeTraceResult(
    string ShaderTypeName,
    int TargetDispatchIndex,
    int MatchedDispatchCount,
    int RecordCapacity,
    int3 SelectedInvocation,
    int3 SelectedWorkgroup,
    int3 SelectedLocalInvocation,
    uint AttemptedCount,
    uint CommittedCount,
    uint DroppedCount,
    bool SelectionMatched,
    bool InvocationCompleted,
    bool IsReplayable,
    uint MaximumCallDepth,
    IReadOnlyList<GpuComputeTraceEvent> Events,
    GpuExecutionHeatDispatch? Dispatch);

/// <summary>
/// Profile-only compiler-instrumented trace for one selected compute invocation. The ordinary
/// generated kernel cache remains uninstrumented. The caller must complete the containing GPU
/// submission before <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuComputeTraceCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 6;
    public const uint EventSchemaVersion = 1;
    public const int HeaderStrideBytes = 64;
    public const int RecordStrideBytes = 64;
    public const int MaximumRecordCapacity = 4_096;
    private const int HeaderWordCount = HeaderStrideBytes / sizeof(uint);
    private const int RecordWordCount = RecordStrideBytes / sizeof(uint);

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly int targetDispatchIndex;
    private readonly int3 selectedInvocation;
    private readonly int recordCapacity;
    private GpuKernel? kernel;
    private GpuBuffer<uint>? stream;
    private GpuBuffer<uint>? sink;
    private FeKernelDiagnosticLayoutV6 layout;
    private GpuExecutionHeatDispatch? dispatch;
    private int matchedDispatchCount;
    private bool targetBound;
    private bool completed;
    private bool disposed;

    internal GpuComputeTraceCapture(
        GpuContext context,
        string shaderTypeName,
        int targetDispatchIndex,
        int3 selectedInvocation,
        int recordCapacity)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.targetDispatchIndex = targetDispatchIndex;
        this.selectedInvocation = selectedInvocation;
        this.recordCapacity = recordCapacity;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (selectedInvocation.X < 0 || selectedInvocation.Y < 0 || selectedInvocation.Z < 0)
            throw new ArgumentOutOfRangeException(nameof(selectedInvocation));
        if (recordCapacity is < 1 or > MaximumRecordCapacity)
            throw new ArgumentOutOfRangeException(nameof(recordCapacity));
    }

    public string ShaderTypeName => shaderTypeName;
    public int TargetDispatchIndex => targetDispatchIndex;
    public int3 SelectedInvocation => selectedInvocation;
    public int RecordCapacity => recordCapacity;

    public int MatchedDispatchCount
    {
        get { lock (gate) { return matchedDispatchCount; } }
    }

    /// <summary>
    /// Queues deterministic stream initialization before measured command intervals begin. The
    /// returned exact fence must complete before the first matching dispatch.
    /// </summary>
    public GpuFence PrepareRecordLayout()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || kernel is not null || stream is not null || sink is not null)
                throw new InvalidOperationException(
                    "The compute-trace record layout can only be prepared once before dispatch.");
        }

        uint[] initial = CreateInitialWords();
        GpuBuffer<uint>? preparedStream = GpuBuffer<uint>.Create(
            context, initial.Length, BufferAccess.ReadWrite);
        GpuBuffer<uint>? preparedSink = GpuBuffer<uint>.Create(
            context, initial.Length, BufferAccess.ReadWrite);
        using var streamSource = GpuBuffer<uint>.Create(context, initial, BufferAccess.ReadOnly);
        using var sinkSource = GpuBuffer<uint>.Create(context, initial, BufferAccess.ReadOnly);
        using var commands = context.Queue.CreateCommandList();
        commands.CopyBuffer(streamSource, preparedStream);
        commands.CopyBuffer(sinkSource, preparedSink);
        commands.Close();
        GpuFence? preparationFence = null;
        try
        {
            preparationFence = context.Queue.Submit(commands);
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (completed || kernel is not null || stream is not null || sink is not null)
                    throw new InvalidOperationException(
                        "The compute-trace record layout changed while it was being prepared.");
                stream = preparedStream;
                sink = preparedSink;
                preparedStream = null;
                preparedSink = null;
            }
            GpuFence result = preparationFence;
            preparationFence = null;
            return result;
        }
        finally
        {
            preparationFence?.Dispose();
            preparedStream?.Dispose();
            preparedSink?.Dispose();
        }
    }

    bool IGpuDiagnosticCapture.TryGetOrCreateKernel<TKernel>(
        bool autoDiff,
        out GpuKernel diagnosticKernel)
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
                throw new InvalidOperationException("The compute-trace capture is already complete.");
            kernel ??= GpuKernel.CreateComputeTrace<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV6 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The compute-trace capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.ComputeTrace ||
                resolvedLayout.SiteCount is < 1 or > GpuExecutionHeatCapture.MaximumSites ||
                resolvedLayout.HeaderStrideBytes != HeaderStrideBytes ||
                resolvedLayout.RecordStrideBytes != RecordStrideBytes ||
                resolvedLayout.RecordCapacity != (uint)recordCapacity ||
                resolvedLayout.EventSchemaVersion != EventSchemaVersion ||
                resolvedLayout.Flags != 0u || resolvedLayout.Reserved != 0u)
            {
                throw new InvalidDataException("The native compute-trace stream ABI is unsupported.");
            }

            if ((stream is null) != (sink is null))
                throw new InvalidDataException("The prepared compute-trace record layout is incomplete.");
            stream ??= GpuBuffer<uint>.Create(context, CreateInitialWords(), BufferAccess.ReadWrite);
            sink ??= GpuBuffer<uint>.Create(context, CreateInitialWords(), BufferAccess.ReadWrite);
            layout = resolvedLayout;
            kernel = attachedKernel;
        }
    }

    void IGpuDiagnosticCapture.Bind(
        GpuKernel dispatchKernel,
        GpuKernelCommand command,
        GpuDispatchSize logicalSize,
        int3 threadGroupSize)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || !ReferenceEquals(kernel, dispatchKernel) || stream is null || sink is null)
                throw new InvalidOperationException("Compute-trace capture binding is not active.");

            int dispatchIndex = matchedDispatchCount;
            matchedDispatchCount = checked(matchedDispatchCount + 1);
            if (dispatchIndex != targetDispatchIndex)
            {
                command.BindDiagnosticBuffer(sink);
                return;
            }
            if (selectedInvocation.X >= logicalSize.X ||
                selectedInvocation.Y >= logicalSize.Y ||
                selectedInvocation.Z >= logicalSize.Z)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(selectedInvocation),
                    "The selected invocation lies outside the target dispatch logical extent.");
            }

            command.BindDiagnosticBuffer(stream);
            targetBound = true;
            dispatch = new GpuExecutionHeatDispatch(
                dispatchIndex,
                logicalSize.X,
                logicalSize.Y,
                logicalSize.Z,
                threadGroupSize.X,
                threadGroupSize.Y,
                threadGroupSize.Z,
                DivRoundUp(logicalSize.X, threadGroupSize.X),
                DivRoundUp(logicalSize.Y, threadGroupSize.Y),
                DivRoundUp(logicalSize.Z, threadGroupSize.Z));
        }
    }

    public GpuComputeTraceResult CompleteAndRead()
    {
        GpuBuffer<uint> capturedStream;
        FeKernelDiagnosticLayoutV6 capturedLayout;
        GpuExecutionHeatDispatch? capturedDispatch;
        int matched;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
                throw new InvalidOperationException("The compute-trace capture was already completed.");
            if (kernel is null || stream is null || !targetBound)
                throw new InvalidOperationException(
                    "The requested matching dispatch was not observed during the compute-trace capture.");
            completed = true;
            capturedStream = stream;
            capturedLayout = layout;
            capturedDispatch = dispatch;
            matched = matchedDispatchCount;
        }

        context.EndDiagnosticCapture(this);
        uint[] words = capturedStream.ToArray();
        uint attempted = words[3];
        uint committed = words[4];
        uint dropped = words[5];
        bool selectionMatched = words[6] == 1u;
        bool invocationCompleted = words[7] == 1u;
        if (words.Length != checked(HeaderWordCount + recordCapacity * RecordWordCount) ||
            words[0] != EventSchemaVersion || words[1] != RecordStrideBytes ||
            words[2] != (uint)recordCapacity || words[6] > 1u || words[7] > 1u ||
            words[8] != (uint)selectedInvocation.X ||
            words[9] != (uint)selectedInvocation.Y ||
            words[10] != (uint)selectedInvocation.Z ||
            words[11] != 0u || words[12] != 0u || words[13] != 0u ||
            words[14] != 0u || words[15] != 0u ||
            committed > (uint)recordCapacity || (ulong)attempted != (ulong)committed + dropped ||
            !selectionMatched || !invocationCompleted || attempted == 0u)
        {
            throw new InvalidDataException("The compute-trace stream header invariants are invalid.");
        }

        var events = new List<GpuComputeTraceEvent>(checked((int)committed));
        var callStack = new List<uint>();
        uint maximumCallDepth = 0u;
        bool sawInvocationEnd = false;
        for (int eventIndex = 0; eventIndex < committed; ++eventIndex)
        {
            int offset = checked(HeaderWordCount + eventIndex * RecordWordCount);
            uint sequence = words[offset];
            uint site = words[offset + 1];
            var kind = (GpuComputeTraceEventKind)words[offset + 2];
            uint depth = words[offset + 3];
            uint functionId = words[offset + 4];
            uint symbolWord = words[offset + 5];
            uint typeTag = words[offset + 6];
            uint componentWord = words[offset + 7];
            var global = new int3(
                checked((int)words[offset + 12]),
                checked((int)words[offset + 13]),
                checked((int)words[offset + 14]));
            uint flags = words[offset + 15];
            if (sequence != (uint)eventIndex || site >= capturedLayout.SiteCount ||
                kind is < GpuComputeTraceEventKind.FunctionEnter or > GpuComputeTraceEventKind.InvocationEnd ||
                typeTag > (uint)GpuLineValueType.Float32 ||
                (typeTag == 0u ? componentWord != 0u : componentWord is < 1u or > 4u) ||
                global != selectedInvocation || flags != 0u)
            {
                throw new InvalidDataException("A compute-trace event has invalid identity or value metadata.");
            }

            int componentCount = checked((int)componentWord);
            uint[] raw = componentCount == 0
                ? []
                : words.AsSpan(offset + 8, componentCount).ToArray();
            for (int unused = componentCount; unused < 4; ++unused)
            {
                if (words[offset + 8 + unused] != 0u)
                    throw new InvalidDataException("A compute-trace event has non-zero unused value words.");
            }

            switch (kind)
            {
                case GpuComputeTraceEventKind.FunctionEnter:
                    if (depth != (uint)callStack.Count)
                        throw new InvalidDataException("A compute-trace function entry has invalid call depth.");
                    callStack.Add(functionId);
                    break;
                case GpuComputeTraceEventKind.FunctionExit:
                    if (callStack.Count == 0 || depth != (uint)(callStack.Count - 1) ||
                        callStack[^1] != functionId)
                    {
                        throw new InvalidDataException("A compute-trace function exit contradicts the call stack.");
                    }
                    callStack.RemoveAt(callStack.Count - 1);
                    break;
                case GpuComputeTraceEventKind.InvocationEnd:
                    if (callStack.Count != 0 || depth != 0u || eventIndex != committed - 1)
                        throw new InvalidDataException("A compute-trace invocation end is not terminal.");
                    sawInvocationEnd = true;
                    break;
                default:
                    if (callStack.Count == 0 || depth != (uint)(callStack.Count - 1) ||
                        callStack[^1] != functionId)
                    {
                        throw new InvalidDataException("A compute-trace event contradicts the active call frame.");
                    }
                    break;
            }
            maximumCallDepth = System.Math.Max(maximumCallDepth, depth);
            events.Add(new GpuComputeTraceEvent(
                sequence,
                site,
                kind,
                depth,
                functionId,
                symbolWord == uint.MaxValue ? null : symbolWord,
                typeTag == 0u ? null : (GpuLineValueType)typeTag,
                componentCount,
                raw,
                global,
                flags));

            if (dropped != 0u && eventIndex == committed - 1)
                callStack.Clear();
        }

        bool replayable = dropped == 0u && sawInvocationEnd && callStack.Count == 0;
        if (dropped == 0u && !replayable)
            throw new InvalidDataException("A complete compute trace has no valid terminal replay state.");

        if (capturedDispatch is null)
            throw new InvalidDataException("The compute-trace dispatch identity is unavailable.");
        var workgroup = new int3(
            selectedInvocation.X / capturedDispatch.Value.ThreadGroupSizeX,
            selectedInvocation.Y / capturedDispatch.Value.ThreadGroupSizeY,
            selectedInvocation.Z / capturedDispatch.Value.ThreadGroupSizeZ);
        var local = new int3(
            selectedInvocation.X % capturedDispatch.Value.ThreadGroupSizeX,
            selectedInvocation.Y % capturedDispatch.Value.ThreadGroupSizeY,
            selectedInvocation.Z % capturedDispatch.Value.ThreadGroupSizeZ);
        return new GpuComputeTraceResult(
            shaderTypeName,
            targetDispatchIndex,
            matched,
            recordCapacity,
            selectedInvocation,
            workgroup,
            local,
            attempted,
            committed,
            dropped,
            selectionMatched,
            invocationCompleted,
            replayable,
            maximumCallDepth,
            events,
            capturedDispatch);
    }

    public void Dispose() => DisposeCore(endCapture: true);

    void IGpuDiagnosticCapture.DisposeForContextShutdown() => DisposeCore(endCapture: false);

    private void DisposeCore(bool endCapture)
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedStream;
        GpuBuffer<uint>? ownedSink;
        lock (gate)
        {
            if (disposed)
                return;
            disposed = true;
            ownedKernel = kernel;
            ownedStream = stream;
            ownedSink = sink;
            kernel = null;
            stream = null;
            sink = null;
        }
        if (endCapture)
            context.EndDiagnosticCapture(this);
        ownedKernel?.Dispose();
        ownedStream?.Dispose();
        ownedSink?.Dispose();
    }

    private uint[] CreateInitialWords()
    {
        int wordCount = checked(HeaderWordCount + recordCapacity * RecordWordCount);
        var words = new uint[wordCount];
        words[0] = EventSchemaVersion;
        words[1] = RecordStrideBytes;
        words[2] = checked((uint)recordCapacity);
        words[8] = checked((uint)selectedInvocation.X);
        words[9] = checked((uint)selectedInvocation.Y);
        words[10] = checked((uint)selectedInvocation.Z);
        return words;
    }

    private static string NormalizeTypeName(string value)
    {
        string normalized = value.Trim();
        if (normalized.StartsWith("global::", StringComparison.Ordinal))
            normalized = normalized["global::".Length..];
        return normalized.Replace('+', '.');
    }

    private static int DivRoundUp(int value, int divisor)
        => divisor <= 0
            ? throw new ArgumentOutOfRangeException(nameof(divisor))
            : checked((value + divisor - 1) / divisor);
}
