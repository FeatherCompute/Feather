using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>Controls which invocations contribute bounded Print/Assert log records.</summary>
public enum GpuPrintAssertFilterMode : uint
{
    AllInvocations = 0,
    SelectedInvocation = 1,
}

/// <summary>Stable user-authored diagnostic record code.</summary>
public enum GpuDebugRecordKind : uint
{
    Print = 1,
    Assert = 2,
}

/// <summary>Stable severity written by the Print/Assert record ABI.</summary>
public enum GpuDebugSeverity : uint
{
    Information = 1,
    Error = 3,
}

/// <summary>Primitive interpretation of raw Print/Assert payload words.</summary>
public enum GpuDebugValueType : uint
{
    Bool32 = 1,
    Int32 = 2,
    UInt32 = 3,
    Float32 = 4,
}

/// <summary>One fixed-width user-authored diagnostic event emitted by a compute invocation.</summary>
public readonly record struct GpuPrintAssertRecord(
    GpuDebugRecordKind Kind,
    uint SourceSiteIndex,
    ShaderStage Stage,
    GpuDebugSeverity Severity,
    int3 Invocation,
    uint LinearInvocationIndex,
    GpuDebugValueType ValueType,
    int ComponentCount,
    IReadOnlyList<uint> RawComponents,
    uint Flags);

/// <summary>
/// Immutable result for one exact dispatch. Log filtering is intentionally independent from the
/// dispatch-wide assertion mask: a selected-invocation filter never hides failures elsewhere.
/// </summary>
public sealed record GpuPrintAssertResult(
    string ShaderTypeName,
    int TargetDispatchIndex,
    int MatchedDispatchCount,
    GpuDispatchSize LogicalSize,
    GpuPrintAssertFilterMode FilterMode,
    int3 SelectedInvocation,
    int RecordCapacity,
    uint AttemptedCount,
    uint CommittedCount,
    uint DroppedCount,
    IReadOnlyList<GpuPrintAssertRecord> Records,
    uint AssertionFailureCount,
    IReadOnlyList<int3> AssertedInvocations,
    GpuExecutionHeatDispatch? Dispatch);

/// <summary>
/// Context-scoped substitution for explicitly authored <see cref="GpuDebug"/> markers. Ordinary
/// kernels are identity-only and allocate no stream. The caller must complete the containing GPU
/// submission before <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuPrintAssertCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 4;
    public const int HeaderStrideBytes = 32;
    public const int RecordStrideBytes = 64;
    public const int MaskHeaderStrideBytes = 16;
    public const int MaskCellStrideBytes = 4;
    public const int MaximumRecordCapacity = 4_096;
    public const int MaximumLogicalInvocations = 16_777_216;
    private const int HeaderWordCount = HeaderStrideBytes / sizeof(uint);
    private const int RecordWordCount = RecordStrideBytes / sizeof(uint);
    private const int MaskHeaderWordCount = MaskHeaderStrideBytes / sizeof(uint);
    private const uint StreamSchemaVersion = 1;

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly int targetDispatchIndex;
    private readonly GpuDispatchSize logicalSize;
    private readonly int logicalInvocationCount;
    private readonly int recordCapacity;
    private readonly GpuPrintAssertFilterMode filterMode;
    private readonly int3 selectedInvocation;
    private readonly List<GpuBuffer<uint>> retiredSinks = [];
    private GpuKernel? kernel;
    private GpuBuffer<uint>? stream;
    private GpuBuffer<uint>? sink;
    private int sinkLogicalCapacity;
    private FeKernelDiagnosticLayoutV4 layout;
    private GpuExecutionHeatDispatch? dispatch;
    private int matchedDispatchCount;
    private bool targetBound;
    private bool completed;
    private bool disposed;

    internal GpuPrintAssertCapture(
        GpuContext context,
        string shaderTypeName,
        int targetDispatchIndex,
        GpuDispatchSize logicalSize,
        int recordCapacity,
        GpuPrintAssertFilterMode filterMode,
        int3 selectedInvocation)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.targetDispatchIndex = targetDispatchIndex;
        this.logicalSize = logicalSize;
        this.recordCapacity = recordCapacity;
        this.filterMode = filterMode;
        this.selectedInvocation = selectedInvocation;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (logicalSize.X <= 0 || logicalSize.Y <= 0 || logicalSize.Z <= 0)
            throw new ArgumentOutOfRangeException(nameof(logicalSize));
        logicalInvocationCount = checked(logicalSize.X * logicalSize.Y * logicalSize.Z);
        if (logicalInvocationCount > MaximumLogicalInvocations)
            throw new ArgumentOutOfRangeException(nameof(logicalSize));
        if (recordCapacity is < 1 or > MaximumRecordCapacity)
            throw new ArgumentOutOfRangeException(nameof(recordCapacity));
        if (filterMode is < GpuPrintAssertFilterMode.AllInvocations or > GpuPrintAssertFilterMode.SelectedInvocation)
            throw new ArgumentOutOfRangeException(nameof(filterMode));
        if (filterMode == GpuPrintAssertFilterMode.SelectedInvocation &&
            (selectedInvocation.X < 0 || selectedInvocation.X >= logicalSize.X ||
             selectedInvocation.Y < 0 || selectedInvocation.Y >= logicalSize.Y ||
             selectedInvocation.Z < 0 || selectedInvocation.Z >= logicalSize.Z))
        {
            throw new ArgumentOutOfRangeException(nameof(selectedInvocation));
        }
    }

    public string ShaderTypeName => shaderTypeName;
    public int TargetDispatchIndex => targetDispatchIndex;
    public GpuDispatchSize LogicalSize => logicalSize;
    public int RecordCapacity => recordCapacity;
    public GpuPrintAssertFilterMode FilterMode => filterMode;
    public int3 SelectedInvocation => selectedInvocation;

    /// <summary>
    /// Number of matching shader dispatches observed so far. Render hosts use this monotonic
    /// capture-local count to authenticate the target dispatch against pass boundaries.
    /// </summary>
    public int MatchedDispatchCount
    {
        get { lock (gate) { return matchedDispatchCount; } }
    }

    /// <summary>
    /// Queues initialization of the exact target stream before measured command intervals begin.
    /// The returned fence must complete before the target dispatch.
    /// </summary>
    public GpuFence PrepareRecordLayout()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || kernel is not null || stream is not null)
                throw new InvalidOperationException(
                    "The Print/Assert record layout can only be prepared once before dispatch.");
        }

        uint[] initial = CreateInitialWords(logicalSize);
        GpuBuffer<uint>? prepared = GpuBuffer<uint>.Create(
            context,
            initial.Length,
            BufferAccess.ReadWrite);
        using var source = GpuBuffer<uint>.Create(context, initial, BufferAccess.ReadOnly);
        using var commands = context.Queue.CreateCommandList();
        commands.CopyBuffer(source, prepared);
        commands.Close();
        GpuFence? preparationFence = null;
        try
        {
            preparationFence = context.Queue.Submit(commands);
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (completed || kernel is not null || stream is not null)
                    throw new InvalidOperationException(
                        "The Print/Assert record layout changed while it was being prepared.");
                stream = prepared;
                prepared = null;
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
                throw new InvalidOperationException("The Print/Assert capture is already complete.");
            kernel ??= GpuKernel.CreatePrintAssert<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV4 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The Print/Assert capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.PrintAssert ||
                resolvedLayout.SiteCount is < 1 or > GpuExecutionHeatCapture.MaximumSites ||
                resolvedLayout.HeaderStrideBytes != HeaderStrideBytes ||
                resolvedLayout.RecordStrideBytes != RecordStrideBytes ||
                resolvedLayout.RecordCapacity != (uint)recordCapacity ||
                resolvedLayout.FilterMode != (uint)filterMode ||
                resolvedLayout.MaskHeaderStrideBytes != MaskHeaderStrideBytes ||
                resolvedLayout.MaskCellStrideBytes != MaskCellStrideBytes ||
                resolvedLayout.LogicalX != (uint)logicalSize.X ||
                resolvedLayout.LogicalY != (uint)logicalSize.Y ||
                resolvedLayout.LogicalZ != (uint)logicalSize.Z)
            {
                throw new InvalidDataException("The native Print/Assert stream ABI is unsupported.");
            }

            stream ??= GpuBuffer<uint>.Create(
                context,
                CreateInitialWords(logicalSize),
                BufferAccess.ReadWrite);
            layout = resolvedLayout;
            kernel = attachedKernel;
        }
    }

    void IGpuDiagnosticCapture.Bind(
        GpuKernel dispatchKernel,
        GpuKernelCommand command,
        GpuDispatchSize dispatchLogicalSize,
        int3 threadGroupSize)
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || !ReferenceEquals(kernel, dispatchKernel) || stream is null)
                throw new InvalidOperationException("Print/Assert capture binding is not active.");

            int dispatchIndex = matchedDispatchCount;
            matchedDispatchCount = checked(matchedDispatchCount + 1);
            if (dispatchIndex != targetDispatchIndex)
            {
                int requiredLogicalCapacity = CheckedLogicalInvocationCount(dispatchLogicalSize);
                if (sink is null || sinkLogicalCapacity < requiredLogicalCapacity)
                {
                    if (sink is not null)
                        retiredSinks.Add(sink);
                    sink = GpuBuffer<uint>.Create(
                        context,
                        CreateInitialWords(dispatchLogicalSize),
                        BufferAccess.ReadWrite);
                    sinkLogicalCapacity = requiredLogicalCapacity;
                }
                command.BindDiagnosticBuffer(sink);
                return;
            }

            if (dispatchLogicalSize != logicalSize)
                throw new InvalidOperationException(
                    "The target Print/Assert dispatch extent differs from the requested immutable snapshot.");
            command.BindDiagnosticBuffer(stream);
            targetBound = true;
            dispatch = new GpuExecutionHeatDispatch(
                dispatchIndex,
                dispatchLogicalSize.X,
                dispatchLogicalSize.Y,
                dispatchLogicalSize.Z,
                threadGroupSize.X,
                threadGroupSize.Y,
                threadGroupSize.Z,
                DivRoundUp(dispatchLogicalSize.X, threadGroupSize.X),
                DivRoundUp(dispatchLogicalSize.Y, threadGroupSize.Y),
                DivRoundUp(dispatchLogicalSize.Z, threadGroupSize.Z));
        }
    }

    public GpuPrintAssertResult CompleteAndRead()
    {
        GpuBuffer<uint> capturedStream;
        FeKernelDiagnosticLayoutV4 capturedLayout;
        GpuExecutionHeatDispatch? capturedDispatch;
        int matched;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
                throw new InvalidOperationException("The Print/Assert capture was already completed.");
            if (kernel is null || stream is null || !targetBound)
                throw new InvalidOperationException(
                    "The requested matching dispatch was not observed during the Print/Assert capture.");
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
        if (words[0] != StreamSchemaVersion || words[1] != RecordStrideBytes ||
            words[2] != (uint)recordCapacity || words[6] != (uint)filterMode || words[7] != 0u ||
            committed > (uint)recordCapacity || (ulong)attempted != (ulong)committed + dropped)
        {
            throw new InvalidDataException("The Print/Assert stream header invariants are invalid.");
        }

        var records = new List<GpuPrintAssertRecord>(checked((int)committed));
        for (int recordIndex = 0; recordIndex < committed; ++recordIndex)
        {
            int offset = checked(HeaderWordCount + recordIndex * RecordWordCount);
            var kind = (GpuDebugRecordKind)words[offset];
            uint sourceSite = words[offset + 1];
            uint stage = words[offset + 2];
            var severity = (GpuDebugSeverity)words[offset + 3];
            var invocation = new int3(
                checked((int)words[offset + 4]),
                checked((int)words[offset + 5]),
                checked((int)words[offset + 6]));
            uint linear = words[offset + 7];
            var valueType = (GpuDebugValueType)words[offset + 8];
            int componentCount = checked((int)words[offset + 9]);
            uint flags = words[offset + 14];
            uint expectedLinear = checked((uint)(
                invocation.X + logicalSize.X * (invocation.Y + logicalSize.Y * invocation.Z)));
            if (kind is < GpuDebugRecordKind.Print or > GpuDebugRecordKind.Assert ||
                sourceSite >= capturedLayout.SiteCount || stage != 3u ||
                (kind == GpuDebugRecordKind.Print && severity != GpuDebugSeverity.Information) ||
                (kind == GpuDebugRecordKind.Assert && severity != GpuDebugSeverity.Error) ||
                invocation.X < 0 || invocation.X >= logicalSize.X ||
                invocation.Y < 0 || invocation.Y >= logicalSize.Y ||
                invocation.Z < 0 || invocation.Z >= logicalSize.Z ||
                linear != expectedLinear ||
                valueType is < GpuDebugValueType.Bool32 or > GpuDebugValueType.Float32 ||
                componentCount is < 1 or > 4 ||
                flags != (kind == GpuDebugRecordKind.Assert ? 1u : 0u) ||
                words[offset + 15] != 0u)
            {
                throw new InvalidDataException("A Print/Assert record has invalid identity or payload metadata.");
            }
            var raw = new uint[componentCount];
            Array.Copy(words, offset + 10, raw, 0, componentCount);
            records.Add(new GpuPrintAssertRecord(
                kind,
                sourceSite,
                ShaderStage.Compute,
                severity,
                invocation,
                linear,
                valueType,
                componentCount,
                raw,
                flags));
        }

        int maskOffset = checked(HeaderWordCount + recordCapacity * RecordWordCount);
        if (words[maskOffset] != (uint)logicalSize.X ||
            words[maskOffset + 1] != (uint)logicalSize.Y ||
            words[maskOffset + 2] != (uint)logicalSize.Z)
        {
            throw new InvalidDataException("The Print/Assert assertion-mask extent is invalid.");
        }
        uint failureCount = words[maskOffset + 3];
        var assertedInvocations = new List<int3>();
        int cellsOffset = maskOffset + MaskHeaderWordCount;
        for (int linear = 0; linear < logicalInvocationCount; ++linear)
        {
            uint cell = words[cellsOffset + linear];
            if (cell > 1u)
                throw new InvalidDataException("The Print/Assert assertion mask contains an invalid cell value.");
            if (cell == 0u)
                continue;
            int x = linear % logicalSize.X;
            int yz = linear / logicalSize.X;
            int y = yz % logicalSize.Y;
            int z = yz / logicalSize.Y;
            assertedInvocations.Add(new int3(x, y, z));
        }
        if (failureCount < assertedInvocations.Count)
            throw new InvalidDataException("The Print/Assert assertion counters contradict the spatial mask.");

        return new GpuPrintAssertResult(
            shaderTypeName,
            targetDispatchIndex,
            matched,
            logicalSize,
            filterMode,
            selectedInvocation,
            recordCapacity,
            attempted,
            committed,
            dropped,
            records,
            failureCount,
            assertedInvocations,
            capturedDispatch);
    }

    public void Dispose() => DisposeCore(endCapture: true);

    void IGpuDiagnosticCapture.DisposeForContextShutdown() => DisposeCore(endCapture: false);

    private void DisposeCore(bool endCapture)
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedStream;
        GpuBuffer<uint>? ownedSink;
        GpuBuffer<uint>[] ownedRetiredSinks;
        lock (gate)
        {
            if (disposed)
                return;
            disposed = true;
            ownedKernel = kernel;
            ownedStream = stream;
            ownedSink = sink;
            ownedRetiredSinks = [.. retiredSinks];
            kernel = null;
            stream = null;
            sink = null;
            retiredSinks.Clear();
        }
        if (endCapture)
            context.EndDiagnosticCapture(this);
        ownedKernel?.Dispose();
        ownedStream?.Dispose();
        ownedSink?.Dispose();
        foreach (GpuBuffer<uint> retired in ownedRetiredSinks)
            retired.Dispose();
    }

    private uint[] CreateInitialWords(GpuDispatchSize size)
    {
        int invocationCount = CheckedLogicalInvocationCount(size);
        int wordCount = checked(
            HeaderWordCount + recordCapacity * RecordWordCount + MaskHeaderWordCount + invocationCount);
        var words = new uint[wordCount];
        words[0] = StreamSchemaVersion;
        words[1] = RecordStrideBytes;
        words[2] = checked((uint)recordCapacity);
        words[6] = (uint)filterMode;
        int maskOffset = HeaderWordCount + recordCapacity * RecordWordCount;
        words[maskOffset] = checked((uint)size.X);
        words[maskOffset + 1] = checked((uint)size.Y);
        words[maskOffset + 2] = checked((uint)size.Z);
        return words;
    }

    private static int CheckedLogicalInvocationCount(GpuDispatchSize size)
    {
        if (size.X <= 0 || size.Y <= 0 || size.Z <= 0)
            throw new ArgumentOutOfRangeException(nameof(size));
        int count = checked(size.X * size.Y * size.Z);
        if (count > MaximumLogicalInvocations)
            throw new ArgumentOutOfRangeException(nameof(size));
        return count;
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
