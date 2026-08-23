using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>Compiler-inserted checks enabled for one GPU UBSan diagnostic variant.</summary>
[Flags]
public enum GpuUbsanChecks : uint
{
    None = 0,
    FloatDivideByZero = 1u << 0,
    SqrtDomain = 1u << 1,
    LogDomain = 1u << 2,
    NonFinite = 1u << 3,
    BufferBounds = 1u << 4,
    All = FloatDivideByZero | SqrtDomain | LogDomain | NonFinite | BufferBounds,
}

/// <summary>Stable issue code written by the GPU UBSan record ABI.</summary>
public enum GpuUbsanIssueKind : uint
{
    FloatDivideByZero = 1,
    SqrtDomain = 2,
    LogDomain = 3,
    NaNValue = 4,
    InfinityValue = 5,
    BufferOutOfBounds = 6,
}

/// <summary>One fixed-width issue emitted by an instrumented compute invocation.</summary>
public readonly record struct GpuUbsanIssue(
    GpuUbsanIssueKind Kind,
    uint SourceSiteIndex,
    int3 Invocation,
    uint Detail0,
    uint Detail1,
    uint Detail2);

/// <summary>
/// Immutable result from one bounded source-instrumented dispatch. Counts are diagnostic events,
/// not timings or hardware PC samples. Invalid operations use the deterministic zero fallback.
/// </summary>
public sealed record GpuUbsanResult(
    string ShaderTypeName,
    int TargetDispatchIndex,
    int MatchedDispatchCount,
    GpuUbsanChecks EnabledChecks,
    int RecordCapacity,
    uint AttemptedCount,
    uint CommittedCount,
    uint DroppedCount,
    IReadOnlyList<GpuUbsanIssue> Issues,
    GpuExecutionHeatDispatch? Dispatch);

/// <summary>
/// Context-scoped GPU UBSan substitution for one exact matching dispatch. The ordinary generated
/// kernel cache is never instrumented. The caller must complete the containing submission before
/// <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuUbsanCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 3;
    public const int HeaderStrideBytes = 16;
    public const int RecordStrideBytes = 32;
    public const int MaximumRecordCapacity = 4_096;
    private const int HeaderWordCount = HeaderStrideBytes / sizeof(uint);
    private const int RecordWordCount = RecordStrideBytes / sizeof(uint);

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly int targetDispatchIndex;
    private readonly int recordCapacity;
    private readonly GpuUbsanChecks enabledChecks;
    private GpuKernel? kernel;
    private GpuBuffer<uint>? recordStream;
    private GpuBuffer<uint>? sink;
    private FeKernelDiagnosticLayoutV3 layout;
    private GpuExecutionHeatDispatch? dispatch;
    private int matchedDispatchCount;
    private bool targetBound;
    private bool completed;
    private bool disposed;

    internal GpuUbsanCapture(
        GpuContext context,
        string shaderTypeName,
        int targetDispatchIndex,
        int recordCapacity,
        GpuUbsanChecks enabledChecks)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.targetDispatchIndex = targetDispatchIndex;
        this.recordCapacity = recordCapacity;
        this.enabledChecks = enabledChecks;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (recordCapacity is < 1 or > MaximumRecordCapacity)
            throw new ArgumentOutOfRangeException(nameof(recordCapacity));
        if (enabledChecks == GpuUbsanChecks.None || (enabledChecks & ~GpuUbsanChecks.All) != 0)
            throw new ArgumentOutOfRangeException(nameof(enabledChecks));
    }

    public string ShaderTypeName => shaderTypeName;
    public int TargetDispatchIndex => targetDispatchIndex;
    public int RecordCapacity => recordCapacity;
    public GpuUbsanChecks EnabledChecks => enabledChecks;

    public int MatchedDispatchCount
    {
        get { lock (gate) { return matchedDispatchCount; } }
    }

    /// <summary>
    /// Allocates and queues initialization of the bounded stream and non-target sink before host
    /// timestamp intervals open. The returned exact fence must complete before dispatch.
    /// </summary>
    public GpuFence PrepareRecordLayout()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || kernel is not null || recordStream is not null || sink is not null)
                throw new InvalidOperationException(
                    "The UBSan record layout can only be prepared once before dispatch.");
        }

        int wordCount = checked(HeaderWordCount + (recordCapacity * RecordWordCount));
        uint[] streamWords = new uint[wordCount];
        streamWords[3] = checked((uint)recordCapacity);
        uint[] sinkWords = new uint[wordCount];
        GpuBuffer<uint>? preparedStream = GpuBuffer<uint>.Create(
            context,
            wordCount,
            BufferAccess.ReadWrite);
        GpuBuffer<uint>? preparedSink = GpuBuffer<uint>.Create(
            context,
            wordCount,
            BufferAccess.ReadWrite);
        using var streamSource = GpuBuffer<uint>.Create(
            context,
            streamWords,
            BufferAccess.ReadOnly);
        using var sinkSource = GpuBuffer<uint>.Create(
            context,
            sinkWords,
            BufferAccess.ReadOnly);
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
                if (completed || kernel is not null || recordStream is not null || sink is not null)
                    throw new InvalidOperationException(
                        "The UBSan record layout changed while it was being prepared.");
                recordStream = preparedStream;
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
                throw new InvalidOperationException("The UBSan capture is already complete.");
            kernel ??= GpuKernel.CreateUbsan<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV3 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The UBSan capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.Ubsan ||
                resolvedLayout.SiteCount is < 1 or > GpuExecutionHeatCapture.MaximumSites ||
                resolvedLayout.HeaderStrideBytes != HeaderStrideBytes ||
                resolvedLayout.RecordStrideBytes != RecordStrideBytes ||
                resolvedLayout.RecordCapacity != (uint)recordCapacity ||
                resolvedLayout.Flags != (uint)enabledChecks)
            {
                throw new InvalidDataException("The native UBSan stream ABI is unsupported.");
            }

            if ((recordStream is null) != (sink is null))
                throw new InvalidDataException("The prepared UBSan record layout is incomplete.");
            int wordCount = checked(HeaderWordCount + (recordCapacity * RecordWordCount));
            if (recordStream is null)
            {
                uint[] streamWords = new uint[wordCount];
                streamWords[3] = checked((uint)recordCapacity);
                recordStream = GpuBuffer<uint>.Create(context, streamWords, BufferAccess.ReadWrite);
                sink = GpuBuffer<uint>.Create(context, new uint[wordCount], BufferAccess.ReadWrite);
            }
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
            if (completed || !ReferenceEquals(kernel, dispatchKernel) ||
                recordStream is null || sink is null)
                throw new InvalidOperationException("UBSan capture binding is not active.");

            int dispatchIndex = matchedDispatchCount;
            matchedDispatchCount = checked(matchedDispatchCount + 1);
            if (dispatchIndex != targetDispatchIndex)
            {
                command.BindDiagnosticBuffer(sink);
                return;
            }

            command.BindDiagnosticBuffer(recordStream);
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

    public GpuUbsanResult CompleteAndRead()
    {
        GpuBuffer<uint> capturedStream;
        FeKernelDiagnosticLayoutV3 capturedLayout;
        GpuExecutionHeatDispatch? capturedDispatch;
        int matched;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
                throw new InvalidOperationException("The UBSan capture was already completed.");
            if (kernel is null || recordStream is null || !targetBound)
                throw new InvalidOperationException(
                    "The requested matching dispatch was not observed during the UBSan capture.");
            completed = true;
            capturedStream = recordStream;
            capturedLayout = layout;
            capturedDispatch = dispatch;
            matched = matchedDispatchCount;
        }

        context.EndDiagnosticCapture(this);
        uint[] words = capturedStream.ToArray();
        uint attempted = words[0];
        uint committed = words[1];
        uint dropped = words[2];
        uint capacity = words[3];
        if (capacity != (uint)recordCapacity || committed > capacity ||
            (ulong)attempted != (ulong)committed + dropped)
        {
            throw new InvalidDataException("The UBSan stream header invariants are invalid.");
        }

        var issues = new List<GpuUbsanIssue>(checked((int)committed));
        for (int recordIndex = 0; recordIndex < committed; recordIndex++)
        {
            int offset = checked(HeaderWordCount + (recordIndex * RecordWordCount));
            var kind = (GpuUbsanIssueKind)words[offset];
            uint site = words[offset + 1];
            var invocation = new int3(
                checked((int)words[offset + 2]),
                checked((int)words[offset + 3]),
                checked((int)words[offset + 4]));
            if (kind is < GpuUbsanIssueKind.FloatDivideByZero or > GpuUbsanIssueKind.BufferOutOfBounds ||
                site >= capturedLayout.SiteCount ||
                capturedDispatch is null ||
                invocation.X >= capturedDispatch.Value.LogicalSizeX ||
                invocation.Y >= capturedDispatch.Value.LogicalSizeY ||
                invocation.Z >= capturedDispatch.Value.LogicalSizeZ)
            {
                throw new InvalidDataException("A UBSan issue record has invalid identity or coordinates.");
            }
            issues.Add(new GpuUbsanIssue(
                kind,
                site,
                invocation,
                words[offset + 5],
                words[offset + 6],
                words[offset + 7]));
        }

        return new GpuUbsanResult(
            shaderTypeName,
            targetDispatchIndex,
            matched,
            enabledChecks,
            recordCapacity,
            attempted,
            committed,
            dropped,
            issues,
            capturedDispatch);
    }

    public void Dispose()
    {
        DisposeCore(endCapture: true);
    }

    void IGpuDiagnosticCapture.DisposeForContextShutdown()
    {
        DisposeCore(endCapture: false);
    }

    private void DisposeCore(bool endCapture)
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedStream;
        GpuBuffer<uint>? ownedSink;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            ownedKernel = kernel;
            ownedStream = recordStream;
            ownedSink = sink;
            kernel = null;
            recordStream = null;
            sink = null;
        }
        if (endCapture)
            context.EndDiagnosticCapture(this);
        ownedKernel?.Dispose();
        ownedStream?.Dispose();
        ownedSink?.Dispose();
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
