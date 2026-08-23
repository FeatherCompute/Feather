using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;
using System.Numerics;

namespace Feather;

/// <summary>One 128-bit subgroup ballot mask in least-significant-word order.</summary>
public readonly record struct GpuSubgroupBallotMask(uint X, uint Y, uint Z, uint W)
{
    public int SetBitCount =>
        BitOperations.PopCount(X) +
        BitOperations.PopCount(Y) +
        BitOperations.PopCount(Z) +
        BitOperations.PopCount(W);

    internal bool IsSubsetOf(GpuSubgroupBallotMask other) =>
        (X & ~other.X) == 0u &&
        (Y & ~other.Y) == 0u &&
        (Z & ~other.Z) == 0u &&
        (W & ~other.W) == 0u;
}

/// <summary>
/// One elected-lane record captured immediately before a selected converged compute branch.
/// Counts and masks describe active predicate lanes, not timing or program-counter samples.
/// </summary>
public readonly record struct GpuBranchDivergenceRecord(
    uint SourceSiteIndex,
    int3 Workgroup,
    uint SubgroupId,
    uint SubgroupSize,
    uint NumSubgroups,
    uint ElectedInvocationId,
    uint ActiveLaneCount,
    uint TrueLaneCount,
    uint FalseLaneCount,
    bool IsMixed,
    GpuSubgroupBallotMask ActiveMask,
    GpuSubgroupBallotMask TrueMask);

/// <summary>Immutable subgroup predicate evidence for one exact matching dispatch.</summary>
public sealed record GpuBranchDivergenceResult(
    string ShaderTypeName,
    uint SourceSiteIndex,
    int TargetDispatchIndex,
    int MatchedDispatchCount,
    int RecordCapacity,
    uint AttemptedCount,
    uint CommittedCount,
    uint DroppedCount,
    uint SubgroupCount,
    uint MixedSubgroupCount,
    uint ActiveLaneCount,
    uint TrueLaneCount,
    uint FalseLaneCount,
    uint UniformTrueSubgroupCount,
    uint UniformFalseSubgroupCount,
    IReadOnlyList<GpuBranchDivergenceRecord> Records,
    GpuExecutionHeatDispatch? Dispatch,
    BackendSubgroupInfo BackendSubgroups);

/// <summary>
/// Profile-only substitution for one retained, converged top-level compute branch. The ordinary
/// generated kernel cache is never instrumented. The caller must complete the containing GPU
/// submission before <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuBranchDivergenceCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 5;
    public const int HeaderStrideBytes = 64;
    public const int RecordStrideBytes = 80;
    public const int MaximumRecordCapacity = 4_096;
    public const uint RequiredSubgroupFeatures = 0x0fu;
    private const uint StreamSchemaVersion = 1;
    private const int HeaderWordCount = HeaderStrideBytes / sizeof(uint);
    private const int RecordWordCount = RecordStrideBytes / sizeof(uint);

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly uint sourceSiteIndex;
    private readonly int targetDispatchIndex;
    private readonly int recordCapacity;
    private readonly BackendSubgroupInfo backendSubgroups;
    private GpuKernel? kernel;
    private GpuBuffer<uint>? stream;
    private GpuBuffer<uint>? sink;
    private FeKernelDiagnosticLayoutV5 layout;
    private GpuExecutionHeatDispatch? dispatch;
    private int matchedDispatchCount;
    private bool targetBound;
    private bool completed;
    private bool disposed;

    internal GpuBranchDivergenceCapture(
        GpuContext context,
        string shaderTypeName,
        uint sourceSiteIndex,
        int targetDispatchIndex,
        int recordCapacity,
        BackendSubgroupInfo backendSubgroups)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.sourceSiteIndex = sourceSiteIndex;
        this.targetDispatchIndex = targetDispatchIndex;
        this.recordCapacity = recordCapacity;
        this.backendSubgroups = backendSubgroups;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (sourceSiteIndex == uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(sourceSiteIndex));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (recordCapacity is < 1 or > MaximumRecordCapacity)
            throw new ArgumentOutOfRangeException(nameof(recordCapacity));
        if (!backendSubgroups.SupportsBranchDivergence)
        {
            throw new NotSupportedException(
                "Branch-divergence capture requires compute basic/vote/ballot subgroup support.");
        }
    }

    public string ShaderTypeName => shaderTypeName;
    public uint SourceSiteIndex => sourceSiteIndex;
    public int TargetDispatchIndex => targetDispatchIndex;
    public int RecordCapacity => recordCapacity;
    public BackendSubgroupInfo BackendSubgroups => backendSubgroups;

    public int MatchedDispatchCount
    {
        get { lock (gate) { return matchedDispatchCount; } }
    }

    /// <summary>
    /// Queues initialization before measured command intervals begin. The returned exact fence
    /// must complete before the first matching dispatch.
    /// </summary>
    public GpuFence PrepareRecordLayout()
    {
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed || kernel is not null || stream is not null || sink is not null)
                throw new InvalidOperationException(
                    "The branch-divergence record layout can only be prepared once before dispatch.");
        }

        uint[] initial = CreateInitialWords();
        GpuBuffer<uint>? preparedStream = GpuBuffer<uint>.Create(
            context,
            initial.Length,
            BufferAccess.ReadWrite);
        GpuBuffer<uint>? preparedSink = GpuBuffer<uint>.Create(
            context,
            initial.Length,
            BufferAccess.ReadWrite);
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
                {
                    throw new InvalidOperationException(
                        "The branch-divergence record layout changed while it was being prepared.");
                }
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
                throw new InvalidOperationException("The branch-divergence capture is already complete.");
            kernel ??= GpuKernel.CreateBranchDivergence<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV5 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The branch-divergence capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.BranchDivergence ||
                resolvedLayout.SiteCount is < 1 or > GpuExecutionHeatCapture.MaximumSites ||
                resolvedLayout.SourceSiteIndex != sourceSiteIndex ||
                resolvedLayout.HeaderStrideBytes != HeaderStrideBytes ||
                resolvedLayout.RecordStrideBytes != RecordStrideBytes ||
                resolvedLayout.RecordCapacity != (uint)recordCapacity ||
                resolvedLayout.RequiredSubgroupFeatures != RequiredSubgroupFeatures ||
                resolvedLayout.Reserved != 0u)
            {
                throw new InvalidDataException("The native branch-divergence stream ABI is unsupported.");
            }

            if ((stream is null) != (sink is null))
                throw new InvalidDataException("The prepared branch-divergence record layout is incomplete.");
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
                throw new InvalidOperationException("Branch-divergence capture binding is not active.");

            int dispatchIndex = matchedDispatchCount;
            matchedDispatchCount = checked(matchedDispatchCount + 1);
            if (dispatchIndex != targetDispatchIndex)
            {
                command.BindDiagnosticBuffer(sink);
                return;
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

    public GpuBranchDivergenceResult CompleteAndRead()
    {
        GpuBuffer<uint> capturedStream;
        FeKernelDiagnosticLayoutV5 capturedLayout;
        GpuExecutionHeatDispatch? capturedDispatch;
        int matched;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
                throw new InvalidOperationException("The branch-divergence capture was already completed.");
            if (kernel is null || stream is null || !targetBound)
            {
                throw new InvalidOperationException(
                    "The requested matching dispatch was not observed during the branch-divergence capture.");
            }
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
        uint subgroupCount = words[6];
        uint mixedSubgroups = words[7];
        uint activeLanes = words[8];
        uint trueLanes = words[9];
        uint falseLanes = words[10];
        uint uniformTrue = words[11];
        uint uniformFalse = words[12];
        if (words.Length != checked(HeaderWordCount + recordCapacity * RecordWordCount) ||
            words[0] != StreamSchemaVersion || words[1] != RecordStrideBytes ||
            words[2] != (uint)recordCapacity || words[13] != sourceSiteIndex ||
            words[14] != RequiredSubgroupFeatures || words[15] != 0u ||
            committed > (uint)recordCapacity || (ulong)attempted != (ulong)committed + dropped ||
            subgroupCount != attempted || mixedSubgroups > subgroupCount ||
            (ulong)trueLanes + falseLanes != activeLanes ||
            (ulong)mixedSubgroups + uniformTrue + uniformFalse != subgroupCount)
        {
            throw new InvalidDataException("The branch-divergence stream header invariants are invalid.");
        }

        var records = new List<GpuBranchDivergenceRecord>(checked((int)committed));
        var identities = new HashSet<(int X, int Y, int Z, uint Subgroup)>();
        ulong recordedActive = 0;
        ulong recordedTrue = 0;
        ulong recordedFalse = 0;
        uint recordedMixed = 0;
        uint recordedUniformTrue = 0;
        uint recordedUniformFalse = 0;
        for (int recordIndex = 0; recordIndex < committed; ++recordIndex)
        {
            int offset = checked(HeaderWordCount + recordIndex * RecordWordCount);
            uint site = words[offset];
            var workgroup = new int3(
                checked((int)words[offset + 1]),
                checked((int)words[offset + 2]),
                checked((int)words[offset + 3]));
            uint subgroupId = words[offset + 4];
            uint subgroupSize = words[offset + 5];
            uint numSubgroups = words[offset + 6];
            uint electedInvocation = words[offset + 7];
            uint active = words[offset + 8];
            uint trueCount = words[offset + 9];
            uint falseCount = words[offset + 10];
            uint mixed = words[offset + 11];
            var activeMask = new GpuSubgroupBallotMask(
                words[offset + 12], words[offset + 13], words[offset + 14], words[offset + 15]);
            var trueMask = new GpuSubgroupBallotMask(
                words[offset + 16], words[offset + 17], words[offset + 18], words[offset + 19]);
            bool expectedMixed = trueCount > 0u && falseCount > 0u;
            if (capturedDispatch is null || site != sourceSiteIndex || site >= capturedLayout.SiteCount ||
                workgroup.X < 0 || workgroup.X >= capturedDispatch.Value.WorkgroupCountX ||
                workgroup.Y < 0 || workgroup.Y >= capturedDispatch.Value.WorkgroupCountY ||
                workgroup.Z < 0 || workgroup.Z >= capturedDispatch.Value.WorkgroupCountZ ||
                subgroupSize == 0u || subgroupSize != backendSubgroups.ReportedSize ||
                numSubgroups == 0u || subgroupId >= numSubgroups ||
                electedInvocation >= subgroupSize || active is 0u || active > subgroupSize ||
                (ulong)trueCount + falseCount != active || mixed > 1u ||
                (mixed == 1u) != expectedMixed || activeMask.SetBitCount != active ||
                trueMask.SetBitCount != trueCount || !trueMask.IsSubsetOf(activeMask) ||
                !identities.Add((workgroup.X, workgroup.Y, workgroup.Z, subgroupId)))
            {
                throw new InvalidDataException(
                    "A branch-divergence record has invalid subgroup identity, counts, or masks.");
            }

            recordedActive += active;
            recordedTrue += trueCount;
            recordedFalse += falseCount;
            if (expectedMixed) recordedMixed++;
            else if (trueCount == active) recordedUniformTrue++;
            else recordedUniformFalse++;
            records.Add(new GpuBranchDivergenceRecord(
                site,
                workgroup,
                subgroupId,
                subgroupSize,
                numSubgroups,
                electedInvocation,
                active,
                trueCount,
                falseCount,
                expectedMixed,
                activeMask,
                trueMask));
        }

        if (dropped == 0u &&
            (recordedActive != activeLanes || recordedTrue != trueLanes ||
             recordedFalse != falseLanes || recordedMixed != mixedSubgroups ||
             recordedUniformTrue != uniformTrue || recordedUniformFalse != uniformFalse))
        {
            throw new InvalidDataException(
                "The branch-divergence aggregate counters contradict the retained records.");
        }

        return new GpuBranchDivergenceResult(
            shaderTypeName,
            sourceSiteIndex,
            targetDispatchIndex,
            matched,
            recordCapacity,
            attempted,
            committed,
            dropped,
            subgroupCount,
            mixedSubgroups,
            activeLanes,
            trueLanes,
            falseLanes,
            uniformTrue,
            uniformFalse,
            records,
            capturedDispatch,
            backendSubgroups);
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
        words[0] = StreamSchemaVersion;
        words[1] = RecordStrideBytes;
        words[2] = checked((uint)recordCapacity);
        words[13] = sourceSiteIndex;
        words[14] = RequiredSubgroupFeatures;
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
