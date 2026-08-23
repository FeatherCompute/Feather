using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather;

/// <summary>Scalar encoding retained by a selected-invocation line-value record.</summary>
public enum GpuLineValueType : uint
{
    Bool32 = 1,
    Int32 = 2,
    UInt32 = 3,
    Float32 = 4,
}

/// <summary>
/// Immutable LAST-occurrence result for one typed FEIR statement and one compute invocation.
/// Raw component words preserve bool/int/uint/float32 bits without GPU-side formatting.
/// </summary>
public sealed record GpuLineValueResult(
    string ShaderTypeName,
    uint SourceSiteIndex,
    int TargetDispatchIndex,
    int MatchedDispatchCount,
    int3 SelectedInvocation,
    bool Executed,
    uint OccurrenceCount,
    GpuLineValueType ValueType,
    int ComponentCount,
    IReadOnlyList<uint> RawComponents,
    bool HasNaN,
    bool HasInfinity,
    GpuExecutionHeatDispatch? Dispatch);

/// <summary>
/// Context-scoped selected-invocation value probe. The diagnostic kernel and fixed 64-byte
/// record are private scratch resources and are never inserted into the ordinary kernel cache.
/// The caller must complete the submission before <see cref="CompleteAndRead"/>.
/// </summary>
public sealed class GpuLineValueCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 2;
    public const int RecordStrideBytes = 64;
    private const int RecordWordCount = RecordStrideBytes / sizeof(uint);

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly uint sourceSiteIndex;
    private readonly int targetDispatchIndex;
    private readonly int3 selectedInvocation;
    private GpuKernel? kernel;
    private GpuBuffer<uint>? record;
    private GpuBuffer<uint>? sink;
    private FeKernelDiagnosticLayoutV2 layout;
    private GpuExecutionHeatDispatch? dispatch;
    private int matchedDispatchCount;
    private bool targetBound;
    private bool completed;
    private bool disposed;

    internal GpuLineValueCapture(
        GpuContext context,
        string shaderTypeName,
        uint sourceSiteIndex,
        int targetDispatchIndex,
        int3 selectedInvocation)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.sourceSiteIndex = sourceSiteIndex;
        this.targetDispatchIndex = targetDispatchIndex;
        this.selectedInvocation = selectedInvocation;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (selectedInvocation.X < 0 || selectedInvocation.Y < 0 || selectedInvocation.Z < 0)
            throw new ArgumentOutOfRangeException(nameof(selectedInvocation));
    }

    public string ShaderTypeName => shaderTypeName;
    public uint SourceSiteIndex => sourceSiteIndex;
    public int TargetDispatchIndex => targetDispatchIndex;
    public int3 SelectedInvocation => selectedInvocation;

    public int MatchedDispatchCount
    {
        get { lock (gate) { return matchedDispatchCount; } }
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
                throw new InvalidOperationException("The line-value capture is already complete.");
            kernel ??= GpuKernel.CreateLineValue<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV2 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The line-value capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.LineValue ||
                resolvedLayout.SourceSiteIndex != sourceSiteIndex ||
                resolvedLayout.RecordStrideBytes != RecordStrideBytes ||
                resolvedLayout.RecordCapacity != 1 ||
                resolvedLayout.ValueTypeTag is < 1 or > 4 ||
                resolvedLayout.ComponentCount is < 1 or > 4)
            {
                throw new InvalidDataException("The native line-value record ABI is unsupported.");
            }

            record = GpuBuffer<uint>.Create(context, new uint[RecordWordCount], BufferAccess.ReadWrite);
            sink = GpuBuffer<uint>.Create(context, new uint[RecordWordCount], BufferAccess.ReadWrite);
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
            if (completed || !ReferenceEquals(kernel, dispatchKernel) || record is null || sink is null)
                throw new InvalidOperationException("Line-value capture binding is not active.");

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

            command.BindDiagnosticBuffer(record);
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

    public GpuLineValueResult CompleteAndRead()
    {
        GpuBuffer<uint> capturedRecord;
        FeKernelDiagnosticLayoutV2 capturedLayout;
        GpuExecutionHeatDispatch? capturedDispatch;
        int matched;
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (completed)
                throw new InvalidOperationException("The line-value capture was already completed.");
            if (kernel is null || record is null || !targetBound)
                throw new InvalidOperationException(
                    "The requested matching dispatch was not observed during the capture.");
            completed = true;
            capturedRecord = record;
            capturedLayout = layout;
            capturedDispatch = dispatch;
            matched = matchedDispatchCount;
        }

        context.EndDiagnosticCapture(this);
        uint[] words = capturedRecord.ToArray();
        bool executed = words[1] != 0;
        if (!executed)
        {
            if (words.Any(static word => word != 0))
                throw new InvalidDataException("An unexecuted line-value record contains payload data.");
            return new GpuLineValueResult(
                shaderTypeName,
                sourceSiteIndex,
                targetDispatchIndex,
                matched,
                selectedInvocation,
                false,
                0,
                (GpuLineValueType)capturedLayout.ValueTypeTag,
                checked((int)capturedLayout.ComponentCount),
                [],
                false,
                false,
                capturedDispatch);
        }

        if (words[0] != 1 || words[2] == 0 || words[3] != sourceSiteIndex ||
            words[4] != (uint)selectedInvocation.X ||
            words[5] != (uint)selectedInvocation.Y ||
            words[6] != (uint)selectedInvocation.Z ||
            words[7] != capturedLayout.ValueTypeTag ||
            words[8] != capturedLayout.ComponentCount)
        {
            throw new InvalidDataException("The line-value record identity or layout is invalid.");
        }

        int components = checked((int)words[8]);
        uint[] raw = words.AsSpan(9, components).ToArray();
        bool hasNaN = false;
        bool hasInfinity = false;
        if ((GpuLineValueType)words[7] == GpuLineValueType.Float32)
        {
            foreach (uint bits in raw)
            {
                float value = BitConverter.UInt32BitsToSingle(bits);
                hasNaN |= float.IsNaN(value);
                hasInfinity |= float.IsInfinity(value);
            }
        }
        return new GpuLineValueResult(
            shaderTypeName,
            sourceSiteIndex,
            targetDispatchIndex,
            matched,
            selectedInvocation,
            true,
            words[2],
            (GpuLineValueType)words[7],
            components,
            raw,
            hasNaN,
            hasInfinity,
            capturedDispatch);
    }

    public void Dispose()
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedRecord;
        GpuBuffer<uint>? ownedSink;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            ownedKernel = kernel;
            ownedRecord = record;
            ownedSink = sink;
            kernel = null;
            record = null;
            sink = null;
        }
        context.EndDiagnosticCapture(this);
        ownedKernel?.Dispose();
        ownedRecord?.Dispose();
        ownedSink?.Dispose();
    }

    void IGpuDiagnosticCapture.DisposeForContextShutdown()
    {
        GpuKernel? ownedKernel;
        GpuBuffer<uint>? ownedRecord;
        GpuBuffer<uint>? ownedSink;
        lock (gate)
        {
            if (disposed) return;
            disposed = true;
            ownedKernel = kernel;
            ownedRecord = record;
            ownedSink = sink;
            kernel = null;
            record = null;
            sink = null;
        }
        ownedKernel?.Dispose();
        ownedRecord?.Dispose();
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
