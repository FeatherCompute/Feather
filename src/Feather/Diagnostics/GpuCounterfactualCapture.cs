using Feather.Interop;
using Feather.Math;
using Feather.Native;

namespace Feather;

/// <summary>
/// Source transformation supported by a private counterfactual compute-kernel variant.
/// Counterfactual transforms intentionally change the program and never establish equivalence.
/// </summary>
public enum GpuCounterfactualTransform
{
    /// <summary>Evaluates the selected <c>if</c> as false, selecting its else path when present.</summary>
    ForceIfFalse = 1
}

/// <summary>
/// Explicit substitution scope for one private counterfactual compute-kernel variant. The scope is
/// inert until <see cref="VariantEnabled"/> is true. Ordinary generated kernels remain in their
/// normal cache and are selected whenever the variant is disabled.
/// </summary>
/// <remarks>
/// This primitive exists for bounded differential profiling. It does not time work, compare
/// outputs, or claim that the transformed program is semantically equivalent to the baseline.
/// </remarks>
public sealed class GpuCounterfactualCapture : IDisposable, IGpuDiagnosticCapture
{
    public const uint AbiVersion = 7;

    private readonly object gate = new();
    private readonly GpuContext context;
    private readonly string shaderTypeName;
    private readonly uint sourceSiteIndex;
    private readonly int targetDispatchIndex;
    private readonly GpuCounterfactualTransform transform;
    private GpuKernel? kernel;
    private FeKernelDiagnosticLayoutV7 layout;
    private bool variantEnabled;
    private int matchingDispatchCount;
    private int lastIterationMatchingDispatchCount;
    private int variantDispatchCount;
    private bool disposed;

    internal GpuCounterfactualCapture(
        GpuContext context,
        string shaderTypeName,
        uint sourceSiteIndex,
        int targetDispatchIndex,
        GpuCounterfactualTransform transform)
    {
        this.context = context;
        this.shaderTypeName = NormalizeTypeName(shaderTypeName);
        this.sourceSiteIndex = sourceSiteIndex;
        this.targetDispatchIndex = targetDispatchIndex;
        this.transform = transform;
        if (this.shaderTypeName.Length == 0)
            throw new ArgumentException("Shader type name is empty after normalization.", nameof(shaderTypeName));
        if (sourceSiteIndex == uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(sourceSiteIndex));
        if (targetDispatchIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(targetDispatchIndex));
        if (transform != GpuCounterfactualTransform.ForceIfFalse)
            throw new ArgumentOutOfRangeException(nameof(transform));
    }

    public string ShaderTypeName => shaderTypeName;
    public uint SourceSiteIndex => sourceSiteIndex;
    public int TargetDispatchIndex => targetDispatchIndex;
    public GpuCounterfactualTransform Transform => transform;

    /// <summary>
    /// Enables or disables substitution of the private variant for matching generated-kernel
    /// lookups. This must be changed only at a caller-owned command-recording boundary.
    /// </summary>
    public bool VariantEnabled
    {
        get { lock (gate) { return variantEnabled; } }
        set
        {
            lock (gate)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                if (variantEnabled == value)
                    return;
                if (value)
                {
                    matchingDispatchCount = 0;
                }
                else
                {
                    lastIterationMatchingDispatchCount = matchingDispatchCount;
                }
                variantEnabled = value;
            }
        }
    }

    /// <summary>Number of matching dispatches recorded through the private variant.</summary>
    public int VariantDispatchCount
    {
        get { lock (gate) { return variantDispatchCount; } }
    }

    /// <summary>
    /// Number of matching generated-kernel lookups observed in the most recently completed
    /// enabled interval. A bounded profiler uses this to reject a stale dispatch index.
    /// </summary>
    public int LastIterationMatchingDispatchCount
    {
        get { lock (gate) { return lastIterationMatchingDispatchCount; } }
    }

    /// <summary>Number of retained typed source sites reported by the configured variant.</summary>
    public uint SiteCount
    {
        get { lock (gate) { return layout.SiteCount; } }
    }

    bool IGpuDiagnosticCapture.TryGetOrCreateKernel<TKernel>(
        bool autoDiff,
        out GpuKernel diagnosticKernel)
    {
        string candidate = NormalizeTypeName(typeof(TKernel).FullName ?? typeof(TKernel).Name);
        lock (gate)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            if (!variantEnabled || !string.Equals(candidate, shaderTypeName, StringComparison.Ordinal))
            {
                diagnosticKernel = null!;
                return false;
            }

            int dispatchIndex = matchingDispatchCount;
            matchingDispatchCount = checked(matchingDispatchCount + 1);
            if (dispatchIndex != targetDispatchIndex)
            {
                diagnosticKernel = null!;
                return false;
            }

            kernel ??= GpuKernel.CreateCounterfactual<TKernel>(context, this, autoDiff);
            diagnosticKernel = kernel;
            return true;
        }
    }

    internal void AttachKernel(GpuKernel attachedKernel, FeKernelDiagnosticLayoutV7 resolvedLayout)
    {
        lock (gate)
        {
            if (kernel is not null)
                throw new InvalidOperationException("The counterfactual capture already owns a kernel variant.");
            if (resolvedLayout.AbiVersion != AbiVersion ||
                resolvedLayout.Mode != (uint)FeKernelDiagnosticMode.Counterfactual ||
                resolvedLayout.SiteCount is < 1 or > GpuExecutionHeatCapture.MaximumSites ||
                resolvedLayout.SourceSiteIndex != sourceSiteIndex ||
                resolvedLayout.TransformKind != (uint)transform ||
                resolvedLayout.Flags != 0u ||
                resolvedLayout.Reserved != 0u)
            {
                throw new InvalidDataException("The native counterfactual variant ABI is unsupported.");
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
            if (!variantEnabled || !ReferenceEquals(kernel, dispatchKernel))
                throw new InvalidOperationException("Counterfactual variant binding is not active.");
            variantDispatchCount = checked(variantDispatchCount + 1);
        }
    }

    public void Dispose() => DisposeCore(endCapture: true);

    void IGpuDiagnosticCapture.DisposeForContextShutdown() => DisposeCore(endCapture: false);

    private void DisposeCore(bool endCapture)
    {
        GpuKernel? ownedKernel;
        lock (gate)
        {
            if (disposed)
                return;
            disposed = true;
            variantEnabled = false;
            ownedKernel = kernel;
            kernel = null;
        }
        if (endCapture)
            context.EndDiagnosticCapture(this);
        ownedKernel?.Dispose();
    }

    private static string NormalizeTypeName(string value)
    {
        string normalized = value.Trim();
        if (normalized.StartsWith("global::", StringComparison.Ordinal))
            normalized = normalized["global::".Length..];
        return normalized.Replace('+', '.');
    }
}
