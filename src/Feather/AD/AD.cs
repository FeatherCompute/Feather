using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;
using System.Text;

namespace Feather.AD;

internal readonly record struct NativeAdGradientMatch(uint Index, string Name);

/// <summary>
/// Shader marker methods for automatic differentiation. These are lowered by the Roslyn generator
/// into IR annotations and must not be called on the CPU.
/// </summary>
public static class AD
{
    /// <summary>Marks a scalar value as a differentiable parameter.</summary>
    public static void Parameter(float value) => Marker();
    /// <summary>Marks a float2 value as a differentiable parameter.</summary>
    public static void Parameter(float2 value) => Marker();
    /// <summary>Marks a float3 value as a differentiable parameter.</summary>
    public static void Parameter(float3 value) => Marker();
    /// <summary>Marks a float4 value as a differentiable parameter.</summary>
    public static void Parameter(float4 value) => Marker();
    /// <summary>Marks the scalar loss value used to seed the backward pass.</summary>
    public static void Loss(float value) => Marker();
    /// <summary>Rejects non-scalar loss markers inside generated GPU code.</summary>
    public static void Loss(float2 value) => Marker();
    /// <summary>Rejects non-scalar loss markers inside generated GPU code.</summary>
    public static void Loss(float3 value) => Marker();
    /// <summary>Rejects non-scalar loss markers inside generated GPU code.</summary>
    public static void Loss(float4 value) => Marker();

    private static void Marker()
        => throw new InvalidOperationException("AD markers can only be used inside Feather-generated GPU code.");
}

/// <summary>
/// A GPU-accelerated automatic differentiation kernel that dispatches forward and backward passes
/// through the EasyGPU AD engine.
/// </summary>
/// <typeparam name="TKernel">The generated kernel type.</typeparam>
public sealed class GpuADKernel<TKernel> : IDisposable
    where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
{
    private bool disposed;
    private readonly GpuKernel gpuKernel;
    private readonly GpuKernel forwardOnlyKernel;
    private DispatchPath lastDispatchPath = DispatchPath.None;

    /// <summary>
    /// Initializes a new AD kernel wrapper.
    /// </summary>
    /// <param name="kernel">The generated kernel struct holding bound resources.</param>
    public GpuADKernel(TKernel kernel)
    {
        Kernel = kernel;
        gpuKernel = GpuKernel.Create<TKernel>(GPU.Context);
        forwardOnlyKernel = GpuKernel.Create<TKernel>(GPU.Context, autoDiff: false);
        Gradients = new GradientSet(ReadBackGradientsCore);
    }

    /// <summary>
    /// Gets the wrapped kernel struct.
    /// </summary>
    public TKernel Kernel { get; }

    /// <summary>
    /// Gets the named gradient set produced by the most recent backward launch.
    /// </summary>
    public GradientSet Gradients { get; }

    /// <summary>
    /// Gets the element count used by the most recent backward launch.
    /// </summary>
    public int LastBackwardCount { get; private set; }

    /// <summary>
    /// Gets a value indicating whether <see cref="Backward"/> has been invoked successfully.
    /// </summary>
    public bool HasBackwardRun { get; private set; }

    /// <summary>
    /// Gets the native route used by the most recent backward launch.
    /// </summary>
    public DispatchPath LastDispatchPath
    {
        get
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            return lastDispatchPath;
        }
    }

    /// <summary>
    /// Runs the forward pass, generates the adjoint kernel via EasyGPU, dispatches the backward pass,
    /// and leaves native gradient buffers available for device-side handoff or explicit readback.
    /// </summary>
    /// <param name="count">The number of logical elements to include.</param>
    public void Backward(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        HasBackwardRun = false;
        Gradients.Clear();

        GpuKernel.Dispatch(GPU.Context, gpuKernel, Kernel, new GpuDispatchSize(count, 1, 1), wait: true);

        LastBackwardCount = count;
        HasBackwardRun = true;
        lastDispatchPath = gpuKernel.LastDispatchPath;
        Gradients.MarkNeedsLoad();
    }

    /// <summary>
    /// Runs only the generated forward pass for an AD-authored kernel and invalidates any previous
    /// backward gradient state.
    /// </summary>
    /// <param name="count">The number of logical elements to include.</param>
    public void Forward(int count)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);

        HasBackwardRun = false;
        LastBackwardCount = 0;
        Gradients.ClearNonMaterialized();

        GpuKernel.Dispatch(GPU.Context, forwardOnlyKernel, Kernel, new GpuDispatchSize(count, 1, 1), wait: true);
        lastDispatchPath = forwardOnlyKernel.LastDispatchPath;
    }

    /// <summary>
    /// Reads the native AD gradients into a managed <see cref="GradientSet"/> for diagnostics and tests.
    /// Prefer <see cref="CopyGradientToBuffer"/> or optimizer device handoff for production training paths.
    /// </summary>
    public GradientSet ReadBackGradients()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (!HasBackwardRun)
        {
            throw new InvalidOperationException("Backward must run successfully before gradients can be read.");
        }

        Gradients.Clear();
        ReadBackGradientsCore();
        return Gradients;
    }

    /// <summary>
    /// Copies a named native AD gradient into a destination GPU buffer, reducing per-dispatch-thread
    /// gradient storage on the device.
    /// </summary>
    public void CopyGradientToBuffer(string gradientName, GpuBuffer<float> destination)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(gradientName);
        ArgumentNullException.ThrowIfNull(destination);
        if (!HasBackwardRun)
        {
            throw new InvalidOperationException("Backward must run successfully before gradients can be copied.");
        }

        var matches = FindGradientMatches([gradientName], destination.Length);
        var index = matches.Length == 0
            ? throw new KeyNotFoundException($"No native AD gradient named '{gradientName}' exists.")
            : matches[0].Index;
        CopyGradientToBuffer(index, destination);
    }

    internal void CopyGradientToBuffer(uint gradientIndex, GpuBuffer<float> destination)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(destination);
        if (!HasBackwardRun)
        {
            throw new InvalidOperationException("Backward must run successfully before gradients can be copied.");
        }

        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_reduce_ad_gradient_to_buffer(
            gpuKernel.Handle,
            gradientIndex,
            destination.Handle,
            0,
            (ulong)destination.SizeInBytes));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed) return;
        forwardOnlyKernel.Dispose();
        gpuKernel.Dispose();
        disposed = true;
    }

    /// <summary>
    /// Gets the merged forward/backward GLSL generated by the most recent successful backward launch.
    /// </summary>
    public string GetBackwardGLSL()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return NativeStringCall.GetString((IntPtr buffer, UIntPtr length, out UIntPtr required) =>
            NativeMethods.fe_kernel_get_ad_backward_glsl(gpuKernel.Handle, buffer, length, out required));
    }

    internal unsafe NativeAdGradientMatch[] FindGradientMatches(IEnumerable<string> gradientNames, int destinationLength)
    {
        ArgumentNullException.ThrowIfNull(gradientNames);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(destinationLength);
        var candidates = new HashSet<string>(gradientNames.Where(name => !string.IsNullOrWhiteSpace(name)), StringComparer.Ordinal);
        if (candidates.Count == 0)
        {
            return [];
        }

        var matches = new List<NativeAdGradientMatch>();
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_count(gpuKernel.Handle, out var count));
        for (uint i = 0; i < count; i++)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_info(gpuKernel.Handle, i, out var info));
            var name = FixedString(info.Name, 128);
            if (string.IsNullOrWhiteSpace(name))
            {
                name = FixedString(info.ResourceName, 128);
            }

            if (!candidates.Contains(name))
            {
                continue;
            }

            var elementType = FixedString(info.ElementType, 64);
            if (NormalizeNativeElementType(elementType) != "float")
            {
                throw new NotSupportedException($"AD gradient '{name}' has element type '{elementType}', but only float gradients can be copied into float parameter buffers.");
            }

            var scalarSlots = checked((int)info.ElementCount * (int)System.Math.Max(info.ComponentCount, 1u));
            if (scalarSlots != destinationLength)
            {
                throw new ArgumentException($"Gradient '{name}' has {scalarSlots} values but destination buffer expects {destinationLength}.", nameof(destinationLength));
            }

            matches.Add(new NativeAdGradientMatch(i, name));
        }

        return [.. matches];
    }

    private unsafe void ReadBackGradientsCore()
    {
        NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_count(gpuKernel.Handle, out var count));
        if (count == 0)
        {
            throw new InvalidOperationException("AD backward completed without producing any gradient buffers.");
        }

        for (uint i = 0; i < count; i++)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_get_ad_gradient_info(gpuKernel.Handle, i, out var info));
            var name = FixedString(info.Name, 128);
            if (string.IsNullOrWhiteSpace(name))
            {
                name = FixedString(info.ResourceName, 128);
            }

            var elementType = FixedString(info.ElementType, 64);
            var floatCount = checked((int)(info.ByteSize / sizeof(float)));
            if (floatCount <= 0 || info.ByteSize % sizeof(float) != 0)
            {
                throw new InvalidOperationException($"AD gradient '{name}' has invalid byte size {info.ByteSize}.");
            }

            var raw = new float[floatCount];
            fixed (float* ptr = raw)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_kernel_read_ad_gradient(
                    gpuKernel.Handle,
                    i,
                    0,
                    info.ByteSize,
                    (IntPtr)ptr));
            }

            var scalarSlots = checked((int)info.ElementCount * (int)System.Math.Max(info.ComponentCount, 1u));
            if (scalarSlots <= 0 || raw.Length % scalarSlots != 0)
            {
                throw new InvalidOperationException($"AD gradient '{name}' has an invalid scalar layout.");
            }

            var reduced = new float[scalarSlots];
            for (var offset = 0; offset < raw.Length; offset += scalarSlots)
            {
                for (var slot = 0; slot < scalarSlots; slot++)
                {
                    reduced[slot] += raw[offset + slot];
                }
            }

            Gradients.RegisterNative(name, elementType, reduced);
        }
    }

    internal static string NormalizeNativeElementType(string elementType)
        => elementType switch
        {
            "Feather.Math.float2" or "global::Feather.Math.float2" => "float2",
            "Feather.Math.float3" or "global::Feather.Math.float3" => "float3",
            "Feather.Math.float4" or "global::Feather.Math.float4" => "float4",
            _ => elementType
        };

    private static unsafe string FixedString(byte* value, int length)
    {
        var span = new ReadOnlySpan<byte>(value, length);
        var nul = span.IndexOf((byte)0);
        if (nul >= 0)
        {
            span = span[..nul];
        }

        return Encoding.UTF8.GetString(span);
    }
}

/// <summary>
/// A named collection of gradient values produced by an AD backward pass.
/// </summary>
public sealed class GradientSet
{
    private readonly Dictionary<string, object> gradients = [];
    private readonly Action? lazyLoader;
    private bool lazyLoaded;

    /// <summary>
    /// Initializes an empty gradient set.
    /// </summary>
    public GradientSet()
    {
    }

    internal GradientSet(Action lazyLoader)
        => this.lazyLoader = lazyLoader;

    /// <summary>
    /// Gets the registered gradient names.
    /// </summary>
    public IReadOnlyCollection<string> Names
    {
        get
        {
            EnsureLoaded();
            return gradients.Keys;
        }
    }

    /// <summary>
    /// Gets a value indicating whether the set currently holds managed gradient values.
    /// </summary>
    public bool HasMaterializedValues => lazyLoaded && gradients.Count > 0;

    /// <summary>
    /// Checks whether a gradient with the given name exists, regardless of element type.
    /// </summary>
    public bool Contains(string name)
    {
        EnsureLoaded();
        return gradients.ContainsKey(name);
    }

    /// <summary>
    /// Registers a scalar gradient for a named parameter.
    /// </summary>
    public void Register<T>(string name, T gradient)
        where T : unmanaged
    {
        lazyLoaded = true;
        gradients[name] = gradient;
    }

    /// <summary>
    /// Registers a vector gradient for a named parameter.
    /// </summary>
    /// <param name="name">The parameter name.</param>
    /// <param name="gradient">The gradient values.</param>
    public void Register<T>(string name, ReadOnlySpan<T> gradient)
        where T : unmanaged
    {
        lazyLoaded = true;
        gradients[name] = gradient.ToArray();
    }

    internal void RegisterNative(string name, string elementType, ReadOnlySpan<float> gradient)
    {
        lazyLoaded = true;
        gradients[name] = NormalizeNativeElementType(elementType) switch
        {
            "float" => gradient.ToArray(),
            "float2" => ConvertNativeFloat2(gradient),
            "float3" => ConvertNativeFloat3(gradient),
            "float4" => ConvertNativeFloat4(gradient),
            _ => throw new NotSupportedException($"AD gradient element type '{elementType}' is not supported by managed readback.")
        };
    }

    /// <summary>
    /// Removes all registered gradients.
    /// </summary>
    public void Clear()
    {
        gradients.Clear();
        lazyLoaded = true;
    }

    /// <summary>
    /// Attempts to read a scalar gradient by name.
    /// </summary>
    public bool TryGet<T>(string name, out T gradient)
        where T : unmanaged
    {
        EnsureLoaded();
        if (gradients.TryGetValue(name, out var value) && value is T typed)
        {
            gradient = typed;
            return true;
        }
        gradient = default;
        return false;
    }

    /// <summary>
    /// Attempts to read a vector gradient by name.
    /// </summary>
    public bool TryGetArray<T>(string name, out T[] gradient)
        where T : unmanaged
    {
        EnsureLoaded();
        if (gradients.TryGetValue(name, out var value) && value is T[] typed)
        {
            gradient = typed;
            return true;
        }
        gradient = [];
        return false;
    }

    /// <summary>
    /// Reads an array gradient by name or throws if the name/type does not match.
    /// </summary>
    public T[] Get<T>(string name)
        where T : unmanaged
        => GetArray<T>(name);

    /// <summary>
    /// Reads an array gradient by name or throws if the name/type does not match.
    /// </summary>
    public T[] GetArray<T>(string name)
        where T : unmanaged
        => TryGetArray<T>(name, out var gradient)
            ? gradient
            : throw new KeyNotFoundException($"No array gradient named '{name}' with element type '{typeof(T).Name}' exists.");

    /// <summary>
    /// Reads a scalar gradient by name or throws if the name/type does not match.
    /// </summary>
    public T GetScalar<T>(string name)
        where T : unmanaged
        => TryGet<T>(name, out var gradient)
            ? gradient
            : throw new KeyNotFoundException($"No scalar gradient named '{name}' with element type '{typeof(T).Name}' exists.");

    private static float2[] ConvertNativeFloat2(ReadOnlySpan<float> gradient)
    {
        EnsureComponentMultiple(gradient, 2);
        var result = new float2[gradient.Length / 2];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new float2(gradient[i * 2], gradient[i * 2 + 1]);
        }

        return result;
    }

    private static float3[] ConvertNativeFloat3(ReadOnlySpan<float> gradient)
    {
        EnsureComponentMultiple(gradient, 3);
        var result = new float3[gradient.Length / 3];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new float3(gradient[i * 3], gradient[i * 3 + 1], gradient[i * 3 + 2]);
        }

        return result;
    }

    private static float4[] ConvertNativeFloat4(ReadOnlySpan<float> gradient)
    {
        EnsureComponentMultiple(gradient, 4);
        var result = new float4[gradient.Length / 4];
        for (var i = 0; i < result.Length; i++)
        {
            result[i] = new float4(gradient[i * 4], gradient[i * 4 + 1], gradient[i * 4 + 2], gradient[i * 4 + 3]);
        }

        return result;
    }

    private static void EnsureComponentMultiple(ReadOnlySpan<float> gradient, int components)
    {
        if (gradient.Length % components != 0)
        {
            throw new InvalidOperationException($"Native AD gradient has {gradient.Length} components, which is not divisible by {components}.");
        }
    }

    private void EnsureLoaded()
    {
        if (lazyLoaded || lazyLoader is null)
        {
            return;
        }

        lazyLoaded = true;
        lazyLoader();
    }

    internal void MarkNeedsLoad()
        => lazyLoaded = false;

    internal void ClearNonMaterialized()
    {
        gradients.Clear();
        lazyLoaded = true;
    }

    private static string NormalizeNativeElementType(string elementType)
        => elementType switch
        {
            "Feather.Math.float2" or "global::Feather.Math.float2" => "float2",
            "Feather.Math.float3" or "global::Feather.Math.float3" => "float3",
            "Feather.Math.float4" or "global::Feather.Math.float4" => "float4",
            _ => elementType
        };
}
