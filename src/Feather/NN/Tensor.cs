using Feather.Resources;

namespace Feather.NN;

/// <summary>
/// Represents a typed tensor stored in a Feather GPU buffer.
/// </summary>
/// <typeparam name="T">The unmanaged element type stored by the tensor.</typeparam>
public sealed class Tensor<T> : IDisposable
    where T : unmanaged
{
    private static int liveInstanceCount;
    private bool disposed;

    /// <summary>
    /// Initializes a new tensor wrapper over an existing GPU buffer.
    /// </summary>
    /// <param name="shape">The logical tensor shape.</param>
    /// <param name="buffer">The backing GPU buffer.</param>
    /// <param name="requiresGrad">Whether operations using this tensor should preserve gradient tracking intent.</param>
    public Tensor(TensorShape shape, GpuBuffer<T> buffer, bool requiresGrad = false)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.Length != shape.ElementCount)
        {
            throw new ArgumentException($"Tensor shape contains {shape.ElementCount} elements but the buffer contains {buffer.Length}.", nameof(buffer));
        }

        Shape = shape;
        Buffer = buffer;
        RequiresGrad = requiresGrad;
        System.Threading.Interlocked.Increment(ref liveInstanceCount);
    }

    /// <summary>
    /// Gets the logical dimensions of the tensor.
    /// </summary>
    public TensorShape Shape { get; }

    /// <summary>
    /// Gets the backing Feather GPU buffer.
    /// </summary>
    public GpuBuffer<T> Buffer { get; }

    /// <summary>
    /// Gets the unmanaged element type stored by the tensor.
    /// </summary>
    public Type ElementType => typeof(T);

    /// <summary>
    /// Gets the number of tensor dimensions.
    /// </summary>
    public int Rank => Shape.Rank;

    /// <summary>
    /// Gets a value indicating whether gradient-aware modules should treat this tensor as differentiable.
    /// </summary>
    public bool RequiresGrad { get; }

    /// <summary>
    /// Gets the total number of logical elements in the tensor.
    /// </summary>
    public int ElementCount => Shape.ElementCount;

    internal bool IsDisposed => disposed;

    internal static int LiveInstanceCount => System.Threading.Volatile.Read(ref liveInstanceCount);

    /// <summary>
    /// Creates a shader-facing read-only buffer view over the tensor data.
    /// </summary>
    public ReadOnlyBuffer<T> AsReadOnlyBuffer() => Buffer.AsReadOnly();

    /// <summary>
    /// Creates a shader-facing read-write buffer view over the tensor data.
    /// </summary>
    public ReadWriteBuffer<T> AsReadWriteBuffer() => Buffer.AsReadWrite();

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Buffer.Dispose();
        disposed = true;
        System.Threading.Interlocked.Decrement(ref liveInstanceCount);
    }
}

/// <summary>
/// Describes the dimensions of a tensor.
/// </summary>
public readonly record struct TensorShape : IEquatable<TensorShape>
{
    private readonly int[]? dimensions;

    /// <summary>
    /// Initializes a new tensor shape and validates that every dimension is positive.
    /// </summary>
    /// <param name="dimensions">The ordered tensor dimensions.</param>
    public TensorShape(params int[] dimensions)
        : this()
    {
        ArgumentNullException.ThrowIfNull(dimensions);
        foreach (var dimension in dimensions)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(dimension);
        }

        this.dimensions = dimensions.ToArray();
    }

    /// <summary>
    /// Gets the ordered tensor dimensions.
    /// </summary>
    public int[] Dimensions => dimensions is null ? [] : dimensions.ToArray();

    /// <summary>
    /// Gets the number of dimensions in the tensor shape.
    /// </summary>
    public int Rank => Dimensions.Length;

    /// <summary>
    /// Gets the total number of elements represented by the dimensions.
    /// </summary>
    public int ElementCount
    {
        get
        {
            var count = 1;
            foreach (var dimension in Dimensions)
            {
                count = checked(count * dimension);
            }

            return count;
        }
    }

    /// <summary>
    /// Compares tensor shapes by dimension values.
    /// </summary>
    public bool Equals(TensorShape other)
        => Dimensions.AsSpan().SequenceEqual(other.Dimensions);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (var dimension in Dimensions)
        {
            hash.Add(dimension);
        }

        return hash.ToHashCode();
    }

}

/// <summary>
/// Describes a learnable module parameter.
/// </summary>
public interface IParameter
{
    /// <summary>
    /// Gets the stable parameter name used by optimizers, checkpoints, and AD gradient handoff.
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Gets the deterministic fully-qualified parameter name used by composed modules.
    /// </summary>
    string FullName { get; }

    /// <summary>
    /// Gets the names accepted when copying AD gradients into this parameter.
    /// </summary>
    IReadOnlyCollection<string> GradientNames { get; }

    /// <summary>
    /// Gets the number of logical scalar/vector elements in the parameter tensor.
    /// </summary>
    int ElementCount { get; }

    /// <summary>
    /// Clears this parameter's gradient tensor on the device when supported.
    /// </summary>
    void ZeroGrad();

    /// <summary>
    /// Qualifies this parameter for a module path.
    /// </summary>
    /// <param name="prefix">The deterministic module path prefix.</param>
    void Qualify(string prefix);

    /// <summary>
    /// Adds an explicit AD gradient alias accepted by optimizer handoff.
    /// </summary>
    /// <param name="name">The AD gradient name.</param>
    void AddGradientAlias(string name);
}

/// <summary>
/// Stores a learnable tensor value and its matching gradient tensor.
/// </summary>
/// <typeparam name="T">The unmanaged parameter element type.</typeparam>
public sealed class Parameter<T> : IDisposable, IParameter
    where T : unmanaged
{
    private readonly HashSet<string> gradientNames = new(StringComparer.Ordinal);

    /// <summary>
    /// Initializes a new parameter.
    /// </summary>
    /// <param name="name">The stable parameter name.</param>
    /// <param name="value">The learnable tensor value.</param>
    /// <param name="gradient">The gradient tensor with the same shape as <paramref name="value" />.</param>
    public Parameter(string name, Tensor<T> value, Tensor<T> gradient)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(gradient);
        if (!value.Shape.Equals(gradient.Shape))
        {
            throw new ArgumentException("Parameter gradient tensor must have the same shape as the value tensor.", nameof(gradient));
        }

        Name = name;
        FullName = name;
        Value = value;
        Gradient = gradient;
        gradientNames.Add(name);
    }

    /// <inheritdoc />
    public string Name { get; }

    /// <inheritdoc />
    public string FullName { get; private set; }

    /// <inheritdoc />
    public IReadOnlyCollection<string> GradientNames => gradientNames;

    /// <summary>
    /// Gets the learnable tensor value.
    /// </summary>
    public Tensor<T> Value { get; }

    /// <summary>
    /// Gets the gradient tensor associated with <see cref="Value" />.
    /// </summary>
    public Tensor<T> Gradient { get; }

    /// <inheritdoc />
    public int ElementCount => Value.Shape.ElementCount;

    /// <inheritdoc />
    public void ZeroGrad()
    {
        if (typeof(T) != typeof(float))
        {
            throw new NotSupportedException($"Device zero-grad is currently supported for float parameters, not {typeof(T).Name}.");
        }

        NnDeviceOps.Fill((Tensor<float>)(object)Gradient, 0f);
    }

    /// <inheritdoc />
    public void Qualify(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        FullName = $"{prefix}.{Name}";
        gradientNames.Clear();
        gradientNames.Add(FullName);
        gradientNames.Add(ToGradientAlias(FullName));
    }

    /// <inheritdoc />
    public void AddGradientAlias(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        gradientNames.Add(name);
    }

    private static string ToGradientAlias(string fullName)
        => fullName.Replace('.', '_');

    /// <inheritdoc />
    public void Dispose()
    {
        Value.Dispose();
        Gradient.Dispose();
    }
}
