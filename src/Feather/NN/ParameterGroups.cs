namespace Feather.NN;

/// <summary>
/// Groups module parameters under a stable optimizer/checkpoint name.
/// </summary>
public sealed class ParameterGroup
{
    /// <summary>
    /// Initializes a parameter group.
    /// </summary>
    public ParameterGroup(string name, IEnumerable<IParameter> parameters, float? learningRate = null, float? weightDecay = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(parameters);
        if (weightDecay is < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(weightDecay), "Weight decay must be non-negative.");
        }

        Name = name;
        Parameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters));
        LearningRate = learningRate;
        WeightDecay = weightDecay;
    }

    /// <summary>
    /// Gets the group name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Gets the parameters in this group.
    /// </summary>
    public IReadOnlyList<IParameter> Parameters { get; }

    /// <summary>
    /// Gets an optional group-specific learning rate.
    /// </summary>
    public float? LearningRate { get; }

    /// <summary>
    /// Gets the group-specific weight decay.
    /// </summary>
    public float? WeightDecay { get; }
}

/// <summary>
/// Common parameter initialization helpers for NN modules.
/// </summary>
public static class ParameterInitializers
{
    /// <summary>
    /// Creates Xavier/Glorot uniform values.
    /// </summary>
    public static float[] XavierUniform(int count, int fanIn, int fanOut, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fanIn);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fanOut);

        var values = new float[count];
        var random = new Random(seed);
        var range = MathF.Sqrt(6.0f / (fanIn + fanOut));
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = ((float)random.NextDouble() * 2f - 1f) * range;
        }

        return values;
    }

    /// <summary>
    /// Creates Xavier/Glorot uniform values using EasyGPU C++'s deterministic LCG stream.
    /// </summary>
    internal static float[] XavierUniformEasyGpuReference(int count, int fanIn, int fanOut, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fanIn);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(fanOut);

        var values = new float[count];
        var state = unchecked((uint)seed);
        var range = MathF.Sqrt(6.0f / (fanIn + fanOut));
        for (var i = 0; i < values.Length; i++)
        {
            state = unchecked((state * 1664525u) + 1013904223u);
            var sample = (float)((double)state / uint.MaxValue);
            values[i] = ((sample * 2f) - 1f) * range;
        }

        return values;
    }

    /// <summary>
    /// Creates a trainable float parameter initialized with EasyGPU C++'s deterministic Xavier stream.
    /// </summary>
    internal static Parameter<float> XavierParameterEasyGpuReference(string name, TensorShape shape, int fanIn, int fanOut, int seed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        var values = XavierUniformEasyGpuReference(shape.ElementCount, fanIn, fanOut, seed);
        return new Parameter<float>(
            name,
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(shape.ElementCount)));
    }

    /// <summary>
    /// Creates a trainable float parameter initialized with Xavier/Glorot uniform values.
    /// </summary>
    public static Parameter<float> XavierParameter(string name, TensorShape shape, int fanIn, int fanOut, int seed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        var values = XavierUniform(shape.ElementCount, fanIn, fanOut, seed);
        return new Parameter<float>(
            name,
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(shape.ElementCount)));
    }

    /// <summary>
    /// Creates a trainable float parameter initialized to a constant value.
    /// </summary>
    public static Parameter<float> ConstantParameter(string name, TensorShape shape, float value = 0f)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        var values = new float[shape.ElementCount];
        if (value != 0f)
        {
            Array.Fill(values, value);
        }

        return new Parameter<float>(
            name,
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(shape, Feather.GPU.CreateBuffer<float>(shape.ElementCount)));
    }
}
