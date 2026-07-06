using Feather.AD;
using Feather.Resources;
using Feather.Math;
using System.ComponentModel;

namespace Feather.NN;

/// <summary>
/// Base type for neural network modules that can own learnable parameters.
/// </summary>
public abstract class Module : IDisposable
{
    /// <summary>
    /// Gets the learnable parameters owned by this module.
    /// </summary>
    public abstract IEnumerable<IParameter> Parameters { get; }

    /// <summary>
    /// Qualifies every owned parameter with a deterministic module path.
    /// </summary>
    /// <param name="prefix">The module path prefix.</param>
    public virtual void QualifyParameters(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        foreach (var parameter in Parameters)
        {
            parameter.Qualify(prefix);
        }
    }

    /// <inheritdoc />
    public virtual void Dispose()
    {
    }
}

/// <summary>
/// Applies an affine transform over the last tensor dimension.
/// </summary>
public sealed class Linear : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a linear layer with weights stored as <c>[outputSize, inputSize]</c>.
    /// </summary>
    /// <param name="inputSize">The number of input features.</param>
    /// <param name="outputSize">The number of output features.</param>
    public Linear(int inputSize, int outputSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputSize);
        InputSize = inputSize;
        OutputSize = outputSize;
        Weight = FloatParameterFactory.Create("weight", new TensorShape(outputSize, inputSize), requiresGrad: true);
        Bias = FloatParameterFactory.Create("bias", new TensorShape(outputSize), requiresGrad: true);
        parameters.Add(Weight);
        parameters.Add(Bias);
    }

    /// <summary>
    /// Gets the expected input feature count.
    /// </summary>
    public int InputSize { get; }

    /// <summary>
    /// Gets the output feature count.
    /// </summary>
    public int OutputSize { get; }

    /// <summary>
    /// Gets the learnable weight matrix stored as <c>[OutputSize, InputSize]</c>.
    /// </summary>
    public Parameter<float> Weight { get; }

    /// <summary>
    /// Gets the learnable bias vector stored as <c>[OutputSize]</c>.
    /// </summary>
    public Parameter<float> Bias { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Applies the linear transform to a rank-1 vector or rank-2 batch tensor.
    /// </summary>
    /// <param name="input">The input tensor with last dimension equal to <see cref="InputSize" />.</param>
    /// <returns>A tensor whose last dimension is <see cref="OutputSize" />.</returns>
    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        var (batch, _) = TensorShapeValidator.GetVectorOrBatchLayout(input.Shape, InputSize, nameof(Linear));

        return NnDeviceOps.Linear(input, Weight.Value, Bias.Value, batch, InputSize, OutputSize);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Weight.Dispose();
        Bias.Dispose();
        disposed = true;
    }
}

/// <summary>
/// Maps integer token or category ids to learned float vectors.
/// </summary>
public sealed class Embedding : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes an embedding table.
    /// </summary>
    /// <param name="vocabularySize">The number of rows in the embedding table.</param>
    /// <param name="embeddingSize">The number of float values returned for each index.</param>
    public Embedding(int vocabularySize, int embeddingSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabularySize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingSize);
        VocabularySize = vocabularySize;
        EmbeddingSize = embeddingSize;
        Weight = FloatParameterFactory.Create("weight", new TensorShape(vocabularySize, embeddingSize), requiresGrad: true);
        parameters.Add(Weight);
    }

    /// <summary>
    /// Gets the number of addressable embedding rows.
    /// </summary>
    public int VocabularySize { get; }

    /// <summary>
    /// Gets the number of float values in each embedding vector.
    /// </summary>
    public int EmbeddingSize { get; }

    /// <summary>
    /// Gets the learnable embedding table stored as <c>[VocabularySize, EmbeddingSize]</c>.
    /// </summary>
    public Parameter<float> Weight { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Looks up embeddings for a span of integer indices.
    /// </summary>
    /// <param name="indices">The embedding row indices to gather.</param>
    /// <returns>A tensor with shape <c>[indices.Length, EmbeddingSize]</c>.</returns>
    public Tensor<float> Forward(ReadOnlySpan<int> indices)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (indices.IsEmpty)
        {
            throw new ArgumentException("Embedding lookup requires at least one index.", nameof(indices));
        }

        return ForwardCore(indices, new TensorShape(indices.Length, EmbeddingSize));
    }

    /// <summary>
    /// Looks up embeddings for every integer stored in an index tensor.
    /// </summary>
    /// <param name="indices">The tensor containing embedding row indices.</param>
    /// <returns>A tensor whose shape appends <see cref="EmbeddingSize" /> to the index tensor shape.</returns>
    public Tensor<float> Forward(Tensor<int> indices)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(indices);
        if (indices.ElementCount == 0)
        {
            throw new ArgumentException("Embedding lookup requires at least one index.", nameof(indices));
        }

        return NnDeviceOps.Embedding(indices, Weight.Value, TensorShapeValidator.AppendDimension(indices.Shape, EmbeddingSize), EmbeddingSize);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Weight.Dispose();
        disposed = true;
    }

    private Tensor<float> ForwardCore(ReadOnlySpan<int> indices, TensorShape outputShape)
    {
        for (var row = 0; row < indices.Length; row++)
        {
            var index = indices[row];
            if ((uint)index >= (uint)VocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(indices), "Embedding index is outside the vocabulary range.");
            }
        }

        using var indexTensor = new Tensor<int>(new TensorShape(indices.Length), Feather.GPU.CreateBuffer<int>(indices));
        return NnDeviceOps.Embedding(indexTensor, Weight.Value, outputShape, EmbeddingSize);
    }
}

/// <summary>
/// Normalizes each vector over its last dimension and applies learned scale and bias.
/// </summary>
public sealed class LayerNorm : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a layer normalization module.
    /// </summary>
    /// <param name="featureSize">The size of the final tensor dimension to normalize.</param>
    /// <param name="epsilon">The numerical stability term added to the variance.</param>
    public LayerNorm(int featureSize, float epsilon = 1e-5f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        FeatureSize = featureSize;
        Epsilon = epsilon;
        Gamma = FloatParameterFactory.Create("gamma", new TensorShape(featureSize), requiresGrad: true, initialValue: 1f);
        Beta = FloatParameterFactory.Create("beta", new TensorShape(featureSize), requiresGrad: true);
        parameters.Add(Gamma);
        parameters.Add(Beta);
    }

    /// <summary>
    /// Gets the normalized feature count.
    /// </summary>
    public int FeatureSize { get; }

    /// <summary>
    /// Gets the numerical stability term added to the variance.
    /// </summary>
    public float Epsilon { get; }

    /// <summary>
    /// Gets the learnable per-feature scale.
    /// </summary>
    public Parameter<float> Gamma { get; }

    /// <summary>
    /// Gets the learnable per-feature bias.
    /// </summary>
    public Parameter<float> Beta { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Normalizes a rank-1 vector or rank-2 batch tensor over the final dimension.
    /// </summary>
    /// <param name="input">The input tensor with last dimension equal to <see cref="FeatureSize" />.</param>
    /// <returns>A tensor with the same shape as <paramref name="input" />.</returns>
    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        var (rows, _) = TensorShapeValidator.GetVectorOrBatchLayout(input.Shape, FeatureSize, nameof(LayerNorm));
        _ = rows;
        return NnDeviceOps.LayerNorm(input, Gamma.Value, Beta.Value, FeatureSize, Epsilon);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Gamma.Dispose();
        Beta.Dispose();
        disposed = true;
    }
}

/// <summary>
/// Normalizes rank-1 or rank-2 tensors per feature using batch statistics.
/// </summary>
public sealed class BatchNorm1D : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a one-dimensional batch normalization module.
    /// </summary>
    /// <param name="featureSize">The size of the feature dimension.</param>
    /// <param name="epsilon">The numerical stability term added to the variance.</param>
    /// <param name="momentum">The running-statistics interpolation factor used during training.</param>
    public BatchNorm1D(int featureSize, float epsilon = 1e-5f, float momentum = 0.1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(featureSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        if (momentum is < 0f or > 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(momentum), "Momentum must be in the inclusive range [0, 1].");
        }

        FeatureSize = featureSize;
        Epsilon = epsilon;
        Momentum = momentum;
        Gamma = FloatParameterFactory.Create("gamma", new TensorShape(featureSize), requiresGrad: true, initialValue: 1f);
        Beta = FloatParameterFactory.Create("beta", new TensorShape(featureSize), requiresGrad: true);
        RunningMean = new Tensor<float>(new TensorShape(featureSize), Feather.GPU.CreateBuffer<float>(new float[featureSize]));

        var initialVariance = new float[featureSize];
        Array.Fill(initialVariance, 1f);
        RunningVariance = new Tensor<float>(new TensorShape(featureSize), Feather.GPU.CreateBuffer<float>(initialVariance));

        parameters.Add(Gamma);
        parameters.Add(Beta);
    }

    /// <summary>
    /// Gets the normalized feature count.
    /// </summary>
    public int FeatureSize { get; }

    /// <summary>
    /// Gets the numerical stability term added to the variance.
    /// </summary>
    public float Epsilon { get; }

    /// <summary>
    /// Gets the running-statistics interpolation factor used during training.
    /// </summary>
    public float Momentum { get; }

    /// <summary>
    /// Gets or sets a value indicating whether forward passes update and use batch statistics.
    /// </summary>
    public bool Training { get; set; } = true;

    /// <summary>
    /// Gets the learnable per-feature scale.
    /// </summary>
    public Parameter<float> Gamma { get; }

    /// <summary>
    /// Gets the learnable per-feature bias.
    /// </summary>
    public Parameter<float> Beta { get; }

    /// <summary>
    /// Gets the running per-feature mean used when <see cref="Training" /> is <see langword="false" />.
    /// </summary>
    public Tensor<float> RunningMean { get; }

    /// <summary>
    /// Gets the running per-feature variance used when <see cref="Training" /> is <see langword="false" />.
    /// </summary>
    public Tensor<float> RunningVariance { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Normalizes a rank-1 vector or rank-2 batch tensor over the feature dimension.
    /// </summary>
    /// <param name="input">The input tensor with last dimension equal to <see cref="FeatureSize" />.</param>
    /// <returns>A tensor with the same shape as <paramref name="input" />.</returns>
    public Tensor<float> Forward(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);

        var (batch, _) = TensorShapeValidator.GetVectorOrBatchLayout(input.Shape, FeatureSize, nameof(BatchNorm1D));
        return NnDeviceOps.BatchNorm(input, Gamma.Value, Beta.Value, RunningMean, RunningVariance, batch, FeatureSize, Epsilon, Momentum, Training);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Gamma.Dispose();
        Beta.Dispose();
        RunningMean.Dispose();
        RunningVariance.Dispose();
        disposed = true;
    }

}

/// <summary>
/// Runs a sequence of float-to-float modules.
/// </summary>
/// <param name="modules">The modules to apply in order.</param>
public sealed class Sequential(params Module[] modules) : Module
{
    private readonly IReadOnlyList<IParameter> parameters = QualifyAndCollectParameters(modules);

    /// <summary>
    /// Gets the modules applied by this sequence.
    /// </summary>
    public IReadOnlyList<Module> Modules { get; } = modules;

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Applies each supported module in order.
    /// </summary>
    /// <param name="input">The input tensor.</param>
    /// <returns>The final output tensor.</returns>
    public Tensor<float> Forward(Tensor<float> input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var current = input;
        var ownsCurrent = false;
        try
        {
            foreach (var module in Modules)
            {
                var next = module switch
                {
                    Linear linear => linear.Forward(current),
                    Activation activation => activation.Forward(current),
                    LayerNorm layerNorm => layerNorm.Forward(current),
                    BatchNorm1D batchNorm => batchNorm.Forward(current),
                    _ => throw new NotSupportedException($"Unsupported module type {module.GetType().FullName}.")
                };

                if (ownsCurrent)
                {
                    current.Dispose();
                }

                current = next;
                ownsCurrent = true;
            }
        }
        catch
        {
            if (ownsCurrent)
            {
                current.Dispose();
            }

            throw;
        }

        return current;
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var module in Modules)
        {
            module.Dispose();
        }
    }

    /// <inheritdoc />
    public override void QualifyParameters(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        for (var i = 0; i < Modules.Count; i++)
        {
            Modules[i].QualifyParameters($"{prefix}.{ModulePathSegment(Modules[i], i)}");
        }
    }

    private static IReadOnlyList<IParameter> QualifyAndCollectParameters(IReadOnlyList<Module> modules)
    {
        ArgumentNullException.ThrowIfNull(modules);
        for (var i = 0; i < modules.Count; i++)
        {
            ArgumentNullException.ThrowIfNull(modules[i]);
            modules[i].QualifyParameters(ModulePathSegment(modules[i], i));
        }

        return ParameterValidation.EnsureUnique(modules.SelectMany(module => module.Parameters), nameof(modules));
    }

    private static string ModulePathSegment(Module module, int index)
        => $"{module.GetType().Name.ToLowerInvariant()}{index}";
}

/// <summary>
/// Base type for optimizers that update module parameters from gradient tensors.
/// </summary>
public abstract class Optimizer : IDisposable
{
    /// <summary>
    /// Initializes an optimizer over a fixed set of parameters.
    /// </summary>
    /// <param name="parameters">The parameters to update.</param>
    protected Optimizer(IEnumerable<IParameter> parameters)
    {
        Parameters = ParameterValidation.EnsureUnique(parameters, nameof(parameters));
        foreach (var parameter in Parameters)
        {
            if (parameter is not Parameter<float>)
            {
                throw new NotSupportedException($"NN optimizers currently support Parameter<float> only. Parameter '{parameter.FullName}' has element type '{ParameterValidation.GetElementType(parameter).Name}'.");
            }
        }
    }

    /// <summary>
    /// Gets the parameters updated by this optimizer.
    /// </summary>
    public IReadOnlyList<IParameter> Parameters { get; }

    /// <summary>
    /// Applies one optimization step using the current gradient tensors.
    /// </summary>
    public abstract void Step();

    /// <summary>
    /// Clears all supported parameter gradient tensors.
    /// </summary>
    public abstract void ZeroGrad();

    /// <inheritdoc />
    public virtual void Dispose()
    {
    }

    /// <summary>
    /// Debug/interop path that materializes named managed AD gradients, uploads them into parameter
    /// gradient buffers, then performs one optimizer step. Prefer <see cref="Step{TKernel}(GpuADKernel{TKernel})" />
    /// for normal device-only training.
    /// </summary>
    /// <param name="gradients">The managed debug gradient set produced by an explicit readback path.</param>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public void Step(GradientSet gradients)
        => StepFromDebugGradients(gradients);

    /// <summary>
    /// Debug/interop path that materializes named managed AD gradients, uploads them into parameter
    /// gradient buffers, then performs one optimizer step.
    /// </summary>
    /// <param name="gradients">The managed debug gradient set produced by an explicit readback path.</param>
    public void StepFromDebugGradients(GradientSet gradients)
    {
        ArgumentNullException.ThrowIfNull(gradients);
        foreach (var parameter in FloatParameters())
        {
            var matches = parameter.GradientNames
                .Where(name => gradients.Contains(name))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            if (matches.Length == 0)
            {
                throw new ArgumentException($"No AD gradient matched parameter '{parameter.FullName}'. Accepted names: {string.Join(", ", parameter.GradientNames)}.", nameof(gradients));
            }

            if (matches.Length > 1)
            {
                throw new ArgumentException($"Multiple AD gradients matched parameter '{parameter.FullName}': {string.Join(", ", matches)}.", nameof(gradients));
            }

            var gradientName = matches[0];
            if (gradients.TryGetArray<float>(gradientName, out var values))
            {
                if (values.Length != parameter.Gradient.Shape.ElementCount)
                {
                    throw new ArgumentException($"Gradient '{gradientName}' has {values.Length} values but parameter '{parameter.FullName}' expects {parameter.Gradient.Shape.ElementCount}.", nameof(gradients));
                }

                // The AD bridge hands optimizers named gradients; optimizers own validation and the concrete update rule.
                parameter.Gradient.Buffer.Upload(values);
            }
            else if (gradients.TryGet<float>(gradientName, out var scalar))
            {
                if (parameter.Gradient.Shape.ElementCount != 1)
                {
                    throw new ArgumentException($"Scalar gradient '{gradientName}' cannot update non-scalar parameter '{parameter.FullName}'.", nameof(gradients));
                }

                parameter.Gradient.Buffer.Upload([scalar]);
            }
            else
            {
                throw new ArgumentException($"Gradient '{gradientName}' exists but is not a float gradient compatible with parameter '{parameter.FullName}'.", nameof(gradients));
            }
        }

        Step();
    }

    /// <summary>
    /// Copies native AD gradients directly into matching parameter gradient tensors on the device,
    /// then applies one optimizer step.
    /// </summary>
    internal void Step<TKernel>(GpuADKernel<TKernel> adKernel)
        where TKernel : struct, IKernel1D, Feather.Interop.IGeneratedKernel<TKernel>
    {
        ArgumentNullException.ThrowIfNull(adKernel);
        foreach (var parameter in FloatParameters())
        {
            var matches = adKernel.FindGradientMatches(parameter.GradientNames, parameter.Gradient.Buffer.Length);
            if (matches.Length == 0)
            {
                throw new ArgumentException($"No native AD gradient matched parameter '{parameter.FullName}'. Accepted names: {string.Join(", ", parameter.GradientNames)}.", nameof(adKernel));
            }

            if (matches.Length > 1)
            {
                throw new ArgumentException($"Multiple native AD gradients matched parameter '{parameter.FullName}': {string.Join(", ", matches.Select(match => match.Name))}.", nameof(adKernel));
            }

            adKernel.CopyGradientToBuffer(matches[0].Index, parameter.Gradient.Buffer);
        }

        Step();
    }

    protected IEnumerable<Parameter<float>> FloatParameters()
        => Parameters.OfType<Parameter<float>>();

    protected void ZeroFloatGradients()
    {
        foreach (var parameter in FloatParameters())
        {
            NnDeviceOps.Fill(parameter.Gradient, 0f);
        }
    }

    protected static Tensor<float> GetState(Dictionary<IParameter, Tensor<float>> state, Parameter<float> parameter)
        => GetState(state, parameter, 0f);

    protected static Tensor<float> GetState(Dictionary<IParameter, Tensor<float>> state, Parameter<float> parameter, float initialValue)
    {
        if (state.TryGetValue(parameter, out var tensor))
        {
            return tensor;
        }

        var values = Feather.GPU.CreateBuffer<float>(parameter.ElementCount);
        tensor = new Tensor<float>(parameter.Value.Shape, values);
        if (initialValue != 0f)
        {
            NnDeviceOps.Fill(tensor, initialValue);
        }

        state.Add(parameter, tensor);
        return tensor;
    }
}

/// <summary>
/// Orchestrates one explicit generated AD loss kernel and an optimizer over module-owned parameters.
/// This is the supported module-training contract today: parameters are owned by <see cref="Module" />
/// instances, losses are authored as generated AD kernels, and optimizer handoff uses device buffers.
/// </summary>
public sealed class TrainingStep<TKernel> : IDisposable
    where TKernel : struct, IKernel1D, Feather.Interop.IGeneratedKernel<TKernel>
{
    private readonly GpuADKernel<TKernel> adKernel;
    private readonly Optimizer optimizer;
    private readonly GpuBuffer<float>? lossBuffer;
    private readonly int count;
    private bool disposed;

    private TrainingStep(GpuADKernel<TKernel> adKernel, Optimizer optimizer, GpuBuffer<float>? lossBuffer, int count)
    {
        this.adKernel = adKernel;
        this.optimizer = optimizer;
        this.lossBuffer = lossBuffer;
        this.count = count;
    }

    /// <summary>
    /// Creates a training step for an already constructed generated AD kernel.
    /// </summary>
    public static TrainingStep<TKernel> Create(
        TKernel kernel,
        IEnumerable<IParameter> parameters,
        Optimizer optimizer,
        GpuBuffer<float>? lossBuffer,
        int count)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        var parameterList = parameters.ToArray();
        if (parameterList.Length == 0)
        {
            throw new ArgumentException("TrainingStep requires at least one module-owned parameter.", nameof(parameters));
        }

        var optimizerParameters = optimizer.Parameters.ToArray();
        if (optimizerParameters.Length != parameterList.Length ||
            optimizerParameters.Where((parameter, index) => !ReferenceEquals(parameter, parameterList[index])).Any())
        {
            throw new ArgumentException("Optimizer parameters must be the same module-owned parameter sequence supplied to the training step.", nameof(optimizer));
        }

        if (lossBuffer is not null && lossBuffer.Length < count)
        {
            throw new ArgumentException("Loss buffer length must be at least the training step count.", nameof(lossBuffer));
        }

        return new TrainingStep<TKernel>(GPU.CreateADKernel(kernel), optimizer, lossBuffer, count);
    }

    /// <summary>
    /// Runs backward, copies native AD gradients to module parameter gradient buffers on device,
    /// applies the optimizer step, and returns the optional scalar loss readback.
    /// </summary>
    public float Run()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        adKernel.Backward(count);
        var loss = 0f;
        if (lossBuffer is not null)
        {
            var values = lossBuffer.ToArray();
            for (var i = 0; i < count; i++)
            {
                loss += values[i];
            }
        }

        optimizer.Step(adKernel);
        LastDispatchPath = adKernel.LastDispatchPath;
        GradientsMaterialized = adKernel.Gradients.HasMaterializedValues;
        LastLoss = loss;
        return LastLoss;
    }

    /// <summary>
    /// Gets the native route used by the most recent generated training dispatch.
    /// </summary>
    public DispatchPath LastDispatchPath { get; private set; } = DispatchPath.None;

    /// <summary>
    /// Gets a value indicating whether managed debug gradient values have been materialized.
    /// </summary>
    public bool GradientsMaterialized { get; private set; }

    /// <summary>
    /// Gets the loss returned by the most recent training call.
    /// </summary>
    public float LastLoss { get; private set; } = float.NaN;

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        adKernel.Dispose();
        disposed = true;
    }
}

/// <summary>
/// Stochastic gradient descent optimizer.
/// </summary>
/// <param name="parameters">The parameters to update.</param>
/// <param name="learningRate">The learning rate applied to every gradient.</param>
public sealed class SGD(IEnumerable<IParameter> parameters, float learningRate = 0.01f, float momentum = 0f, float weightDecay = 0f) : Optimizer(parameters)
{
    private readonly Dictionary<IParameter, Tensor<float>> momentumState = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Gets the learning rate applied to every gradient.
    /// </summary>
    public float LearningRate { get; } = learningRate;

    /// <summary>
    /// Gets the momentum factor. A value of zero runs vanilla SGD.
    /// </summary>
    public float Momentum { get; } = momentum;

    /// <summary>
    /// Gets coupled L2 weight decay applied to gradients.
    /// </summary>
    public float WeightDecay { get; } = weightDecay;

    /// <inheritdoc />
    public override void Step()
    {
        foreach (var parameter in FloatParameters())
        {
            if (Momentum == 0f)
            {
                NnDeviceOps.Sgd(parameter, LearningRate, WeightDecay);
                continue;
            }

            var momentumTensor = GetState(momentumState, parameter);
            NnDeviceOps.Momentum(parameter, momentumTensor, LearningRate, Momentum, WeightDecay);
        }
    }

    /// <inheritdoc />
    public override void ZeroGrad() => ZeroFloatGradients();

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var tensor in momentumState.Values)
        {
            tensor.Dispose();
        }

        momentumState.Clear();
    }
}

/// <summary>
/// Adam optimizer with first- and second-moment state.
/// </summary>
public sealed class Adam : Optimizer
{
    private readonly Dictionary<IParameter, Tensor<float>> firstMoments = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<IParameter, Tensor<float>> secondMoments = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<IParameter, (float LearningRate, float WeightDecay)> parameterOptions;
    private int step;

    /// <summary>
    /// Initializes an Adam optimizer.
    /// </summary>
    /// <param name="parameters">The parameters to update.</param>
    /// <param name="learningRate">The optimizer learning rate.</param>
    /// <param name="beta1">The first-moment decay coefficient.</param>
    /// <param name="beta2">The second-moment decay coefficient.</param>
    /// <param name="epsilon">The numerical stability term used in the denominator.</param>
    public Adam(IEnumerable<IParameter> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0f, float gradientClip = 0f)
        : base(parameters)
    {
        ValidateAdamOptions(learningRate, beta1, beta2, epsilon, weightDecay, gradientClip);
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        GradientClip = gradientClip;
        parameterOptions = new Dictionary<IParameter, (float LearningRate, float WeightDecay)>(ReferenceEqualityComparer.Instance);
        foreach (var parameter in Parameters)
        {
            parameterOptions.Add(parameter, (learningRate, weightDecay));
        }
    }

    /// <summary>
    /// Initializes an Adam optimizer from parameter groups. Group-specific learning rates and
    /// weight decay override the optimizer defaults for parameters inside that group.
    /// </summary>
    public Adam(IEnumerable<ParameterGroup> parameterGroups, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0f, float gradientClip = 0f)
        : this(SnapshotParameterGroups(parameterGroups), learningRate, beta1, beta2, epsilon, weightDecay, gradientClip)
    {
    }

    private Adam(IReadOnlyList<ParameterGroup> parameterGroups, float learningRate, float beta1, float beta2, float epsilon, float weightDecay, float gradientClip)
        : base(parameterGroups.SelectMany(group => group.Parameters))
    {
        ValidateAdamOptions(learningRate, beta1, beta2, epsilon, weightDecay, gradientClip);
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
        GradientClip = gradientClip;
        parameterOptions = BuildParameterOptions(parameterGroups, learningRate, weightDecay);
    }

    private static void ValidateAdamOptions(float learningRate, float beta1, float beta2, float epsilon, float weightDecay, float gradientClip)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta1);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(beta1, 1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta2);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(beta2, 1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        ArgumentOutOfRangeException.ThrowIfNegative(weightDecay);
        ArgumentOutOfRangeException.ThrowIfNegative(gradientClip);
    }

    /// <summary>
    /// Gets the optimizer learning rate.
    /// </summary>
    public float LearningRate { get; }

    /// <summary>
    /// Gets the first-moment decay coefficient.
    /// </summary>
    public float Beta1 { get; }

    /// <summary>
    /// Gets the second-moment decay coefficient.
    /// </summary>
    public float Beta2 { get; }

    /// <summary>
    /// Gets the numerical stability term used in the Adam denominator.
    /// </summary>
    public float Epsilon { get; }

    /// <summary>
    /// Gets coupled L2 weight decay applied to gradients.
    /// </summary>
    public float WeightDecay { get; }

    /// <summary>
    /// Gets the optional elementwise gradient clipping threshold. A value of zero disables clipping.
    /// </summary>
    public float GradientClip { get; }

    /// <summary>
    /// Gets the number of optimization steps completed.
    /// </summary>
    public int StepCount => step;

    /// <inheritdoc />
    public override void Step()
    {
        step++;

        foreach (var parameter in FloatParameters())
        {
            var options = parameterOptions[parameter];
            NnDeviceOps.Adam(
                parameter,
                GetState(firstMoments, parameter),
                GetState(secondMoments, parameter),
                options.LearningRate,
                Beta1,
                Beta2,
                Epsilon,
                step,
                options.WeightDecay,
                GradientClip,
                decoupledWeightDecay: false);
        }
    }

    /// <inheritdoc />
    public override void ZeroGrad() => ZeroFloatGradients();

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var tensor in firstMoments.Values.Concat(secondMoments.Values))
        {
            tensor.Dispose();
        }

        firstMoments.Clear();
        secondMoments.Clear();
    }

    private static IReadOnlyList<ParameterGroup> SnapshotParameterGroups(IEnumerable<ParameterGroup> parameterGroups)
    {
        ArgumentNullException.ThrowIfNull(parameterGroups);
        var groups = parameterGroups.ToArray();
        if (groups.Length == 0)
        {
            throw new ArgumentException("Adam requires at least one parameter group.", nameof(parameterGroups));
        }

        foreach (var group in groups)
        {
            ArgumentNullException.ThrowIfNull(group);
        }

        return groups;
    }

    private static Dictionary<IParameter, (float LearningRate, float WeightDecay)> BuildParameterOptions(
        IEnumerable<ParameterGroup> parameterGroups,
        float defaultLearningRate,
        float defaultWeightDecay)
    {
        var options = new Dictionary<IParameter, (float LearningRate, float WeightDecay)>(ReferenceEqualityComparer.Instance);
        foreach (var group in parameterGroups)
        {
            var groupLearningRate = group.LearningRate ?? defaultLearningRate;
            var groupWeightDecay = group.WeightDecay ?? defaultWeightDecay;
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(groupLearningRate);
            ArgumentOutOfRangeException.ThrowIfNegative(groupWeightDecay);
            foreach (var parameter in group.Parameters)
            {
                if (!options.TryAdd(parameter, (groupLearningRate, groupWeightDecay)))
                {
                    throw new ArgumentException($"Parameter '{parameter.FullName}' appears in more than one Adam parameter group.", nameof(parameterGroups));
                }
            }
        }

        return options;
    }
}

/// <summary>
/// AdamW optimizer with decoupled weight decay.
/// </summary>
public sealed class AdamW : Optimizer
{
    private readonly Dictionary<IParameter, Tensor<float>> firstMoments = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<IParameter, Tensor<float>> secondMoments = new(ReferenceEqualityComparer.Instance);
    private int step;

    public AdamW(IEnumerable<IParameter> parameters, float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weightDecay = 0.01f)
        : base(parameters)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta1);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(beta1, 1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(beta2);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(beta2, 1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        ArgumentOutOfRangeException.ThrowIfNegative(weightDecay);
        LearningRate = learningRate;
        Beta1 = beta1;
        Beta2 = beta2;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
    }

    public float LearningRate { get; }

    public float Beta1 { get; }

    public float Beta2 { get; }

    public float Epsilon { get; }

    public float WeightDecay { get; }

    /// <inheritdoc />
    public override void Step()
    {
        step++;
        foreach (var parameter in FloatParameters())
        {
            NnDeviceOps.Adam(
                parameter,
                GetState(firstMoments, parameter),
                GetState(secondMoments, parameter),
                LearningRate,
                Beta1,
                Beta2,
                Epsilon,
                step,
                WeightDecay,
                gradientClip: 0f,
                decoupledWeightDecay: true);
        }
    }

    /// <inheritdoc />
    public override void ZeroGrad() => ZeroFloatGradients();

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var tensor in firstMoments.Values.Concat(secondMoments.Values))
        {
            tensor.Dispose();
        }

        firstMoments.Clear();
        secondMoments.Clear();
    }
}

/// <summary>
/// RMSProp optimizer with a decaying squared-gradient average.
/// </summary>
public sealed class RMSProp : Optimizer
{
    private readonly Dictionary<IParameter, Tensor<float>> squareAverages = new(ReferenceEqualityComparer.Instance);

    /// <summary>
    /// Initializes an RMSProp optimizer.
    /// </summary>
    /// <param name="parameters">The parameters to update.</param>
    /// <param name="learningRate">The optimizer learning rate.</param>
    /// <param name="alpha">The decay coefficient for the squared-gradient moving average.</param>
    /// <param name="epsilon">The numerical stability term used in the denominator.</param>
    public RMSProp(IEnumerable<IParameter> parameters, float learningRate = 0.001f, float alpha = 0.99f, float epsilon = 1e-8f, float weightDecay = 0f)
        : base(parameters)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(alpha);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(alpha, 1);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epsilon);
        LearningRate = learningRate;
        Alpha = alpha;
        Epsilon = epsilon;
        WeightDecay = weightDecay;
    }

    /// <summary>
    /// Gets the optimizer learning rate.
    /// </summary>
    public float LearningRate { get; }

    /// <summary>
    /// Gets the decay coefficient for the squared-gradient moving average.
    /// </summary>
    public float Alpha { get; }

    /// <summary>
    /// Gets the numerical stability term used in the RMSProp denominator.
    /// </summary>
    public float Epsilon { get; }

    /// <summary>
    /// Gets coupled L2 weight decay applied to gradients.
    /// </summary>
    public float WeightDecay { get; }

    /// <inheritdoc />
    public override void Step()
    {
        foreach (var parameter in FloatParameters())
        {
            NnDeviceOps.RmsProp(parameter, GetState(squareAverages, parameter), LearningRate, Alpha, Epsilon, WeightDecay);
        }
    }

    /// <inheritdoc />
    public override void ZeroGrad() => ZeroFloatGradients();

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var tensor in squareAverages.Values)
        {
            tensor.Dispose();
        }

        squareAverages.Clear();
    }
}

internal static class FloatParameterFactory
{
    internal static Parameter<float> Create(string name, TensorShape shape, bool requiresGrad, float initialValue = 0f)
    {
        var initialValues = new float[shape.ElementCount];
        if (initialValue != 0f)
        {
            Array.Fill(initialValues, initialValue);
        }

        // Parameters keep value and gradient storage paired so optimizers can operate over one stable handle.
        var values = Feather.GPU.CreateBuffer<float>(initialValues);
        var gradients = Feather.GPU.CreateBuffer<float>(shape.ElementCount);
        return new Parameter<float>(
            name,
            new Tensor<float>(shape, values, requiresGrad),
            new Tensor<float>(shape, gradients));
    }
}

internal static class TensorShapeValidator
{
    internal static (int Rows, int FeatureCount) GetVectorOrBatchLayout(TensorShape shape, int expectedFeatures, string moduleName)
    {
        if (shape.Rank is not 1 and not 2)
        {
            throw new ArgumentException($"{moduleName} input must be a rank-1 vector or rank-2 batch.", nameof(shape));
        }

        var rows = shape.Rank == 1 ? 1 : shape[0];
        var features = shape.Rank == 1 ? shape[0] : shape[1];
        if (rows <= 0 || features <= 0)
        {
            throw new ArgumentException($"{moduleName} input dimensions must be positive.", nameof(shape));
        }

        if (features != expectedFeatures)
        {
            throw new ArgumentException($"{moduleName} expected the last input dimension to be {expectedFeatures}, but received {features}.", nameof(shape));
        }

        // Rank-1 tensors are treated as one row so modules can share one row-major implementation.
        return (rows, features);
    }

    internal static TensorShape AppendDimension(TensorShape shape, int dimension)
    {
        if (shape.Rank == 0)
        {
            return new TensorShape(dimension);
        }

        var dimensions = new int[shape.Rank + 1];
        shape.AsSpan().CopyTo(dimensions);
        dimensions[^1] = dimension;
        return new TensorShape(dimensions);
    }
}

internal static class ParameterValidation
{
    public static IReadOnlyList<IParameter> EnsureUnique(IEnumerable<IParameter> parameters, string paramName)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        var result = parameters.ToArray();
        var names = new Dictionary<string, IParameter>(StringComparer.Ordinal);
        foreach (var parameter in result)
        {
            ArgumentNullException.ThrowIfNull(parameter);
            if (!names.TryAdd(parameter.FullName, parameter))
            {
                throw new ArgumentException($"Duplicate parameter name '{parameter.FullName}'. Compose modules through Sequential or qualify parameters before optimizing/checkpointing.", paramName);
            }
        }

        return result;
    }

    public static Type GetElementType(IParameter parameter)
    {
        var type = parameter.GetType();
        while (type is not null)
        {
            if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(Parameter<>))
            {
                return type.GetGenericArguments()[0];
            }

            type = type.BaseType;
        }

        return typeof(void);
    }
}
