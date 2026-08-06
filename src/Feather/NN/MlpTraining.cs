using Feather.AD;
using Feather.Math;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.NN;

/// <summary>
/// The packed weight layout shared by the MLP training kernel and by shader-side inference.
/// </summary>
/// <remarks>
/// A 3→h→h→1 network lives in one flat <c>float</c> buffer rather than one buffer per layer. Two
/// reasons: a single buffer costs one <see cref="AD.Parameter(float)"/> marker for the whole network
/// (the native bridge registers a whole bound buffer per marker), and it keeps an inference kernel's
/// binding count at one regardless of depth. <see cref="SelfAttention"/> already packs its four
/// projections the same way.
///
/// Every offset is a pure function of <c>hiddenSize</c>, so host code sizing a buffer and shader code
/// indexing it derive the same numbers from the same place instead of agreeing by convention.
///
/// Layout, in order: <c>w1[h,3]</c>, <c>b1[h]</c>, <c>w2[h,h]</c>, <c>b2[h]</c>, <c>w3[1,h]</c>,
/// <c>b3[1]</c>. Weight matrices are row-major per layer: <c>w[outIndex * inputSize + inIndex]</c>.
/// </remarks>
public static class MlpLayout
{
    /// <summary>The number of inputs the 3→h→h→1 network takes.</summary>
    public const int InputSize = 3;

    /// <summary>Flat element count for a packed 3→h→h→1 network, for buffer sizing.</summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    public static int PackedElementCount3To1(int hiddenSize)
    {
        ValidateHiddenSize(hiddenSize);
        return (hiddenSize * hiddenSize) + (6 * hiddenSize) + 1;
    }

    /// <summary>Scratch elements one lane needs to evaluate a 3→h→h→1 network.</summary>
    /// <remarks>
    /// Sized for the training kernel, which needs a pre-activation and a post-activation per hidden
    /// unit across two layers. The pre-activations are kept because the adjoint pass reads them back;
    /// they are not dead stores.
    ///
    /// Shader-side inference needs less — three staged inputs plus two activation vectors, so
    /// <c>3 + 2 * hiddenSize</c> — and that fits inside this stride for any hidden size of 2 or more.
    /// One stride for both sides means one scratch buffer can serve training and inference, which is
    /// why <see cref="MinimumHiddenSize" /> is 2 rather than 1.
    /// </remarks>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    public static int ScratchElementsPerLane3To1(int hiddenSize)
    {
        ValidateHiddenSize(hiddenSize);
        return 4 * hiddenSize;
    }

    /// <summary>The narrowest hidden layer the shared scratch stride supports.</summary>
    public const int MinimumHiddenSize = 2;

    private static void ValidateHiddenSize(int hiddenSize)
    {
        if (hiddenSize < MinimumHiddenSize)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, $"A packed 3→h→h→1 network needs a hidden size of at least {MinimumHiddenSize}.");
        }
    }

    /// <summary>Offset of the first hidden layer's weight matrix, shaped <c>[hiddenSize, 3]</c>.</summary>
    public static int Layer1WeightOffset(int hiddenSize) => 0;

    /// <summary>Offset of the first hidden layer's bias vector, shaped <c>[hiddenSize]</c>.</summary>
    public static int Layer1BiasOffset(int hiddenSize) => InputSize * hiddenSize;

    /// <summary>Offset of the second hidden layer's weight matrix, shaped <c>[hiddenSize, hiddenSize]</c>.</summary>
    public static int Layer2WeightOffset(int hiddenSize) => 4 * hiddenSize;

    /// <summary>Offset of the second hidden layer's bias vector, shaped <c>[hiddenSize]</c>.</summary>
    public static int Layer2BiasOffset(int hiddenSize) => (4 * hiddenSize) + (hiddenSize * hiddenSize);

    /// <summary>Offset of the output layer's weight row, shaped <c>[1, hiddenSize]</c>.</summary>
    public static int OutputWeightOffset(int hiddenSize) => (5 * hiddenSize) + (hiddenSize * hiddenSize);

    /// <summary>Offset of the output layer's single bias value.</summary>
    public static int OutputBiasOffset(int hiddenSize) => (6 * hiddenSize) + (hiddenSize * hiddenSize);

    /// <summary>
    /// Evaluates the packed network on the host, matching the shader and kernel arithmetic.
    /// </summary>
    /// <remarks>
    /// Present so a test or a sample can state an expected value without a GPU dispatch, and so the
    /// inference smoke test has something to compare a shader result against.
    /// </remarks>
    /// <param name="weights">The packed weights, at least <see cref="PackedElementCount3To1" /> long.</param>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    /// <param name="x">The three input values.</param>
    public static float Evaluate3To1(ReadOnlySpan<float> weights, int hiddenSize, ReadOnlySpan<float> x)
    {
        ValidateHiddenSize(hiddenSize);
        if (x.Length != InputSize)
        {
            throw new ArgumentException($"A 3→h→h→1 network takes {InputSize} inputs but {x.Length} were supplied.", nameof(x));
        }

        var required = PackedElementCount3To1(hiddenSize);
        if (weights.Length < required)
        {
            throw new ArgumentException($"A hidden size of {hiddenSize} needs {required} packed weights but {weights.Length} were supplied.", nameof(weights));
        }

        var w1 = Layer1WeightOffset(hiddenSize);
        var b1 = Layer1BiasOffset(hiddenSize);
        var w2 = Layer2WeightOffset(hiddenSize);
        var b2 = Layer2BiasOffset(hiddenSize);
        var w3 = OutputWeightOffset(hiddenSize);
        var b3 = OutputBiasOffset(hiddenSize);

        var hidden1 = new float[hiddenSize];
        for (var j = 0; j < hiddenSize; j++)
        {
            var value = weights[b1 + j];
            for (var i = 0; i < InputSize; i++)
            {
                value += weights[w1 + (j * InputSize) + i] * x[i];
            }

            hidden1[j] = MathF.Max(value, 0f);
        }

        var hidden2 = new float[hiddenSize];
        for (var j = 0; j < hiddenSize; j++)
        {
            var value = weights[b2 + j];
            for (var k = 0; k < hiddenSize; k++)
            {
                value += weights[w2 + (j * hiddenSize) + k] * hidden1[k];
            }

            hidden2[j] = MathF.Max(value, 0f);
        }

        var prediction = weights[b3];
        for (var j = 0; j < hiddenSize; j++)
        {
            prediction += weights[w3 + j] * hidden2[j];
        }

        return prediction;
    }
}

/// <summary>
/// Squared-error training kernel for a packed 3→h→h→1 ReLU regression MLP.
/// </summary>
/// <remarks>
/// One lane per sample. Each lane reads its three inputs, runs the forward pass through
/// <see cref="MlpLayout" />'s packed layout using its own <paramref name="scratch" /> range, writes
/// a per-sample squared error into <paramref name="loss" />, and marks the whole weight buffer as one
/// AD parameter group.
///
/// Everything here is inside the AD subset by construction: a 1D kernel, one scalar loss, counted
/// <c>for</c> loops only, no <c>while</c>/<c>break</c>, and every weight read at an index built from
/// loop counters and uniforms rather than from data. That last point is the one worth restating —
/// The XIR AD path does not differentiate a parameter read at a data-dependent index, so the layout is
/// arithmetic on <c>hiddenSize</c> and never a value loaded from a buffer.
///
/// The arithmetic is duplicated in <see cref="MlpLayout.Evaluate3To1" /> for the host and in
/// <c>MlpShader.Evaluate3To1</c> for inference. It is not shared with either, because Feather.dll
/// does not compile the injected shader library, so a kernel inside the SDK cannot call into it.
/// </remarks>
[Kernel]
[AutoDiff]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MlpRegression3To1LossKernel(
    ReadOnlyBuffer<float> inputs,
    ReadOnlyBuffer<float> targets,
    ReadWriteBuffer<float> weights,
    ReadWriteBuffer<float> scratch,
    ReadWriteBuffer<float> loss,
    Uniform<int> hiddenSize,
    Uniform<float> lossScale) : IKernel1D
{
    /// <summary>Runs one sample's forward pass and records its squared error as the loss.</summary>
    public void Execute()
    {
        int sample = ThreadIds.X;
        int layer1Weight = 0;
        int layer1Bias = 3 * hiddenSize.Value;
        int layer2Weight = 4 * hiddenSize.Value;
        int layer2Bias = (4 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);
        int outputWeight = (5 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);
        int outputBias = (6 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);

        int laneBase = sample * (4 * hiddenSize.Value);
        int preActivation1 = laneBase;
        int activation1 = laneBase + hiddenSize.Value;
        int preActivation2 = laneBase + (2 * hiddenSize.Value);
        int activation2 = laneBase + (3 * hiddenSize.Value);

        int inputBase = sample * 3;
        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            float value = weights[layer1Bias + j];
            for (int i = 0; i < 3; i = i + 1)
            {
                value = value + (weights[layer1Weight + (j * 3) + i] * inputs[inputBase + i]);
            }

            scratch[preActivation1 + j] = value;
            scratch[activation1 + j] = ShaderMath.Max(scratch[preActivation1 + j], 0f);
        }

        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            float value = weights[layer2Bias + j];
            for (int k = 0; k < hiddenSize.Value; k = k + 1)
            {
                value = value + (weights[layer2Weight + (j * hiddenSize.Value) + k] * scratch[activation1 + k]);
            }

            scratch[preActivation2 + j] = value;
            scratch[activation2 + j] = ShaderMath.Max(scratch[preActivation2 + j], 0f);
        }

        float prediction = weights[outputBias];
        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            prediction = prediction + (weights[outputWeight + j] * scratch[activation2 + j]);
        }

        float error = prediction - targets[sample];
        float l = error * error * lossScale.Value;
        loss[sample] = l;

        ADMarker.Parameter(weights[0]);
        ADMarker.Loss(l);
    }
}

/// <summary>
/// Evaluates a packed 3→h→h→1 ReLU network on the GPU, one lane per input.
/// </summary>
/// <remarks>
/// The inference counterpart to <see cref="MlpRegression3To1LossKernel" />: same layout, same arithmetic,
/// no loss and no AD markers. Bind trained weights straight from
/// <see cref="InferenceWeights.Buffer" /> and dispatch one thread per input.
///
/// This is a kernel rather than a <c>[Callable]</c> helper in <c>MlpShader</c> because a callable cannot
/// take a buffer parameter — <c>GPU::IR::Type</c> has no buffer kind, so the native lowerer rejects one.
/// A pass that wants inference inline in its own shader instead of through a dispatch writes the loop
/// itself using <c>MlpShader</c>'s offset helpers; see that class's remarks.
///
/// <paramref name="scratch" /> must hold
/// <see cref="MlpLayout.ScratchElementsPerLane3To1" /> elements per lane. Lanes that share a range
/// corrupt each other's activations.
/// </remarks>
[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MlpInference3To1Kernel(
    ReadOnlyBuffer<float> inputs,
    ReadOnlyBuffer<float> weights,
    ReadWriteBuffer<float> scratch,
    ReadWriteBuffer<float> predictions,
    Uniform<int> hiddenSize) : IKernel1D
{
    /// <summary>Evaluates one input triple and writes its prediction.</summary>
    public void Execute()
    {
        int lane = ThreadIds.X;
        int layer1Weight = 0;
        int layer1Bias = 3 * hiddenSize.Value;
        int layer2Weight = 4 * hiddenSize.Value;
        int layer2Bias = (4 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);
        int outputWeight = (5 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);
        int outputBias = (6 * hiddenSize.Value) + (hiddenSize.Value * hiddenSize.Value);

        int laneBase = lane * (4 * hiddenSize.Value);
        int activation1 = laneBase;
        int activation2 = laneBase + hiddenSize.Value;

        int inputBase = lane * 3;
        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            float value = weights[layer1Bias + j];
            for (int i = 0; i < 3; i = i + 1)
            {
                value = value + (weights[layer1Weight + (j * 3) + i] * inputs[inputBase + i]);
            }

            scratch[activation1 + j] = ShaderMath.Max(value, 0f);
        }

        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            float value = weights[layer2Bias + j];
            for (int k = 0; k < hiddenSize.Value; k = k + 1)
            {
                value = value + (weights[layer2Weight + (j * hiddenSize.Value) + k] * scratch[activation1 + k]);
            }

            scratch[activation2 + j] = ShaderMath.Max(value, 0f);
        }

        float prediction = weights[outputBias];
        for (int j = 0; j < hiddenSize.Value; j = j + 1)
        {
            prediction = prediction + (weights[outputWeight + j] * scratch[activation2 + j]);
        }

        predictions[lane] = prediction;
    }
}

/// <summary>
/// A host-drivable training job for a packed 3→h→h→1 ReLU regression MLP.
/// </summary>
/// <remarks>
/// The common case reduced to a few lines: construct with a dataset, let a host call
/// <see cref="Initialize" /> then <see cref="Step" /> in a loop, and checkpoint
/// <see cref="Parameters" /> whenever it likes. All device state is allocated in
/// <see cref="Initialize" /> and held until <see cref="Dispose" />, so the kernel is built once and the
/// per-step cost is one dispatch plus one optimizer step.
///
/// The full dataset is one batch. Batching across steps waits on a dataset abstraction; with a fixed
/// dataset uploaded once, every step sees every sample, which is correct if not scalable.
///
/// Loss is read back only when <see cref="ReadLossEveryStep" /> is set or a host asks for it via
/// <see cref="StepAndReadLoss" />. That is the point of splitting the readback out of the step: a host
/// reporting every 25 steps should not pay a blocking readback 25 times.
/// </remarks>
public sealed class MlpRegressionJob : ITrainingJob
{
    private readonly int hiddenSize;
    private readonly float[] inputValues;
    private readonly float[] targetValues;
    private readonly float learningRate;
    private readonly int seed;

    private Parameter<float>? weights;
    private GpuBuffer<float>? inputs;
    private GpuBuffer<float>? targets;
    private GpuBuffer<float>? scratch;
    private GpuBuffer<float>? loss;
    private Optimizer? optimizer;
    private TrainingStep<MlpRegression3To1LossKernel>? trainingStep;
    private IParameter[] parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a regression job over a fixed dataset.
    /// </summary>
    /// <param name="hiddenSize">The width of both hidden layers.</param>
    /// <param name="inputs">Three float values per sample, laid out sample-major.</param>
    /// <param name="targets">One float target per sample.</param>
    /// <param name="learningRate">The Adam learning rate.</param>
    /// <param name="seed">The seed for Xavier weight initialization.</param>
    public MlpRegressionJob(
        int hiddenSize,
        ReadOnlySpan<float> inputs,
        ReadOnlySpan<float> targets,
        float learningRate = 0.01f,
        int seed = 1234)
    {
        if (hiddenSize < MlpLayout.MinimumHiddenSize)
        {
            throw new ArgumentOutOfRangeException(nameof(hiddenSize), hiddenSize, $"A packed 3→h→h→1 network needs a hidden size of at least {MlpLayout.MinimumHiddenSize}.");
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(targets.Length);
        if (inputs.Length != targets.Length * MlpLayout.InputSize)
        {
            throw new ArgumentException($"{targets.Length} targets need {targets.Length * MlpLayout.InputSize} input values but {inputs.Length} were supplied.", nameof(inputs));
        }

        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(learningRate);
        this.hiddenSize = hiddenSize;
        inputValues = inputs.ToArray();
        targetValues = targets.ToArray();
        this.learningRate = learningRate;
        this.seed = seed;
    }

    /// <summary>Gets the number of samples in the dataset.</summary>
    public int SampleCount => targetValues.Length;

    /// <summary>Gets the width of both hidden layers.</summary>
    public int HiddenSize => hiddenSize;

    /// <inheritdoc />
    public int PlannedSteps { get; init; }

    /// <summary>
    /// Gets or sets a value indicating whether every step reads its loss back.
    /// </summary>
    /// <remarks>
    /// False by default, which leaves <see cref="TrainingStepReport.Loss" /> as NaN on steps a host did
    /// not ask about. Set it when a caller wants a loss on every step and accepts the per-step stall —
    /// a short console sample, typically.
    /// </remarks>
    public bool ReadLossEveryStep { get; set; }

    /// <inheritdoc />
    public IReadOnlyList<IParameter> Parameters => parameters;

    /// <summary>Gets the packed weight parameter, once <see cref="Initialize" /> has run.</summary>
    public Parameter<float> Weights
        => weights ?? throw new InvalidOperationException("Initialize must run before the weight parameter exists.");

    /// <summary>The name the packed weight tensor is checkpointed under.</summary>
    public const string WeightParameterName = "mlp.weights";

    /// <inheritdoc />
    public void Initialize(TrainingContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        ObjectDisposedException.ThrowIf(disposed, this);
        if (trainingStep is not null)
        {
            throw new InvalidOperationException("This training job has already been initialized.");
        }

        var packedCount = MlpLayout.PackedElementCount3To1(hiddenSize);
        var shape = new TensorShape(packedCount);
        var initialValues = ParameterInitializers.XavierUniform(packedCount, MlpLayout.InputSize, 1, seed);
        weights = new Parameter<float>(
            WeightParameterName,
            new Tensor<float>(shape, GPU.CreateBuffer<float>(initialValues), requiresGrad: true),
            new Tensor<float>(shape, GPU.CreateBuffer<float>(packedCount)));

        // One marker covers the whole bound buffer, so the native gradient carries the resource name.
        weights.AddGradientAlias("weights");
        parameters = [weights];

        inputs = GPU.CreateBuffer<float>(inputValues);
        targets = GPU.CreateBuffer<float>(targetValues);
        scratch = GPU.CreateBuffer<float>(MlpLayout.ScratchElementsPerLane3To1(hiddenSize) * SampleCount);
        loss = GPU.CreateBuffer<float>(SampleCount);
        optimizer = new Adam(parameters, learningRate: learningRate);
        trainingStep = TrainingStep<MlpRegression3To1LossKernel>.Create(
            new MlpRegression3To1LossKernel(
                inputs.AsReadOnly(),
                targets.AsReadOnly(),
                weights.Value.AsReadWriteBuffer(),
                scratch.AsReadWrite(),
                loss.AsReadWrite(),
                new Uniform<int>(hiddenSize),
                new Uniform<float>(1f / SampleCount)),
            parameters,
            optimizer,
            loss,
            SampleCount);
    }

    /// <inheritdoc />
    public TrainingStepReport Step(TrainingContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var step = EnsureInitialized();
        step.RunWithoutLossReadback();
        return ReadLossEveryStep
            ? new TrainingStepReport(context.Step, step.ReadLoss(), step.LastDispatchPath)
            : TrainingStepReport.Unreported(context.Step, step.LastDispatchPath);
    }

    /// <summary>
    /// Runs one step and reads its loss back, regardless of <see cref="ReadLossEveryStep" />.
    /// </summary>
    /// <remarks>
    /// What a host calls on its reporting steps, so the readback happens exactly where a value is
    /// wanted rather than being a property of the whole run.
    /// </remarks>
    /// <param name="context">The host-supplied context, carrying the step index about to run.</param>
    public TrainingStepReport StepAndReadLoss(TrainingContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        var step = EnsureInitialized();
        step.RunWithoutLossReadback();
        return new TrainingStepReport(context.Step, step.ReadLoss(), step.LastDispatchPath);
    }

    /// <summary>Reads the current loss without running a step.</summary>
    public float ReadLoss() => EnsureInitialized().ReadLoss();

    /// <summary>Evaluates the trained network on the host for a single input triple.</summary>
    /// <remarks>
    /// Reads the weights back, so it is a diagnostic rather than something to call per sample in a loop.
    /// </remarks>
    /// <param name="x">The three input values.</param>
    /// <param name="y">The second input value.</param>
    /// <param name="z">The third input value.</param>
    public float Predict(float x, float y, float z)
    {
        EnsureInitialized();
        return MlpLayout.Evaluate3To1(Weights.Value.Buffer.ToArray(), hiddenSize, [x, y, z]);
    }

    private TrainingStep<MlpRegression3To1LossKernel> EnsureInitialized()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return trainingStep ?? throw new InvalidOperationException("Initialize must run before the job can step.");
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        disposed = true;
        trainingStep?.Dispose();
        optimizer?.Dispose();
        loss?.Dispose();
        scratch?.Dispose();
        targets?.Dispose();
        inputs?.Dispose();
        weights?.Dispose();
        parameters = [];
    }
}
