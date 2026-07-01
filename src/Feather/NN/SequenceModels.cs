using Feather.AD;
using Feather.Math;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.NN;

/// <summary>
/// Learned positional embeddings for fixed-length sequence models.
/// </summary>
public sealed class PositionalEmbedding : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a positional embedding table.
    /// </summary>
    public PositionalEmbedding(int blockSize, int embeddingSize, int seed = 123)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingSize);
        BlockSize = blockSize;
        EmbeddingSize = embeddingSize;
        Weight = ParameterInitializers.XavierParameterEasyGpuReference(
            "weight",
            new TensorShape(blockSize, embeddingSize),
            blockSize,
            embeddingSize,
            seed);
        parameters.Add(Weight);
    }

    /// <summary>
    /// Gets the maximum sequence length supported by the table.
    /// </summary>
    public int BlockSize { get; }

    /// <summary>
    /// Gets the embedding width.
    /// </summary>
    public int EmbeddingSize { get; }

    /// <summary>
    /// Gets the learnable position table stored as <c>[BlockSize, EmbeddingSize]</c>.
    /// </summary>
    public Parameter<float> Weight { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Reads one position embedding vector. This host helper is intended for generation and tests.
    /// </summary>
    public float[] GetVector(int position)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if ((uint)position >= (uint)BlockSize)
        {
            throw new ArgumentOutOfRangeException(nameof(position), "Position is outside the embedding table.");
        }

        var values = Weight.Value.Buffer.ToArray();
        var vector = new float[EmbeddingSize];
        Array.Copy(values, position * EmbeddingSize, vector, 0, EmbeddingSize);
        return vector;
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
}

/// <summary>
/// Causal self-attention weights for GPT-style sequence models.
/// </summary>
public sealed class SelfAttention : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a causal self-attention module.
    /// </summary>
    public SelfAttention(int embeddingSize, int headCount = 1, int seed = 456)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        if (embeddingSize % headCount != 0)
        {
            throw new ArgumentException("SelfAttention requires embeddingSize to be divisible by headCount.", nameof(headCount));
        }

        EmbeddingSize = embeddingSize;
        HeadCount = headCount;
        HeadSize = embeddingSize / headCount;
        var values = new float[4 * embeddingSize * embeddingSize];
        var projectionSize = embeddingSize * embeddingSize;
        for (var layer = 0; layer < 4; layer++)
        {
            var layerValues = ParameterInitializers.XavierUniformEasyGpuReference(
                projectionSize,
                embeddingSize,
                embeddingSize,
                seed + layer);
            Array.Copy(layerValues, 0, values, layer * projectionSize, projectionSize);
        }

        Weights = new Parameter<float>(
            "weights",
            new Tensor<float>(new TensorShape(4, embeddingSize, embeddingSize), GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(new TensorShape(4, embeddingSize, embeddingSize), GPU.CreateBuffer<float>(values.Length)));
        parameters.Add(Weights);
    }

    /// <summary>
    /// Gets the embedding width.
    /// </summary>
    public int EmbeddingSize { get; }

    /// <summary>
    /// Gets the number of attention heads.
    /// </summary>
    public int HeadCount { get; }

    /// <summary>
    /// Gets the size of each attention head.
    /// </summary>
    public int HeadSize { get; }

    /// <summary>
    /// Gets the packed Q/K/V/O projection weights stored as <c>[4, EmbeddingSize, EmbeddingSize]</c>.
    /// </summary>
    public Parameter<float> Weights { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Runs host-readback causal self-attention for a rank-2 <c>[sequence, embedding]</c> tensor.
    /// </summary>
    public Tensor<float> ForwardHost(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);
        var dimensions = input.Shape.Dimensions;
        if (input.Shape.Rank != 2 || dimensions[1] != EmbeddingSize)
        {
            throw new ArgumentException($"SelfAttention expected a rank-2 [sequence, {EmbeddingSize}] tensor, but received [{string.Join(", ", dimensions)}].", nameof(input));
        }

        var output = RunHost(input.Buffer.ToArray(), dimensions[0], includeOutputProjection: true);
        return new Tensor<float>(input.Shape, GPU.CreateBuffer<float>(output), input.RequiresGrad || Weights.Value.RequiresGrad);
    }

    /// <summary>
    /// Runs host-side attention math from a flattened sequence buffer.
    /// </summary>
    public float[] RunHost(float[] input, int sequenceLength, bool includeOutputProjection = true)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);
        if (sequenceLength <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sequenceLength), "Sequence length must be positive.");
        }

        if (input.Length != sequenceLength * EmbeddingSize)
        {
            throw new ArgumentException("Input length must equal sequenceLength * embeddingSize.", nameof(input));
        }

        return TransformerMath.CausalSelfAttention(
            input,
            Weights.Value.Buffer.ToArray(),
            sequenceLength,
            EmbeddingSize,
            HeadCount,
            includeOutputProjection);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Weights.Dispose();
        disposed = true;
    }
}

/// <summary>
/// Pre-norm causal Transformer block with attention, residuals, and a ReLU MLP.
/// </summary>
public sealed class TransformerBlock : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a Transformer block.
    /// </summary>
    public TransformerBlock(int blockSize, int embeddingSize, int headCount, int? mlpSize = null, int seed = 456)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headCount);
        BlockSize = blockSize;
        EmbeddingSize = embeddingSize;
        HeadCount = headCount;
        MlpSize = mlpSize ?? checked(4 * embeddingSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(MlpSize);

        Attention = new SelfAttention(embeddingSize, headCount, seed);
        Fc1 = ParameterInitializers.XavierParameterEasyGpuReference("fc1", new TensorShape(MlpSize, embeddingSize), embeddingSize, MlpSize, seed + 10);
        Fc2 = ParameterInitializers.XavierParameterEasyGpuReference("fc2", new TensorShape(embeddingSize, MlpSize), MlpSize, embeddingSize, seed + 11);

        parameters.AddRange(Attention.Parameters);
        parameters.Add(Fc1);
        parameters.Add(Fc2);
    }

    /// <summary>
    /// Gets the maximum sequence length.
    /// </summary>
    public int BlockSize { get; }

    /// <summary>
    /// Gets the residual stream width.
    /// </summary>
    public int EmbeddingSize { get; }

    /// <summary>
    /// Gets the attention head count.
    /// </summary>
    public int HeadCount { get; }

    /// <summary>
    /// Gets the hidden MLP width.
    /// </summary>
    public int MlpSize { get; }

    /// <summary>
    /// Gets the attention module.
    /// </summary>
    public SelfAttention Attention { get; }

    /// <summary>
    /// Gets the first MLP projection stored as <c>[MlpSize, EmbeddingSize]</c>.
    /// </summary>
    public Parameter<float> Fc1 { get; }

    /// <summary>
    /// Gets the second MLP projection stored as <c>[EmbeddingSize, MlpSize]</c>.
    /// </summary>
    public Parameter<float> Fc2 { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Runs host-readback block inference for a rank-2 <c>[sequence, embedding]</c> tensor.
    /// </summary>
    public Tensor<float> ForwardHost(Tensor<float> input)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        ArgumentNullException.ThrowIfNull(input);
        var dimensions = input.Shape.Dimensions;
        if (input.Shape.Rank != 2 || dimensions[0] > BlockSize || dimensions[1] != EmbeddingSize)
        {
            throw new ArgumentException($"TransformerBlock expected [sequence <= {BlockSize}, {EmbeddingSize}], but received [{string.Join(", ", dimensions)}].", nameof(input));
        }

        var output = RunHost(input.Buffer.ToArray(), dimensions[0]);
        return new Tensor<float>(input.Shape, GPU.CreateBuffer<float>(output), input.RequiresGrad || Parameters.Any(parameter => parameter is Parameter<float> { Value.RequiresGrad: true }));
    }

    /// <summary>
    /// Runs host-side block inference from a flattened sequence buffer.
    /// </summary>
    public float[] RunHost(float[] input, int sequenceLength)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return TransformerMath.TransformerBlock(
            input,
            Attention.Weights.Value.Buffer.ToArray(),
            Fc1.Value.Buffer.ToArray(),
            Fc2.Value.Buffer.ToArray(),
            sequenceLength,
            EmbeddingSize,
            HeadCount,
            MlpSize);
    }

    /// <inheritdoc />
    public override void QualifyParameters(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        Attention.QualifyParameters($"{prefix}.attention");
        Fc1.Qualify(prefix);
        Fc2.Qualify(prefix);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Attention.Dispose();
        Fc1.Dispose();
        Fc2.Dispose();
        disposed = true;
    }
}

/// <summary>
/// A compact GPT-style language model with token embeddings, positional embeddings, one Transformer
/// block, and an LM head. The model owns typed parameters; generated training kernels are created
/// through <see cref="CreateTrainer"/>.
/// </summary>
public sealed class GptLanguageModel : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes a GPT-style language model.
    /// </summary>
    public GptLanguageModel(int vocabularySize, int blockSize, int embeddingSize, int headCount, int seed = 42)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocabularySize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(embeddingSize);
        VocabularySize = vocabularySize;
        BlockSize = blockSize;
        EmbeddingSize = embeddingSize;
        HeadCount = headCount;

        TokenEmbedding = new Embedding(vocabularySize, embeddingSize);
        TokenEmbedding.Weight.Value.Buffer.Upload(ParameterInitializers.XavierUniformEasyGpuReference(vocabularySize * embeddingSize, vocabularySize, embeddingSize, seed));
        PositionalEmbedding = new PositionalEmbedding(blockSize, embeddingSize, seed + 81);
        Block = new TransformerBlock(blockSize, embeddingSize, headCount, seed: seed + 414);
        LmHead = ParameterInitializers.XavierParameterEasyGpuReference("lmHead", new TensorShape(vocabularySize, embeddingSize), vocabularySize, embeddingSize, seed + 747);

        TokenEmbedding.QualifyParameters("tokenEmbedding");
        PositionalEmbedding.QualifyParameters("positionEmbedding");
        Block.QualifyParameters("block0");
        LmHead.Qualify("lmHead");
        AddTrainerGradientAliases();

        parameters.AddRange(TokenEmbedding.Parameters);
        parameters.AddRange(PositionalEmbedding.Parameters);
        parameters.AddRange(Block.Parameters);
        parameters.Add(LmHead);
    }

    /// <summary>
    /// Gets the vocabulary size.
    /// </summary>
    public int VocabularySize { get; }

    /// <summary>
    /// Gets the context window length.
    /// </summary>
    public int BlockSize { get; }

    /// <summary>
    /// Gets the embedding width.
    /// </summary>
    public int EmbeddingSize { get; }

    /// <summary>
    /// Gets the attention head count.
    /// </summary>
    public int HeadCount { get; }

    /// <summary>
    /// Gets the token embedding module.
    /// </summary>
    public Embedding TokenEmbedding { get; }

    /// <summary>
    /// Gets the positional embedding module.
    /// </summary>
    public PositionalEmbedding PositionalEmbedding { get; }

    /// <summary>
    /// Gets the Transformer block.
    /// </summary>
    public TransformerBlock Block { get; }

    /// <summary>
    /// Gets the LM head stored as <c>[VocabularySize, EmbeddingSize]</c>.
    /// </summary>
    public Parameter<float> LmHead { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Creates an AD-backed trainer over batches shaped <c>[batchSize, blockSize + 1]</c>.
    /// </summary>
    public GptLanguageModelTrainer CreateTrainer(int batchSize, Optimizer optimizer)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new GptLanguageModelTrainer(this, batchSize, optimizer);
    }

    /// <summary>
    /// Runs host-side inference for one full context and returns logits for the final position.
    /// </summary>
    public float[] PredictNextHost(ReadOnlySpan<int> context)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (context.Length != BlockSize)
        {
            throw new ArgumentException($"Context must contain exactly {BlockSize} tokens.", nameof(context));
        }

        var wte = TokenEmbedding.Weight.Value.Buffer.ToArray();
        var wpe = PositionalEmbedding.Weight.Value.Buffer.ToArray();
        var x = new float[BlockSize * EmbeddingSize];
        for (var pos = 0; pos < BlockSize; pos++)
        {
            var token = context[pos];
            if ((uint)token >= (uint)VocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(context), "Context token is outside the vocabulary range.");
            }

            for (var d = 0; d < EmbeddingSize; d++)
            {
                x[(pos * EmbeddingSize) + d] = wte[(token * EmbeddingSize) + d] + wpe[(pos * EmbeddingSize) + d];
            }
        }

        var block = Block.RunHost(x, BlockSize);
        var lm = LmHead.Value.Buffer.ToArray();
        var logits = new float[VocabularySize];
        var finalOffset = (BlockSize - 1) * EmbeddingSize;
        for (var token = 0; token < VocabularySize; token++)
        {
            var value = 0f;
            for (var d = 0; d < EmbeddingSize; d++)
            {
                value += lm[(token * EmbeddingSize) + d] * block[finalOffset + d];
            }

            logits[token] = value;
        }

        return logits;
    }

    /// <inheritdoc />
    public override void QualifyParameters(string prefix)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prefix);
        TokenEmbedding.QualifyParameters($"{prefix}.tokenEmbedding");
        PositionalEmbedding.QualifyParameters($"{prefix}.positionEmbedding");
        Block.QualifyParameters($"{prefix}.block0");
        LmHead.Qualify(prefix);
        AddTrainerGradientAliases();
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        TokenEmbedding.Dispose();
        PositionalEmbedding.Dispose();
        Block.Dispose();
        LmHead.Dispose();
        disposed = true;
    }

    private void AddTrainerGradientAliases()
    {
        TokenEmbedding.Weight.AddGradientAlias("tokenEmbedding");
        PositionalEmbedding.Weight.AddGradientAlias("positionEmbedding");
        Block.Attention.Weights.AddGradientAlias("attentionWeights");
        Block.Fc1.AddGradientAlias("fc1");
        Block.Fc2.AddGradientAlias("fc2");
        LmHead.AddGradientAlias("lmHead");
    }
}

/// <summary>
/// AD trainer for <see cref="GptLanguageModel"/>.
/// </summary>
public sealed class GptLanguageModelTrainer : IDisposable
{
    private readonly GptLanguageModel model;
    private readonly Optimizer optimizer;
    private readonly GpuBuffer<int> tokens;
    private readonly GpuBuffer<float> scratch;
    private readonly GpuBuffer<float> loss;
    private readonly GpuADKernel<GptLanguageModelTrainingKernel> adKernel;
    private bool disposed;

    internal GptLanguageModelTrainer(GptLanguageModel model, int batchSize, Optimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);

        this.model = model;
        this.optimizer = optimizer;
        BatchSize = batchSize;
        tokens = GPU.CreateBuffer<int>(checked(batchSize * (model.BlockSize + 1)));
        scratch = GPU.CreateBuffer<float>(checked(batchSize * TrainingWorkspaceElementsPerBatch(model.BlockSize, model.EmbeddingSize, model.HeadCount, model.Block.MlpSize, model.VocabularySize)));
        loss = GPU.CreateBuffer<float>(batchSize);
        adKernel = GPU.CreateADKernel(new GptLanguageModelTrainingKernel(
            tokens.AsReadOnly(),
            scratch.AsReadWrite(),
            model.TokenEmbedding.Weight.Value.AsReadWriteBuffer(),
            model.PositionalEmbedding.Weight.Value.AsReadWriteBuffer(),
            model.Block.Attention.Weights.Value.AsReadWriteBuffer(),
            model.Block.Fc1.Value.AsReadWriteBuffer(),
            model.Block.Fc2.Value.AsReadWriteBuffer(),
            model.LmHead.Value.AsReadWriteBuffer(),
            loss.AsReadWrite(),
            new Uniform<int>(model.BlockSize),
            new Uniform<int>(model.EmbeddingSize),
            new Uniform<int>(model.HeadCount),
            new Uniform<int>(model.Block.MlpSize),
            new Uniform<int>(model.VocabularySize),
            new Uniform<float>(1f / checked(batchSize * model.BlockSize))));
    }

    /// <summary>
    /// Gets the batch size.
    /// </summary>
    public int BatchSize { get; }

    /// <summary>
    /// Gets the native route used by the most recent generated trainer dispatch.
    /// </summary>
    public DispatchPath LastDispatchPath { get; private set; } = DispatchPath.None;

    /// <summary>
    /// Gets a value indicating whether managed debug gradient values have been materialized.
    /// </summary>
    public bool GradientsMaterialized { get; private set; }

    /// <summary>
    /// Gets the loss returned by the most recent train or evaluation call.
    /// </summary>
    public float LastLoss { get; private set; } = float.NaN;

    /// <summary>
    /// Uploads a token batch, runs backward, applies the optimizer, and returns mean next-token loss.
    /// </summary>
    public float TrainBatch(ReadOnlySpan<int> tokenBatch)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        UploadTokenBatch(tokenBatch);
        adKernel.Backward(BatchSize);
        optimizer.Step(adKernel);
        return UpdateDiagnosticsFromLossBuffer();
    }

    /// <summary>
    /// Uploads a token batch, runs the generated loss kernel, and returns mean next-token loss without
    /// updating parameters. This is intended for progress reporting and smoke tests.
    /// </summary>
    public float EvaluateBatch(ReadOnlySpan<int> tokenBatch)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        UploadTokenBatch(tokenBatch);
        adKernel.Forward(BatchSize);
        return UpdateDiagnosticsFromLossBuffer();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        adKernel.Dispose();
        tokens.Dispose();
        scratch.Dispose();
        loss.Dispose();
        disposed = true;
    }

    internal static int TrainingWorkspaceElementsPerBatch(int blockSize, int embeddingSize, int headCount, int mlpSize, int vocabularySize)
        => checked(
            (11 * blockSize * embeddingSize) +
            (2 * blockSize * mlpSize) +
            (blockSize * headCount * blockSize) +
            (blockSize * headCount) +
            (blockSize * headCount) +
            (blockSize * vocabularySize) +
            blockSize +
            (6 * blockSize));

    private void UploadTokenBatch(ReadOnlySpan<int> tokenBatch)
    {
        var expected = checked(BatchSize * (model.BlockSize + 1));
        if (tokenBatch.Length != expected)
        {
            throw new ArgumentException($"Token batch must contain {expected} values shaped [batchSize, blockSize + 1].", nameof(tokenBatch));
        }

        ValidateTokens(tokenBatch, model.VocabularySize);
        tokens.Upload(tokenBatch);
    }

    private float UpdateDiagnosticsFromLossBuffer()
    {
        LastDispatchPath = adKernel.LastDispatchPath;
        GradientsMaterialized = adKernel.Gradients.HasMaterializedValues;
        LastLoss = loss.ToArray().Sum();
        return LastLoss;
    }

    private static void ValidateTokens(ReadOnlySpan<int> tokenBatch, int vocabularySize)
    {
        for (var i = 0; i < tokenBatch.Length; i++)
        {
            if ((uint)tokenBatch[i] >= (uint)vocabularySize)
            {
                throw new ArgumentOutOfRangeException(nameof(tokenBatch), $"Token at flat index {i} is outside the vocabulary range.");
            }
        }
    }

    internal float[] ReadGradientForTesting(Parameter<float> parameter)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        using var gradient = GPU.CreateBuffer<float>(parameter.ElementCount);
        var matches = adKernel.FindGradientMatches(parameter.GradientNames, parameter.ElementCount);
        if (matches.Length != 1)
        {
            throw new InvalidOperationException($"Expected exactly one gradient for parameter '{parameter.FullName}', found {matches.Length}.");
        }

        adKernel.CopyGradientToBuffer(matches[0].Index, gradient);
        return gradient.ToArray();
    }
}

/// <summary>
/// AD trainer for the compact two-token attention classification example.
/// </summary>
public sealed class SelfAttentionClassifier : Module
{
    private readonly List<IParameter> parameters = [];
    private bool disposed;

    /// <summary>
    /// Initializes the compact self-attention classifier used by the transformer sample.
    /// </summary>
    public SelfAttentionClassifier(int seed = 123)
    {
        Weights = ParameterInitializers.XavierParameter("weights", new TensorShape(9), 2, 2, seed);
        Weights.Value.Buffer.Upload(CreateInitialWeights(seed));
        parameters.Add(Weights);
    }

    /// <summary>
    /// Gets the packed trainable weights.
    /// </summary>
    public Parameter<float> Weights { get; }

    /// <inheritdoc />
    public override IEnumerable<IParameter> Parameters => parameters;

    /// <summary>
    /// Creates an AD-backed trainer.
    /// </summary>
    public SelfAttentionClassifierTrainer CreateTrainer(ReadOnlySpan<float> features, ReadOnlySpan<float> labels, Optimizer optimizer)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        return new SelfAttentionClassifierTrainer(this, features, labels, optimizer);
    }

    /// <summary>
    /// Runs host-side prediction for one two-token sample.
    /// </summary>
    public float PredictHost(float x0, float x1)
        => TransformerMath.TwoTokenAttentionClassifier(Weights.Value.Buffer.ToArray(), x0, x1);

    /// <inheritdoc />
    public override void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Weights.Dispose();
        disposed = true;
    }

    private static float[] CreateInitialWeights(int seed)
    {
        var jitter = ParameterInitializers.XavierUniform(8, 2, 2, seed);
        return
        [
            0.2f + (0.05f * jitter[0]),
            0.2f + (0.05f * jitter[1]),
            0.4f + (0.05f * jitter[2]),
            0.4f + (0.05f * jitter[3]),
            0.4f + (0.05f * jitter[4]),
            0.05f * jitter[5],
            0.05f * jitter[6],
            0.8f + (0.05f * jitter[7]),
            0f
        ];
    }
}

/// <summary>
/// AD trainer for <see cref="SelfAttentionClassifier"/>.
/// </summary>
public sealed class SelfAttentionClassifierTrainer : IDisposable
{
    private readonly Optimizer optimizer;
    private readonly GpuBuffer<float> features;
    private readonly GpuBuffer<float> labels;
    private readonly GpuBuffer<float> loss;
    private readonly GpuADKernel<SelfAttentionClassifierTrainingKernel> adKernel;
    private bool disposed;

    internal SelfAttentionClassifierTrainer(SelfAttentionClassifier model, ReadOnlySpan<float> features, ReadOnlySpan<float> labels, Optimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);
        if (features.Length == 0 || features.Length % 2 != 0)
        {
            throw new ArgumentException("Self-attention classifier features must be shaped [sampleCount, 2].", nameof(features));
        }

        if (labels.Length != features.Length / 2)
        {
            throw new ArgumentException("Self-attention classifier labels must contain one value per sample.", nameof(labels));
        }

        this.optimizer = optimizer;
        SampleCount = labels.Length;
        this.features = GPU.CreateBuffer<float>(features, BufferAccess.ReadOnly);
        this.labels = GPU.CreateBuffer<float>(labels, BufferAccess.ReadOnly);
        loss = GPU.CreateBuffer<float>(SampleCount);
        adKernel = GPU.CreateADKernel(new SelfAttentionClassifierTrainingKernel(
            this.features.AsReadOnly(),
            this.labels.AsReadOnly(),
            model.Weights.Value.AsReadWriteBuffer(),
            loss.AsReadWrite()));
    }

    /// <summary>
    /// Gets the sample count.
    /// </summary>
    public int SampleCount { get; }

    /// <summary>
    /// Gets the native route used by the most recent generated trainer dispatch.
    /// </summary>
    public DispatchPath LastDispatchPath { get; private set; } = DispatchPath.None;

    /// <summary>
    /// Gets a value indicating whether managed debug gradient values have been materialized.
    /// </summary>
    public bool GradientsMaterialized { get; private set; }

    /// <summary>
    /// Gets the loss returned by the most recent train or evaluation call.
    /// </summary>
    public float LastLoss { get; private set; } = float.NaN;

    /// <summary>
    /// Runs one train step and returns the mean loss.
    /// </summary>
    public float TrainStep()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        adKernel.Backward(SampleCount);
        optimizer.Step(adKernel);
        return UpdateDiagnosticsFromLossBuffer();
    }

    /// <summary>
    /// Evaluates the loss without applying an optimizer step.
    /// </summary>
    public float EvaluateLoss()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        adKernel.Backward(SampleCount);
        return UpdateDiagnosticsFromLossBuffer();
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        adKernel.Dispose();
        features.Dispose();
        labels.Dispose();
        loss.Dispose();
        disposed = true;
    }

    private float UpdateDiagnosticsFromLossBuffer()
    {
        LastDispatchPath = adKernel.LastDispatchPath;
        GradientsMaterialized = adKernel.Gradients.HasMaterializedValues;
        LastLoss = loss.ToArray().Average();
        return LastLoss;
    }
}

internal static class TransformerMath
{
    public static float[] CausalSelfAttention(float[] input, float[] weights, int sequenceLength, int embeddingSize, int headCount, bool includeOutputProjection)
    {
        var headSize = embeddingSize / headCount;
        var keys = new float[sequenceLength * embeddingSize];
        var values = new float[sequenceLength * embeddingSize];
        var queries = new float[sequenceLength * embeddingSize];
        var attention = new float[sequenceLength * embeddingSize];
        var scale = 1f / MathF.Sqrt(headSize);

        for (var pos = 0; pos < sequenceLength; pos++)
        {
            for (var o = 0; o < embeddingSize; o++)
            {
                var q = 0f;
                var k = 0f;
                var v = 0f;
                for (var i = 0; i < embeddingSize; i++)
                {
                    var xi = input[(pos * embeddingSize) + i];
                    q += weights[(0 * embeddingSize * embeddingSize) + (o * embeddingSize) + i] * xi;
                    k += weights[(1 * embeddingSize * embeddingSize) + (o * embeddingSize) + i] * xi;
                    v += weights[(2 * embeddingSize * embeddingSize) + (o * embeddingSize) + i] * xi;
                }

                queries[(pos * embeddingSize) + o] = q;
                keys[(pos * embeddingSize) + o] = k;
                values[(pos * embeddingSize) + o] = v;
            }

            for (var head = 0; head < headCount; head++)
            {
                var headStart = head * headSize;
                var scores = new float[pos + 1];
                var maxScore = -1e9f;
                for (var t = 0; t <= pos; t++)
                {
                    var score = 0f;
                    for (var d = 0; d < headSize; d++)
                    {
                        score += queries[(pos * embeddingSize) + headStart + d] * keys[(t * embeddingSize) + headStart + d];
                    }

                    scores[t] = score * scale;
                    maxScore = MathF.Max(maxScore, scores[t]);
                }

                var sumExp = 0f;
                for (var t = 0; t < scores.Length; t++)
                {
                    scores[t] = MathF.Exp(scores[t] - maxScore);
                    sumExp += scores[t];
                }

                for (var d = 0; d < headSize; d++)
                {
                    var sum = 0f;
                    for (var t = 0; t <= pos; t++)
                    {
                        sum += scores[t] / sumExp * values[(t * embeddingSize) + headStart + d];
                    }

                    attention[(pos * embeddingSize) + headStart + d] = sum;
                }
            }
        }

        if (!includeOutputProjection)
        {
            return attention;
        }

        var output = new float[attention.Length];
        for (var pos = 0; pos < sequenceLength; pos++)
        {
            for (var o = 0; o < embeddingSize; o++)
            {
                var sum = 0f;
                for (var i = 0; i < embeddingSize; i++)
                {
                    sum += weights[(3 * embeddingSize * embeddingSize) + (o * embeddingSize) + i] * attention[(pos * embeddingSize) + i];
                }

                output[(pos * embeddingSize) + o] = sum;
            }
        }

        return output;
    }

    public static float[] TransformerBlock(float[] input, float[] attentionWeights, float[] fc1, float[] fc2, int sequenceLength, int embeddingSize, int headCount, int mlpSize)
    {
        var x = (float[])input.Clone();
        var norm1 = new float[x.Length];
        for (var pos = 0; pos < sequenceLength; pos++)
        {
            RmsNorm(x, norm1, pos * embeddingSize, embeddingSize);
        }

        var attention = CausalSelfAttention(norm1, attentionWeights, sequenceLength, embeddingSize, headCount, includeOutputProjection: true);
        for (var i = 0; i < x.Length; i++)
        {
            x[i] += attention[i];
        }

        var norm2 = new float[x.Length];
        for (var pos = 0; pos < sequenceLength; pos++)
        {
            RmsNorm(x, norm2, pos * embeddingSize, embeddingSize);
            var hidden = new float[mlpSize];
            for (var h = 0; h < mlpSize; h++)
            {
                var sum = 0f;
                for (var d = 0; d < embeddingSize; d++)
                {
                    sum += fc1[(h * embeddingSize) + d] * norm2[(pos * embeddingSize) + d];
                }

                hidden[h] = MathF.Max(sum, 0f);
            }

            for (var o = 0; o < embeddingSize; o++)
            {
                var sum = 0f;
                for (var h = 0; h < mlpSize; h++)
                {
                    sum += fc2[(o * mlpSize) + h] * hidden[h];
                }

                x[(pos * embeddingSize) + o] += sum;
            }
        }

        return x;
    }

    public static float TwoTokenAttentionClassifier(float[] weights, float x0, float x1)
    {
        var wq = weights[0];
        var wk = weights[1];
        var wv = weights[2];
        var wo0 = weights[3];
        var wo1 = weights[4];
        var bq = weights[5];
        var bk = weights[6];
        var bv = weights[7];

        var q0 = (wq * x0) + bq;
        var q1 = (wq * x1) + bq;
        var k0 = (wk * x0) + bk;
        var k1 = (wk * x1) + bk;
        var v0 = (wv * x0) + bv;
        var v1 = (wv * x1) + bv;

        var e00 = MathF.Exp(q0 * k0);
        var e01 = MathF.Exp(q0 * k1);
        var a00 = e00 / (e00 + e01);
        var a01 = e01 / (e00 + e01);
        var e10 = MathF.Exp(q1 * k0);
        var e11 = MathF.Exp(q1 * k1);
        var a10 = e10 / (e10 + e11);
        var a11 = e11 / (e10 + e11);

        var y0 = (a00 * v0) + (a01 * v1);
        var y1 = (a10 * v0) + (a11 * v1);
        return (wo0 * MathF.Max(y0, 0f)) + (wo1 * MathF.Max(y1, 0f));
    }

    private static void RmsNorm(float[] input, float[] output, int offset, int count)
    {
        var meanSquare = 0f;
        for (var i = 0; i < count; i++)
        {
            var value = input[offset + i];
            meanSquare += value * value;
        }

        var scale = 1f / MathF.Sqrt((meanSquare / count) + 1e-5f);
        for (var i = 0; i < count; i++)
        {
            output[offset + i] = input[offset + i] * scale;
        }
    }
}

[Kernel]
[AutoDiff]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct SelfAttentionClassifierTrainingKernel(
    ReadOnlyBuffer<float> features,
    ReadOnlyBuffer<float> labels,
    ReadWriteBuffer<float> weights,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int id = ThreadIds.X;
        float x0 = features[(id * 2) + 0];
        float x1 = features[(id * 2) + 1];
        float yTrue = labels[id];

        float wq = weights[0];
        float wk = weights[1];
        float wv = weights[2];
        float wo0 = weights[3];
        float wo1 = weights[4];
        float bq = weights[5];
        float bk = weights[6];
        float bv = weights[7];

        float q0 = (wq * x0) + bq;
        float q1 = (wq * x1) + bq;
        float k0 = (wk * x0) + bk;
        float k1 = (wk * x1) + bk;
        float v0 = (wv * x0) + bv;
        float v1 = (wv * x1) + bv;

        float e00 = ShaderMath.Exp(q0 * k0);
        float e01 = ShaderMath.Exp(q0 * k1);
        float e10 = ShaderMath.Exp(q1 * k0);
        float e11 = ShaderMath.Exp(q1 * k1);

        float y0 = ((e00 / (e00 + e01)) * v0) + ((e01 / (e00 + e01)) * v1);
        float y1 = ((e10 / (e10 + e11)) * v0) + ((e11 / (e10 + e11)) * v1);
        float prediction = (wo0 * ShaderMath.Max(y0, 0f)) + (wo1 * ShaderMath.Max(y1, 0f));
        float diff = prediction - yTrue;
        float l = diff * diff;
        loss[id] = l;

        ADMarker.Parameter(weights[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[AutoDiff]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct GptLanguageModelTrainingKernel(
    ReadOnlyBuffer<int> tokens,
    ReadWriteBuffer<float> scratch,
    ReadWriteBuffer<float> tokenEmbedding,
    ReadWriteBuffer<float> positionEmbedding,
    ReadWriteBuffer<float> attentionWeights,
    ReadWriteBuffer<float> fc1,
    ReadWriteBuffer<float> fc2,
    ReadWriteBuffer<float> lmHead,
    ReadWriteBuffer<float> loss,
    Uniform<int> blockSize,
    Uniform<int> embeddingSize,
    Uniform<int> headCount,
    Uniform<int> mlpSize,
    Uniform<int> vocabularySize,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute()
    {
        int batch = ThreadIds.X;
        int seq = blockSize.Value + 1;
        int tokenBase = batch * seq;
        int workspaceElementsPerBatch =
            (11 * blockSize.Value * embeddingSize.Value) +
            (2 * blockSize.Value * mlpSize.Value) +
            (blockSize.Value * headCount.Value * blockSize.Value) +
            (blockSize.Value * headCount.Value) +
            (blockSize.Value * headCount.Value) +
            (blockSize.Value * vocabularySize.Value) +
            blockSize.Value +
            (6 * blockSize.Value);
        int baseOffset = batch * workspaceElementsPerBatch;
        int initialBase = baseOffset;
        int norm1Base = initialBase + (blockSize.Value * embeddingSize.Value);
        int queryBase = norm1Base + (blockSize.Value * embeddingSize.Value);
        int keyBase = queryBase + (blockSize.Value * embeddingSize.Value);
        int valueBase = keyBase + (blockSize.Value * embeddingSize.Value);
        int attentionBase = valueBase + (blockSize.Value * embeddingSize.Value);
        int attentionProjectedBase = attentionBase + (blockSize.Value * embeddingSize.Value);
        int residual1Base = attentionProjectedBase + (blockSize.Value * embeddingSize.Value);
        int norm2Base = residual1Base + (blockSize.Value * embeddingSize.Value);
        int mlpOutputBase = norm2Base + (blockSize.Value * embeddingSize.Value);
        int finalBase = mlpOutputBase + (blockSize.Value * embeddingSize.Value);
        int mlpPreActivationBase = finalBase + (blockSize.Value * embeddingSize.Value);
        int hiddenBase = mlpPreActivationBase + (blockSize.Value * mlpSize.Value);
        int dotsBase = hiddenBase + (blockSize.Value * mlpSize.Value);
        int attentionMaxBase = dotsBase + (blockSize.Value * headCount.Value * blockSize.Value);
        int attentionDenomBase = attentionMaxBase + (blockSize.Value * headCount.Value);
        int logitsBase = attentionDenomBase + (blockSize.Value * headCount.Value);
        int classMaxBase = logitsBase + (blockSize.Value * vocabularySize.Value);
        int classDenomBase = classMaxBase + blockSize.Value;
        int lossTermBase = classDenomBase + blockSize.Value;
        int norm1MeanBase = lossTermBase + blockSize.Value;
        int norm1ScaleBase = norm1MeanBase + blockSize.Value;
        int norm2MeanBase = norm1ScaleBase + blockSize.Value;
        int norm2ScaleBase = norm2MeanBase + blockSize.Value;
        int headSize = embeddingSize.Value / headCount.Value;
        float attentionScale = 1f / ShaderMath.Sqrt((float)headSize);

        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            for (int d = 0; d < embeddingSize.Value; d = d + 1)
            {
                scratch[initialBase + (pos * embeddingSize.Value) + d] =
                    tokenEmbedding[(tokens[tokenBase + pos] * embeddingSize.Value) + d] +
                    positionEmbedding[(pos * embeddingSize.Value) + d];
            }
        }

        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            float norm1MeanSquare = 0f;
            for (int d = 0; d < embeddingSize.Value; d = d + 1)
            {
                norm1MeanSquare = norm1MeanSquare +
                    (scratch[initialBase + (pos * embeddingSize.Value) + d] *
                     scratch[initialBase + (pos * embeddingSize.Value) + d]);
            }

            scratch[norm1MeanBase + pos] = norm1MeanSquare / (float)embeddingSize.Value;
            scratch[norm1ScaleBase + pos] = 1f / ShaderMath.Sqrt(scratch[norm1MeanBase + pos] + 0.00001f);
            for (int d = 0; d < embeddingSize.Value; d = d + 1)
            {
                scratch[norm1Base + (pos * embeddingSize.Value) + d] =
                    scratch[initialBase + (pos * embeddingSize.Value) + d] * scratch[norm1ScaleBase + pos];
            }

            for (int o = 0; o < embeddingSize.Value; o = o + 1)
            {
                float query = 0f;
                float key = 0f;
                float val = 0f;
                for (int i = 0; i < embeddingSize.Value; i = i + 1)
                {
                    query = query +
                        (attentionWeights[(0 * embeddingSize.Value * embeddingSize.Value) + (o * embeddingSize.Value) + i] *
                         scratch[norm1Base + (pos * embeddingSize.Value) + i]);
                    key = key +
                        (attentionWeights[(1 * embeddingSize.Value * embeddingSize.Value) + (o * embeddingSize.Value) + i] *
                         scratch[norm1Base + (pos * embeddingSize.Value) + i]);
                    val = val +
                        (attentionWeights[(2 * embeddingSize.Value * embeddingSize.Value) + (o * embeddingSize.Value) + i] *
                         scratch[norm1Base + (pos * embeddingSize.Value) + i]);
                }

                scratch[queryBase + (pos * embeddingSize.Value) + o] = query;
                scratch[keyBase + (pos * embeddingSize.Value) + o] = key;
                scratch[valueBase + (pos * embeddingSize.Value) + o] = val;
            }
        }

        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            for (int head = 0; head < headCount.Value; head = head + 1)
            {
                float maxScore = -1000000000f;
                for (int t = 0; t < blockSize.Value; t = t + 1)
                {
                    if (t <= pos)
                    {
                        float score = 0f;
                        for (int d = 0; d < headSize; d = d + 1)
                        {
                            score = score +
                                (scratch[queryBase + (pos * embeddingSize.Value) + (head * headSize) + d] *
                                 scratch[keyBase + (t * embeddingSize.Value) + (head * headSize) + d]);
                        }

                        scratch[dotsBase + (pos * headCount.Value * blockSize.Value) + (head * blockSize.Value) + t] = score * attentionScale;
                        maxScore = ShaderMath.Max(maxScore, scratch[dotsBase + (pos * headCount.Value * blockSize.Value) + (head * blockSize.Value) + t]);
                    }
                }

                scratch[attentionMaxBase + (pos * headCount.Value) + head] = maxScore;
                float sumExp = 0f;
                for (int t = 0; t < blockSize.Value; t = t + 1)
                {
                    if (t <= pos)
                    {
                        sumExp = sumExp + ShaderMath.Exp(
                            scratch[dotsBase + (pos * headCount.Value * blockSize.Value) + (head * blockSize.Value) + t] -
                            scratch[attentionMaxBase + (pos * headCount.Value) + head]);
                    }
                }

                scratch[attentionDenomBase + (pos * headCount.Value) + head] = sumExp;
                for (int d = 0; d < headSize; d = d + 1)
                {
                    float value = 0f;
                    for (int t = 0; t < blockSize.Value; t = t + 1)
                    {
                        if (t <= pos)
                        {
                            value = value +
                                ((ShaderMath.Exp(
                                      scratch[dotsBase + (pos * headCount.Value * blockSize.Value) + (head * blockSize.Value) + t] -
                                      scratch[attentionMaxBase + (pos * headCount.Value) + head]) /
                                  scratch[attentionDenomBase + (pos * headCount.Value) + head]) *
                                 scratch[valueBase + (t * embeddingSize.Value) + (head * headSize) + d]);
                        }
                    }

                    scratch[attentionBase + (pos * embeddingSize.Value) + (head * headSize) + d] = value;
                }
            }

            for (int o = 0; o < embeddingSize.Value; o = o + 1)
            {
                float projected = 0f;
                for (int i = 0; i < embeddingSize.Value; i = i + 1)
                {
                    projected = projected +
                        (attentionWeights[(3 * embeddingSize.Value * embeddingSize.Value) + (o * embeddingSize.Value) + i] *
                         scratch[attentionBase + (pos * embeddingSize.Value) + i]);
                }

                scratch[attentionProjectedBase + (pos * embeddingSize.Value) + o] = projected;
                scratch[residual1Base + (pos * embeddingSize.Value) + o] = scratch[initialBase + (pos * embeddingSize.Value) + o] + scratch[attentionProjectedBase + (pos * embeddingSize.Value) + o];
            }
        }

        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            float norm2MeanSquare = 0f;
            for (int d = 0; d < embeddingSize.Value; d = d + 1)
            {
                norm2MeanSquare = norm2MeanSquare +
                    (scratch[residual1Base + (pos * embeddingSize.Value) + d] *
                     scratch[residual1Base + (pos * embeddingSize.Value) + d]);
            }

            scratch[norm2MeanBase + pos] = norm2MeanSquare / (float)embeddingSize.Value;
            scratch[norm2ScaleBase + pos] = 1f / ShaderMath.Sqrt(scratch[norm2MeanBase + pos] + 0.00001f);
            for (int d = 0; d < embeddingSize.Value; d = d + 1)
            {
                scratch[norm2Base + (pos * embeddingSize.Value) + d] =
                    scratch[residual1Base + (pos * embeddingSize.Value) + d] * scratch[norm2ScaleBase + pos];
            }

            for (int h = 0; h < mlpSize.Value; h = h + 1)
            {
                float value = 0f;
                for (int i = 0; i < embeddingSize.Value; i = i + 1)
                {
                    value = value +
                        (fc1[(h * embeddingSize.Value) + i] *
                         scratch[norm2Base + (pos * embeddingSize.Value) + i]);
                }

                scratch[mlpPreActivationBase + (pos * mlpSize.Value) + h] = value;
                scratch[hiddenBase + (pos * mlpSize.Value) + h] =
                    ShaderMath.Max(scratch[mlpPreActivationBase + (pos * mlpSize.Value) + h], 0f);
            }

            for (int o = 0; o < embeddingSize.Value; o = o + 1)
            {
                float value = 0f;
                for (int h = 0; h < mlpSize.Value; h = h + 1)
                {
                    value = value + (fc2[(o * mlpSize.Value) + h] * scratch[hiddenBase + (pos * mlpSize.Value) + h]);
                }

                scratch[mlpOutputBase + (pos * embeddingSize.Value) + o] = value;
                scratch[finalBase + (pos * embeddingSize.Value) + o] =
                    scratch[residual1Base + (pos * embeddingSize.Value) + o] +
                    scratch[mlpOutputBase + (pos * embeddingSize.Value) + o];
            }
        }

        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            float maxClassLogit = -1000000000f;
            for (int v = 0; v < vocabularySize.Value; v = v + 1)
            {
                float classLogit = 0f;
                for (int d = 0; d < embeddingSize.Value; d = d + 1)
                {
                    classLogit = classLogit + (lmHead[(v * embeddingSize.Value) + d] * scratch[finalBase + (pos * embeddingSize.Value) + d]);
                }

                scratch[logitsBase + (pos * vocabularySize.Value) + v] = classLogit;
                maxClassLogit = ShaderMath.Max(maxClassLogit, scratch[logitsBase + (pos * vocabularySize.Value) + v]);
            }

            scratch[classMaxBase + pos] = maxClassLogit;
            float sumClassExp = 0f;
            for (int v = 0; v < vocabularySize.Value; v = v + 1)
            {
                sumClassExp = sumClassExp + ShaderMath.Exp(scratch[logitsBase + (pos * vocabularySize.Value) + v] - scratch[classMaxBase + pos]);
            }

            scratch[classDenomBase + pos] = sumClassExp;
            scratch[lossTermBase + pos] =
                ShaderMath.Log(scratch[classDenomBase + pos]) +
                scratch[classMaxBase + pos] -
                scratch[logitsBase + (pos * vocabularySize.Value) + tokens[tokenBase + pos + 1]];
        }

        float totalLoss = 0f;
        for (int pos = 0; pos < blockSize.Value; pos = pos + 1)
        {
            totalLoss = totalLoss + scratch[lossTermBase + pos];
        }

        float l = totalLoss * lossScale.Value;
        loss[batch] = l;

        ADMarker.Parameter(tokenEmbedding[0]);
        ADMarker.Parameter(positionEmbedding[0]);
        ADMarker.Parameter(attentionWeights[0]);
        ADMarker.Parameter(fc1[0]);
        ADMarker.Parameter(fc2[0]);
        ADMarker.Parameter(lmHead[0]);
        ADMarker.Loss(l);
    }
}
