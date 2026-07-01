using Feather.AD;
using Feather.NN;

namespace Feather.NN.Tests;

public class NNSurfaceTests
{
    [Fact]
    public void TensorShapeComputesElementCount()
    {
        var shape = new TensorShape(2, 3, 4);

        Assert.Equal(3, shape.Rank);
        Assert.Equal(24, shape.ElementCount);
    }

    [Fact]
    public void TensorShapeDimensionsAreImmutableCopiesAndTensorExposesLayout()
    {
        var dimensions = new[] { 2, 3 };
        var shape = new TensorShape(dimensions);
        dimensions[0] = 99;
        var returned = shape.Dimensions;
        returned[1] = 99;
        using var tensor = new Tensor<float>(shape, GPU.CreateBuffer<float>(6), requiresGrad: true);

        Assert.Equal([2, 3], shape.Dimensions);
        Assert.Equal(typeof(float), tensor.ElementType);
        Assert.Equal(2, tensor.Rank);
        Assert.True(tensor.RequiresGrad);
    }

    [Fact]
    public void Tensor2DAndTensorViewExposeTypedMatrixLayout()
    {
        using var matrix = new Tensor2D<float>(2, 3, GPU.CreateBuffer<float>([1, 2, 3, 4, 5, 6]), requiresGrad: true);

        var view = matrix.AsView();

        Assert.Equal(2, matrix.Rows);
        Assert.Equal(3, matrix.Columns);
        Assert.Equal([2, 3], matrix.Shape.Dimensions);
        Assert.Equal(6, view.ElementCount);
        Assert.True(view.Tensor.RequiresGrad);
    }

    [Fact]
    public void TensorRejectsShapeAndBufferLengthMismatch()
    {
        using var buffer = GPU.CreateBuffer<float>(3);

        var ex = Assert.Throws<ArgumentException>(() => new Tensor<float>(new TensorShape(2, 2), buffer));

        Assert.Contains("shape", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ParameterOwnsGradientBufferAndCanZeroGrad()
    {
        using var parameter = new Parameter<float>(
            "weight",
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([1f, 2f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([3f, 4f])));

        parameter.ZeroGrad();

        Assert.Equal([0f, 0f], parameter.Gradient.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SequentialExposesChildParameters()
    {
        using var sequential = new Sequential(new Linear(4, 8), new Linear(8, 2));

        Assert.Equal(2, sequential.Modules.Count);
        Assert.Equal(4, sequential.Parameters.Count());
    }

    [Fact]
    public void SequentialQualifiesParameterNamesWithoutCollisions()
    {
        using var sequential = new Sequential(new Linear(4, 8), new Linear(8, 2));

        var names = sequential.Parameters.Select(parameter => parameter.FullName).ToArray();

        Assert.Equal(
            ["linear0.weight", "linear0.bias", "linear1.weight", "linear1.bias"],
            names);
        Assert.All(sequential.Parameters, parameter => Assert.DoesNotContain(parameter.Name, parameter.GradientNames));
        Assert.Contains("linear0_weight", sequential.Parameters.First().GradientNames);
    }

    [Fact]
    public void LinearForwardSupportsVectorInput()
    {
        using var input = new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 2, 3]));
        using var linear = new Linear(3, 2);

        linear.Weight.Value.Buffer.Upload([1, 0, -1, 2, 3, 4]);
        linear.Bias.Value.Buffer.Upload([10, -1]);

        using var output = linear.Forward(input);

        Assert.Equal(1, output.Shape.Rank);
        Assert.Equal([2], output.Shape.Dimensions);
        Assert.Equal([8, 19], output.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void LinearForwardSupportsBatchedNonSquareInput()
    {
        using var input = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1, 2, 3, -1, 0, 4]));
        using var linear = new Linear(3, 4);

        linear.Weight.Value.Buffer.Upload([
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            2, -1, 0.5f
        ]);
        linear.Bias.Value.Buffer.Upload([10, 20, 30, -2]);

        using var output = linear.Forward(input);

        Assert.Equal([2, 4], output.Shape.Dimensions);
        AssertClose([11, 22, 33, -0.5f, 9, 20, 34, -2], output.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void LinearForwardRejectsInvalidInputShape()
    {
        using var input = new Tensor<float>(new TensorShape(2, 2, 3), GPU.CreateBuffer<float>(12));
        using var linear = new Linear(3, 2);

        var ex = Assert.Throws<ArgumentException>(() => linear.Forward(input));

        Assert.Contains("rank-1 vector or rank-2 batch", ex.Message);
    }

    [Fact]
    public void ActivationsTransformTensorValues()
    {
        using var input = new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([-1, 0, 2]));
        using var relu = new ReLU();
        using var sigmoid = new Sigmoid();
        using var tanh = new Tanh();
        using var silu = new SiLU();

        using var reluOutput = relu.Forward(input);
        using var sigmoidOutput = sigmoid.Forward(input);
        using var tanhOutput = tanh.Forward(input);
        using var siluOutput = silu.Forward(input);

        Assert.Equal([0, 0, 2], reluOutput.Buffer.ToArray());
        AssertClose([1f / (1f + MathF.Exp(1)), 0.5f, 1f / (1f + MathF.Exp(-2))], sigmoidOutput.Buffer.ToArray());
        AssertClose([MathF.Tanh(-1), 0, MathF.Tanh(2)], tanhOutput.Buffer.ToArray());
        AssertClose([-1f / (1f + MathF.Exp(1)), 0, 2f / (1f + MathF.Exp(-2))], siluOutput.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void TensorOpsAndSoftmaxUseDeviceKernels()
    {
        using var left = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1, 2, 3, 4, 5, 6]));
        using var right = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([6, 5, 4, 3, 2, 1]));
        using var add = TensorOps.Add(left, right);
        using var subtract = TensorOps.Subtract(left, right);
        using var multiply = TensorOps.Multiply(left, right);
        using var divide = TensorOps.Divide(left, 2f);
        using var destination = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>(6));
        TensorOps.Copy(left, destination);
        TensorOps.Fill(destination, -1f);

        using var logits = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1, 2, 3, 3, 1, -1]));
        using var softmax = TensorOps.Softmax(logits);
        using var logSoftmax = TensorOps.LogSoftmax(logits);

        Assert.Equal([7, 7, 7, 7, 7, 7], add.Buffer.ToArray());
        Assert.Equal([-5, -3, -1, 1, 3, 5], subtract.Buffer.ToArray());
        Assert.Equal([6, 10, 12, 12, 10, 6], multiply.Buffer.ToArray());
        Assert.Equal([0.5f, 1, 1.5f, 2, 2.5f, 3], divide.Buffer.ToArray());
        Assert.Equal([-1, -1, -1, -1, -1, -1], destination.Buffer.ToArray());
        AssertRowsSumToOne(softmax.Buffer.ToArray(), rows: 2, columns: 3);
        AssertClose(ReferenceLogSoftmax([1, 2, 3, 3, 1, -1], rows: 2, columns: 3), logSoftmax.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(17)]
    [InlineData(257)]
    [InlineData(513)]
    public void TensorOpsHandleAwkwardNonWorkgroupMultipleLengths(int length)
    {
        var leftValues = Enumerable.Range(0, length).Select(i => (float)i).ToArray();
        var rightValues = Enumerable.Range(0, length).Select(i => 1000f + i).ToArray();
        using var left = new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(leftValues));
        using var right = new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(rightValues));

        using var add = TensorOps.Add(left, right);
        using var scaled = TensorOps.Multiply(left, 2f);
        using var relu = new ReLU();
        using var activationInput = new Tensor<float>(
            new TensorShape(length),
            GPU.CreateBuffer<float>(Enumerable.Range(0, length).Select(i => i % 2 == 0 ? -(float)i : (float)i).ToArray()));
        using var activated = relu.Forward(activationInput);

        AssertClose(leftValues.Zip(rightValues, static (l, r) => l + r).ToArray(), add.Buffer.ToArray());
        AssertClose(leftValues.Select(static value => value * 2f).ToArray(), scaled.Buffer.ToArray());
        AssertClose(Enumerable.Range(0, length).Select(i => i % 2 == 0 ? 0f : (float)i).ToArray(), activated.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void OptimizerKernelsHandleNonWorkgroupMultipleParameterLengths()
    {
        var length = 257;
        var values = Enumerable.Range(0, length).Select(i => 1f + (i * 0.01f)).ToArray();
        var gradients = Enumerable.Range(0, length).Select(i => i % 3 == 0 ? 0.25f : -0.5f).ToArray();
        using var parameter = new Parameter<float>(
            "p",
            new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(gradients)));
        var optimizer = new AdamW([parameter], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f);

        optimizer.Step();

        var expected = AdamWReference(values, gradients, learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f, step: 1, firstMoment: new float[length], secondMoment: new float[length]);
        AssertClose(expected.Value, parameter.Value.Buffer.ToArray(), tolerance: 1e-4f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SoftmaxActivationModulesCanBeUsedInSequential()
    {
        using var input = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([1, 2, 4, 1]));
        using var sequential = new Sequential(new Softmax());

        using var output = sequential.Forward(input);

        AssertRowsSumToOne(output.Buffer.ToArray(), rows: 2, columns: 2);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SequentialSupportsActivationModules()
    {
        using var input = new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([2, -4]));
        using var linear = new Linear(2, 2);
        using var sequential = new Sequential(linear, new ReLU());
        linear.Weight.Value.Buffer.Upload([1, 0, 0, 1]);
        linear.Bias.Value.Buffer.Upload([-5, 1]);

        using var output = sequential.Forward(input);

        Assert.Equal([0, 0], output.Buffer.ToArray());
    }

    [Fact]
    public void SequentialForwardDisposesIntermediateTensorsAcrossRepeatedCalls()
    {
        using var input = new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([2, -4]));
        using var sequential = new Sequential(new Linear(2, 2), new ReLU(), new Linear(2, 1));
        var first = (Linear)sequential.Modules[0];
        var second = (Linear)sequential.Modules[2];
        first.Weight.Value.Buffer.Upload([1, 0, 0, 1]);
        first.Bias.Value.Buffer.Upload([-5, 1]);
        second.Weight.Value.Buffer.Upload([1, 1]);
        second.Bias.Value.Buffer.Upload([2]);
        var baselineLive = Tensor<float>.LiveInstanceCount;

        for (var iteration = 0; iteration < 12; iteration++)
        {
            using var output = sequential.Forward(input);
            Assert.Equal([2], output.Buffer.ToArray());
            Assert.Equal(baselineLive + 1, Tensor<float>.LiveInstanceCount);
            Assert.False(input.IsDisposed);
            Assert.False(output.IsDisposed);
        }

        Assert.Equal(baselineLive, Tensor<float>.LiveInstanceCount);
        Assert.Equal([2, -4], input.Buffer.ToArray());
    }

    [Fact]
    public void SequentialForwardDisposesOwnedIntermediateWhenLaterModuleThrows()
    {
        using var input = new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([1, 2]));
        using var sequential = new Sequential(new Linear(2, 2), new Linear(3, 1));
        var first = (Linear)sequential.Modules[0];
        first.Weight.Value.Buffer.Upload([1, 0, 0, 1]);
        first.Bias.Value.Buffer.Upload([0, 0]);
        var baselineLive = Tensor<float>.LiveInstanceCount;

        var ex = Assert.Throws<ArgumentException>(() => sequential.Forward(input));

        Assert.Contains("Linear", ex.Message);
        Assert.Equal(baselineLive, Tensor<float>.LiveInstanceCount);
        Assert.False(input.IsDisposed);
        Assert.Equal([1, 2], input.Buffer.ToArray());
    }

    [Fact]
    public void EmbeddingGathersRowsForSpanAndTensorIndices()
    {
        using var embedding = new Embedding(4, 3);
        embedding.Weight.Value.Buffer.Upload([0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32]);

        using var spanOutput = embedding.Forward([2, 0, 3]);
        using var indexTensor = new Tensor<int>(new TensorShape(2, 2), GPU.CreateBuffer<int>([1, 3, 0, 2]));
        using var tensorOutput = embedding.Forward(indexTensor);

        Assert.Equal([3, 3], spanOutput.Shape.Dimensions);
        Assert.Equal([20, 21, 22, 0, 1, 2, 30, 31, 32], spanOutput.Buffer.ToArray());
        Assert.Equal([2, 2, 3], tensorOutput.Shape.Dimensions);
        Assert.Equal([10, 11, 12, 30, 31, 32, 0, 1, 2, 20, 21, 22], tensorOutput.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void LayerNormNormalizesEachRowOverLastDimension()
    {
        using var input = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1, 2, 3, 2, 4, 6]));
        using var layerNorm = new LayerNorm(3, epsilon: 1f);

        using var output = layerNorm.Forward(input);

        Assert.Equal([2, 3], output.Shape.Dimensions);
        Assert.True(output.RequiresGrad);
        AssertClose(
            [-0.7745967f, 0, 0.7745967f, -1.0444659f, 0, 1.0444659f],
            output.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void BatchNormNormalizesFeaturesAndTracksRunningStatistics()
    {
        using var input = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1, 2, 3, 3, 6, 9]));
        using var batchNorm = new BatchNorm1D(3, epsilon: 1f, momentum: 0.5f);

        using var output = batchNorm.Forward(input);

        Assert.Equal([2, 3], output.Shape.Dimensions);
        AssertClose(
            [-0.70710677f, -0.8944272f, -0.9486833f, 0.70710677f, 0.8944272f, 0.9486833f],
            output.Buffer.ToArray());
        Assert.Equal([1, 2, 3], batchNorm.RunningMean.Buffer.ToArray());
        Assert.Equal([1, 2.5f, 5], batchNorm.RunningVariance.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SequentialSupportsNormalizationModules()
    {
        using var input = new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 2, 3]));
        using var sequential = new Sequential(new LayerNorm(3, epsilon: 1f), new ReLU());

        using var output = sequential.Forward(input);

        AssertClose([0, 0, 0.7745967f], output.Buffer.ToArray());
    }

    [Fact]
    public void SelfAttentionForwardHostMatchesCpuReferenceAndValidatesShape()
    {
        using var attention = new SelfAttention(embeddingSize: 2, headCount: 1, seed: 1);
        attention.Weights.Value.Buffer.Upload([
            1, 0, 0, 1,
            1, 0, 0, 1,
            1, 0, 0, 1,
            1, 0, 0, 1
        ]);
        using var input = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([1, 0, 0, 1]));

        using var output = attention.ForwardHost(input);
        var expected = attention.RunHost([1, 0, 0, 1], sequenceLength: 2);

        Assert.Equal([2, 2], output.Shape.Dimensions);
        AssertClose(expected, output.Buffer.ToArray());

        using var wrong = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>(6));
        var ex = Assert.Throws<ArgumentException>(() => attention.ForwardHost(wrong));
        Assert.Contains("SelfAttention expected", ex.Message);
    }

    [Fact]
    public void TransformerBlockOwnsQualifiedParametersAndValidatesShape()
    {
        using var block = new TransformerBlock(blockSize: 4, embeddingSize: 2, headCount: 1, mlpSize: 3, seed: 1);
        block.QualifyParameters("encoder");

        Assert.Equal(
            ["encoder.attention.weights", "encoder.fc1", "encoder.fc2"],
            block.Parameters.Select(parameter => parameter.FullName).ToArray());

        using var input = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([1, 0, 0, 1]));
        using var output = block.ForwardHost(input);
        Assert.Equal([2, 2], output.Shape.Dimensions);
        Assert.All(output.Buffer.ToArray(), value => Assert.True(float.IsFinite(value)));

        using var wrong = new Tensor<float>(new TensorShape(5, 2), GPU.CreateBuffer<float>(10));
        var ex = Assert.Throws<ArgumentException>(() => block.ForwardHost(wrong));
        Assert.Contains("TransformerBlock expected", ex.Message);
    }

    [Fact]
    public void GptLanguageModelExposesTypedParametersTrainerAndPrediction()
    {
        using var model = new GptLanguageModel(vocabularySize: 5, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 7);
        using var optimizer = new Adam(model.Parameters, learningRate: 0.001f);
        using var trainer = model.CreateTrainer(batchSize: 2, optimizer);

        var logits = model.PredictNextHost([0, 1, 2, 3]);

        Assert.Equal(5, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.Contains(model.Parameters, parameter => parameter.FullName == "tokenEmbedding.weight");
        Assert.Contains(model.Parameters, parameter => parameter.FullName == "block0.attention.weights");
        Assert.Equal(2, trainer.BatchSize);
        Assert.Equal(DispatchPath.None, trainer.LastDispatchPath);
        Assert.False(trainer.GradientsMaterialized);
        Assert.True(float.IsNaN(trainer.LastLoss));

        var wrongBatch = Assert.Throws<ArgumentException>(() => trainer.TrainBatch([0, 1, 2]));
        Assert.Contains("Token batch must contain", wrongBatch.Message);
        var wrongToken = Assert.Throws<ArgumentOutOfRangeException>(() => trainer.TrainBatch([0, 1, 2, 3, 4, 0, 1, 2, 3, 99]));
        Assert.Contains("outside the vocabulary", wrongToken.Message);
    }

    [Fact]
    public void GptLanguageModelUsesEasyGpuReferenceInitialization()
    {
        using var model = new GptLanguageModel(vocabularySize: 36, blockSize: 32, embeddingSize: 16, headCount: 4, seed: 42);

        AssertClose(
            [-0.168248323f, -0.279813931f, 0.052502236f, -0.188487260f],
            Take(model.TokenEmbedding.Weight.Value.Buffer.ToArray(), 0, 4),
            tolerance: 1e-6f);
        AssertClose(
            [-0.152921089f, -0.045870000f, -0.326222824f, -0.197367712f],
            Take(model.PositionalEmbedding.Weight.Value.Buffer.ToArray(), 0, 4),
            tolerance: 1e-6f);

        var attention = model.Block.Attention.Weights.Value.Buffer.ToArray();
        var projectionStride = model.EmbeddingSize * model.EmbeddingSize;
        AssertClose(
            [-0.075524448f, 0.120796534f, -0.392737370f, 0.040988768f],
            Take(attention, 0, 4),
            tolerance: 1e-6f);
        AssertClose(
            [-0.075188817f, 0.199348299f, 0.200237388f, 0.073045882f],
            Take(attention, projectionStride, 4),
            tolerance: 1e-6f);

        AssertClose(
            [-0.045643143f, 0.025480861f, 0.215577393f, 0.228670559f],
            Take(model.Block.Fc1.Value.Buffer.ToArray(), 0, 4),
            tolerance: 1e-6f);
        AssertClose(
            [-0.045430871f, 0.075161359f, 0.042885002f, 0.248945258f],
            Take(model.Block.Fc2.Value.Buffer.ToArray(), 0, 4),
            tolerance: 1e-6f);
        AssertClose(
            [0.028429328f, 0.233591666f, -0.302752330f, 0.253933235f],
            Take(model.LmHead.Value.Buffer.ToArray(), 0, 4),
            tolerance: 1e-6f);
    }

    [Fact]
    public void LossesComputeMeanSquaredErrorAndCrossEntropy()
    {
        using var prediction = new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 2, 4]));
        using var target = new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 4, 1]));
        using var probabilities = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([0.1f, 0.8f, 0.1f, 0.25f, 0.25f, 0.5f]));
        using var labels = new Tensor<int>(new TensorShape(2), GPU.CreateBuffer<int>([1, 2]));
        using var logits = new Tensor<float>(new TensorShape(2, 3), GPU.CreateBuffer<float>([1f, 3f, -1f, 0.5f, -0.5f, 2f]));
        var crossEntropyLoss = new CrossEntropyLoss();

        Assert.Equal(13f / 3f, Losses.MeanSquaredError(prediction, target));
        Assert.Equal((-MathF.Log(0.8f) - MathF.Log(0.5f)) / 2f, Losses.CrossEntropy(probabilities, [1, 2]));
        using var deviceCrossEntropy = Losses.CrossEntropyTensor(probabilities, labels);
        using var logitsCrossEntropy = Losses.CrossEntropyFromLogitsTensor(logits, labels);
        using var facadeCrossEntropy = crossEntropyLoss.Forward(probabilities, labels);
        using var facadeLogitsCrossEntropy = crossEntropyLoss.FromLogits(logits, labels);
        AssertClose([(-MathF.Log(0.8f) - MathF.Log(0.5f)) / 2f], deviceCrossEntropy.Buffer.ToArray());
        AssertClose(deviceCrossEntropy.Buffer.ToArray(), facadeCrossEntropy.Buffer.ToArray());
        AssertClose([ReferenceCrossEntropyFromLogits([1f, 3f, -1f, 0.5f, -0.5f, 2f], [1, 2], 2, 3)], logitsCrossEntropy.Buffer.ToArray());
        AssertClose(logitsCrossEntropy.Buffer.ToArray(), facadeLogitsCrossEntropy.Buffer.ToArray());
        Assert.InRange(
            MathF.Abs(ReferenceCrossEntropyFromLogits([1f, 3f, -1f, 0.5f, -0.5f, 2f], [1, 2], 2, 3) - Losses.CrossEntropyFromLogits(logits, [1, 2])),
            0,
            1e-5f);
        Assert.InRange(MathF.Abs(crossEntropyLoss.FromLogits(logits, [1, 2]) - Losses.CrossEntropyFromLogits(logits, [1, 2])), 0, 1e-5f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void DeviceClassificationLossReturnsNaNForInvalidDeviceLabelsWithoutHostReadback()
    {
        using var probabilities = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([0.8f, 0.2f, 0.4f, 0.6f]));
        using var logits = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([2f, 1f, -1f, 3f]));
        using var labels = new Tensor<int>(new TensorShape(2), GPU.CreateBuffer<int>([0, 4]));

        using var probabilityLoss = Losses.CrossEntropyTensor(probabilities, labels);
        using var logitsLoss = Losses.CrossEntropyFromLogitsTensor(logits, labels);

        Assert.True(float.IsNaN(probabilityLoss.Buffer.ToArray()[0]));
        Assert.True(float.IsNaN(logitsLoss.Buffer.ToArray()[0]));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void MeanSquaredErrorUsesMeanReductionAcrossBatch()
    {
        using var prediction = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([1, 3, 5, 7]));
        using var target = new Tensor<float>(new TensorShape(2, 2), GPU.CreateBuffer<float>([0, 1, 2, 3]));

        var loss = Losses.MeanSquaredError(prediction, target);

        Assert.Equal((1f + 4f + 9f + 16f) / 4f, loss);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(257)]
    [InlineData(4097)]
    public void ScalarTensorLossesUseScalableGpuReductions(int length)
    {
        var predictionValues = Enumerable.Range(0, length).Select(i => ((i % 17) - 8) * 0.125f).ToArray();
        var targetValues = Enumerable.Range(0, length).Select(i => ((i % 11) - 5) * 0.2f).ToArray();
        using var prediction = new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(predictionValues));
        using var target = new Tensor<float>(new TensorShape(length), GPU.CreateBuffer<float>(targetValues));

        NnDispatchTrace.Reset();
        using var mse = Losses.MeanSquaredErrorTensor(prediction, target);
        var expectedMse = predictionValues.Zip(targetValues, static (p, t) => (p - t) * (p - t)).Sum() / length;
        AssertClose([expectedMse], mse.Buffer.ToArray(), tolerance: 1e-4f);
        Assert.Contains("Reduce.PartialSum", NnDispatchTrace.Operations);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);

        NnDispatchTrace.Reset();
        using var mae = Losses.MeanAbsoluteErrorTensor(prediction, target);
        var expectedMae = predictionValues.Zip(targetValues, static (p, t) => MathF.Abs(p - t)).Sum() / length;
        AssertClose([expectedMae], mae.Buffer.ToArray(), tolerance: 1e-4f);
        Assert.Contains("Reduce.PartialSum", NnDispatchTrace.Operations);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(257)]
    [InlineData(4097)]
    public void ClassificationTensorLossesUseScalableGpuReductions(int batch)
    {
        const int classes = 4;
        var probabilities = new float[batch * classes];
        var logits = new float[batch * classes];
        var labels = new int[batch];
        for (var row = 0; row < batch; row++)
        {
            labels[row] = row % classes;
            var rowSum = 0f;
            for (var cls = 0; cls < classes; cls++)
            {
                var raw = 1f + ((row + cls) % 7);
                probabilities[(row * classes) + cls] = raw;
                rowSum += raw;
                logits[(row * classes) + cls] = ((row % 5) - 2) * 0.2f + cls * 0.35f;
            }

            for (var cls = 0; cls < classes; cls++)
            {
                probabilities[(row * classes) + cls] /= rowSum;
            }
        }

        using var probabilityTensor = new Tensor<float>(new TensorShape(batch, classes), GPU.CreateBuffer<float>(probabilities));
        using var logitsTensor = new Tensor<float>(new TensorShape(batch, classes), GPU.CreateBuffer<float>(logits));
        using var labelTensor = new Tensor<int>(new TensorShape(batch), GPU.CreateBuffer<int>(labels));

        NnDispatchTrace.Reset();
        using var probabilityLoss = Losses.CrossEntropyTensor(probabilityTensor, labelTensor);
        var expectedCrossEntropy = labels.Select((label, row) => -MathF.Log(probabilities[(row * classes) + label])).Sum() / batch;
        AssertClose([expectedCrossEntropy], probabilityLoss.Buffer.ToArray(), tolerance: 1e-4f);
        Assert.Contains("Reduce.PartialSum", NnDispatchTrace.Operations);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);

        NnDispatchTrace.Reset();
        using var logitsLoss = Losses.CrossEntropyFromLogitsTensor(logitsTensor, labelTensor);
        AssertClose([ReferenceCrossEntropyFromLogits(logits, labels, batch, classes)], logitsLoss.Buffer.ToArray(), tolerance: 1e-4f);
        Assert.Contains("Reduce.PartialSum", NnDispatchTrace.Operations);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SgdStepUpdatesParametersAndZeroGradClearsGradients()
    {
        using var parameter = new Parameter<float>(
            "p",
            new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 2, 3]), requiresGrad: true),
            new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([0.5f, -1, 2])));
        var optimizer = new SGD([parameter], learningRate: 0.1f);

        optimizer.Step();

        Assert.Equal([0.95f, 2.1f, 2.8f], parameter.Value.Buffer.ToArray());

        optimizer.ZeroGrad();

        Assert.Equal([0, 0, 0], parameter.Gradient.Buffer.ToArray());
    }

    [Fact]
    public void OptimizerCanStepFromNamedDebugGradients()
    {
        using var parameter = new Parameter<float>(
            "weight",
            new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>([1, 2, 3]), requiresGrad: true),
            new Tensor<float>(new TensorShape(3), GPU.CreateBuffer<float>(3)));
        var gradients = new GradientSet();
        gradients.Register<float>("weight", [0.5f, -1, 2]);
        var optimizer = new SGD([parameter], learningRate: 0.1f);

        optimizer.StepFromDebugGradients(gradients);

        Assert.Equal([0.95f, 2.1f, 2.8f], parameter.Value.Buffer.ToArray());
        Assert.Equal([0.5f, -1, 2], parameter.Gradient.Buffer.ToArray());
    }

    [Fact]
    public void OptimizerStepFromDebugGradientsUpdatesQualifiedParameters()
    {
        using var sequential = new Sequential(new Linear(1, 1), new Linear(1, 1));
        var first = (Linear)sequential.Modules[0];
        var second = (Linear)sequential.Modules[1];
        first.Weight.Value.Buffer.Upload([1]);
        first.Bias.Value.Buffer.Upload([2]);
        second.Weight.Value.Buffer.Upload([10]);
        second.Bias.Value.Buffer.Upload([20]);

        var gradients = new GradientSet();
        gradients.Register<float>("linear0_weight", [0.5f]);
        gradients.Register<float>("linear0_bias", [1f]);
        gradients.Register<float>("linear1.weight", [2f]);
        gradients.Register<float>("linear1.bias", [4f]);
        var optimizer = new SGD(sequential.Parameters, learningRate: 0.1f);

        optimizer.StepFromDebugGradients(gradients);

        Assert.Equal([0.95f], first.Weight.Value.Buffer.ToArray());
        Assert.Equal([1.9f], first.Bias.Value.Buffer.ToArray());
        Assert.Equal([9.8f], second.Weight.Value.Buffer.ToArray());
        Assert.Equal([19.6f], second.Bias.Value.Buffer.ToArray());
        Assert.Equal([0.5f], first.Weight.Gradient.Buffer.ToArray());
        Assert.Equal([2f], second.Weight.Gradient.Buffer.ToArray());
    }

    [Fact]
    public void OptimizerStepFromDebugGradientsFailsForMissingWrongShapeAndAmbiguousGradients()
    {
        using var parameter = new Parameter<float>(
            "weight",
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([1, 2]), requiresGrad: true),
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>(2)));
        parameter.AddGradientAlias("w");
        var optimizer = new SGD([parameter], learningRate: 0.1f);

        var missing = Assert.Throws<ArgumentException>(() => optimizer.StepFromDebugGradients(new GradientSet()));
        Assert.Contains("No AD gradient matched", missing.Message);

        var wrongShape = new GradientSet();
        wrongShape.Register<float>("weight", [1, 2, 3]);
        var shapeEx = Assert.Throws<ArgumentException>(() => optimizer.StepFromDebugGradients(wrongShape));
        Assert.Contains("expects 2", shapeEx.Message);

        var wrongType = new GradientSet();
        wrongType.Register<int>("weight", [1, 2]);
        var typeEx = Assert.Throws<ArgumentException>(() => optimizer.StepFromDebugGradients(wrongType));
        Assert.Contains("not a float gradient", typeEx.Message);

        var ambiguous = new GradientSet();
        ambiguous.Register<float>("weight", [1, 2]);
        ambiguous.Register<float>("w", [1, 2]);
        var ambiguousEx = Assert.Throws<ArgumentException>(() => optimizer.StepFromDebugGradients(ambiguous));
        Assert.Contains("Multiple AD gradients", ambiguousEx.Message);
    }

    [Fact]
    public void OptimizersConsumeParameterGradientsAfterHandoff()
    {
        using var adamParameter = new Parameter<float>(
            "p",
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([1f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>(1)));
        using var rmsParameter = new Parameter<float>(
            "q",
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([1f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>(1)));
        using var momentumParameter = new Parameter<float>(
            "m",
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([1f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([0.5f])));
        using var adamWParameter = new Parameter<float>(
            "w",
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([1f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(1), GPU.CreateBuffer<float>([0.25f])));
        var adamGradients = new GradientSet();
        adamGradients.Register<float>("p", [0.25f]);
        var rmsGradients = new GradientSet();
        rmsGradients.Register<float>("q", [0.25f]);

        var adam = new Adam([adamParameter], learningRate: 0.1f);
        var rms = new RMSProp([rmsParameter], learningRate: 0.1f, alpha: 0.9f);
        var momentum = new SGD([momentumParameter], learningRate: 0.1f, momentum: 0.9f);
        var adamW = new AdamW([adamWParameter], learningRate: 0.1f, weightDecay: 0.1f);
        adam.StepFromDebugGradients(adamGradients);
        adam.StepFromDebugGradients(adamGradients);
        rms.StepFromDebugGradients(rmsGradients);
        momentum.Step();
        momentum.Step();
        adamW.Step();

        Assert.Equal([0.25f], adamParameter.Gradient.Buffer.ToArray());
        Assert.InRange(adamParameter.Value.Buffer.ToArray()[0], 0.79f, 0.81f);
        Assert.Equal([0.25f], rmsParameter.Gradient.Buffer.ToArray());
        Assert.InRange(rmsParameter.Value.Buffer.ToArray()[0], 0.68f, 0.69f);
        Assert.InRange(momentumParameter.Value.Buffer.ToArray()[0], 0.85f, 0.86f);
        Assert.InRange(adamWParameter.Value.Buffer.ToArray()[0], 0.88f, 0.90f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void OptimizersMatchCpuReferenceFormulasAcrossMultipleParameters()
    {
        using var first = new Parameter<float>(
            "first",
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([1f, -2f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([0.25f, -0.5f])));
        using var second = new Parameter<float>(
            "second",
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([0.5f, 3f]), requiresGrad: true),
            new Tensor<float>(new TensorShape(2), GPU.CreateBuffer<float>([0f, 1f])));
        var adamW = new AdamW([first, second], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f);

        adamW.Step();
        var firstAfterStep1 = first.Value.Buffer.ToArray();
        var secondAfterStep1 = second.Value.Buffer.ToArray();
        first.Gradient.Buffer.Upload([0.5f, -0.25f]);
        second.Gradient.Buffer.Upload([0f, -0.5f]);
        adamW.Step();

        var expectedFirstStep1 = AdamWReference([1f, -2f], [0.25f, -0.5f], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f, step: 1, firstMoment: [0, 0], secondMoment: [0, 0]);
        var expectedSecondStep1 = AdamWReference([0.5f, 3f], [0f, 1f], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f, step: 1, firstMoment: [0, 0], secondMoment: [0, 0]);
        var firstMoments = expectedFirstStep1.FirstMoment;
        var firstSecondMoments = expectedFirstStep1.SecondMoment;
        var secondMoments = expectedSecondStep1.FirstMoment;
        var secondSecondMoments = expectedSecondStep1.SecondMoment;
        var expectedFirstStep2 = AdamWReference(expectedFirstStep1.Value, [0.5f, -0.25f], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f, step: 2, firstMoment: firstMoments, secondMoment: firstSecondMoments);
        var expectedSecondStep2 = AdamWReference(expectedSecondStep1.Value, [0f, -0.5f], learningRate: 0.05f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f, step: 2, firstMoment: secondMoments, secondMoment: secondSecondMoments);

        AssertClose(expectedFirstStep1.Value, firstAfterStep1);
        AssertClose(expectedSecondStep1.Value, secondAfterStep1);
        AssertClose(expectedFirstStep2.Value, first.Value.Buffer.ToArray());
        AssertClose(expectedSecondStep2.Value, second.Value.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SgdMatchesCpuReferenceAcrossIndustrialEdgeCases()
    {
        using var scalar = FloatParameter("scalar", [1f], [0.25f]);
        using var vector = FloatParameter(
            "vector",
            Sequence(257, i => 1f + (i * 0.01f)),
            Sequence(257, i => i % 3 == 0 ? 0.5f : -0.25f));
        var optimizer = new SGD([scalar, vector], learningRate: 0.05f, weightDecay: 0.1f);
        var expectedScalar = scalar.Value.Buffer.ToArray();
        var expectedVector = vector.Value.Buffer.ToArray();
        float[][] scalarGradients = [[0.25f], [0f], [-0.5f]];
        float[][] vectorGradients =
        [
            Sequence(257, i => i % 3 == 0 ? 0.5f : -0.25f),
            new float[257],
            Sequence(257, i => i % 2 == 0 ? -0.125f : 0.375f)
        ];

        for (var step = 0; step < scalarGradients.Length; step++)
        {
            scalar.Gradient.Buffer.Upload(scalarGradients[step]);
            vector.Gradient.Buffer.Upload(vectorGradients[step]);
            optimizer.Step();
            ApplySgdReference(expectedScalar, scalarGradients[step], learningRate: 0.05f, weightDecay: 0.1f);
            ApplySgdReference(expectedVector, vectorGradients[step], learningRate: 0.05f, weightDecay: 0.1f);

            AssertClose(expectedScalar, scalar.Value.Buffer.ToArray(), tolerance: 1e-5f);
            AssertClose(expectedVector, vector.Value.Buffer.ToArray(), tolerance: 1e-4f);
            Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
        }

        optimizer.ZeroGrad();
        optimizer.ZeroGrad();

        Assert.Equal([0f], scalar.Gradient.Buffer.ToArray());
        Assert.All(vector.Gradient.Buffer.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void SgdMomentumMatchesCpuReferenceAcrossIndustrialEdgeCases()
    {
        using var scalar = FloatParameter("scalar", [1f], [0.25f]);
        using var vector = FloatParameter(
            "vector",
            Sequence(257, i => 0.5f + (i * 0.005f)),
            Sequence(257, i => i % 4 == 0 ? -0.5f : 0.2f));
        var optimizer = new SGD([scalar, vector], learningRate: 0.03f, momentum: 0.8f, weightDecay: 0.05f);
        var expectedScalar = scalar.Value.Buffer.ToArray();
        var expectedVector = vector.Value.Buffer.ToArray();
        var scalarMomentum = new float[1];
        var vectorMomentum = new float[257];
        float[][] scalarGradients = [[0.25f], [0f], [-0.5f]];
        float[][] vectorGradients =
        [
            Sequence(257, i => i % 4 == 0 ? -0.5f : 0.2f),
            new float[257],
            Sequence(257, i => i % 5 == 0 ? 0.75f : -0.1f)
        ];

        for (var step = 0; step < scalarGradients.Length; step++)
        {
            scalar.Gradient.Buffer.Upload(scalarGradients[step]);
            vector.Gradient.Buffer.Upload(vectorGradients[step]);
            optimizer.Step();
            ApplyMomentumReference(expectedScalar, scalarGradients[step], scalarMomentum, learningRate: 0.03f, momentumFactor: 0.8f, weightDecay: 0.05f);
            ApplyMomentumReference(expectedVector, vectorGradients[step], vectorMomentum, learningRate: 0.03f, momentumFactor: 0.8f, weightDecay: 0.05f);

            AssertClose(expectedScalar, scalar.Value.Buffer.ToArray(), tolerance: 1e-5f);
            AssertClose(expectedVector, vector.Value.Buffer.ToArray(), tolerance: 1e-4f);
            Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
        }

        optimizer.ZeroGrad();
        optimizer.ZeroGrad();

        Assert.Equal([0f], scalar.Gradient.Buffer.ToArray());
        Assert.All(vector.Gradient.Buffer.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void RmsPropMatchesCpuReferenceAcrossIndustrialEdgeCases()
    {
        using var scalar = FloatParameter("scalar", [1f], [0.25f]);
        using var vector = FloatParameter(
            "vector",
            Sequence(257, i => -1f + (i * 0.01f)),
            Sequence(257, i => i % 2 == 0 ? 0.3f : -0.2f));
        var optimizer = new RMSProp([scalar, vector], learningRate: 0.02f, alpha: 0.9f, epsilon: 1e-6f, weightDecay: 0.03f);
        var expectedScalar = scalar.Value.Buffer.ToArray();
        var expectedVector = vector.Value.Buffer.ToArray();
        var scalarAverage = new float[1];
        var vectorAverage = new float[257];
        float[][] scalarGradients = [[0.25f], [0f], [-0.5f]];
        float[][] vectorGradients =
        [
            Sequence(257, i => i % 2 == 0 ? 0.3f : -0.2f),
            new float[257],
            Sequence(257, i => i % 7 == 0 ? 0.6f : -0.15f)
        ];

        for (var step = 0; step < scalarGradients.Length; step++)
        {
            scalar.Gradient.Buffer.Upload(scalarGradients[step]);
            vector.Gradient.Buffer.Upload(vectorGradients[step]);
            optimizer.Step();
            ApplyRmsPropReference(expectedScalar, scalarGradients[step], scalarAverage, learningRate: 0.02f, alpha: 0.9f, epsilon: 1e-6f, weightDecay: 0.03f);
            ApplyRmsPropReference(expectedVector, vectorGradients[step], vectorAverage, learningRate: 0.02f, alpha: 0.9f, epsilon: 1e-6f, weightDecay: 0.03f);

            AssertClose(expectedScalar, scalar.Value.Buffer.ToArray(), tolerance: 1e-4f);
            AssertClose(expectedVector, vector.Value.Buffer.ToArray(), tolerance: 1e-4f);
            Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
        }

        optimizer.ZeroGrad();
        optimizer.ZeroGrad();

        Assert.Equal([0f], scalar.Gradient.Buffer.ToArray());
        Assert.All(vector.Gradient.Buffer.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamMatchesCpuReferenceAcrossIndustrialEdgeCases()
    {
        using var scalar = FloatParameter("scalar", [1f], [0.25f]);
        using var vector = FloatParameter(
            "vector",
            Sequence(257, i => 1.5f - (i * 0.002f)),
            Sequence(257, i => i % 2 == 0 ? 0.4f : -0.35f));
        var optimizer = new Adam([scalar, vector], learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.04f);
        var expectedScalar = scalar.Value.Buffer.ToArray();
        var expectedVector = vector.Value.Buffer.ToArray();
        var scalarFirstMoment = new float[1];
        var scalarSecondMoment = new float[1];
        var vectorFirstMoment = new float[257];
        var vectorSecondMoment = new float[257];
        float[][] scalarGradients = [[0.25f], [0f], [-0.5f]];
        float[][] vectorGradients =
        [
            Sequence(257, i => i % 2 == 0 ? 0.4f : -0.35f),
            new float[257],
            Sequence(257, i => i % 11 == 0 ? -0.6f : 0.1f)
        ];

        for (var step = 0; step < scalarGradients.Length; step++)
        {
            scalar.Gradient.Buffer.Upload(scalarGradients[step]);
            vector.Gradient.Buffer.Upload(vectorGradients[step]);
            optimizer.Step();
            ApplyAdamReference(expectedScalar, scalarGradients[step], scalarFirstMoment, scalarSecondMoment, learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, step: step + 1, weightDecay: 0.04f, gradientClip: 0f, decoupledWeightDecay: false);
            ApplyAdamReference(expectedVector, vectorGradients[step], vectorFirstMoment, vectorSecondMoment, learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, step: step + 1, weightDecay: 0.04f, gradientClip: 0f, decoupledWeightDecay: false);

            AssertClose(expectedScalar, scalar.Value.Buffer.ToArray(), tolerance: 1e-4f);
            AssertClose(expectedVector, vector.Value.Buffer.ToArray(), tolerance: 1e-4f);
            Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
        }

        optimizer.ZeroGrad();
        optimizer.ZeroGrad();

        Assert.Equal([0f], scalar.Gradient.Buffer.ToArray());
        Assert.All(vector.Gradient.Buffer.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamCoupledWeightDecayUsesSingleL2GradientTerm()
    {
        using var parameter = FloatParameter("weight", [2f, -3f, 0.5f], [-0.6f, 0.9f, -0.15f]);
        var optimizer = new Adam([parameter], learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, weightDecay: 0.2f);
        var expected = parameter.Value.Buffer.ToArray();
        var doubledDecay = parameter.Value.Buffer.ToArray();
        var firstMoment = new float[expected.Length];
        var secondMoment = new float[expected.Length];
        var doubledFirstMoment = new float[expected.Length];
        var doubledSecondMoment = new float[expected.Length];

        optimizer.Step();
        ApplyAdamReference(expected, [-0.6f, 0.9f, -0.15f], firstMoment, secondMoment, learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, step: 1, weightDecay: 0.2f, gradientClip: 0f, decoupledWeightDecay: false);
        ApplyAdamReference(doubledDecay, [-0.6f, 0.9f, -0.15f], doubledFirstMoment, doubledSecondMoment, learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, step: 1, weightDecay: 0.4f, gradientClip: 0f, decoupledWeightDecay: false);

        var actual = parameter.Value.Buffer.ToArray();
        AssertClose(expected, actual, tolerance: 1e-5f);
        Assert.Contains(Enumerable.Range(0, actual.Length), i => MathF.Abs(actual[i] - doubledDecay[i]) > 1e-4f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamParameterGroupsUseGroupSpecificLearningRatesAndTrackStepCount()
    {
        using var fast = FloatParameter("fast", [1f], [0.5f]);
        using var slow = FloatParameter("slow", [1f], [0.5f]);
        var optimizer = new Adam(
            [
                new ParameterGroup("fast", [fast], learningRate: 0.1f),
                new ParameterGroup("slow", [slow], learningRate: 0.01f)
            ],
            learningRate: 0.001f,
            beta1: 0.9f,
            beta2: 0.999f,
            epsilon: 1e-8f);

        optimizer.Step();

        Assert.Equal(1, optimizer.StepCount);
        Assert.InRange(fast.Value.Buffer.ToArray()[0], 0.89f, 0.91f);
        Assert.InRange(slow.Value.Buffer.ToArray()[0], 0.989f, 0.991f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamGradientClipMatchesCpuReference()
    {
        using var parameter = FloatParameter("weight", [1f, -2f, 0.5f], [3f, -4f, 0.1f]);
        var optimizer = new Adam([parameter], learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, gradientClip: 0.25f);
        var expected = parameter.Value.Buffer.ToArray();
        var firstMoment = new float[expected.Length];
        var secondMoment = new float[expected.Length];

        optimizer.Step();
        ApplyAdamReference(expected, [3f, -4f, 0.1f], firstMoment, secondMoment, learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, step: 1, weightDecay: 0f, gradientClip: 0.25f, decoupledWeightDecay: false);

        AssertClose(expected, parameter.Value.Buffer.ToArray(), tolerance: 1e-5f);
        Assert.Equal(0.25f, optimizer.GradientClip);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamWMatchesCpuReferenceAcrossIndustrialEdgeCases()
    {
        using var scalar = FloatParameter("scalar", [1f], [0.25f]);
        using var vector = FloatParameter(
            "vector",
            Sequence(257, i => 0.25f + (i * 0.004f)),
            Sequence(257, i => i % 3 == 0 ? 0.2f : -0.45f));
        var optimizer = new AdamW([scalar, vector], learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, weightDecay: 0.1f);
        var expectedScalar = scalar.Value.Buffer.ToArray();
        var expectedVector = vector.Value.Buffer.ToArray();
        var scalarFirstMoment = new float[1];
        var scalarSecondMoment = new float[1];
        var vectorFirstMoment = new float[257];
        var vectorSecondMoment = new float[257];
        float[][] scalarGradients = [[0.25f], [0f], [-0.5f]];
        float[][] vectorGradients =
        [
            Sequence(257, i => i % 3 == 0 ? 0.2f : -0.45f),
            new float[257],
            Sequence(257, i => i % 13 == 0 ? 0.7f : -0.05f)
        ];

        for (var step = 0; step < scalarGradients.Length; step++)
        {
            scalar.Gradient.Buffer.Upload(scalarGradients[step]);
            vector.Gradient.Buffer.Upload(vectorGradients[step]);
            optimizer.Step();
            ApplyAdamReference(expectedScalar, scalarGradients[step], scalarFirstMoment, scalarSecondMoment, learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, step: step + 1, weightDecay: 0.1f, gradientClip: 0f, decoupledWeightDecay: true);
            ApplyAdamReference(expectedVector, vectorGradients[step], vectorFirstMoment, vectorSecondMoment, learningRate: 0.02f, beta1: 0.8f, beta2: 0.9f, epsilon: 1e-6f, step: step + 1, weightDecay: 0.1f, gradientClip: 0f, decoupledWeightDecay: true);

            AssertClose(expectedScalar, scalar.Value.Buffer.ToArray(), tolerance: 1e-4f);
            AssertClose(expectedVector, vector.Value.Buffer.ToArray(), tolerance: 1e-4f);
            Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
        }

        optimizer.ZeroGrad();
        optimizer.ZeroGrad();

        Assert.Equal([0f], scalar.Gradient.Buffer.ToArray());
        Assert.All(vector.Gradient.Buffer.ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void AdamWDecoupledWeightDecayDoesNotEnterMomentGradients()
    {
        using var parameter = FloatParameter("weight", [2f, -3f, 0.5f], [0.1f, 0.2f, -0.4f]);
        var optimizer = new AdamW([parameter], learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, weightDecay: 0.2f);
        var expected = parameter.Value.Buffer.ToArray();
        var coupledDecay = parameter.Value.Buffer.ToArray();
        var firstMoment = new float[expected.Length];
        var secondMoment = new float[expected.Length];
        var coupledFirstMoment = new float[expected.Length];
        var coupledSecondMoment = new float[expected.Length];

        optimizer.Step();
        ApplyAdamReference(expected, [0.1f, 0.2f, -0.4f], firstMoment, secondMoment, learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, step: 1, weightDecay: 0.2f, gradientClip: 0f, decoupledWeightDecay: true);
        ApplyAdamReference(coupledDecay, [0.1f, 0.2f, -0.4f], coupledFirstMoment, coupledSecondMoment, learningRate: 0.03f, beta1: 0.7f, beta2: 0.8f, epsilon: 1e-5f, step: 1, weightDecay: 0.2f, gradientClip: 0f, decoupledWeightDecay: false);

        var actual = parameter.Value.Buffer.ToArray();
        AssertClose(expected, actual, tolerance: 1e-5f);
        Assert.Contains(Enumerable.Range(0, actual.Length), i => MathF.Abs(actual[i] - coupledDecay[i]) > 1e-4f);
        Assert.Equal(DispatchPath.TypedEasyGpu, NnDispatchTrace.LastPath);
    }

    [Fact]
    public void OptimizerRejectsUnsupportedNonFloatParameters()
    {
        using var parameter = new Parameter<int>(
            "indices",
            new Tensor<int>(new TensorShape(3), GPU.CreateBuffer<int>([1, 2, 3]), requiresGrad: true),
            new Tensor<int>(new TensorShape(3), GPU.CreateBuffer<int>(3)));

        var ex = Assert.Throws<NotSupportedException>(() => new SGD([parameter]));

        Assert.Contains("Parameter<float>", ex.Message);
        Assert.Contains("indices", ex.Message);
    }

    [Fact]
    public void OptimizerRejectsDuplicateParameterNames()
    {
        using var first = new Linear(1, 1);
        using var second = new Linear(1, 1);

        var ex = Assert.Throws<ArgumentException>(() => new SGD(first.Parameters.Concat(second.Parameters)));

        Assert.Contains("Duplicate parameter name 'weight'", ex.Message);
    }

    [Fact]
    public void CheckpointRoundTripsFloatParametersByName()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-nn-{Guid.NewGuid():N}.ckpt");
        using var source = new Linear(2, 2);
        using var target = new Linear(2, 2);
        source.Weight.Value.Buffer.Upload([1, 2, 3, 4]);
        source.Bias.Value.Buffer.Upload([5, 6]);

        try
        {
            Checkpoint.Save(path, source.Parameters);
            Checkpoint.Load(path, target.Parameters);

            Assert.Equal([1, 2, 3, 4], target.Weight.Value.Buffer.ToArray());
            Assert.Equal([5, 6], target.Bias.Value.Buffer.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CheckpointRoundTripsSequentialParametersByQualifiedName()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-nn-{Guid.NewGuid():N}.ckpt");
        using var source = new Sequential(new Linear(1, 1), new Linear(1, 1));
        using var target = new Sequential(new Linear(1, 1), new Linear(1, 1));
        var sourceFirst = (Linear)source.Modules[0];
        var sourceSecond = (Linear)source.Modules[1];
        var targetFirst = (Linear)target.Modules[0];
        var targetSecond = (Linear)target.Modules[1];
        sourceFirst.Weight.Value.Buffer.Upload([1]);
        sourceFirst.Bias.Value.Buffer.Upload([2]);
        sourceSecond.Weight.Value.Buffer.Upload([3]);
        sourceSecond.Bias.Value.Buffer.Upload([4]);

        try
        {
            Checkpoint.Save(path, source.Parameters);
            Checkpoint.Load(path, target.Parameters);

            Assert.Equal([1], targetFirst.Weight.Value.Buffer.ToArray());
            Assert.Equal([2], targetFirst.Bias.Value.Buffer.ToArray());
            Assert.Equal([3], targetSecond.Weight.Value.Buffer.ToArray());
            Assert.Equal([4], targetSecond.Bias.Value.Buffer.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void CheckpointRejectsDuplicateParameterNames()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-nn-{Guid.NewGuid():N}.ckpt");
        using var first = new Linear(1, 1);
        using var second = new Linear(1, 1);

        try
        {
            var ex = Assert.Throws<ArgumentException>(() => Checkpoint.Save(path, first.Parameters.Concat(second.Parameters)));

            Assert.Contains("Duplicate parameter name 'weight'", ex.Message);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static float[] Take(float[] values, int start, int count)
    {
        var slice = new float[count];
        Array.Copy(values, start, slice, 0, count);
        return slice;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-5f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.InRange(MathF.Abs(actual[i] - expected[i]), 0, tolerance);
        }
    }

    private static void AssertRowsSumToOne(float[] values, int rows, int columns)
    {
        for (var row = 0; row < rows; row++)
        {
            var sum = 0f;
            for (var column = 0; column < columns; column++)
            {
                sum += values[(row * columns) + column];
            }

            Assert.InRange(MathF.Abs(sum - 1f), 0, 1e-5f);
        }
    }

    private static Parameter<float> FloatParameter(string name, float[] value, float[] gradient)
    {
        var shape = new TensorShape(value.Length);
        return new Parameter<float>(
            name,
            new Tensor<float>(shape, GPU.CreateBuffer<float>(value), requiresGrad: true),
            new Tensor<float>(shape, GPU.CreateBuffer<float>(gradient)));
    }

    private static float[] Sequence(int count, Func<int, float> selector)
    {
        var values = new float[count];
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = selector(i);
        }

        return values;
    }

    private static void ApplySgdReference(float[] value, float[] gradient, float learningRate, float weightDecay)
    {
        for (var i = 0; i < value.Length; i++)
        {
            var grad = gradient[i] + (weightDecay * value[i]);
            value[i] -= learningRate * grad;
        }
    }

    private static void ApplyMomentumReference(float[] value, float[] gradient, float[] momentum, float learningRate, float momentumFactor, float weightDecay)
    {
        for (var i = 0; i < value.Length; i++)
        {
            var grad = gradient[i] + (weightDecay * value[i]);
            momentum[i] = (momentumFactor * momentum[i]) + grad;
            value[i] -= learningRate * momentum[i];
        }
    }

    private static void ApplyRmsPropReference(float[] value, float[] gradient, float[] squareAverage, float learningRate, float alpha, float epsilon, float weightDecay)
    {
        for (var i = 0; i < value.Length; i++)
        {
            var grad = gradient[i] + (weightDecay * value[i]);
            squareAverage[i] = (alpha * squareAverage[i]) + ((1f - alpha) * grad * grad);
            value[i] -= learningRate * grad / (MathF.Sqrt(squareAverage[i]) + epsilon);
        }
    }

    private static void ApplyAdamReference(
        float[] value,
        float[] gradient,
        float[] firstMoment,
        float[] secondMoment,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        int step,
        float weightDecay,
        float gradientClip,
        bool decoupledWeightDecay)
    {
        for (var i = 0; i < value.Length; i++)
        {
            var current = value[i];
            var grad = gradient[i];
            if (!decoupledWeightDecay)
            {
                grad += weightDecay * current;
            }

            if (gradientClip > 0f)
            {
                grad = System.Math.Clamp(grad, -gradientClip, gradientClip);
            }

            firstMoment[i] = (beta1 * firstMoment[i]) + ((1f - beta1) * grad);
            secondMoment[i] = (beta2 * secondMoment[i]) + ((1f - beta2) * grad * grad);
            var correctedM = firstMoment[i] / (1f - MathF.Pow(beta1, step));
            var correctedV = secondMoment[i] / (1f - MathF.Pow(beta2, step));
            var next = current - (learningRate * correctedM / (MathF.Sqrt(correctedV) + epsilon));
            if (decoupledWeightDecay)
            {
                next -= learningRate * weightDecay * current;
            }

            value[i] = next;
        }
    }

    private static float[] ReferenceLogSoftmax(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (var row = 0; row < rows; row++)
        {
            var offset = row * columns;
            var max = values[offset];
            for (var column = 1; column < columns; column++)
            {
                max = MathF.Max(max, values[offset + column]);
            }

            var sumExp = 0f;
            for (var column = 0; column < columns; column++)
            {
                sumExp += MathF.Exp(values[offset + column] - max);
            }

            var logSum = MathF.Log(sumExp);
            for (var column = 0; column < columns; column++)
            {
                result[offset + column] = values[offset + column] - max - logSum;
            }
        }

        return result;
    }

    private static float ReferenceCrossEntropyFromLogits(float[] logits, int[] labels, int batch, int classes)
    {
        var logSoftmax = ReferenceLogSoftmax(logits, batch, classes);
        var sum = 0f;
        for (var row = 0; row < batch; row++)
        {
            sum -= logSoftmax[(row * classes) + labels[row]];
        }

        return sum / batch;
    }

    private static (float[] Value, float[] FirstMoment, float[] SecondMoment) AdamWReference(
        float[] value,
        float[] gradient,
        float learningRate,
        float beta1,
        float beta2,
        float epsilon,
        float weightDecay,
        int step,
        float[] firstMoment,
        float[] secondMoment)
    {
        var nextValue = new float[value.Length];
        var nextFirstMoment = new float[value.Length];
        var nextSecondMoment = new float[value.Length];
        for (var i = 0; i < value.Length; i++)
        {
            var m = (beta1 * firstMoment[i]) + ((1f - beta1) * gradient[i]);
            var v = (beta2 * secondMoment[i]) + ((1f - beta2) * gradient[i] * gradient[i]);
            var correctedM = m / (1f - MathF.Pow(beta1, step));
            var correctedV = v / (1f - MathF.Pow(beta2, step));
            nextValue[i] = value[i] - (learningRate * correctedM / (MathF.Sqrt(correctedV) + epsilon)) - (learningRate * weightDecay * value[i]);
            nextFirstMoment[i] = m;
            nextSecondMoment[i] = v;
        }

        return (nextValue, nextFirstMoment, nextSecondMoment);
    }
}
