using Feather.AD;
using Feather.Math;
using Feather.NN;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.Integration.Tests;

public class NNTrainingIntegrationTests
{
    [Fact]
    public void NNTrainingLinearRegressionWithGpuADAndSgdDecreasesMeanLoss()
    {
        using var model = new Linear(1, 1);
        model.Weight.Value.Buffer.Upload([0f]);
        model.Bias.Value.Buffer.Upload([0f]);

        using var x = GPU.CreateBuffer<float>([-2f, -1f, 0f, 1f, 2f]);
        using var y = GPU.CreateBuffer<float>([-3f, -1f, 1f, 3f, 5f]);
        using var loss = GPU.CreateBuffer<float>(x.Length);
        using var ad = GPU.CreateADKernel(new NNLinearRegressionMeanLossKernel(
            model.Weight.Value.AsReadWriteBuffer(),
            model.Bias.Value.AsReadWriteBuffer(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            loss.AsReadWrite(),
            new Uniform<float>(1f / x.Length)));
        var optimizer = new SGD(model.Parameters, learningRate: 0.08f);

        var initialLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        var lastLoss = initialLoss;
        for (var step = 0; step < 80; step++)
        {
            lastLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        }

        var weight = model.Weight.Value.Buffer.ToArray()[0];
        var bias = model.Bias.Value.Buffer.ToArray()[0];
        Assert.True(lastLoss < initialLoss * 0.05f, $"Expected loss to decrease substantially, initial={initialLoss}, final={lastLoss}.");
        Assert.InRange(weight, 1.85f, 2.15f);
        Assert.InRange(bias, 0.85f, 1.15f);
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
        Assert.False(ad.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void TrainingStepRunsModuleOwnedParametersThroughDeviceAdHandoff()
    {
        using var model = new Linear(1, 1);
        model.Weight.Value.Buffer.Upload([0f]);
        model.Bias.Value.Buffer.Upload([0f]);

        using var x = GPU.CreateBuffer<float>([-2f, -1f, 0f, 1f, 2f]);
        using var y = GPU.CreateBuffer<float>([-3f, -1f, 1f, 3f, 5f]);
        using var loss = GPU.CreateBuffer<float>(x.Length);
        var optimizer = new SGD(model.Parameters, learningRate: 0.08f);
        using var step = TrainingStep<NNLinearRegressionMeanLossKernel>.Create(
            new NNLinearRegressionMeanLossKernel(
                model.Weight.Value.AsReadWriteBuffer(),
                model.Bias.Value.AsReadWriteBuffer(),
                x.AsReadOnly(),
                y.AsReadOnly(),
                loss.AsReadWrite(),
                new Uniform<float>(1f / x.Length)),
            model.Parameters,
            optimizer,
            loss,
            x.Length);

        var initialLoss = step.Run();
        var lastLoss = initialLoss;
        for (var i = 0; i < 80; i++)
        {
            lastLoss = step.Run();
        }

        Assert.True(lastLoss < initialLoss * 0.05f, $"Expected TrainingStep loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, step.LastDispatchPath);
        Assert.False(step.GradientsMaterialized);
        Assert.Equal(lastLoss, step.LastLoss);
    }

    [Fact]
    public void NNTrainingTinyReluMlpWithGpuADAndAdamDecreasesLoss()
    {
        using var hiddenWeight = FloatParameter("hiddenWeight", "hiddenWeight", [1f, -1f]);
        using var hiddenBias = FloatParameter("hiddenBias", "hiddenBias", [0f, 0f]);
        using var outputWeight = FloatParameter("outputWeight", "outputWeight", [0.15f, 0.15f]);
        using var outputBias = FloatParameter("outputBias", "outputBias", [0f]);
        IParameter[] parameters = [hiddenWeight, hiddenBias, outputWeight, outputBias];

        using var x = GPU.CreateBuffer<float>([-2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f]);
        using var y = GPU.CreateBuffer<float>(ComputeAbsoluteValueTargets(x.ToArray()));
        using var loss = GPU.CreateBuffer<float>(x.Length);
        using var ad = GPU.CreateADKernel(new NNReluMlpMeanLossKernel(
            hiddenWeight.Value.AsReadWriteBuffer(),
            hiddenBias.Value.AsReadWriteBuffer(),
            outputWeight.Value.AsReadWriteBuffer(),
            outputBias.Value.AsReadWriteBuffer(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            loss.AsReadWrite(),
            new Uniform<float>(1f / x.Length)));
        var optimizer = new Adam(parameters, learningRate: 0.04f);

        var initialLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        var lastLoss = initialLoss;
        for (var step = 0; step < 90; step++)
        {
            lastLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        }

        Assert.True(lastLoss < initialLoss * 0.2f, $"Expected nonlinear loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
        Assert.False(ad.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void SequentialReluMlpTrainingStepWithAdamUsesModuleOwnedParameters()
    {
        using var model = new Sequential(new Linear(1, 2), new ReLU(), new Linear(2, 1));
        var hidden = (Linear)model.Modules[0];
        var output = (Linear)model.Modules[2];
        hidden.Weight.Value.Buffer.Upload([1f, -1f]);
        hidden.Bias.Value.Buffer.Upload([0f, 0f]);
        output.Weight.Value.Buffer.Upload([0.15f, 0.15f]);
        output.Bias.Value.Buffer.Upload([0f]);

        using var x = GPU.CreateBuffer<float>([-2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f]);
        using var y = GPU.CreateBuffer<float>(ComputeAbsoluteValueTargets(x.ToArray()));
        using var loss = GPU.CreateBuffer<float>(x.Length);
        using var optimizer = new Adam(model.Parameters, learningRate: 0.04f);
        using var step = TrainingStep<NNSequentialReluMlpMeanLossKernel>.Create(
            new NNSequentialReluMlpMeanLossKernel(
                hidden.Weight.Value.AsReadWriteBuffer(),
                hidden.Bias.Value.AsReadWriteBuffer(),
                output.Weight.Value.AsReadWriteBuffer(),
                output.Bias.Value.AsReadWriteBuffer(),
                x.AsReadOnly(),
                y.AsReadOnly(),
                loss.AsReadWrite(),
                new Uniform<float>(1f / x.Length)),
            model.Parameters,
            optimizer,
            loss,
            x.Length);

        var initialLoss = step.Run();
        var lastLoss = initialLoss;
        for (var i = 0; i < 90; i++)
        {
            lastLoss = step.Run();
        }

        Assert.True(lastLoss < initialLoss * 0.2f, $"Expected Sequential MLP loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.Contains(model.Parameters, parameter => parameter.FullName == "linear0.weight");
        Assert.Contains(model.Parameters, parameter => parameter.FullName == "linear2.weight");
        Assert.Equal(DispatchPath.TypedEasyGpu, step.LastDispatchPath);
        Assert.False(step.GradientsMaterialized);
        Assert.Equal(lastLoss, step.LastLoss);
    }

    [Fact]
    public void NNTrainingOptimizerHandoffSupportsRmsPropAndZeroGrad()
    {
        using var parameter = FloatParameter("weight", "weight", [0f]);
        using var x = GPU.CreateBuffer<float>([2f]);
        using var y = GPU.CreateBuffer<float>([4f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NNSingleWeightMeanLossKernel(
            parameter.Value.AsReadWriteBuffer(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            loss.AsReadWrite()));
        var optimizer = new RMSProp([parameter], learningRate: 0.05f, alpha: 0.9f);

        ad.Backward(1);
        optimizer.Step(ad);
        var afterFirstStep = parameter.Value.Buffer.ToArray()[0];
        var firstGradient = parameter.Gradient.Buffer.ToArray()[0];
        ad.Backward(1);
        optimizer.Step(ad);
        var afterSecondStep = parameter.Value.Buffer.ToArray()[0];

        Assert.True(firstGradient < 0f);
        Assert.True(afterFirstStep > 0f);
        Assert.True(afterSecondStep > afterFirstStep);
        optimizer.ZeroGrad();
        Assert.Equal([0f], parameter.Gradient.Buffer.ToArray());
        Assert.False(ad.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void EmbeddingGradientAccumulatesRowsThroughNativeAd()
    {
        using var embedding = new Embedding(vocabularySize: 3, embeddingSize: 2);
        embedding.Weight.Value.Buffer.Upload([1f, 2f, 3f, 4f, 5f, 6f]);
        using var indices = GPU.CreateBuffer<int>([0, 2, 0]);
        using var loss = GPU.CreateBuffer<float>(3);
        using var ad = GPU.CreateADKernel(new NNEmbeddingGatherLossKernel(
            embedding.Weight.Value.AsReadWriteBuffer(),
            indices.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(3);
        ad.CopyGradientToBuffer("embedding", embedding.Weight.Gradient.Buffer);

        Assert.Equal([2f, 2f, 0f, 0f, 1f, 1f], embedding.Weight.Gradient.Buffer.ToArray());
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
        Assert.False(ad.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void NNTrainingBinaryClassificationWithGpuADAndAdamWDecreasesCrossEntropy()
    {
        using var weight = FloatParameter("weight", "weight", [0f]);
        using var bias = FloatParameter("bias", "bias", [0f]);
        IParameter[] parameters = [weight, bias];

        using var x = GPU.CreateBuffer<float>([-2f, -1f, -0.25f, 0.25f, 1f, 2f]);
        using var labels = GPU.CreateBuffer<float>([0f, 0f, 0f, 1f, 1f, 1f]);
        using var loss = GPU.CreateBuffer<float>(x.Length);
        using var ad = GPU.CreateADKernel(new NNBinaryClassificationCrossEntropyKernel(
            weight.Value.AsReadWriteBuffer(),
            bias.Value.AsReadWriteBuffer(),
            x.AsReadOnly(),
            labels.AsReadOnly(),
            loss.AsReadWrite(),
            new Uniform<float>(1f / x.Length)));
        var optimizer = new AdamW(parameters, learningRate: 0.05f, weightDecay: 0.001f);

        var initialLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        var lastLoss = initialLoss;
        for (var step = 0; step < 120; step++)
        {
            lastLoss = RunBackwardAndStep(ad, x.Length, loss, optimizer);
        }

        var learnedWeight = weight.Value.Buffer.ToArray()[0];
        Assert.True(lastLoss < initialLoss * 0.35f, $"Expected cross entropy to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.True(learnedWeight > 0f, $"Expected positive class logit weight to be positive, weight={learnedWeight}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
        Assert.False(ad.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void ModuleBackedBinaryClassifierWithAdamWDecreasesCrossEntropy()
    {
        using var model = new Linear(1, 1);
        model.Weight.Value.Buffer.Upload([0f]);
        model.Bias.Value.Buffer.Upload([0f]);

        using var x = GPU.CreateBuffer<float>([-2f, -1f, -0.25f, 0.25f, 1f, 2f]);
        using var labels = GPU.CreateBuffer<float>([0f, 0f, 0f, 1f, 1f, 1f]);
        using var loss = GPU.CreateBuffer<float>(x.Length);
        using var optimizer = new AdamW(model.Parameters, learningRate: 0.05f, weightDecay: 0.001f);
        using var step = TrainingStep<NNBinaryClassificationCrossEntropyKernel>.Create(
            new NNBinaryClassificationCrossEntropyKernel(
                model.Weight.Value.AsReadWriteBuffer(),
                model.Bias.Value.AsReadWriteBuffer(),
                x.AsReadOnly(),
                labels.AsReadOnly(),
                loss.AsReadWrite(),
                new Uniform<float>(1f / x.Length)),
            model.Parameters,
            optimizer,
            loss,
            x.Length);

        var initialLoss = step.Run();
        var lastLoss = initialLoss;
        for (var i = 0; i < 120; i++)
        {
            lastLoss = step.Run();
        }

        var learnedWeight = model.Weight.Value.Buffer.ToArray()[0];
        Assert.True(lastLoss < initialLoss * 0.35f, $"Expected module classifier loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.True(learnedWeight > 0f, $"Expected positive class logit weight to be positive, weight={learnedWeight}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, step.LastDispatchPath);
        Assert.False(step.GradientsMaterialized);
        Assert.Equal(lastLoss, step.LastLoss);
    }

    [Fact]
    public void SelfAttentionClassifierTrainerWithAdamDecreasesLoss()
    {
        var (features, labels) = CreateAttentionClassificationData(128, seed: 42);
        using var model = new SelfAttentionClassifier(seed: 123);
        using var optimizer = new Adam(model.Parameters, learningRate: 0.03f, beta1: 0.85f, beta2: 0.99f, epsilon: 1e-6f);
        using var trainer = model.CreateTrainer(features, labels, optimizer);

        var initialLoss = trainer.EvaluateLoss();
        var lastLoss = initialLoss;
        for (var step = 0; step < 80; step++)
        {
            lastLoss = trainer.TrainStep();
        }

        Assert.True(lastLoss < initialLoss, $"Expected attention classifier loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, trainer.LastDispatchPath);
        Assert.False(trainer.GradientsMaterialized);
        Assert.Equal(lastLoss, trainer.LastLoss);
    }

    [Fact]
    public void GptLanguageModelTrainerWithTransformerBlockDecreasesTinyBatchLoss()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 11);
        using var optimizer = new Adam(model.Parameters, learningRate: 0.002f, beta1: 0.85f, beta2: 0.99f, epsilon: 1e-6f);
        using var trainer = model.CreateTrainer(batchSize: 4, optimizer);
        var batch = new[]
        {
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 5,
            2, 3, 4, 5, 0,
            3, 4, 5, 0, 1
        };

        var initialLoss = trainer.EvaluateBatch(batch);
        var lastLoss = initialLoss;
        for (var step = 0; step < 25; step++)
        {
            lastLoss = trainer.TrainBatch(batch);
        }

        lastLoss = trainer.EvaluateBatch(batch);

        Assert.True(lastLoss < initialLoss, $"Expected GPT trainer loss to decrease, initial={initialLoss}, final={lastLoss}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, trainer.LastDispatchPath);
        Assert.False(trainer.GradientsMaterialized);
        Assert.Equal(lastLoss, trainer.LastLoss);
    }

    [Fact]
    public void GptEvaluationIsForwardOnlyAndDoesNotPerturbNextTrainStep()
    {
        var batch = new[]
        {
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 5
        };

        using var modelWithEval = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 23);
        using var optimizerWithEval = new SGD(modelWithEval.Parameters, learningRate: 0.001f);
        using var trainerWithEval = modelWithEval.CreateTrainer(batchSize: 2, optimizerWithEval);
        var evalLoss = trainerWithEval.EvaluateBatch(batch);
        Assert.False(trainerWithEval.GradientsMaterialized);
        Assert.Equal(DispatchPath.TypedEasyGpu, trainerWithEval.LastDispatchPath);

        var beforeTrain = SnapshotFloatParameters(modelWithEval);
        var trainAfterEval = trainerWithEval.TrainBatch(batch);
        var afterTrain = SnapshotFloatParameters(modelWithEval);
        Assert.False(trainerWithEval.GradientsMaterialized);
        Assert.True(trainAfterEval > 0f);
        Assert.NotEqual(beforeTrain[0][0], afterTrain[0][0]);

        using var modelWithoutEval = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 23);
        using var optimizerWithoutEval = new SGD(modelWithoutEval.Parameters, learningRate: 0.001f);
        using var trainerWithoutEval = modelWithoutEval.CreateTrainer(batchSize: 2, optimizerWithoutEval);
        var trainWithoutEval = trainerWithoutEval.TrainBatch(batch);
        var afterDirectTrain = SnapshotFloatParameters(modelWithoutEval);

        Assert.InRange(MathF.Abs(trainAfterEval - trainWithoutEval), 0f, 1e-6f);
        AssertParameterSnapshotsClose(afterDirectTrain, afterTrain, tolerance: 2e-6f);
        Assert.True(evalLoss > 0f);
    }

    [Fact]
    public void GptLanguageModelTrainingKernelProducesGradientsForEveryParameterGroup()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 11);
        using var tokens = GPU.CreateBuffer<int>([
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 5
        ]);
        using var workspace = GPU.CreateBuffer<float>(checked(2 * GptLanguageModelTrainer.TrainingWorkspaceElementsPerBatch(model.BlockSize, model.EmbeddingSize, model.HeadCount, model.Block.MlpSize, model.VocabularySize)));
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new GptLanguageModelTrainingKernel(
            tokens.AsReadOnly(),
            workspace.AsReadWrite(),
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
            new Uniform<float>(1f / (2 * model.BlockSize))));

        ad.Backward(2);

        AssertNonZeroGradient(ad, "tokenEmbedding", model.TokenEmbedding.Weight.ElementCount);
        AssertNonZeroGradient(ad, "positionEmbedding", model.PositionalEmbedding.Weight.ElementCount);
        AssertNonZeroGradient(ad, "attentionWeights", model.Block.Attention.Weights.ElementCount);
        AssertNonZeroGradient(ad, "fc1", model.Block.Fc1.ElementCount);
        AssertNonZeroGradient(ad, "fc2", model.Block.Fc2.ElementCount);
        AssertNonZeroGradient(ad, "lmHead", model.LmHead.ElementCount);
        Assert.False(ad.Gradients.HasMaterializedValues);
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
    }

    [Fact]
    public void GptNativeGradientReductionMatchesManagedReadbackAndOptimizerHandoff()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 31);
        var tokenWindows = new[]
        {
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 0
        };
        using var probe = GptGradientProbe.Create(model, tokenWindows);

        probe.Backward();
        var managed = probe.ReadManagedGradient("tokenEmbedding");
        var reduced = probe.ReadGradient("tokenEmbedding", model.TokenEmbedding.Weight.ElementCount);
        AssertClose(managed, reduced, tolerance: 1e-6f);

        var index = 1 * model.EmbeddingSize + 2;
        probe.Dispose();
        using var handoffProbe = GptGradientProbe.Create(model, tokenWindows);
        handoffProbe.Backward();
        reduced = handoffProbe.ReadGradient("tokenEmbedding", model.TokenEmbedding.Weight.ElementCount);
        var before = model.TokenEmbedding.Weight.Value.Buffer.ToArray();
        using var optimizer = new SGD([model.TokenEmbedding.Weight], learningRate: 0.005f);
        optimizer.Step(handoffProbe.AdKernel);
        var after = model.TokenEmbedding.Weight.Value.Buffer.ToArray();

        var expected = before[index] - (0.005f * reduced[index]);
        Assert.InRange(MathF.Abs(after[index] - expected), 0f, 1e-6f);
        Assert.Equal(reduced, model.TokenEmbedding.Weight.Gradient.Buffer.ToArray());
        Assert.False(handoffProbe.AdKernel.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void GptParameterGroupsMapToNativeGradientsAndClippingDoesNotZeroThem()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 37);
        var groups = new[]
        {
            new ParameterGroup("embeddings", model.TokenEmbedding.Parameters.Concat(model.PositionalEmbedding.Parameters), learningRate: 0.001f),
            new ParameterGroup("block", model.Block.Parameters, learningRate: 0.001f),
            new ParameterGroup("head", [model.LmHead], learningRate: 0.001f)
        };
        using var optimizer = new Adam(groups, learningRate: 0.001f, beta1: 0.85f, beta2: 0.99f, epsilon: 1e-6f, gradientClip: 0.05f);
        Assert.Equal(model.Parameters.Count(), optimizer.Parameters.Count);
        Assert.Equal(model.Parameters.Select(parameter => parameter.FullName), optimizer.Parameters.Select(parameter => parameter.FullName));

        var tokenWindows = new[]
        {
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 0
        };
        using var probe = GptGradientProbe.Create(model, tokenWindows);
        probe.Backward();
        optimizer.Step(probe.AdKernel);

        Assert.Contains(model.Parameters.OfType<Parameter<float>>(), parameter =>
            parameter.Gradient.Buffer.ToArray().Any(value => MathF.Abs(value) > 1e-7f));
        Assert.Contains(model.TokenEmbedding.Weight.Gradient.Buffer.ToArray(), value => MathF.Abs(value) > 1e-7f);
        Assert.Contains(model.LmHead.Gradient.Buffer.ToArray(), value => MathF.Abs(value) > 1e-7f);
        Assert.False(probe.AdKernel.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void GptLanguageModelTrainingKernelGradientsMatchFiniteDifferences()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 17);
        var tokenWindows = new[]
        {
            0, 1, 2, 3, 4,
            1, 2, 3, 4, 0
        };
        using var probe = GptGradientProbe.Create(model, tokenWindows);

        probe.Backward();
        var tokenEmbedding = probe.ReadGradient("tokenEmbedding", model.TokenEmbedding.Weight.ElementCount);
        var positionEmbedding = probe.ReadGradient("positionEmbedding", model.PositionalEmbedding.Weight.ElementCount);
        var attentionWeights = probe.ReadGradient("attentionWeights", model.Block.Attention.Weights.ElementCount);
        var fc1 = probe.ReadGradient("fc1", model.Block.Fc1.ElementCount);
        var fc2 = probe.ReadGradient("fc2", model.Block.Fc2.ElementCount);
        var lmHead = probe.ReadGradient("lmHead", model.LmHead.ElementCount);

        const float epsilon = 2e-2f;
        AssertGradientMatchesFiniteDifference(
            "token embedding used row",
            probe.FiniteDifference(model.TokenEmbedding.Weight, index: 1 * model.EmbeddingSize + 2, epsilon),
            tokenEmbedding[1 * model.EmbeddingSize + 2]);
        AssertGradientMatchesFiniteDifference(
            "token embedding unused row",
            probe.FiniteDifference(model.TokenEmbedding.Weight, index: 5 * model.EmbeddingSize + 1, epsilon),
            tokenEmbedding[5 * model.EmbeddingSize + 1],
            absoluteTolerance: 2e-4f,
            relativeTolerance: 0.05f);
        AssertGradientMatchesFiniteDifference(
            "position embedding first position",
            probe.FiniteDifference(model.PositionalEmbedding.Weight, index: 0, epsilon),
            positionEmbedding[0]);
        AssertGradientMatchesFiniteDifference(
            "position embedding final position",
            probe.FiniteDifference(model.PositionalEmbedding.Weight, index: (3 * model.EmbeddingSize) + 1, epsilon),
            positionEmbedding[(3 * model.EmbeddingSize) + 1]);

        var projectionStride = model.EmbeddingSize * model.EmbeddingSize;
        AssertGradientMatchesFiniteDifference(
            "attention Q slice",
            probe.FiniteDifference(model.Block.Attention.Weights, index: 0 * projectionStride + 2, epsilon),
            attentionWeights[0 * projectionStride + 2]);
        AssertGradientMatchesFiniteDifference(
            "attention K slice",
            probe.FiniteDifference(model.Block.Attention.Weights, index: 1 * projectionStride + 5, epsilon),
            attentionWeights[1 * projectionStride + 5]);
        AssertGradientMatchesFiniteDifference(
            "attention V slice",
            probe.FiniteDifference(model.Block.Attention.Weights, index: 2 * projectionStride + 10, epsilon),
            attentionWeights[2 * projectionStride + 10]);
        AssertGradientMatchesFiniteDifference(
            "attention O slice",
            probe.FiniteDifference(model.Block.Attention.Weights, index: 3 * projectionStride + 15, epsilon),
            attentionWeights[3 * projectionStride + 15]);

        AssertGradientMatchesFiniteDifference(
            "fc1 weight",
            probe.FiniteDifference(model.Block.Fc1, index: (2 * model.EmbeddingSize) + 1, epsilon),
            fc1[(2 * model.EmbeddingSize) + 1]);
        AssertGradientMatchesFiniteDifference(
            "fc2 weight",
            probe.FiniteDifference(model.Block.Fc2, index: model.Block.MlpSize + 3, epsilon),
            fc2[model.Block.MlpSize + 3]);
        AssertGradientMatchesFiniteDifference(
            "lm head target row",
            probe.FiniteDifference(model.LmHead, index: (1 * model.EmbeddingSize) + 0, epsilon),
            lmHead[(1 * model.EmbeddingSize) + 0]);
        AssertGradientMatchesFiniteDifference(
            "lm head non-target row",
            probe.FiniteDifference(model.LmHead, index: (5 * model.EmbeddingSize) + 2, epsilon),
            lmHead[(5 * model.EmbeddingSize) + 2]);

        Assert.Equal(DispatchPath.TypedEasyGpu, probe.LastDispatchPath);
    }

    [Fact]
    public void GptLanguageModelTrainingKernelLossMatchesHostInference()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 11);
        var tokenWindow = new[] { 0, 1, 2, 3, 4 };
        using var tokens = GPU.CreateBuffer<int>(tokenWindow);
        using var workspace = GPU.CreateBuffer<float>(GptLanguageModelTrainer.TrainingWorkspaceElementsPerBatch(model.BlockSize, model.EmbeddingSize, model.HeadCount, model.Block.MlpSize, model.VocabularySize));
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new GptLanguageModelTrainingKernel(
            tokens.AsReadOnly(),
            workspace.AsReadWrite(),
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
            new Uniform<float>(1f)));

        ad.Backward(1);

        var expectedLoss = GptAllPositionHostLoss(model, tokenWindow);
        var kernelLoss = loss.ToArray()[0];
        Assert.InRange(MathF.Abs(kernelLoss - expectedLoss), 0f, 1e-4f);
        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
    }

    [Fact]
    public void GptLanguageModelTrainingKernelBatchLossMatchesHostInferenceWithoutWorkspaceOverlap()
    {
        using var model = new GptLanguageModel(vocabularySize: 6, blockSize: 4, embeddingSize: 4, headCount: 2, seed: 11);
        var tokenWindows = new[]
        {
            0, 1, 2, 3, 4,
            3, 2, 1, 0, 5,
            4, 5, 0, 1, 2
        };
        using var tokens = GPU.CreateBuffer<int>(tokenWindows);
        using var workspace = GPU.CreateBuffer<float>(checked(3 * GptLanguageModelTrainer.TrainingWorkspaceElementsPerBatch(model.BlockSize, model.EmbeddingSize, model.HeadCount, model.Block.MlpSize, model.VocabularySize)));
        using var loss = GPU.CreateBuffer<float>(3);
        using var ad = GPU.CreateADKernel(new GptLanguageModelTrainingKernel(
            tokens.AsReadOnly(),
            workspace.AsReadWrite(),
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
            new Uniform<float>(1f)));

        ad.Backward(3);

        var actualLoss = loss.ToArray();
        for (var row = 0; row < 3; row++)
        {
            var window = tokenWindows[(row * (model.BlockSize + 1))..((row + 1) * (model.BlockSize + 1))];
            var expectedLoss = GptAllPositionHostLoss(model, window);
            Assert.InRange(MathF.Abs(actualLoss[row] - expectedLoss), 0f, 1e-4f);
        }

        Assert.Equal(DispatchPath.TypedEasyGpu, ad.LastDispatchPath);
    }

    [Fact]
    public void OptimizerStepAdKernelReportsMissingMismatchedAmbiguousAndUnsupportedNativeGradients()
    {
        using var missingParameter = FloatParameter("missing", "missing", [0f]);
        using var missingX = GPU.CreateBuffer<float>([2f]);
        using var missingY = GPU.CreateBuffer<float>([4f]);
        using var missingLoss = GPU.CreateBuffer<float>(1);
        using var missingAd = GPU.CreateADKernel(new NNSingleWeightMeanLossKernel(
            missingParameter.Value.AsReadWriteBuffer(),
            missingX.AsReadOnly(),
            missingY.AsReadOnly(),
            missingLoss.AsReadWrite()));
        using var missingOptimizer = new SGD([missingParameter], learningRate: 0.1f);
        missingAd.Backward(1);

        var missing = Assert.Throws<ArgumentException>(() => missingOptimizer.Step(missingAd));
        Assert.Contains("No native AD gradient matched", missing.Message);
        Assert.False(missingAd.Gradients.HasMaterializedValues);

        using var wrongShapeParameter = FloatParameter("weight", "weight", [0f]);
        using var wrongShapeWeight = GPU.CreateBuffer<float>([0f, 0f]);
        using var wrongShapeX = GPU.CreateBuffer<float>([2f]);
        using var wrongShapeY = GPU.CreateBuffer<float>([4f]);
        using var wrongShapeLoss = GPU.CreateBuffer<float>(1);
        using var wrongShapeAd = GPU.CreateADKernel(new NNSingleWeightMeanLossKernel(
            wrongShapeWeight.AsReadWrite(),
            wrongShapeX.AsReadOnly(),
            wrongShapeY.AsReadOnly(),
            wrongShapeLoss.AsReadWrite()));
        using var wrongShapeOptimizer = new SGD([wrongShapeParameter], learningRate: 0.1f);
        wrongShapeAd.Backward(1);

        var wrongShape = Assert.Throws<ArgumentException>(() => wrongShapeOptimizer.Step(wrongShapeAd));
        Assert.Contains("destination buffer expects 1", wrongShape.Message);
        Assert.False(wrongShapeAd.Gradients.HasMaterializedValues);

        using var ambiguousParameter = FloatParameter("weight", "weight", [0f]);
        ambiguousParameter.AddGradientAlias("aliasWeight");
        using var ambiguousX = GPU.CreateBuffer<float>([2f]);
        using var ambiguousLoss = GPU.CreateBuffer<float>(1);
        using var ambiguousAd = GPU.CreateADKernel(new NNAmbiguousGradientAliasKernel(
            ambiguousParameter.Value.AsReadWriteBuffer(),
            ambiguousParameter.Value.AsReadWriteBuffer(),
            ambiguousX.AsReadOnly(),
            ambiguousLoss.AsReadWrite()));
        using var ambiguousOptimizer = new SGD([ambiguousParameter], learningRate: 0.1f);
        ambiguousAd.Backward(1);

        var ambiguous = Assert.Throws<ArgumentException>(() => ambiguousOptimizer.Step(ambiguousAd));
        Assert.Contains("Multiple native AD gradients matched", ambiguous.Message);
        Assert.False(ambiguousAd.Gradients.HasMaterializedValues);

    }

    [Fact]
    public void NNCheckpointLoadResumesGpuAdTraining()
    {
        var path = Path.Combine(Path.GetTempPath(), $"feather-nn-resume-{Guid.NewGuid():N}.ckpt");
        using var source = new Linear(1, 1);
        source.Weight.Value.Buffer.Upload([0f]);
        source.Bias.Value.Buffer.Upload([0f]);

        using var x = GPU.CreateBuffer<float>([-2f, -1f, 0f, 1f, 2f]);
        using var y = GPU.CreateBuffer<float>([-3f, -1f, 1f, 3f, 5f]);
        using var sourceLoss = GPU.CreateBuffer<float>(x.Length);
        using var sourceAd = GPU.CreateADKernel(new NNLinearRegressionMeanLossKernel(
            source.Weight.Value.AsReadWriteBuffer(),
            source.Bias.Value.AsReadWriteBuffer(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            sourceLoss.AsReadWrite(),
            new Uniform<float>(1f / x.Length)));
        var sourceOptimizer = new SGD(source.Parameters, learningRate: 0.08f);

        var initialLoss = RunBackwardAndStep(sourceAd, x.Length, sourceLoss, sourceOptimizer);
        var checkpointLoss = initialLoss;
        for (var step = 0; step < 20; step++)
        {
            checkpointLoss = RunBackwardAndStep(sourceAd, x.Length, sourceLoss, sourceOptimizer);
        }

        try
        {
            Checkpoint.Save(path, source.Parameters);

            using var resumed = new Linear(1, 1);
            Checkpoint.Load(path, resumed.Parameters);
            using var resumedLossBuffer = GPU.CreateBuffer<float>(x.Length);
            using var resumedAd = GPU.CreateADKernel(new NNLinearRegressionMeanLossKernel(
                resumed.Weight.Value.AsReadWriteBuffer(),
                resumed.Bias.Value.AsReadWriteBuffer(),
                x.AsReadOnly(),
                y.AsReadOnly(),
                resumedLossBuffer.AsReadWrite(),
                new Uniform<float>(1f / x.Length)));
            var resumedOptimizer = new SGD(resumed.Parameters, learningRate: 0.08f);

            var resumeStartLoss = RunBackwardAndStep(resumedAd, x.Length, resumedLossBuffer, resumedOptimizer);
            var resumedLoss = resumeStartLoss;
            for (var step = 0; step < 30; step++)
            {
                resumedLoss = RunBackwardAndStep(resumedAd, x.Length, resumedLossBuffer, resumedOptimizer);
            }

            Assert.True(checkpointLoss < initialLoss, $"Expected checkpoint source training to reduce loss, initial={initialLoss}, checkpoint={checkpointLoss}.");
            Assert.InRange(MathF.Abs(resumeStartLoss - checkpointLoss), 0, 0.1f);
            Assert.True(resumedLoss < resumeStartLoss, $"Expected resumed training to reduce loss, start={resumeStartLoss}, final={resumedLoss}.");
            Assert.Equal(DispatchPath.TypedEasyGpu, resumedAd.LastDispatchPath);
            Assert.False(resumedAd.Gradients.HasMaterializedValues);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static float RunBackwardAndStep<TKernel>(
        GpuADKernel<TKernel> ad,
        int count,
        GpuBuffer<float> loss,
        Optimizer optimizer)
        where TKernel : struct, IKernel1D, Feather.Interop.IGeneratedKernel<TKernel>
    {
        ad.Backward(count);
        var meanLoss = loss.ToArray().Sum();
        optimizer.Step(ad);
        return meanLoss;
    }

    private static void AssertNonZeroGradient<TKernel>(GpuADKernel<TKernel> ad, string name, int expectedLength)
        where TKernel : struct, IKernel1D, Feather.Interop.IGeneratedKernel<TKernel>
    {
        using var gradient = GPU.CreateBuffer<float>(expectedLength);
        ad.CopyGradientToBuffer(name, gradient);
        var values = gradient.ToArray();
        Assert.Equal(expectedLength, values.Length);
        Assert.Contains(values, value => MathF.Abs(value) > 1e-7f);
    }

    private static void AssertGradientMatchesFiniteDifference(
        string label,
        float finiteDifference,
        float nativeGradient,
        float absoluteTolerance = 8e-3f,
        float relativeTolerance = 0.12f)
    {
        var tolerance = MathF.Max(absoluteTolerance, MathF.Abs(finiteDifference) * relativeTolerance);
        var error = MathF.Abs(nativeGradient - finiteDifference);
        Assert.True(
            error <= tolerance,
            $"{label}: native gradient {nativeGradient:R} did not match finite difference {finiteDifference:R}; error={error:R}, tolerance={tolerance:R}.");
    }

    private static float[][] SnapshotFloatParameters(Module model)
        => model.Parameters
            .OfType<Parameter<float>>()
            .Select(parameter => parameter.Value.Buffer.ToArray())
            .ToArray();

    private static void AssertParameterSnapshotsClose(float[][] expected, float[][] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var p = 0; p < expected.Length; p++)
        {
            AssertClose(expected[p], actual[p], tolerance);
        }
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, tolerance);
        }
    }

    private static float CrossEntropyFromLogits(float[] logits, int target)
    {
        var max = logits.Max();
        var sum = 0f;
        foreach (var logit in logits)
        {
            sum += MathF.Exp(logit - max);
        }

        return MathF.Log(sum) + max - logits[target];
    }

    private static float GptAllPositionHostLoss(GptLanguageModel model, int[] tokenWindow)
    {
        Assert.Equal(model.BlockSize + 1, tokenWindow.Length);
        var tokenEmbedding = model.TokenEmbedding.Weight.Value.Buffer.ToArray();
        var positionEmbedding = model.PositionalEmbedding.Weight.Value.Buffer.ToArray();
        var x = new float[model.BlockSize * model.EmbeddingSize];
        for (var pos = 0; pos < model.BlockSize; pos++)
        {
            var token = tokenWindow[pos];
            for (var d = 0; d < model.EmbeddingSize; d++)
            {
                x[(pos * model.EmbeddingSize) + d] =
                    tokenEmbedding[(token * model.EmbeddingSize) + d] +
                    positionEmbedding[(pos * model.EmbeddingSize) + d];
            }
        }

        var block = model.Block.RunHost(x, model.BlockSize);
        var lmHead = model.LmHead.Value.Buffer.ToArray();
        var logits = new float[model.VocabularySize];
        var totalLoss = 0f;
        for (var pos = 0; pos < model.BlockSize; pos++)
        {
            Array.Clear(logits);
            var offset = pos * model.EmbeddingSize;
            for (var token = 0; token < model.VocabularySize; token++)
            {
                var value = 0f;
                for (var d = 0; d < model.EmbeddingSize; d++)
                {
                    value += lmHead[(token * model.EmbeddingSize) + d] * block[offset + d];
                }

                logits[token] = value;
            }

            totalLoss += CrossEntropyFromLogits(logits, tokenWindow[pos + 1]);
        }

        return totalLoss;
    }

    private static Parameter<float> FloatParameter(string name, string gradientAlias, float[] values)
    {
        var shape = new TensorShape(values.Length);
        var parameter = new Parameter<float>(
            name,
            new Tensor<float>(shape, GPU.CreateBuffer<float>(values), requiresGrad: true),
            new Tensor<float>(shape, GPU.CreateBuffer<float>(values.Length)));
        if (!string.Equals(name, gradientAlias, StringComparison.Ordinal))
        {
            parameter.AddGradientAlias(gradientAlias);
        }

        return parameter;
    }

    private static float[] ComputeAbsoluteValueTargets(float[] x)
    {
        var y = new float[x.Length];
        for (var i = 0; i < x.Length; i++)
        {
            y[i] = MathF.Abs(x[i]);
        }

        return y;
    }

    private static (float[] Features, float[] Labels) CreateAttentionClassificationData(int count, int seed)
    {
        var random = new Random(seed);
        var features = new float[count * 2];
        var labels = new float[count];
        for (var i = 0; i < count; i++)
        {
            var x0 = NextGaussian(random);
            var x1 = NextGaussian(random);
            features[(i * 2) + 0] = x0;
            features[(i * 2) + 1] = x1;
            labels[i] = x0 * x1 > 0f ? 1f : 0f;
        }

        return (features, labels);
    }

    private static float NextGaussian(Random random)
    {
        var u1 = 1.0 - random.NextDouble();
        var u2 = 1.0 - random.NextDouble();
        return (float)(System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2.0 * System.Math.PI * u2));
    }

    private sealed class GptGradientProbe : IDisposable
    {
        private readonly GptLanguageModel model;
        private readonly GpuBuffer<int> tokens;
        private readonly GpuBuffer<float> workspace;
        private readonly GpuBuffer<float> loss;
        private readonly GpuADKernel<GptLanguageModelTrainingKernel> ad;
        private readonly int batchSize;
        private bool disposed;

        private GptGradientProbe(
            GptLanguageModel model,
            GpuBuffer<int> tokens,
            GpuBuffer<float> workspace,
            GpuBuffer<float> loss,
            GpuADKernel<GptLanguageModelTrainingKernel> ad,
            int batchSize)
        {
            this.model = model;
            this.tokens = tokens;
            this.workspace = workspace;
            this.loss = loss;
            this.ad = ad;
            this.batchSize = batchSize;
        }

        public DispatchPath LastDispatchPath => ad.LastDispatchPath;

        public GpuADKernel<GptLanguageModelTrainingKernel> AdKernel => ad;

        public static GptGradientProbe Create(GptLanguageModel model, int[] tokenWindows)
        {
            var windowSize = model.BlockSize + 1;
            Assert.Equal(0, tokenWindows.Length % windowSize);
            var batchSize = tokenWindows.Length / windowSize;
            var tokens = GPU.CreateBuffer<int>(tokenWindows);
            var workspace = GPU.CreateBuffer<float>(checked(batchSize * GptLanguageModelTrainer.TrainingWorkspaceElementsPerBatch(
                model.BlockSize,
                model.EmbeddingSize,
                model.HeadCount,
                model.Block.MlpSize,
                model.VocabularySize)));
            var loss = GPU.CreateBuffer<float>(batchSize);
            var ad = GPU.CreateADKernel(new GptLanguageModelTrainingKernel(
                tokens.AsReadOnly(),
                workspace.AsReadWrite(),
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

            return new GptGradientProbe(model, tokens, workspace, loss, ad, batchSize);
        }

        public void Backward()
            => ad.Backward(batchSize);

        public float[] ReadGradient(string name, int expectedLength)
        {
            using var gradient = GPU.CreateBuffer<float>(expectedLength);
            ad.CopyGradientToBuffer(name, gradient);
            return gradient.ToArray();
        }

        public float[] ReadManagedGradient(string name)
            => ad.ReadBackGradients().Get<float>(name);

        public float FiniteDifference(Parameter<float> parameter, int index, float epsilon)
        {
            var values = parameter.Value.Buffer.ToArray();
            var original = values[index];
            values[index] = original + epsilon;
            parameter.Value.Buffer.Upload(values);
            var plus = EvaluateLoss();

            values[index] = original - epsilon;
            parameter.Value.Buffer.Upload(values);
            var minus = EvaluateLoss();

            values[index] = original;
            parameter.Value.Buffer.Upload(values);

            return (plus - minus) / (2f * epsilon);
        }

        private float EvaluateLoss()
        {
            ad.Forward(batchSize);
            return loss.ToArray().Sum();
        }

        public void Dispose()
        {
            if (disposed)
            {
                return;
            }

            ad.Dispose();
            loss.Dispose();
            workspace.Dispose();
            tokens.Dispose();
            disposed = true;
        }
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNLinearRegressionMeanLossKernel(
    ReadWriteBuffer<float> weight,
    ReadWriteBuffer<float> bias,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float w = weight[0];
        float b = bias[0];
        float prediction = (w * x[i]) + b;
        float error = prediction - y[i];
        float l = error * error * lossScale.Value;
        loss[i] = l;
        ADMarker.Parameter(weight[0]);
        ADMarker.Parameter(bias[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNReluMlpMeanLossKernel(
    ReadWriteBuffer<float> hiddenWeight,
    ReadWriteBuffer<float> hiddenBias,
    ReadWriteBuffer<float> outputWeight,
    ReadWriteBuffer<float> outputBias,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float h0Weight = hiddenWeight[0];
        float h1Weight = hiddenWeight[1];
        float h0Bias = hiddenBias[0];
        float h1Bias = hiddenBias[1];
        float o0Weight = outputWeight[0];
        float o1Weight = outputWeight[1];
        float oBias = outputBias[0];

        float h0Pre = (h0Weight * x[i]) + h0Bias;
        float h1Pre = (h1Weight * x[i]) + h1Bias;
        float h0;
        if (h0Pre > 0f)
        {
            h0 = h0Pre;
        }
        else
        {
            h0 = 0f;
        }

        float h1;
        if (h1Pre > 0f)
        {
            h1 = h1Pre;
        }
        else
        {
            h1 = 0f;
        }

        float prediction = (o0Weight * h0) + (o1Weight * h1) + oBias;
        float error = prediction - y[i];
        float l = error * error * lossScale.Value;
        loss[i] = l;

        ADMarker.Parameter(hiddenWeight[0]);
        ADMarker.Parameter(hiddenWeight[1]);
        ADMarker.Parameter(hiddenBias[0]);
        ADMarker.Parameter(hiddenBias[1]);
        ADMarker.Parameter(outputWeight[0]);
        ADMarker.Parameter(outputWeight[1]);
        ADMarker.Parameter(outputBias[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNSequentialReluMlpMeanLossKernel(
    ReadWriteBuffer<float> linear0_weight,
    ReadWriteBuffer<float> linear0_bias,
    ReadWriteBuffer<float> linear2_weight,
    ReadWriteBuffer<float> linear2_bias,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float h0Weight = linear0_weight[0];
        float h1Weight = linear0_weight[1];
        float h0Bias = linear0_bias[0];
        float h1Bias = linear0_bias[1];
        float o0Weight = linear2_weight[0];
        float o1Weight = linear2_weight[1];
        float oBias = linear2_bias[0];

        float h0Pre = (h0Weight * x[i]) + h0Bias;
        float h1Pre = (h1Weight * x[i]) + h1Bias;
        float h0;
        if (h0Pre > 0f)
        {
            h0 = h0Pre;
        }
        else
        {
            h0 = 0f;
        }

        float h1;
        if (h1Pre > 0f)
        {
            h1 = h1Pre;
        }
        else
        {
            h1 = 0f;
        }

        float prediction = (o0Weight * h0) + (o1Weight * h1) + oBias;
        float error = prediction - y[i];
        float l = error * error * lossScale.Value;
        loss[i] = l;

        ADMarker.Parameter(linear0_weight[0]);
        ADMarker.Parameter(linear0_weight[1]);
        ADMarker.Parameter(linear0_bias[0]);
        ADMarker.Parameter(linear0_bias[1]);
        ADMarker.Parameter(linear2_weight[0]);
        ADMarker.Parameter(linear2_weight[1]);
        ADMarker.Parameter(linear2_bias[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNSingleWeightMeanLossKernel(
    ReadWriteBuffer<float> weight,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float w = weight[0];
        float error = (w * x[i]) - y[i];
        float l = error * error;
        loss[i] = l;
        ADMarker.Parameter(weight[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNAmbiguousGradientAliasKernel(
    ReadWriteBuffer<float> weight,
    ReadWriteBuffer<float> aliasWeight,
    ReadOnlyBuffer<float> x,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float l = (weight[0] + aliasWeight[0]) * x[ThreadIds.X];
        loss[ThreadIds.X] = l;
        ADMarker.Parameter(weight[0]);
        ADMarker.Parameter(aliasWeight[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
[AutoDiff]
public readonly partial struct NNEmbeddingGatherLossKernel(
    ReadWriteBuffer<float> embedding,
    ReadOnlyBuffer<int> indices,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int row = ThreadIds.X;
        int token = indices[row];
        float value0 = embedding[(token * 2) + 0];
        float value1 = embedding[(token * 2) + 1];
        float l = value0 + value1;
        loss[row] = l;
        ADMarker.Parameter(embedding[(token * 2) + 0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NNBinaryClassificationCrossEntropyKernel(
    ReadWriteBuffer<float> weight,
    ReadWriteBuffer<float> bias,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> labels,
    ReadWriteBuffer<float> loss,
    Uniform<float> lossScale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float logit = (weight[0] * x[i]) + bias[0];
        float label = labels[i];
        float expLogit = ShaderMath.Exp(logit);
        float logTerm = ShaderMath.Log(1f + expLogit);
        float l = (logTerm - (label * logit)) * lossScale.Value;
        loss[i] = l;

        ADMarker.Parameter(weight[0]);
        ADMarker.Parameter(bias[0]);
        ADMarker.Loss(l);
    }
}
