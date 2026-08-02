using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.NN;
using Feather.Resources;
using Feather.Shaders;

namespace Feather.Integration.Tests;

/// <summary>
/// Closes the training-to-rendering loop: train a tiny MLP on the GPU, checkpoint it, load the weights
/// back through <see cref="InferenceWeights" />, and evaluate them on the GPU through
/// <see cref="MlpInference3To1Kernel" />.
/// </summary>
/// <remarks>
/// The forward arithmetic exists in three places — the host mirror in <see cref="MlpLayout" />, the
/// training kernel's <c>Execute</c>, and the inference kernel's — because a <c>[Callable]</c> cannot take
/// a buffer parameter, so no shared helper can read weights out of one. See
/// <see cref="MlpLoweringBoundaryTests" /> for the boundary and <see cref="MlpShader" /> for what is left
/// shareable. Nothing but a test keeps the three in agreement: the gradient tests pin the first two
/// against each other, and this pins the third against the host mirror.
///
/// The offsets <see cref="MlpShader" /> exposes are checked against <see cref="MlpLayout" /> here too,
/// since an inline pass evaluating per pixel indexes weights through those rather than through the kernel.
/// </remarks>
public class MlpInferenceSmokeTests(Xunit.Abstractions.ITestOutputHelper output)
{
    private const int HiddenSize = 8;
    private const int SampleCount = 64;

    [Fact]
    public void TrainedWeightsLoadedFromACheckpointEvaluateThroughTheShaderLibrary()
    {
        var (inputs, targets) = MakeSamples();
        var directory = Path.Combine(Path.GetTempPath(), $"feather-mlp-inference-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var context = new TrainingContext(directory);
            float trainedLoss;
            float initialLoss;
            using (var job = new MlpRegressionJob(HiddenSize, inputs, targets, learningRate: 0.02f) { PlannedSteps = 400 })
            {
                job.Initialize(context);
                context.AdvanceTo(0);
                initialLoss = job.StepAndReadLoss(context).Loss;
                for (var step = 1; step < 400; step++)
                {
                    context.AdvanceTo(step);
                    job.Step(context);
                }

                trainedLoss = job.ReadLoss();
                Checkpoint.SaveAtomic(
                    context.ResolveProjectPath("weights.fthc"),
                    job.Parameters,
                    new CheckpointMetadata(399, trainedLoss, ModelKind: "mlp-regression-3to1"));
            }

            output.WriteLine($"initialLoss={initialLoss}, trainedLoss={trainedLoss}");
            Assert.True(trainedLoss < initialLoss * 0.1f, $"Expected training to reduce loss, initial={initialLoss}, final={trainedLoss}.");

            // The checkpoint is the only thing carried from training to inference. Nothing GPU-resident
            // survives the job's disposal, which is the point: a render host loads from a file.
            using var weights = InferenceWeights.Load(Path.Combine(directory, "weights.fthc"));
            Assert.Equal([MlpRegressionJob.WeightParameterName], weights.Names.ToArray());
            Assert.Equal("mlp-regression-3to1", weights.Metadata!.ModelKind);
            Assert.False(weights.IsStale());

            var packed = weights[MlpRegressionJob.WeightParameterName].Buffer.ToArray();

            // Evaluate the loaded weights on the GPU, the same route a render pass takes: bind the
            // checkpoint's buffer view straight into the inference kernel.
            const int probeCount = 16;
            var probeInputs = new float[probeCount * MlpLayout.InputSize];
            Array.Copy(inputs, probeInputs, probeInputs.Length);

            using var probeBuffer = GPU.CreateBuffer<float>(probeInputs);
            using var scratch = GPU.CreateBuffer<float>(MlpLayout.ScratchElementsPerLane3To1(HiddenSize) * probeCount);
            using var predictions = GPU.CreateBuffer<float>(probeCount);
            GpuKernel.Dispatch(
                GPU.Context,
                new MlpInference3To1Kernel(
                    probeBuffer.AsReadOnly(),
                    weights.Buffer(MlpRegressionJob.WeightParameterName),
                    scratch.AsReadWrite(),
                    predictions.AsReadWrite(),
                    new Uniform<int>(HiddenSize)),
                new GpuDispatchSize(probeCount, 1, 1),
                wait: true);

            var devicePredictions = predictions.ToArray();
            var worstDelta = 0f;
            for (var i = 0; i < probeCount; i++)
            {
                var expected = MlpLayout.Evaluate3To1(
                    packed,
                    HiddenSize,
                    [
                        probeInputs[(i * MlpLayout.InputSize) + 0],
                        probeInputs[(i * MlpLayout.InputSize) + 1],
                        probeInputs[(i * MlpLayout.InputSize) + 2]
                    ]);
                worstDelta = MathF.Max(worstDelta, MathF.Abs(expected - devicePredictions[i]));
                Assert.Equal(expected, devicePredictions[i], 4);
            }

            output.WriteLine($"worst device-vs-host delta={worstDelta}");

            // Not just self-consistent with the host mirror — actually a trained network. Without this
            // the test would pass on a network that predicts a constant.
            var meanAbsoluteError = 0f;
            for (var i = 0; i < probeCount; i++)
            {
                meanAbsoluteError += MathF.Abs(devicePredictions[i] - targets[i]);
            }

            meanAbsoluteError /= probeCount;
            output.WriteLine($"meanAbsoluteError against targets={meanAbsoluteError}");
            Assert.True(meanAbsoluteError < 0.1f, $"Expected the loaded network to predict its training targets, mean absolute error was {meanAbsoluteError}.");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void InferenceKernelLowersToRealShaderCode()
    {
        var glsl = ShaderInspection.GetGLSL<MlpInference3To1Kernel>();

        // A kernel that type-checks can still fail to become a shader, so assert on the lowered GLSL.
        Assert.NotEmpty(glsl);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(8)]
    [InlineData(31)]
    public void ShaderOffsetHelpersAgreeWithTheHostLayout(int hiddenSize)
    {
        // A pass evaluating inline uses MlpShader's offsets rather than the inference kernel, so those
        // offsets have to name the same slots the trainer wrote.
        Assert.Equal(MlpLayout.PackedElementCount3To1(hiddenSize), MlpShader.PackedElementCount3To1(hiddenSize));
        Assert.Equal(MlpLayout.ScratchElementsPerLane3To1(hiddenSize), MlpShader.ScratchElementsPerLane3To1(hiddenSize));
        Assert.Equal(MlpLayout.Layer1WeightOffset(hiddenSize), MlpShader.Layer1WeightOffset3To1(hiddenSize));
        Assert.Equal(MlpLayout.Layer1BiasOffset(hiddenSize), MlpShader.Layer1BiasOffset3To1(hiddenSize));
        Assert.Equal(MlpLayout.Layer2WeightOffset(hiddenSize), MlpShader.Layer2WeightOffset3To1(hiddenSize));
        Assert.Equal(MlpLayout.Layer2BiasOffset(hiddenSize), MlpShader.Layer2BiasOffset3To1(hiddenSize));
        Assert.Equal(MlpLayout.OutputWeightOffset(hiddenSize), MlpShader.OutputWeightOffset3To1(hiddenSize));
        Assert.Equal(MlpLayout.OutputBiasOffset(hiddenSize), MlpShader.OutputBiasOffset3To1(hiddenSize));
        Assert.Equal(5, MlpShader.WeightIndex(1, 2, 3));
    }

    [Fact]
    public void WeightsCacheLoadsOncePerCheckpointChange()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"feather-mlp-cache-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            var path = Path.Combine(directory, "weights.fthc");
            using var parameter = new Parameter<float>(
                "mlp.weights",
                new Tensor<float>(new TensorShape(4), GPU.CreateBuffer<float>([1f, 2f, 3f, 4f])),
                new Tensor<float>(new TensorShape(4), GPU.CreateBuffer<float>(4)));
            Checkpoint.SaveAtomic(path, [parameter]);

            using var cache = new InferenceWeightsCache(directory);
            var first = cache.GetOrLoad("weights.fthc");
            var second = cache.GetOrLoad("weights.fthc");

            // A pass asking on every frame must get the same instance back, not a fresh upload.
            Assert.Same(first, second);
            Assert.Equal(1, cache.Count);
            Assert.Equal(0, cache.ReloadCount);
            Assert.Equal([1f, 2f, 3f, 4f], first["mlp.weights"].Buffer.ToArray());

            // Retraining rewrites the file; the next request must pick it up without a host restart.
            parameter.Value.Buffer.Upload([5f, 6f, 7f, 8f]);
            File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddSeconds(1));
            Checkpoint.SaveAtomic(path, [parameter]);
            File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddSeconds(2));

            var reloaded = cache.GetOrLoad("weights.fthc");
            Assert.Equal(1, cache.ReloadCount);
            Assert.Equal([5f, 6f, 7f, 8f], reloaded["mlp.weights"].Buffer.ToArray());
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void WeightsCacheRejectsPathsEscapingTheProjectRoot()
    {
        var directory = Path.Combine(Path.GetTempPath(), $"feather-mlp-escape-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            using var cache = new InferenceWeightsCache(directory);

            // A checkpoint path is graph-node data, so it is untrusted input. It must not reach outside
            // the project it belongs to.
            Assert.Throws<ArgumentException>(() => cache.GetOrLoad("../escaped.fthc"));
            Assert.Throws<ArgumentException>(() => cache.GetOrLoad(Path.Combine(Path.GetTempPath(), "absolute.fthc")));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static (float[] Inputs, float[] Targets) MakeSamples()
    {
        var random = new Random(7351);
        var inputs = new float[SampleCount * MlpLayout.InputSize];
        var targets = new float[SampleCount];
        for (var i = 0; i < SampleCount; i++)
        {
            var x = (float)((random.NextDouble() * 2.0) - 1.0);
            var y = (float)((random.NextDouble() * 2.0) - 1.0);
            var z = (float)((random.NextDouble() * 2.0) - 1.0);
            inputs[(i * MlpLayout.InputSize) + 0] = x;
            inputs[(i * MlpLayout.InputSize) + 1] = y;
            inputs[(i * MlpLayout.InputSize) + 2] = z;
            targets[i] = (0.5f * x) - (0.4f * y) + (0.3f * x * z) + 0.1f;
        }

        return (inputs, targets);
    }
}
