using Feather.AD;
using Feather.NN;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// The go/no-go gate for the NN platform work: the packed MLP training kernel's analytic gradients
/// must match central finite differences.
/// </summary>
/// <remarks>
/// Every host, checkpoint, and inference API in this workstream assumes
/// <see cref="MlpRegression3To1LossKernel" /> produces a correct adjoint. The former backend listed variable
/// buffer indexing of a parameter as non-differentiable, and the boundary between "index built from
/// loop counters and uniforms" (which the GPT kernel proves works) and "index loaded from data" is
/// not documented precisely. This test is what pins which side of that boundary the packed layout
/// falls on.
/// </remarks>
public class MlpTrainingGradientTests(Xunit.Abstractions.ITestOutputHelper output)
{
    [Fact]
    public void MlpRegression3To1LossKernelGradientsMatchCentralFiniteDifferences()
    {
        const int hiddenSize = 4;
        const int sampleCount = 6;
        var weights = MakeDeterministicWeights(hiddenSize, seed: 20250815);
        var (inputs, targets) = MakeRegressionSamples(sampleCount, seed: 7);

        using var probe = MlpGradientProbe.Create(hiddenSize, inputs, targets, weights);

        probe.Backward();
        var gradients = probe.ReadGradient("weights", weights.Length);

        // The kernel must execute through the sole Luisa AD route.
        Assert.Equal(DispatchPath.Luisa, probe.LastDispatchPath);
        Assert.Equal(weights.Length, gradients.Length);

        const float epsilon = 1e-2f;
        var failures = new List<string>();
        var maximumRelativeError = 0f;
        for (var index = 0; index < weights.Length; index++)
        {
            var finiteDifference = probe.FiniteDifference(index, epsilon);
            var analytic = gradients[index];
            var tolerance = MathF.Max(4e-3f, MathF.Abs(finiteDifference) * 0.05f);
            var error = MathF.Abs(analytic - finiteDifference);
            if (MathF.Abs(finiteDifference) > 1e-4f)
            {
                maximumRelativeError = MathF.Max(maximumRelativeError, error / MathF.Abs(finiteDifference));
            }

            if (error > tolerance)
            {
                failures.Add($"index {index}: analytic={analytic:R}, finiteDifference={finiteDifference:R}, error={error:R}, tolerance={tolerance:R}");
            }
        }

        Assert.True(
            failures.Count == 0,
            $"{failures.Count} of {weights.Length} packed MLP gradients did not match central finite differences:{Environment.NewLine}{string.Join(Environment.NewLine, failures)}");

        // At least some gradients must be substantial, otherwise a kernel that emitted zeros
        // everywhere would pass on tolerance alone.
        Assert.Contains(gradients, value => MathF.Abs(value) > 1e-2f);
        Assert.True(maximumRelativeError < 0.05f, $"Worst relative gradient error was {maximumRelativeError:R}.");

        output.WriteLine($"packed weights={weights.Length}, dispatchPath={probe.LastDispatchPath}, worstRelativeError={maximumRelativeError:R}");
        for (var index = 0; index < System.Math.Min(6, weights.Length); index++)
        {
            output.WriteLine($"  index {index}: analytic={gradients[index]:R}, finiteDifference={probe.FiniteDifference(index, epsilon):R}");
        }
    }

    [Fact]
    public void MlpRegression3To1LossKernelForwardMatchesHostEvaluation()
    {
        const int hiddenSize = 5;
        const int sampleCount = 4;
        var weights = MakeDeterministicWeights(hiddenSize, seed: 4242);
        var (inputs, targets) = MakeRegressionSamples(sampleCount, seed: 99);

        using var probe = MlpGradientProbe.Create(hiddenSize, inputs, targets, weights);
        var kernelLosses = probe.ForwardLosses();

        for (var sample = 0; sample < sampleCount; sample++)
        {
            var prediction = MlpLayout.Evaluate3To1(weights, hiddenSize, inputs.AsSpan(sample * 3, 3));
            var error = prediction - targets[sample];
            var expected = error * error / sampleCount;
            Assert.InRange(MathF.Abs(kernelLosses[sample] - expected), 0f, 1e-4f);
        }
    }

    [Fact]
    public void MlpLayoutOffsetsPartitionThePackedBufferWithoutOverlap()
    {
        const int hiddenSize = 7;
        var total = MlpLayout.PackedElementCount3To1(hiddenSize);
        var covered = new bool[total];

        MarkRange(covered, MlpLayout.Layer1WeightOffset(hiddenSize), MlpLayout.InputSize * hiddenSize);
        MarkRange(covered, MlpLayout.Layer1BiasOffset(hiddenSize), hiddenSize);
        MarkRange(covered, MlpLayout.Layer2WeightOffset(hiddenSize), hiddenSize * hiddenSize);
        MarkRange(covered, MlpLayout.Layer2BiasOffset(hiddenSize), hiddenSize);
        MarkRange(covered, MlpLayout.OutputWeightOffset(hiddenSize), hiddenSize);
        MarkRange(covered, MlpLayout.OutputBiasOffset(hiddenSize), 1);

        Assert.DoesNotContain(false, covered);
    }

    private static void MarkRange(bool[] covered, int offset, int length)
    {
        for (var i = offset; i < offset + length; i++)
        {
            Assert.False(covered[i], $"Packed MLP element {i} is claimed by two layers.");
            covered[i] = true;
        }
    }

    private static float[] MakeDeterministicWeights(int hiddenSize, int seed)
    {
        var random = new Random(seed);
        var weights = new float[MlpLayout.PackedElementCount3To1(hiddenSize)];
        for (var i = 0; i < weights.Length; i++)
        {
            // Kept away from zero so no ReLU sits exactly on its kink, where a central difference
            // straddles the non-differentiable point and disagrees with any one-sided derivative.
            var magnitude = 0.25f + (float)random.NextDouble();
            weights[i] = random.Next(2) == 0 ? -magnitude : magnitude;
        }

        return weights;
    }

    private static (float[] Inputs, float[] Targets) MakeRegressionSamples(int sampleCount, int seed)
    {
        var random = new Random(seed);
        var inputs = new float[sampleCount * 3];
        var targets = new float[sampleCount];
        for (var sample = 0; sample < sampleCount; sample++)
        {
            for (var component = 0; component < 3; component++)
            {
                inputs[(sample * 3) + component] = (float)((random.NextDouble() * 2.0) - 1.0);
            }

            targets[sample] = (float)((random.NextDouble() * 2.0) - 1.0);
        }

        return (inputs, targets);
    }

    private sealed class MlpGradientProbe : IDisposable
    {
        private readonly GpuBuffer<float> inputs;
        private readonly GpuBuffer<float> targets;
        private readonly GpuBuffer<float> weights;
        private readonly GpuBuffer<float> scratch;
        private readonly GpuBuffer<float> loss;
        private readonly GpuADKernel<MlpRegression3To1LossKernel> ad;
        private readonly int sampleCount;

        private MlpGradientProbe(
            GpuBuffer<float> inputs,
            GpuBuffer<float> targets,
            GpuBuffer<float> weights,
            GpuBuffer<float> scratch,
            GpuBuffer<float> loss,
            GpuADKernel<MlpRegression3To1LossKernel> ad,
            int sampleCount)
        {
            this.inputs = inputs;
            this.targets = targets;
            this.weights = weights;
            this.scratch = scratch;
            this.loss = loss;
            this.ad = ad;
            this.sampleCount = sampleCount;
        }

        public static MlpGradientProbe Create(int hiddenSize, float[] inputValues, float[] targetValues, float[] weightValues)
        {
            var sampleCount = targetValues.Length;
            var inputs = GPU.CreateBuffer<float>(inputValues);
            var targets = GPU.CreateBuffer<float>(targetValues);
            var weights = GPU.CreateBuffer<float>(weightValues);
            var scratch = GPU.CreateBuffer<float>(MlpLayout.ScratchElementsPerLane3To1(hiddenSize) * sampleCount);
            var loss = GPU.CreateBuffer<float>(sampleCount);
            var ad = GPU.CreateADKernel(new MlpRegression3To1LossKernel(
                inputs.AsReadOnly(),
                targets.AsReadOnly(),
                weights.AsReadWrite(),
                scratch.AsReadWrite(),
                loss.AsReadWrite(),
                new Uniform<int>(hiddenSize),
                new Uniform<float>(1f / sampleCount)));

            return new MlpGradientProbe(inputs, targets, weights, scratch, loss, ad, sampleCount);
        }

        public DispatchPath LastDispatchPath => ad.LastDispatchPath;

        public void Backward() => ad.Backward(sampleCount);

        public float[] ReadGradient(string name, int elementCount)
        {
            using var destination = GPU.CreateBuffer<float>(elementCount);
            ad.CopyGradientToBuffer(name, destination);
            return destination.ToArray();
        }

        public float[] ForwardLosses()
        {
            ad.Forward(sampleCount);
            return loss.ToArray();
        }

        public float FiniteDifference(int index, float epsilon)
        {
            var values = weights.ToArray();
            var original = values[index];

            values[index] = original + epsilon;
            weights.Upload(values);
            var plus = EvaluateLoss();

            values[index] = original - epsilon;
            weights.Upload(values);
            var minus = EvaluateLoss();

            values[index] = original;
            weights.Upload(values);

            return (plus - minus) / (2f * epsilon);
        }

        private float EvaluateLoss()
        {
            ad.Forward(sampleCount);
            return loss.ToArray().Sum();
        }

        public void Dispose()
        {
            ad.Dispose();
            loss.Dispose();
            scratch.Dispose();
            weights.Dispose();
            targets.Dispose();
            inputs.Dispose();
        }
    }
}
