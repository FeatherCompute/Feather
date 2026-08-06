using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;
using Feather.Shaders;

namespace Feather.Integration.Tests;

/// <summary>
/// Guards the scalar and noise helpers that stand in for intrinsics the generator does not provide.
/// </summary>
/// <remarks>
/// These are ordinary <c>[Callable]</c> functions rather than compiler intrinsics, which is the
/// point: an effect that needs <c>mod</c> or value noise does not need a generator change to build.
/// Asserted on lowered GLSL because a helper that type-checks can still fail to become a shader.
/// </remarks>
public class ShaderNoiseTests
{
    [Fact]
    public void ModFollowsTheDivisorSign()
    {
        // GLSL's mod, not C#'s %. A truncated remainder would return -1 here and mirror a tiled
        // pattern either side of the origin.
        Assert.Equal(2.0f, FeatherMathEx.Mod(-1.0f, 3.0f), 5);
        Assert.Equal(1.0f, FeatherMathEx.Mod(7.0f, 3.0f), 5);
        Assert.Equal(0.0f, FeatherMathEx.Mod(6.0f, 3.0f), 5);
    }

    [Fact]
    public void SignAndStepMatchTheirGlslCounterparts()
    {
        Assert.Equal(-1.0f, FeatherMathEx.Sign(-4.0f), 5);
        Assert.Equal(0.0f, FeatherMathEx.Sign(0.0f), 5);
        Assert.Equal(1.0f, FeatherMathEx.Sign(4.0f), 5);

        Assert.Equal(0.0f, FeatherMathEx.Step(1.0f, 0.5f), 5);
        Assert.Equal(1.0f, FeatherMathEx.Step(1.0f, 1.0f), 5);
    }

    [Fact]
    public void TruncRoundsTowardZero()
    {
        Assert.Equal(1.0f, FeatherMathEx.Trunc(1.7f), 5);
        Assert.Equal(-1.0f, FeatherMathEx.Trunc(-1.7f), 5);
    }

    [Fact]
    public void ExponentHelpersRoundTrip()
    {
        Assert.Equal(8.0f, FeatherMathEx.Exp2(3.0f), 3);
        Assert.Equal(3.0f, FeatherMathEx.Log2(8.0f), 3);
    }

    [Fact]
    public void NoiseIsBoundedAndRepeatable()
    {
        // Bounded, because the wave sum that consumes it would otherwise drift off the surface.
        for (var step = 0; step < 64; step++)
        {
            var position = new float2(step * 0.37f, step * -0.21f);
            var value = FeatherNoise.Value(position);
            Assert.InRange(value, -1.0f, 1.0f);
        }

        // Repeatable, which is what makes a rendered frame reproducible.
        var once = FeatherNoise.Value(new float2(3.5f, -2.25f));
        var twice = FeatherNoise.Value(new float2(3.5f, -2.25f));
        Assert.Equal(once, twice, 6);
    }

    [Fact]
    public void NoiseVariesAcrossCells()
    {
        // A hash that collapsed would leave the surface flat, which is hard to spot in an image.
        var samples = new float[16];
        for (var index = 0; index < samples.Length; index++)
        {
            samples[index] = FeatherNoise.Value(new float2(index * 1.7f, index * 0.9f));
        }

        var mean = 0.0f;
        foreach (var sample in samples)
        {
            mean += sample;
        }

        mean /= samples.Length;

        var variance = 0.0f;
        foreach (var sample in samples)
        {
            variance += (sample - mean) * (sample - mean);
        }

        variance /= samples.Length;
        Assert.True(variance > 0.01f, $"noise variance was {variance}");
    }
}

[Kernel]
[ThreadGroupSize(64)]
public readonly partial struct NoiseProbeKernel(ReadWriteBuffer<float4> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        var position = new float2(i * 0.5f, i * 0.25f);

        output[i] = new float4(
            FeatherMathEx.Mod(i, 3.0f),
            FeatherMathEx.Step(1.0f, i) + FeatherMathEx.Sign(i - 8.0f),
            FeatherNoise.Value(position),
            FeatherMathEx.Exp2(i * 0.1f) + FeatherMathEx.Trunc(i * 0.3f) + FeatherMathEx.Log2(i + 1.0f));
    }
}
