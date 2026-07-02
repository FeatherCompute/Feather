using Feather.Math;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.AD.Tests;

public class ADNumericalCorrectnessTests
{
    [Fact]
    public void ScalarQuadraticGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([3.5f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new QuadraticKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(7f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([3.5f], parameters.ToArray());
    }

    [Fact]
    public void LinearRegressionSingleElementGradientsMatchAnalyticDerivatives()
    {
        using var weights = GPU.CreateBuffer<float>([2f]);
        using var biases = GPU.CreateBuffer<float>([-0.5f]);
        using var x = GPU.CreateBuffer<float>([4f]);
        using var y = GPU.CreateBuffer<float>([6f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new LinearRegressionKernel(
            weights.AsReadWrite(),
            biases.AsReadWrite(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var pred = 2f * 4f - 0.5f;
        var error = pred - 6f;
        AssertNear(2f * error * 4f, ad.Gradients.Get<float>("weights")[0]);
        AssertNear(2f * error, ad.Gradients.Get<float>("biases")[0]);
        Assert.Equal([2f], weights.ToArray());
        Assert.Equal([-0.5f], biases.ToArray());
    }

    [Fact]
    public void ScalarAffineAndReusedParameterGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([2.5f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ReusedAffineScalarAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(10f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(23.75f, loss.ToArray()[0]);
        Assert.Equal([2.5f], parameters.ToArray());
    }

    [Fact]
    public void RepeatedScalarLinearRegressionBackwardMatchesUpdatedParameters()
    {
        using var weights = GPU.CreateBuffer<float>([-0.5f]);
        using var biases = GPU.CreateBuffer<float>([0f]);
        using var x = GPU.CreateBuffer<float>([2f]);
        using var y = GPU.CreateBuffer<float>([5f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ScalarLinearRegressionKernel(
            weights.AsReadWrite(),
            biases.AsReadWrite(),
            x.AsReadOnly(),
            y.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(1);
        AssertNear(-24f, ad.Gradients.Get<float>("weights")[0]);
        AssertNear(-12f, ad.Gradients.Get<float>("biases")[0]);

        weights.Upload([0.7f]);
        biases.Upload([0.6f]);

        ad.Backward(1);
        AssertNear(-12f, ad.Gradients.Get<float>("weights")[0]);
        AssertNear(-6f, ad.Gradients.Get<float>("biases")[0]);
        AssertNear(9f, loss.ToArray()[0]);
    }

    [Fact]
    public void MultiParameterAliasesAndRepeatedBackwardReplaceGradients()
    {
        using var left = GPU.CreateBuffer<float>([2f]);
        using var right = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new AliasedTwoParameterAdKernel(
            left.AsReadWrite(),
            right.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);
        AssertNear(120f, ad.Gradients.Get<float>("left")[0]);
        AssertNear(48f, ad.Gradients.Get<float>("right")[0]);
        AssertNear(144f, loss.ToArray()[0]);

        left.Upload([-1f]);
        right.Upload([4f]);

        ad.Backward(1);
        AssertNear(-16f, ad.Gradients.Get<float>("left")[0]);
        AssertNear(32f, ad.Gradients.Get<float>("right")[0]);
        AssertNear(64f, loss.ToArray()[0]);
    }

    [Fact]
    public void RepeatedBufferParameterMarkersShareOneUpdatedGradientStorage()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var auxiliary = GPU.CreateBuffer<float>([3f, 4f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new RepeatedBufferParameterMarkerAdKernel(
            parameters.AsReadWrite(),
            auxiliary.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(1);

        Assert.Equal(["parameters"], ad.Gradients.Names);
        AssertNear(18f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(81f, loss.ToArray()[0]);
    }

    [Fact]
    public void SharedScalarParameterReadByMultipleThreadsReducesGradients()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(4);
        using var ad = GPU.CreateADKernel(new SharedScalarParameterAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(4);

        AssertNear(120f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear([4f, 16f, 36f, 64f], loss.ToArray());
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void SharedScalarParameterReadBySixteenThreadsSumsGradients()
    {
        using var parameters = GPU.CreateBuffer<float>([0.5f]);
        using var loss = GPU.CreateBuffer<float>(16);
        using var ad = GPU.CreateADKernel(new SharedScalarParameterSixteenLaneAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(16);

        AssertNear(352f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear([
            12.25f, 20.25f, 30.25f, 42.25f,
            56.25f, 72.25f, 90.25f, 110.25f,
            132.25f, 156.25f, 182.25f, 210.25f,
            240.25f, 272.25f, 306.25f, 342.25f
        ], loss.ToArray());
    }

    [Fact]
    public void ThreadGroupSizeGreaterThanOneWritesDistinctGradients()
    {
        using var parameters = GPU.CreateBuffer<float>([1f, 2f, 3f, 4f]);
        using var loss = GPU.CreateBuffer<float>(4);
        using var ad = GPU.CreateADKernel(new ThreadGroupSizeFourQuadraticAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(4);

        AssertNear([2f, 4f, 6f, 8f], ad.Gradients.Get<float>("parameters"));
        AssertNear([1f, 4f, 9f, 16f], loss.ToArray());
    }

    [Fact]
    public void BufferVectorElementwiseWritesDistinctGradients()
    {
        using var parameters = GPU.CreateBuffer<float>([1f, 2f, 3f, 4f]);
        using var loss = GPU.CreateBuffer<float>(4);
        using var ad = GPU.CreateADKernel(new QuadraticKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(4);

        AssertNear([2f, 4f, 6f, 8f], ad.Gradients.Get<float>("parameters"));
        Assert.Equal([1f, 2f, 3f, 4f], parameters.ToArray());
    }

    [Fact]
    public void CallableHelperParticipatesInGradientFlow()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableQuadraticKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(54f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([3f], parameters.ToArray());
    }

    [Fact]
    public void FiniteDifferenceParityForScalarQuadratic()
    {
        const float w = 2.25f;
        const float eps = 1e-2f;
        using var parameters = GPU.CreateBuffer<float>([w]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new QuadraticKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        var finiteDifference = (EvaluateQuadraticLoss(w + eps) - EvaluateQuadraticLoss(w - eps)) / (2f * eps);
        AssertNear(finiteDifference, ad.Gradients.Get<float>("parameters")[0], 1e-2f);
    }

    [Fact]
    public void IfElseControlFlowGradientMatchesAnalyticDerivativeForBothBranches()
    {
        using var parameters = GPU.CreateBuffer<float>([2f, -2f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new IfElseControlFlowAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(2);

        AssertNear([16f, -36f], ad.Gradients.Get<float>("parameters"));
        AssertNear([16f, 36f], loss.ToArray());
    }

    [Fact]
    public void ForLoopAccumulationGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([1.5f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ForLoopAccumulationAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(75f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(56.25f, loss.ToArray()[0]);
    }

    [Fact]
    public void NonZeroStartPositiveStepForLoopGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NonZeroStartStepForLoopAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(324f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(324f, loss.ToArray()[0]);
    }

    [Fact]
    public void NestedControlFlowGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NestedControlFlowAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(64f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(64f, loss.ToArray()[0]);
    }

    [Fact]
    public void BranchConditionCanDependOnNonParameterData()
    {
        using var parameters = GPU.CreateBuffer<float>([2f, 2f]);
        using var flags = GPU.CreateBuffer<float>([1f, -1f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new NonParameterBranchConditionAdKernel(
            parameters.AsReadWrite(),
            flags.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(2);

        AssertNear([16f, 36f], ad.Gradients.Get<float>("parameters"));
        AssertNear([16f, 36f], loss.ToArray());
    }

    [Fact]
    public void IntrinsicSqrtPowAndMixGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new IntrinsicSqrtPowMixAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var expectedLoss = MathF.Sqrt(5f) + 8f + 2.5f;
        var expectedGradient = (2f / MathF.Sqrt(5f)) + 12f + 1.25f;
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(expectedLoss, loss.ToArray()[0]);
    }

    [Fact]
    public void ScalarElementaryIntrinsicGradientsMatchAnalyticDerivatives()
    {
        using var parameters = GPU.CreateBuffer<float>([0.5f, -0.25f, 0.25f, 0.75f, 1.5f, 2.25f]);
        using var loss = GPU.CreateBuffer<float>(6);
        using var ad = GPU.CreateADKernel(new ScalarElementaryIntrinsicsAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(6);

        var expectedLoss = new[]
        {
            MathF.Sin(0.5f),
            MathF.Cos(-0.25f),
            MathF.Tan(0.25f),
            MathF.Exp(0.75f),
            MathF.Log(1.5f),
            MathF.Sqrt(2.25f)
        };
        var expectedGradients = new[]
        {
            MathF.Cos(0.5f),
            -MathF.Sin(-0.25f),
            1f + (MathF.Tan(0.25f) * MathF.Tan(0.25f)),
            MathF.Exp(0.75f),
            1f / 1.5f,
            1f / (2f * MathF.Sqrt(2.25f))
        };
        AssertNear(expectedGradients, ad.Gradients.Get<float>("parameters"), 2e-3f);
        AssertNear(expectedLoss, loss.ToArray(), 2e-3f);
    }

    [Fact]
    public void IntrinsicArgumentExpressionGradientMatchesSoftplusDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([-1f, 0f, 1.25f]);
        using var loss = GPU.CreateBuffer<float>(3);
        using var ad = GPU.CreateADKernel(new IntrinsicArgumentExpressionAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(3);

        var expectedLoss = new float[3];
        var expectedGradients = new float[3];
        var inputs = parameters.ToArray();
        for (var i = 0; i < inputs.Length; i++)
        {
            var expValue = MathF.Exp(inputs[i]);
            expectedLoss[i] = MathF.Log(1f + expValue);
            expectedGradients[i] = expValue / (1f + expValue);
        }

        AssertNear(expectedGradients, ad.Gradients.Get<float>("parameters"), 2e-3f);
        AssertNear(expectedLoss, loss.ToArray(), 2e-3f);
    }

    [Fact]
    public void PowAndMixGradientsMatchAnalyticDerivativesForAllArguments()
    {
        using var bases = GPU.CreateBuffer<float>([2f]);
        using var exponents = GPU.CreateBuffer<float>([3f]);
        using var left = GPU.CreateBuffer<float>([1.25f]);
        using var right = GPU.CreateBuffer<float>([4.5f]);
        using var t = GPU.CreateBuffer<float>([0.2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new PowAndMixAllArgumentsAdKernel(
            bases.AsReadWrite(),
            exponents.AsReadWrite(),
            left.AsReadWrite(),
            right.AsReadWrite(),
            t.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var expectedLoss = MathF.Pow(2f, 3f) + ShaderMath.Mix(1.25f, 4.5f, 0.2f);
        AssertNear(expectedLoss, loss.ToArray()[0]);
        AssertNear(3f * MathF.Pow(2f, 2f), ad.Gradients.Get<float>("bases")[0]);
        AssertNear(MathF.Pow(2f, 3f) * MathF.Log(2f), ad.Gradients.Get<float>("exponents")[0], 2e-3f);
        AssertNear(0.8f, ad.Gradients.Get<float>("left")[0]);
        AssertNear(0.2f, ad.Gradients.Get<float>("right")[0]);
        AssertNear(3.25f, ad.Gradients.Get<float>("t")[0]);
    }

    [Fact]
    public void PiecewiseScalarIntrinsicGradientsMatchActiveRegions()
    {
        using var parameters = GPU.CreateBuffer<float>([
            -1.25f, 1.5f,
            0.25f, 0.75f,
            0.25f, 0.75f,
            -0.5f, 0.25f, 1.25f
        ]);
        using var loss = GPU.CreateBuffer<float>(9);
        using var ad = GPU.CreateADKernel(new PiecewiseScalarIntrinsicsAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(9);

        AssertNear(
            [-1f, 1f, 1f, 0f, 0f, 1f, 0f, 1f, 0f],
            ad.Gradients.Get<float>("parameters"));
        AssertNear(
            [1.25f, 1.5f, 0.25f, 0.5f, 0.5f, 0.75f, -0.25f, 0.25f, 0.75f],
            loss.ToArray());
    }

    [Fact]
    public void VectorIntrinsicGradientsMatchAnalyticDerivatives()
    {
        using var parameters = GPU.CreateBuffer<float3>([
            new float3(1f, -2f, 4f),
            new float3(3f, 4f, 12f),
            new float3(1f, 2f, 2f)
        ]);
        using var loss = GPU.CreateBuffer<float>(3);
        using var ad = GPU.CreateADKernel(new VectorIntrinsicAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(3);

        var gradients = ad.Gradients.Get<float3>("parameters");
        AssertNear(new float3(2f, -3f, 0.5f), gradients[0], 2e-3f);
        AssertNear(new float3(3f / 13f, 4f / 13f, 12f / 13f), gradients[1], 2e-3f);
        AssertNear(new float3(2f / 9f, -8f / 9f, 7f / 9f), gradients[2], 2e-3f);
        AssertNear([10f, 13f, 1f], loss.ToArray(), 2e-3f);
    }

    [Fact]
    public void IfElseControlFlowWithDifferentIntrinsicsMatchesAnalyticDerivativesForBothBranches()
    {
        using var parameters = GPU.CreateBuffer<float>([0.5f, 1.25f]);
        using var flags = GPU.CreateBuffer<float>([1f, -1f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new IntrinsicIfElseAdKernel(
            parameters.AsReadWrite(),
            flags.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(2);

        var shifted = 1.25f + 2f;
        AssertNear(
            [
                MathF.Cos(0.5f) + 1f,
                (1f / shifted) + (1f / (2f * MathF.Sqrt(shifted)))
            ],
            ad.Gradients.Get<float>("parameters"),
            2e-3f);
        AssertNear(
            [
                MathF.Sin(0.5f) + (0.5f * 0.5f),
                MathF.Log(shifted) + MathF.Sqrt(shifted)
            ],
            loss.ToArray(),
            2e-3f);
    }

    [Fact]
    public void CallableControlFlowWithIntrinsicMatchesAnalyticDerivativeForBothBranches()
    {
        using var parameters = GPU.CreateBuffer<float>([0.5f, -0.5f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new CallableControlFlowIntrinsicAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(2);

        AssertNear(
            [MathF.Cos(0.5f), -MathF.Sin(-0.5f)],
            ad.Gradients.Get<float>("parameters"),
            2e-3f);
        AssertNear(
            [MathF.Sin(0.5f), MathF.Cos(-0.5f)],
            loss.ToArray(),
            2e-3f);
    }

    [Fact]
    public void TernarySelectionGradientMatchesSelectedArmForBothBranches()
    {
        using var parameters = GPU.CreateBuffer<float>([2f, -2f]);
        using var flags = GPU.CreateBuffer<float>([1f, -1f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new TernarySelectAdKernel(
            parameters.AsReadWrite(),
            flags.AsReadOnly(),
            loss.AsReadWrite()));

        ad.Backward(2);

        AssertNear([16f, -36f], ad.Gradients.Get<float>("parameters"));
        AssertNear([16f, 36f], loss.ToArray());
    }

    [Fact]
    public void CallableIfElseParticipatesInGradientFlow()
    {
        using var parameters = GPU.CreateBuffer<float>([2f, -2f]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new CallableIfElseAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(2);

        AssertNear([16f, -36f], ad.Gradients.Get<float>("parameters"));
        AssertNear([16f, 36f], loss.ToArray());
    }

    [Fact]
    public void CallableForLoopParticipatesInGradientFlow()
    {
        using var parameters = GPU.CreateBuffer<float>([1.5f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableForLoopAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(75f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(56.25f, loss.ToArray()[0]);
    }

    [Fact]
    public void TraceableLocalAliasParameterGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([4f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new AliasParameterAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(8f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(16f, loss.ToArray()[0]);
    }

    [Fact]
    public void ConstantIndexAliasParameterGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ConstantIndexAliasParameterAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(6f, ad.Gradients.Get<float>("parameters")[0]);
        AssertNear(9f, loss.ToArray()[0]);
    }

    [Fact]
    public void Float2VectorGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float2>([new float2(1.5f, -2f)]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new Float2QuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(new float2(3f, -4f), ad.Gradients.Get<float2>("parameters")[0]);
        AssertNear(6.25f, loss.ToArray()[0]);
    }

    [Fact]
    public void Float3VectorGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float3>([new float3(1f, -2f, 3f)]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new Float3QuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(new float3(2f, -4f, 6f), ad.Gradients.Get<float3>("parameters")[0]);
        AssertNear(14f, loss.ToArray()[0]);
    }

    [Fact]
    public void Float4VectorGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float4>([new float4(1f, -2f, 3f, -4f)]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new Float4QuadraticAdKernel(parameters.AsReadWrite(), loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(new float4(2f, -4f, 6f, -8f), ad.Gradients.Get<float4>("parameters")[0]);
        AssertNear(30f, loss.ToArray()[0]);
    }

    [Fact]
    public void VectorAliasGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float3>([new float3(2f, -3f, 4f)]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new Float3AliasQuadraticAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(new float3(4f, -6f, 8f), ad.Gradients.Get<float3>("parameters")[0]);
        AssertNear(29f, loss.ToArray()[0]);
    }

    [Fact]
    public void VectorMemberAccessAliasGradientMatchesAnalyticDerivative()
    {
        const float p = 0.25f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NormalizedZAliasAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var expectedGradient = EvaluateNormalizedZSquaredDerivative(p);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-3f);
        AssertNear(EvaluateNormalizedZSquared(p), loss.ToArray()[0], 1e-3f);
        Assert.DoesNotContain("d_(", ad.GetBackwardGLSL(), StringComparison.Ordinal);
    }

    [Fact]
    public void VectorMemberAccessExpressionGradientMatchesAnalyticDerivative()
    {
        const float p = 0.25f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NormalizedZExpressionAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var expectedGradient = EvaluateNormalizedZSquaredDerivative(p);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-3f);
        AssertNear(EvaluateNormalizedZSquared(p), loss.ToArray()[0], 1e-3f);
        Assert.DoesNotContain("d_(", ad.GetBackwardGLSL(), StringComparison.Ordinal);
    }

    [Fact]
    public void VectorCallableGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float2>([new float2(1.5f, -2f)]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new Float2CallableAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(new float2(27f, -36f), ad.Gradients.Get<float2>("parameters")[0]);
        AssertNear(56.25f, loss.ToArray()[0]);
    }

    [Fact]
    public void VectorControlFlowGradientMatchesAnalyticDerivative()
    {
        using var parameters = GPU.CreateBuffer<float2>([new float2(1f, -2f), new float2(-1f, 3f)]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var ad = GPU.CreateADKernel(new Float2IfElseAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(2);

        var gradients = ad.Gradients.Get<float2>("parameters");
        AssertNear(new float2(8f, -16f), gradients[0]);
        AssertNear(new float2(-18f, 54f), gradients[1]);
        AssertNear([20f, 90f], loss.ToArray());
    }

    [Fact]
    public void OverwrittenDifferentiableScratchKeepsOriginalPrimalForBackward()
    {
        const float p = 3f;
        const float eps = 1e-2f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var scratch = GPU.CreateBuffer<float>(1);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new OverwrittenDifferentiableScratchAdKernel(
            parameters.AsReadWrite(),
            scratch.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        var expectedFiniteDifference =
            (EvaluateOverwrittenScratchLoss(p + eps) - EvaluateOverwrittenScratchLoss(p - eps)) / (2f * eps);
        AssertNear(expectedFiniteDifference, ad.Gradients.Get<float>("parameters")[0], 1e-2f);
        AssertNear(EvaluateOverwrittenScratchLoss(p), loss.ToArray()[0]);
    }

    private static float EvaluateQuadraticLoss(float value) => value * value;

    private static float EvaluateOverwrittenScratchLoss(float value)
    {
        var scratch = value * 2f;
        return scratch * scratch;
    }

    private static float EvaluateNormalizedZSquared(float value)
    {
        var denominator = (value * value) + 1.25f;
        return 1f / denominator;
    }

    private static float EvaluateNormalizedZSquaredDerivative(float value)
    {
        var denominator = (value * value) + 1.25f;
        return (-2f * value) / (denominator * denominator);
    }

    private static void AssertNear(float expected, float actual, float tolerance = 1e-3f)
        => Assert.InRange(actual, expected - tolerance, expected + tolerance);

    private static void AssertNear(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance = 1e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (var i = 0; i < expected.Length; i++)
        {
            AssertNear(expected[i], actual[i], tolerance);
        }
    }

    private static void AssertNear(float2 expected, float2 actual, float tolerance = 1e-3f)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
    }

    private static void AssertNear(float3 expected, float3 actual, float tolerance = 1e-3f)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
        AssertNear(expected.Z, actual.Z, tolerance);
    }

    private static void AssertNear(float4 expected, float4 actual, float tolerance = 1e-3f)
    {
        AssertNear(expected.X, actual.X, tolerance);
        AssertNear(expected.Y, actual.Y, tolerance);
        AssertNear(expected.Z, actual.Z, tolerance);
        AssertNear(expected.W, actual.W, tolerance);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct QuadraticKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LinearRegressionKernel(
    ReadWriteBuffer<float> weights,
    ReadWriteBuffer<float> biases,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float w = weights[i];
        float b = biases[i];
        float pred = w * x[i] + b;
        float error = pred - y[i];
        float l = error * error;
        loss[i] = l;
        ADMarker.Parameter(weights[i]);
        ADMarker.Parameter(biases[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ScalarLinearRegressionKernel(
    ReadWriteBuffer<float> weights,
    ReadWriteBuffer<float> biases,
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float w = weights[0];
        float b = biases[0];
        float pred = w * x[i] + b;
        float error = pred - y[i];
        float l = error * error;
        loss[i] = l;
        ADMarker.Parameter(weights[0]);
        ADMarker.Parameter(biases[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct AliasedTwoParameterAdKernel(
    ReadWriteBuffer<float> left,
    ReadWriteBuffer<float> right,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float a = left[0];
        float b = right[0];
        float aliasA = a;
        float aliasB = b;
        float y = (aliasA * aliasA) + aliasA + (aliasB * 2f);
        float l = y * y;
        loss[0] = l;
        ADMarker.Parameter(a);
        ADMarker.Parameter(b);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ReusedAffineScalarAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = (p * p) + (p * 3f) + (2f * p) + 5f;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct RepeatedBufferParameterMarkerAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadOnlyBuffer<float> auxiliary,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float scale = auxiliary[0] + auxiliary[1];
        float y = p + scale;
        float l = y * y;
        loss[0] = l;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct SharedScalarParameterAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[0];
        float scale = (float)(i + 1);
        float y = p * scale;
        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(16, 1, 1)]
[AutoDiff]
public readonly partial struct SharedScalarParameterSixteenLaneAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[0];
        float scale = (float)(i + 3);
        float y = p + scale;
        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(4, 1, 1)]
[AutoDiff]
public readonly partial struct ThreadGroupSizeFourQuadraticAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableQuadraticKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float scaled = Scale(p);
        float l = Square(scaled);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float Scale(float value) => value * 3f;

    [Callable]
    private static float Square(float value) => value * value;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct IfElseControlFlowAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y;
        if (p > 0f)
        {
            y = p * 2f;
        }
        else
        {
            y = p * 3f;
        }

        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ForLoopAccumulationAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 0; j < 5; j = j + 1)
        {
            sum = sum + p;
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NonZeroStartStepForLoopAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 1; j < 7; j = j + 2)
        {
            sum = sum + (p * (float)j);
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NestedControlFlowAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float sum = 0f;
        for (int j = 0; j < 3; j = j + 1)
        {
            if (j < 2)
            {
                sum = sum + p;
            }
            else
            {
                sum = sum + (p * 2f);
            }
        }

        float l = sum * sum;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NonParameterBranchConditionAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadOnlyBuffer<float> flags,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y;
        if (flags[i] > 0f)
        {
            y = p * 2f;
        }
        else
        {
            y = p * 3f;
        }

        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct IntrinsicSqrtPowMixAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float squared = p * p;
        float shifted = squared + 1f;
        float root = ShaderMath.Sqrt(shifted);
        float cubic = ShaderMath.Pow(p, 3f);
        float doubled = p * 2f;
        float blended = ShaderMath.Mix(p, doubled, 0.25f);
        float l = root + cubic + blended;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ScalarElementaryIntrinsicsAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l;
        if (i == 0)
        {
            l = ShaderMath.Sin(p);
        }
        else if (i == 1)
        {
            l = ShaderMath.Cos(p);
        }
        else if (i == 2)
        {
            l = ShaderMath.Tan(p);
        }
        else if (i == 3)
        {
            l = ShaderMath.Exp(p);
        }
        else if (i == 4)
        {
            l = ShaderMath.Log(p);
        }
        else
        {
            l = ShaderMath.Sqrt(p);
        }

        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct IntrinsicArgumentExpressionAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float expValue = ShaderMath.Exp(p);
        float l = ShaderMath.Log(1f + expValue);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct PowAndMixAllArgumentsAdKernel(
    ReadWriteBuffer<float> bases,
    ReadWriteBuffer<float> exponents,
    ReadWriteBuffer<float> left,
    ReadWriteBuffer<float> right,
    ReadWriteBuffer<float> t,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float a = bases[0];
        float b = exponents[0];
        float l = left[0];
        float r = right[0];
        float amount = t[0];
        float powered = ShaderMath.Pow(a, b);
        float blended = ShaderMath.Mix(l, r, amount);
        float y = powered + blended;
        loss[0] = y;
        ADMarker.Parameter(bases[0]);
        ADMarker.Parameter(exponents[0]);
        ADMarker.Parameter(left[0]);
        ADMarker.Parameter(right[0]);
        ADMarker.Parameter(t[0]);
        ADMarker.Loss(y);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct PiecewiseScalarIntrinsicsAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float threshold = 0.5f;
        float lo = -0.25f;
        float hi = 0.75f;
        float l;
        if (i < 2)
        {
            l = ShaderMath.Abs(p);
        }
        else if (i < 4)
        {
            l = ShaderMath.Min(p, threshold);
        }
        else if (i < 6)
        {
            l = ShaderMath.Max(p, threshold);
        }
        else
        {
            l = ShaderMath.Clamp(p, lo, hi);
        }

        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct VectorIntrinsicAdKernel(
    ReadWriteBuffer<float3> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 p = parameters[i];
        float3 dotWeights = new(2f, -3f, 0.5f);
        float3 normalizeWeights = new(1f, -2f, 3f);
        float3 n = ShaderMath.Normalize(p);
        float l;
        if (i == 0)
        {
            l = ShaderMath.Dot(p, dotWeights);
        }
        else if (i == 1)
        {
            l = ShaderMath.Length(p);
        }
        else
        {
            l = ShaderMath.Dot(n, normalizeWeights);
        }

        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct IntrinsicIfElseAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadOnlyBuffer<float> flags,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float shifted = p + 2f;
        float sinPart = ShaderMath.Sin(p);
        float squarePart = p * p;
        float logPart = ShaderMath.Log(shifted);
        float rootPart = ShaderMath.Sqrt(shifted);
        float l;
        if (flags[i] > 0f)
        {
            l = sinPart + squarePart;
        }
        else
        {
            l = logPart + rootPart;
        }

        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableControlFlowIntrinsicAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = Shape(p);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float Shape(float value)
    {
        float result;
        if (value > 0f)
        {
            result = ShaderMath.Sin(value);
        }
        else
        {
            result = ShaderMath.Cos(value);
        }

        return result;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct TernarySelectAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadOnlyBuffer<float> flags,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        bool useDouble = flags[i] > 0f;
        float doubled = p * 2f;
        float tripled = p * 3f;
        float y = useDouble ? doubled : tripled;
        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableIfElseAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y = Shape(p);
        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float Shape(float value)
    {
        float y;
        if (value > 0f)
        {
            y = value * 2f;
        }
        else
        {
            y = value * 3f;
        }

        return y;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableForLoopAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y = RepeatFive(p);
        float l = y * y;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float RepeatFive(float value)
    {
        float sum = 0f;
        for (int j = 0; j < 5; j = j + 1)
        {
            sum = sum + value;
        }

        return sum;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct AliasParameterAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(p);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ConstantIndexAliasParameterAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[0];
        float l = p * p;
        loss[i] = l;
        ADMarker.Parameter(p);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float2QuadraticAdKernel(
    ReadWriteBuffer<float2> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 p = parameters[i];
        float l = Hlsl.Dot(p, p);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float3AliasQuadraticAdKernel(
    ReadWriteBuffer<float3> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 p = parameters[i];
        float3 alias = p;
        float l = Hlsl.Dot(alias, alias);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float2CallableAdKernel(
    ReadWriteBuffer<float2> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 p = parameters[i];
        float2 y = Triple(p);
        float l = Hlsl.Dot(y, y);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float2 Triple(float2 value) => value + value + value;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float2IfElseAdKernel(
    ReadWriteBuffer<float2> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 p = parameters[i];
        float2 y;
        if (i == 0)
        {
            y = p * 2f;
        }
        else
        {
            y = p * 3f;
        }

        float l = Hlsl.Dot(y, y);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct OverwrittenDifferentiableScratchAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> scratch,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        scratch[0] = p * 2f;
        float used = scratch[0];
        float l = used * used;
        scratch[0] = p * 7f;
        loss[0] = l;

        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NormalizedZAliasAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float3 v = ShaderMath.Normalize(new float3(p, 0.5f, 1.0f));
        float z = v.Z;
        float l = z * z;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NormalizedZExpressionAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float3 v = ShaderMath.Normalize(new float3(p, 0.5f, 1.0f));
        float l = v.Z * v.Z;
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float3QuadraticAdKernel(
    ReadWriteBuffer<float3> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 p = parameters[i];
        float l = Hlsl.Dot(p, p);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct Float4QuadraticAdKernel(
    ReadWriteBuffer<float4> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float4 p = parameters[i];
        float l = Hlsl.Dot(p, p);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }
}
