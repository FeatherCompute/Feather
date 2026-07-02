using Feather.Math;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.AD.Tests;

public class ADCallableRegressionTests
{
    [Fact]
    public void CallableDeclarationOrderDoesNotAffectGradient()
    {
        using var parameters = GPU.CreateBuffer<float>([3f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableDeclarationOrderAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(27f, loss.ToArray()[0]);
        AssertNear(18f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([3f], parameters.ToArray());
    }

    [Fact]
    public void SameCallableCalledTwiceReusesCorrectSubTape()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new SameCallableTwiceAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(16f, loss.ToArray()[0]);
        AssertNear(32f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void DistinctCallablesCanBeCalledMultipleTimesInMixedOrder()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new MixedCallableReuseAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(30f, loss.ToArray()[0]);
        AssertNear(9f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void CallableWithLocalVariablesAndReturnExpressionParticipatesInGradient()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableLocalReturnExpressionAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(121f, loss.ToArray()[0]);
        AssertNear(66f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void NestedCallableToCallableParticipatesInGradient()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NestedCallableAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(12f, loss.ToArray()[0]);
        AssertNear(12f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void NestedShaderLibraryCallableToCallableParticipatesInGradient()
    {
        using var parameters = GPU.CreateBuffer<float>([2f]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NestedShaderLibraryCallableAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        AssertNear(12f, loss.ToArray()[0]);
        AssertNear(12f, ad.Gradients.Get<float>("parameters")[0]);
        Assert.Equal([2f], parameters.ToArray());
    }

    [Fact]
    public void ShaderLibraryCallableLocalPrimalsParticipateInBackward()
    {
        const float p = 0.3f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ShaderLibraryMisLikeAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float ltcValue = (p * p) + 0.5f;
        float error = MathF.Abs(0.7f - ltcValue);
        float denominator = (ltcValue / 1.2f) + 0.2f;
        float expectedLoss = error * error * error / denominator;
        float dLossDValue = ((-3f * error * error) * denominator - (error * error * error / 1.2f)) /
            (denominator * denominator);
        float expectedGradient = dLossDValue * 2f * p;

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-5f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-4f);

        string backward = ad.GetBackwardGLSL();
        string backwardSection = BackwardSection(backward);
        Assert.Contains("denominator", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("brdfValue", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("ltcMagnitude", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("d_(", backwardSection, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadOnlyBufferLeavesDoNotBecomeGradientTargets()
    {
        const float targetValue = 1.1f;
        const float pdfValue = 0.25f;
        const float p = 0.4f;
        using var targets = GPU.CreateBuffer<float>([targetValue]);
        using var pdfs = GPU.CreateBuffer<float>([pdfValue]);
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new ReadOnlyTargetMisLikeAdKernel(
            targets.AsReadOnly(),
            pdfs.AsReadOnly(),
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float error = MathF.Abs(targetValue - p);
        float denominator = pdfValue + p;
        float expectedLoss = error * error * error / denominator;
        float expectedGradient = ((-3f * error * error) * denominator - (error * error * error)) /
            (denominator * denominator);

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-5f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-4f);
        Assert.False(ad.Gradients.Contains("targets"));
        Assert.False(ad.Gradients.Contains("pdfs"));

        string backward = ad.GetBackwardGLSL();
        Assert.DoesNotContain("grad_targets", backward, StringComparison.Ordinal);
        Assert.DoesNotContain("grad_pdfs", backward, StringComparison.Ordinal);
        Assert.DoesNotContain("grad_fe_1", backward, StringComparison.Ordinal);
    }

    [Fact]
    public void CallableLocalNamedLikeGlslIntrinsicDoesNotRemapFunctionName()
    {
        const float p = 0.4f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableLengthNameCollisionAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float expectedLoss = (5f * p * p) + 9f;
        float expectedGradient = 10f * p;

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-4f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-4f);

        string backwardSection = BackwardSection(ad.GetBackwardGLSL());
        Assert.Contains("length(", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("_length(", backwardSection, StringComparison.Ordinal);
    }

    [Fact]
    public void NestedCallableSwizzleArgumentsParticipateInGradient()
    {
        const float p = 0.3f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new NestedCallableSwizzleArgumentAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float expectedLoss = ((2f * p) - 0.5f) * ((2f * p) - 0.5f) + (p + 0.5f) * (p + 0.5f);
        float expectedGradient = (10f * p) - 1f;

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-4f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-4f);

        string backwardSection = BackwardSection(ad.GetBackwardGLSL());
        Assert.DoesNotContain("+()", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("abs(()", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("()-", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("d_(", backwardSection, StringComparison.Ordinal);
    }

    [Fact]
    public void LtcStyleNestedShaderLibraryPathCompilesAndDifferentiates()
    {
        const float p = 0.3f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new LtcStyleNestedShaderLibraryAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float brdfValue = (2f * p) + 0.1f;
        float brdfPdf = p + 0.2f;
        float ltcValue = (5f * p * p) + 9f;
        float magnitude = p + 2f;
        float error = ltcValue - brdfValue;
        float denominator = (ltcValue / magnitude) + brdfPdf;
        float expectedLoss = error * error * error / denominator;

        float dError = (10f * p) - 2f;
        float dDenominator = (((10f * p) * magnitude) - ltcValue) / (magnitude * magnitude) + 1f;
        float expectedGradient =
            (3f * error * error * dError / denominator) -
            (error * error * error * dDenominator / (denominator * denominator));

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-4f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 1e-3f);

        string backwardSection = BackwardSection(ad.GetBackwardGLSL());
        Assert.Contains("length(", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("_length(", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("+()", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("abs(()", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("()-", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("d_(", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("grad_targets", backwardSection, StringComparison.Ordinal);
    }

    [Fact]
    public void CallableRematerializesTransitiveLocalDependencyChain()
    {
        const float p = 0.23f;
        using var parameters = GPU.CreateBuffer<float>([p]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var ad = GPU.CreateADKernel(new CallableEvalLikeDependencyChainAdKernel(
            parameters.AsReadWrite(),
            loss.AsReadWrite()));

        ad.Backward(1);

        float expectedLoss = EvalLikeCpu(p);
        const float eps = 1e-3f;
        float expectedGradient = (EvalLikeCpu(p + eps) - EvalLikeCpu(p - eps)) / (2f * eps);

        AssertNear(expectedLoss, loss.ToArray()[0], 1e-4f);
        AssertNear(expectedGradient, ad.Gradients.Get<float>("parameters")[0], 3e-3f);

        string backwardSection = BackwardSection(ad.GetBackwardGLSL());
        int original = backwardSection.IndexOf("_original = normalize", StringComparison.Ordinal);
        int transformed = backwardSection.IndexOf("_transformed = ", StringComparison.Ordinal);
        Assert.True(original >= 0, backwardSection);
        Assert.True(transformed > original, backwardSection);
        Assert.DoesNotContain("_length(", backwardSection, StringComparison.Ordinal);
        Assert.DoesNotContain("d_(", backwardSection, StringComparison.Ordinal);
    }

    private static void AssertNear(float expected, float actual, float tolerance = 1e-3f)
        => Assert.InRange(actual, expected - tolerance, expected + tolerance);

    private static float EvalLikeCpu(float p)
    {
        float3 direction = NormalizeCpu(new float3(0.2f, 0.1f, 0.97f));
        float3 original = NormalizeCpu(new float3(
            direction.X / (1f + p),
            direction.Y,
            direction.Z));
        float3 transformed = new(
            (1f + (0.5f * p)) * original.X,
            1.2f * original.Y,
            0.9f * original.Z);
        float length = LengthCpu(transformed);
        float d = MathF.Max(original.Z, 0.0f);
        float scale = MathF.Exp(p);
        return scale * d / length;
    }

    private static float3 NormalizeCpu(float3 value)
    {
        float length = LengthCpu(value);
        return value / length;
    }

    private static float LengthCpu(float3 value)
        => MathF.Sqrt((value.X * value.X) + (value.Y * value.Y) + (value.Z * value.Z));

    private static string BackwardSection(string glsl)
    {
        const string marker = "// === Backward pass";
        int markerIndex = glsl.IndexOf(marker, StringComparison.Ordinal);
        if (markerIndex < 0)
        {
            return glsl;
        }

        int mainIndex = glsl.IndexOf("void main()", StringComparison.Ordinal);
        int openBrace = mainIndex < 0 ? -1 : glsl.IndexOf('{', mainIndex);
        if (openBrace < 0 || markerIndex < openBrace)
        {
            return glsl[markerIndex..];
        }

        int depth = 0;
        for (int i = openBrace; i < glsl.Length; i++)
        {
            if (glsl[i] == '{')
            {
                depth++;
            }
            else if (glsl[i] == '}')
            {
                depth--;
                if (depth == 0)
                {
                    return glsl[markerIndex..(i + 1)];
                }
            }
        }

        return glsl[markerIndex..];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableDeclarationOrderAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float a = Square(p);
        float l = Scale(a);
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
public readonly partial struct SameCallableTwiceAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float a = Square(p);
        float b = Square(a);
        loss[i] = b;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(b);
    }

    [Callable]
    private static float Square(float value) => value * value;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct MixedCallableReuseAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float a = AddOne(p);
        float b = Triple(a);
        float c = AddOne(b);
        float l = Triple(c);
        loss[i] = l;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(l);
    }

    [Callable]
    private static float Triple(float value) => value * 3f;

    [Callable]
    private static float AddOne(float value) => value + 1f;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableLocalReturnExpressionAdKernel(
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
        float doubled = value * 2f;
        float shifted = doubled + 5f;
        return shifted + value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NestedCallableAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y = Outer(p);
        loss[i] = y;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(y);
    }

    [Callable]
    private static float Outer(float value)
    {
        return Inner(value) * 3f;
    }

    [Callable]
    private static float Inner(float value)
    {
        return value * value;
    }
}

[ShaderLibrary]
public static class NestedAdShaderLibrary
{
    [Callable]
    public static float Outer(float value)
    {
        return Inner(value) * 3f;
    }

    [Callable]
    public static float Inner(float value)
    {
        return value * value;
    }
}

[ShaderLibrary]
public static class LtcAdRegressionShaderLibrary
{
    [Callable]
    public static float MisLike(float brdfValue, float brdfPdf, float ltcValue, float ltcMagnitude)
    {
        float pdfLtc = ltcValue / ShaderMath.Max(ltcMagnitude, 1e-5f);
        float denominator = ShaderMath.Max(pdfLtc + brdfPdf, 1e-5f);
        float error = ShaderMath.Abs(brdfValue - ltcValue);
        return error * error * error / denominator;
    }

    [Callable]
    public static float LengthNameCollision(float3 value)
    {
        float length = ShaderMath.Length(value);
        return length * length;
    }

    [Callable]
    public static float2 BrdfPair(float value)
    {
        return new float2((2f * value) + 0.1f, value + 0.2f);
    }

    [Callable]
    public static float LtcStyleNestedPath(float value)
    {
        float2 brdf = BrdfPair(value);
        float ltcValue = LengthNameCollision(new float3(value, value * 2f, 3f));
        float magnitude = value + 2f;
        return MisLike(brdf.X, brdf.Y, ltcValue, magnitude);
    }

    [Callable]
    public static float EvalLike(float3 direction, float3x3 m, float3x3 invM, float scale)
    {
        float3 original = ShaderMath.Normalize(invM * direction);
        float3 transformed = m * original;
        float length = ShaderMath.Length(transformed);
        float d = ShaderMath.Max(original.Z, 0.0f);
        return scale * d / length;
    }
}

[ShaderLibrary]
public static class NestedSwizzleArgumentShaderLibrary
{
    [Callable]
    public static float Outer(float2 value, float target)
    {
        return Inner(value.X, target) + Inner(value.Y, target);
    }

    [Callable]
    public static float Inner(float x, float y)
    {
        float delta = x - y;
        return delta * delta;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NestedShaderLibraryCallableAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float y = NestedAdShaderLibrary.Outer(p);
        loss[i] = y;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(y);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ShaderLibraryMisLikeAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float ltcValue = (p * p) + 0.5f;
        float y = LtcAdRegressionShaderLibrary.MisLike(0.7f, 0.2f, ltcValue, 1.2f);
        loss[0] = y;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(y);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct ReadOnlyTargetMisLikeAdKernel(
    ReadOnlyBuffer<float> targets,
    ReadOnlyBuffer<float> pdfs,
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float model = parameters[0];
        float error = ShaderMath.Abs(targets[0] - model);
        float error2 = error * error;
        float denominator = ShaderMath.Max(pdfs[0] + model, 1e-5f);
        float y = error2 * error / denominator;
        loss[0] = y;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(y);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableLengthNameCollisionAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float y = LtcAdRegressionShaderLibrary.LengthNameCollision(new float3(p, p * 2f, 3f));
        loss[0] = y;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(y);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct NestedCallableSwizzleArgumentAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float2 pair = new float2(p * 2f, p + 1f);
        float result = NestedSwizzleArgumentShaderLibrary.Outer(pair, 0.5f);
        loss[0] = result;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(result);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LtcStyleNestedShaderLibraryAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float result = LtcAdRegressionShaderLibrary.LtcStyleNestedPath(p);
        loss[0] = result;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(result);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct CallableEvalLikeDependencyChainAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        float p = parameters[0];
        float3 direction = ShaderMath.Normalize(new float3(0.2f, 0.1f, 0.97f));
        float3x3 m = new(
            1f + (0.5f * p), 0f, 0f,
            0f, 1.2f, 0f,
            0f, 0f, 0.9f);
        float3x3 invM = new(
            1f / (1f + p), 0f, 0f,
            0f, 1f, 0f,
            0f, 0f, 1f);
        float scale = ShaderMath.Exp(p);
        float result = LtcAdRegressionShaderLibrary.EvalLike(direction, m, invM, scale);
        loss[0] = result;
        ADMarker.Parameter(parameters[0]);
        ADMarker.Loss(result);
    }
}
