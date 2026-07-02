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

    private static void AssertNear(float expected, float actual, float tolerance = 1e-3f)
        => Assert.InRange(actual, expected - tolerance, expected + tolerance);

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
