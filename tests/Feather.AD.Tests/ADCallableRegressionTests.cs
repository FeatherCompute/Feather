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

    private static void AssertNear(float expected, float actual, float tolerance = 1e-3f)
        => Assert.InRange(actual, expected - tolerance, expected + tolerance);
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
