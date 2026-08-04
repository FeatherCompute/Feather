using Feather.AD;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

namespace Feather.Integration.Tests;

public class LuisaBackendAdTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void ReverseModeGradientsMatchEasyGpu()
    {
        float[] parameters = [2f, -3f, 0.5f, 4f];
        var expectedLoss = parameters.Select(static value => (value * value) + (3f * value)).ToArray();
        var expectedGradient = parameters.Select(static value => (2f * value) + 3f).ToArray();

        using var easyParameters = GPU.CreateBuffer<float>(parameters);
        using var easyLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var easyAd = GPU.CreateADKernel(new LuisaScalarAdKernel(
            easyParameters.AsReadWrite(), easyLoss.AsReadWrite()));

        using var luisaParameters = GPU.CreateBuffer<float>(parameters);
        using var luisaLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var luisaAd = GPU.CreateADKernel(new LuisaScalarAdKernel(
            luisaParameters.AsReadWrite(), luisaLoss.AsReadWrite()), GpuExecutionBackend.Luisa);

        easyAd.Backward(parameters.Length);
        luisaAd.Backward(parameters.Length);

        Assert.Equal(expectedLoss, easyLoss.ToArray());
        Assert.Equal(expectedLoss, luisaLoss.ToArray());
        Assert.Equal(expectedGradient, easyAd.ReadBackGradients().Get<float>("parameters"));
        Assert.Equal(expectedGradient, luisaAd.ReadBackGradients().Get<float>("parameters"));
        Assert.Equal(DispatchPath.Luisa, luisaAd.LastDispatchPath);

        using var reducedGradient = GPU.CreateBuffer<float>(parameters.Length);
        luisaAd.CopyGradientToBuffer("parameters", reducedGradient);
        Assert.Equal(expectedGradient, reducedGradient.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void VectorParameterGradientsMatchEasyGpu()
    {
        float2[] parameters = [new(2f, -3f), new(0.5f, 4f)];
        float2[] expectedGradient = [new(4f, -6f), new(1f, 8f)];

        using var easyParameters = GPU.CreateBuffer<float2>(parameters);
        using var easyLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var easyAd = GPU.CreateADKernel(new LuisaVectorAdKernel(
            easyParameters.AsReadWrite(), easyLoss.AsReadWrite()));
        using var luisaParameters = GPU.CreateBuffer<float2>(parameters);
        using var luisaLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var luisaAd = GPU.CreateADKernel(new LuisaVectorAdKernel(
            luisaParameters.AsReadWrite(), luisaLoss.AsReadWrite()), GpuExecutionBackend.Luisa);

        easyAd.Backward(parameters.Length);
        luisaAd.Backward(parameters.Length);

        Assert.Equal(easyLoss.ToArray(), luisaLoss.ToArray());
        Assert.Equal(expectedGradient, easyAd.ReadBackGradients().Get<float2>("parameters"));
        Assert.Equal(expectedGradient, luisaAd.ReadBackGradients().Get<float2>("parameters"));
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void CallableGradientsExecuteThroughLuisa()
    {
        float[] parameters = [-2f, -1f, 0.5f, 3f];
        float[] expectedLoss = parameters.Select(static value => 3f * value * value).ToArray();
        float[] expectedGradient = parameters.Select(static value => 6f * value).ToArray();

        using var luisaParameters = GPU.CreateBuffer<float>(parameters);
        using var luisaLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var luisaAd = GPU.CreateADKernel(new LuisaCallableAdKernel(
            luisaParameters.AsReadWrite(), luisaLoss.AsReadWrite()), GpuExecutionBackend.Luisa);

        luisaAd.Backward(parameters.Length);

        Assert.Equal(expectedLoss, luisaLoss.ToArray());
        var luisaGradient = luisaAd.ReadBackGradients().Get<float>("parameters");
        for (var i = 0; i < expectedGradient.Length; i++)
        {
            Assert.InRange(MathF.Abs(luisaGradient[i] - expectedGradient[i]), 0f, 1e-4f);
        }
        Assert.Equal(DispatchPath.Luisa, luisaAd.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ConditionalIntrinsicGradientsMatchEasyGpu()
    {
        float[] parameters = [0.5f, -0.5f];
        float[] expectedGradient = [MathF.Cos(0.5f), -MathF.Sin(-0.5f)];

        using var easyParameters = GPU.CreateBuffer<float>(parameters);
        using var easyLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var easyAd = GPU.CreateADKernel(new LuisaConditionalAdKernel(
            easyParameters.AsReadWrite(), easyLoss.AsReadWrite()));
        using var luisaParameters = GPU.CreateBuffer<float>(parameters);
        using var luisaLoss = GPU.CreateBuffer<float>(parameters.Length);
        using var luisaAd = GPU.CreateADKernel(new LuisaConditionalAdKernel(
            luisaParameters.AsReadWrite(), luisaLoss.AsReadWrite()), GpuExecutionBackend.Luisa);

        easyAd.Backward(parameters.Length);
        luisaAd.Backward(parameters.Length);

        var easyGradient = easyAd.ReadBackGradients().Get<float>("parameters");
        var luisaGradient = luisaAd.ReadBackGradients().Get<float>("parameters");
        Assert.Equal(easyLoss.ToArray(), luisaLoss.ToArray());
        for (var i = 0; i < expectedGradient.Length; i++)
        {
            Assert.InRange(MathF.Abs(easyGradient[i] - expectedGradient[i]), 0f, 2e-3f);
            Assert.InRange(MathF.Abs(luisaGradient[i] - expectedGradient[i]), 0f, 2e-3f);
            Assert.InRange(MathF.Abs(luisaGradient[i] - easyGradient[i]), 0f, 2e-3f);
        }
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LuisaScalarAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float parameter = parameters[i];
        float value = (parameter * parameter) + (3f * parameter);
        loss[i] = value;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(value);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LuisaVectorAdKernel(
    ReadWriteBuffer<float2> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float2 parameter = parameters[i];
        float value = (parameter.X * parameter.X) + (parameter.Y * parameter.Y);
        loss[i] = value;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(value);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LuisaCallableAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float parameter = parameters[i];
        float value = Differentiate(parameter);
        loss[i] = value;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(value);
    }

    [Callable]
    private static float Differentiate(float value)
        => Square(value) * 3f;

    [Callable]
    private static float Square(float value) => value * value;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct LuisaConditionalAdKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float parameter = parameters[i];
        float value;
        if (parameter > 0f)
        {
            value = ShaderMath.Sin(parameter);
        }
        else
        {
            value = ShaderMath.Cos(parameter);
        }
        loss[i] = value;
        ADMarker.Parameter(parameters[i]);
        ADMarker.Loss(value);
    }
}
