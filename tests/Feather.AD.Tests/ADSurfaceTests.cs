using Feather.AD;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather.AD.Tests;

public class ADSurfaceTests
{
    [Fact]
    public void AdMarkersThrowWhenInvokedOnCpu()
    {
        AssertMarkerThrows("Parameter", typeof(float), 1f);
        AssertMarkerThrows("Parameter", typeof(float2), new float2(1f, 2f));
        AssertMarkerThrows("Parameter", typeof(float3), new float3(1f, 2f, 3f));
        AssertMarkerThrows("Parameter", typeof(float4), new float4(1f, 2f, 3f, 4f));
        AssertMarkerThrows("Loss", typeof(float), 1f);
        AssertMarkerThrows("Loss", typeof(float2), new float2(1f, 2f));
        AssertMarkerThrows("Loss", typeof(float3), new float3(1f, 2f, 3f));
        AssertMarkerThrows("Loss", typeof(float4), new float4(1f, 2f, 3f, 4f));
    }

    [Fact]
    public void GradientSetStoresTypedGradients()
    {
        var gradients = new GradientSet();
        gradients.Register("weight", 3.5f);
        gradients.Register<float>("bias", [1, 2]);

        Assert.True(gradients.TryGet<float>("weight", out var value));
        Assert.Equal(3.5f, value);
        Assert.True(gradients.TryGetArray<float>("bias", out var vector));
        Assert.Equal([1, 2], vector);
        Assert.Equal([1, 2], gradients.Get<float>("bias"));
        Assert.Equal(3.5f, gradients.GetScalar<float>("weight"));
        Assert.True(gradients.Contains("weight"));
        Assert.False(gradients.Contains("missing"));
        Assert.Contains("weight", gradients.Names);
        Assert.False(gradients.TryGet<int>("weight", out _));
        Assert.False(gradients.TryGetArray<int>("bias", out var wrongType));
        Assert.Empty(wrongType);
        Assert.Throws<KeyNotFoundException>(() => gradients.GetArray<int>("bias"));
        Assert.Throws<KeyNotFoundException>(() => gradients.GetScalar<int>("missing"));

        gradients.Clear();
        Assert.Empty(gradients.Names);
    }

    [Fact]
    public void BackwardDispatchesThroughNativeAdBridgeAndReadsGradients()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        using var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), output.AsReadWrite()));

        adKernel.Backward(4);

        Assert.True(adKernel.HasBackwardRun);
        Assert.Equal(4, adKernel.LastBackwardCount);
        Assert.Equal(DispatchPath.TypedEasyGpu, adKernel.LastDispatchPath);
        Assert.Contains("Backward pass", adKernel.GetBackwardGLSL(), StringComparison.Ordinal);
        Assert.False(adKernel.Gradients.HasMaterializedValues);

        Assert.Same(adKernel.Gradients, adKernel.ReadBackGradients());
        Assert.True(adKernel.Gradients.HasMaterializedValues);
        var gradients = adKernel.Gradients.Get<float>("parameters");
        Assert.True(adKernel.Gradients.HasMaterializedValues);
        Assert.Equal(4, gradients.Length);
        AssertNear(2f, gradients[0]);
        AssertNear(4f, gradients[1]);
        AssertNear(6f, gradients[2]);
        AssertNear(8f, gradients[3]);
        Assert.Equal([1, 2, 3, 4], input.ToArray());
    }

    [Fact]
    public void CopyGradientToBufferReducesNativeGradientOnDevice()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var loss = GPU.CreateBuffer<float>(4);
        using var destination = GPU.CreateBuffer<float>(4);
        using var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), loss.AsReadWrite()));

        adKernel.Backward(4);
        adKernel.CopyGradientToBuffer("parameters", destination);

        Assert.Equal([2, 4, 6, 8], destination.ToArray());
        Assert.False(adKernel.Gradients.HasMaterializedValues);
    }

    [Fact]
    public void ForwardClearsBackwardGradientStateUntilNextBackward()
    {
        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var loss = GPU.CreateBuffer<float>(4);
        using var destination = GPU.CreateBuffer<float>(4);
        using var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), loss.AsReadWrite()));

        adKernel.Backward(4);
        adKernel.CopyGradientToBuffer("parameters", destination);
        Assert.Equal([2, 4, 6, 8], destination.ToArray());

        input.Upload([2, 3, 4, 5]);
        adKernel.Forward(4);

        Assert.False(adKernel.HasBackwardRun);
        Assert.Equal(0, adKernel.LastBackwardCount);
        Assert.False(adKernel.Gradients.HasMaterializedValues);
        Assert.Empty(adKernel.Gradients.Names);
        Assert.Equal(DispatchPath.TypedEasyGpu, adKernel.LastDispatchPath);
        Assert.Throws<InvalidOperationException>(() => adKernel.ReadBackGradients());
        Assert.Throws<InvalidOperationException>(() => adKernel.CopyGradientToBuffer("parameters", destination));

        adKernel.Backward(4);
        adKernel.CopyGradientToBuffer("parameters", destination);

        Assert.True(adKernel.HasBackwardRun);
        Assert.Equal(4, adKernel.LastBackwardCount);
        Assert.Equal([4, 6, 8, 10], destination.ToArray());
    }

    [Fact]
    public void CopyGradientToBufferRejectsInvalidStateNameAndShape()
    {
        using var input = GPU.CreateBuffer<float>([1, 2]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var destination = GPU.CreateBuffer<float>(2);
        using var wrongSize = GPU.CreateBuffer<float>(1);
        using var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), loss.AsReadWrite()));

        Assert.Throws<InvalidOperationException>(() => adKernel.ReadBackGradients());
        Assert.Throws<InvalidOperationException>(() => adKernel.CopyGradientToBuffer("parameters", destination));

        adKernel.Backward(2);

        Assert.Throws<ArgumentException>(() => adKernel.CopyGradientToBuffer("", destination));
        Assert.Throws<ArgumentNullException>(() => adKernel.CopyGradientToBuffer("parameters", null!));
        Assert.Throws<KeyNotFoundException>(() => adKernel.CopyGradientToBuffer("missing", destination));
        Assert.Throws<ArgumentException>(() => adKernel.CopyGradientToBuffer("parameters", wrongSize));
    }

    [Fact]
    public void InternalGradientCopyRejectsMissingBackwardAndEmptyCandidateNames()
    {
        using var input = GPU.CreateBuffer<float>([1, 2]);
        using var loss = GPU.CreateBuffer<float>(2);
        using var destination = GPU.CreateBuffer<float>(2);
        using var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), loss.AsReadWrite()));

        Assert.Throws<InvalidOperationException>(() => adKernel.CopyGradientToBuffer(0, destination));

        adKernel.Backward(2);

        Assert.Empty(adKernel.FindGradientMatches(["", " ", "\t"], destination.Length));
        Assert.Throws<ArgumentNullException>(() => adKernel.FindGradientMatches(null!, destination.Length));
        Assert.Throws<ArgumentOutOfRangeException>(() => adKernel.FindGradientMatches(["parameters"], 0));
        Assert.Throws<ArgumentNullException>(() => adKernel.CopyGradientToBuffer(0, null!));
    }

    [Fact]
    public void CopyGradientToBufferRejectsVectorGradientTypes()
    {
        using var parameters2 = GPU.CreateBuffer<float2>([new float2(1f, 2f)]);
        using var loss2 = GPU.CreateBuffer<float>(1);
        using var destination2 = GPU.CreateBuffer<float>(2);
        using var ad2 = GPU.CreateADKernel(new Float2QuadraticAdKernel(parameters2.AsReadWrite(), loss2.AsReadWrite()));

        using var parameters3 = GPU.CreateBuffer<float3>([new float3(1f, 2f, 3f)]);
        using var loss3 = GPU.CreateBuffer<float>(1);
        using var destination3 = GPU.CreateBuffer<float>(3);
        using var ad3 = GPU.CreateADKernel(new Float3QuadraticAdKernel(parameters3.AsReadWrite(), loss3.AsReadWrite()));

        using var parameters4 = GPU.CreateBuffer<float4>([new float4(1f, 2f, 3f, 4f)]);
        using var loss4 = GPU.CreateBuffer<float>(1);
        using var destination4 = GPU.CreateBuffer<float>(4);
        using var ad4 = GPU.CreateADKernel(new Float4QuadraticAdKernel(parameters4.AsReadWrite(), loss4.AsReadWrite()));

        ad2.Backward(1);
        ad3.Backward(1);
        ad4.Backward(1);

        Assert.Throws<NotSupportedException>(() => ad2.CopyGradientToBuffer("parameters", destination2));
        Assert.Throws<NotSupportedException>(() => ad3.CopyGradientToBuffer("parameters", destination3));
        Assert.Throws<NotSupportedException>(() => ad4.CopyGradientToBuffer("parameters", destination4));
    }

    [Fact]
    public void AdKernelMembersThrowAfterDispose()
    {
        using var input = GPU.CreateBuffer<float>([1]);
        using var loss = GPU.CreateBuffer<float>(1);
        using var destination = GPU.CreateBuffer<float>(1);
        var adKernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), loss.AsReadWrite()));

        adKernel.Backward(1);
        adKernel.Dispose();

        Assert.Throws<ObjectDisposedException>(() => _ = adKernel.LastDispatchPath);
        Assert.Throws<ObjectDisposedException>(() => adKernel.Backward(1));
        Assert.Throws<ObjectDisposedException>(() => adKernel.ReadBackGradients());
        Assert.Throws<ObjectDisposedException>(() => adKernel.CopyGradientToBuffer("parameters", destination));
        Assert.Throws<ObjectDisposedException>(() => adKernel.GetBackwardGLSL());
        adKernel.Dispose();
    }

    [Fact]
    public void BackwardRejectsInvalidCount()
    {
        using var input = GPU.CreateBuffer<float>(4);
        using var output = GPU.CreateBuffer<float>(4);
        using var kernel = new GpuADKernel<AdSmokeKernel>(new AdSmokeKernel(input.AsReadWrite(), output.AsReadWrite()));

        Assert.Throws<ArgumentOutOfRangeException>(() => kernel.Backward(0));
    }

    [Fact]
    public void BackwardThrowsAndKeepsStateClearWhenNativeAdFails()
    {
        using var input = GPU.CreateBuffer<float>([2]);
        using var kernel = GPU.CreateADKernel(new MissingLossManagedKernel(input.AsReadWrite()));

        var ex = Assert.Throws<FeatherNativeException>(() => kernel.Backward(1));

        Assert.Equal(FeResult.ErrorUnsupported, ex.Result);
        Assert.False(kernel.HasBackwardRun);
        Assert.Equal(0, kernel.LastBackwardCount);
        Assert.Empty(kernel.Gradients.Names);
    }

    [Fact]
    public void GpuFacadeCreatesAdKernel()
    {
        using var input = GPU.CreateBuffer<float>(4);
        using var output = GPU.CreateBuffer<float>(4);
        using var kernel = GPU.CreateADKernel(new AdSmokeKernel(input.AsReadWrite(), output.AsReadWrite()));

        kernel.Backward(4);

        Assert.True(kernel.HasBackwardRun);
        Assert.Equal(4, kernel.LastBackwardCount);
        Assert.Equal([0, 0, 0, 0], output.ToArray());
        Assert.Equal([0, 0, 0, 0], kernel.Gradients.Get<float>("parameters"));
    }

    private static void AssertNear(float expected, float actual)
        => Assert.InRange(actual, expected - 0.01f, expected + 0.01f);

    private static void AssertMarkerThrows(string name, Type argumentType, object value)
    {
        var method = typeof(AD).GetMethod(name, [argumentType])
            ?? throw new InvalidOperationException($"Missing AD marker overload {name}({argumentType.Name}).");
        var exception = Assert.Throws<System.Reflection.TargetInvocationException>(() => method.Invoke(null, [value]));
        Assert.IsType<InvalidOperationException>(exception.InnerException);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct AdSmokeKernel(
    ReadWriteBuffer<float> parameters,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        float l = p * p;
        loss[i] = l;
        AD.Parameter(parameters[i]);
        AD.Loss(l);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
[AutoDiff]
public readonly partial struct MissingLossManagedKernel(ReadWriteBuffer<float> parameters) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float p = parameters[i];
        AD.Parameter(parameters[i]);
        parameters[i] = p;
    }
}
