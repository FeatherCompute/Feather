using Feather.Interop;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class GpuMultiContextCollection
{
    public const string Name = "GPU multi-context";
}

[Collection(GpuMultiContextCollection.Name)]
public class GpuMultiContextTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void SameDeviceContextsDispatchIndependentlyAndRejectCrossContextUse()
    {
        var runtime = GpuRuntime.Create();
        using var firstContext = runtime.CreateContext();
        using var secondContext = runtime.CreateContext();
        var first = GPU.WithContext(firstContext);
        var second = GPU.WithContext(secondContext);

        Assert.Equal(runtime.DefaultDevice.Backend, firstContext.Backend);
        Assert.Equal(runtime.DefaultDevice.Backend, secondContext.Backend);

        using var firstInput = first.CreateBuffer<float>([1.0f, 2.0f, 3.0f]);
        using var firstOutput = first.CreateBuffer<float>(3);
        using var secondInput = second.CreateBuffer<float>([10.0f, 20.0f, 30.0f]);
        using var secondOutput = second.CreateBuffer<float>(3);

        var firstPath = first.Dispatch(
            new MultiContextIncrementKernel(firstInput.AsReadOnly(), firstOutput.AsReadWrite()), 3);
        var secondPath = second.Dispatch(
            new MultiContextIncrementKernel(secondInput.AsReadOnly(), secondOutput.AsReadWrite()), 3);
        firstInput.Upload([4.0f, 5.0f, 6.0f]);
        var resumedFirstPath = first.Dispatch(
            new MultiContextIncrementKernel(firstInput.AsReadOnly(), firstOutput.AsReadWrite()), 3);

        Assert.Equal(DispatchPath.Luisa, firstPath);
        Assert.Equal(DispatchPath.Luisa, secondPath);
        Assert.Equal(DispatchPath.Luisa, resumedFirstPath);
        Assert.Equal([5.0f, 6.0f, 7.0f], firstOutput.ToArray());
        Assert.Equal([11.0f, 21.0f, 31.0f], secondOutput.ToArray());
        Assert.Equal(GPU.Context.Device, runtime.DefaultDevice);

        using var firstKernel = GpuKernel.Create<MultiContextIncrementKernel>(firstContext);
        Assert.Throws<ArgumentException>(() => GpuKernel.Dispatch(
            secondContext,
            firstKernel,
            new MultiContextIncrementKernel(firstInput.AsReadOnly(), firstOutput.AsReadWrite()),
            new GpuDispatchSize(3, 1, 1),
            wait: true));

        var crossResource = Assert.Throws<FeatherNativeException>(() => GpuKernel.Dispatch(
            firstContext,
            firstKernel,
            new MultiContextIncrementKernel(secondInput.AsReadOnly(), secondOutput.AsReadWrite()),
            new GpuDispatchSize(3, 1, 1),
            wait: true));
        Assert.Equal(FeResult.ErrorInvalidArgument, crossResource.Result);
        Assert.Contains("different GPU context", crossResource.Message, StringComparison.Ordinal);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ContextDisposalInvalidatesOwnedResourcesWithoutChangingDefaultContext()
    {
        var defaultContext = GPU.Context;
        var context = GpuRuntime.Default.CreateContext();
        var operations = GPU.WithContext(context);
        var buffer = operations.CreateBuffer<float>([1.0f]);

        context.Dispose();

        Assert.Same(defaultContext, GPU.Context);
        Assert.Throws<ObjectDisposedException>(() => operations.CreateBuffer<float>(1));
        Assert.Throws<ObjectDisposedException>(() => buffer.ToArray());
        buffer.Dispose();
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct MultiContextIncrementKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = input[ThreadIds.X] + 1.0f;
    }
}
