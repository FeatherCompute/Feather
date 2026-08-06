using Feather.Interop;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(GpuMultiContextCollection.Name)]
public class GpuStreamTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public async Task AsyncDispatchFenceCompletesAndRestoresHostResults()
    {
        using var context = GpuRuntime.Default.CreateContext();
        using var stream = context.CreateStream();
        using var input = GpuBuffer<float>.Create(context, [1.0f, 2.0f, 3.0f], BufferAccess.ReadWrite);
        using var output = GpuBuffer<float>.Create(context, 3, BufferAccess.ReadWrite);

        using var fence = stream.Dispatch(
            new StreamIncrementKernel(input.AsReadOnly(), output.AsReadWrite()), 3);
        await fence.WaitAsync();

        Assert.True(fence.IsCompleted);
        Assert.Equal([2.0f, 3.0f, 4.0f], output.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void TwoStreamsExecuteIndependentlyAndCanOrderSharedResources()
    {
        using var context = GpuRuntime.Default.CreateContext();
        using var first = context.CreateStream();
        using var second = context.CreateStream();
        using var firstInput = GpuBuffer<float>.Create(context, [2.0f, 4.0f], BufferAccess.ReadWrite);
        using var secondInput = GpuBuffer<float>.Create(context, [10.0f, 20.0f], BufferAccess.ReadWrite);
        using var intermediate = GpuBuffer<float>.Create(context, 2, BufferAccess.ReadWrite);
        using var secondOutput = GpuBuffer<float>.Create(context, 2, BufferAccess.ReadWrite);
        using var dependentOutput = GpuBuffer<float>.Create(context, 2, BufferAccess.ReadWrite);
        using var orderedOutput = GpuBuffer<float>.Create(context, 2, BufferAccess.ReadWrite);

        using var firstFence = first.Dispatch(
            new StreamIncrementKernel(firstInput.AsReadOnly(), intermediate.AsReadWrite()), 2);
        using var secondFence = second.Dispatch(
            new StreamIncrementKernel(secondInput.AsReadOnly(), secondOutput.AsReadWrite()), 2);
        second.Wait(firstFence);
        using var dependentFence = second.Dispatch(
            new StreamIncrementKernel(intermediate.AsReadOnly(), dependentOutput.AsReadWrite()), 2);
        using var earlierFence = first.Dispatch(
            new StreamIncrementKernel(firstInput.AsReadOnly(), orderedOutput.AsReadWrite()), 2);
        using var laterFence = first.Dispatch(
            new StreamIncrementKernel(secondInput.AsReadOnly(), orderedOutput.AsReadWrite()), 2);

        dependentFence.Wait();
        secondFence.Wait();
        laterFence.Wait();
        Assert.Equal([11.0f, 21.0f], secondOutput.ToArray());
        Assert.Equal([4.0f, 6.0f], dependentOutput.ToArray());
        Assert.Equal([11.0f, 21.0f], orderedOutput.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void CrossContextWaitIsRejectedAndDisposalSynchronizesInFlightWork()
    {
        using var firstContext = GpuRuntime.Default.CreateContext();
        using var secondContext = GpuRuntime.Default.CreateContext();
        using var firstStream = firstContext.CreateStream();
        using var secondStream = secondContext.CreateStream();
        using var input = GpuBuffer<float>.Create(firstContext, [3.0f], BufferAccess.ReadWrite);
        var output = GpuBuffer<float>.Create(firstContext, 1, BufferAccess.ReadWrite);
        using var fence = firstStream.Dispatch(
            new StreamIncrementKernel(input.AsReadOnly(), output.AsReadWrite()), 1);

        var error = Assert.Throws<ArgumentException>(() => secondStream.Wait(fence));
        Assert.Contains("different GPU context", error.Message, StringComparison.Ordinal);

        output.Dispose();
        Assert.True(fence.IsCompleted);
        firstStream.Dispose();
        fence.Wait();
        Assert.Throws<ObjectDisposedException>(() => firstStream.Synchronize());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void LegacyWaitFalseRetainsNativeSubmissionStateUntilHostRead()
    {
        using var context = GpuRuntime.Default.CreateContext();
        using var input = GpuBuffer<float>.Create(context, [5.0f, 7.0f], BufferAccess.ReadWrite);
        using var output = GpuBuffer<float>.Create(context, 2, BufferAccess.ReadWrite);
        using var kernel = GpuKernel.Create<StreamIncrementKernel>(context);

        GpuKernel.Dispatch(
            context,
            kernel,
            new StreamIncrementKernel(input.AsReadOnly(), output.AsReadWrite()),
            new GpuDispatchSize(2, 1, 1),
            wait: false);

        Assert.Equal([6.0f, 8.0f], output.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void VulkanStreamsAndFencesSurviveContextActivationSwitches()
    {
        var runtime = GpuRuntime.Create();
        var device = Assert.Single(runtime.EnumerateDevices(), static device =>
            device.Backend == GpuBackend.Vulkan && device.DeviceIndex == 0);
        var options = new GpuContextOptions
        {
            Backend = GpuBackend.Vulkan,
            DeviceIndex = device.DeviceIndex
        };
        using var firstContext = runtime.CreateContext(options);
        using var secondContext = runtime.CreateContext(options);
        using var firstStream = firstContext.CreateStream();
        using var secondStream = secondContext.CreateStream();
        using var firstInput = GpuBuffer<float>.Create(firstContext, [1.0f, 2.0f], BufferAccess.ReadWrite);
        using var firstOutput = GpuBuffer<float>.Create(firstContext, 2, BufferAccess.ReadWrite);
        using var secondInput = GpuBuffer<float>.Create(secondContext, [10.0f, 20.0f], BufferAccess.ReadWrite);
        using var secondOutput = GpuBuffer<float>.Create(secondContext, 2, BufferAccess.ReadWrite);

        using var firstFence = firstStream.Dispatch(
            new StreamIncrementKernel(firstInput.AsReadOnly(), firstOutput.AsReadWrite()), 2);
        using var secondFence = secondStream.Dispatch(
            new StreamIncrementKernel(secondInput.AsReadOnly(), secondOutput.AsReadWrite()), 2);

        Assert.True(firstFence.IsCompleted);
        using var resumedFence = firstStream.Dispatch(
            new StreamIncrementKernel(firstInput.AsReadOnly(), firstOutput.AsReadWrite()), 2);
        resumedFence.Wait();
        secondFence.Wait();

        Assert.Equal([2.0f, 3.0f], firstOutput.ToArray());
        Assert.Equal([11.0f, 21.0f], secondOutput.ToArray());
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct StreamIncrementKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = input[ThreadIds.X] + 1.0f;
    }
}
