using Feather.Interop;
using Feather.Resources;

namespace Feather.Integration.Tests;

public class LuisaBackendTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void VectorAddExecutesThroughLuisaXirVulkan()
    {
        float[] leftValues = [1.25f, -2.0f, 3.5f, 8.0f, 0.125f, 42.0f, -7.0f];
        float[] rightValues = [0.75f, 5.0f, -1.5f, 2.0f, 0.875f, -40.0f, 9.0f];
        var expected = leftValues.Zip(rightValues, static (left, right) => left + right).ToArray();

        using var left = GPU.CreateBuffer<float>(leftValues);
        using var right = GPU.CreateBuffer<float>(rightValues);
        using var easyGpuOutput = GPU.CreateBuffer<float>(expected.Length);
        using var luisaOutput = GPU.CreateBuffer<float>(expected.Length);

        GPU.Dispatch(
            new LuisaVectorAddKernel(left.AsReadOnly(), right.AsReadOnly(), easyGpuOutput.AsReadWrite()),
            expected.Length);

        using var luisaKernel = GpuKernel.Create<LuisaVectorAddKernel>(
            GPU.Context,
            GpuExecutionBackend.Luisa);
        GpuKernel.Dispatch(
            GPU.Context,
            luisaKernel,
            new LuisaVectorAddKernel(left.AsReadOnly(), right.AsReadOnly(), luisaOutput.AsReadWrite()),
            new GpuDispatchSize(expected.Length, 1, 1),
            wait: true);

        Assert.Equal(expected, easyGpuOutput.ToArray());
        Assert.Equal(expected, luisaOutput.ToArray());
        Assert.Equal(DispatchPath.Luisa, luisaKernel.LastDispatchPath);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LuisaVectorAddKernel(
    ReadOnlyBuffer<float> left,
    ReadOnlyBuffer<float> right,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = left[ThreadIds.X] + right[ThreadIds.X];
    }
}
