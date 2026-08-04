using Feather.Interop;

namespace Feather.Integration.Tests;

public class LuisaBackendControlMemoryTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void StructuredControlFlowAndMutableLocalsMatchEasyGpu()
    {
        float[] values = [1, 2, 3, 4, 5, 6, 7];
        using var input = GPU.CreateBuffer<float>(values);

        AssertKernelParity(
            3,
            easy => new BreakContinueLoopKernel(input.AsReadOnly(), easy),
            luisa => new BreakContinueLoopKernel(input.AsReadOnly(), luisa));
        AssertKernelParity(
            3,
            easy => new WhileAccumulateKernel(input.AsReadOnly(), easy),
            luisa => new WhileAccumulateKernel(input.AsReadOnly(), luisa));
        AssertKernelParity(
            3,
            easy => new DoWhileAccumulateKernel(input.AsReadOnly(), easy),
            luisa => new DoWhileAccumulateKernel(input.AsReadOnly(), luisa));

        float[] signed = [-2, 0, 3];
        using var signedInput = GPU.CreateBuffer<float>(signed);
        AssertKernelParity(
            signed.Length,
            easy => new IfThresholdKernel(signedInput.AsReadOnly(), easy),
            luisa => new IfThresholdKernel(signedInput.AsReadOnly(), luisa));
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void SharedMemoryLocalIdsAndBarrierMatchEasyGpu()
    {
        float[] values = [1.25f, -2, 9, 4.5f];
        using var input = GPU.CreateBuffer<float>(values);
        AssertKernelParity(
            values.Length,
            easy => new SharedFloatCopyKernel(input.AsReadOnly(), easy),
            luisa => new SharedFloatCopyKernel(input.AsReadOnly(), luisa));
    }

    private static void AssertKernelParity<TKernel>(
        int count,
        Func<Resources.ReadWriteBuffer<float>, TKernel> easyKernel,
        Func<Resources.ReadWriteBuffer<float>, TKernel> luisaKernel)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
    {
        using var easy = GPU.CreateBuffer<float>(count);
        using var luisa = GPU.CreateBuffer<float>(count);
        GPU.Dispatch(easyKernel(easy.AsReadWrite()), count);
        using var compiled = GpuKernel.Create<TKernel>(GPU.Context, GpuExecutionBackend.Luisa);
        GpuKernel.Dispatch(
            GPU.Context,
            compiled,
            luisaKernel(luisa.AsReadWrite()),
            new GpuDispatchSize(count, 1, 1),
            wait: true);
        Assert.Equal(DispatchPath.Luisa, compiled.LastDispatchPath);
        Assert.Equal(easy.ToArray(), luisa.ToArray());
    }
}
