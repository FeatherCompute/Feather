using Feather.Interop;

namespace Feather.Integration.Tests;

public class LuisaBackendControlMemoryTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void StructuredControlFlowAndMutableLocalsStaticAndExplicitLuisaAgree()
    {
        float[] values = [1, 2, 3, 4, 5, 6, 7];
        using var input = GPU.CreateBuffer<float>(values);

        AssertKernelParity(
            3,
            staticOutput => new BreakContinueLoopKernel(input.AsReadOnly(), staticOutput),
            luisa => new BreakContinueLoopKernel(input.AsReadOnly(), luisa));
        AssertKernelParity(
            3,
            staticOutput => new WhileAccumulateKernel(input.AsReadOnly(), staticOutput),
            luisa => new WhileAccumulateKernel(input.AsReadOnly(), luisa));
        AssertKernelParity(
            3,
            staticOutput => new DoWhileAccumulateKernel(input.AsReadOnly(), staticOutput),
            luisa => new DoWhileAccumulateKernel(input.AsReadOnly(), luisa));

        float[] signed = [-2, 0, 3];
        using var signedInput = GPU.CreateBuffer<float>(signed);
        AssertKernelParity(
            signed.Length,
            staticOutput => new IfThresholdKernel(signedInput.AsReadOnly(), staticOutput),
            luisa => new IfThresholdKernel(signedInput.AsReadOnly(), luisa));
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void SharedMemoryLocalIdsAndBarrierStaticAndExplicitLuisaAgree()
    {
        float[] values = [1.25f, -2, 9, 4.5f];
        using var input = GPU.CreateBuffer<float>(values);
        AssertKernelParity(
            values.Length,
            staticOutput => new SharedFloatCopyKernel(input.AsReadOnly(), staticOutput),
            luisa => new SharedFloatCopyKernel(input.AsReadOnly(), luisa));
    }

    private static void AssertKernelParity<TKernel>(
        int count,
        Func<Resources.ReadWriteBuffer<float>, TKernel> easyKernel,
        Func<Resources.ReadWriteBuffer<float>, TKernel> luisaKernel)
        where TKernel : struct, IKernel1D, IGeneratedKernel<TKernel>
    {
        using var staticOutput = GPU.CreateBuffer<float>(count);
        using var luisa = GPU.CreateBuffer<float>(count);
        GPU.Dispatch(easyKernel(staticOutput.AsReadWrite()), count);
        using var compiled = GpuKernel.Create<TKernel>(GPU.Context, GpuExecutionBackend.Luisa);
        GpuKernel.Dispatch(
            GPU.Context,
            compiled,
            luisaKernel(luisa.AsReadWrite()),
            new GpuDispatchSize(count, 1, 1),
            wait: true);
        Assert.Equal(DispatchPath.Luisa, compiled.LastDispatchPath);
        Assert.Equal(staticOutput.ToArray(), luisa.ToArray());
    }
}
