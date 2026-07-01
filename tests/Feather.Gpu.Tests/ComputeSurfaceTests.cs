using Feather.Interop;
using Feather.Math;

namespace Feather.Gpu.Tests;

public class ComputeSurfaceTests
{
    [Fact]
    public void DispatchSizePreservesThreeDimensions()
    {
        var size = new GpuDispatchSize(5, 6, 7);

        Assert.Equal(5, size.X);
        Assert.Equal(6, size.Y);
        Assert.Equal(7, size.Z);
    }

    [Fact]
    public void GeneratedKernelContractCanBeImplementedByUserPartial()
    {
        var descriptor = GeneratedDescriptor<SmokeKernel>();

        Assert.Equal(KernelDimension.One, descriptor.Dimension);
        Assert.Equal(new int3(256, 1, 1), descriptor.ThreadGroupSize);
        Assert.StartsWith("46454952", Convert.ToHexString(GeneratedIR<SmokeKernel>()));
    }

    private static ReadOnlySpan<byte> GeneratedIR<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
        => TKernel.IR;

    private static KernelDescriptor GeneratedDescriptor<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
        => TKernel.Descriptor;

    private readonly partial struct SmokeKernel : IKernel1D, IGeneratedKernel<SmokeKernel>
    {
        public void Execute()
        {
        }

        static ReadOnlySpan<byte> IGeneratedKernel<SmokeKernel>.IR => "FEIR"u8;

        static KernelDescriptor IGeneratedKernel<SmokeKernel>.Descriptor => new(
            KernelDimension.One,
            new int3(256, 1, 1),
            [],
            [],
            BoundsCheck: true,
            AutoDiff: false,
            DebugName: nameof(SmokeKernel));

        static void IGeneratedKernel<SmokeKernel>.Bind(in SmokeKernel kernel, GpuKernelCommand command)
        {
        }

    }
}
