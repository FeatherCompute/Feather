using Feather.Interop;
using Feather.Math;

namespace Feather.Integration.Tests;

public class LuisaBackendTypeFeatureTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void ScalarsComparisonsLogicAndBitOperationsStaticAndExplicitLuisaAgree()
    {
        float[] floatValues = [-3.5f, -1.0f, 0.0f, 1.25f, 8.0f];
        using var floats = GPU.CreateBuffer<float>(floatValues);
        using var easyFloat = GPU.CreateBuffer<float>(floatValues.Length);
        using var luisaFloat = GPU.CreateBuffer<float>(floatValues.Length);

        GPU.Dispatch(new LogicalPredicateKernel(floats.AsReadOnly(), easyFloat.AsReadWrite()), floatValues.Length);
        DispatchLuisa(new LogicalPredicateKernel(floats.AsReadOnly(), luisaFloat.AsReadWrite()), floatValues.Length);
        Assert.Equal(easyFloat.ToArray(), luisaFloat.ToArray());

        int[] intValues = [-8, -1, 0, 3, 17];
        using var ints = GPU.CreateBuffer<int>(intValues);
        using var easyInt = GPU.CreateBuffer<int>(intValues.Length);
        using var luisaInt = GPU.CreateBuffer<int>(intValues.Length);
        GPU.Dispatch(new BitwiseShiftKernel(ints.AsReadOnly(), easyInt.AsReadWrite()), intValues.Length);
        DispatchLuisa(new BitwiseShiftKernel(ints.AsReadOnly(), luisaInt.AsReadWrite()), intValues.Length);
        Assert.Equal(easyInt.ToArray(), luisaInt.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void UnsignedConstantsBitOperationsComparisonsAndConversionsStaticAndExplicitLuisaAgree()
    {
        int[] signedValues = [-8, -1, 0, 3, 17];
        uint[] unsignedValues = [0u, 1u, 7u, 31u, 0x80000000u];
        float[] floatValues = [-3.5f, -1.0f, 0.0f, 1.25f, 8.0f];
        using var signed = GPU.CreateBuffer<int>(signedValues);
        using var unsigned = GPU.CreateBuffer<uint>(unsignedValues);
        using var floats = GPU.CreateBuffer<float>(floatValues);
        using var staticOutput = GPU.CreateBuffer<int>(signedValues.Length);
        using var luisa = GPU.CreateBuffer<int>(signedValues.Length);

        GPU.Dispatch(new LuisaScalarMatrixKernel(
            signed.AsReadOnly(), unsigned.AsReadOnly(), floats.AsReadOnly(), staticOutput.AsReadWrite()), signedValues.Length);
        DispatchLuisa(new LuisaScalarMatrixKernel(
            signed.AsReadOnly(), unsigned.AsReadOnly(), floats.AsReadOnly(), luisa.AsReadWrite()), signedValues.Length);
        Assert.Equal(staticOutput.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void VectorConstructionAndSwizzlesStaticAndExplicitLuisaAgree()
    {
        float4[] values =
        [
            new(1, 2, 3, 4),
            new(-2, 0.5f, 9, -7),
            new(10, 20, 30, 40)
        ];
        using var input = GPU.CreateBuffer<float4>(values);
        using var staticOutput = GPU.CreateBuffer<float4>(values.Length);
        using var luisa = GPU.CreateBuffer<float4>(values.Length);
        GPU.Dispatch(new ExpandedSwizzleKernel(input.AsReadOnly(), staticOutput.AsReadWrite()), values.Length);
        DispatchLuisa(new ExpandedSwizzleKernel(input.AsReadOnly(), luisa.AsReadWrite()), values.Length);
        Assert.Equal(staticOutput.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void MatrixConstructionAndLinearAlgebraStaticAndExplicitLuisaAgree()
    {
        const int count = 5;
        using var staticOutput = GPU.CreateBuffer<float2>(count);
        using var luisa = GPU.CreateBuffer<float2>(count);
        GPU.Dispatch(new Matrix2VectorMultiplyKernel(staticOutput.AsReadWrite()), count);
        DispatchLuisa(new Matrix2VectorMultiplyKernel(luisa.AsReadWrite()), count);
        Assert.Equal(staticOutput.ToArray(), luisa.ToArray());
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void StructAggregateLoadsAndNestedFieldExtractionStaticAndExplicitLuisaAgree()
    {
        NestedScene[] values =
        [
            new() { Scene = new TypedScene { LightDir = new float3(1, 2, 3), Intensity = 4 }, Weight = new float4(10, 0, 0, 0) },
            new() { Scene = new TypedScene { LightDir = new float3(5, 6, 7), Intensity = 8 }, Weight = new float4(1, 0, 0, 0) }
        ];
        using var input = GPU.CreateBuffer<NestedScene>(values);
        using var staticOutput = GPU.CreateBuffer<float>(values.Length);
        using var luisa = GPU.CreateBuffer<float>(values.Length);
        GPU.Dispatch(new NestedStructFieldReadKernel(input.AsReadOnly(), staticOutput.AsReadWrite()), values.Length);
        DispatchLuisa(new NestedStructFieldReadKernel(input.AsReadOnly(), luisa.AsReadWrite()), values.Length);
        Assert.Equal(staticOutput.ToArray(), luisa.ToArray());
    }

    private static void DispatchLuisa<TKernel>(TKernel kernel, int count)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        using var compiled = GpuKernel.Create<TKernel>(GPU.Context, GpuExecutionBackend.Luisa);
        GpuKernel.Dispatch(GPU.Context, compiled, kernel, new GpuDispatchSize(count, 1, 1), wait: true);
        Assert.Equal(DispatchPath.Luisa, compiled.LastDispatchPath);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LuisaScalarMatrixKernel(
    Resources.ReadOnlyBuffer<int> signed,
    Resources.ReadOnlyBuffer<uint> unsigned,
    Resources.ReadOnlyBuffer<float> floats,
    Resources.ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        uint bits = ((unsigned[i] << 2) ^ 0xAu) | 1u;
        bool predicate = bits > unsigned[i] && floats[i] != 0.0f;
        int converted = (int)bits + (int)floats[i] + signed[i];
        output[i] = predicate ? converted : -converted;
    }
}
