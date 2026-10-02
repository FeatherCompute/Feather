using Feather.Interop;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>Bisection kernels for the native "stage:LowerStatement" failures.</summary>
public class ShaderDslGapBisectTests
{
    [Fact]
    public void A1_FloatBufferCallable()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA1Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void A2_StructBufferCallableScalarArg()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA2Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void A3_StructBufferCallablePropertyArg()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA3Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void B1_ArrayFieldStructDeclareOnly()
    {
        var glsl = ShaderInspection.GetGLSL<BisectB1Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void B2_ArrayFieldWriteOnly()
    {
        var glsl = ShaderInspection.GetGLSL<BisectB2Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void B3_ArrayFieldReadWrite()
    {
        var glsl = ShaderInspection.GetGLSL<BisectB3Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void C1_InstanceCallableReadOnly()
    {
        var glsl = ShaderInspection.GetGLSL<BisectC1Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void C2_InstanceCallableMutating()
    {
        var glsl = ShaderInspection.GetGLSL<BisectC2Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void A5_ShaderLibraryFloatBuffer()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA5Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void A6_ShaderLibraryStructBuffer()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA6Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void A7_ShaderLibraryStructBufferPropertyArg()
    {
        var glsl = ShaderInspection.GetGLSL<BisectA7Kernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA1Kernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ReadAt(input, i);
    }

    [Callable]
    private static float ReadAt(ReadOnlyBuffer<float> buffer, int index) => buffer[index];
}

[GpuStruct]
public readonly partial record struct BisectWeighted(float Value, float Weight);

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA2Kernel(ReadOnlyBuffer<BisectWeighted> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ScaleAt(input, i, 2.0f);
    }

    [Callable]
    private static float ScaleAt(ReadOnlyBuffer<BisectWeighted> shapes, int index, float factor)
    {
        BisectWeighted shape = shapes[index];
        return shape.Weight * factor;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA3Kernel(ReadOnlyBuffer<BisectWeighted> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = ScaleAt(input, i, input[i].Value);
    }

    [Callable]
    private static float ScaleAt(ReadOnlyBuffer<BisectWeighted> shapes, int index, float factor)
    {
        BisectWeighted shape = shapes[index];
        return shape.Weight * factor;
    }
}

[GpuStruct]
public partial struct BisectLanePacket
{
    public GpuArray4<float> Lanes;
    public int Count;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectB1Kernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var packet = new BisectLanePacket();
        packet.Count = 7;
        output[ThreadIds.X] = packet.Count;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectB2Kernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var packet = new BisectLanePacket();
        packet.Lanes[0] = 3;
        packet.Count = 7;
        output[ThreadIds.X] = packet.Count;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectB3Kernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var packet = new BisectLanePacket();
        packet.Lanes[0] = 3;
        packet.Count = 7;
        output[ThreadIds.X] = packet.Lanes[0] + packet.Count;
    }
}

[GpuStruct]
public partial struct BisectAccumulator
{
    public float Total;

    [Callable]
    public readonly float Get() => Total;

    [Callable]
    public void Add(float value)
    {
        Total = Total + value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectC1Kernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var acc = new BisectAccumulator();
        acc.Total = 5;
        output[ThreadIds.X] = acc.Get();
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectC2Kernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var acc = new BisectAccumulator();
        acc.Total = 5;
        acc.Add(2);
        output[ThreadIds.X] = acc.Total;
    }
}

[ShaderLibrary]
public static class BisectLib
{
    [Callable]
    public static float ReadAt(ReadOnlyBuffer<float> buffer, int index) => buffer[index];

    [Callable]
    public static float ScaleAt(ReadOnlyBuffer<BisectWeighted> shapes, int index, float factor)
    {
        BisectWeighted shape = shapes[index];
        return shape.Weight * factor;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA5Kernel(ReadOnlyBuffer<float> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = BisectLib.ReadAt(input, i);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA6Kernel(ReadOnlyBuffer<BisectWeighted> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = BisectLib.ScaleAt(input, i, 2.0f);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct BisectA7Kernel(ReadOnlyBuffer<BisectWeighted> input, ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = BisectLib.ScaleAt(input, i, input[i].Value);
    }
}
