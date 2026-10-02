using Feather.Interop;
using Feather.Math;
using Feather.Resources;
using System.Runtime.CompilerServices;

namespace Feather.Integration.Tests;

/// <summary>
/// Runtime coverage for the five shader-DSL gaps fixed in 0.4.0-preview.2:
/// local GpuStruct field mutation (incl. instance-method this mutation), [InlineArray]
/// fields in shader-local structs, [Callable] ref/out parameters, generic callables taking
/// ReadOnlyBuffer&lt;TShape&gt;, and generic type-parameter locals. GLSL inspection covers the
/// native typed-IR lowering path; dispatch assertions cover end-to-end execution.
/// </summary>
public class ShaderDslGapRuntimeTests
{
    [Fact]
    public void LocalStructFieldMutation_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapFieldMutationKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapFieldMutationKernel(output.AsReadWrite()), 4);
        Assert.Equal([1.5f, 1.5f, 1.5f, 1.5f], output.ToArray());
    }

    [Fact]
    public void InstanceMethodThisMutation_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapThisMutationKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapThisMutationKernel(output.AsReadWrite()), 4);
        Assert.Equal([2f, 2f, 2f, 2f], output.ToArray());
    }

    [Fact]
    public void InlineArrayField_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapInlineArrayKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapInlineArrayKernel(output.AsReadWrite()), 4);
        Assert.Equal([11f, 11f, 11f, 11f], output.ToArray());
    }

    [Fact]
    public void RefOutCallable_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapRefOutKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapRefOutKernel(output.AsReadWrite()), 4);
        Assert.Equal([6f, 6f, 6f, 6f], output.ToArray());
    }

    [Fact]
    public void GenericCallableWithBufferOfTypeParameter_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapGenericBufferKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<GapWeighted>([new GapWeighted(5, 2), new GapWeighted(7, 3)]);
        using var output = GPU.CreateBuffer<float>(2);
        GPU.Dispatch(new GapGenericBufferKernel(input.AsReadOnly(), output.AsReadWrite()), 2);
        Assert.Equal([10f, 21f], output.ToArray());
    }

    [Fact]
    public void GenericTypeParameterLocal_LowersAndDispatches()
    {
        var glsl = ShaderInspection.GetGLSL<GapGenericLocalKernel>();
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);

        using var input = GPU.CreateBuffer<float>([1, 2, 3, 4]);
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapGenericLocalKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
        Assert.Equal([2f, 4f, 6f, 8f], output.ToArray());
    }
}

[GpuStruct]
public partial struct GapCounter
{
    public int Count;
    public float Weight;

    public GapCounter(int count, float weight)
    {
        Count = count;
        Weight = weight;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapFieldMutationKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var s = new GapCounter(3, 0.5f);
        s.Count = 0;
        s.Weight = s.Weight + 1;
        output[ThreadIds.X] = s.Count + s.Weight;
    }
}

[GpuStruct]
public partial struct GapAccumulator
{
    public float Total;

    [Callable]
    public void Reset()
    {
        Total = 0;
    }

    [Callable]
    public void Add(float value)
    {
        Total = Total + value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapThisMutationKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var acc = new GapAccumulator();
        acc.Total = 5;
        acc.Reset();
        acc.Add(2);
        output[ThreadIds.X] = acc.Total;
    }
}

[InlineArray(4)]
public struct GapFloatLanes
{
    private float _element0;
}

[GpuStruct]
public partial struct GapLanePacket
{
    public GapFloatLanes Lanes;
    public int Count;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapInlineArrayKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var packet = new GapLanePacket();
        packet.Lanes[0] = 3;
        packet.Lanes[1] = 8;
        packet.Count = 2;
        output[ThreadIds.X] = packet.Lanes[0] + packet.Lanes[packet.Count - 1];
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapRefOutKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        float a = 1;
        float b = 2;
        Swap(ref a, ref b);
        float sum;
        TakeSum(a, b, out sum);
        output[ThreadIds.X] = sum + a + b;
    }

    [Callable]
    private static void Swap(ref float x, ref float y)
    {
        float t = x;
        x = y;
        y = t;
    }

    [Callable]
    private static void TakeSum(float x, float y, out float result)
    {
        result = x + y;
    }
}

public interface IGapWeighted
{
    float Weight { get; }
}

[GpuStruct]
public readonly partial record struct GapWeighted(float Value, float Weight) : IGapWeighted;

[ShaderLibrary]
public static class GapQueries
{
    [Callable]
    public static float ScaleAt<TShape>(ReadOnlyBuffer<TShape> shapes, int index, float factor)
        where TShape : unmanaged, IGapWeighted
    {
        TShape shape = shapes[index];
        return shape.Weight * factor;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapGenericBufferKernel(
    ReadOnlyBuffer<GapWeighted> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = GapQueries.ScaleAt(input, i, input[i].Value);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapGenericLocalKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = Echo(input[i]) + input[i];
    }

    [Callable]
    private static T Echo<T>(T value) where T : unmanaged
    {
        T local = value;
        T copy = local;
        return copy;
    }
}

public class ShaderDslGapBaselineTests
{
    [Fact]
    public void NonGenericBufferCallable_LowersAndDispatches()
    {
        using var input = GPU.CreateBuffer<GapWeightedPlain>([new GapWeightedPlain(5, 2), new GapWeightedPlain(7, 3)]);
        using var output = GPU.CreateBuffer<float>(2);
        GPU.Dispatch(new GapPlainBufferKernel(input.AsReadOnly(), output.AsReadWrite()), 2);
        Assert.Equal([10f, 21f], output.ToArray());
    }

    [Fact]
    public void GpuArrayField_LowersAndDispatches()
    {
        using var output = GPU.CreateBuffer<float>(4);
        GPU.Dispatch(new GapGpuArrayKernel(output.AsReadWrite()), 4);
        Assert.Equal([11f, 11f, 11f, 11f], output.ToArray());
    }
}

[GpuStruct]
public readonly partial record struct GapWeightedPlain(float Value, float Weight);

[ShaderLibrary]
public static class GapPlainQueries
{
    [Callable]
    public static float ScaleAt(ReadOnlyBuffer<GapWeightedPlain> shapes, int index, float factor)
    {
        GapWeightedPlain shape = shapes[index];
        return shape.Weight * factor;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapPlainBufferKernel(
    ReadOnlyBuffer<GapWeightedPlain> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = GapPlainQueries.ScaleAt(input, i, input[i].Value);
    }
}

[GpuStruct]
public partial struct GapGpuLanePacket
{
    public GpuArray4<float> Lanes;
    public int Count;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct GapGpuArrayKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        var packet = new GapGpuLanePacket();
        packet.Lanes[0] = 3;
        packet.Lanes[1] = 8;
        packet.Count = 2;
        output[ThreadIds.X] = packet.Lanes[0] + packet.Lanes[packet.Count - 1];
    }
}
