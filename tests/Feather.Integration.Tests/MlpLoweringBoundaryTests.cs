using Feather;
using Feather.Interop;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// Pins the <c>[Callable]</c> resource-parameter boundary that shapes how MLP inference is exposed.
/// </summary>
/// <remarks>
/// Read-only scalar and vector buffers lower by specializing the callable's resource access to the bound
/// SSBO. Writable buffers remain outside the phase-1 subset. <c>MlpShader</c> still exposes layout
/// arithmetic and inference still ships as <see cref="Feather.NN.MlpInference3To1Kernel" /> until that
/// API is deliberately consolidated.
/// </remarks>
public class MlpLoweringBoundaryTests
{
    [Fact]
    public void ScalarCallableParametersLower()
    {
        var glsl = ShaderInspection.GetGLSL<CallableScalarProbeKernel>();

        Assert.Contains("ScalarOnly", glsl, StringComparison.Ordinal);
        Assert.DoesNotContain("Feather native stub", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadOnlyBufferCallableParametersLowerAsGlobalResourceAccessors()
    {
        var glsl = ShaderInspection.GetGLSL<CallableReadOnlyBufferProbeKernel>();

        Assert.Contains("FromReadOnlyBuffer", glsl, StringComparison.Ordinal);
        Assert.Contains("(int index)", glsl, StringComparison.Ordinal);
        Assert.Contains("fe_0[", glsl, StringComparison.Ordinal);
    }

    [Fact]
    public void ReadWriteBufferCallableParametersDoNotLower()
    {
        var ex = Assert.Throws<FeatherNativeException>(ShaderInspection.GetGLSL<CallableReadWriteBufferProbeKernel>);

        Assert.Contains("typed IR", ex.Message, StringComparison.Ordinal);
    }
}

/// <summary>Minimal callables isolating the parameter-type boundary.</summary>
[ShaderLibrary]
public static class CallableBoundaryProbeLibrary
{
    /// <summary>A scalar-only callable, which lowers.</summary>
    /// <param name="value">Any integer.</param>
    [Callable]
    public static int ScalarOnly(int value)
    {
        return value * 2;
    }

    /// <summary>A callable taking a read-only buffer, lowered as a global resource accessor.</summary>
    /// <param name="source">The buffer to read.</param>
    /// <param name="index">The element to read.</param>
    [Callable]
    public static float FromReadOnlyBuffer(ReadOnlyBuffer<float> source, int index)
    {
        return source[index];
    }

    /// <summary>A callable taking a read-write buffer, which does not lower.</summary>
    /// <param name="destination">The buffer to write.</param>
    /// <param name="index">The element to write.</param>
    /// <param name="value">The value to store.</param>
    [Callable]
    public static void IntoReadWriteBuffer(ReadWriteBuffer<float> destination, int index, float value)
    {
        destination[index] = value;
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CallableScalarProbeKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = CallableBoundaryProbeLibrary.ScalarOnly(ThreadIds.X);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CallableReadOnlyBufferProbeKernel(
    ReadOnlyBuffer<float> source,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = CallableBoundaryProbeLibrary.FromReadOnlyBuffer(source, ThreadIds.X);
    }
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct CallableReadWriteBufferProbeKernel(ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        CallableBoundaryProbeLibrary.IntoReadWriteBuffer(output, ThreadIds.X, 1f);
    }
}
