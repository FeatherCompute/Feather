using Feather;
using Feather.Interop;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

/// <summary>
/// Pins the <c>[Callable]</c> parameter-type boundary that shapes how MLP inference is exposed.
/// </summary>
/// <remarks>
/// The generator's <c>IsSupportedCallableType</c> accepts shader resource views as <c>[Callable]</c>
/// parameters, but <c>GPU::IR::Type</c> has no buffer kind, so <c>GPU::IR::CallableParameter</c> cannot
/// represent one and the native typed-IR lowerer fails at dispatch rather than at compile time. That
/// disagreement is why <c>MlpShader</c> exposes only layout arithmetic and inference ships as
/// <see cref="Feather.NN.MlpInference3To1Kernel" />.
///
/// These tests assert the current behavior, including the failure. If a future EasyGPU release gains a
/// resource type in its IR, the buffer cases here start passing and this file is the signal that
/// <c>MlpShader</c> can be given a real <c>Evaluate3To1</c> and the duplicated arithmetic collapsed.
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
    public void ReadOnlyBufferCallableParametersDoNotLower()
    {
        // Accepted by the generator, rejected by the native lowerer. Documented here because the failure
        // is opaque and arrives at dispatch time.
        var ex = Assert.Throws<FeatherNativeException>(ShaderInspection.GetGLSL<CallableReadOnlyBufferProbeKernel>);

        Assert.Contains("typed IR", ex.Message, StringComparison.Ordinal);
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

    /// <summary>A callable taking a read-only buffer, which does not lower.</summary>
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
