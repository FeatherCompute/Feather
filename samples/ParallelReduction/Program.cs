using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int N = 4;

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<BarrierCopyKernel>();

float[] data = [1.0f, 2.0f, 3.0f, 4.0f];
using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);

// Kernel that copies buffer with a barrier.
var path = GPU.DispatchAndGetPath(new BarrierCopyKernel(input.AsReadOnly(), output.AsReadWrite()), N);
SampleProof.AssertTypedEasyGpu(path);

float[] result = output.ToArray();
Console.WriteLine($"Input:  {string.Join(", ", data)}");
Console.WriteLine($"Output: {string.Join(", ", result)}");
Console.WriteLine($"Dispatch path: {path}");

if (!data.SequenceEqual(result))
{
    throw new InvalidOperationException("ParallelReduction validation failed.");
}

Console.WriteLine("PASS");

/// <summary>
/// Copies input to output after issuing a workgroup barrier.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct BarrierCopyKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Waits at a workgroup barrier, then copies one element.
    /// </summary>
    public void Execute()
    {
        var i = ThreadIds.X;
        GpuBarrier.Workgroup();
        output[i] = input[i];
    }
}

/// <summary>
/// Common runtime checks used by the sample before it prints PASS.
/// </summary>
internal static class SampleProof
{
    /// <summary>
    /// Prints the active backend and workgroup limits reported by EasyGPU.
    /// </summary>
    public static void PrintBackend(GpuContext context)
    {
        var caps = context.Caps;
        Console.WriteLine($"Backend: {caps.BackendType}");
        Console.WriteLine($"Max workgroup size: {caps.MaxWorkGroupSizeX}x{caps.MaxWorkGroupSizeY}x{caps.MaxWorkGroupSizeZ}");
    }

    /// <summary>
    /// Verifies the generated source comes from the EasyGPU GLSL bridge.
    /// </summary>
    public static void AssertEasyGpuGlsl<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var glsl = ShaderInspection.GetGLSL<TKernel>();
        if (!glsl.Contains("gl_GlobalInvocationID", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce EasyGPU GLSL.");
        }

        Console.WriteLine("EasyGPU GLSL bridge: OK");
    }

    /// <summary>
    /// Requires the dispatch to have used the typed EasyGPU backend path.
    /// </summary>
    public static void AssertTypedEasyGpu(DispatchPath path)
    {
        if (path != DispatchPath.TypedEasyGpu)
        {
            throw new InvalidOperationException($"Expected TypedEasyGpu dispatch, got {path}.");
        }
    }
}
