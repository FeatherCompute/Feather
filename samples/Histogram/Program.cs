using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Simple buffer copy kernel (avoids atomic race conditions)
const int N = 1024;
SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<CopyKernel>();

float[] data = new float[N];
for (int i = 0; i < N; i++) data[i] = i;

using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);
var path = GPU.DispatchAndGetPath(new CopyKernel(input.AsReadOnly(), output.AsReadWrite()), N);
SampleProof.AssertTypedEasyGpu(path);

float[] result = output.ToArray();
bool pass = true;
for (int i = 0; i < N; i++)
    if (Math.Abs(result[i] - data[i]) > 0.01f)
        pass = false;
Console.WriteLine($"Dispatch path: {path}");
if (!pass)
{
    throw new InvalidOperationException("Histogram copy validation failed.");
}

Console.WriteLine($"PASS: {N} elements copied correctly");

/// <summary>
/// Copies one input value to the output buffer.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct CopyKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Copies the current global thread's element.
    /// </summary>
    public void Execute()
    {
        int i = ThreadIds.X;
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
        Console.WriteLine(
            $"Max workgroup size: {caps.MaxWorkGroupSizeX}x{caps.MaxWorkGroupSizeY}x{caps.MaxWorkGroupSizeZ}");
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
