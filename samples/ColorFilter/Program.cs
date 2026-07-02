using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Color filter using simple buffer copy (no bit operations)
const int N = 16;
SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<FilterKernel>();

var input = new float[N];
var output = new float[N];
for (var i = 0; i < N; i++)
{
    input[i] = i * 16;
}

using var bufInput = GPU.CreateBuffer<float>(input, BufferAccess.ReadOnly);
using var bufOutput = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);

var path = GPU.DispatchAndGetPath(new FilterKernel(bufInput.AsReadOnly(), bufOutput.AsReadWrite()), N);
SampleProof.AssertTypedEasyGpu(path);

var result = bufOutput.ToArray();
Console.Write("Input:  ");
foreach (var v in input)
{
    Console.Write($"{v:F0} ");
}

Console.WriteLine();
Console.Write("Output: ");
foreach (var v in result)
{
    Console.Write($"{v:F0} ");
}

Console.WriteLine();
Console.WriteLine($"Dispatch path: {path}");

var pass = true;
for (var i = 0; i < N; i++)
{
    if (Math.Abs(result[i] - input[i] * 0.5f) > 0.01f)
    {
        pass = false;
    }
}

if (!pass)
{
    throw new InvalidOperationException("ColorFilter validation failed.");
}

Console.WriteLine("PASS");

/// <summary>
/// Scales each color channel value by half.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct FilterKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Writes one filtered value for the current global thread index.
    /// </summary>
    public void Execute()
    {
        var i = ThreadIds.X;
        output[i] = input[i] * 0.5f;
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
