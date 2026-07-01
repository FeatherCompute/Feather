using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

SampleProof.PrintBackend(GPU.Context);

Console.WriteLine("Shader Inspection:");
string glsl = ShaderInspection.GetGLSL<InspectKernel>();
Console.WriteLine("GLSL source obtained successfully");
Console.WriteLine($"Contains gl_GlobalInvocationID: {glsl.Contains("gl_GlobalInvocationID", StringComparison.Ordinal)}");
Console.WriteLine($"Contains #version: {glsl.Contains("#version", StringComparison.Ordinal)}");
if (!SampleProof.IsEasyGpuGlsl(glsl))
{
    throw new InvalidOperationException("InspectKernel did not produce EasyGPU GLSL.");
}

using var input = GPU.CreateBuffer<float>([1.0f, 2.0f, 3.0f, 4.0f], BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(4, BufferAccess.ReadWrite);
var path = GPU.DispatchAndGetPath(new InspectKernel(input.AsReadOnly(), output.AsReadWrite()), 4);
SampleProof.AssertTypedEasyGpu(path);
AssertOutput(output.ToArray());
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine("PASS");

static void AssertOutput(float[] output)
{
    float[] expected = [2.0f, 4.0f, 6.0f, 8.0f];
    for (var i = 0; i < expected.Length; i++)
    {
        if (MathF.Abs(output[i] - expected[i]) > 1e-6f)
        {
            throw new InvalidOperationException("SpirvOptInspection validation failed.");
        }
    }
}

/// <summary>
/// Doubles each input element so shader inspection can be paired with dispatch validation.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct InspectKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Writes twice the current input element.
    /// </summary>
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * 2.0f;
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
    /// Checks whether GLSL came from the EasyGPU bridge rather than a stub.
    /// </summary>
    public static bool IsEasyGpuGlsl(string glsl)
        => glsl.Contains("gl_GlobalInvocationID", StringComparison.Ordinal) &&
           glsl.Contains("#version", StringComparison.Ordinal) &&
           !glsl.Contains("Feather native stub", StringComparison.Ordinal);

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
