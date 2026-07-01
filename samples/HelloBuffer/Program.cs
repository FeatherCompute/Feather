using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Create input data and output buffer
float[] data = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f];
SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<DoubleKernel>();

using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(data.Length, BufferAccess.ReadWrite);

// Create and dispatch the kernel
var path = GPU.DispatchAndGetPath(new DoubleKernel(input.AsReadOnly(), output.AsReadWrite()), data.Length);
SampleProof.AssertTypedEasyGpu(path);

// Read back the results
float[] result = output.ToArray();
Console.WriteLine("Input:  " + string.Join(", ", data));
Console.WriteLine("Output: " + string.Join(", ", result));
Console.WriteLine($"Dispatch path: {path}");

// Verify
bool pass = true;
for (int i = 0; i < data.Length; i++)
{
    if (MathF.Abs(result[i] - data[i] * 2.0f) > 1e-6f)
    {
        pass = false;
        break;
    }
}
if (!pass)
{
    throw new InvalidOperationException("HelloBuffer validation failed.");
}

Console.WriteLine("PASS");

// A simple compute kernel that doubles each element of an input buffer.
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct DoubleKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Doubles one input element for the current global thread index.
    /// </summary>
    public void Execute()
    {
        var i = ThreadIds.X;
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
