using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

int count = args.Length > 0 && int.TryParse(args[0], out var parsed) ? parsed : 25600;
SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<IncrementKernel>();

int[] data = new int[count];
for (int idx = 0; idx < count; idx++)
{
    data[idx] = idx + 1;
}

using var input = GPU.CreateBuffer<int>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<int>(count, BufferAccess.ReadWrite);

var path = GPU.DispatchAndGetPath(
    new IncrementKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<int>(count)),
    count);
SampleProof.AssertTypedEasyGpu(path);

int[] result = output.ToArray();
bool allCorrect = true;
for (int i = 0; i < count && allCorrect; i++)
{
    if (result[i] != data[i] + 1)
    {
        allCorrect = false;
    }
}

Console.WriteLine($"Dispatch path: {path}");
if (!allCorrect)
{
    throw new InvalidOperationException("HelloWorld validation failed.");
}

Console.WriteLine($"PASS: All {count} elements incremented correctly");

/// <summary>
/// Increments each input element while respecting the logical element count.
/// </summary>
[Kernel]
[ThreadGroupSize(256, 1, 1)]
public readonly partial struct IncrementKernel(
    ReadOnlyBuffer<int> input,
    ReadWriteBuffer<int> output,
    Uniform<int> count) : IKernel1D
{
    /// <summary>
    /// Adds one to the current input element if the global thread is in range.
    /// </summary>
    public void Execute()
    {
        int i = ThreadIds.X;
        if (i < count.Value)
        {
            output[i] = input[i] + 1;
        }
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
            !glsl.Contains("layout(push_constant)", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce EasyGPU GLSL with push constants.");
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
