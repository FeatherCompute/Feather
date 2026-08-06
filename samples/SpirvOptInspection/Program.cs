using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

SampleProof.PrintBackend(GPU.Context);

Console.WriteLine("FEIR Inspection:");
string ir = ShaderInspection.GetIR<InspectKernel>();
Console.WriteLine($"Serialized FEIR bytes: {ir.Length / 2}");
if (ir.Length == 0)
{
    throw new InvalidOperationException("InspectKernel did not produce serialized FEIR.");
}

using var input = GPU.CreateBuffer<float>([1.0f, 2.0f, 3.0f, 4.0f], BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(4, BufferAccess.ReadWrite);
var path = GPU.DispatchAndGetPath(new InspectKernel(input.AsReadOnly(), output.AsReadWrite()), 4, GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);
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
    /// Prints the selected Luisa device.
    /// </summary>
    public static void PrintBackend(GpuContext context)
    {
        Console.WriteLine($"Backend: {context.Device.BackendName}");
        Console.WriteLine($"Device: {context.Device.Name} (index {context.Device.DeviceIndex})");
    }

    /// <summary>
    /// Requires the dispatch to have used the Luisa backend path.
    /// </summary>
    public static void AssertLuisa(DispatchPath path)
    {
        if (path != DispatchPath.Luisa)
        {
            throw new InvalidOperationException($"Expected Luisa dispatch, got {path}.");
        }
    }
}
