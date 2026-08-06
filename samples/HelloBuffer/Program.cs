using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Create input data and output buffer
float[] data = [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f];
SampleProof.PrintBackend(GPU.Context);

using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(data.Length, BufferAccess.ReadWrite);

// Create and dispatch the kernel
var path = GPU.DispatchAndGetPath(new DoubleKernel(input.AsReadOnly(), output.AsReadWrite()), data.Length, GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

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
