using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Simple buffer copy kernel (avoids atomic race conditions)
const int N = 1024;
SampleProof.PrintBackend(GPU.Context);

float[] data = new float[N];
for (int i = 0; i < N; i++) data[i] = i;

using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);
var path = GPU.DispatchAndGetPath(new CopyKernel(input.AsReadOnly(), output.AsReadWrite()), N, GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

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
