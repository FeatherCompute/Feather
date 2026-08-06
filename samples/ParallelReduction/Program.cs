using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int N = 4;

SampleProof.PrintBackend(GPU.Context);

float[] data = [1.0f, 2.0f, 3.0f, 4.0f];
using var input = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);

// Kernel that copies buffer with a barrier.
var path = GPU.DispatchAndGetPath(new BarrierCopyKernel(input.AsReadOnly(), output.AsReadWrite()), N, GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

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
