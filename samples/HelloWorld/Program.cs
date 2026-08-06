using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

int count = args.Length > 0 && int.TryParse(args[0], out var parsed) ? parsed : 25600;
SampleProof.PrintBackend(GPU.Context);

int[] data = new int[count];
for (int idx = 0; idx < count; idx++)
{
    data[idx] = idx + 1;
}

using var input = GPU.CreateBuffer<int>(data, BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<int>(count, BufferAccess.ReadWrite);

var path = GPU.DispatchAndGetPath(
    new IncrementKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<int>(count)),
    count,
    GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

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
