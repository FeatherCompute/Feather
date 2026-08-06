using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

// Color filter using simple buffer copy (no bit operations)
const int N = 16;
SampleProof.PrintBackend(GPU.Context);

var input = new float[N];
var output = new float[N];
for (var i = 0; i < N; i++)
{
    input[i] = i * 16;
}

using var bufInput = GPU.CreateBuffer<float>(input, BufferAccess.ReadOnly);
using var bufOutput = GPU.CreateBuffer<float>(N, BufferAccess.ReadWrite);

var path = GPU.DispatchAndGetPath(new FilterKernel(bufInput.AsReadOnly(), bufOutput.AsReadWrite()), N, GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

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
