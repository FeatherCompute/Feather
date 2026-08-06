using Feather;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

var frameLimit = ReadFrameLimit(args);
var runtime = GpuRuntime.Create();
var devices = runtime.EnumerateDevices();
var selectedDevice = runtime.DefaultDevice;

Console.WriteLine($"Discovered {devices.Count} Luisa device(s).");
Console.WriteLine($"Platform default: {selectedDevice.BackendName}[{selectedDevice.DeviceIndex}] {selectedDevice.Name}");

// Keep the static facade exercised alongside the explicit context below. M9
// will redirect this compatibility entry point to the same Luisa-only runtime.
using var legacyInput = GPU.CreateBuffer<int>([1, 2, 3], BufferAccess.ReadOnly);
using var legacyOutput = GPU.CreateBuffer<int>(3, BufferAccess.ReadWrite);
var legacyPath = GPU.DispatchAndGetPath(
    new IncrementKernel(legacyInput.AsReadOnly(), legacyOutput.AsReadWrite()), 3);
if (!legacyOutput.ToArray().SequenceEqual([2, 3, 4]))
{
    throw new InvalidOperationException("Static GPU.Dispatch compatibility check failed.");
}
Console.WriteLine($"Static GPU.Dispatch path: {legacyPath}");

using var context = runtime.CreateContext(new GpuContextOptions
{
    Backend = selectedDevice.Backend,
    DeviceIndex = selectedDevice.DeviceIndex
});
using var stream = context.CreateStream();
using var streamInput = GpuBuffer<int>.Create(context, [40], BufferAccess.ReadWrite);
using var streamOutput = GpuBuffer<int>.Create(context, 1, BufferAccess.ReadWrite);
using var streamKernel = GpuKernel.Create<IncrementKernel>(context, GpuExecutionBackend.Luisa);
using (var fence = stream.Dispatch(
           streamKernel,
           new IncrementKernel(streamInput.AsReadOnly(), streamOutput.AsReadWrite()),
           new GpuDispatchSize(1, 1, 1)))
{
    fence.Wait();
}
if (!streamOutput.ToArray().SequenceEqual([41]))
{
    throw new InvalidOperationException("Explicit GpuStream dispatch check failed.");
}
Console.WriteLine($"Explicit context: {context.Backend} {context.Device.Name}; GpuStream fence: PASS");

using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 450,
    Title = "Feather Compute Texture"
});
using var presenter = window.CreateTexturePresenter();
using var color = GpuTexture2D<float4, float4>.Create(
    context, window.Width, window.Height, PixelFormat.Rgba32Float, TextureAccess.ReadWrite);
using var pixelsKernel = GpuKernel.Create<ComputePixels>(context, GpuExecutionBackend.Luisa);

var frame = 0;
while (window.IsOpen && (frameLimit == 0 || frame < frameLimit))
{
    window.PollEvents();
    while (window.TryPollEvent(out var windowEvent))
    {
        if (windowEvent is WindowKeyEvent { Key: WindowKey.Escape, Pressed: true })
        {
            window.Close();
        }
    }

    GpuKernel.Dispatch(
        context,
        pixelsKernel,
        new ComputePixels(color.AsReadWrite(), new Uniform<int>(frame)),
        new GpuDispatchSize(color.Width, color.Height, 1),
        wait: true);
    if (frame == 0)
    {
        Console.WriteLine($"Explicit context dispatch path: {pixelsKernel.LastDispatchPath}");
    }
    presenter.Present(color);
    frame++;
}

if (frameLimit != 0)
{
    Console.WriteLine($"PASS: rendered {frame} frame(s) through the explicit Luisa context.");
}

static int ReadFrameLimit(string[] arguments)
{
    for (var index = 0; index < arguments.Length; index++)
    {
        if (arguments[index] != "--frames")
        {
            continue;
        }
        if (index + 1 >= arguments.Length || !int.TryParse(arguments[++index], out var count) || count <= 0)
        {
            throw new ArgumentException("--frames requires a positive integer.");
        }
        return count;
    }
    return 0;
}

[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct IncrementKernel(
    ReadOnlyBuffer<int> input,
    ReadWriteBuffer<int> output) : IKernel1D
{
    public void Execute()
    {
        output[ThreadIds.X] = input[ThreadIds.X] + 1;
    }
}

[Kernel]
[ThreadGroupSize(8, 8, 1)]
public readonly partial struct ComputePixels(ReadWriteTexture2D<float4> output, Uniform<int> frame) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        int t = frame.Value;
        float r = ((p.X + t) & 255) / 255.0f;
        float g = ((p.Y * 2 + t) & 255) / 255.0f;
        float b = ((p.X + p.Y + t * 3) & 255) / 255.0f;
        output[p] = new float4(r, g, b, 1.0f);
    }
}
