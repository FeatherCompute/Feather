using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int Width = 4;
const int Height = 4;

SampleProof.PrintBackend(GPU.Context);

// Create a simple 4x4 texture with RGBA pixel data.
var pixels = new Rgba32[]
{
    new(255, 0, 0, 255),   new(0, 255, 0, 255),   new(0, 0, 255, 255),   new(255, 255, 255, 255),
    new(127, 0, 0, 255),   new(0, 127, 0, 255),   new(0, 0, 127, 255),   new(127, 127, 127, 255),
    new(255, 128, 0, 255), new(128, 255, 0, 255), new(0, 255, 128, 255), new(128, 128, 255, 255),
    new(64, 64, 64, 255),  new(192, 192, 192, 255), new(32, 64, 128, 255), new(255, 0, 128, 255),
};

using var input = GPU.CreateTexture2D<Rgba32, Rgba32>(Width, Height, PixelFormat.Rgba8, TextureAccess.ReadOnly);
using var output = GPU.CreateTexture2D<Rgba32, Rgba32>(Width, Height, PixelFormat.Rgba8, TextureAccess.ReadWrite);

input.Upload(pixels);

var path = GPU.DispatchAndGetPath(new TextureCopyKernel(input.AsReadOnly(), output.AsReadWrite()), new int2(Width, Height), GpuExecutionBackend.Luisa);
SampleProof.AssertLuisa(path);

var readback = new Rgba32[pixels.Length];
output.Read(readback);

var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "texture-copy.tga"));
Directory.CreateDirectory(Path.GetDirectoryName(imagePath)!);
output.Save(imagePath);

Console.WriteLine("Texture Copy Sample");
Console.WriteLine("==================");
Console.WriteLine($"Input pixel count: {pixels.Length}");
Console.WriteLine($"Output pixel count: {readback.Length}");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine($"Image written: {imagePath}");

if (!pixels.SequenceEqual(readback) || !SampleProof.HasMeaningfulPixels(readback))
{
    throw new InvalidOperationException("TextureCopy validation failed.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("TextureCopy image artifact is empty.");
}

Console.WriteLine("PASS");

/// <summary>
/// Copies a texture pixel-by-pixel through the generated compute kernel.
/// </summary>
[Kernel]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct TextureCopyKernel(
    ReadOnlyTexture2D<Rgba32> input,
    ReadWriteTexture2D<Rgba32> output) : IKernel2D
{
    /// <summary>
    /// Copies the current two-dimensional pixel.
    /// </summary>
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        output[p] = input[p];
    }
}

/// <summary>
/// RGBA pixel struct used for texture data.
/// </summary>
[GpuStruct]
public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

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

    /// <summary>
    /// Checks that copied pixels contain more than transparent or all-zero data.
    /// </summary>
    public static bool HasMeaningfulPixels(ReadOnlySpan<Rgba32> pixels)
    {
        var visiblePixels = 0;
        foreach (var pixel in pixels)
        {
            if (pixel.A != 0 && (pixel.R != 0 || pixel.G != 0 || pixel.B != 0))
            {
                visiblePixels++;
            }
        }

        return visiblePixels > pixels.Length / 2 && pixels.ToArray().Distinct().Count() > 4;
    }
}
