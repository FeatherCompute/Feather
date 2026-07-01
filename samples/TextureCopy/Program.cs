using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int Width = 4;
const int Height = 4;

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<TextureCopyKernel>(glsl =>
    glsl.Contains("imageLoad", StringComparison.Ordinal) &&
    glsl.Contains("imageStore", StringComparison.Ordinal));

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

var path = GPU.DispatchAndGetPath(new TextureCopyKernel(input.AsReadOnly(), output.AsReadWrite()), new int2(Width, Height));
SampleProof.AssertTypedEasyGpu(path);

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
/// Copies a texture pixel-by-pixel through the EasyGPU imageLoad/imageStore bridge.
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
    public static void AssertEasyGpuGlsl<TKernel>(Func<string, bool>? extraCheck = null)
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var glsl = ShaderInspection.GetGLSL<TKernel>();
        if (!glsl.Contains("gl_GlobalInvocationID", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal) ||
            (extraCheck is not null && !extraCheck(glsl)))
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
