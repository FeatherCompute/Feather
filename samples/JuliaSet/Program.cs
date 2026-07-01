using System.Diagnostics;
using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

var width = args.Length > 0 && int.TryParse(args[0], out var parsedWidth) ? parsedWidth : 1024;
var height = args.Length > 1 && int.TryParse(args[1], out var parsedHeight) ? parsedHeight : 1024;
var maxIterations = args.Length > 2 && int.TryParse(args[2], out var parsedIterations) ? parsedIterations : 256;
ArgumentOutOfRangeException.ThrowIfLessThan(width, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(height, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(maxIterations, 1);

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<JuliaKernel>();

using var output = GPU.CreateBuffer<float4>(width * height, BufferAccess.ReadWrite);
var stopwatch = Stopwatch.StartNew();
var path = GPU.DispatchAndGetPath(
    new JuliaKernel(
        output.AsReadWrite(),
        new Uniform<int>(width),
        new Uniform<int>(height),
        new Uniform<int>(maxIterations)),
    new int2(width, height));
stopwatch.Stop();
SampleProof.AssertTypedEasyGpu(path);

var pixels = output.ToArray();
var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "julia-set.tga"));
JuliaImageWriter.SaveFloat4Tga(imagePath, pixels, width, height);
var pngPath = JuliaImageWriter.TryConvertToPng(imagePath);
var proof = SampleProof.MeasureImage(pixels);

Console.WriteLine("Feather Julia Set Renderer");
Console.WriteLine("Parameter c = -0.8 + 0.156i");
Console.WriteLine($"Image: {width}x{height}, iterations={maxIterations}");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine($"Render time: {stopwatch.ElapsedMilliseconds} ms");
Console.WriteLine($"Lit pixels: {proof.LitPixels}");
Console.WriteLine($"Distinct quantized colors: {proof.DistinctColors}");
Console.WriteLine($"Image written: {imagePath}");
if (pngPath is not null)
{
    Console.WriteLine($"PNG written: {pngPath}");
}

if (proof.LitPixels <= pixels.Length / 8 || proof.DistinctColors <= 16)
{
    throw new InvalidOperationException("Julia kernel produced an image without meaningful color variation.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("Julia image artifact is empty.");
}

Console.WriteLine("PASS");

/// <summary>
/// Ports EasyGPU's Julia set compute example to Feather's C# kernel DSL.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct JuliaKernel(
    ReadWriteBuffer<float4> output,
    Uniform<int> width,
    Uniform<int> height,
    Uniform<int> maxIterations) : IKernel2D
{
    private const float CenterX = 0.0f;
    private const float CenterY = 0.0f;
    private const float Zoom = 1.5f;
    private const float JuliaCx = -0.8f;
    private const float JuliaCy = 0.156f;

    /// <summary>
    /// Computes one Julia set pixel for the current global thread coordinate.
    /// </summary>
    public void Execute()
    {
        int2 pixel = ThreadIds.XY;
        int imageWidth = width.Value;
        int imageHeight = height.Value;
        if (pixel.X >= imageWidth || pixel.Y >= imageHeight)
        {
            return;
        }

        float aspectRatio = (float)imageWidth / (float)imageHeight;
        float scaleX = Zoom * aspectRatio;
        float scaleY = Zoom;

        float u = ((float)pixel.X + 0.5f) / (float)imageWidth;
        float v = ((float)pixel.Y + 0.5f) / (float)imageHeight;

        float zx = CenterX + (((u * 2.0f) - 1.0f) * scaleX);
        float zy = CenterY + (((v * 2.0f) - 1.0f) * scaleY);
        int iter = Julia(zx, zy, maxIterations.Value);
        float3 color = GetColor(iter, maxIterations.Value);

        output[(pixel.Y * imageWidth) + pixel.X] = new float4(color, 1.0f);
    }

    [Callable]
    private static int Julia(float zx, float zy, int maxIterations)
    {
        int iter = maxIterations;
        for (int i = 0; i < maxIterations; i++)
        {
            float zx2 = zx * zx;
            float zy2 = zy * zy;

            if (zx2 + zy2 > 4.0f)
            {
                iter = i;
                break;
            }

            zy = (2.0f * zx * zy) + JuliaCy;
            zx = zx2 - zy2 + JuliaCx;
        }

        return iter;
    }

    [Callable]
    private static float3 GetColor(int iter, int maxIterations)
    {
        float3 color = new float3(0.02f, 0.02f, 0.05f);
        if (iter != maxIterations)
        {
            float t = (float)iter / (float)maxIterations;
            float freq = 4.71239f;

            float r = 0.2f + (0.6f * ShaderMath.Sin((freq * t) + 4.0f));
            float g = 0.3f + (0.5f * ShaderMath.Sin((freq * t) + 2.0f));
            float b = 0.6f + (0.4f * ShaderMath.Sin(freq * t));

            color = new float3(
                ShaderMath.Clamp(r, 0.0f, 1.0f),
                ShaderMath.Clamp(g, 0.0f, 1.0f),
                ShaderMath.Clamp(b * 1.1f, 0.0f, 1.0f));
        }

        return color;
    }
}

/// <summary>
/// Writes Julia sample output as dependency-free image artifacts.
/// </summary>
internal static class JuliaImageWriter
{
    /// <summary>
    /// Saves top-left-origin BGRA TGA pixels after clamping float channels to [0, 1].
    /// </summary>
    public static void SaveFloat4Tga(string path, ReadOnlySpan<float4> pixels, int width, int height)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        if (pixels.Length < width * height)
        {
            throw new ArgumentException("Pixel span is shorter than the image area.", nameof(pixels));
        }

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = File.Create(path);
        Span<byte> header = stackalloc byte[18];
        header[2] = 2;
        header[12] = (byte)width;
        header[13] = (byte)(width >> 8);
        header[14] = (byte)height;
        header[15] = (byte)(height >> 8);
        header[16] = 32;
        header[17] = 0x28;
        stream.Write(header);

        var encoded = new byte[checked(width * height * 4)];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var srcIndex = ((height - 1 - y) * width) + x;
                var offset = ((y * width) + x) * 4;
                var pixel = pixels[srcIndex];
                encoded[offset] = ToByte(pixel.Z);
                encoded[offset + 1] = ToByte(pixel.Y);
                encoded[offset + 2] = ToByte(pixel.X);
                encoded[offset + 3] = ToByte(pixel.W);
            }
        }

        stream.Write(encoded);
    }

    /// <summary>
    /// Converts the TGA artifact to PNG through macOS sips when it is available.
    /// </summary>
    public static string? TryConvertToPng(string tgaPath)
    {
        if (!OperatingSystem.IsMacOS())
        {
            return null;
        }

        var pngPath = Path.ChangeExtension(tgaPath, ".png");
        try
        {
            using var process = Process.Start(new ProcessStartInfo
            {
                FileName = "sips",
                ArgumentList = { "-s", "format", "png", tgaPath, "--out", pngPath },
                RedirectStandardError = true,
                RedirectStandardOutput = true
            });
            process?.WaitForExit(5000);
            return process?.ExitCode == 0 && File.Exists(pngPath) ? pngPath : null;
        }
        catch
        {
            return null;
        }
    }

    private static byte ToByte(float value)
        => (byte)(256.0f * Math.Clamp(value, 0.0f, 0.999f));
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
            !glsl.Contains("sin", StringComparison.Ordinal) ||
            !glsl.Contains("layout(push_constant)", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce EasyGPU GLSL with structured control flow and push constants.");
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
    /// Measures whether the rendered image contains non-empty color variation.
    /// </summary>
    public static ImageProof MeasureImage(ReadOnlySpan<float4> pixels)
    {
        var litPixels = 0;
        var colors = new HashSet<int>();
        foreach (var pixel in pixels)
        {
            var r = Quantize(pixel.X);
            var g = Quantize(pixel.Y);
            var b = Quantize(pixel.Z);
            if (r + g + b > 0)
            {
                litPixels++;
                colors.Add(r | (g << 8) | (b << 16));
            }
        }

        return new ImageProof(litPixels, colors.Count);
    }

    private static int Quantize(float value)
        => (int)(255.0f * Math.Clamp(value, 0.0f, 1.0f));
}

internal readonly record struct ImageProof(int LitPixels, int DistinctColors);
