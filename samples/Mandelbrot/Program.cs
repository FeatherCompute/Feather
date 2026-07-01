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
SampleProof.AssertEasyGpuGlsl<MandelbrotKernel>();

using var image = GPU.CreateBuffer<float4>(width * height, BufferAccess.ReadWrite);

var stopwatch = Stopwatch.StartNew();
var path = GPU.DispatchAndGetPath(
    new MandelbrotKernel(
        image.AsReadWrite(),
        new Uniform<int>(width),
        new Uniform<int>(height),
        new Uniform<int>(maxIterations)),
    new int2(width, height));
stopwatch.Stop();
SampleProof.AssertTypedEasyGpu(path);

var pixels = image.ToArray();
var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "mandelbrot-feather.tga"));
MandelbrotImageWriter.SaveFloat4Tga(imagePath, pixels, width, height);
var pngPath = MandelbrotImageWriter.TryConvertToPng(imagePath);
var proof = SampleProof.MeasureImage(pixels);

Console.WriteLine("Feather Mandelbrot Renderer");
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

if (proof.LitPixels <= pixels.Length / 4 || proof.DistinctColors <= 16)
{
    throw new InvalidOperationException("Mandelbrot kernel produced an image without meaningful fractal color variation.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("Mandelbrot image artifact is empty.");
}

Console.WriteLine("PASS");

/// <summary>
/// Ports EasyGPU's Mandelbrot compute example to Feather's C# kernel DSL.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct MandelbrotKernel(
    ReadWriteBuffer<float4> image,
    Uniform<int> width,
    Uniform<int> height,
    Uniform<int> maxIterations) : IKernel2D
{
    private const float CenterX = -0.5f;
    private const float CenterY = 0.0f;
    private const float Zoom = 1.5f;

    /// <summary>
    /// Renders one Mandelbrot pixel into the output buffer.
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

        float cx = CenterX + (((u * 2.0f) - 1.0f) * scaleX);
        float cy = CenterY + (((v * 2.0f) - 1.0f) * scaleY);
        int iter = Mandelbrot(cx, cy, maxIterations.Value);
        float3 color = GetColor(iter, maxIterations.Value);

        image[(pixel.Y * imageWidth) + pixel.X] = new float4(color, 1.0f);
    }

    [Callable]
    private static int Mandelbrot(float cx, float cy, int maxIterations)
    {
        float zx = 0.0f;
        float zy = 0.0f;
        int iter = 0;

        for (int i = 0; i < maxIterations; i++)
        {
            float zx2 = zx * zx;
            float zy2 = zy * zy;

            if (zx2 + zy2 > 4.0f)
            {
                iter = i;
                break;
            }

            zy = (2.0f * zx * zy) + cy;
            zx = zx2 - zy2 + cx;
            iter = i;
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
            float freq = 6.28318f;
            float r = 0.5f + (0.5f * ShaderMath.Sin((freq * t) + 0.0f) * ShaderMath.Cos(freq * t * 0.5f));
            float g = 0.5f + (0.5f * ShaderMath.Sin((freq * t) + 2.094f) * ShaderMath.Cos((freq * t * 0.3f) + 1.0f));
            float b = 0.5f + (0.5f * ShaderMath.Sin((freq * t) + 4.188f) * ShaderMath.Cos((freq * t * 0.7f) + 2.0f));

            r = ShaderMath.Pow(ShaderMath.Clamp(r, 0.0f, 1.0f), 0.8f);
            g = ShaderMath.Pow(ShaderMath.Clamp(g, 0.0f, 1.0f), 0.8f);
            b = ShaderMath.Pow(ShaderMath.Clamp(b, 0.0f, 1.0f), 0.8f);

            float intensity = 1.2f;
            color = new float3(
                ShaderMath.Clamp(r * intensity, 0.0f, 1.0f),
                ShaderMath.Clamp(g * intensity, 0.0f, 1.0f),
                ShaderMath.Clamp(b * intensity, 0.0f, 1.0f));
        }

        return color;
    }
}

internal static class MandelbrotImageWriter
{
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

internal static class SampleProof
{
    public static void PrintBackend(GpuContext context)
    {
        var caps = context.Caps;
        Console.WriteLine($"Backend: {caps.BackendType}");
        Console.WriteLine($"Max workgroup size: {caps.MaxWorkGroupSizeX}x{caps.MaxWorkGroupSizeY}x{caps.MaxWorkGroupSizeZ}");
    }

    public static void AssertEasyGpuGlsl<TKernel>()
        where TKernel : struct, IGeneratedKernel<TKernel>
    {
        var glsl = ShaderInspection.GetGLSL<TKernel>();
        if (!glsl.Contains("gl_GlobalInvocationID", StringComparison.Ordinal) ||
            !glsl.Contains("sin", StringComparison.Ordinal) ||
            !glsl.Contains("pow", StringComparison.Ordinal) ||
            !glsl.Contains("for", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce the expected EasyGPU Mandelbrot GLSL.");
        }

        Console.WriteLine("EasyGPU GLSL bridge: OK");
    }

    public static void AssertTypedEasyGpu(DispatchPath path)
    {
        if (path != DispatchPath.TypedEasyGpu)
        {
            throw new InvalidOperationException($"Expected TypedEasyGpu dispatch, got {path}.");
        }
    }

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
        => Math.Clamp((int)(value * 31.0f), 0, 31);
}

internal readonly record struct ImageProof(int LitPixels, int DistinctColors);
