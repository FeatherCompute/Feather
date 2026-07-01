using System.Diagnostics;
using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int DefaultWidth = 1024;
const int DefaultHeight = 1024;
const int DefaultSamplesPerPixel = 2560 * 4;

var width = args.Length > 0 && int.TryParse(args[0], out var parsedWidth) ? parsedWidth : DefaultWidth;
var height = args.Length > 1 && int.TryParse(args[1], out var parsedHeight) ? parsedHeight : DefaultHeight;
var samplesPerPixel = args.Length > 2 && int.TryParse(args[2], out var parsedSamples) ? parsedSamples : DefaultSamplesPerPixel;
ArgumentOutOfRangeException.ThrowIfLessThan(width, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(height, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(samplesPerPixel, 1);

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<RayTracingKernel>();

using var image = GPU.CreateBuffer<float4>(width * height, BufferAccess.ReadWrite);
using var rng = GPU.CreateBuffer<int>(CreateSeeds(width, height), BufferAccess.ReadWrite);

var stopwatch = Stopwatch.StartNew();
var path = GPU.DispatchAndGetPath(
    new RayTracingKernel(
        image.AsReadWrite(),
        rng.AsReadWrite(),
        new Uniform<int>(width),
        new Uniform<int>(height),
        new Uniform<int>(samplesPerPixel)),
    new int2(width, height));
stopwatch.Stop();
SampleProof.AssertTypedEasyGpu(path);

var pixels = image.ToArray();
var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "cornell-box-feather.tga"));
RayTracingImageWriter.SaveFloat4Tga(imagePath, pixels, width, height);
var pngPath = RayTracingImageWriter.TryConvertToPng(imagePath);

var proof = SampleProof.MeasureImage(pixels);
Console.WriteLine("Feather Cornell Box Path Tracer");
Console.WriteLine($"Image: {width}x{height}, {samplesPerPixel} spp");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine($"Render time: {stopwatch.ElapsedMilliseconds} ms");
Console.WriteLine($"Lit pixels: {proof.LitPixels}");
Console.WriteLine($"Distinct quantized colors: {proof.DistinctColors}");
Console.WriteLine($"Image written: {imagePath}");
if (pngPath is not null)
{
    Console.WriteLine($"PNG written: {pngPath}");
}

if (proof.LitPixels <= Math.Max(4, pixels.Length / 128) || proof.DistinctColors <= 4)
{
    throw new InvalidOperationException("Ray tracing kernel produced an image without meaningful lighting variation.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("Ray tracing image artifact is empty.");
}

Console.WriteLine("PASS");

static int[] CreateSeeds(int width, int height)
{
    var seeds = new int[checked(width * height)];
    for (var i = 0; i < seeds.Length; i++)
    {
        seeds[i] = (i + 1) * 9781;
    }

    return seeds;
}

/// <summary>
/// Ports the EasyGPU C++ Cornell-box compute path tracer to Feather's C# compute DSL.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct RayTracingKernel(
    ReadWriteBuffer<float4> image,
    ReadWriteBuffer<int> rng,
    Uniform<int> width,
    Uniform<int> height,
    Uniform<int> samplesPerPixel) : IKernel2D
{
    /// <summary>
    /// Renders one pixel with stochastic bounces through a Cornell-box scene.
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

        int sampleCount = samplesPerPixel.Value;
        int pixelIndex = pixel.Y * imageWidth + pixel.X;
        int seed = rng[pixelIndex];
        float3 accumulated = new float3(0.0f);

        for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
        {
            int stream = sampleIndex * 64;
            float u = ((float)pixel.X + Random01(seed, stream)) / (float)imageWidth;
            float v = ((float)pixel.Y + Random01(seed, stream + 1)) / (float)imageHeight;
            float3 rayOrigin = new float3(0.0f, 0.0f, 2.5f);
            float3 rayDirection = ShaderMath.Normalize(new float3(-1.0f + (2.0f * u), -1.0f + (2.0f * v), 0.5f) - rayOrigin);

            float3 throughput = new float3(1.0f);
            float3 radiance = new float3(0.0f);

            for (int depth = 0; depth < 8; depth++)
            {
                float closest = 1000.0f;
                int hit = 0;
                int materialType = 0;
                float3 hitAlbedo = new float3(0.73f, 0.73f, 0.73f);
                float3 hitNormal = new float3(0.0f, 0.0f, 1.0f);
                float3 hitPoint = new float3(0.0f);
                float t = closest;

                t = HitBoxDistance(new float3(-1.0f, -1.0f, -1.0f), new float3(1.0f, -0.75f, 1.0f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-1.0f, -1.0f, -1.0f), new float3(1.0f, -0.75f, 1.0f), hitPoint);
                    hitAlbedo = new float3(0.73f, 0.73f, 0.73f);
                    materialType = 0;
                }

                t = HitBoxDistance(new float3(-1.0f, 0.75f, -1.0f), new float3(1.0f, 1.0f, 1.0f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-1.0f, 0.75f, -1.0f), new float3(1.0f, 1.0f, 1.0f), hitPoint);
                    hitAlbedo = new float3(0.73f, 0.73f, 0.73f);
                    materialType = 0;
                }

                t = HitBoxDistance(new float3(-1.0f, -0.75f, -1.0f), new float3(1.0f, 0.75f, -0.75f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-1.0f, -0.75f, -1.0f), new float3(1.0f, 0.75f, -0.75f), hitPoint);
                    hitAlbedo = new float3(0.73f, 0.73f, 0.73f);
                    materialType = 0;
                }

                t = HitBoxDistance(new float3(-1.0f, -0.75f, -0.75f), new float3(-0.75f, 0.75f, 1.0f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-1.0f, -0.75f, -0.75f), new float3(-0.75f, 0.75f, 1.0f), hitPoint);
                    hitAlbedo = new float3(0.65f, 0.05f, 0.05f);
                    materialType = 0;
                }

                t = HitBoxDistance(new float3(0.75f, -0.75f, -0.75f), new float3(1.0f, 0.75f, 1.0f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(0.75f, -0.75f, -0.75f), new float3(1.0f, 0.75f, 1.0f), hitPoint);
                    hitAlbedo = new float3(0.12f, 0.45f, 0.15f);
                    materialType = 0;
                }

                t = HitBoxDistance(new float3(-0.25f, 0.74f, -0.25f), new float3(0.25f, 0.75f, 0.25f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-0.25f, 0.74f, -0.25f), new float3(0.25f, 0.75f, 0.25f), hitPoint);
                    hitAlbedo = new float3(15.0f, 15.0f, 15.0f);
                    materialType = 2;
                }

                t = HitBoxDistance(new float3(0.15f, -0.75f, -0.4f), new float3(0.45f, -0.15f, -0.1f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(0.15f, -0.75f, -0.4f), new float3(0.45f, -0.15f, -0.1f), hitPoint);
                    hitAlbedo = new float3(0.8f, 0.85f, 0.88f);
                    materialType = 1;
                }

                t = HitBoxDistance(new float3(-0.4f, -0.75f, 0.0f), new float3(-0.1f, -0.4f, 0.3f), rayOrigin, rayDirection, 0.001f, closest);
                if (t < closest)
                {
                    closest = t;
                    hit = 1;
                    hitPoint = rayOrigin + (rayDirection * t);
                    hitNormal = BoxNormal(new float3(-0.4f, -0.75f, 0.0f), new float3(-0.1f, -0.4f, 0.3f), hitPoint);
                    hitAlbedo = new float3(0.73f, 0.73f, 0.73f);
                    materialType = 0;
                }

                if (hit == 0)
                {
                    break;
                }

                if (materialType == 2)
                {
                    radiance = radiance + (throughput * hitAlbedo);
                    break;
                }

                float3 randomDirection = RandomUnitVector(seed, stream + (depth * 7) + 2);
                throughput = throughput * hitAlbedo;
                if (materialType == 1)
                {
                    rayDirection = ShaderMath.Normalize(ReflectVector(ShaderMath.Normalize(rayDirection), hitNormal) + (0.2f * randomDirection));
                }
                else
                {
                    float3 lightBias = ShaderMath.Normalize(new float3(0.0f, 0.74f, 0.0f) - hitPoint);
                    rayDirection = ShaderMath.Normalize(hitNormal + randomDirection + (0.35f * lightBias));
                }

                rayOrigin = hitPoint + (0.002f * hitNormal);
            }

            accumulated = accumulated + radiance;
        }

        float scale = 1.0f / (float)sampleCount;
        float3 color = accumulated * scale;
        image[pixelIndex] = new float4(
            ShaderMath.Sqrt(ShaderMath.Clamp(color.X, 0.0f, 1.0f)),
            ShaderMath.Sqrt(ShaderMath.Clamp(color.Y, 0.0f, 1.0f)),
            ShaderMath.Sqrt(ShaderMath.Clamp(color.Z, 0.0f, 1.0f)),
            1.0f);
        rng[pixelIndex] = seed + (sampleCount * 131) + (pixel.X * 17) + (pixel.Y * 29);
    }

    [Callable]
    private static float Random01(int seed, int stream)
    {
        int state = seed + (stream * 747796405);
        state = state ^ (state >> 16);
        state = state * 1103515245 + 12345;
        state = state ^ (state >> 15);
        int positive = state & 2147483647;
        return (float)positive / 2147483647.0f;
    }

    [Callable]
    private static float3 RandomUnitVector(int seed, int stream)
    {
        float x = (Random01(seed, stream) * 2.0f) - 1.0f;
        float y = (Random01(seed, stream + 1) * 2.0f) - 1.0f;
        float z = (Random01(seed, stream + 2) * 2.0f) - 1.0f;
        float3 value = new float3(x, y, z);
        if (ShaderMath.Dot(value, value) < 0.0001f)
        {
            value = new float3(0.17f, 0.59f, 0.73f);
        }

        return ShaderMath.Normalize(value);
    }

    [Callable]
    private static float3 ReflectVector(float3 direction, float3 normal)
    {
        return direction - ((2.0f * ShaderMath.Dot(direction, normal)) * normal);
    }

    [Callable]
    private static float HitBoxDistance(float3 bmin, float3 bmax, float3 origin, float3 direction, float tmin, float closest)
    {
        float candidate = closest;
        float t = (bmin.X - origin.X) / direction.X;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.Y > bmin.Y && p.Y < bmax.Y && p.Z > bmin.Z && p.Z < bmax.Z)
            {
                candidate = t;
            }
        }

        t = (bmax.X - origin.X) / direction.X;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.Y > bmin.Y && p.Y < bmax.Y && p.Z > bmin.Z && p.Z < bmax.Z)
            {
                candidate = t;
            }
        }

        t = (bmin.Y - origin.Y) / direction.Y;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.X > bmin.X && p.X < bmax.X && p.Z > bmin.Z && p.Z < bmax.Z)
            {
                candidate = t;
            }
        }

        t = (bmax.Y - origin.Y) / direction.Y;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.X > bmin.X && p.X < bmax.X && p.Z > bmin.Z && p.Z < bmax.Z)
            {
                candidate = t;
            }
        }

        t = (bmin.Z - origin.Z) / direction.Z;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.X > bmin.X && p.X < bmax.X && p.Y > bmin.Y && p.Y < bmax.Y)
            {
                candidate = t;
            }
        }

        t = (bmax.Z - origin.Z) / direction.Z;
        if (t > tmin && t < candidate)
        {
            float3 p = origin + (direction * t);
            if (p.X > bmin.X && p.X < bmax.X && p.Y > bmin.Y && p.Y < bmax.Y)
            {
                candidate = t;
            }
        }

        return candidate;
    }

    [Callable]
    private static float3 BoxNormal(float3 bmin, float3 bmax, float3 p)
    {
        float eps = 0.0025f;
        float3 normal = new float3(0.0f, 0.0f, 1.0f);
        if (ShaderMath.Abs(p.X - bmin.X) < eps)
        {
            normal = new float3(-1.0f, 0.0f, 0.0f);
        }
        else if (ShaderMath.Abs(p.X - bmax.X) < eps)
        {
            normal = new float3(1.0f, 0.0f, 0.0f);
        }
        else if (ShaderMath.Abs(p.Y - bmin.Y) < eps)
        {
            normal = new float3(0.0f, -1.0f, 0.0f);
        }
        else if (ShaderMath.Abs(p.Y - bmax.Y) < eps)
        {
            normal = new float3(0.0f, 1.0f, 0.0f);
        }
        else if (ShaderMath.Abs(p.Z - bmin.Z) < eps)
        {
            normal = new float3(0.0f, 0.0f, -1.0f);
        }

        return normal;
    }
}

/// <summary>
/// Writes floating-point render output as dependency-free image artifacts.
/// </summary>
internal static class RayTracingImageWriter
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
/// Runtime checks used by the sample before it prints PASS.
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
            !glsl.Contains("normalize", StringComparison.Ordinal) ||
            !glsl.Contains("sqrt", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce the expected EasyGPU GLSL path-tracing shader.");
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
