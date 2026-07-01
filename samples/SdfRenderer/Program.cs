using System.Diagnostics;
using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

var width = args.Length > 0 && int.TryParse(args[0], out var parsedWidth) ? parsedWidth : 1280;
var height = args.Length > 1 && int.TryParse(args[1], out var parsedHeight) ? parsedHeight : 720;
var samplesPerPixel = args.Length > 2 && int.TryParse(args[2], out var parsedSamples) ? parsedSamples : 1240;
ArgumentOutOfRangeException.ThrowIfLessThan(width, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(height, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(samplesPerPixel, 1);

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<SdfRendererKernel>();

using var seedBuffer = GPU.CreateBuffer<int>(CreateSeeds(width, height), BufferAccess.ReadWrite);
using var accumBuffer = GPU.CreateBuffer<float4>(CreateZeroPixels(width, height), BufferAccess.ReadWrite);

var stopwatch = Stopwatch.StartNew();
var path = GPU.DispatchAndGetPath(
    new SdfRendererKernel(
        seedBuffer.AsReadWrite(),
        accumBuffer.AsReadWrite(),
        new Uniform<int>(width),
        new Uniform<int>(height),
        new Uniform<int>(samplesPerPixel)),
    new int2(width, height));
stopwatch.Stop();
SampleProof.AssertTypedEasyGpu(path);

var pixels = accumBuffer.ToArray();
var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "sdf-renderer-feather.tga"));
SdfImageWriter.SaveExposedSrgbTga(imagePath, pixels, width, height, 2.0f);
var pngPath = SdfImageWriter.TryConvertToPng(imagePath);
var proof = SampleProof.MeasureImage(pixels);

Console.WriteLine("Feather SDF Path Tracer");
Console.WriteLine($"Image: {width}x{height}, spp={samplesPerPixel}");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine($"Render time: {stopwatch.ElapsedMilliseconds} ms");
Console.WriteLine($"Lit pixels: {proof.LitPixels}");
Console.WriteLine($"Distinct quantized colors: {proof.DistinctColors}");
Console.WriteLine($"Image written: {imagePath}");
if (pngPath is not null)
{
    Console.WriteLine($"PNG written: {pngPath}");
}

if (proof.LitPixels <= Math.Max(16, pixels.Length / 256) || proof.DistinctColors <= 8)
{
    throw new InvalidOperationException("SDF renderer produced an image without meaningful lighting variation.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("SDF renderer image artifact is empty.");
}

Console.WriteLine("PASS");

static int[] CreateSeeds(int width, int height)
{
    var seeds = new int[checked(width * height)];
    for (var y = 0; y < height; y++)
    {
        for (var x = 0; x < width; x++)
        {
            var v0 = x;
            var v1 = y;
            var s0 = 0;
            for (var n = 0; n < 4; n++)
            {
                s0 += unchecked((int)0x9e3779b9);
                v0 += (((v1 << 4) + unchecked((int)0xa341316c)) ^ (v1 + s0) ^ ((v1 >> 5) + unchecked((int)0xc8013ea4)));
                v1 += (((v0 << 4) + unchecked((int)0xad90777d)) ^ (v0 + s0) ^ ((v0 >> 5) + unchecked((int)0x7e95761e)));
            }

            seeds[(y * width) + x] = v0;
        }
    }

    return seeds;
}

static float4[] CreateZeroPixels(int width, int height)
    => new float4[checked(width * height)];

/// <summary>
/// Ports EasyGPU's SDF path tracing example to Feather's C# kernel DSL.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct SdfRendererKernel(
    ReadWriteBuffer<int> seedBuffer,
    ReadWriteBuffer<float4> accumBuffer,
    Uniform<int> width,
    Uniform<int> height,
    Uniform<int> samplesPerPixel) : IKernel2D
{
    private const int MaxRayDepth = 6;
    private const float Eps = 1e-4f;
    private const float Inf = 1e10f;
    private const float Fov = 0.23f;
    private const float DistLimit = 100.0f;
    private const float LightRadius = 2.0f;

    /// <summary>
    /// Accumulates all requested path tracing samples for one SDF pixel.
    /// </summary>
    [Entry]
    public void Execute()
    {
        var pixel = ThreadIds.XY;
        var imageWidth = width.Value;
        var imageHeight = height.Value;
        if (pixel.X >= imageWidth || pixel.Y >= imageHeight)
        {
            return;
        }

        var idx = (pixel.Y * imageWidth) + pixel.X;
        var seed = seedBuffer[idx];
        var accum4 = accumBuffer[idx];
        var accum = accum4.XYZ;
        var aspectRatio = imageWidth / (float)imageHeight;

        for (var frame = 0; frame < samplesPerPixel.Value; frame++)
        {
            var pos = new float3(0.0f, 0.32f, 3.7f);

            var sampleStream = frame * 64;
            var ux = RandF(seed, sampleStream);
            var uy = RandF(seed, sampleStream + 1);
            var u = (float)pixel.X + ux;
            var v = (float)pixel.Y + uy;

            var du = ((2.0f * Fov * u) / (float)imageHeight) - (Fov * aspectRatio) - 1e-5f;
            var dv = ((2.0f * Fov * v) / (float)imageHeight) - Fov - 1e-5f;
            var direction = ShaderMath.Normalize(new float3(du, dv, -1.0f));

            var throughput = new float3(1.0f);
            var hitLight = 0.0f;

            for (var depth = 0; depth < MaxRayDepth; depth++)
            {
                var hit = NextHit(pos, direction);
                var distToLight = IntersectLight(pos, direction);
                if (distToLight < hit.Closest)
                {
                    hitLight = 1.0f;
                    break;
                }

                if (ShaderMath.Length(hit.Normal) == 0.0f)
                {
                    break;
                }

                var hitPos = pos + (hit.Closest * direction);
                direction = OutDir(hit.Normal, seed, sampleStream + (depth * 7) + 2);
                pos = hitPos + (1e-4f * direction);
                throughput = throughput * hit.Color;
            }

            var sampleColor = throughput * hitLight;
            accum = ShaderMath.Mix(accum, sampleColor, 1.0f / ((float)frame + 1.0f));
        }

        accumBuffer[idx] = new float4(accum, 1.0f);
        seedBuffer[idx] = seed;
    }

    [Callable]
    private static float IntersectLight(float3 pos, float3 direction)
    {
        var lightPos = new float3(-1.5f, 0.6f, 0.3f);
        var lightNormal = new float3(1.0f, 0.0f, 0.0f);
        var cosW = ShaderMath.Dot(-direction, lightNormal);
        var dist = ShaderMath.Dot(direction, lightPos - pos);
        var candidate = dist / cosW;
        var hitPoint = pos + (candidate * direction);
        var distToCenter = ShaderMath.Length(lightPos - hitPoint);

        var result = Inf;
        if (cosW > 0.0f && dist > 0.0f && distToCenter < LightRadius)
        {
            result = candidate;
        }

        return result;
    }

    [Callable]
    private static float RandF(int seed, int stream)
    {
        var state = seed + (stream * 747796405);
        state = state ^ (state >> 16);
        state = (1664525 * state) + 1013904223;
        state = state ^ (state >> 15);
        var positive = state & 2147483647;
        return (float)positive / 2147483648.0f;
    }

    [Callable]
    private static float3 OutDir(float3 normal, int seed, int stream)
    {
        var u = new float3(1.0f, 0.0f, 0.0f);
        if (ShaderMath.Abs(normal.Y) < 1.0f - Eps)
        {
            u = ShaderMath.Normalize(ShaderMath.Cross(normal, new float3(0.0f, 1.0f, 0.0f)));
        }

        var v = ShaderMath.Cross(normal, u);
        var phi = 6.28318530718f * RandF(seed, stream);
        var ay = ShaderMath.Sqrt(RandF(seed, stream + 1));
        var ax = ShaderMath.Sqrt(1.0f - (ay * ay));

        return (ax * ((ShaderMath.Cos(phi) * u) + (ShaderMath.Sin(phi) * v))) + (ay * normal);
    }

    [Callable]
    private static int FloatToInt(float f)
    {
        var result = (int)f;
        if (f < 0.0f)
        {
            result -= 1;
        }

        return result;
    }

    [Callable]
    private static float MakeNested(float f)
    {
        var freq = 40.0f;
        var scaled = f * freq;

        var result = scaled;
        if (scaled < 0.0f)
        {
            var fInt = FloatToInt(scaled);
            var fractVal = ShaderMath.Fract(scaled);
            if (fInt % 2 == 0)
            {
                result = 1.0f - fractVal;
            }
            else
            {
                result = fractVal;
            }
        }

        return (result - 0.2f) * (1.0f / freq);
    }

    [Callable]
    private static float Sdf(float3 o)
    {
        var wall = ShaderMath.Min(o.Y + 0.1f, o.Z + 0.4f);
        var sphere = ShaderMath.Length(o - new float3(0.0f, 0.35f, 0.0f)) - 0.36f;

        var q = ShaderMath.Abs(o - new float3(0.8f, 0.3f, 0.0f)) - new float3(0.3f);
        var box = ShaderMath.Length(ShaderMath.Max(q, new float3(0.0f))) + ShaderMath.Min(ShaderMath.Max(ShaderMath.Max(q.X, q.Y), q.Z), 0.0f);

        var shifted = o - new float3(-0.8f, 0.3f, 0.0f);
        var d = new float2(ShaderMath.Length(new float2(shifted.X, shifted.Z)) - 0.3f, ShaderMath.Abs(shifted.Y) - 0.3f);
        var cylinder = ShaderMath.Min(ShaderMath.Max(d.X, d.Y), 0.0f) + ShaderMath.Length(ShaderMath.Max(d, new float2(0.0f)));

        var geometry = MakeNested(ShaderMath.Min(ShaderMath.Min(sphere, box), cylinder));
        var g = ShaderMath.Max(geometry, -(0.32f - ((o.Y * 0.6f) + (o.Z * 0.8f))));

        return ShaderMath.Min(wall, g);
    }

    [Callable]
    private static float RayMarch(float3 p, float3 direction)
    {
        var dist = 0.0f;

        for (var i = 0; i < 100; i++)
        {
            var s = Sdf(p + (dist * direction));
            if (s <= 1e-6f || dist >= Inf)
            {
                break;
            }

            dist += s;
        }

        return ShaderMath.Min(dist, Inf);
    }

    [Callable]
    private static float3 SdfNormal(float3 p)
    {
        var delta = 1e-3f;
        var sdfCenter = Sdf(p);

        var incX = p + new float3(delta, 0.0f, 0.0f);
        var incY = p + new float3(0.0f, delta, 0.0f);
        var incZ = p + new float3(0.0f, 0.0f, delta);

        var nx = (1.0f / delta) * (Sdf(incX) - sdfCenter);
        var ny = (1.0f / delta) * (Sdf(incY) - sdfCenter);
        var nz = (1.0f / delta) * (Sdf(incZ) - sdfCenter);

        return ShaderMath.Normalize(new float3(nx, ny, nz));
    }

    [Callable]
    private static HitInfo NextHit(float3 pos, float3 direction)
    {
        var closest = Inf;
        var normal = new float3(0.0f);
        var color = new float3(0.0f);

        var rayMarchDist = RayMarch(pos, direction);
        if (rayMarchDist < ShaderMath.Min(DistLimit, closest))
        {
            closest = rayMarchDist;
            var hitPos = pos + (direction * closest);
            normal = SdfNormal(hitPos);
            var t = FloatToInt(((hitPos.X + 10.0f) * 1.1f) + 0.5f) % 3;

            var baseColor = new float3(0.4f, 0.4f, 0.4f);
            var pattern = new float3(0.3f, 0.2f, 0.3f);

            if (t == 0)
            {
                color = baseColor + pattern;
            }
            else if (t == 1)
            {
                color = baseColor + new float3(pattern.Y, pattern.X, pattern.Z);
            }
            else
            {
                color = baseColor + new float3(pattern.Z, pattern.Y, pattern.X);
            }
        }

        return new HitInfo(closest, normal, color);
    }
}

[GpuStruct]
public readonly partial record struct HitInfo(float Closest, float3 Normal, float3 Color);

internal static class SdfImageWriter
{
    public static void SaveExposedSrgbTga(string path, ReadOnlySpan<float4> pixels, int width, int height, float exposureScale)
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
                encoded[offset] = ToByte(ToSrgb(pixel.Z * exposureScale));
                encoded[offset + 1] = ToByte(ToSrgb(pixel.Y * exposureScale));
                encoded[offset + 2] = ToByte(ToSrgb(pixel.X * exposureScale));
                encoded[offset + 3] = 255;
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

    private static float ToSrgb(float c)
    {
        if (c <= 0.00031308f)
        {
            return c * 12.92f;
        }

        return (1.055f * MathF.Pow(c, 1.0f / 2.4f)) - 0.055f;
    }

    private static byte ToByte(float value)
        => (byte)(255.0f * Math.Clamp(value, 0.0f, 1.0f));
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
            !glsl.Contains("normalize", StringComparison.Ordinal) ||
            !glsl.Contains("cross", StringComparison.Ordinal) ||
            !glsl.Contains("for", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce the expected EasyGPU SDF GLSL.");
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
