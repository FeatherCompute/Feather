using System.Diagnostics;
using Feather;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

var width = args.Length > 0 && int.TryParse(args[0], out var parsedWidth) ? parsedWidth : 1280;
var height = args.Length > 1 && int.TryParse(args[1], out var parsedHeight) ? parsedHeight : 720;
ArgumentOutOfRangeException.ThrowIfLessThan(width, 2);
ArgumentOutOfRangeException.ThrowIfLessThan(height, 2);

SampleProof.PrintBackend(GPU.Context);
SampleProof.AssertEasyGpuGlsl<VolumetricFogKernel>();

using var image = GPU.CreateBuffer<float4>(width * height, BufferAccess.ReadWrite);

var stopwatch = Stopwatch.StartNew();
var path = GPU.DispatchAndGetPath(
    new VolumetricFogKernel(
        image.AsReadWrite(),
        new Uniform<int>(width),
        new Uniform<int>(height)),
    new int2(width, height));
stopwatch.Stop();
SampleProof.AssertTypedEasyGpu(path);

var pixels = image.ToArray();
var imagePath = Path.GetFullPath(Path.Combine("artifacts", "images", "volumetric-fog-feather.tga"));
VolumetricFogImageWriter.SaveFloat4Tga(imagePath, pixels, width, height);
var pngPath = VolumetricFogImageWriter.TryConvertToPng(imagePath);
var proof = SampleProof.MeasureImage(pixels);

Console.WriteLine("Feather Volumetric Cloud Renderer");
Console.WriteLine($"Image: {width}x{height}, ray marching steps=256");
Console.WriteLine($"Dispatch path: {path}");
Console.WriteLine($"Render time: {stopwatch.ElapsedMilliseconds} ms");
Console.WriteLine($"Lit pixels: {proof.LitPixels}");
Console.WriteLine($"Distinct quantized colors: {proof.DistinctColors}");
Console.WriteLine($"Image written: {imagePath}");
if (pngPath is not null)
{
    Console.WriteLine($"PNG written: {pngPath}");
}

if (proof.LitPixels <= pixels.Length / 2 || proof.DistinctColors <= 16)
{
    throw new InvalidOperationException("Volumetric fog kernel produced an image without meaningful cloud and sky variation.");
}

if (new FileInfo(imagePath).Length <= 18)
{
    throw new InvalidOperationException("Volumetric fog image artifact is empty.");
}

Console.WriteLine("PASS");

/// <summary>
/// Ports EasyGPU's volumetric fog/cloud renderer to Feather's C# kernel DSL.
/// </summary>
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct VolumetricFogKernel(
    ReadWriteBuffer<float4> image,
    Uniform<int> width,
    Uniform<int> height) : IKernel2D
{
    private const int MaxSteps = 256;
    private const float StepSize = 0.02f;
    private const float DensityScale = 8.0f;
    private const float Absorption = 0.8f;
    private const float Scattering = 0.9f;
    private const float Ambient = 0.1f;
    private const float Fov = 1.2f;
    private const float LightIntensity = 15.0f;
    private const float NoiseFreq = 2.0f;
    private const float NoiseAmp = 1.0f;
    private const int NoiseOctaves = 4;

    /// <summary>
    /// Renders one volumetric cloud pixel.
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

        float3 cameraPos = new float3(0.0f, 1.5f, 4.0f);
        float3 cameraLook = new float3(0.0f, -0.2f, -1.0f);
        float3 cameraUpInput = new float3(0.0f, 1.0f, 0.0f);

        float3 cameraForward = ShaderMath.Normalize(cameraLook);
        float3 cameraRight = ShaderMath.Normalize(ShaderMath.Cross(cameraForward, cameraUpInput));
        float3 cameraUp = ShaderMath.Cross(cameraRight, cameraForward);

        float aspectRatio = (float)imageWidth / (float)imageHeight;
        float scale = ShaderMath.Tan(Fov * 0.5f);

        float u = ((2.0f * ((float)pixel.X + 0.5f) / (float)imageWidth) - 1.0f) * aspectRatio * scale;
        float v = (1.0f - (2.0f * ((float)pixel.Y + 0.5f) / (float)imageHeight)) * scale;

        float3 rayDir = ShaderMath.Normalize(cameraForward + (cameraRight * u) + (cameraUp * v));
        float3 color = RenderVolume(cameraPos, rayDir);
        color = ToneMapAces(color);

        float gamma = 1.0f / 2.2f;
        color = new float3(
            ShaderMath.Pow(color.X, gamma),
            ShaderMath.Pow(color.Y, gamma),
            ShaderMath.Pow(color.Z, gamma));

        image[(pixel.Y * imageWidth) + pixel.X] = new float4(color, 1.0f);
    }

    [Callable]
    private static float Hash(float3 p)
    {
        float3 q = p * new float3(0.1031f, 0.1030f, 0.0973f);
        float h = ShaderMath.Fract(q.X + q.Y + q.Z);
        h *= 0.47f;
        return ShaderMath.Fract(h * 93.31f);
    }

    [Callable]
    private static float ValueNoise(float3 p)
    {
        float3 i = ShaderMath.Floor(p);
        float3 f = ShaderMath.Fract(p);
        f = f * f * (new float3(3.0f) - (new float3(2.0f) * f));

        float n = 0.0f;
        for (int idx = 0; idx < 8; idx++)
        {
            float dx = (float)(idx % 2);
            float dy = (float)((idx / 2) % 2);
            float dz = (float)(idx / 4);

            float3 offset = new float3(dx, dy, dz);
            float3 samplePos = i + offset;
            float h = Hash(samplePos);
            float weight = (1.0f - ShaderMath.Abs(f.X - dx)) *
                (1.0f - ShaderMath.Abs(f.Y - dy)) *
                (1.0f - ShaderMath.Abs(f.Z - dz));

            n += h * weight;
        }

        return n;
    }

    [Callable]
    private static float Fbm(float3 p)
    {
        float value = 0.0f;
        float amplitude = NoiseAmp;
        float frequency = NoiseFreq;

        for (int i = 0; i < NoiseOctaves; i++)
        {
            float3 scaledP = p * frequency;
            value += amplitude * ValueNoise(scaledP);
            amplitude *= 0.5f;
            frequency *= 2.0f;
        }

        return value;
    }

    [Callable]
    private static float GetDensity(float3 pos)
    {
        float3 center = new float3(0.0f, 0.5f, 0.0f);
        float3 scale = new float3(1.5f, 0.8f, 1.5f);

        float3 diff = pos - center;
        float dist = ShaderMath.Length(diff / scale);

        float baseShape = 0.0f;
        if (dist < 1.0f)
        {
            baseShape = ShaderMath.Pow(1.0f - dist, 2.0f);
        }

        float noise = Fbm(pos + new float3(1.5f, 2.3f, 0.7f));
        float density = baseShape * noise * DensityScale;
        density = ShaderMath.Max(density - 0.5f, 0.0f) * 2.0f;

        return density;
    }

    [Callable]
    private static float LightTransmittance(float3 from, float3 to)
    {
        float3 dir = ShaderMath.Normalize(to - from);
        float dist = ShaderMath.Length(to - from);

        float transmittance = 1.0f;
        float t = 0.1f;

        for (int i = 0; i < 32; i++)
        {
            if (t >= dist)
            {
                break;
            }

            float3 samplePos = from + (dir * t);
            float density = GetDensity(samplePos);
            transmittance *= ShaderMath.Exp(-density * Absorption * 0.05f);
            t += 0.05f;
        }

        return ShaderMath.Max(transmittance, 0.001f);
    }

    [Callable]
    private static float PhaseHg(float cosTheta)
    {
        float g = 0.3f;
        float gg = g * g;
        float denom = 1.0f + gg - (2.0f * g * cosTheta);
        float result = (1.0f - gg) / ShaderMath.Pow(denom, 1.5f);
        return result / (4.0f * 3.14159f);
    }

    [Callable]
    private static float3 SkyColor(float3 rayDir)
    {
        float y = rayDir.Y;
        float3 horizonColor = new float3(1.0f, 0.5f, 0.2f);
        float3 zenithColor = new float3(0.1f, 0.3f, 0.6f);

        float t = ShaderMath.Clamp((y * 0.5f) + 0.5f, 0.0f, 1.0f);
        t = ShaderMath.Pow(t, 0.7f);

        float3 color = ShaderMath.Mix(horizonColor, zenithColor, t);

        float3 lightPos = new float3(4.0f, 2.0f, -2.0f);
        float3 sunDir = ShaderMath.Normalize(lightPos);
        float cosAngle = ShaderMath.Dot(rayDir, sunDir);
        if (cosAngle > 0.995f)
        {
            float sunIntensity = ShaderMath.Pow((cosAngle - 0.995f) / 0.005f, 2.0f);
            color += new float3(1.0f, 0.9f, 0.7f) * sunIntensity * 5.0f;
        }

        return color;
    }

    [Callable]
    private static float3 RenderVolume(float3 rayOrigin, float3 rayDir)
    {
        float3 bgColor = SkyColor(rayDir);
        float3 result = bgColor;

        float3 center = new float3(0.0f, 0.5f, 0.0f);
        float radius = 2.5f;

        float3 oc = rayOrigin - center;
        float a = ShaderMath.Dot(rayDir, rayDir);
        float b = 2.0f * ShaderMath.Dot(oc, rayDir);
        float c = ShaderMath.Dot(oc, oc) - (radius * radius);
        float discriminant = (b * b) - (4.0f * a * c);

        if (discriminant >= 0.0f)
        {
            float sqrtDisc = ShaderMath.Sqrt(discriminant);
            float tNear = (-b - sqrtDisc) / (2.0f * a);
            float tFar = (-b + sqrtDisc) / (2.0f * a);

            if (tNear < 0.0f)
            {
                tNear = 0.0f;
            }

            if (tFar >= 0.0f)
            {
                float3 accumulatedLight = new float3(0.0f);
                float transmittance = 1.0f;
                float t = tNear;

                for (int i = 0; i < MaxSteps; i++)
                {
                    if (t > tFar || transmittance < 0.001f)
                    {
                        break;
                    }

                    float3 pos = rayOrigin + (rayDir * t);
                    float density = GetDensity(pos);

                    if (density > 0.01f)
                    {
                        float3 lightPos = new float3(4.0f, 2.0f, -2.0f);
                        float3 lightColor = new float3(1.0f, 0.7f, 0.4f);
                        float3 toLight = lightPos - pos;
                        float lightDist = ShaderMath.Length(toLight);
                        float3 lightDir = ShaderMath.Normalize(toLight);

                        float lightTrans = LightTransmittance(pos + (lightDir * 0.05f), lightPos);
                        float cosTheta = ShaderMath.Dot(rayDir, lightDir);
                        float phase = PhaseHg(cosTheta);

                        float3 sunLight = lightColor * LightIntensity * lightTrans * phase * Scattering;
                        float3 ambient = new float3(0.3f, 0.4f, 0.6f) * Ambient;
                        float3 lighting = sunLight + ambient;

                        float stepTrans = ShaderMath.Exp(-density * Absorption * StepSize);
                        float3 stepLight = lighting * density * StepSize * transmittance;

                        accumulatedLight += stepLight;
                        transmittance *= stepTrans;
                    }

                    t += StepSize;
                }

                result = accumulatedLight + (bgColor * transmittance);
            }
        }

        return result;
    }

    [Callable]
    private static float3 ToneMapAces(float3 color)
    {
        float a = 2.51f;
        float b = 0.03f;
        float c = 2.43f;
        float d = 0.59f;
        float e = 0.14f;

        float3 x = color;
        float3 result = (x * ((a * x) + new float3(b))) / ((x * ((c * x) + new float3(d))) + new float3(e));
        return ShaderMath.Clamp(result, 0.0f, 1.0f);
    }
}

internal static class VolumetricFogImageWriter
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
            !glsl.Contains("exp", StringComparison.Ordinal) ||
            !glsl.Contains("pow", StringComparison.Ordinal) ||
            !glsl.Contains("for", StringComparison.Ordinal) ||
            glsl.Contains("Feather native stub", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"{typeof(TKernel).Name} did not produce the expected EasyGPU volumetric GLSL.");
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
