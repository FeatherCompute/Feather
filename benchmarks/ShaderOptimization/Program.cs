using System.Diagnostics;
using System.Text.Json;
using Feather;
using Feather.Interop;
using Feather.Resources;

var options = ExportOptions.Parse(args);
Directory.CreateDirectory(options.OutputDirectory);
foreach (var scenario in new[] { "fused-mlp", "particle-sim" })
{
    var source = Path.Combine(options.HandwrittenSourceDirectory, $"handwritten-{scenario}.comp");
    var destination = Path.Combine(options.OutputDirectory, Path.GetFileName(source));
    File.Copy(source, destination, overwrite: true);
}

var exports = new[]
{
    Export<FusedMlpBenchmarkKernel>("fused-mlp", options),
    Export<ParticleSimulationBenchmarkKernel>("particle-sim", options)
};

var metadataPath = Path.Combine(options.OutputDirectory, "feather-source-metadata.json");
File.WriteAllText(metadataPath, JsonSerializer.Serialize(exports, new JsonSerializerOptions { WriteIndented = true }));

foreach (var export in exports)
{
    Console.WriteLine(FormattableString.Invariant(
        $"{export.Name}: {export.SourceBytes} bytes, source lowering median {export.SourceLoweringMedianMs:F3} ms"));
}

Console.WriteLine($"Metadata: {metadataPath}");

static SourceExport Export<TKernel>(string name, ExportOptions options)
    where TKernel : struct, IGeneratedKernel<TKernel>
{
    _ = ShaderInspection.GetGLSL<TKernel>();

    var samples = new double[options.Samples];
    string source = string.Empty;
    for (var i = 0; i < samples.Length; i++)
    {
        var stopwatch = Stopwatch.StartNew();
        source = ShaderInspection.GetGLSL<TKernel>();
        stopwatch.Stop();
        samples[i] = stopwatch.Elapsed.TotalMilliseconds;
    }

    Array.Sort(samples);
    var sourcePath = Path.Combine(options.OutputDirectory, $"feather-{name}.comp");
    File.WriteAllText(sourcePath, source);
    return new SourceExport(
        name,
        Path.GetFileName(sourcePath),
        source.Length,
        source.Count(static c => c == '\n') + 1,
        samples[samples.Length / 2]);
}

file readonly record struct ExportOptions(string OutputDirectory, string HandwrittenSourceDirectory, int Samples)
{
    public static ExportOptions Parse(string[] args)
    {
        var outputDirectory = Path.GetFullPath(Path.Combine("artifacts", "optimization-benchmark", "sources"));
        var handwrittenSourceDirectory = Path.GetFullPath(
            Path.Combine("benchmarks", "ShaderOptimization", "shaders"));
        var samples = 9;
        for (var i = 0; i < args.Length; i++)
        {
            if (args[i] == "--output" && i + 1 < args.Length)
            {
                outputDirectory = Path.GetFullPath(args[++i]);
            }
            else if (args[i].StartsWith("--output=", StringComparison.Ordinal))
            {
                outputDirectory = Path.GetFullPath(args[i]["--output=".Length..]);
            }
            else if (args[i] == "--handwritten-sources" && i + 1 < args.Length)
            {
                handwrittenSourceDirectory = Path.GetFullPath(args[++i]);
            }
            else if (args[i].StartsWith("--handwritten-sources=", StringComparison.Ordinal))
            {
                handwrittenSourceDirectory = Path.GetFullPath(args[i]["--handwritten-sources=".Length..]);
            }
            else if (args[i] == "--samples" && i + 1 < args.Length && int.TryParse(args[++i], out var parsed))
            {
                samples = parsed;
            }
            else if (args[i].StartsWith("--samples=", StringComparison.Ordinal) &&
                     int.TryParse(args[i]["--samples=".Length..], out parsed))
            {
                samples = parsed;
            }
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(samples, 1);
        return new ExportOptions(outputDirectory, handwrittenSourceDirectory, samples);
    }
}

file readonly record struct SourceExport(
    string Name,
    string SourceFile,
    int SourceBytes,
    int SourceLines,
    double SourceLoweringMedianMs);

/// <summary>
/// A fused, arithmetic-heavy residual block representative of elementwise NN work.
/// </summary>
[Kernel(BoundsCheck = false)]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct FusedMlpBenchmarkKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Applies 64 fused affine, activation, polynomial, and residual steps.
    /// </summary>
    public void Execute()
    {
        int index = ThreadIds.X;
        float value = input[index];
        float residual = 0.0f;

        for (int layer = 0; layer < 64; layer++)
        {
            float bias = (float)((layer % 7) - 3) * 0.0005f;
            float affine = (value * 1.0009765625f) + bias;
            float activated = affine > 0.0f ? affine : affine * 0.125f;
            value = activated + ((activated * activated) * 0.000244140625f);
            residual += value * 0.015625f;
        }

        output[index] = value + residual;
    }
}

/// <summary>
/// A fixed-step particle integrator representative of branch-heavy physical simulation kernels.
/// </summary>
[Kernel(BoundsCheck = false)]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct ParticleSimulationBenchmarkKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    /// <summary>
    /// Integrates one particle for 128 fixed steps with bounded collision response.
    /// </summary>
    public void Execute()
    {
        int index = ThreadIds.X;
        float seed = input[index];
        float position = (seed * 2.0f) - 1.0f;
        float velocity = (seed - 0.5f) * 0.01f;
        float energy = 0.0f;

        for (int step = 0; step < 128; step++)
        {
            float wind = (float)((step % 11) - 5) * 0.00002f;
            float force = (-position * 0.0015f) + wind;
            velocity += force * 0.125f;
            position += velocity * 0.125f;

            if (position > 1.0f)
            {
                position = 2.0f - position;
                velocity = -velocity * 0.85f;
            }
            else if (position < -1.0f)
            {
                position = -2.0f - position;
                velocity = -velocity * 0.85f;
            }

            energy += (position * position) + (velocity * velocity);
        }

        output[index] = position + (energy * 0.0001f);
    }
}
