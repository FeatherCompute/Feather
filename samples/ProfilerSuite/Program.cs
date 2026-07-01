using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Feather;
using Feather.AD;
using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.NN;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

var options = ProfilerOptions.Parse(args);
Directory.CreateDirectory(options.OutputDirectory);

var caps = GPU.Context.Caps;
var suite = new ProfilerSuite(options, new BackendInfo(
    caps.BackendType.ToString(),
    caps.MaxWorkGroupSizeX,
    caps.MaxWorkGroupSizeY,
    caps.MaxWorkGroupSizeZ));

Console.WriteLine("=== Feather Profiler Suite ===");
Console.WriteLine($"backend={suite.Backend.BackendType} maxWorkGroup={suite.Backend.MaxWorkGroupSizeX}x{suite.Backend.MaxWorkGroupSizeY}x{suite.Backend.MaxWorkGroupSizeZ}");
Console.WriteLine($"warmup={options.Warmup} iterations={options.Iterations} output={options.OutputDirectory}");

suite.Run("compute.buffer.increment.1m", "Compute", "1,048,576 int increments", MeasureComputeBuffer);
suite.Run("compute.texture.copy.512", "Compute", "512x512 RGBA8 image load/store copy", MeasureTextureCopy);
suite.Run("ad.linear.backward.4096", "AD", "4,096 scalar backward launches over two parameters", MeasureAdLinear);
suite.Run("nn.gpt.train.small", "NN", "tiny GPT trainer, 8x8 tokens, Adam step", MeasureSmallGptTraining);
suite.Run("graphics.triangle.1280x720", "Graphics", "non-indexed triangle draw to RGBA8 render target", MeasureTriangleDraw);
suite.Run("graphics.textured_quad.1280x720", "Graphics", "indexed sampled-texture quad draw to RGBA8 render target", MeasureTexturedQuadDraw);
suite.TryRunSponzaCapture();

suite.WriteReports();
Console.WriteLine($"JSON: {suite.JsonPath}");
Console.WriteLine($"Markdown: {suite.MarkdownPath}");

static CaseMeasurement MeasureComputeBuffer(ProfilerCaseContext context)
{
    const int count = 1_048_576;
    var inputData = new int[count];
    for (var i = 0; i < inputData.Length; i++)
    {
        inputData[i] = i;
    }

    using var input = GPU.CreateBuffer<int>(inputData, BufferAccess.ReadOnly);
    using var output = GPU.CreateBuffer<int>(count, BufferAccess.ReadWrite);
    var kernel = new ProfileIncrementKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<int>(count));
    var path = DispatchPath.None;

    var profiler = context.Measure("ProfileIncrementKernel", () =>
    {
        path = GPU.DispatchAndGetPath(kernel, count);
    });

    var result = output.ToArray();
    Validate(result[0] == 1 && result[^1] == count, "compute buffer increment validation failed.");
    Validate(path == DispatchPath.TypedEasyGpu, $"compute buffer path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["elements"] = count,
            ["bytesRead"] = count * sizeof(int),
            ["bytesWritten"] = count * sizeof(int),
            ["lastElement"] = result[^1]
        });
}

static CaseMeasurement MeasureTextureCopy(ProfilerCaseContext context)
{
    const int width = 512;
    const int height = 512;
    var pixels = new Rgba32[width * height];
    for (var y = 0; y < height; y++)
    {
        for (var x = 0; x < width; x++)
        {
            pixels[(y * width) + x] = new Rgba32((byte)(x & 255), (byte)(y & 255), (byte)((x + y) & 255), 255);
        }
    }

    using var input = GPU.CreateTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8, TextureAccess.ReadOnly);
    using var output = GPU.CreateTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8, TextureAccess.ReadWrite);
    input.Upload(pixels);
    var kernel = new ProfileTextureCopyKernel(input.AsReadOnly(), output.AsReadWrite());
    var path = DispatchPath.None;

    var profiler = context.Measure("ProfileTextureCopyKernel", () =>
    {
        path = GPU.DispatchAndGetPath(kernel, new int2(width, height));
    });

    var readback = new Rgba32[pixels.Length];
    output.Read(readback);
    Validate(readback[0].Equals(pixels[0]) && readback[^1].Equals(pixels[^1]), "texture copy validation failed.");
    Validate(path == DispatchPath.TypedEasyGpu, $"texture copy path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["width"] = width,
            ["height"] = height,
            ["pixels"] = pixels.Length,
            ["bytesRead"] = pixels.Length * 4,
            ["bytesWritten"] = pixels.Length * 4
        });
}

static CaseMeasurement MeasureAdLinear(ProfilerCaseContext context)
{
    const int count = 4096;
    var x = new float[count];
    var y = new float[count];
    for (var i = 0; i < count; i++)
    {
        x[i] = 0.25f + (i % 37) * 0.01f;
        y[i] = 0.75f + (i % 17) * 0.02f;
    }

    using var xBuffer = GPU.CreateBuffer<float>(x, BufferAccess.ReadOnly);
    using var yBuffer = GPU.CreateBuffer<float>(y, BufferAccess.ReadOnly);
    using var w = GPU.CreateBuffer<float>([0.3f], BufferAccess.ReadWrite);
    using var b = GPU.CreateBuffer<float>([0.1f], BufferAccess.ReadWrite);
    using var loss = GPU.CreateBuffer<float>(count, BufferAccess.ReadWrite);
    using var ad = GPU.CreateADKernel(new ProfileLinearAdKernel(
        xBuffer.AsReadOnly(),
        yBuffer.AsReadOnly(),
        w.AsReadWrite(),
        b.AsReadWrite(),
        loss.AsReadWrite()));
    var path = DispatchPath.None;

    var profiler = context.Measure("ProfileLinearAdKernel", () =>
    {
        ad.Backward(count);
        path = ad.LastDispatchPath;
    });

    using var wGradient = GPU.CreateBuffer<float>(1, BufferAccess.ReadWrite);
    using var bGradient = GPU.CreateBuffer<float>(1, BufferAccess.ReadWrite);
    ad.CopyGradientToBuffer("w", wGradient);
    ad.CopyGradientToBuffer("b", bGradient);
    var gradients = new[] { wGradient.ToArray()[0], bGradient.ToArray()[0] };
    var lossValues = loss.ToArray();
    Validate(gradients.All(float.IsFinite), "AD gradients were not finite.");
    Validate(path == DispatchPath.TypedEasyGpu, $"AD path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["elements"] = count,
            ["lossSum"] = lossValues.Sum(),
            ["gradientW"] = gradients[0],
            ["gradientB"] = gradients[1],
            ["backwardGlslBytes"] = ad.GetBackwardGLSL().Length
        });
}

static CaseMeasurement MeasureSmallGptTraining(ProfilerCaseContext context)
{
    const int vocabSize = 16;
    const int blockSize = 8;
    const int embeddingSize = 8;
    const int headCount = 2;
    const int batchSize = 8;
    using var model = new GptLanguageModel(vocabSize, blockSize, embeddingSize, headCount, seed: 7);
    using var optimizer = new Adam(model.Parameters, learningRate: 0.00005f, beta1: 0.85f, beta2: 0.99f, epsilon: 1e-5f, weightDecay: 1e-5f, gradientClip: 0.25f);
    using var trainer = model.CreateTrainer(batchSize, optimizer);
    var batch = new int[batchSize * (blockSize + 1)];
    for (var row = 0; row < batchSize; row++)
    {
        for (var i = 0; i < blockSize + 1; i++)
        {
            batch[(row * (blockSize + 1)) + i] = (row + i) % vocabSize;
        }
    }

    var path = DispatchPath.None;
    var profiler = context.Measure("GptLanguageModelTrainingKernel", () =>
    {
        _ = trainer.TrainBatch(batch);
        path = trainer.LastDispatchPath;
    });

    var evalLoss = trainer.EvaluateBatch(batch);
    Validate(float.IsFinite(evalLoss), "NN eval loss was not finite.");
    Validate(path == DispatchPath.TypedEasyGpu, $"NN trainer path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["batchSize"] = batchSize,
            ["blockSize"] = blockSize,
            ["embeddingSize"] = embeddingSize,
            ["headCount"] = headCount,
            ["vocabularySize"] = vocabSize,
            ["adamStep"] = optimizer.StepCount,
            ["evalLoss"] = evalLoss,
            ["gradientsMaterialized"] = trainer.GradientsMaterialized
        });
}

static CaseMeasurement MeasureTriangleDraw(ProfilerCaseContext context)
{
    const int width = 1280;
    const int height = 720;
    using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8);
    using var vertices = GPU.CreateBuffer<float4>(
    [
        new float4(-0.8f, -0.7f, 0, 1),
        new float4(0.8f, -0.7f, 0, 1),
        new float4(0, 0.75f, 0, 1)
    ], BufferAccess.ReadOnly);
    using var pipeline = GPU.CreateGraphicsPipeline<ProfileTriangleVS, ProfileTriangleFS, float4>(
        new GraphicsPipelineDesc { DebugName = "ProfilerTrianglePipeline" });
    var path = DispatchPath.None;

    var profiler = context.Measure("ProfilerTrianglePipeline", () =>
    {
        pipeline.Draw(new ProfileTriangleVS(vertices.AsReadOnly()), new ProfileTriangleFS(), target, vertexCount: 3);
        path = pipeline.LastDispatchPath;
    });

    var pixels = new Rgba32[width * height];
    target.Read(pixels);
    var visible = CountVisiblePixels(pixels);
    Validate(visible > pixels.Length / 32, "triangle draw produced too few visible pixels.");
    Validate(path == DispatchPath.TypedEasyGpu, $"triangle draw path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["width"] = width,
            ["height"] = height,
            ["vertices"] = 3,
            ["visiblePixels"] = visible
        });
}

static CaseMeasurement MeasureTexturedQuadDraw(ProfilerCaseContext context)
{
    const int width = 1280;
    const int height = 720;
    using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8);
    using var vertices = GPU.CreateBuffer<float4>(
    [
        new float4(-0.85f, -0.75f, 0, 1),
        new float4(0.85f, -0.75f, 0, 1),
        new float4(0.85f, 0.75f, 0, 1),
        new float4(-0.85f, 0.75f, 0, 1)
    ], BufferAccess.ReadOnly);
    using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2, 0, 2, 3]);
    using var texture = GPU.CreateTexture2D<Rgba32, float4>(2, 2, PixelFormat.Rgba8, TextureAccess.Sampled);
    using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
    using var pipeline = GPU.CreateGraphicsPipeline<ProfileQuadVS, ProfileQuadFS, float4>(
        new GraphicsPipelineDesc { DebugName = "ProfilerTexturedQuadPipeline" });
    texture.Upload(
    [
        new Rgba32(255, 70, 80, 255),
        new Rgba32(50, 210, 120, 255),
        new Rgba32(70, 130, 255, 255),
        new Rgba32(255, 220, 70, 255)
    ]);
    var path = DispatchPath.None;

    var profiler = context.Measure("ProfilerTexturedQuadPipeline", () =>
    {
        pipeline.DrawIndexed(new ProfileQuadVS(vertices.AsReadOnly()), new ProfileQuadFS(texture.AsSampled(), sampler), target, indices);
        path = pipeline.LastDispatchPath;
    });

    var pixels = new Rgba32[width * height];
    target.Read(pixels);
    var visible = CountVisiblePixels(pixels);
    Validate(visible > pixels.Length / 8, "textured quad draw produced too few visible pixels.");
    Validate(path == DispatchPath.TypedEasyGpu, $"textured quad path was {path}.");

    return new CaseMeasurement(
        DispatchPath: path.ToString(),
        Profiler: profiler,
        Metrics: new Dictionary<string, object?>
        {
            ["width"] = width,
            ["height"] = height,
            ["indices"] = indices.Length,
            ["visiblePixels"] = visible
        });
}

static int CountVisiblePixels(ReadOnlySpan<Rgba32> pixels)
{
    var visible = 0;
    foreach (var pixel in pixels)
    {
        if (pixel.A != 0 && (pixel.R != 0 || pixel.G != 0 || pixel.B != 0))
        {
            visible++;
        }
    }

    return visible;
}

static void Validate(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct ProfileIncrementKernel(
    ReadOnlyBuffer<int> input,
    ReadWriteBuffer<int> output,
    Uniform<int> count) : IKernel1D
{
    public void Execute()
    {
        var i = ThreadIds.X;
        if (i < count.Value)
        {
            output[i] = input[i] + 1;
        }
    }
}

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.XY)]
public readonly partial struct ProfileTextureCopyKernel(
    ReadOnlyTexture2D<Rgba32> input,
    ReadWriteTexture2D<Rgba32> output) : IKernel2D
{
    public void Execute()
    {
        var p = ThreadIds.XY;
        output[p] = input[p];
    }
}

[Kernel]
[AutoDiff]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct ProfileLinearAdKernel(
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> w,
    ReadWriteBuffer<float> b,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        var i = ThreadIds.X;
        var prediction = (w[0] * x[i]) + b[0];
        var error = prediction - y[i];
        var l = error * error;
        loss[i] = l;
        ADMarker.Parameter(w[0]);
        ADMarker.Parameter(b[0]);
        ADMarker.Loss(l);
    }
}

[VertexShader]
public readonly partial struct ProfileTriangleVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute() => vertices[VertexIds.Index];
}

[FragmentShader]
public readonly partial struct ProfileTriangleFS : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        var tint = new float3(0.25f + (input.X * 0.25f), 0.45f + (input.Y * 0.25f), 0.9f);
        return new float4(tint, 1.0f);
    }
}

[VertexShader]
public readonly partial struct ProfileQuadVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute() => vertices[VertexIds.Index];
}

[FragmentShader]
public readonly partial struct ProfileQuadFS(SampledTexture2D<float4> texture, SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        var uv = (input.XY * 0.5f) + new float2(0.5f);
        return texture.Sample(sampler, uv);
    }
}

[GpuStruct]
public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);

file sealed class ProfilerSuite(ProfilerOptions options, BackendInfo backend)
{
    private readonly List<ProfilerCaseResult> results = [];
    private readonly List<ExternalProcessResult> externalResults = [];

    public BackendInfo Backend { get; } = backend;
    public string JsonPath => Path.Combine(options.OutputDirectory, "feather-profiler-report.json");
    public string MarkdownPath => Path.Combine(options.OutputDirectory, "feather-profiler-report.md");

    public void Run(string name, string category, string workload, Func<ProfilerCaseContext, CaseMeasurement> measure)
    {
        Console.WriteLine();
        Console.WriteLine($"==> {name}");
        var context = new ProfilerCaseContext(options.Warmup, options.Iterations);
        var stopwatch = Stopwatch.StartNew();
        var measurement = measure(context);
        stopwatch.Stop();
        var result = new ProfilerCaseResult(
            Name: name,
            Category: category,
            Workload: workload,
            DispatchPath: measurement.DispatchPath,
            WarmupIterations: options.Warmup,
            MeasuredIterations: options.Iterations,
            HarnessWallTimeMs: stopwatch.Elapsed.TotalMilliseconds,
            Profiler: measurement.Profiler,
            Metrics: measurement.Metrics);
        results.Add(result);
        Console.WriteLine($"    native avg={measurement.Profiler.AverageTimeMs:F3}ms total={measurement.Profiler.TotalTimeMs:F3}ms count={measurement.Profiler.Count}");
    }

    public void TryRunSponzaCapture()
    {
        var sponzaObj = Path.Combine("Sponza", "sponza.obj");
        if (!File.Exists(sponzaObj))
        {
            externalResults.Add(new ExternalProcessResult(
                "graphics.sponza.capture",
                "Graphics",
                "Sponza capture skipped because Sponza/sponza.obj was not found.",
                false,
                null,
                null,
                null,
                []));
            return;
        }

        Console.WriteLine();
        Console.WriteLine("==> graphics.sponza.capture");
        var outputPath = Path.Combine(options.OutputDirectory, "sponza-profiler.tga");
        var arguments = $"run --no-restore --project samples/SponzaRenderer/SponzaRenderer.csproj -- Sponza --capture \"{outputPath}\"";
        var startInfo = new ProcessStartInfo("dotnet", arguments)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        };
        var stopwatch = Stopwatch.StartNew();
        using var process = Process.Start(startInfo) ?? throw new InvalidOperationException("Failed to start SponzaRenderer.");
        var stdout = process.StandardOutput.ReadToEnd();
        var stderr = process.StandardError.ReadToEnd();
        process.WaitForExit();
        stopwatch.Stop();

        var lines = (stdout + stderr)
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Where(static line => line.Contains("Positions:", StringComparison.Ordinal) ||
                                  line.Contains("Vertices:", StringComparison.Ordinal) ||
                                  line.Contains("Capture pixels:", StringComparison.Ordinal) ||
                                  line.Contains("Captured ", StringComparison.Ordinal))
            .ToArray();
        var fileSize = File.Exists(outputPath) ? new FileInfo(outputPath).Length : (long?)null;
        externalResults.Add(new ExternalProcessResult(
            "graphics.sponza.capture",
            "Graphics",
            "SponzaRenderer sample capture, full asset load plus one draw plus readback.",
            process.ExitCode == 0,
            process.ExitCode,
            stopwatch.Elapsed.TotalMilliseconds,
            outputPath,
            lines.Concat(fileSize is null ? [] : [$"output bytes={fileSize.Value:N0}"]).ToArray()));

        Console.WriteLine($"    exit={process.ExitCode} wall={stopwatch.Elapsed.TotalMilliseconds:F1}ms");
    }

    public void WriteReports()
    {
        var report = new ProfilerReport(
            GeneratedAtUtc: DateTimeOffset.UtcNow,
            Backend: Backend,
            Options: options,
            Cases: results,
            ExternalProcesses: externalResults);

        var jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };
        File.WriteAllText(JsonPath, JsonSerializer.Serialize(report, jsonOptions));
        File.WriteAllText(MarkdownPath, RenderMarkdown(report));
    }

    private static string RenderMarkdown(ProfilerReport report)
    {
        var sb = new StringBuilder();
        sb.AppendLine("# Feather Profiler Report");
        sb.AppendLine();
        sb.AppendLine($"Generated UTC: `{report.GeneratedAtUtc:O}`");
        sb.AppendLine();
        sb.AppendLine("## Backend");
        sb.AppendLine();
        sb.AppendLine($"- Backend: `{report.Backend.BackendType}`");
        sb.AppendLine($"- Max workgroup size: `{report.Backend.MaxWorkGroupSizeX}x{report.Backend.MaxWorkGroupSizeY}x{report.Backend.MaxWorkGroupSizeZ}`");
        sb.AppendLine($"- Warmup iterations per case: `{report.Options.Warmup}`");
        sb.AppendLine($"- Measured iterations per case: `{report.Options.Iterations}`");
        sb.AppendLine();
        sb.AppendLine("## Native Profiler Cases");
        sb.AppendLine();
        sb.AppendLine("| Case | Category | Workload | Path | Count | Avg ms | Min ms | Max ms | Main total ms | All native total ms | Harness wall ms |");
        sb.AppendLine("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
        foreach (var item in report.Cases)
        {
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| `{item.Name}` | {item.Category} | {item.Workload} | `{item.DispatchPath}` | {item.Profiler.Count} | {item.Profiler.AverageTimeMs:F3} | {item.Profiler.MinTimeMs:F3} | {item.Profiler.MaxTimeMs:F3} | {item.Profiler.TotalTimeMs:F3} | {item.Profiler.ProcessProfilerTotalTimeMs:F3} | {item.HarnessWallTimeMs:F3} |");
        }

        sb.AppendLine();
        sb.AppendLine("## Case Metrics");
        foreach (var item in report.Cases)
        {
            sb.AppendLine();
            sb.AppendLine($"### {item.Name}");
            foreach (var metric in item.Metrics)
            {
                sb.AppendLine($"- `{metric.Key}`: `{metric.Value}`");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## External Processes");
        sb.AppendLine();
        sb.AppendLine("| Case | Category | Success | Exit | Wall ms | Output |");
        sb.AppendLine("| --- | --- | ---: | ---: | ---: | --- |");
        foreach (var item in report.ExternalProcesses)
        {
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| `{item.Name}` | {item.Category} | {item.Success} | {item.ExitCode?.ToString(CultureInfo.InvariantCulture) ?? ""} | {item.WallTimeMs?.ToString("F3", CultureInfo.InvariantCulture) ?? ""} | `{item.OutputPath ?? ""}` |");
        }

        foreach (var item in report.ExternalProcesses)
        {
            if (item.KeyLines.Count == 0)
            {
                continue;
            }

            sb.AppendLine();
            sb.AppendLine($"### {item.Name} output");
            foreach (var line in item.KeyLines)
            {
                sb.AppendLine($"- {line}");
            }
        }

        sb.AppendLine();
        sb.AppendLine("## Raw Native Profiler Reports");
        foreach (var item in report.Cases)
        {
            sb.AppendLine();
            sb.AppendLine($"### {item.Name}");
            sb.AppendLine();
            sb.AppendLine("```text");
            sb.AppendLine(item.Profiler.FormattedReport.TrimEnd());
            sb.AppendLine("```");
        }

        return sb.ToString();
    }
}

file sealed class ProfilerCaseContext(int warmupIterations, int measuredIterations)
{
    public ProfilerSnapshot Measure(string profilerName, Action action)
    {
        for (var i = 0; i < warmupIterations; i++)
        {
            action();
        }

        GpuProfiler.SetEnabled(true);
        GpuProfiler.Clear();
        for (var i = 0; i < measuredIterations; i++)
        {
            action();
        }

        var query = GpuProfiler.Query(profilerName);
        var total = GpuProfiler.GetTotalTimeMs();
        if (query.Count != (ulong)measuredIterations)
        {
            throw new InvalidOperationException($"profiler expected {measuredIterations} records for {profilerName}, got {query.Count}.");
        }

        return new ProfilerSnapshot(
            profilerName,
            query.Count,
            query.MinTimeMs,
            query.MaxTimeMs,
            query.AverageTimeMs,
            query.TotalTimeMs,
            total,
            GpuProfiler.GetFormattedReport());
    }
}

file readonly record struct ProfilerOptions(int Warmup, int Iterations, string OutputDirectory)
{
    public static ProfilerOptions Parse(string[] args)
    {
        var warmup = 3;
        var iterations = 30;
        var outputDirectory = Path.GetFullPath(Path.Combine("artifacts", "profiler"));

        for (var i = 0; i < args.Length; i++)
        {
            var arg = args[i];
            if (arg == "--quick")
            {
                warmup = 1;
                iterations = 5;
                continue;
            }

            if (arg == "--warmup" && i + 1 < args.Length && int.TryParse(args[++i], out var parsedWarmup))
            {
                warmup = parsedWarmup;
                continue;
            }

            if (arg.StartsWith("--warmup=", StringComparison.Ordinal) && int.TryParse(arg["--warmup=".Length..], out parsedWarmup))
            {
                warmup = parsedWarmup;
                continue;
            }

            if (arg == "--iterations" && i + 1 < args.Length && int.TryParse(args[++i], out var parsedIterations))
            {
                iterations = parsedIterations;
                continue;
            }

            if (arg.StartsWith("--iterations=", StringComparison.Ordinal) && int.TryParse(arg["--iterations=".Length..], out parsedIterations))
            {
                iterations = parsedIterations;
                continue;
            }

            if (arg == "--output" && i + 1 < args.Length)
            {
                outputDirectory = Path.GetFullPath(args[++i]);
                continue;
            }

            if (arg.StartsWith("--output=", StringComparison.Ordinal))
            {
                outputDirectory = Path.GetFullPath(arg["--output=".Length..]);
            }
        }

        ArgumentOutOfRangeException.ThrowIfNegative(warmup);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(iterations);
        return new ProfilerOptions(warmup, iterations, outputDirectory);
    }
}

file readonly record struct CaseMeasurement(
    string DispatchPath,
    ProfilerSnapshot Profiler,
    IReadOnlyDictionary<string, object?> Metrics);

file readonly record struct ProfilerSnapshot(
    string Name,
    ulong Count,
    double MinTimeMs,
    double MaxTimeMs,
    double AverageTimeMs,
    double TotalTimeMs,
    double ProcessProfilerTotalTimeMs,
    string FormattedReport);

file readonly record struct BackendInfo(
    string BackendType,
    uint MaxWorkGroupSizeX,
    uint MaxWorkGroupSizeY,
    uint MaxWorkGroupSizeZ);

file readonly record struct ProfilerCaseResult(
    string Name,
    string Category,
    string Workload,
    string DispatchPath,
    int WarmupIterations,
    int MeasuredIterations,
    double HarnessWallTimeMs,
    ProfilerSnapshot Profiler,
    IReadOnlyDictionary<string, object?> Metrics);

file readonly record struct ExternalProcessResult(
    string Name,
    string Category,
    string Workload,
    bool Success,
    int? ExitCode,
    double? WallTimeMs,
    string? OutputPath,
    IReadOnlyList<string> KeyLines);

file readonly record struct ProfilerReport(
    DateTimeOffset GeneratedAtUtc,
    BackendInfo Backend,
    ProfilerOptions Options,
    IReadOnlyList<ProfilerCaseResult> Cases,
    IReadOnlyList<ExternalProcessResult> ExternalProcesses);
