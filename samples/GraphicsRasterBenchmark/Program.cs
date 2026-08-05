using System.Diagnostics;
using Feather;
using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

const int width = 512;
const int height = 512;
const int iterations = 5;

using var vertices = GPU.CreateBuffer<float4>(
[
    new float4(-0.9f, -0.9f, 0.5f, 1.0f),
    new float4(0.9f, -0.9f, 0.5f, 1.0f),
    new float4(0.0f, 0.9f, 0.5f, 1.0f)
], BufferAccess.ReadOnly);
using var target = GPU.CreateRenderTexture2D<float4, float4>(width, height, PixelFormat.Rgba32Float);
using var pipeline = GPU.CreateGraphicsPipeline<BenchmarkVertexShader, BenchmarkFragmentShader, float4>();
target.Upload(new float4[width * height]);

pipeline.Draw(new BenchmarkVertexShader(vertices.AsReadOnly()), new BenchmarkFragmentShader(),
              target, vertexCount: 3, wait: true);

var timings = new double[iterations];
for (var i = 0; i < iterations; ++i)
{
    var stopwatch = Stopwatch.StartNew();
    pipeline.Draw(new BenchmarkVertexShader(vertices.AsReadOnly()), new BenchmarkFragmentShader(),
                  target, vertexCount: 3, wait: true);
    stopwatch.Stop();
    timings[i] = stopwatch.Elapsed.TotalMilliseconds;
    Console.WriteLine($"iteration={i + 1} elapsed_ms={timings[i]:F3}");
}

var pixels = new float4[width * height];
target.Read(pixels);
var visible = pixels.Count(static pixel => pixel.W > 0.9f && pixel.Z > 0.7f);
if (visible < width * height / 4)
{
    throw new InvalidOperationException($"Raster validation failed: visible={visible}.");
}

Array.Sort(timings);
Console.WriteLine($"path={pipeline.LastDispatchPath}");
Console.WriteLine($"size={width}x{height} warmup=1 iterations={iterations} visible_pixels={visible}");
Console.WriteLine($"median_ms={timings[iterations / 2]:F3}");

[VertexShader]
public readonly partial struct BenchmarkVertexShader(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute() => vertices[VertexIds.Index];
}

[FragmentShader]
public readonly partial struct BenchmarkFragmentShader : IFragmentShader<float4>
{
    public float4 Execute(float4 input) => new(0.2f + input.X * 0.1f, 0.4f, 0.8f, 1.0f);
}
