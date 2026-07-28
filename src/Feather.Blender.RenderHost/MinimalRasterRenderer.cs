using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Blender.RenderHost;

internal sealed class MinimalRasterRenderer : IDisposable
{
    private GpuGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings>? pipeline;
    private SampleCount pipelineSampleCount;

    public RenderedFrame Render(
        RenderGeometry geometry,
        int width,
        int height,
        float4x4 viewProjection,
        SampleCount sampleCount,
        MinimalRasterSettings settings)
    {
        var pixels = CreateBackground(width, height, settings.ClearColor);
        using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8);
        color.Upload(pixels);

        var dispatchPath = DispatchPath.None;
        if (geometry.Indices.Length > 0)
        {
            if (geometry.Vertices.Length == 0 || geometry.Indices.Length % 3 != 0)
            {
                throw new InvalidDataException("Render geometry must contain complete indexed triangles.");
            }

            using var depth = GPU.CreateDepthTexture2D(width, height);
            using var vertexBuffer = GPU.CreateBuffer<MinimalRasterVertex>(geometry.Vertices, BufferAccess.ReadOnly);
            using var indexBuffer = GPU.CreateIndexBuffer<uint>(geometry.Indices);
            if (pipeline is null || pipelineSampleCount != sampleCount)
            {
                var replacement = GPU.CreateGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings>(
                new GraphicsPipelineDesc
                {
                    SampleCount = sampleCount,
                    DepthStencil = DepthStencilState.Default with
                    {
                        DepthTest = true,
                        DepthWrite = true,
                        DepthCompare = CompareOp.Less
                    },
                    Raster = RasterState.Default with { CullMode = CullMode.None },
                    DebugName = "Feather Blender MinimalRaster"
                });
                var previous = pipeline;
                pipeline = replacement;
                pipelineSampleCount = sampleCount;
                previous?.Dispose();
            }

            IGpuTexture2D[] targets = [color];
            pipeline.DrawIndexed(
                new MinimalRasterVertexShader(
                    vertexBuffer.AsReadOnly(),
                    new Uniform<float4x4>(viewProjection)),
                new MinimalRasterFragmentShader(
                    new Uniform<float3>(settings.LightDirection),
                    new Uniform<float>(settings.Ambient)),
                targets,
                depth,
                indexBuffer,
                new GraphicsDrawDesc
                {
                    ColorLoadOp = GraphicsColorLoadOp.Clear,
                    ClearColor = settings.ClearColor,
                    DepthLoadOp = GraphicsDepthLoadOp.Clear,
                    ClearDepth = 1.0f
                });
            dispatchPath = pipeline.LastDispatchPath;
        }

        color.Read(pixels);
        return new RenderedFrame(width, height, pixels, dispatchPath);
    }

    public void Dispose()
    {
        pipeline?.Dispose();
        pipeline = null;
    }

    private static Rgba32[] CreateBackground(int width, int height, float4 color)
    {
        var pixel = new Rgba32(
            ToUnorm(color.X),
            ToUnorm(color.Y),
            ToUnorm(color.Z),
            ToUnorm(color.W));
        return Enumerable.Repeat(pixel, checked(width * height)).ToArray();
    }

    private static byte ToUnorm(float value)
        => (byte)System.Math.Clamp(MathF.Round(value * 255.0f), 0.0f, 255.0f);
}

internal sealed record RenderedFrame(
    int Width,
    int Height,
    Rgba32[] Pixels,
    DispatchPath DispatchPath);

internal readonly record struct Rgba32(byte R, byte G, byte B, byte A);

[GpuStruct]
public partial struct MinimalRasterVertex
{
    public float3 Position;
    public float3 Normal;
}

[GpuStruct]
public partial struct MinimalRasterVaryings
{
    [Position]
    public float4 Position;
    public float3 Normal;
}

[VertexShader]
public readonly partial struct MinimalRasterVertexShader(
    ReadOnlyBuffer<MinimalRasterVertex> vertices,
    Uniform<float4x4> viewProjection) : IVertexShader<MinimalRasterVaryings>
{
    public MinimalRasterVaryings Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new MinimalRasterVaryings
        {
            Position = ShaderMath.Mul(viewProjection.Value, new float4(vertex.Position, 1.0f)),
            Normal = vertex.Normal
        };
    }
}

[FragmentShader]
public readonly partial struct MinimalRasterFragmentShader(
    Uniform<float3> lightDirection,
    Uniform<float> ambient) : IFragmentShader<MinimalRasterVaryings>
{
    public float4 Execute(MinimalRasterVaryings input)
    {
        var normal = ShaderMath.Normalize(input.Normal);
        var diffuse = ShaderMath.Max(ShaderMath.Dot(normal, lightDirection.Value), 0.0f);
        var shade = ambient.Value + ((1.0f - ambient.Value) * diffuse);
        return new float4(shade, shade, shade, 1.0f);
    }
}
