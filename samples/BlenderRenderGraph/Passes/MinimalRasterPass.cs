using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;

namespace BlenderRenderGraph.Passes;

[FeatherPass(
    "01c671a1-9b4e-5cab-b7e1-c101348af596",
    Name = "Minimal Raster",
    Category = "Raster",
    Version = 1)]
public sealed class MinimalRasterPass : IRasterPass, IDisposable
{
    private GpuGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings>? pipeline;

    [Input("6d6eb2d5-bb7a-55a4-a85a-c58e36715c53")]
    public SceneGeometryHandle Geometry { get; init; }

    [Input("a6eed590-b632-5f91-a69d-09b6eb4bb5ac")]
    public MaterialTableHandle Materials { get; init; }

    [Input("cc78191c-ac9a-57b6-bcac-91cce5e298f5")]
    public CameraHandle Camera { get; init; }

    [Output(
        "bd711ea6-36f9-56cd-863a-cfec58727a46",
        Format = TextureFormat.Rgba8)]
    public TextureHandle Color { get; init; }

    [Parameter(
        "2a1df649-2b96-558f-ae30-9d9bf6858d43",
        Min = 0.0,
        Max = 8.0)]
    public float Exposure { get; set; } = 1.0f;

    public void Execute(RenderContext context)
    {
        var geometry = context.GetSceneGeometry(Geometry);
        var camera = context.GetCamera(Camera);
        using var color = GPU.CreateRenderTexture2D<Rgba8, Rgba8>(
            context.Width,
            context.Height,
            PixelFormat.Rgba8);
        color.Upload(CreateBackground(context.Width, context.Height));

        var dispatchPath = DispatchPath.None;
        if (!geometry.Indices.IsEmpty)
        {
            var shaderVertices = new MinimalRasterVertex[geometry.Vertices.Length];
            for (var index = 0; index < shaderVertices.Length; index++)
            {
                var vertex = geometry.Vertices.Span[index];
                shaderVertices[index] = new MinimalRasterVertex
                {
                    Position = vertex.Position,
                    Normal = vertex.Normal
                };
            }
            using var vertices = GPU.CreateBuffer<MinimalRasterVertex>(shaderVertices, BufferAccess.ReadOnly);
            using var indices = GPU.CreateIndexBuffer(geometry.Indices.Span);
            pipeline = GPU.CreateGraphicsPipeline<MinimalRasterVertexShader, MinimalRasterFragmentShader, MinimalRasterVaryings>(
                new GraphicsPipelineDesc
                {
                    SampleCount = context.SampleCount,
                    DepthStencil = DepthStencilState.Default with
                    {
                        DepthTest = true,
                        DepthWrite = true,
                        DepthCompare = CompareOp.Less
                    },
                    Raster = RasterState.Default with { CullMode = CullMode.None },
                    DebugName = "Project MinimalRaster"
                });
            using var depth = GPU.CreateDepthTexture2D(context.Width, context.Height);
            IGpuTexture2D[] targets = [color];
            pipeline.DrawIndexed(
                new MinimalRasterVertexShader(
                    vertices.AsReadOnly(),
                    new Uniform<float4x4>(camera.ViewProjection)),
                new MinimalRasterFragmentShader(
                    new Uniform<float3>(new float3(0.0f, 0.0f, 1.0f)),
                    new Uniform<float>(0.1f),
                    new Uniform<float>(Exposure),
                    new Uniform<float3>(new float3(1.0f, 1.0f, 1.0f))),
                targets,
                depth,
                indices,
                new GraphicsDrawDesc
                {
                    ColorLoadOp = GraphicsColorLoadOp.Clear,
                    ClearColor = new float4(0.01f, 0.02f, 0.03f, 1.0f),
                    DepthLoadOp = GraphicsDepthLoadOp.Clear,
                    ClearDepth = 1.0f
                });
            dispatchPath = pipeline.LastDispatchPath;
        }

        context.SetColorOutput(Color, color, dispatchPath);
    }

    public void Dispose()
    {
        pipeline?.Dispose();
        pipeline = null;
    }

    private static Rgba8[] CreateBackground(int width, int height)
    {
        var pixels = new Rgba8[checked(width * height)];
        Array.Fill(pixels, new Rgba8(3, 5, 8, 255));
        return pixels;
    }
}

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
    Uniform<float> ambient,
    Uniform<float> exposure,
    Uniform<float3> tint) : IFragmentShader<MinimalRasterVaryings>
{
    public float4 Execute(MinimalRasterVaryings input)
    {
        var normal = ShaderMath.Normalize(input.Normal);
        var diffuse = ShaderMath.Max(ShaderMath.Dot(normal, lightDirection.Value), 0.0f);
        var shade = ShaderMath.Min(
            (ambient.Value + ((1.0f - ambient.Value) * diffuse)) * exposure.Value,
            1.0f);
        return new float4(tint.Value * shade, 1.0f);
    }
}
