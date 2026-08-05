using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ComputeRasterizerCollection
{
    public const string Name = "ComputeRasterizer";
}

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerTests
{
    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerCoversTriangleAndInterpolatesVertexValues()
    {
        const int size = 8;
        var sentinel = new float4(9.0f, 8.0f, 7.0f, 6.0f);
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.75f, -0.75f, 0.0f, 1.0f),
            new float4(0.75f, -0.75f, 0.0f, 1.0f),
            new float4(-0.75f, 0.75f, 1.0f, 1.0f)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>();
        target.Upload([.. Enumerable.Repeat(sentinel, size * size)]);

        var previous = Environment.GetEnvironmentVariable("FEATHER_GRAPHICS_COMPUTE");
        try
        {
            Environment.SetEnvironmentVariable("FEATHER_GRAPHICS_COMPUTE", "1");
            pipeline.Draw(
                new GeneratedVertexShader(vertices.AsReadOnly()),
                new GeneratedFragmentShader(sampler),
                target,
                vertexCount: 3);
        }
        finally
        {
            Environment.SetEnvironmentVariable("FEATHER_GRAPHICS_COMPUTE", previous);
        }

        var pixels = new float4[size * size];
        target.Read(pixels);

        Assert.Equal(sentinel, pixels[7 * size + 7]);
        Assert.NotEqual(sentinel, pixels[3 * size + 2]);
        Assert.InRange(pixels[3 * size + 2].X, -0.37501f, -0.37499f);
        Assert.InRange(pixels[3 * size + 2].Y, 0.12499f, 0.12501f);
        Assert.InRange(pixels[3 * size + 2].Z, 0.58332f, 0.58335f);
        Assert.InRange(pixels[3 * size + 2].W, 0.99999f, 1.00001f);
        Assert.True(pixels.Count(pixel => pixel != sentinel) >= 15);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerLoadsAndTestsDepthAcrossDraws()
    {
        const int size = 8;
        using var nearVertices = GPU.CreateBuffer<float4>(FullTargetTriangle(0.25f));
        using var farVertices = GPU.CreateBuffer<float4>(FullTargetTriangle(0.75f));
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var depth = GPU.CreateDepthTexture2D(size, size);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    DepthTest = true,
                    DepthWrite = true,
                    DepthCompare = CompareOp.Less
                }
            });
        target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);
        depth.Upload([.. Enumerable.Repeat(1.0f, size * size)]);

        pipeline.Draw(
            new GeneratedVertexShader(nearVertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ClearDepth = 1.0f });
        pipeline.Draw(
            new GeneratedVertexShader(farVertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load });

        var colorPixels = new float4[size * size];
        var depthPixels = new float[size * size];
        target.Read(colorPixels);
        depth.Read(depthPixels);
        Assert.InRange(colorPixels[4 * size + 4].Z, 0.24999f, 0.25001f);
        Assert.InRange(depthPixels[4 * size + 4], 0.24999f, 0.25001f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerAppliesViewportScissorAndCull()
    {
        const int size = 8;
        var sentinel = new float4(9.0f, 8.0f, 7.0f, 6.0f);
        using var vertices = GPU.CreateBuffer<float4>(FullTargetTriangle(0.5f));
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(sentinel, size * size)]);

        using (var culled = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
                   new GraphicsPipelineDesc
                   {
                       Raster = RasterState.Default with
                       {
                           CullMode = CullMode.Back,
                           FrontFace = FrontFace.CounterClockwise
                       }
                   }))
        {
            culled.Draw(
                new GeneratedVertexShader(vertices.AsReadOnly()),
                new GeneratedFragmentShader(sampler),
                target,
                vertexCount: 3);
            var culledPixels = new float4[size * size];
            target.Read(culledPixels);
            Assert.All(culledPixels, pixel => Assert.Equal(sentinel, pixel));
        }

        using var visible = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                Raster = RasterState.Default with
                {
                    CullMode = CullMode.Back,
                    FrontFace = FrontFace.Clockwise
                }
            });
        visible.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                Viewport = new GraphicsRect(2, 2, 4, 4),
                Scissor = new GraphicsRect(3, 3, 2, 2)
            });

        var pixels = new float4[size * size];
        target.Read(pixels);
        Assert.Equal(4, pixels.Count(pixel => pixel != sentinel));
        Assert.Equal(sentinel, pixels[2 * size + 2]);
        Assert.NotEqual(sentinel, pixels[3 * size + 3]);
        Assert.Equal(DispatchPath.Luisa, visible.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerExecutesVertexFeirBeforeRasterization()
    {
        const int size = 8;
        var sentinel = new float4(9.0f, 8.0f, 7.0f, 6.0f);
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.75f, -0.75f, 0.0f, 1.0f),
            new float4(0.75f, -0.75f, 0.0f, 1.0f),
            new float4(-0.75f, 0.75f, 1.0f, 1.0f)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var pipeline = GPU.CreateGraphicsPipeline<ComputeRasterOffsetVertexShader, ComputeRasterIdentityFragmentShader, float4>();
        target.Upload([.. Enumerable.Repeat(sentinel, size * size)]);

        pipeline.Draw(
            new ComputeRasterOffsetVertexShader(vertices.AsReadOnly(), new Uniform<float2>(new float2(0.5f, 0.0f))),
            new ComputeRasterIdentityFragmentShader(),
            target,
            vertexCount: 3);

        var pixels = new float4[size * size];
        target.Read(pixels);
        var shiftedOnlyPixel = pixels[3 * size + 4];
        Assert.NotEqual(sentinel, shiftedOnlyPixel);
        Assert.InRange(shiftedOnlyPixel.X, 0.12499f, 0.12501f);
        Assert.InRange(shiftedOnlyPixel.Y, 0.12499f, 0.12501f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerExecutesFragmentFeirForCoveredPixels()
    {
        const int size = 8;
        var sentinel = new float4(9.0f, 8.0f, 7.0f, 6.0f);
        var fragmentColor = new float4(0.2f, 0.4f, 0.6f, 0.8f);
        using var vertices = GPU.CreateBuffer<float4>(FullTargetTriangle(0.5f));
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>();
        target.Upload([.. Enumerable.Repeat(sentinel, size * size)]);

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(fragmentColor)),
            target,
            vertexCount: 3);

        var pixels = new float4[size * size];
        target.Read(pixels);
        Assert.Equal(fragmentColor, pixels[4 * size + 4]);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerSamplesTextureInFragmentFeir()
    {
        const int size = 8;
        var sourceColor = new float4(0.2f, 0.7f, 0.4f, 1.0f);
        using var vertices = GPU.CreateBuffer<float4>(FullTargetTriangle(0.5f));
        using var source = GPU.CreateTexture2D<float4, float4>(
            1, 1, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedTextureFragmentShader, float4>();
        source.Upload([sourceColor]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedTextureFragmentShader(source.AsSampled(), sampler),
            target,
            vertexCount: 3);

        var pixels = new float4[size * size];
        target.Read(pixels);
        var sampled = pixels[4 * size + 4];
        Assert.InRange(sampled.X, 0.69999f, 0.70001f);
        Assert.InRange(sampled.Y, 0.19999f, 0.20001f);
        Assert.InRange(sampled.Z, 0.39999f, 0.40001f);
        Assert.InRange(sampled.W, 0.99999f, 1.00001f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerInterpolatesStructuredVaryings()
    {
        const int size = 8;
        var expected = new float4(0.15f, 0.35f, 0.65f, 1.0f);
        using var positions = GPU.CreateBuffer<float4>(FullTargetTriangle(0.5f));
        using var colors = GPU.CreateBuffer<float4>([expected, expected, expected]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var pipeline = GPU.CreateGraphicsPipeline<ComputeRasterVaryingVertexShader, ComputeRasterVaryingFragmentShader, ComputeRasterVaryings>();
        target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);

        pipeline.Draw(
            new ComputeRasterVaryingVertexShader(positions.AsReadOnly(), colors.AsReadOnly()),
            new ComputeRasterVaryingFragmentShader(),
            target,
            vertexCount: 3);

        var pixels = new float4[size * size];
        target.Read(pixels);
        var interpolated = pixels[4 * size + 4];
        Assert.InRange(interpolated.X, expected.X - 0.00001f, expected.X + 0.00001f);
        Assert.InRange(interpolated.Y, expected.Y - 0.00001f, expected.Y + 0.00001f);
        Assert.InRange(interpolated.Z, expected.Z - 0.00001f, expected.Z + 0.00001f);
        Assert.InRange(interpolated.W, expected.W - 0.00001f, expected.W + 0.00001f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerAssemblesMultipleTriangles()
    {
        const int size = 8;
        var left = new float4(1.0f, 0.0f, 0.0f, 1.0f);
        var right = new float4(0.0f, 1.0f, 0.0f, 1.0f);
        using var positions = GPU.CreateBuffer<float4>(
        [
            new float4(-0.9f, -0.8f, 0.5f, 1.0f),
            new float4(-0.1f, -0.8f, 0.5f, 1.0f),
            new float4(-0.5f, 0.8f, 0.5f, 1.0f),
            new float4(0.1f, -0.8f, 0.5f, 1.0f),
            new float4(0.9f, -0.8f, 0.5f, 1.0f),
            new float4(0.5f, 0.8f, 0.5f, 1.0f)
        ]);
        using var colors = GPU.CreateBuffer<float4>([left, left, left, right, right, right]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var pipeline = GPU.CreateGraphicsPipeline<ComputeRasterVaryingVertexShader, ComputeRasterVaryingFragmentShader, ComputeRasterVaryings>();
        target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);

        pipeline.Draw(
            new ComputeRasterVaryingVertexShader(positions.AsReadOnly(), colors.AsReadOnly()),
            new ComputeRasterVaryingFragmentShader(),
            target,
            vertexCount: 6);

        var pixels = new float4[size * size];
        target.Read(pixels);
        Assert.Equal(left, pixels[4 * size + 2]);
        Assert.Equal(right, pixels[4 * size + 5]);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    private static float4[] FullTargetTriangle(float depth)
        =>
        [
            new float4(-1.0f, -1.0f, depth, 1.0f),
            new float4(3.0f, -1.0f, depth, 1.0f),
            new float4(-1.0f, 3.0f, depth, 1.0f)
        ];
}

[VertexShader]
public readonly partial struct ComputeRasterOffsetVertexShader(
    ReadOnlyBuffer<float4> vertices,
    Uniform<float2> offset) : IVertexShader<float4>
{
    public float4 Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new float4(vertex.X + offset.Value.X, vertex.Y + offset.Value.Y, vertex.Z, vertex.W);
    }
}

[FragmentShader]
public readonly partial struct ComputeRasterIdentityFragmentShader : IFragmentShader<float4>
{
    public float4 Execute(float4 input) => input;
}

[GpuStruct]
public partial struct ComputeRasterVaryings
{
    [Position]
    public float4 Position;

    public float4 Color;
}

[VertexShader]
public readonly partial struct ComputeRasterVaryingVertexShader(
    ReadOnlyBuffer<float4> positions,
    ReadOnlyBuffer<float4> colors) : IVertexShader<ComputeRasterVaryings>
{
    public ComputeRasterVaryings Execute()
    {
        return new ComputeRasterVaryings
        {
            Position = positions[VertexIds.Index],
            Color = colors[VertexIds.Index]
        };
    }
}

[FragmentShader]
public readonly partial struct ComputeRasterVaryingFragmentShader : IFragmentShader<ComputeRasterVaryings>
{
    public float4 Execute(ComputeRasterVaryings input) => input.Color;
}
