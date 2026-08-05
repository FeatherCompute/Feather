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
        Assert.NotEqual(sentinel, pixels[3 * size + 3]);
        Assert.InRange(pixels[3 * size + 3].X, -0.12501f, -0.12499f);
        Assert.InRange(pixels[3 * size + 3].Y, -0.12501f, -0.12499f);
        Assert.InRange(pixels[3 * size + 3].Z, 0.41665f, 0.41668f);
        Assert.InRange(pixels[3 * size + 3].W, 0.99999f, 1.00001f);
        Assert.True(pixels.Count(pixel => pixel != sentinel) >= 15);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }
}
