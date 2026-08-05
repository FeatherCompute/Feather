using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerPolygonModeTests
{
    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerDistinguishesFillLineAndPointCoverage()
    {
        const int size = 16;
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.8f, -0.8f, 0.5f, 1),
            new float4(0.8f, -0.8f, 0.5f, 1),
            new float4(0.0f, 0.8f, 0.5f, 1)
        ]);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        var fill = Draw(PolygonMode.Fill);
        var line = Draw(PolygonMode.Line);
        var point = Draw(PolygonMode.Point);

        Assert.True(fill > line, $"Expected fill coverage ({fill}) to exceed line coverage ({line}).");
        Assert.True(line > point, $"Expected line coverage ({line}) to exceed point coverage ({point}).");
        Assert.InRange(point, 1, 12);

        int Draw(PolygonMode mode)
        {
            using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
            using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
                new GraphicsPipelineDesc { Raster = RasterState.Default with { PolygonMode = mode } });
            target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);
            pipeline.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                          new GeneratedConstantColorFragmentShader(
                              sampler, new Uniform<float4>(new float4(1, 1, 0, 1))),
                          target, vertexCount: 3);
            var pixels = new float4[size * size];
            target.Read(pixels);
            Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
            return pixels.Count(pixel => pixel.X > 0.5f && pixel.Y > 0.5f);
        }
    }
}
