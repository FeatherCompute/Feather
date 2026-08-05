using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerMrtTests
{
    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerWritesAndMasksMultipleColorTargets()
    {
        const int size = 4;
        var destination = new float4(0.25f, 0.25f, 0.25f, 0.5f);
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1.0f, -1.0f, 0.5f, 1.0f),
            new float4(3.0f, -1.0f, 0.5f, 1.0f),
            new float4(-1.0f, 3.0f, 0.5f, 1.0f)
        ]);
        using var target0 = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var target1 = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMrtFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                ColorAttachmentCount = 2,
                BlendAttachments =
                [
                    BlendState.Opaque with { WriteMask = ColorWriteMask.Red | ColorWriteMask.Alpha },
                    BlendState.Opaque with { WriteMask = ColorWriteMask.Green }
                ]
            });
        target0.Upload([.. Enumerable.Repeat(destination, size * size)]);
        target1.Upload([.. Enumerable.Repeat(destination, size * size)]);

        IGpuTexture2D[] targets = [target0, target1];
        pipeline.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                      new GeneratedMrtFragmentShader(sampler), targets, vertexCount: 3);

        var pixels0 = new float4[size * size];
        var pixels1 = new float4[size * size];
        target0.Read(pixels0);
        target1.Read(pixels1);
        Assert.Equal(new float4(1.0f, 0.25f, 0.25f, 1.0f), pixels0[5]);
        Assert.Equal(new float4(0.25f, 1.0f, 0.25f, 0.5f), pixels1[5]);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }
}
