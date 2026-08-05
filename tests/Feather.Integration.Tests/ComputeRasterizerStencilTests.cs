using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerStencilTests
{
    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerAppliesStencilOperationsAndMasksAcrossDraws()
    {
        const int size = 4;
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.5f, 1),
            new float4(3, -1, 0.5f, 1),
            new float4(-1, 3, 0.5f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var depthStencil = GPU.CreateDepthStencilTexture2D(size, size);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var writer = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    StencilTest = true,
                    Front = StencilFaceState.KeepAlways with { PassOp = StencilOp.Replace },
                    Back = StencilFaceState.KeepAlways with { PassOp = StencilOp.Replace },
                    StencilWriteMask = 0x0f,
                    StencilReference = 0xa5
                }
            });
        using var tester = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    StencilTest = true,
                    Front = StencilFaceState.KeepAlways with { Compare = CompareOp.Equal },
                    Back = StencilFaceState.KeepAlways with { Compare = CompareOp.Equal },
                    StencilReadMask = 0x0f,
                    StencilReference = 0x15
                }
            });
        target.Upload([.. Enumerable.Repeat(float4.Zero, size * size)]);

        writer.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                    new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(float4.Zero)),
                    target, depthStencil, vertexCount: 3,
                    drawDesc: new GraphicsDrawDesc { ClearDepth = 1.0f });
        tester.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                    new GeneratedConstantColorFragmentShader(
                        sampler, new Uniform<float4>(new float4(0, 1, 1, 1))),
                    target, depthStencil, vertexCount: 3,
                    drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load });

        var pixels = new float4[size * size];
        target.Read(pixels);
        Assert.All(pixels, pixel => Assert.Equal(new float4(0, 1, 1, 1), pixel));
        Assert.Equal(DispatchPath.Luisa, tester.LastDispatchPath);
    }
}
