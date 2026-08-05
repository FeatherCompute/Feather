using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerBlendTests
{
    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerAppliesAlphaBlend()
    {
        var pixel = Draw(new float4(0.25f, 0.0f, 0.0f, 1.0f),
                         new float4(0.0f, 0.0f, 1.0f, 0.5f),
                         BlendState.AlphaBlend);

        AssertClose(new float4(0.125f, 0.0f, 0.5f, 1.0f), pixel);
    }

    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerUsesIndependentColorAndAlphaBlendOperations()
    {
        var blend = BlendState.Opaque with
        {
            Enabled = true,
            SrcColor = BlendFactor.One,
            DstColor = BlendFactor.One,
            ColorOp = BlendOp.ReverseSubtract,
            SrcAlpha = BlendFactor.One,
            DstAlpha = BlendFactor.One,
            AlphaOp = BlendOp.Subtract
        };
        var pixel = Draw(new float4(0.2f, 0.4f, 0.6f, 0.8f),
                         new float4(0.7f, 0.5f, 0.3f, 0.25f), blend);

        AssertClose(new float4(-0.5f, -0.1f, 0.3f, -0.55f), pixel);
    }

    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerPreservesMaskedColorChannels()
    {
        var blend = BlendState.Opaque with { WriteMask = ColorWriteMask.Red | ColorWriteMask.Alpha };
        var pixel = Draw(new float4(0.1f, 0.2f, 0.3f, 0.4f),
                         new float4(0.9f, 0.8f, 0.7f, 0.6f), blend);

        AssertClose(new float4(0.9f, 0.2f, 0.3f, 0.6f), pixel);
    }

    private static float4 Draw(float4 destination, float4 source, BlendState blend)
    {
        const int size = 4;
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1.0f, -1.0f, 0.5f, 1.0f),
            new float4(3.0f, -1.0f, 0.5f, 1.0f),
            new float4(-1.0f, 3.0f, 0.5f, 1.0f)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(size, size, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { Blend = blend });
        target.Upload([.. Enumerable.Repeat(destination, size * size)]);

        pipeline.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                      new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(source)),
                      target, vertexCount: 3);

        var pixels = new float4[size * size];
        target.Read(pixels);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
        return pixels[5];
    }

    private static void AssertClose(float4 expected, float4 actual)
    {
        Assert.True(MathF.Abs(actual.X - expected.X) <= 0.00001f &&
                    MathF.Abs(actual.Y - expected.Y) <= 0.00001f &&
                    MathF.Abs(actual.Z - expected.Z) <= 0.00001f &&
                    MathF.Abs(actual.W - expected.W) <= 0.00001f,
                    $"Expected {expected}, got {actual}.");
    }
}
