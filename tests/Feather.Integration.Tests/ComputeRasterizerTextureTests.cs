using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

namespace Feather.Integration.Tests;

[Collection(ComputeRasterizerCollection.Name)]
public class ComputeRasterizerTextureTests
{
    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerHonorsTextureFilterAndAddressModes()
    {
        using var vertices = GPU.CreateBuffer<float4>(FullTargetTriangle());
        using var source = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var clamp = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var repeat = GPU.CreateSampler(SamplerDesc.LinearRepeat);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantUvTextureFragmentShader, float4>();
        source.Upload(
        [
            new float4(0, 0, 0, 1), new float4(1, 0, 0, 1),
            new float4(0, 1, 0, 1), new float4(1, 1, 0, 1)
        ]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);

        pipeline.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                      new GeneratedConstantUvTextureFragmentShader(
                          source.AsSampled(), clamp, new Uniform<float2>(new float2(1.25f, 0.25f))),
                      target, vertexCount: 3);
        var clampPixels = new float4[16];
        target.Read(clampPixels);

        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        pipeline.Draw(new GeneratedVertexShader(vertices.AsReadOnly()),
                      new GeneratedConstantUvTextureFragmentShader(
                          source.AsSampled(), repeat, new Uniform<float2>(new float2(1.25f, 0.25f))),
                      target, vertexCount: 3);
        var repeatPixels = new float4[16];
        target.Read(repeatPixels);

        Assert.True(MathF.Abs(clampPixels[5].X - repeatPixels[5].X) > 0.2f ||
                    MathF.Abs(clampPixels[5].Y - repeatPixels[5].Y) > 0.2f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerEvaluatesVaryingDerivativesForSampleGrad()
    {
        using var vertices = GPU.CreateBuffer<GeneratedMeshVertex>(
        [
            MeshVertex(new float3(-1, -1, 0.25f), new float2(0.25f, 0.25f)),
            MeshVertex(new float3(3, -1, 0.25f), new float2(0.75f, 0.25f)),
            MeshVertex(new float3(-1, 3, 0.25f), new float2(0.25f, 0.75f))
        ], BufferAccess.ReadOnly);
        using var atlas = GPU.CreateTexture2D<Rgba32, float4>(2, 2, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedMeshVertexShader, GeneratedMeshFragmentShader, GeneratedMeshVaryings>();
        atlas.Upload([.. Enumerable.Repeat(new Rgba32(220, 40, 20, 255), 4)]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);

        pipeline.Draw(new GeneratedMeshVertexShader(vertices.AsReadOnly(), new Uniform<float4x4>(float4x4.Identity)),
                      new GeneratedMeshFragmentShader(atlas.AsSampled(), sampler), target, vertexCount: 3);

        var pixels = new float4[64];
        target.Read(pixels);
        Assert.Contains(pixels, pixel => pixel.X > pixel.Y && pixel.X > pixel.Z && pixel.W > 0.5f);
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    [ComputeRasterFact]
    [Trait("Category", "Gpu")]
    public void ComputeRasterizerUsesSharedVertexAndFragmentPushConstantLayout()
    {
        using var vertices = GPU.CreateBuffer<float4>(FullTargetTriangle());
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedCrossStageFloat3UniformVertexShader,
            GeneratedCrossStageFloat3UniformFragmentShader, GeneratedCrossStageFloat3Varyings>();
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);

        pipeline.Draw(
            new GeneratedCrossStageFloat3UniformVertexShader(
                vertices.AsReadOnly(), new Uniform<float>(0.01f),
                new Uniform<float3>(new float3(0.02f, 0.03f, 0.04f)), new Uniform<float>(0.05f)),
            new GeneratedCrossStageFloat3UniformFragmentShader(
                new Uniform<float>(0.06f), new Uniform<float3>(new float3(0.07f, 0.08f, 0.09f)),
                new Uniform<float3>(new float3(0.10f, 0.11f, 0.12f)), new Uniform<float>(0.13f)),
            target, vertexCount: 3);

        var pixels = new float4[16];
        target.Read(pixels);
        Assert.All(pixels, pixel => AssertClose(new float4(0.21f, 0.14f, 0.36f, 1.0f), pixel, 0.02f));
        Assert.Equal(DispatchPath.Luisa, pipeline.LastDispatchPath);
    }

    private static GeneratedMeshVertex MeshVertex(float3 position, float2 uv)
        => new()
        {
            Position = position,
            Normal = new float3(0, 0, 1),
            Uv = uv,
            AtlasTransform = new float4(0, 0, 1, 1)
        };

    private static float4[] FullTargetTriangle()
        =>
        [
            new float4(-1, -1, 0.5f, 1),
            new float4(3, -1, 0.5f, 1),
            new float4(-1, 3, 0.5f, 1)
        ];

    private static void AssertClose(float4 expected, float4 actual, float tolerance)
    {
        Assert.InRange(actual.X, expected.X - tolerance, expected.X + tolerance);
        Assert.InRange(actual.Y, expected.Y - tolerance, expected.Y + tolerance);
        Assert.InRange(actual.Z, expected.Z - tolerance, expected.Z + tolerance);
        Assert.InRange(actual.W, expected.W - tolerance, expected.W + tolerance);
    }
}
