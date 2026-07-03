using Feather.Graphics;
using Feather.Interop;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

public class GeneratedGraphicsPipelineTests
{
    [Fact]
    public void GeneratedGraphicsInspectionReportsSeparateStageIr()
    {
        var source = ShaderInspection.GetGraphicsSource<GeneratedVertexShader, GeneratedTextureFragmentShader, float4>();

        Assert.NotEmpty(source.IR);
        Assert.NotEmpty(source.VertexIR);
        Assert.NotEmpty(source.FragmentIR);
        Assert.Equal(source.IR, source.VertexIR);
        Assert.NotEqual(source.VertexIR, source.FragmentIR);
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawRasterizesTarget()
    {
        using var vertices = GPU.CreateBuffer<float4>([new float4(0, 0, 0, 1), new float4(1, 0, 0, 1), new float4(0, 1, 0, 1)]);
        using var target = GPU.CreateTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new Rgba32(10, 20, 30, 40), 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            vertexCount: 3);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(10, 20, 30, 40));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineBindsUniformPushConstants()
    {
        using var vertices = GPU.CreateBuffer<float4>([new float4(0, 0, 0, 1), new float4(1, 0, 0, 1), new float4(0, 1, 0, 1)]);
        using var target = GPU.CreateTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new Rgba32(1, 2, 3, 4), 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedUniformVertexShader, GeneratedUniformFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedUniformVertexShader(vertices.AsReadOnly(), new Uniform<float4>(new float4(1, 2, 3, 4))),
            new GeneratedUniformFragmentShader(sampler, new Uniform<float4>(new float4(5, 6, 7, 8))),
            target,
            vertexCount: 3);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(1, 2, 3, 4));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineAlignsMixedFloat3UniformPushConstants()
    {
        var descriptor = GetGraphicsDescriptor<GeneratedVertexShader, GeneratedMixedFloat3UniformFragmentShader, float4>();
        var pushConstants = descriptor.PushConstants.ToDictionary(static pushConstant => pushConstant.Name);

        Assert.Equal(8, pushConstants.Count);
        Assert.Equal(0u, pushConstants["aspect"].Offset);
        Assert.Equal(4u, pushConstants["roughness"].Offset);
        Assert.Equal(16u, pushConstants["lightP0"].Offset);
        Assert.Equal(32u, pushConstants["lightP1"].Offset);
        Assert.Equal(48u, pushConstants["lightP2"].Offset);
        Assert.Equal(64u, pushConstants["lightP3"].Offset);
        Assert.Equal(76u, pushConstants["intensity"].Offset);
        Assert.Equal(80u, pushConstants["twoSided"].Offset);
        Assert.Equal(12u, pushConstants["lightP0"].Size);
        Assert.Equal(12u, pushConstants["lightP1"].Size);
        Assert.Equal(12u, pushConstants["lightP2"].Size);
        Assert.Equal(12u, pushConstants["lightP3"].Size);

        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMixedFloat3UniformFragmentShader, float4>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "MixedFloat3UniformPushConstants" });

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedMixedFloat3UniformFragmentShader(
                new Uniform<float>(0.10f),
                new Uniform<float>(0.20f),
                new Uniform<float3>(new float3(0.01f, 0.02f, 0.03f)),
                new Uniform<float3>(new float3(0.04f, 0.05f, 0.06f)),
                new Uniform<float3>(new float3(0.07f, 0.08f, 0.09f)),
                new Uniform<float3>(new float3(0.11f, 0.12f, 0.13f)),
                new Uniform<float>(0.14f),
                new Uniform<int>(1)),
            target,
            vertexCount: 3);

        var readback = new float4[16];
        target.Read(readback);
        var expected = new float4(0.15f, 0.21f, 0.34f, 1.0f);
        Assert.All(readback, pixel => AssertColorNear(pixel, expected, 0.02f));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelinePreservesAlignedFloat3UniformsAcrossVertexAndFragmentStages()
    {
        var descriptor = GetGraphicsDescriptor<GeneratedCrossStageFloat3UniformVertexShader, GeneratedCrossStageFloat3UniformFragmentShader, GeneratedCrossStageFloat3Varyings>();
        var pushConstants = descriptor.PushConstants.ToDictionary(static pushConstant => pushConstant.Name);

        Assert.Equal(7, pushConstants.Count);
        Assert.Equal(0u, pushConstants["fragmentScalar0"].Offset);
        Assert.Equal(4u, pushConstants["vertexScalar0"].Offset);
        Assert.Equal(16u, pushConstants["fragmentVector0"].Offset);
        Assert.Equal(32u, pushConstants["vertexVector"].Offset);
        Assert.Equal(48u, pushConstants["fragmentVector1"].Offset);
        Assert.Equal(60u, pushConstants["vertexScalar1"].Offset);
        Assert.Equal(64u, pushConstants["fragmentScalar1"].Offset);
        Assert.Equal(12u, pushConstants["fragmentVector0"].Size);
        Assert.Equal(12u, pushConstants["vertexVector"].Size);
        Assert.Equal(12u, pushConstants["fragmentVector1"].Size);

        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedCrossStageFloat3UniformVertexShader, GeneratedCrossStageFloat3UniformFragmentShader, GeneratedCrossStageFloat3Varyings>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "CrossStageFloat3UniformPushConstants" });

        pipeline.Draw(
            new GeneratedCrossStageFloat3UniformVertexShader(
                vertices.AsReadOnly(),
                new Uniform<float>(0.01f),
                new Uniform<float3>(new float3(0.02f, 0.03f, 0.04f)),
                new Uniform<float>(0.05f)),
            new GeneratedCrossStageFloat3UniformFragmentShader(
                new Uniform<float>(0.06f),
                new Uniform<float3>(new float3(0.07f, 0.08f, 0.09f)),
                new Uniform<float3>(new float3(0.10f, 0.11f, 0.12f)),
                new Uniform<float>(0.13f)),
            target,
            vertexCount: 3);

        var readback = new float4[16];
        target.Read(readback);
        var expected = new float4(0.21f, 0.14f, 0.36f, 1.0f);
        Assert.All(readback, pixel => AssertColorNear(pixel, expected, 0.02f));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineBindsSquareMatrixUniformPushConstants()
    {
        var descriptor = GetGraphicsDescriptor<GeneratedMatrixUniformVertexShader, GeneratedMatrixUniformFragmentShader, GeneratedMatrixUniformVaryings>();
        var pushConstants = descriptor.PushConstants.ToDictionary(static pushConstant => pushConstant.Name);

        Assert.Equal(5, pushConstants.Count);
        Assert.Equal(16u, pushConstants["vertexMatrix2"].Size);
        Assert.Equal(64u, pushConstants["vertexMatrix4"].Size);
        Assert.Equal(16u, pushConstants["fragmentMatrix2"].Size);
        Assert.Equal(48u, pushConstants["fragmentMatrix3"].Size);
        Assert.Equal(4u, pushConstants["scalar"].Size);
        Assert.Equal(0u, pushConstants["vertexMatrix2"].Offset % 8u);
        Assert.Equal(0u, pushConstants["vertexMatrix4"].Offset % 16u);
        Assert.Equal(0u, pushConstants["fragmentMatrix2"].Offset % 8u);
        Assert.Equal(0u, pushConstants["fragmentMatrix3"].Offset % 16u);

        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedMatrixUniformVertexShader, GeneratedMatrixUniformFragmentShader, GeneratedMatrixUniformVaryings>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "SquareMatrixUniformPushConstants" });

        pipeline.Draw(
            new GeneratedMatrixUniformVertexShader(
                vertices.AsReadOnly(),
                new Uniform<float2x2>(new float2x2(new float2(0.01f, 0.02f), new float2(0.03f, 0.04f))),
                new Uniform<float4x4>(float4x4.Identity)),
            new GeneratedMatrixUniformFragmentShader(
                new Uniform<float2x2>(new float2x2(new float2(0.1f, 0.2f), new float2(0.3f, 0.4f))),
                new Uniform<float3x3>(new float3x3(
                    new float3(0.01f, 0.02f, 0.03f),
                    new float3(0.04f, 0.05f, 0.06f),
                    new float3(0.07f, 0.08f, 0.09f))),
                new Uniform<float>(0.2f)),
            target,
            vertexCount: 3);

        var readback = new float4[16];
        target.Read(readback);
        var expected = new float4(0.7747f, 1.0864f, 0.2981f, 1.0f);
        Assert.All(readback, pixel => AssertColorNear(pixel, expected, 0.02f));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineUsesEntryAttributedShaders()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<EntryAttributedVertexShader, EntryAttributedFragmentShader, float4>();

        pipeline.Draw(
            new EntryAttributedVertexShader(vertices.AsReadOnly()),
            new EntryAttributedFragmentShader(sampler),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X > 0.8f && pixel.Y is > 0.2f and < 0.4f && pixel.Z is > 0.4f and < 0.6f && pixel.W > 0.9f);
        Assert.DoesNotContain(readback, pixel => pixel.X < 0.1f && pixel.Y > 0.8f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedFragmentShaderUniformExpressionAffectsEasyGpuOutput()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new Rgba32(0, 0, 0, 255), 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedUniformVertexShader, GeneratedSwizzledUniformFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedUniformVertexShader(vertices.AsReadOnly(), new Uniform<float4>(new float4(9, 9, 9, 9))),
            new GeneratedSwizzledUniformFragmentShader(sampler, new Uniform<float4>(new float4(0.1f, 0.8f, 0.3f, 1.0f))),
            target,
            vertexCount: 3);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.R > 190 && pixel.G < 40 && pixel.B is > 60 and < 100 && pixel.A == 255);
        Assert.DoesNotContain(readback, pixel => pixel.R is > 20 and < 40 && pixel.G > 190 && pixel.B is > 60 and < 100 && pixel.A == 255);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void ProfilerRecordsGeneratedGraphicsDraw()
    {
        try
        {
            GpuProfiler.SetEnabled(true);
            GpuProfiler.Clear();

            using var vertices = GPU.CreateBuffer<float4>([new float4(0, 0, 0, 1), new float4(1, 0, 0, 1), new float4(0, 1, 0, 1)]);
            using var target = GPU.CreateTexture2D<Rgba32, Rgba32>(1, 1, PixelFormat.Rgba8, TextureAccess.RenderTarget);
            using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
            using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
                new GraphicsPipelineDesc { DebugName = "ProfiledGraphicsDraw" });

            pipeline.Draw(
                new GeneratedVertexShader(vertices.AsReadOnly()),
                new GeneratedFragmentShader(sampler),
                target,
                vertexCount: 3);

            var query = GpuProfiler.Query("ProfiledGraphicsDraw");
            Assert.Equal(1UL, query.Count);
            Assert.True(query.TotalTimeMs >= 0.0);
        }
        finally
        {
            GpuProfiler.Clear();
            GpuProfiler.SetEnabled(false);
        }
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawsToRgba32FloatTargetThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new float4(0, 0, 0, 0), 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.W > 0.5f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawsWithMsaaAndResolvesToColorTarget()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(16, 16, PixelFormat.Rgba8);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new Rgba32(0, 0, 0, 255), 256)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "GeneratedMsaaResolve" });

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            vertexCount: 3);

        var readback = new Rgba32[256];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.R > 0 || pixel.G > 0 || pixel.B > 0);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawsWithMsaaDepthTargetUsingDefaultDepthClear()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.5f, 1),
            new float4(1, -1, 0.5f, 1),
            new float4(0, 1, 0.5f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(16, 16, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(16, 16);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new Rgba32(0, 0, 0, 255), 256)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                SampleCount = SampleCount.X4,
                DepthStencil = DepthStencilState.Default with { DepthTest = true, DepthWrite = true },
                DebugName = "GeneratedMsaaDepthDefaultClear"
            });

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            depth,
            vertexCount: 3);

        var readback = new Rgba32[256];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.R > 0 || pixel.G > 0 || pixel.B > 0);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineRejectsExplicitMsaaDepthLoad()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.5f, 1),
            new float4(1, -1, 0.5f, 1),
            new float4(0, 1, 0.5f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(16, 16, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(16, 16);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                SampleCount = SampleCount.X4,
                DepthStencil = DepthStencilState.Default with { DepthTest = true, DepthWrite = true },
                DebugName = "GeneratedMsaaDepthLoadReject"
            });

        var ex = Assert.Throws<FeatherNativeException>(() => pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedFragmentShader(sampler),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load }));

        Assert.Contains("MSAA depth load is not supported", ex.Message);
        Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineColorLoadPreservesPreviousPassOutsideSecondGeometry()
    {
        using var floorVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var quadVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.5f, -0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(0.5f, 0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorLoadPreserve" });

        pipeline.Draw(
            new GeneratedVertexShader(floorVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.25f, 0.5f, 0.75f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        pipeline.Draw(
            new GeneratedVertexShader(quadVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            target,
            vertexCount: 6,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = GraphicsColorLoadOp.Load });

        var readback = new float4[64];
        target.Read(readback);
        AssertColorNear(readback[0], new float4(0.25f, 0.5f, 0.75f, 1.0f));
        AssertColorNear(readback[(4 * 8) + 4], new float4(1.0f, 1.0f, 1.0f, 1.0f));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineColorClearDiscardsPreviousPassOutsideSecondGeometry()
    {
        using var fullscreenVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var quadVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.5f, -0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(0.5f, 0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorClearDiscard" });

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 0.0f, 0.0f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        pipeline.Draw(
            new GeneratedVertexShader(quadVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            target,
            vertexCount: 6,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 1.0f, 1.0f)
            });

        var readback = new float4[64];
        target.Read(readback);
        AssertColorNear(readback[0], new float4(0.0f, 0.0f, 1.0f, 1.0f));
        AssertColorNear(readback[(4 * 8) + 4], new float4(1.0f, 1.0f, 1.0f, 1.0f));
        Assert.DoesNotContain(readback, pixel => pixel.X > 0.8f && pixel.Y < 0.2f && pixel.Z < 0.2f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineDefaultColorLoadClearsWhenClearColorIsSpecified()
    {
        using var fullscreenVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorDefaultClearColor" });
        target.Upload([.. Enumerable.Repeat(new float4(1.0f, 0.0f, 0.0f, 1.0f), 16)]);

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 1.0f, 0.0f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ClearColor = new float4(0.0f, 0.0f, 1.0f, 1.0f),
                Scissor = new GraphicsRect(0, 0, 1, 1)
            });

        var readback = new float4[16];
        target.Read(readback);
        Assert.Contains(readback, pixel => IsColorNear(pixel, new float4(0.0f, 1.0f, 0.0f, 1.0f)));
        Assert.Contains(readback, pixel => IsColorNear(pixel, new float4(0.0f, 0.0f, 1.0f, 1.0f)));
        Assert.DoesNotContain(readback, pixel => pixel.X > 0.8f && pixel.Y < 0.2f && pixel.Z < 0.2f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineColorDontCareAcceptsFullscreenOverwrite()
    {
        using var fullscreenVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorDontCare" });

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.2f, 0.4f, 0.8f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = GraphicsColorLoadOp.DontCare });

        var readback = new float4[16];
        target.Read(readback);
        Assert.All(readback, pixel => AssertColorNear(pixel, new float4(0.2f, 0.4f, 0.8f, 1.0f)));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineIndexedDrawUsesColorLoadDesc()
    {
        using var fullscreenVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var quadVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.5f, -0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1),
            new float4(0.5f, 0.5f, 0, 1)
        ]);
        using var indices = GPU.CreateBuffer<uint>([0u, 1u, 2u, 1u, 3u, 2u], BufferAccess.ReadOnly);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorLoadIndexed" });

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.1f, 0.7f, 0.2f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        pipeline.DrawIndexed(
            new GeneratedVertexShader(quadVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            [target],
            indices,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = GraphicsColorLoadOp.Load });

        var readback = new float4[64];
        target.Read(readback);
        AssertColorNear(readback[0], new float4(0.1f, 0.7f, 0.2f, 1.0f));
        AssertColorNear(readback[(4 * 8) + 4], new float4(1.0f, 1.0f, 1.0f, 1.0f));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineRejectsColorLoadWithClearColor()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorLoadRejectsClear" });

        var ex = Assert.Throws<FeatherNativeException>(() => pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Load,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            }));

        Assert.Contains("ClearColor when ColorLoadOp is Load", ex.Message);
        Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineRejectsInvalidColorLoadOp()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedColorLoadInvalid" });

        var ex = Assert.Throws<FeatherNativeException>(() => pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = (GraphicsColorLoadOp)99u }));

        Assert.Contains("color load op contains an unsupported value", ex.Message);
        Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineMsaaColorLoadPreservesPreviousPassOutsideSecondGeometry()
    {
        using var floorVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var quadVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.5f, -0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(0.5f, 0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(16, 16, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var floorPipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "GeneratedMsaaColorLoadFloor" });
        using var lightPipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "GeneratedMsaaColorLoadLight" });

        floorPipeline.Draw(
            new GeneratedVertexShader(floorVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.15f, 0.45f, 0.75f, 1.0f))),
            target,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        lightPipeline.Draw(
            new GeneratedVertexShader(quadVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            target,
            vertexCount: 6,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = GraphicsColorLoadOp.Load });

        var readback = new float4[256];
        target.Read(readback);
        AssertColorNear(readback[0], new float4(0.15f, 0.45f, 0.75f, 1.0f));
        AssertColorNear(readback[(8 * 16) + 8], new float4(1.0f, 1.0f, 1.0f, 1.0f));
        Assert.Equal(DispatchPath.TypedEasyGpu, floorPipeline.LastDispatchPath);
        Assert.Equal(DispatchPath.TypedEasyGpu, lightPipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineMsaaColorLoadKeepsSameSizedTargetsIsolated()
    {
        using var fullscreenVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var quadVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.5f, -0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1),
            new float4(0.5f, -0.5f, 0, 1),
            new float4(0.5f, 0.5f, 0, 1),
            new float4(-0.5f, 0.5f, 0, 1)
        ]);
        using var targetA = GPU.CreateRenderTexture2D<float4, float4>(16, 16, PixelFormat.Rgba32Float);
        using var targetB = GPU.CreateRenderTexture2D<float4, float4>(16, 16, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { SampleCount = SampleCount.X4, DebugName = "GeneratedMsaaColorLoadTargetIsolation" });

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 0.8f, 0.2f, 1.0f))),
            targetA,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        pipeline.Draw(
            new GeneratedVertexShader(fullscreenVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.8f, 0.0f, 0.1f, 1.0f))),
            targetB,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc
            {
                ColorLoadOp = GraphicsColorLoadOp.Clear,
                ClearColor = new float4(0.0f, 0.0f, 0.0f, 1.0f)
            });

        pipeline.Draw(
            new GeneratedVertexShader(quadVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 1.0f, 1.0f))),
            targetA,
            vertexCount: 6,
            drawDesc: new GraphicsDrawDesc { ColorLoadOp = GraphicsColorLoadOp.Load });

        var readbackA = new float4[256];
        var readbackB = new float4[256];
        targetA.Read(readbackA);
        targetB.Read(readbackB);
        AssertColorNear(readbackA[0], new float4(0.0f, 0.8f, 0.2f, 1.0f));
        AssertColorNear(readbackA[(8 * 16) + 8], new float4(1.0f, 1.0f, 1.0f, 1.0f));
        AssertColorNear(readbackB[0], new float4(0.8f, 0.0f, 0.1f, 1.0f));
        Assert.DoesNotContain(readbackA, pixel => pixel.X > 0.7f && pixel.Y < 0.1f && pixel.Z < 0.2f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineSamplesTextureThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var source = GPU.CreateTexture2D<float4, float4>(1, 1, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        source.Upload([new float4(0.2f, 0.7f, 0.4f, 1.0f)]);
        target.Upload([.. Enumerable.Repeat(new float4(0, 0, 0, 0), 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedTextureFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedTextureFragmentShader(source.AsSampled(), sampler),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X > 0.6f && pixel.Y is > 0.15f and < 0.25f && pixel.Z is > 0.35f and < 0.45f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineSamplesTextureWithTextureCoordinateSwizzleThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var source = GPU.CreateTexture2D<float4, float4>(1, 1, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        source.Upload([new float4(0.25f, 0.5f, 0.75f, 1.0f)]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedTextureCoordinateFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedTextureCoordinateFragmentShader(source.AsSampled(), sampler),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X is > 0.2f and < 0.3f && pixel.Y is > 0.45f and < 0.55f && pixel.Z is > 0.7f and < 0.8f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineSamplesGeneratedMipLevelThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var source = GPU.CreateTexture2D<Rgba32, float4>(4, 4, 3, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        source.Upload(
        [
            new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255),
            new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255),
            new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255),
            new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255), new Rgba32(0, 0, 0, 255), new Rgba32(255, 255, 255, 255)
        ]);
        source.GenerateMipmaps();
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);

        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMipSampleFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedMipSampleFragmentShader(source.AsSampled(), sampler),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X is > 0.45f and < 0.55f && pixel.Y is > 0.45f and < 0.55f && pixel.Z is > 0.45f and < 0.55f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineImplicitSampleUsesGeneratedMipChainForMinification()
    {
        using var vertices = GPU.CreateBuffer<GeneratedImplicitMipVertex>(
        [
            new() { Position = new float3(-1, -1, 0), Uv = new float2(0, 0) },
            new() { Position = new float3(3, -1, 0), Uv = new float2(100, 0) },
            new() { Position = new float3(-1, 3, 0), Uv = new float2(0, 100) }
        ], BufferAccess.ReadOnly);
        var checker = CreateCheckerboard(64);
        using var unmipped = GPU.CreateTexture2D<Rgba32, float4>(64, 64, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var mipped = GPU.CreateTexture2D<Rgba32, float4>(64, 64, 7, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(16, 16, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedImplicitMipVertexShader, GeneratedImplicitMipFragmentShader, GeneratedImplicitMipVaryings>();
        unmipped.Upload(checker);
        mipped.Upload(checker);
        mipped.GenerateMipmaps();

        pipeline.Draw(
            new GeneratedImplicitMipVertexShader(vertices.AsReadOnly()),
            new GeneratedImplicitMipFragmentShader(unmipped.AsSampled(), sampler),
            target,
            vertexCount: 3);

        var readback = new float4[256];
        target.Read(readback);
        var unmippedGrayPixels = readback.Count(IsGray);

        target.Upload([.. Enumerable.Repeat(float4.Zero, 256)]);
        pipeline.Draw(
            new GeneratedImplicitMipVertexShader(vertices.AsReadOnly()),
            new GeneratedImplicitMipFragmentShader(mipped.AsSampled(), sampler),
            target,
            vertexCount: 3);

        target.Read(readback);
        var mippedGrayPixels = readback.Count(IsGray);

        Assert.True(unmippedGrayPixels < 32, $"Unmipped sample unexpectedly produced {unmippedGrayPixels} gray pixels.");
        Assert.True(mippedGrayPixels > 128, $"Mipmapped sample produced only {mippedGrayPixels} gray pixels.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineSamplerDescriptorAffectsFiltering()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var source = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var nearest = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var linear = GPU.CreateSampler(SamplerDesc.LinearClamp);
        source.Upload(
        [
            new float4(0, 0, 0, 1), new float4(1, 0, 0, 1),
            new float4(0, 1, 0, 1), new float4(1, 1, 0, 1)
        ]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantUvTextureFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantUvTextureFragmentShader(source.AsSampled(), nearest, new Uniform<float2>(new float2(0.5f, 0.5f))),
            target,
            vertexCount: 3);

        var nearestPixels = new float4[16];
        target.Read(nearestPixels);
        var nearestPixel = FirstDrawn(nearestPixels);

        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantUvTextureFragmentShader(source.AsSampled(), linear, new Uniform<float2>(new float2(0.5f, 0.5f))),
            target,
            vertexCount: 3);

        var linearPixels = new float4[16];
        target.Read(linearPixels);
        var linearPixel = FirstDrawn(linearPixels);

        Assert.True(System.Math.Abs(linearPixel.X - 0.5f) < 0.2f, $"Expected linear sample near 0.5 red, got {linearPixel.X}.");
        Assert.True(System.Math.Abs(linearPixel.Y - 0.5f) < 0.2f, $"Expected linear sample near 0.5 green, got {linearPixel.Y}.");
        Assert.True(System.Math.Abs(nearestPixel.X - linearPixel.X) > 0.2f || System.Math.Abs(nearestPixel.Y - linearPixel.Y) > 0.2f,
            $"Nearest and linear samples were too similar: nearest={nearestPixel}, linear={linearPixel}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineSamplerDescriptorAffectsAddressMode()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var source = GPU.CreateTexture2D<float4, float4>(2, 2, PixelFormat.Rgba32Float, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var clamp = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);
        source.Upload(
        [
            new float4(0, 0, 0, 1), new float4(1, 0, 0, 1),
            new float4(0, 1, 0, 1), new float4(1, 1, 0, 1)
        ]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantUvTextureFragmentShader, float4>();

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantUvTextureFragmentShader(source.AsSampled(), clamp, new Uniform<float2>(new float2(1.25f, 0.25f))),
            target,
            vertexCount: 3);
        var clampPixels = new float4[16];
        target.Read(clampPixels);
        var clampPixel = FirstDrawn(clampPixels);

        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantUvTextureFragmentShader(source.AsSampled(), sampler, new Uniform<float2>(new float2(1.25f, 0.25f))),
            target,
            vertexCount: 3);
        var repeatPixels = new float4[16];
        target.Read(repeatPixels);
        var repeatPixel = FirstDrawn(repeatPixels);

        Assert.True(System.Math.Abs(clampPixel.X - repeatPixel.X) > 0.2f || System.Math.Abs(clampPixel.Y - repeatPixel.Y) > 0.2f,
            $"Clamp and repeat samples were too similar: clamp={clampPixel}, repeat={repeatPixel}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawsMultipleRenderTargetsThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target0 = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var target1 = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target0.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        target1.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMrtFragmentShader, float4>(
            new GraphicsPipelineDesc { ColorAttachmentCount = 2, DebugName = "GeneratedMrt" });

        IGpuTexture2D[] targets = [target0, target1];
        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedMrtFragmentShader(sampler),
            targets,
            vertexCount: 3);

        var readback0 = new float4[16];
        var readback1 = new float4[16];
        target0.Read(readback0);
        target1.Read(readback1);
        Assert.Contains(readback0, pixel => pixel.X > 0.8f && pixel.Y < 0.2f && pixel.Z < 0.2f && pixel.W > 0.9f);
        Assert.Contains(readback1, pixel => pixel.X < 0.2f && pixel.Y > 0.8f && pixel.Z < 0.2f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineRejectsInvalidColorAttachmentAndBlendAttachmentCounts()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMrtFragmentShader, float4>(
                new GraphicsPipelineDesc { ColorAttachmentCount = 9 }));

        Assert.Throws<ArgumentException>(() =>
            GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMrtFragmentShader, float4>(
                new GraphicsPipelineDesc
                {
                    ColorAttachmentCount = 2,
                    BlendAttachments = [BlendState.Opaque]
                }));
    }

    [Fact]
    public void GeneratedGraphicsPipelineAppliesPerAttachmentColorWriteMasksThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target0 = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var target1 = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target0.Upload([.. Enumerable.Repeat(new float4(0.25f, 0.25f, 0.25f, 1.0f), 16)]);
        target1.Upload([.. Enumerable.Repeat(new float4(0.25f, 0.25f, 0.25f, 1.0f), 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedMrtFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                ColorAttachmentCount = 2,
                BlendAttachments =
                [
                    BlendState.Opaque with { WriteMask = ColorWriteMask.Red },
                    BlendState.Opaque with { WriteMask = ColorWriteMask.Green }
                ],
                DebugName = "GeneratedMrtPerAttachmentWriteMask"
            });

        IGpuTexture2D[] targets = [target0, target1];
        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedMrtFragmentShader(sampler),
            targets,
            vertexCount: 3);

        var readback0 = new float4[16];
        var readback1 = new float4[16];
        target0.Read(readback0);
        target1.Read(readback1);
        var pixel0 = readback0[5];
        var pixel1 = readback1[5];
        Assert.True(pixel0.X > 0.8f && pixel0.Y is > 0.2f and < 0.3f && pixel0.Z is > 0.2f and < 0.3f, $"Expected only target0 red channel to change, got {pixel0}.");
        Assert.True(pixel1.X is > 0.2f and < 0.3f && pixel1.Y > 0.8f && pixel1.Z is > 0.2f and < 0.3f, $"Expected only target1 green channel to change, got {pixel1}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineAppliesAlphaBlendThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(new float4(0.25f, 0.0f, 0.0f, 1.0f), 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc { Blend = BlendState.AlphaBlend, DebugName = "GeneratedAlphaBlend" });

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 0.0f, 1.0f, 0.5f))),
            target,
            vertexCount: 3);

        var readback = new float4[16];
        target.Read(readback);
        var pixel = readback[5];
        Assert.True(pixel.X is > 0.10f and < 0.15f, $"Expected blended red near 0.125, got {pixel}.");
        Assert.True(pixel.Z is > 0.45f and < 0.55f, $"Expected blended blue near 0.5, got {pixel}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineAppliesDepthCompareGreaterThroughEasyGpu()
    {
        using var nearVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.25f, 1),
            new float4(3, -1, 0.25f, 1),
            new float4(-1, 3, 0.25f, 1)
        ]);
        using var farVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.75f, 1),
            new float4(3, -1, 0.75f, 1),
            new float4(-1, 3, 0.75f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var depth = GPU.CreateDepthTexture2D(4, 4);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        depth.Upload([.. Enumerable.Repeat(0.0f, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    DepthTest = true,
                    DepthWrite = true,
                    DepthCompare = CompareOp.Greater
                },
                DebugName = "GeneratedDepthGreater"
            });

        pipeline.Draw(
            new GeneratedVertexShader(nearVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 0.0f, 0.0f, 1.0f))),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ClearDepth = 0.0f });
        pipeline.Draw(
            new GeneratedVertexShader(farVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 1.0f, 0.0f, 1.0f))),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load });

        var readback = new float4[16];
        target.Read(readback);
        var pixel = readback[5];
        Assert.True(pixel.Y > 0.8f && pixel.X < 0.2f, $"Expected later greater-depth draw to win, got {pixel}.");
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineLegacyDepthFieldsEnableDepthState()
    {
        using var nearVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.25f, 1),
            new float4(3, -1, 0.25f, 1),
            new float4(-1, 3, 0.25f, 1)
        ]);
        using var farVertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.75f, 1),
            new float4(3, -1, 0.75f, 1),
            new float4(-1, 3, 0.75f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var depth = GPU.CreateDepthTexture2D(4, 4);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthTest = true,
                DepthWrite = true,
                DebugName = "GeneratedLegacyDepthFields"
            });

        pipeline.Draw(
            new GeneratedVertexShader(nearVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 0.0f, 0.0f, 1.0f))),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ClearDepth = 1.0f });
        pipeline.Draw(
            new GeneratedVertexShader(farVertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 1.0f, 0.0f, 1.0f))),
            target,
            depth,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load });

        var readback = new float4[16];
        target.Read(readback);
        var pixel = readback[5];
        Assert.True(pixel.X > 0.8f && pixel.Y < 0.2f, $"Expected legacy depth fields to preserve the near draw, got {pixel}.");
        Assert.True(pipeline.Desc.DepthStencil.DepthTest);
        Assert.True(pipeline.Desc.DepthStencil.DepthWrite);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineAppliesStencilReplaceAndEqualThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0.5f, 1),
            new float4(3, -1, 0.5f, 1),
            new float4(-1, 3, 0.5f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var depthStencil = GPU.CreateDepthStencilTexture2D(4, 4);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        using var writeStencil = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    StencilTest = true,
                    Front = StencilFaceState.KeepAlways with { PassOp = StencilOp.Replace },
                    Back = StencilFaceState.KeepAlways with { PassOp = StencilOp.Replace },
                    StencilReference = 7
                },
                DebugName = "GeneratedStencilWrite"
            });
        using var testStencil = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    StencilTest = true,
                    Front = StencilFaceState.KeepAlways with { Compare = CompareOp.Equal },
                    Back = StencilFaceState.KeepAlways with { Compare = CompareOp.Equal },
                    StencilReference = 7
                },
                DebugName = "GeneratedStencilEqual"
            });

        writeStencil.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 0.0f, 0.0f, 0.0f))),
            target,
            depthStencil,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { ClearDepth = 1.0f });
        testStencil.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.0f, 1.0f, 1.0f, 1.0f))),
            target,
            depthStencil,
            vertexCount: 3,
            drawDesc: new GraphicsDrawDesc { DepthLoadOp = GraphicsDepthLoadOp.Load });

        var readback = new float4[16];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.Y > 0.8f && pixel.Z > 0.8f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, testStencil.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineAppliesCullModeAndFrontFaceThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                Raster = RasterState.Default with
                {
                    CullMode = CullMode.Back,
                    FrontFace = FrontFace.CounterClockwise
                },
                DebugName = "GeneratedCullFrontFace"
            });

        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 0.0f, 0.0f, 1.0f))),
            target,
            vertexCount: 3);

        var readback = new float4[16];
        target.Read(readback);
        Assert.DoesNotContain(readback, pixel => pixel.X > 0.5f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineUsesPolygonLineModeWhenDeviceSupportsIt()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-0.8f, -0.8f, 0, 1),
            new float4(0.8f, -0.8f, 0, 1),
            new float4(0.0f, 0.8f, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(16, 16, PixelFormat.Rgba32Float);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                Raster = RasterState.Default with { PolygonMode = PolygonMode.Line },
                DebugName = "GeneratedPolygonLine"
            });

        target.Upload([.. Enumerable.Repeat(float4.Zero, 256)]);
        try
        {
            pipeline.Draw(
                new GeneratedVertexShader(vertices.AsReadOnly()),
                new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(1.0f, 1.0f, 0.0f, 1.0f))),
                target,
                vertexCount: 3);

            var readback = new float4[256];
            target.Read(readback);
            var litPixels = readback.Count(pixel => pixel.X > 0.5f && pixel.Y > 0.5f);
            Assert.True(litPixels is > 0 and < 96, $"Expected line rasterization to draw edges without filling the triangle, got {litPixels} lit pixels.");
            Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
        }
        catch (FeatherNativeException ex) when (ex.Message.Contains("fillModeNonSolid", StringComparison.Ordinal))
        {
            Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
        }
    }

    [Fact]
    public void GeneratedGraphicsPipelineUsesDepthClampWhenDeviceSupportsIt()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, -0.5f, 1),
            new float4(3, -1, -0.5f, 1),
            new float4(-1, 3, -0.5f, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(4, 4, PixelFormat.Rgba32Float);
        using var depth = GPU.CreateDepthTexture2D(4, 4);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedConstantColorFragmentShader, float4>(
            new GraphicsPipelineDesc
            {
                DepthStencil = DepthStencilState.Default with
                {
                    DepthTest = true,
                    DepthWrite = true,
                    DepthCompare = CompareOp.LessOrEqual
                },
                Raster = RasterState.Default with { DepthClamp = true },
                DebugName = "GeneratedDepthClamp"
            });

        target.Upload([.. Enumerable.Repeat(float4.Zero, 16)]);
        try
        {
            pipeline.Draw(
                new GeneratedVertexShader(vertices.AsReadOnly()),
                new GeneratedConstantColorFragmentShader(sampler, new Uniform<float4>(new float4(0.4f, 0.2f, 1.0f, 1.0f))),
                target,
                depth,
                vertexCount: 3,
                drawDesc: new GraphicsDrawDesc { ClearDepth = 1.0f });

            var readback = new float4[16];
            target.Read(readback);
            Assert.Contains(readback, pixel => pixel.Z > 0.8f && pixel.W > 0.9f);
            Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
        }
        catch (FeatherNativeException ex) when (ex.Message.Contains("depthClamp", StringComparison.Ordinal))
        {
            Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
        }
    }

    [Fact]
    public void GeneratedGraphicsPipelineDrawsGpuStructMeshWithSampledRgba8TextureAndDepthThroughEasyGpu()
    {
        using var vertices = GPU.CreateBuffer<GeneratedMeshVertex>(
        [
            new()
            {
                Position = new float3(-1, -1, 0.25f),
                Normal = new float3(0, 0, 1),
                Uv = new float2(0.25f, 0.25f),
                AtlasTransform = new float4(0, 0, 1, 1)
            },
            new()
            {
                Position = new float3(3, -1, 0.25f),
                Normal = new float3(0, 0, 1),
                Uv = new float2(0.75f, 0.25f),
                AtlasTransform = new float4(0, 0, 1, 1)
            },
            new()
            {
                Position = new float3(-1, 3, 0.25f),
                Normal = new float3(0, 0, 1),
                Uv = new float2(0.25f, 0.75f),
                AtlasTransform = new float4(0, 0, 1, 1)
            }
        ], BufferAccess.ReadOnly);
        using var atlas = GPU.CreateTexture2D<Rgba32, float4>(2, 2, PixelFormat.Rgba8, TextureAccess.Sampled);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var depth = GPU.CreateDepthTexture2D(8, 8);
        using var sampler = GPU.CreateSampler(SamplerDesc.NearestClamp);
        atlas.Upload(
        [
            new Rgba32(220, 40, 20, 255),
            new Rgba32(220, 40, 20, 255),
            new Rgba32(220, 40, 20, 255),
            new Rgba32(220, 40, 20, 255)
        ]);
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedMeshVertexShader, GeneratedMeshFragmentShader, GeneratedMeshVaryings>(
            new GraphicsPipelineDesc { DepthTest = true, DepthWrite = true, DebugName = "GeneratedMeshTextureDepth" });

        pipeline.Draw(
            new GeneratedMeshVertexShader(vertices.AsReadOnly(), new Uniform<float4x4>(float4x4.Identity)),
            new GeneratedMeshFragmentShader(atlas.AsSampled(), sampler),
            target,
            depth,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X > pixel.Y && pixel.X > pixel.Z && pixel.W > 0.5f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void GeneratedGraphicsPipelineLowersCallablesStructParametersAndFullFragmentBody()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(3, -1, 0, 1),
            new float4(-1, 3, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<float4, float4>(8, 8, PixelFormat.Rgba32Float);
        using var pipeline = GPU.CreateGraphicsPipeline<GeneratedVertexShader, GeneratedCallableBodyFragmentShader, float4>(
            new GraphicsPipelineDesc { DebugName = "GeneratedCallableBody" });
        target.Upload([.. Enumerable.Repeat(float4.Zero, 64)]);

        pipeline.Draw(
            new GeneratedVertexShader(vertices.AsReadOnly()),
            new GeneratedCallableBodyFragmentShader(new Uniform<float>(0.8f)),
            target,
            vertexCount: 3);

        var readback = new float4[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel.X > 0.5f && pixel.Y > 0.2f && pixel.Z < 0.5f && pixel.W > 0.9f);
        Assert.Contains(readback, pixel => pixel.X < 0.05f && pixel.Y < 0.05f && pixel.Z < 0.05f && pixel.W > 0.9f);
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    private static Rgba32[] CreateCheckerboard(int size)
    {
        var pixels = new Rgba32[size * size];
        for (var y = 0; y < size; y++)
        {
            for (var x = 0; x < size; x++)
            {
                var value = ((x + y) & 1) == 0 ? byte.MaxValue : (byte)0;
                pixels[(y * size) + x] = new Rgba32(value, value, value, byte.MaxValue);
            }
        }

        return pixels;
    }

    private static bool IsGray(float4 pixel)
        => pixel.X is > 0.35f and < 0.65f &&
           pixel.Y is > 0.35f and < 0.65f &&
           pixel.Z is > 0.35f and < 0.65f &&
           pixel.W > 0.9f;

    private static float4 FirstDrawn(float4[] pixels)
        => pixels.First(pixel => pixel.W > 0.5f);

    private static void AssertColorNear(float4 actual, float4 expected, float tolerance = 0.02f)
    {
        Assert.True(IsColorNear(actual, expected, tolerance),
            $"Expected color near {expected}, got {actual}.");
    }

    private static bool IsColorNear(float4 actual, float4 expected, float tolerance = 0.02f)
        => System.Math.Abs(actual.X - expected.X) <= tolerance &&
           System.Math.Abs(actual.Y - expected.Y) <= tolerance &&
           System.Math.Abs(actual.Z - expected.Z) <= tolerance &&
           System.Math.Abs(actual.W - expected.W) <= tolerance;

    private static GraphicsPipelineDescriptor GetGraphicsDescriptor<TVertexShader, TFragmentShader, TVaryings>()
        where TVertexShader : struct, IVertexShader<TVaryings>, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TFragmentShader : struct, IGeneratedGraphicsPipeline<TVertexShader, TFragmentShader, TVaryings>
        where TVaryings : unmanaged
        => TVertexShader.Descriptor;

    private readonly record struct Rgba32(byte R, byte G, byte B, byte A);
}

[VertexShader]
public readonly partial struct GeneratedVertexShader(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute()
    {
        return vertices[VertexIds.Index];
    }
}

[FragmentShader]
public readonly partial struct GeneratedFragmentShader(SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return input;
    }
}

[VertexShader]
public readonly partial struct GeneratedUniformVertexShader(ReadOnlyBuffer<float4> vertices, Uniform<float4> transform) : IVertexShader<float4>
{
    public float4 Execute()
    {
        return vertices[VertexIds.Index];
    }
}

[FragmentShader]
public readonly partial struct GeneratedUniformFragmentShader(SamplerState sampler, Uniform<float4> color) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return input;
    }
}

[FragmentShader]
public readonly partial struct GeneratedMixedFloat3UniformFragmentShader(
    Uniform<float> aspect,
    Uniform<float> roughness,
    Uniform<float3> lightP0,
    Uniform<float3> lightP1,
    Uniform<float3> lightP2,
    Uniform<float3> lightP3,
    Uniform<float> intensity,
    Uniform<int> twoSided) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        var alpha = twoSided.Value > 0 ? 1.0f : 0.25f;
        return new float4(
            lightP0.Value.X + lightP1.Value.Y + lightP2.Value.Z,
            lightP3.Value.X + aspect.Value,
            roughness.Value + intensity.Value,
            alpha);
    }
}

[GpuStruct]
public partial struct GeneratedCrossStageFloat3Varyings
{
    [Position]
    public float4 Position;

    public float4 Data;
}

[VertexShader]
public readonly partial struct GeneratedCrossStageFloat3UniformVertexShader(
    ReadOnlyBuffer<float4> vertices,
    Uniform<float> vertexScalar0,
    Uniform<float3> vertexVector,
    Uniform<float> vertexScalar1) : IVertexShader<GeneratedCrossStageFloat3Varyings>
{
    public GeneratedCrossStageFloat3Varyings Execute()
    {
        return new GeneratedCrossStageFloat3Varyings
        {
            Position = vertices[VertexIds.Index],
            Data = new float4(
                vertexScalar0.Value + vertexVector.Value.X,
                vertexVector.Value.Y + vertexScalar1.Value,
                vertexVector.Value.Z,
                1.0f)
        };
    }
}

[FragmentShader]
public readonly partial struct GeneratedCrossStageFloat3UniformFragmentShader(
    Uniform<float> fragmentScalar0,
    Uniform<float3> fragmentVector0,
    Uniform<float3> fragmentVector1,
    Uniform<float> fragmentScalar1) : IFragmentShader<GeneratedCrossStageFloat3Varyings>
{
    public float4 Execute(GeneratedCrossStageFloat3Varyings input)
    {
        return new float4(
            input.Data.X + fragmentVector0.Value.X + fragmentVector1.Value.Y,
            input.Data.Y + fragmentScalar0.Value,
            input.Data.Z + fragmentVector0.Value.Z + fragmentVector1.Value.X + fragmentScalar1.Value,
            input.Data.W);
    }
}

[GpuStruct]
public partial struct GeneratedMatrixUniformVaryings
{
    [Position]
    public float4 Position;

    public float4 Data;
}

[VertexShader]
public readonly partial struct GeneratedMatrixUniformVertexShader(
    ReadOnlyBuffer<float4> vertices,
    Uniform<float2x2> vertexMatrix2,
    Uniform<float4x4> vertexMatrix4) : IVertexShader<GeneratedMatrixUniformVaryings>
{
    public GeneratedMatrixUniformVaryings Execute()
    {
        var data = vertexMatrix2.Value * new float2(1.0f, 2.0f);
        return new GeneratedMatrixUniformVaryings
        {
            Position = vertexMatrix4.Value * vertices[VertexIds.Index],
            Data = new float4(data.X, data.Y, 0.0f, 1.0f)
        };
    }
}

[FragmentShader]
public readonly partial struct GeneratedMatrixUniformFragmentShader(
    Uniform<float2x2> fragmentMatrix2,
    Uniform<float3x3> fragmentMatrix3,
    Uniform<float> scalar) : IFragmentShader<GeneratedMatrixUniformVaryings>
{
    public float4 Execute(GeneratedMatrixUniformVaryings input)
    {
        var from2 = fragmentMatrix2.Value * new float2(1.0f, 2.0f);
        var from3 = fragmentMatrix3.Value * new float3(input.Data.X, input.Data.Y, 1.0f);
        return new float4(from2.X + from3.X, from2.Y + from3.Y, from3.Z + scalar.Value, input.Data.W);
    }
}

[VertexShader]
public readonly partial struct EntryAttributedVertexShader(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    [Entry]
    public float4 Run()
    {
        return vertices[VertexIds.Index];
    }

    public float4 Execute()
    {
        return new float4(2, 2, 0, 1);
    }
}

[FragmentShader]
public readonly partial struct EntryAttributedFragmentShader(SamplerState sampler) : IFragmentShader<float4>
{
    [Entry]
    public float4 Shade(float4 input)
    {
        return new float4(0.9f, 0.3f, 0.5f, 1.0f);
    }

    public float4 Execute(float4 input)
    {
        return new float4(0.0f, 1.0f, 0.0f, 1.0f);
    }
}

[FragmentShader]
public readonly partial struct GeneratedSwizzledUniformFragmentShader(SamplerState sampler, Uniform<float4> color) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return new float4(color.Value.G, color.Value.R, color.Value.B, color.Value.A);
    }
}

[FragmentShader]
public readonly partial struct GeneratedConstantColorFragmentShader(SamplerState sampler, Uniform<float4> color) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return color.Value;
    }
}

[FragmentShader]
public readonly partial struct GeneratedTextureFragmentShader(SampledTexture2D<float4> texture, SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        var sampled = texture.Sample(sampler, input.XY);
        return new float4(sampled.G, sampled.R, sampled.B, sampled.A);
    }
}

[FragmentShader]
public readonly partial struct GeneratedTextureCoordinateFragmentShader(SampledTexture2D<float4> texture, SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return texture.Sample(sampler, input.ST);
    }
}

[FragmentShader]
public readonly partial struct GeneratedMipSampleFragmentShader(SampledTexture2D<float4> texture, SamplerState sampler) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return texture.SampleLevel(sampler, new float2(0.5f, 0.5f), 2.0f);
    }
}

[FragmentShader]
public readonly partial struct GeneratedConstantUvTextureFragmentShader(
    SampledTexture2D<float4> texture,
    SamplerState sampler,
    Uniform<float2> uv) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return texture.Sample(sampler, uv.Value);
    }
}

[GpuStruct]
public partial struct GeneratedMrtOutput
{
    [Color(0)]
    public float4 Target0;

    [Color(1)]
    public float4 Target1;
}

[FragmentShader]
public readonly partial struct GeneratedMrtFragmentShader(SamplerState sampler) : IFragmentShader<float4, GeneratedMrtOutput>
{
    public GeneratedMrtOutput Execute(float4 input)
    {
        return new GeneratedMrtOutput
        {
            Target0 = new float4(1.0f, 0.0f, 0.0f, 1.0f),
            Target1 = new float4(0.0f, 1.0f, 0.0f, 1.0f)
        };
    }
}

[GpuStruct]
public partial struct GeneratedImplicitMipVertex
{
    public float3 Position;
    public float2 Uv;
}

[GpuStruct]
public partial struct GeneratedImplicitMipVaryings
{
    [Position]
    public float4 Position;
    public float2 Uv;
}

[VertexShader]
public readonly partial struct GeneratedImplicitMipVertexShader(ReadOnlyBuffer<GeneratedImplicitMipVertex> vertices) : IVertexShader<GeneratedImplicitMipVaryings>
{
    public GeneratedImplicitMipVaryings Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new GeneratedImplicitMipVaryings
        {
            Position = new float4(vertex.Position, 1.0f),
            Uv = vertex.Uv
        };
    }
}

[FragmentShader]
public readonly partial struct GeneratedImplicitMipFragmentShader(
    SampledTexture2D<float4> texture,
    SamplerState sampler) : IFragmentShader<GeneratedImplicitMipVaryings>
{
    public float4 Execute(GeneratedImplicitMipVaryings input)
    {
        return texture.Sample(sampler, ShaderMath.Fract(input.Uv));
    }
}

[GpuStruct]
public partial struct GeneratedMeshVertex
{
    public float3 Position;
    public float3 Normal;
    public float2 Uv;
    public float4 AtlasTransform;
}

[GpuStruct]
public partial struct GeneratedMeshVaryings
{
    [Position]
    public float4 Position;
    public float3 Normal;
    public float2 Uv;
    public float4 AtlasTransform;
}

[GpuStruct]
public partial struct GeneratedCallableRect
{
    public float3 Center;
    public float3 DirX;
    public float3 DirY;
    public float HalfX;
    public float HalfY;
}

[VertexShader]
public readonly partial struct GeneratedMeshVertexShader(
    ReadOnlyBuffer<GeneratedMeshVertex> vertices,
    Uniform<float4x4> mvp) : IVertexShader<GeneratedMeshVaryings>
{
    public GeneratedMeshVaryings Execute()
    {
        var vertex = vertices[VertexIds.Index];
        return new GeneratedMeshVaryings
        {
            Position = ShaderMath.Mul(mvp.Value, new float4(vertex.Position, 1.0f)),
            Normal = ShaderMath.Normalize(vertex.Normal),
            Uv = vertex.Uv,
            AtlasTransform = vertex.AtlasTransform
        };
    }
}

[FragmentShader]
public readonly partial struct GeneratedMeshFragmentShader(
    SampledTexture2D<float4> texture,
    SamplerState sampler) : IFragmentShader<GeneratedMeshVaryings>
{
    public float4 Execute(GeneratedMeshVaryings input)
    {
        var tiled = ShaderMath.Fract(input.Uv);
        var uv = input.AtlasTransform.XY + (tiled * input.AtlasTransform.ZW);
        var ddx = ShaderMath.Ddx(input.Uv) * input.AtlasTransform.ZW;
        var ddy = ShaderMath.Ddy(input.Uv) * input.AtlasTransform.ZW;
        var normal = ShaderMath.Normalize(input.Normal);
        var light = ShaderMath.Max(ShaderMath.Dot(normal, ShaderMath.Normalize(new float3(0.3f, 0.6f, 0.4f))), 0.2f);
        var sampled = texture.SampleGrad(sampler, uv, ddx, ddy);
        return new float4(sampled.R * light, sampled.G * light, sampled.B * light, sampled.A);
    }
}

[FragmentShader]
public readonly partial struct GeneratedCallableBodyFragmentShader(Uniform<float> scale) : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        var rect = BuildRect();
        var direction = ShaderMath.Normalize(float3x3.Identity * new float3(input.X, input.Y, 1.0f));
        var hit = IntersectRect(float3.Zero, direction, rect);
        if (input.X < -0.75f)
        {
            return new float4(float3.Zero, 1.0f);
        }

        if (hit <= 0.0f)
        {
            return new float4(0.05f, 0.0f, 0.0f, 1.0f);
        }

        var color = ShadeRect(rect, direction, scale.Value);
        var greenBoost = rect.HalfY * 0.1f;
        greenBoost += 0.02f;
        return new float4(color.X, color.Y + greenBoost, color.Z, 1.0f);
    }

    [Callable]
    public static GeneratedCallableRect BuildRect()
    {
        var rect = new GeneratedCallableRect
        {
            Center = new float3(0.0f, 0.0f, 1.0f),
            DirX = new float3(1.0f, 0.0f, 0.0f),
            DirY = new float3(0.0f, 1.0f, 0.0f),
            HalfX = 0.5f,
            HalfY = 0.5f
        };
        rect.HalfX = rect.HalfX + 0.25f;
        return rect;
    }

    [Callable]
    public static float IntersectRect(float3 origin, float3 direction, GeneratedCallableRect rect)
    {
        var normal = ShaderMath.Normalize(ShaderMath.Cross(rect.DirX, rect.DirY));
        var denom = ShaderMath.Dot(normal, direction);
        if (ShaderMath.Abs(denom) <= 0.000001f)
        {
            return -1.0f;
        }

        var t = ShaderMath.Dot(normal, rect.Center - origin) / denom;
        if (t <= 0.0f)
        {
            return -1.0f;
        }

        var local = origin + (direction * t) - rect.Center;
        var x = ShaderMath.Dot(local, rect.DirX);
        var y = ShaderMath.Dot(local, rect.DirY);
        return (ShaderMath.Abs(x) <= rect.HalfX && ShaderMath.Abs(y) <= rect.HalfY) ? t : -1.0f;
    }

    [Callable]
    public static float3 ShadeRect(GeneratedCallableRect rect, float3 direction, float scale)
    {
        var normal = ShaderMath.Normalize(ShaderMath.Cross(rect.DirX, rect.DirY));
        var facing = ShaderMath.Max(ShaderMath.Dot(normal, direction), 0.0f);
        var value = facing * scale;
        for (int i = 0; i < 2; i++)
        {
            value += 0.025f;
        }

        return new float3(value, rect.HalfX * 0.4f, 1.0f - facing);
    }
}
