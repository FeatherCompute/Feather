using Feather.Graphics;
using Feather.Math;
using Feather.Native;
using Feather.Resources;

namespace Feather.Integration.Tests;

public class GraphicsRasterizationTests
{
    [Fact]
    public void DrawRasterizesColorTargetThroughTypedPath()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8, TextureAccess.RenderTarget);
        target.Upload(
        [
            .. Enumerable.Repeat(new Rgba32(1, 2, 3, 4), 64)
        ]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>();

        pipeline.Draw(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, vertexCount: 3);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(1, 2, 3, 4));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void DrawWithDepthTargetRasterizesColorTargetThroughTypedPath()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(8, 8);
        target.Upload([.. Enumerable.Repeat(new Rgba32(1, 2, 3, 4), 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>(
            new GraphicsPipelineDesc { DepthTest = true, DepthWrite = true });

        pipeline.Draw(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, depth, vertexCount: 3);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(1, 2, 3, 4));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void DrawIndexedWithDepthTargetRasterizesColorTargetThroughTypedPath()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(8, 8);
        using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2]);
        target.Upload([.. Enumerable.Repeat(new Rgba32(5, 6, 7, 8), 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>(
            new GraphicsPipelineDesc { DepthTest = true, DepthWrite = true });

        pipeline.DrawIndexed(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, depth, indices);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(5, 6, 7, 8));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void DrawIndexedBindsIndexBufferThroughTypedPath()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2]);
        target.Upload([.. Enumerable.Repeat(new Rgba32(9, 10, 11, 12), 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>();

        pipeline.DrawIndexed(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, indices);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(9, 10, 11, 12));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void DrawIndexedBindsIndexAndDepthTargetsThroughTypedPath()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        using var depth = GPU.CreateDepthTexture2D(8, 8);
        using var indices = GPU.CreateIndexBuffer<ushort>([0, 1, 2]);
        target.Upload([.. Enumerable.Repeat(new Rgba32(13, 14, 15, 16), 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>(
            new GraphicsPipelineDesc { DepthTest = true, DepthWrite = true });

        pipeline.DrawIndexed(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, depth, indices);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(13, 14, 15, 16));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    [Fact]
    public void NativePersistentIndexBufferBindingIsRejected()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>();

        var command = new GpuGraphicsCommand(pipeline.Handle);
        command.SetVertexBuffer(vertices.Handle, (uint)vertices.ElementStride);

        var staleStateException = Assert.Throws<FeatherNativeException>(
            () => NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_set_index_buffer(pipeline.Handle, indices.Handle)));
        Assert.Equal(FeResult.ErrorUnsupported, staleStateException.Result);

        var missingIndexException = Assert.Throws<FeatherNativeException>(
            () => NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_draw_indexed(
                pipeline.Handle,
                target.GetNativeHandle(),
                FeTextureHandle.Null,
                FeBufferHandle.Null,
                3,
                wait: true)));
        Assert.Equal(FeResult.ErrorInvalidHandle, missingIndexException.Result);
        Assert.Equal(DispatchPath.Rejected, pipeline.LastDispatchPath);
    }

    [Fact]
    public void DrawIndexedStillAcceptsExplicitIndexBufferAfterRejectedPersistentBinding()
    {
        using var vertices = GPU.CreateBuffer<float4>(
        [
            new float4(-1, -1, 0, 1),
            new float4(1, -1, 0, 1),
            new float4(0, 1, 0, 1)
        ]);
        using var indices = GPU.CreateIndexBuffer<uint>([0, 1, 2]);
        using var target = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(8, 8, PixelFormat.Rgba8);
        target.Upload([.. Enumerable.Repeat(new Rgba32(21, 22, 23, 24), 64)]);
        using var pipeline = GPU.CreateGraphicsPipeline<RasterVertexShader, RasterFragmentShader, float4>();

        _ = Assert.Throws<FeatherNativeException>(
            () => NativeMethods.ThrowIfFailed(NativeMethods.fe_graphics_pipeline_set_index_buffer(pipeline.Handle, indices.Handle)));

        pipeline.DrawIndexed(new RasterVertexShader(vertices.AsReadOnly()), new RasterFragmentShader(), target, indices);

        var readback = new Rgba32[64];
        target.Read(readback);
        Assert.Contains(readback, pixel => pixel != new Rgba32(21, 22, 23, 24));
        Assert.Equal(DispatchPath.TypedEasyGpu, pipeline.LastDispatchPath);
    }

    private readonly record struct Rgba32(byte R, byte G, byte B, byte A);
}

[VertexShader]
public readonly partial struct RasterVertexShader(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute() => vertices[VertexIds.Index];
}

[FragmentShader]
public readonly partial struct RasterFragmentShader : IFragmentShader<float4>
{
    public float4 Execute(float4 input) => input;
}
