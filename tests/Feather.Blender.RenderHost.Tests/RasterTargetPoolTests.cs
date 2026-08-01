using Feather.Math;
using Feather.RenderGraph;
using Feather.Resources;

namespace Feather.Blender.RenderHost.Tests;

public sealed class RasterTargetPoolTests
{
    [Fact]
    public void ExistingBackendsCanOmitReusableRasterTargets()
    {
        var context = new RenderContext(new LegacyRenderContextBackend());

        Assert.Throws<NotSupportedException>(
            () => context.GetOrCreateRenderTarget<Rgba8, float4>(
                new TextureHandle(1),
                PixelFormat.Rgba8));
        Assert.Throws<NotSupportedException>(
            () => context.GetOrCreateDepthTarget(new TextureHandle(1)));
    }

    [Fact]
    [Trait("Category", "Gpu")]
    public void MatchingTargetsAreReusedAndInvalidatedTargetsAreDisposed()
    {
        using var pool = new RasterTargetPool();
        pool.PrepareForGraph("graph-a");

        var color = pool.GetOrCreateRenderTarget<Rgba8, Rgba8>(
            "socket|pass|color",
            8,
            8,
            PixelFormat.Rgba8);
        var depth = pool.GetOrCreateDepthTarget("socket|pass|color", 8, 8);

        pool.PrepareForGraph("graph-a");
        Assert.Same(
            color,
            pool.GetOrCreateRenderTarget<Rgba8, Rgba8>(
                "socket|pass|color",
                8,
                8,
                PixelFormat.Rgba8));
        Assert.Same(depth, pool.GetOrCreateDepthTarget("socket|pass|color", 8, 8));

        var resizedColor = pool.GetOrCreateRenderTarget<Rgba8, Rgba8>(
            "socket|pass|color",
            16,
            8,
            PixelFormat.Rgba8);
        Assert.NotSame(color, resizedColor);
        AssertDisposed(color);

        pool.PrepareForGraph("graph-b");
        AssertDisposed(resizedColor);
        AssertDisposed(depth);

        var replacement = pool.GetOrCreateDepthTarget("socket|pass|color", 8, 8);
        pool.Dispose();
        AssertDisposed(replacement);
        Assert.Throws<ObjectDisposedException>(
            () => pool.GetOrCreateDepthTarget("socket|pass|color", 8, 8));
    }

    private static void AssertDisposed<TPixel, TValue>(GpuTexture2D<TPixel, TValue> texture)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        var pixels = new TPixel[checked(texture.Width * texture.Height)];
        Assert.Throws<ObjectDisposedException>(() => texture.Read(pixels));
    }

    private sealed class LegacyRenderContextBackend : IRenderContextBackend
    {
        public int Width => 1;
        public int Height => 1;
        public SampleCount SampleCount => SampleCount.X1;

        public SceneGeometry GetSceneGeometry(SceneGeometryHandle handle)
            => new(Array.Empty<SceneVertex>(), Array.Empty<uint>());

        public RenderCamera GetCamera(CameraHandle handle)
            => new(float4x4.Identity);

        public ReadOnlyMemory<Rgba8> GetColorInput(TextureHandle handle)
            => ReadOnlyMemory<Rgba8>.Empty;

        public void SetColorOutput(
            TextureHandle handle,
            Rgba8[] pixels,
            DispatchPath dispatchPath)
        {
        }
    }
}
