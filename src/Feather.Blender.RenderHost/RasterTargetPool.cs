using Feather.Resources;

namespace Feather.Blender.RenderHost;

/// <summary>
/// Owns render and depth targets used by project raster passes across executions.
/// </summary>
/// <remarks>
/// The pool follows the lifetime of one project assembly generation. This keeps targets alive
/// between frames without letting resources created for an old pass assembly survive a reload.
/// Graph-stable texture identities keep allocations attached to the output that owns them rather
/// than to execution-local handle values.
/// </remarks>
internal sealed class RasterTargetPool : IDisposable
{
    private readonly Dictionary<TargetKey, PooledTarget> targets = new();
    private string fingerprint = string.Empty;
    private bool disposed;

    public void PrepareForGraph(string graphFingerprint)
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (string.Equals(fingerprint, graphFingerprint, StringComparison.Ordinal))
        {
            return;
        }

        ReleaseAll();
        fingerprint = graphFingerprint;
    }

    public GpuTexture2D<TPixel, TValue> GetOrCreateRenderTarget<TPixel, TValue>(
        string identity,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
        => GetOrCreate<TPixel, TValue>(
            new TargetKey(identity, TextureAccess.RenderTarget),
            width,
            height,
            format,
            TextureAccess.RenderTarget);

    public GpuTexture2D<float, float> GetOrCreateDepthTarget(
        string identity,
        int width,
        int height)
        => GetOrCreate<float, float>(
            new TargetKey(identity, TextureAccess.DepthStencil),
            width,
            height,
            PixelFormat.Depth32Float,
            TextureAccess.DepthStencil);

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        ReleaseAll();
        disposed = true;
    }

    private GpuTexture2D<TPixel, TValue> GetOrCreate<TPixel, TValue>(
        TargetKey key,
        int width,
        int height,
        PixelFormat format,
        TextureAccess access)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (targets.TryGetValue(key, out var existing))
        {
            if (existing.Texture is GpuTexture2D<TPixel, TValue> reusable &&
                reusable.Width == width &&
                reusable.Height == height &&
                reusable.Format == format &&
                reusable.Access == access)
            {
                return reusable;
            }

            existing.Lifetime.Dispose();
            targets.Remove(key);
        }

        var target = GPU.CreateTexture2D<TPixel, TValue>(width, height, format, access);
        targets.Add(key, new PooledTarget(target, target));
        return target;
    }

    private void ReleaseAll()
    {
        foreach (var target in targets.Values)
        {
            target.Lifetime.Dispose();
        }
        targets.Clear();
    }

    private readonly record struct TargetKey(string Identity, TextureAccess Access);

    private sealed record PooledTarget(IGpuTexture2D Texture, IDisposable Lifetime);
}
