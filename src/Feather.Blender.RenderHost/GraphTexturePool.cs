using Feather.RenderGraph;
using Feather.Resources;

namespace Feather.Blender.RenderHost;

/// <summary>
/// Owns the GPU textures that carry data between render-graph passes and between frames.
///
/// Passes used to exchange images through system memory, which forced a synchronous readback at
/// every pass boundary. Iterative work such as a fluid solver cannot afford that, so the host owns
/// the intermediates instead and only reads back once, when the final frame is published.
/// </summary>
internal sealed class GraphTexturePool : IDisposable
{
    /// <summary>
    /// Textures are keyed by a graph-stable identity rather than by <see cref="TextureHandle"/>.
    /// Handle values are assigned in resolution order and are only meaningful within a single
    /// execution, so keying on them would hand a pass another socket's texture after an edit.
    /// </summary>
    private readonly Dictionary<string, PooledTexture> textures = new(StringComparer.Ordinal);
    private string fingerprint = string.Empty;
    private bool disposed;

    /// <summary>
    /// Discards every pooled texture when the graph topology changes, because socket identities
    /// only remain comparable within one topology.
    /// </summary>
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

    /// <summary>
    /// Reports whether the next <see cref="GetOrCreate{TPixel,TValue}"/> for this identity would
    /// allocate. Callers use this to clear a texture the first time it is handed out, since a
    /// fresh allocation has undefined contents.
    /// </summary>
    public bool IsAllocated(string identity)
        => textures.ContainsKey(identity);

    /// <summary>
    /// Returns the pooled texture for <paramref name="identity"/>, allocating it on first use and
    /// reallocating it when the render size or pixel format changes.
    /// </summary>
    public GpuTexture2D<TPixel, TValue> GetOrCreate<TPixel, TValue>(
        string identity,
        int width,
        int height,
        PixelFormat format)
        where TPixel : unmanaged
        where TValue : unmanaged
    {
        ObjectDisposedException.ThrowIf(disposed, this);
        if (textures.TryGetValue(identity, out var existing))
        {
            if (existing.Texture is GpuTexture2D<TPixel, TValue> reusable &&
                existing.Width == width &&
                existing.Height == height &&
                existing.Format == format)
            {
                return reusable;
            }

            // A resize, a format change, or a different element type invalidates the allocation.
            existing.Lifetime.Dispose();
            textures.Remove(identity);
        }

        // ReadWrite grants both storage and sampled usage in the native layer, so one allocation
        // serves compute writes, shader reads, and the final readback.
        var texture = GPU.CreateTexture2D<TPixel, TValue>(
            width,
            height,
            format,
            TextureAccess.ReadWrite);
        textures.Add(identity, new PooledTexture(texture, texture, width, height, format));
        return texture;
    }

    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        ReleaseAll();
        disposed = true;
    }

    private void ReleaseAll()
    {
        foreach (var pooled in textures.Values)
        {
            pooled.Lifetime.Dispose();
        }
        textures.Clear();
    }

    /// <summary>
    /// Retains the concrete texture as <see cref="IGpuTexture2D"/> so downstream passes can read
    /// its metadata, plus the disposal handle, since the generic texture type is not known here.
    /// </summary>
    private sealed record PooledTexture(
        IGpuTexture2D Texture,
        IDisposable Lifetime,
        int Width,
        int Height,
        PixelFormat Format);
}
