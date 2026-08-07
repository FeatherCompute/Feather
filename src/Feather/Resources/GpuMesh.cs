namespace Feather.Resources;

/// <summary>
/// A triangle mesh for hardware acceleration: a flat float3 vertex buffer
/// (3N floats) plus a uint index buffer whose length is a multiple of three.
/// The mesh does not own its buffers — they must outlive every acceleration
/// structure built from them.
/// </summary>
public sealed class GpuMesh : IDisposable
{
    private bool disposed;

    internal GpuMesh(GpuBuffer<float> vertices, GpuBuffer<uint> indices)
    {
        Vertices = vertices;
        Indices = indices;
    }

    /// <summary>Gets the flat float3 vertex buffer backing this mesh.</summary>
    public GpuBuffer<float> Vertices { get; }

    /// <summary>Gets the uint triangle index buffer backing this mesh.</summary>
    public GpuBuffer<uint> Indices { get; }

    /// <summary>
    /// Releases this mesh wrapper. The underlying buffers are owned by the
    /// caller and are not disposed.
    /// </summary>
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        disposed = true;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}
