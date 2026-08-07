using Feather.Native;

namespace Feather.Resources;

/// <summary>
/// A GPU-accelerated triangle acceleration structure (BLAS + TLAS) built from
/// vertex and index buffers. The native accel copies the mesh data into its
/// own device memory at creation time; later buffer uploads do not change it.
/// </summary>
public sealed class GpuAccel : IDisposable
{
    private bool disposed;

    private GpuAccel(GpuContext context, FeAccelHandle handle)
    {
        Context = context;
        Handle = handle;
    }

    internal GpuContext Context { get; }

    internal FeAccelHandle Handle { get; }

    /// <summary>
    /// Creates an acceleration structure from one or more triangle meshes.
    /// Each mesh must have a float3 vertex buffer and a uint index buffer
    /// whose length is a multiple of three.
    /// </summary>
    public static GpuAccel Create(GpuContext context, params (GpuBuffer<float> vertices, GpuBuffer<uint> indices)[] meshes)
    {
        ArgumentNullException.ThrowIfNull(context);
        context.ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(meshes.Length);

        var descs = new NativeMethods.FeAccelMeshDesc[meshes.Length];
        for (var i = 0; i < meshes.Length; i++)
        {
            if (meshes[i].vertices is null || meshes[i].indices is null)
            {
                throw new ArgumentException("Accel meshes require non-null vertex and index buffers.", nameof(meshes));
            }
            if (meshes[i].vertices.ElementStride != 4 || meshes[i].vertices.Length % 3 != 0)
            {
                throw new ArgumentException("Accel vertex buffers must be flat float3 (3N floats).", nameof(meshes));
            }
            if (meshes[i].indices.ElementStride != 4 || meshes[i].indices.Length % 3 != 0)
            {
                throw new ArgumentException("Accel index buffers must be uint triplets.", nameof(meshes));
            }
            descs[i] = new NativeMethods.FeAccelMeshDesc
            {
                VertexBuffer = meshes[i].vertices.Handle.RawValue,
                IndexBuffer = meshes[i].indices.Handle.RawValue,
            };
        }

        NativeMethods.ThrowIfFailed(NativeMethods.fe_accel_create(context.Handle, (uint)descs.Length, descs, out var handle));
        return new GpuAccel(context, handle);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }
        if (!Handle.IsInvalid)
        {
            Handle.Dispose();
        }
        disposed = true;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(disposed, this);
}
