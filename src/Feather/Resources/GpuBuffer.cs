using Feather.Native;
using Feather.Interop;

namespace Feather.Resources;

/// <summary>
/// Owns a typed EasyGPU storage buffer.
/// </summary>
/// <typeparam name="T">The unmanaged shader element type.</typeparam>
public sealed class GpuBuffer<T> : IDisposable
    where T : unmanaged
{
    private bool disposed;

    private GpuBuffer(GpuContext context, FeBufferHandle handle, int length, BufferAccess access)
    {
        Context = context;
        Handle = handle;
        Length = length;
        Access = access;
    }

    internal GpuContext Context { get; }
    internal FeBufferHandle Handle { get; }

    /// <summary>
    /// Gets the number of logical elements in the buffer.
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// Gets the GPU-side byte size of the buffer.
    /// </summary>
    public int SizeInBytes => checked(Length * ElementStride);

    /// <summary>
    /// Gets the buffer access mode.
    /// </summary>
    public BufferAccess Access { get; }

    /// <summary>
    /// Gets the EasyGPU std430 array stride for one element.
    /// </summary>
    internal int ElementStride => GpuValueLayout<T>.BufferElementStride;

    /// <summary>
    /// Creates an empty typed GPU buffer.
    /// </summary>
    /// <param name="context">The GPU context that owns the buffer.</param>
    /// <param name="count">The number of logical elements.</param>
    /// <param name="access">The shader access mode.</param>
    /// <returns>The created buffer.</returns>
    public static GpuBuffer<T> Create(GpuContext context, int count, BufferAccess access)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var desc = new FeBufferDesc((ulong)checked(count * GpuValueLayout<T>.BufferElementStride), (uint)access, (uint)GpuValueLayout<T>.BufferElementStride);
        NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_create(context.Handle, in desc, IntPtr.Zero, out var handle));
        return new GpuBuffer<T>(context, handle, count, access);
    }

    /// <summary>
    /// Creates a typed GPU buffer and uploads initial data.
    /// </summary>
    /// <param name="context">The GPU context that owns the buffer.</param>
    /// <param name="data">The initial CPU values.</param>
    /// <param name="access">The shader access mode.</param>
    /// <returns>The created buffer.</returns>
    public static GpuBuffer<T> Create(GpuContext context, ReadOnlySpan<T> data, BufferAccess access)
    {
        var buffer = Create(context, data.Length, access);
        buffer.Upload(data);
        return buffer;
    }

    /// <summary>
    /// Creates a read-only shader binding for this buffer.
    /// </summary>
    /// <returns>The read-only binding view.</returns>
    public ReadOnlyBuffer<T> AsReadOnly() => new(Handle, Length);

    /// <summary>
    /// Creates a write-only shader binding for this buffer.
    /// </summary>
    /// <returns>The write-only binding view.</returns>
    public WriteOnlyBuffer<T> AsWriteOnly() => new(Handle, Length);

    /// <summary>
    /// Creates a read-write shader binding for this buffer.
    /// </summary>
    /// <returns>The read-write binding view.</returns>
    public ReadWriteBuffer<T> AsReadWrite() => new(Handle, Length);

    /// <summary>
    /// Uploads CPU values starting at the first element.
    /// </summary>
    /// <param name="data">The CPU values to upload.</param>
    public void Upload(ReadOnlySpan<T> data) => Upload(0, data);

    /// <summary>
    /// Uploads CPU values into a subrange of the buffer.
    /// </summary>
    /// <param name="startIndex">The first logical element to replace.</param>
    /// <param name="data">The CPU values to upload.</param>
    public unsafe void Upload(int startIndex, ReadOnlySpan<T> data)
    {
        ThrowIfDisposed();
        ArgumentOutOfRangeException.ThrowIfNegative(startIndex);
        if (startIndex + data.Length > Length)
        {
            throw new ArgumentOutOfRangeException(nameof(data), "Upload range exceeds buffer length.");
        }

        if (GpuValueLayout<T>.RequiresBufferRepacking)
        {
            var gpuBytes = new byte[checked(data.Length * GpuValueLayout<T>.BufferElementStride)];
            GpuValueLayout<T>.PackBuffer(data, gpuBytes);
            fixed (byte* ptr = gpuBytes)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_upload(Handle, (ulong)checked(startIndex * GpuValueLayout<T>.BufferElementStride), (ulong)gpuBytes.Length, (IntPtr)ptr));
            }

            return;
        }

        fixed (T* ptr = data)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_upload(Handle, (ulong)checked(startIndex * GpuValueLayout<T>.BufferElementStride), (ulong)checked(data.Length * GpuValueLayout<T>.BufferElementStride), (IntPtr)ptr));
        }
    }

    /// <summary>
    /// Reads all logical elements back into CPU layout.
    /// </summary>
    /// <param name="destination">The destination CPU span.</param>
    public unsafe void Read(Span<T> destination)
    {
        ThrowIfDisposed();
        if (destination.Length < Length)
        {
            throw new ArgumentException("Destination span is shorter than the buffer.", nameof(destination));
        }

        if (GpuValueLayout<T>.RequiresBufferRepacking)
        {
            var gpuBytes = new byte[SizeInBytes];
            fixed (byte* ptr = gpuBytes)
            {
                NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_download(Handle, 0, (ulong)gpuBytes.Length, (IntPtr)ptr));
            }

            GpuValueLayout<T>.UnpackBuffer(gpuBytes, destination[..Length]);
            return;
        }

        fixed (T* ptr = destination)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_buffer_download(Handle, 0, (ulong)SizeInBytes, (IntPtr)ptr));
        }
    }

    /// <summary>
    /// Reads all logical elements into a new CPU array.
    /// </summary>
    /// <returns>The readback values.</returns>
    public T[] ToArray()
    {
        var data = new T[Length];
        Read(data);
        return data;
    }

    /// <summary>
    /// Releases the native buffer handle.
    /// </summary>
    public void Dispose()
    {
        if (disposed)
        {
            return;
        }

        Handle.Dispose();
        disposed = true;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(disposed, this);
    }
}

/// <summary>
/// Read-only shader view over a GPU buffer.
/// </summary>
/// <typeparam name="T">The unmanaged shader element type.</typeparam>
public readonly struct ReadOnlyBuffer<T> : IGpuBufferBinding
    where T : unmanaged
{
    internal ReadOnlyBuffer(FeBufferHandle handle, int length)
    {
        Handle = handle;
        Length = length;
    }

    internal FeBufferHandle Handle { get; }

    /// <summary>
    /// Gets the number of logical elements visible to the shader.
    /// </summary>
    public int Length { get; }
    FeBufferHandle IGpuBufferBinding.NativeBufferHandle => Handle;

    /// <summary>
    /// Gets a shader-only indexed value.
    /// </summary>
    /// <param name="index">The logical element index.</param>
    /// <returns>A marker value consumed by the generator.</returns>
    public T this[int index] => ShaderRuntimeMarker<T>.Value;
}

/// <summary>
/// Write-only shader view over a GPU buffer.
/// </summary>
/// <typeparam name="T">The unmanaged shader element type.</typeparam>
public readonly struct WriteOnlyBuffer<T> : IGpuBufferBinding
    where T : unmanaged
{
    internal WriteOnlyBuffer(FeBufferHandle handle, int length)
    {
        Handle = handle;
        Length = length;
    }

    internal FeBufferHandle Handle { get; }

    /// <summary>
    /// Gets the number of logical elements visible to the shader.
    /// </summary>
    public int Length { get; }
    FeBufferHandle IGpuBufferBinding.NativeBufferHandle => Handle;

    /// <summary>
    /// Sets a shader-only indexed value.
    /// </summary>
    /// <param name="index">The logical element index.</param>
    public T this[int index]
    {
        set => _ = value;
    }
}

/// <summary>
/// Read-write shader view over a GPU buffer.
/// </summary>
/// <typeparam name="T">The unmanaged shader element type.</typeparam>
public readonly struct ReadWriteBuffer<T> : IGpuBufferBinding
    where T : unmanaged
{
    internal ReadWriteBuffer(FeBufferHandle handle, int length)
    {
        Handle = handle;
        Length = length;
    }

    internal FeBufferHandle Handle { get; }

    /// <summary>
    /// Gets the number of logical elements visible to the shader.
    /// </summary>
    public int Length { get; }
    FeBufferHandle IGpuBufferBinding.NativeBufferHandle => Handle;

    /// <summary>
    /// Gets a shader-only indexed value by reference.
    /// </summary>
    /// <param name="index">The logical element index.</param>
    /// <returns>A marker reference consumed by the generator.</returns>
    public ref T this[int index] => ref ShaderRuntimeMarker<T>.RefValue;
}
