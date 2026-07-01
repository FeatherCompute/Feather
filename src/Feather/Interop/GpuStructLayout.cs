namespace Feather.Interop;

/// <summary>
/// Describes a managed structure whose fields can be packed into a GPU layout.
/// </summary>
/// <typeparam name="T">The unmanaged structure type.</typeparam>
public interface IGpuStruct<T>
    where T : unmanaged
{
    /// <summary>
    /// Gets the GPU layout metadata for <typeparamref name="T"/>.
    /// </summary>
    static abstract GpuStructLayout Layout { get; }

    /// <summary>
    /// Packs managed values into the GPU byte layout.
    /// </summary>
    /// <param name="source">The managed values to pack.</param>
    /// <param name="destination">The destination byte span. Its length must be at least <c>source.Length * Layout.SizeInBytes</c>.</param>
    static abstract void Pack(ReadOnlySpan<T> source, Span<byte> destination);

    /// <summary>
    /// Unpacks GPU layout bytes into managed values.
    /// </summary>
    /// <param name="source">The GPU byte layout. Its length must be at least <c>destination.Length * Layout.SizeInBytes</c>.</param>
    /// <param name="destination">The managed destination values.</param>
    static abstract void Unpack(ReadOnlySpan<byte> source, Span<T> destination);
}

/// <summary>
/// Describes the GPU memory layout for a generated structure.
/// </summary>
/// <param name="Name">The structure name.</param>
/// <param name="Layout">The GPU layout convention.</param>
/// <param name="Fields">The field layout metadata.</param>
/// <param name="SizeInBytes">The total stride of one structure value in bytes.</param>
/// <param name="Alignment">The structure alignment in bytes.</param>
public sealed record GpuStructLayout(string Name, GpuLayout Layout, IReadOnlyList<GpuStructField> Fields, int SizeInBytes, int Alignment);

/// <summary>
/// Describes one field in a generated GPU structure layout.
/// </summary>
/// <param name="Name">The field or property name.</param>
/// <param name="Type">The managed field type.</param>
/// <param name="Offset">The std430 byte offset.</param>
/// <param name="SizeInBytes">The field size in bytes.</param>
/// <param name="Alignment">The field alignment in bytes.</param>
/// <param name="ArrayLength">The fixed <c>GpuArrayN&lt;T&gt;</c> element count, or zero when the field is not an array.</param>
/// <param name="ArrayStride">The std430 byte stride between fixed array elements, or zero when the field is not an array.</param>
public sealed record GpuStructField(
    string Name,
    Type Type,
    int Offset,
    int SizeInBytes,
    int Alignment,
    int ArrayLength = 0,
    int ArrayStride = 0);
