using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Feather.Math;

namespace Feather.Interop;

/// <summary>
/// Provides helpers for byte offsets that follow EasyGPU's GPU layout rules.
/// </summary>
public static class GpuValueLayout
{
    /// <summary>
    /// Aligns an offset to the next address accepted by the GPU layout.
    /// </summary>
    /// <param name="offset">The unaligned byte offset.</param>
    /// <param name="alignment">The required byte alignment.</param>
    /// <returns>The aligned byte offset.</returns>
    public static int AlignOffset(int offset, int alignment)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(offset);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(alignment);

        var remainder = offset % alignment;
        return remainder == 0 ? offset : checked(offset + alignment - remainder);
    }
}

/// <summary>
/// Describes the EasyGPU-compatible byte layout for a managed GPU value type.
/// </summary>
/// <typeparam name="T">The unmanaged managed value type.</typeparam>
public static class GpuValueLayout<T>
    where T : unmanaged
{
    private static readonly IGpuValueLayout<T> Layout = CreateLayout();

    /// <summary>
    /// Gets the managed CPU size of <typeparamref name="T"/>.
    /// </summary>
    public static int CpuSizeInBytes => Layout.CpuSizeInBytes;

    /// <summary>
    /// Gets the byte stride used for <typeparamref name="T"/> inside an EasyGPU <c>layout(std430)</c> buffer array.
    /// </summary>
    public static int BufferElementStride => Layout.BufferElementStride;

    /// <summary>
    /// Gets the byte size used when <typeparamref name="T"/> is stored as a struct field or push constant.
    /// </summary>
    public static int FieldSizeInBytes => Layout.FieldSizeInBytes;

    /// <summary>
    /// Gets the byte alignment used when <typeparamref name="T"/> is stored as a struct field or push constant.
    /// </summary>
    public static int Alignment => Layout.Alignment;

    /// <summary>
    /// Gets a value indicating whether buffer upload/download must convert between CPU and EasyGPU layout.
    /// </summary>
    public static bool RequiresBufferRepacking => BufferElementStride != CpuSizeInBytes || Layout.RequiresBufferRepacking;

    /// <summary>
    /// Packs CPU values into the EasyGPU buffer-array layout.
    /// </summary>
    /// <param name="source">The source CPU values.</param>
    /// <param name="destination">The destination GPU byte span.</param>
    public static void PackBuffer(ReadOnlySpan<T> source, Span<byte> destination)
    {
        var required = checked(source.Length * BufferElementStride);
        if (destination.Length < required)
        {
            throw new ArgumentException("Destination span is too small for the EasyGPU buffer layout.", nameof(destination));
        }

        Layout.PackBuffer(source, destination[..required]);
    }

    /// <summary>
    /// Unpacks EasyGPU buffer-array bytes into CPU values.
    /// </summary>
    /// <param name="source">The source GPU byte span.</param>
    /// <param name="destination">The destination CPU values.</param>
    public static void UnpackBuffer(ReadOnlySpan<byte> source, Span<T> destination)
    {
        var required = checked(destination.Length * BufferElementStride);
        if (source.Length < required)
        {
            throw new ArgumentException("Source span is too small for the EasyGPU buffer layout.", nameof(source));
        }

        Layout.UnpackBuffer(source[..required], destination);
    }

    /// <summary>
    /// Packs one CPU value into its EasyGPU struct-field or push-constant layout.
    /// </summary>
    /// <param name="value">The source CPU value.</param>
    /// <param name="destination">The destination GPU byte span.</param>
    public static void PackValue(in T value, Span<byte> destination)
    {
        if (destination.Length < FieldSizeInBytes)
        {
            throw new ArgumentException("Destination span is too small for the EasyGPU value layout.", nameof(destination));
        }

        Layout.PackValue(in value, destination[..FieldSizeInBytes]);
    }

    /// <summary>
    /// Unpacks one EasyGPU struct-field or push-constant value into CPU layout.
    /// </summary>
    /// <param name="source">The source GPU byte span.</param>
    /// <returns>The unpacked CPU value.</returns>
    public static T UnpackValue(ReadOnlySpan<byte> source)
    {
        if (source.Length < FieldSizeInBytes)
        {
            throw new ArgumentException("Source span is too small for the EasyGPU value layout.", nameof(source));
        }

        return Layout.UnpackValue(source[..FieldSizeInBytes]);
    }

    private static IGpuValueLayout<T> CreateLayout()
    {
        if (EasyGpuBuiltinLayout<T>.TryCreate(out var builtin))
        {
            return builtin;
        }

        if (typeof(IGpuStruct<T>).IsAssignableFrom(typeof(T)))
        {
            return (IGpuValueLayout<T>)Activator.CreateInstance(typeof(GpuStructValueLayout<>).MakeGenericType(typeof(T)))!;
        }

        var cpuSize = Unsafe.SizeOf<T>();
        return new BlittableGpuValueLayout<T>(cpuSize, cpuSize, cpuSize, System.Math.Min(cpuSize, 16));
    }
}

internal interface IGpuValueLayout<T>
    where T : unmanaged
{
    int CpuSizeInBytes { get; }

    int BufferElementStride { get; }

    int FieldSizeInBytes { get; }

    int Alignment { get; }

    bool RequiresBufferRepacking { get; }

    void PackBuffer(ReadOnlySpan<T> source, Span<byte> destination);

    void UnpackBuffer(ReadOnlySpan<byte> source, Span<T> destination);

    void PackValue(in T value, Span<byte> destination);

    T UnpackValue(ReadOnlySpan<byte> source);
}

internal sealed class BlittableGpuValueLayout<T> : IGpuValueLayout<T>
    where T : unmanaged
{
    private readonly int bufferCopySize;
    private readonly int valueCopySize;

    public BlittableGpuValueLayout(
        int cpuSizeInBytes,
        int bufferElementStride,
        int fieldSizeInBytes,
        int alignment,
        int? bufferCopySize = null,
        int? valueCopySize = null)
    {
        CpuSizeInBytes = cpuSizeInBytes;
        BufferElementStride = bufferElementStride;
        FieldSizeInBytes = fieldSizeInBytes;
        Alignment = alignment;
        this.bufferCopySize = bufferCopySize ?? System.Math.Min(CpuSizeInBytes, BufferElementStride);
        this.valueCopySize = valueCopySize ?? System.Math.Min(CpuSizeInBytes, FieldSizeInBytes);
    }

    public int CpuSizeInBytes { get; }

    public int BufferElementStride { get; }

    public int FieldSizeInBytes { get; }

    public int Alignment { get; }

    public bool RequiresBufferRepacking => BufferElementStride != CpuSizeInBytes;

    public void PackBuffer(ReadOnlySpan<T> source, Span<byte> destination)
    {
        if (!RequiresBufferRepacking)
        {
            MemoryMarshal.AsBytes(source).CopyTo(destination);
            return;
        }

        // EasyGPU Runtime/Buffer.h writes arrays using Std430Converter<T>::GetGPULayoutSize().
        // For vec3/ivec3 this means a 12-byte CPU payload in a 16-byte array slot.
        destination.Clear();
        for (var i = 0; i < source.Length; i++)
        {
            MemoryMarshal.AsBytes(source.Slice(i, 1))[..bufferCopySize]
                .CopyTo(destination.Slice(checked(i * BufferElementStride), bufferCopySize));
        }
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<T> destination)
    {
        if (!RequiresBufferRepacking)
        {
            source.CopyTo(MemoryMarshal.AsBytes(destination));
            return;
        }

        var destinationBytes = MemoryMarshal.AsBytes(destination);
        for (var i = 0; i < destination.Length; i++)
        {
            source.Slice(checked(i * BufferElementStride), bufferCopySize)
                .CopyTo(destinationBytes.Slice(checked(i * CpuSizeInBytes), bufferCopySize));
        }
    }

    public void PackValue(in T value, Span<byte> destination)
    {
        destination.Clear();
        MemoryMarshal.AsBytes(MemoryMarshal.CreateReadOnlySpan(ref Unsafe.AsRef(in value), 1))
            .Slice(0, valueCopySize)
            .CopyTo(destination);
    }

    public T UnpackValue(ReadOnlySpan<byte> source)
    {
        var value = default(T);
        source[..valueCopySize].CopyTo(MemoryMarshal.AsBytes(MemoryMarshal.CreateSpan(ref value, 1)));
        return value;
    }
}

internal sealed class GpuStructValueLayout<T> : IGpuValueLayout<T>
    where T : unmanaged, IGpuStruct<T>
{
    private static readonly GpuStructLayout StructLayout = T.Layout;

    public int CpuSizeInBytes => Unsafe.SizeOf<T>();

    public int BufferElementStride => StructLayout.SizeInBytes;

    public int FieldSizeInBytes => StructLayout.SizeInBytes;

    public int Alignment => StructLayout.Alignment;

    public bool RequiresBufferRepacking => true;

    public void PackBuffer(ReadOnlySpan<T> source, Span<byte> destination)
    {
        destination.Clear();
        T.Pack(source, destination);
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<T> destination)
        => T.Unpack(source, destination);

    public void PackValue(in T value, Span<byte> destination)
    {
        var local = value;
        destination.Clear();
        T.Pack(MemoryMarshal.CreateReadOnlySpan(ref local, 1), destination);
    }

    public T UnpackValue(ReadOnlySpan<byte> source)
    {
        Span<T> value = stackalloc T[1];
        T.Unpack(source, value);
        return value[0];
    }
}

internal static class EasyGpuBuiltinLayout<T>
    where T : unmanaged
{
    public static bool TryCreate(out IGpuValueLayout<T> layout)
    {
        var type = typeof(T);
        var cpuSize = Unsafe.SizeOf<T>();

        // These sizes mirror EasyGPU/include/Utility/Meta/Std430Layout.h for buffer arrays
        // and EasyGPU/include/Utility/Meta/StructMeta.h for fields and push constants.
        if (type == typeof(float) || type == typeof(int) || type == typeof(uint))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 4, 4, 4);
            return true;
        }

        if (type == typeof(bool))
        {
            layout = (IGpuValueLayout<T>)(object)new BoolGpuValueLayout();
            return true;
        }

        if (type == typeof(bool2))
        {
            layout = (IGpuValueLayout<T>)(object)new Bool2GpuValueLayout();
            return true;
        }

        if (type == typeof(bool3))
        {
            layout = (IGpuValueLayout<T>)(object)new Bool3GpuValueLayout();
            return true;
        }

        if (type == typeof(bool4))
        {
            layout = (IGpuValueLayout<T>)(object)new Bool4GpuValueLayout();
            return true;
        }

        if (type == typeof(float2) || type == typeof(int2))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 8, 8, 8);
            return true;
        }

        if (type == typeof(float3) || type == typeof(int3))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 16, 12, 16);
            return true;
        }

        if (type == typeof(float4) || type == typeof(int4))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 16, 16, 16);
            return true;
        }

        if (type == typeof(float2x2))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 32, 32, 16, bufferCopySize: 16, valueCopySize: 16);
            return true;
        }

        if (type == typeof(float3x3))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 48, 48, 16, bufferCopySize: 36, valueCopySize: 36);
            return true;
        }

        if (type == typeof(float4x4))
        {
            layout = new BlittableGpuValueLayout<T>(cpuSize, 64, 64, 16);
            return true;
        }

        layout = null!;
        return false;
    }
}

internal sealed class BoolGpuValueLayout : IGpuValueLayout<bool>
{
    public int CpuSizeInBytes => Unsafe.SizeOf<bool>();

    public int BufferElementStride => 4;

    public int FieldSizeInBytes => 4;

    public int Alignment => 4;

    public bool RequiresBufferRepacking => true;

    public void PackBuffer(ReadOnlySpan<bool> source, Span<byte> destination)
    {
        destination.Clear();
        for (var i = 0; i < source.Length; i++)
        {
            BoolLayoutHelpers.WriteBool32(destination.Slice(checked(i * BufferElementStride), FieldSizeInBytes), source[i]);
        }
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<bool> destination)
    {
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = BoolLayoutHelpers.ReadBool32(source.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void PackValue(in bool value, Span<byte> destination)
    {
        destination.Clear();
        BoolLayoutHelpers.WriteBool32(destination, value);
    }

    public bool UnpackValue(ReadOnlySpan<byte> source)
        => BoolLayoutHelpers.ReadBool32(source);
}

internal sealed class Bool2GpuValueLayout : IGpuValueLayout<bool2>
{
    public int CpuSizeInBytes => Unsafe.SizeOf<bool2>();

    public int BufferElementStride => 8;

    public int FieldSizeInBytes => 8;

    public int Alignment => 8;

    public bool RequiresBufferRepacking => true;

    public void PackBuffer(ReadOnlySpan<bool2> source, Span<byte> destination)
    {
        destination.Clear();
        for (var i = 0; i < source.Length; i++)
        {
            Pack(source[i], destination.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<bool2> destination)
    {
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = Unpack(source.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void PackValue(in bool2 value, Span<byte> destination)
    {
        destination.Clear();
        Pack(value, destination);
    }

    public bool2 UnpackValue(ReadOnlySpan<byte> source)
        => Unpack(source);

    private static void Pack(bool2 value, Span<byte> destination)
    {
        BoolLayoutHelpers.WriteBool32(destination.Slice(0, 4), value.X);
        BoolLayoutHelpers.WriteBool32(destination.Slice(4, 4), value.Y);
    }

    private static bool2 Unpack(ReadOnlySpan<byte> source)
        => new(
            BoolLayoutHelpers.ReadBool32(source.Slice(0, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(4, 4)));
}

internal sealed class Bool3GpuValueLayout : IGpuValueLayout<bool3>
{
    public int CpuSizeInBytes => Unsafe.SizeOf<bool3>();

    public int BufferElementStride => 16;

    public int FieldSizeInBytes => 12;

    public int Alignment => 16;

    public bool RequiresBufferRepacking => true;

    public void PackBuffer(ReadOnlySpan<bool3> source, Span<byte> destination)
    {
        destination.Clear();
        for (var i = 0; i < source.Length; i++)
        {
            Pack(source[i], destination.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<bool3> destination)
    {
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = Unpack(source.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void PackValue(in bool3 value, Span<byte> destination)
    {
        destination.Clear();
        Pack(value, destination);
    }

    public bool3 UnpackValue(ReadOnlySpan<byte> source)
        => Unpack(source);

    private static void Pack(bool3 value, Span<byte> destination)
    {
        BoolLayoutHelpers.WriteBool32(destination.Slice(0, 4), value.X);
        BoolLayoutHelpers.WriteBool32(destination.Slice(4, 4), value.Y);
        BoolLayoutHelpers.WriteBool32(destination.Slice(8, 4), value.Z);
    }

    private static bool3 Unpack(ReadOnlySpan<byte> source)
        => new(
            BoolLayoutHelpers.ReadBool32(source.Slice(0, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(4, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(8, 4)));
}

internal sealed class Bool4GpuValueLayout : IGpuValueLayout<bool4>
{
    public int CpuSizeInBytes => Unsafe.SizeOf<bool4>();

    public int BufferElementStride => 16;

    public int FieldSizeInBytes => 16;

    public int Alignment => 16;

    public bool RequiresBufferRepacking => true;

    public void PackBuffer(ReadOnlySpan<bool4> source, Span<byte> destination)
    {
        destination.Clear();
        for (var i = 0; i < source.Length; i++)
        {
            Pack(source[i], destination.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void UnpackBuffer(ReadOnlySpan<byte> source, Span<bool4> destination)
    {
        for (var i = 0; i < destination.Length; i++)
        {
            destination[i] = Unpack(source.Slice(checked(i * BufferElementStride), FieldSizeInBytes));
        }
    }

    public void PackValue(in bool4 value, Span<byte> destination)
    {
        destination.Clear();
        Pack(value, destination);
    }

    public bool4 UnpackValue(ReadOnlySpan<byte> source)
        => Unpack(source);

    private static void Pack(bool4 value, Span<byte> destination)
    {
        BoolLayoutHelpers.WriteBool32(destination.Slice(0, 4), value.X);
        BoolLayoutHelpers.WriteBool32(destination.Slice(4, 4), value.Y);
        BoolLayoutHelpers.WriteBool32(destination.Slice(8, 4), value.Z);
        BoolLayoutHelpers.WriteBool32(destination.Slice(12, 4), value.W);
    }

    private static bool4 Unpack(ReadOnlySpan<byte> source)
        => new(
            BoolLayoutHelpers.ReadBool32(source.Slice(0, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(4, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(8, 4)),
            BoolLayoutHelpers.ReadBool32(source.Slice(12, 4)));
}

internal static class BoolLayoutHelpers
{
    public static void WriteBool32(Span<byte> destination, bool value)
        => BinaryPrimitives.WriteInt32LittleEndian(destination, value ? 1 : 0);

    public static bool ReadBool32(ReadOnlySpan<byte> source)
        => BinaryPrimitives.ReadInt32LittleEndian(source) != 0;
}
