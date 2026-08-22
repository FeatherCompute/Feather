namespace Feather.Native;

public static class NativeByteCall
{
    public unsafe delegate FeResult Getter(
        IntPtr buffer,
        UIntPtr bufferSize,
        out UIntPtr requiredSize,
        out FeShaderBinaryFormat format);

    public static unsafe (FeShaderBinaryFormat Format, byte[] Bytes) GetBytes(Getter getter)
    {
        var result = getter(IntPtr.Zero, UIntPtr.Zero, out var required, out var format);
        NativeMethods.ThrowIfFailed(result);

        var size = checked((int)required);
        if (size == 0)
        {
            return (format, []);
        }

        var bytes = new byte[size];
        fixed (byte* pointer = bytes)
        {
            NativeMethods.ThrowIfFailed(
                getter((IntPtr)pointer, (UIntPtr)bytes.Length, out var copiedSize, out var copiedFormat));
            if (copiedSize != required || copiedFormat != format)
            {
                throw new InvalidOperationException("Native shader binary changed during a point-in-time inspection.");
            }
        }

        return (format, bytes);
    }
}
