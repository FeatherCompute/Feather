using System.Text;

namespace Feather.Native;

public static class NativeStringCall
{
    public unsafe delegate FeResult Getter(IntPtr buffer, UIntPtr bufferSize, out UIntPtr requiredSize);

    public static unsafe string GetString(Getter getter)
    {
        var result = getter(IntPtr.Zero, UIntPtr.Zero, out var required);
        if (result != FeResult.Ok && required == UIntPtr.Zero)
        {
            NativeMethods.ThrowIfFailed(result);
        }

        var size = checked((int)required) + 1;
        var bytes = new byte[size];
        fixed (byte* ptr = bytes)
        {
            NativeMethods.ThrowIfFailed(getter((IntPtr)ptr, (UIntPtr)bytes.Length, out _));
        }

        var length = Array.IndexOf(bytes, (byte)0);
        if (length < 0)
        {
            length = bytes.Length;
        }

        return Encoding.UTF8.GetString(bytes, 0, length);
    }
}
