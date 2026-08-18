using Feather.Native;

namespace Feather.Resources;

internal static class NativeHandleExtensions
{
    internal static FeBufferHandle GetNativeHandle<T>(this GpuBuffer<T> buffer)
        where T : unmanaged
        => buffer.Handle;

    internal static FeTextureHandle GetNativeHandle<TPixel, TValue>(this GpuTexture2D<TPixel, TValue> texture)
        where TPixel : unmanaged
        where TValue : unmanaged
        => texture.Handle;

    internal static FeTextureHandle GetNativeHandle<TPixel, TValue>(this GpuTexture3D<TPixel, TValue> texture)
        where TPixel : unmanaged
        where TValue : unmanaged
        => texture.Handle;
}
