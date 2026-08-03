using Feather.Native;

namespace Feather.RenderGraph;

/// <summary>Host-side resampling for display-ready RGBA8 frames.</summary>
public static class Rgba8Resampling
{
    public static unsafe void BilinearUpscale(
        ReadOnlySpan<Rgba8> source,
        int sourceWidth,
        int sourceHeight,
        Span<Rgba8> destination,
        int width,
        int height)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 || width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Image dimensions must be positive.");
        }
        if (source.Length != checked(sourceWidth * sourceHeight))
        {
            throw new ArgumentException("Source length does not match its dimensions.", nameof(source));
        }
        if (destination.Length != checked(width * height))
        {
            throw new ArgumentException("Destination length does not match its dimensions.", nameof(destination));
        }

        fixed (Rgba8* sourcePointer = source)
        fixed (Rgba8* destinationPointer = destination)
        {
            NativeMethods.ThrowIfFailed(NativeMethods.fe_bilinear_upscale_rgba8(
                (IntPtr)sourcePointer,
                (uint)sourceWidth,
                (uint)sourceHeight,
                (IntPtr)destinationPointer,
                (uint)width,
                (uint)height));
        }
    }
}
