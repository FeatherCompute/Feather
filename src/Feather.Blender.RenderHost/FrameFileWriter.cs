using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace Feather.Blender.RenderHost;

internal static class FrameFileWriter
{
    private static ReadOnlySpan<byte> Magic => "FTHRFRM1"u8;
    public const ushort Version = 1;
    public const ushort HeaderSize = 40;
    public const ushort PixelFormatRgba8 = 1;
    public const ushort OriginTopLeft = 2;

    public static void WriteAtomic(string outputPath, ulong frameId, RenderedFrame frame)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);
        if (frame.Width is < 1 or > RenderRequest.MaximumDimension ||
            frame.Height is < 1 or > RenderRequest.MaximumDimension)
        {
            throw new InvalidDataException("Frame dimensions are outside the viewport protocol limits.");
        }

        var rowStride = checked(frame.Width * 4);
        var payloadSize = checked(rowStride * frame.Height);
        if (payloadSize != checked(frame.Pixels.Length * 4) ||
            payloadSize > RenderRequest.MaximumFramePayloadSize)
        {
            throw new InvalidDataException("Frame pixel storage does not match its dimensions.");
        }

        outputPath = Path.GetFullPath(outputPath);
        var directory = Path.GetDirectoryName(outputPath)
            ?? throw new InvalidDataException("Frame output path has no parent directory.");
        Directory.CreateDirectory(directory);
        var temporaryPath = Path.Combine(
            directory,
            $".{Path.GetFileName(outputPath)}.{Environment.ProcessId}.{Guid.NewGuid():N}.tmp");

        try
        {
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                1024 * 1024,
                FileOptions.SequentialScan))
            {
                Span<byte> header = stackalloc byte[HeaderSize];
                Magic.CopyTo(header);
                BinaryPrimitives.WriteUInt16LittleEndian(header[8..10], Version);
                BinaryPrimitives.WriteUInt16LittleEndian(header[10..12], HeaderSize);
                BinaryPrimitives.WriteUInt16LittleEndian(header[12..14], PixelFormatRgba8);
                BinaryPrimitives.WriteUInt16LittleEndian(header[14..16], OriginTopLeft);
                BinaryPrimitives.WriteUInt32LittleEndian(header[16..20], checked((uint)frame.Width));
                BinaryPrimitives.WriteUInt32LittleEndian(header[20..24], checked((uint)frame.Height));
                BinaryPrimitives.WriteUInt32LittleEndian(header[24..28], checked((uint)rowStride));
                BinaryPrimitives.WriteUInt32LittleEndian(header[28..32], checked((uint)payloadSize));
                BinaryPrimitives.WriteUInt64LittleEndian(header[32..40], frameId);
                stream.Write(header);
                stream.Write(MemoryMarshal.AsBytes(frame.Pixels.AsSpan()));
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporaryPath, outputPath, overwrite: true);
        }
        finally
        {
            try
            {
                File.Delete(temporaryPath);
            }
            catch (IOException)
            {
            }
        }
    }
}
