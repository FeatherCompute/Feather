using Feather.RenderGraph;
using Feather.Resources;

namespace Feather.Blender.RenderHost;

/// <summary>
/// One frame of history for a History Read/Write pair.
///
/// History used to be a CPU pixel array, which forced a readback every frame even when both the
/// producer and the consumer were GPU passes. An entry now carries either a CPU frame, for software
/// passes, or a GPU-resident texture that stays on the device between frames.
/// </summary>
internal sealed class GraphHistoryEntry
{
    private GraphHistoryEntry(RenderedFrame? frame, IGpuTexture2D? texture, int width, int height)
    {
        Frame = frame;
        Texture = texture;
        Width = width;
        Height = height;
    }

    public RenderedFrame? Frame { get; }

    public IGpuTexture2D? Texture { get; }

    public int Width { get; }

    public int Height { get; }

    public bool IsGpuResident => Texture is not null;

    public static GraphHistoryEntry FromFrame(RenderedFrame frame)
        => new(frame, null, frame.Width, frame.Height);

    public static GraphHistoryEntry FromTexture(IGpuTexture2D texture)
        => new(null, texture, texture.Width, texture.Height);
}
