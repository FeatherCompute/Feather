namespace Feather.Windowing;

public readonly record struct GpuWindowOptions
{
    public int Width { get; init; } = 1280;
    public int Height { get; init; } = 720;
    public string Title { get; init; } = "Feather";
    public bool Resizable { get; init; } = true;
    public bool Visible { get; init; } = true;
    public bool VSync { get; init; } = true;
    public bool HighDpi { get; init; } = true;
    public bool CenterOnCreate { get; init; } = true;

    public GpuWindowOptions()
    {
    }
}

public enum PresentMode : uint
{
    Auto = 0,
    CopyToCpu = 1,
    Direct = 2
}

public static class GpuColor
{
    public static uint Rgba(byte r, byte g, byte b, byte a = 255)
        => (uint)r | ((uint)g << 8) | ((uint)b << 16) | ((uint)a << 24);
}
