using Feather.Windowing;

using var window = GpuWindow.Create(new()
{
    Width = 640,
    Height = 360,
    Title = "Feather CPU Pixels"
});
using var pixels = new GpuPixelBuffer(window.Width, window.Height);

var frame = 0;
while (window.IsOpen)
{
    window.PollEvents();
    while (window.TryPollEvent(out var windowEvent))
    {
        if (windowEvent is WindowKeyEvent { Key: WindowKey.Escape, Pressed: true })
        {
            window.Close();
        }
    }

    for (var y = 0; y < pixels.Height; y++)
    {
        for (var x = 0; x < pixels.Width; x++)
        {
            var r = (byte)((x + frame) & 255);
            var g = (byte)((y * 2 + frame) & 255);
            var b = (byte)((x + y + frame * 3) & 255);
            pixels.SetPixel(x, y, GpuColor.Rgba(r, g, b));
        }
    }

    window.Present(pixels);
    frame++;
}
