using Feather.Windowing;

using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 480,
    Title = "Feather Window Hello",
    VSync = true,
    Resizable = true
});

while (window.IsOpen)
{
    window.PollEvents();
    while (window.TryPollEvent(out var windowEvent))
    {
        Console.WriteLine(windowEvent);
        if (windowEvent is WindowKeyEvent { Key: WindowKey.Escape, Pressed: true })
        {
            window.Close();
        }
    }
}
