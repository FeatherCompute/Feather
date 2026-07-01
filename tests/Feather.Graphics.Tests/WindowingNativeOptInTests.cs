using Feather.Resources;
using Feather.Windowing;

namespace Feather.Graphics.Tests;

public class WindowingNativeOptInTests
{
    [WindowFact]
    public void OptInWindowCreatePollPresentAndDisposePathsDoNotThrow()
    {
        using var window = CreateSmallWindow("Feather Window Test");
        Assert.True(window.Width > 0);
        Assert.True(window.Height > 0);

        window.PollEvents();
        _ = window.TryPollEvent(out _);
        window.SetTitle("Feather Window Test Updated");
        window.SetVSync(false);

        using var pixels = new GpuPixelBuffer(window.Width, window.Height);
        pixels.Clear(GpuColor.Rgba(4, 8, 16));
        window.Present(pixels);

        window.Close();
        Assert.False(window.IsOpen);
    }

    [WindowFact]
    public void OptInTexturePresenterPresentsGpuTextureAndSupportsDisposeOrdering()
    {
        using var window = CreateSmallWindow("Feather Texture Presenter Test");
        using var texture = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(window.Width, window.Height, PixelFormat.Rgba8);
        texture.Upload([.. Enumerable.Repeat(new Rgba32(32, 64, 96, 255), window.Width * window.Height)]);

        var presenter = window.CreateTexturePresenter();
        presenter.Present(texture, PresentMode.Auto);
        presenter.Dispose();
        Assert.Throws<ObjectDisposedException>(() => presenter.Present(texture));

        var presenterDisposedWithWindow = window.CreateTexturePresenter();
        window.Dispose();
        Assert.Throws<ObjectDisposedException>(() => presenterDisposedWithWindow.Present(texture));
    }

    [WindowFact]
    public void OptInPixelPresenterDisposedUsageThrows()
    {
        using var window = CreateSmallWindow("Feather Pixel Presenter Test");
        using var pixels = new GpuPixelBuffer(window.Width, window.Height);
        pixels.Clear(GpuColor.Rgba(12, 24, 48));

        var presenter = window.CreateTexturePresenter();
        presenter.Present(pixels);
        presenter.Dispose();

        Assert.Throws<ObjectDisposedException>(() => presenter.Present(pixels));
    }

    private static GpuWindow CreateSmallWindow(string title)
        => GpuWindow.Create(new()
        {
            Width = 64,
            Height = 64,
            Title = title,
            Visible = false,
            Resizable = false,
            VSync = false,
            HighDpi = false,
            CenterOnCreate = false
        });

    private readonly record struct Rgba32(byte R, byte G, byte B, byte A);
}

public sealed class WindowFactAttribute : FactAttribute
{
    public WindowFactAttribute()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("FEATHER_WINDOW_TESTS"), "1", StringComparison.Ordinal))
        {
            Skip = "Set FEATHER_WINDOW_TESTS=1 to run native window tests.";
        }
    }
}
