using Feather.Windowing;

namespace Feather.Graphics.Tests;

public class WindowingSurfaceTests
{
    [Fact]
    public void PixelBufferSupportsClearSetGetAndResize()
    {
        using var pixels = new GpuPixelBuffer(2, 2);

        pixels.Clear(GpuColor.Rgba(1, 2, 3, 4));
        Assert.All(pixels.Pixels.ToArray(), pixel => Assert.Equal(GpuColor.Rgba(1, 2, 3, 4), pixel));

        pixels.SetPixel(1, 0, GpuColor.Rgba(9, 8, 7));
        Assert.Equal(GpuColor.Rgba(9, 8, 7), pixels.GetPixel(1, 0));

        pixels.Resize(1, 3);
        Assert.Equal(1, pixels.Width);
        Assert.Equal(3, pixels.Height);
        Assert.Equal(3, pixels.Pixels.Length);
    }

    [Fact]
    public void WindowOptionsDefaultToVisibleResizableVSyncWindow()
    {
        var options = new GpuWindowOptions();

        Assert.Equal(1280, options.Width);
        Assert.Equal(720, options.Height);
        Assert.Equal("Feather", options.Title);
        Assert.True(options.Visible);
        Assert.True(options.Resizable);
        Assert.True(options.VSync);
    }

    [Fact]
    public void InputEnumValuesMatchNativeWindowAbi()
    {
        Assert.Equal(256, (int)WindowKey.Escape);
        Assert.Equal(65, (int)WindowKey.A);
        Assert.Equal(0u, (uint)MouseButton.Left);
        Assert.Equal(1u, (uint)KeyModifiers.Shift);
    }

    [Fact]
    public void WindowInputApiUsesDomainNamesInsteadOfGpuInputNames()
    {
        var featherAssembly = typeof(GpuWindow).Assembly;

        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowKey"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowKeyEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowResizeEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowCloseEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowCharInputEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowMouseButtonEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowMouseMoveEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowMouseScrollEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.WindowFocusEvent"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.MouseButton"));
        Assert.NotNull(featherAssembly.GetType("Feather.Windowing.KeyModifiers"));

        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuWindowEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuKey"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuKeyEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuWindowResizeEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuWindowCloseEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuCharInputEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuMouseButton"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuMouseButtonEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuMouseMoveEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuMouseScrollEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuWindowFocusEvent"));
        Assert.Null(featherAssembly.GetType("Feather.Windowing.GpuModifierFlags"));
    }
}
