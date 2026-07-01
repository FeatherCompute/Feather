# Windowing

Feather windows are presentation shells. They do not act as render targets directly; compute kernels and graphics pipelines render into `GpuTexture2D`, then a `GpuTexturePresenter` presents that texture.

```text
C# compute kernel or graphics pipeline
    -> GpuTexture2D
    -> GpuTexturePresenter
    -> GpuWindow
```

![Volumetric fog presented from a Feather texture](img/volumetric-fog.png)

## Create A Window

```csharp
using Feather.Windowing;

using var window = GpuWindow.Create(new GpuWindowOptions
{
    Width = 800,
    Height = 480,
    Title = "Feather Window",
    VSync = true,
    Resizable = true
});

while (window.IsOpen)
{
    window.PollEvents();
    while (window.TryPollEvent(out var e))
    {
        if (e is WindowKeyEvent { Key: WindowKey.Escape, Pressed: true })
        {
            window.Close();
        }
    }
}
```

`GpuWindowOptions` defaults to a visible, resizable, high-DPI, centered 1280 by 720 window.

| Option | Meaning |
| --- | --- |
| `Width`, `Height` | Initial window size. |
| `Title` | Native title-bar text. |
| `Resizable` | Allows native resize events. |
| `Visible` | Creates a visible or hidden window. |
| `VSync` | Requests synchronized presentation. |
| `HighDpi` | Requests high-DPI framebuffer behavior when supported. |
| `CenterOnCreate` | Centers the window on creation. |

## Events And Input

Poll events each frame:

```csharp
window.PollEvents();

while (window.TryPollEvent(out var e))
{
    switch (e)
    {
        case WindowResizeEvent resize:
            Console.WriteLine($"{resize.Width}x{resize.Height}");
            break;
        case WindowCloseEvent:
            window.Close();
            break;
        case WindowKeyEvent { Key: WindowKey.Escape, Pressed: true }:
            window.Close();
            break;
    }
}
```

You can also query current state:

```csharp
bool left = window.IsMouseDown(MouseButton.Left);
bool escape = window.IsKeyDown(WindowKey.Escape);
int2 mouse = window.MousePosition;
float2 scroll = window.MouseScroll;
float aspect = window.Aspect;
```

`WaitEvents()` is available for event-driven tools that do not need a continuous render loop.

## Present CPU Pixels

For simple CPU-side demos, use `GpuPixelBuffer`:

```csharp
using var pixels = new GpuPixelBuffer(window.Width, window.Height);

pixels.Clear(GpuColor.Rgba(16, 24, 32));
window.Present(pixels);
```

This path uploads CPU memory to the window. It is useful for UI tests and simple demos, but GPU-generated images should stay on the GPU and use a texture presenter.

## Present GPU Textures

For GPU output, render or compute into a texture:

```csharp
using Feather;
using Feather.Math;
using Feather.Resources;
using Feather.Windowing;

using var window = GpuWindow.Create(new() { Width = 800, Height = 450 });
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<float4, float4>(
    window.Width,
    window.Height,
    PixelFormat.Rgba32Float);

var frame = 0;
while (window.IsOpen)
{
    window.PollEvents();
    GPU.Dispatch(new ComputePixels(color.AsReadWrite(), new Uniform<int>(frame)), color.Size);
    presenter.Present(color);
    frame++;
}
```

This is the pattern used by `samples/WindowCompute`, `samples/WindowGraphicsTriangle`, and `samples/SponzaRenderer`.

## Resize Strategy

`GpuTexture2D` sizes are fixed. When a window is resized, recreate render targets and any depth textures that must match the window size:

```csharp
if (e is WindowResizeEvent resize)
{
    color.Dispose();
    depth.Dispose();
    color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(resize.Width, resize.Height, PixelFormat.Rgba8);
    depth = GPU.CreateDepthTexture2D(resize.Width, resize.Height);
}
```

Keep the ownership simple: dispose old textures after the GPU work that uses them has completed, then create replacements for the next frame.

## Build Notes

Native window support is enabled by default:

```bash
cmake -S native -B native/build -DFEATHER_BUILD_WINDOW=ON
```

For headless/core-only builds:

```bash
cmake -S native -B native/build -DFEATHER_BUILD_WINDOW=OFF
```

Real window tests are opt-in:

```bash
FEATHER_WINDOW_TESTS=1 dotnet test tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj --filter WindowingNativeOptInTests
```

On macOS, create windows and poll events on the main thread.

## Current Limits

- Feather does not expose ImGui.
- Windows are presentation targets, not graphics pipeline swapchain render targets.
- Windowing is runtime opt-in; compute-only workloads do not open a display.
- Headless builds should use `FEATHER_BUILD_WINDOW=OFF`.

## API Reference

See [API: Windowing](api/windowing.md).
