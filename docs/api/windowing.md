# API Reference: Windowing

## Purpose

Windowing APIs create native windows, poll input/events, present CPU pixel buffers, and present GPU textures.

## Typical Usage

```csharp
using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 450,
    Title = "Feather"
});
using var presenter = window.CreateTexturePresenter();

while (window.IsOpen)
{
    window.PollEvents();
    presenter.Present(colorTexture);
}
```

## `GpuWindow`

| API | Purpose |
| --- | --- |
| `Create(options)` | Creates a native window. |
| `IsOpen` | Whether the window should continue running. |
| `Width`, `Height`, `Aspect` | Current window size/aspect. |
| `MousePosition`, `MouseScroll` | Current pointer state. |
| `PollEvents()` | Processes native events without blocking. |
| `WaitEvents()` | Waits for native events. |
| `TryPollEvent(out WindowEvent)` | Retrieves queued high-level events. |
| `IsKeyDown(key)` | Current key state. |
| `IsMouseDown(button)` | Current mouse-button state. |
| `SetTitle(title)` | Updates the native title. |
| `SetVSync(enabled)` | Toggles vsync where supported. |
| `Present(GpuPixelBuffer)` | Presents CPU pixels. |
| `CreateTexturePresenter()` | Creates a GPU texture presenter. |
| `Close()` | Requests close. |
| `Dispose()` | Destroys the window and presenters. |

## `GpuWindowOptions`

| Property | Default | Purpose |
| --- | --- | --- |
| `Width` | `1280` | Initial width. |
| `Height` | `720` | Initial height. |
| `Title` | `"Feather"` | Title-bar text. |
| `Resizable` | `true` | Allows resize. |
| `Visible` | `true` | Creates visible window. |
| `VSync` | `true` | Requests synchronized presentation. |
| `HighDpi` | `true` | Requests high-DPI behavior. |
| `CenterOnCreate` | `true` | Centers on creation. |

## Events

`WindowEvent` derived records include:

- `WindowResizeEvent`
- `WindowCloseEvent`
- `WindowKeyEvent`
- `WindowCharInputEvent`
- `WindowMouseButtonEvent`
- `WindowMouseMoveEvent`
- `WindowMouseScrollEvent`
- `WindowFocusEvent`

Input enums include `WindowKey`, `MouseButton`, and `KeyModifiers`.

## Pixel Buffers

```csharp
using var pixels = new GpuPixelBuffer(width, height);
pixels.Clear(GpuColor.Rgba(16, 24, 32));
window.Present(pixels);
```

`GpuColor.Rgba(byte r, byte g, byte b, byte a = 255)` packs a CPU pixel value.

## Texture Presentation

`GpuTexturePresenter` presents Feather textures:

```csharp
using var presenter = window.CreateTexturePresenter();
presenter.Present(colorTexture);
```

It also supports presenting CPU pixel buffers through `Present(GpuPixelBuffer)`.

## Platform Notes

- Build with `FEATHER_BUILD_WINDOW=ON` to include native window support.
- Use `FEATHER_BUILD_WINDOW=OFF` for headless builds.
- On macOS, create and poll real windows on the main thread.
- Compute-only workloads do not create windows.

## Host Vs Shader

Windowing is entirely host-side. Generated GPU work renders into textures; window APIs present those textures or CPU pixel buffers.

## Lifetime And Errors

- `GpuWindow`, `GpuTexturePresenter`, and `GpuPixelBuffer` are disposable.
- Presenters are tied to the window that created them.
- Native window failures surface as `FeatherNativeException`.

## Guide

See [Windowing](../window.md).

## Samples And Tests

- `samples/WindowHello`
- `samples/WindowCompute`
- `samples/WindowPixels`
- `samples/WindowGraphicsTriangle`
- `tests/Feather.Graphics.Tests/WindowingSurfaceTests.cs`
