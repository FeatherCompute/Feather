# API Reference: Graphics

## Purpose

Graphics APIs create generated vertex/fragment pipelines, configure draw state, and render into Feather textures.

## Typical Usage

```csharp
using var pipeline = GPU.CreateGraphicsPipeline<TriangleVS, TriangleFS, float4>();
pipeline.Draw(new TriangleVS(vertices.AsReadOnly()), new TriangleFS(), color, vertexCount: 3);
```

## Shader Interfaces

| Interface | Purpose |
| --- | --- |
| `IVertexShader<TVaryings>` | Vertex shader returning varyings. |
| `IFragmentShader<TVaryings>` | Fragment shader returning `float4`. |
| `IFragmentShader<TVaryings,TOutput>` | Fragment shader returning an explicit output struct. |

Graphics shader structs use `[VertexShader]` and `[FragmentShader]`.

## Built-In IDs

| API | Meaning |
| --- | --- |
| `VertexIds.Index` | Current vertex index. |
| `VertexIds.Instance` | Current instance index. |
| `FragmentIds.Coord` | Fragment coordinate. |

## `GpuGraphicsPipeline<TVS,TFS,TVaryings>`

| API | Purpose |
| --- | --- |
| `Desc` | Normalized pipeline descriptor. |
| `LastDispatchPath` | Last native graphics route. |
| `Draw(vs, fs, target, vertexCount, wait)` | Draws to one color target. |
| `Draw(vs, fs, target, depth, vertexCount, wait)` | Draws with depth/stencil target. |
| `Draw(vs, fs, targets, vertexCount, drawDesc, wait)` | Draws to MRT targets. |
| `DrawIndexed(...)` | Draws with a per-draw index buffer. |
| `Dispose()` | Releases native pipeline handle. |

## Draw State

`GraphicsDrawDesc` controls state that belongs to one draw call rather than the reusable graphics pipeline.

| Property | Purpose |
| --- | --- |
| `Viewport` | Optional viewport rectangle. Defaults to the full color target. |
| `Scissor` | Optional scissor rectangle. Defaults to the full color target. |
| `ColorLoadOp` | Per-pass color attachment load behavior: `Default`, `Load`, `Clear`, or `DontCare`. |
| `ClearColor` | Optional color clear value. Valid with `Default` or `Clear`. |
| `DepthLoadOp` | Per-pass depth attachment load behavior: `Default`, `Load`, or `Clear`. |
| `ClearDepth` | Optional depth clear value, clamped to `[0, 1]`. |

`GraphicsColorLoadOp.Load` is used for multi-pass rendering into the same target:

```csharp
floorPipeline.Draw(floorVS, floorFS, color, 3, new GraphicsDrawDesc
{
    ColorLoadOp = GraphicsColorLoadOp.Clear,
    ClearColor = new float4(0, 0, 0, 1)
});

lightPipeline.Draw(lightVS, lightFS, color, 6, new GraphicsDrawDesc
{
    ColorLoadOp = GraphicsColorLoadOp.Load
});
```

This clears before the first pass, then preserves the first pass anywhere the second pass does not rasterize. `GraphicsColorLoadOp.DontCare` is for passes that overwrite every pixel they later read.

For MSAA pipelines, Feather preserves an internal multisampled color attachment per resolved render target, so a later `Load` draw against the same target keeps previous multisampled color before resolving again.

## Pipeline State

`GraphicsPipelineDesc`:

| Property | Purpose |
| --- | --- |
| `Topology` | Primitive topology. |
| `SampleCount` | MSAA sample count. |
| `DepthTest`, `DepthWrite` | Convenience depth flags. |
| `Blend` | Default blend state. |
| `BlendAttachments` | Per-attachment blend states. |
| `DepthStencil` | Full depth/stencil state. |
| `Raster` | Cull/front-face/polygon/depth-clamp state. |
| `ColorAttachmentCount` | MRT count, 1 through 8. |
| `DebugName` | Native debug/profiler name. |

Common state types:

- `BlendState`
- `DepthStencilState`
- `RasterState`
- `GraphicsDrawDesc`
- `GraphicsRect`

## Render Targets

Graphics render into `GpuTexture2D<,>`:

```csharp
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8);
using var depth = GPU.CreateDepthTexture2D(width, height);
```

Present with `GpuTexturePresenter` or save/read the texture after drawing.

## Errors And Limits

- Optional raster states depend on backend feature support.
- Window swapchains are not public render targets.
- Persistent native index-buffer binding is not exposed; pass index buffers per draw.
- Unsupported draw shapes return native errors and set `LastDispatchPath` to rejected.

## Host Vs Shader

- `GpuGraphicsPipeline<TVS,TFS,TVaryings>`, `GraphicsPipelineDesc`, and draw calls are host APIs.
- Vertex/fragment shader `Execute` methods, `VertexIds`, `FragmentIds`, varyings, and texture sampling are shader-facing.
- Render targets are host-owned `GpuTexture2D<,>` objects passed into draw calls.

## Lifetime And Errors

- `GpuGraphicsPipeline<TVS,TFS,TVaryings>` is disposable.
- Render/depth textures and samplers used by a draw remain owned by the caller.
- Unsupported device features return native errors rather than silently changing state.

## Samples And Tests

- `samples/WindowGraphicsTriangle`
- `samples/WindowGraphicsTexturedQuad`
- `samples/SponzaRenderer`
- `samples/ProfilerSuite`
- `tests/Feather.Integration.Tests/GeneratedGraphicsPipelineTests.cs`
- `tests/Feather.Graphics.Tests/GraphicsSurfaceTests.cs`

## Guide

See [Graphics Pipeline](../graphics-pipeline.md).
