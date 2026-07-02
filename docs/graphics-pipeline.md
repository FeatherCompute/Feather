# Graphics Pipeline

Feather's graphics pipeline API lets you write preview vertex and fragment shaders as C# generated structs. The pipeline renders into Feather textures, which can be saved, read back, or presented in a `GpuWindow`.

![Sponza renderer in Feather](img/sponza.png)

Graphics is a preview surface. It is useful for samples and experiments, and it already supports real render targets, depth, MSAA, MRT, texture sampling, and indexed draws through the typed EasyGPU graphics path.

## Mental Model

Compute kernels answer: "run this code once per logical element."

Graphics pipelines answer: "run a vertex shader for vertices, interpolate varyings, then run a fragment shader for pixels."

In Feather:

- Vertex shaders implement `IVertexShader<TVaryings>`.
- Fragment shaders implement `IFragmentShader<TVaryings>` or `IFragmentShader<TVaryings, TOutput>`.
- `TVaryings` is an unmanaged value shared between stages.
- Color and depth targets are `GpuTexture2D<,>` objects.
- Window presentation is a separate step through `GpuTexturePresenter`.

## Minimal Triangle

```csharp
using Feather;
using Feather.Graphics;
using Feather.Math;
using Feather.Resources;

using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(800, 600, PixelFormat.Rgba8);
using var vertices = GPU.CreateBuffer<float4>(
[
    new float4(-0.8f, -0.7f, 0, 1),
    new float4( 0.8f, -0.7f, 0, 1),
    new float4( 0.0f,  0.75f, 0, 1)
]);
using var pipeline = GPU.CreateGraphicsPipeline<TriangleVS, TriangleFS, float4>();

pipeline.Draw(new TriangleVS(vertices.AsReadOnly()), new TriangleFS(), color, vertexCount: 3);

public readonly record struct Rgba32(byte R, byte G, byte B, byte A);

[VertexShader]
public readonly partial struct TriangleVS(ReadOnlyBuffer<float4> vertices) : IVertexShader<float4>
{
    public float4 Execute()
    {
        return vertices[VertexIds.Index];
    }
}

[FragmentShader]
public readonly partial struct TriangleFS : IFragmentShader<float4>
{
    public float4 Execute(float4 input)
    {
        return input;
    }
}
```

Run the full interactive version:

```bash
dotnet run --project samples/WindowGraphicsTriangle/WindowGraphicsTriangle.csproj
```

## Varyings

For real shaders, define a `[GpuStruct]` varying type. Use `[Position]` for clip-space position and optional `[Color(index)]` fields for color outputs when using fragment-output structs.

```csharp
[GpuStruct]
public readonly partial record struct MeshVaryings(
    [property: Position] float4 Position,
    float2 Uv,
    float3 Normal);
```

Vertex shader:

```csharp
[VertexShader]
public readonly partial struct MeshVS(ReadOnlyBuffer<Vertex> vertices, Uniform<float4x4> mvp)
    : IVertexShader<MeshVaryings>
{
    public MeshVaryings Execute()
    {
        Vertex v = vertices[VertexIds.Index];
        return new MeshVaryings(
            ShaderMath.Mul(mvp.Value, new float4(v.Position, 1.0f)),
            v.Uv,
            v.Normal);
    }
}
```

Fragment shader:

```csharp
[FragmentShader]
public readonly partial struct MeshFS(SampledTexture2D<float4> albedo, SamplerState sampler)
    : IFragmentShader<MeshVaryings>
{
    public float4 Execute(MeshVaryings input)
    {
        return albedo.Sample(sampler, input.Uv);
    }
}
```

## Render Targets And Depth

Create render targets through `GPU`:

```csharp
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(width, height, PixelFormat.Rgba8);
using var hdr = GPU.CreateRenderTexture2D<float4, float4>(width, height, PixelFormat.Rgba32Float);
using var depth = GPU.CreateDepthTexture2D(width, height);
```

Depth-enabled draw:

```csharp
using var pipeline = GPU.CreateGraphicsPipeline<MeshVS, MeshFS, MeshVaryings>(
    new GraphicsPipelineDesc
    {
        DepthTest = true,
        DepthWrite = true,
        DebugName = "Mesh"
    });

pipeline.Draw(new MeshVS(vertices.AsReadOnly(), new Uniform<float4x4>(mvp)),
              new MeshFS(texture.AsSampled(), sampler),
              color,
              depth,
              vertexCount);
```

## Per-Draw Color Load And Clear

Every `Draw` starts a native render pass. Use `GraphicsDrawDesc.ColorLoadOp` when a frame is split into multiple passes that render into the same color target.

| Value | Meaning |
| --- | --- |
| `GraphicsColorLoadOp.Default` | Compatibility mode. A supplied `ClearColor` clears the target; otherwise non-MSAA draws keep the previous contents and MSAA draws clear their internal multisampled color attachment. |
| `GraphicsColorLoadOp.Load` | Load the existing color target contents at the start of the pass. This is the option for a second pass that should preserve pixels it does not cover. |
| `GraphicsColorLoadOp.Clear` | Clear the color target before drawing. `ClearColor` defaults to `(0, 0, 0, 1)` when omitted. |
| `GraphicsColorLoadOp.DontCare` | The previous color contents are undefined. Use only when the draw overwrites every pixel you will read later. |

Two-pass example:

```csharp
floorPipeline.Draw(
    new FloorVS(floorVertices.AsReadOnly()),
    new FloorFS(...),
    color,
    vertexCount: 3,
    drawDesc: new GraphicsDrawDesc
    {
        ColorLoadOp = GraphicsColorLoadOp.Clear,
        ClearColor = new float4(0, 0, 0, 1)
    });

lightPipeline.Draw(
    new LightQuadVS(lightVertices.AsReadOnly()),
    new LightQuadFS(...),
    color,
    vertexCount: 6,
    drawDesc: new GraphicsDrawDesc
    {
        ColorLoadOp = GraphicsColorLoadOp.Load
    });
```

The first pass clears and fills the floor. The second pass loads that color target, so pixels outside the light quad keep the floor result while pixels covered by the quad are overwritten by the light shader.

`ClearColor` is only valid with `Default` or `Clear`. `Load` and `DontCare` reject `ClearColor` because no clear is performed.

With MSAA enabled, Feather keeps an internal multisampled color attachment for each resolved render target. A later `ColorLoadOp.Load` draw against the same target loads that multisampled attachment, preserves the earlier pass, rasterizes more geometry, and resolves back into the target at the end of the draw.

## Pipeline State

`GraphicsPipelineDesc` controls fixed-function state:

| Property | Purpose |
| --- | --- |
| `Topology` | Primitive topology, default `TriangleList`. |
| `SampleCount` | MSAA sample count, default `X1`. |
| `DepthTest`, `DepthWrite` | Convenience flags merged into `DepthStencil`. |
| `DepthStencil` | Depth and stencil compare/write state. |
| `Blend` | Default color blend state. |
| `BlendAttachments` | Per-attachment blend states for MRT. |
| `Raster` | Cull mode, front face, polygon mode, depth clamp. |
| `ColorAttachmentCount` | Number of render targets, 1 through 8. |
| `DebugName` | Native profiler/debug name. |

Convenience states:

- `BlendState.Opaque`
- `BlendState.AlphaBlend`
- `BlendState.Additive`
- `DepthStencilState.Default`
- `RasterState.Default`

Some raster states require optional Vulkan features. Unsupported devices return explicit errors rather than silently changing state.

## Multiple Render Targets

Pipelines can target more than one color attachment by setting `ColorAttachmentCount` and drawing with a texture span:

```csharp
using var pipeline = GPU.CreateGraphicsPipeline<GBufferVS, GBufferFS, GBufferVaryings>(
    new GraphicsPipelineDesc { ColorAttachmentCount = 2 });

pipeline.Draw(new GBufferVS(...), new GBufferFS(...), [albedo, normal], vertexCount);
```

Use matching fragment output structs when writing multiple outputs.

## Indexed Draws

Create an index buffer with `GPU.CreateIndexBuffer`:

```csharp
using var indices = GPU.CreateIndexBuffer<ushort>([0, 1, 2, 2, 3, 0]);

pipeline.DrawIndexed(new QuadVS(vertices.AsReadOnly()),
                     new QuadFS(),
                     color,
                     indices,
                     indexCount: 6);
```

Persistent native index-buffer binding is intentionally not exposed; pass the index buffer for the draw that uses it.

## Window Presentation

Graphics does not render directly to a window swapchain. Render to a texture and present:

```csharp
using var window = GpuWindow.Create(new() { Width = 800, Height = 600 });
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<Rgba32, Rgba32>(window.Width, window.Height, PixelFormat.Rgba8);

while (window.IsOpen)
{
    window.PollEvents();
    pipeline.Draw(vs, fs, color, vertexCount);
    presenter.Present(color);
}
```

## Samples

| Sample | Shows |
| --- | --- |
| `WindowGraphicsTriangle` | Minimal vertex/fragment pipeline. |
| `WindowGraphicsTexturedQuad` | Texture sampling in a fragment shader. |
| `SponzaRenderer` | OBJ loading, atlas texture, depth, MSAA, camera movement. |
| `ProfilerSuite` | Graphics profiling and dispatch path validation. |

## Current Limits

- Graphics is preview-quality.
- Swapchain render targets and ImGui overlays are not part of Feather's public API.
- Render targets are Feather textures, not window backbuffers.
- Optional raster features depend on the active backend/device.
- The supported shader subset is the same style of generated C# subset described in [C# Shader Subset](csharp-subset.md).

## Next Reading

- [API: Graphics](api/graphics.md)
- [Windowing](window.md)
- [Examples](examples.md)
- [Support Status](support-status.md)
