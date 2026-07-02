# API Reference: Core Runtime

## Purpose

Core runtime APIs create resources, dispatch kernels, inspect backend capabilities, define generated shader shapes, and collect profiler data.

## Typical Usage

```csharp
GpuContext context = GPU.Context;
BackendCaps caps = context.Caps;

using var input = GPU.CreateBuffer<float>([1, 2, 3], BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(3, BufferAccess.ReadWrite);

DispatchPath path = GPU.DispatchAndGetPath(
    new CopyKernel(input.AsReadOnly(), output.AsReadWrite()),
    input.Length);
```

## `GPU`

`GPU` is the main host entry point.

| API | Purpose |
| --- | --- |
| `Context` | Lazily creates and returns the default `GpuContext`. |
| `CreateBuffer<T>(count, access)` | Allocates a typed GPU buffer. |
| `CreateBuffer<T>(data, access)` | Allocates and uploads a typed GPU buffer. |
| `CreateReadOnlyBuffer<T>(data)` | Convenience allocation returning `ReadOnlyBuffer<T>`. |
| `CreateWriteOnlyBuffer<T>(count)` | Convenience allocation returning `WriteOnlyBuffer<T>`. |
| `CreateReadWriteBuffer<T>(count/data)` | Convenience allocation returning `ReadWriteBuffer<T>`. |
| `CreateIndexBuffer<T>(data)` | Creates a read-only buffer for indexed graphics draws. |
| `CreateTexture2D<TPixel,TValue>(...)` | Allocates a 2D texture. |
| `CreateTexture3D<TPixel,TValue>(...)` | Allocates a 3D texture. |
| `CreateRenderTexture2D<TPixel,TValue>(...)` | Allocates a render-target texture. |
| `CreateDepthTexture2D(width, height)` | Allocates a `Depth32Float` texture. |
| `CreateDepthStencilTexture2D(width, height)` | Allocates a `Depth24Stencil8` texture. |
| `LoadReadWriteTexture2D<TPixel,TValue>(path)` | Loads a TGA as a read-write texture. |
| `LoadSampledTexture2D<TPixel,TValue>(path)` | Loads a TGA as a sampled texture. |
| `CreateSampler(desc)` | Creates a sampler state. |
| `CreateGraphicsPipeline<TVS,TFS,TVaryings>(desc)` | Creates a generated graphics pipeline. |
| `CreateADKernel<TKernel>(kernel)` | Creates an AD wrapper for a generated 1D kernel. |
| `Dispatch(kernel, int/int2/int3, wait)` | Dispatches generated 1D/2D/3D compute kernels. |
| `DispatchAndGetPath(...)` | Dispatches and returns the native route used. |

## `GpuContext`

| API | Purpose |
| --- | --- |
| `BackendType` | Active backend type. |
| `Caps` | Backend capabilities and limits. |
| `GetDefault()` | Creates the default context. |
| `Dispose()` | Releases the native context handle. |

Most applications use `GPU.Context` rather than constructing contexts manually.

## Capabilities

`BackendCaps` reports backend type, max workgroup dimensions, and feature flags. Use it to gate optional graphics features or to print runtime diagnostics.

```csharp
var caps = GPU.Context.Caps;
Console.WriteLine($"{caps.BackendType}: {caps.MaxWorkGroupSizeX}x{caps.MaxWorkGroupSizeY}x{caps.MaxWorkGroupSizeZ}");
```

## Attributes

| Attribute | Target | Purpose |
| --- | --- | --- |
| `[Kernel]` | Struct | Marks a generated compute kernel. `BoundsCheck` defaults to `true`. |
| `[AutoDiff]` | Struct | Adds AD metadata for a generated 1D kernel. |
| `[ThreadGroupSize]` | Struct | Sets local workgroup size. |
| `[VertexShader]` | Struct | Marks a generated vertex shader. |
| `[FragmentShader]` | Struct | Marks a generated fragment shader. |
| `[Entry]` | Method | Selects an explicit entry method. |
| `[Callable]` | Method | Emits a helper method into the shader module. |
| `[ShaderLibrary]` | Class/struct | Enables source-available static `[Callable]` helpers to be imported by generated shaders. |
| `[GpuStruct]` | Struct | Requests deterministic GPU layout metadata. |
| `[Position]` | Field/property | Marks graphics position output. |
| `[Color(index)]` | Field/property | Marks a fragment color output. |
| `[Binding(index)]` | Parameter/field | Overrides a resource binding where supported. |

## Enums

| Enum | Purpose |
| --- | --- |
| `BufferAccess` | Buffer read/write mode. |
| `TextureAccess` | Texture read/write/sample/render/depth mode. |
| `PixelFormat` | Texture pixel format. |
| `DefaultThreadGroupSizes` | Common local group-size presets. |
| `PrimitiveTopology` | Graphics topology. |
| `SampleCount` | Graphics MSAA sample count. |
| `GpuLayout` | GPU struct layout selection. |
| `DispatchPath` | Native route used by a dispatch/draw. |

## Profiler

```csharp
GpuProfiler.SetEnabled(true);
GpuProfiler.Clear();

// Dispatch or draw work here.

GpuProfilerQuery query = GpuProfiler.Query("MyKernel");
Console.WriteLine(GpuProfiler.GetFormattedReport());
```

| API | Purpose |
| --- | --- |
| `GpuProfiler.IsEnabled` | Reads global profiler state. |
| `SetEnabled(bool)` | Enables/disables profiling. |
| `Clear()` | Clears accumulated profiler data. |
| `GetTotalTimeMs()` | Total recorded GPU time. |
| `Query(name)` | Gets count/min/max/average/total for one name. |
| `GetFormattedReport()` | Gets a textual report. |

## Errors

Native failures throw `FeatherNativeException`. Generator failures appear as `FE0001`-style diagnostics. See [Diagnostics](../diagnostics.md).

## Host Vs Shader

Core runtime APIs are host APIs. Attributes are source-generation metadata. Thread IDs, barriers, atomics, and shader resource views are documented in [Kernels](kernels.md) and are shader-facing.

## Lifetime And Errors

- `GpuContext` is disposable, though most applications use the process-wide `GPU.Context`.
- Resources created through `GPU` are disposable and are documented in [Resources](resources.md).
- Native failures throw `FeatherNativeException`.
- Generator failures appear as `FE0001`-style diagnostics.

## Samples And Tests

- `samples/HelloBuffer`
- `samples/ProfilerSuite`
- `samples/SpirvOptInspection`
- `tests/Feather.Tests/PublicApiTests.cs`
- `tests/Feather.Integration.Tests/GeneratedComputeDispatchTests.cs`
