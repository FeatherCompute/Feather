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
| `CreateReadOnlyBuffer<T>(data)` | Returns an owning `GpuBuffer<T>` configured for read-only shader access. |
| `CreateWriteOnlyBuffer<T>(count)` | Returns an owning `GpuBuffer<T>` configured for write-only shader access. |
| `CreateReadWriteBuffer<T>(count/data)` | Returns an owning read-write `GpuBuffer<T>`. |
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
| `Compile<TKernel>()` | Precompiles and caches a generated compute pipeline without dispatching it. |
| `Queue` | Default queue used to create command lists, submit work, and wait on fences. |

## `GpuContext`

| API | Purpose |
| --- | --- |
| `BackendType` | Active backend type. |
| `Caps` | Backend capabilities and limits. |
| `GetDefault()` | Creates the default context. |
| `Compile<TKernel>()` | Precompiles a context-owned generated kernel. |
| `WaitIdle()` | Waits for queued GPU work and releases retained submission resources. |
| `Dispose()` | Releases the native context handle. |

Most applications use `GPU.Context` rather than constructing contexts manually.

Buffer views returned by `.AsReadOnly()`, `.AsWriteOnly()`, and `.AsReadWrite()` are non-owning shader bindings. Keep and dispose the `GpuBuffer<T>` owner. The convenience allocation methods return that owner rather than a view whose owner would otherwise be lost.
Owning buffers implicitly convert to a compatible shader view, so existing constructor calls can continue to pass a convenience-factory result directly.

Owning buffers and textures implement `IGpuResourceAllocation`. Its `AllocationInfo` query is side-effect free: before a
lazy resource is materialized, or when the backend cannot expose exact allocation evidence, `Available` is false. When
available, `PhysicalBytes` is the backend's exact allocation size and `AllocationGroup` is an opaque identity used only
to compare whether resources share that allocation.

## Queue, Command Lists, And Fences

`GPU.Queue` is the context's ordered submission queue. A `GpuCommandList` starts in the recording state, must be closed before submission, and can then be submitted repeatedly. `Reset()` clears it and returns it to recording. `IsClosed`, `IsDisposed`, and `Count` expose its current recording state without changing it.

```csharp
using var source = GPU.CreateBuffer<float>([1, 2, 3, 4]);
using var destination = GPU.CreateBuffer<float>(4);
using var commands = GPU.Queue.CreateCommandList();

commands.CopyBuffer(source, destination);
commands.MemoryBarrier(GpuMemoryBarrier.Buffer);
commands.Dispatch(new MyKernel(destination.AsReadWrite()), destination.Length);
commands.Close();

await using var fence = GPU.Queue.Submit(commands);
await fence.WaitAsync();
```

Queue submission APIs are:

| API | Semantics |
| --- | --- |
| `Submit(commandList)` | Submits one closed list and returns its completion fence. |
| `Submit(commandLists)` | Validates and snapshots every closed list before atomically replaying them in order behind one fence. An empty list is valid. |
| `SubmitTimestamped(record)` | Records one callback under exclusive queue ownership, submits that exact command stream, and returns the callback result plus its completion fence. Unsupported backends still submit normally and report timing unavailable. |
| `Signal()` | Inserts an empty queue-ordered completion point after work already recorded on the queue. |
| `WaitIdle()` | Waits for all outstanding work; prefer a fence for local dependencies. |

Command lists record these command families:

| API | Recorded operation |
| --- | --- |
| `Dispatch(kernel, int/int2/int3)` | Generated compute dispatch. |
| `CopyBuffer(source, destination)` | Full type-safe GPU buffer copy. |
| `CopyBuffer(source, sourceIndex, destination, destinationIndex, count)` | Element-range GPU buffer copy. |
| `MemoryBarrier(flags)` | Explicit buffer, texture, uniform, or all-resource dependency. |
| `Draw(...)` | Non-indexed graphics draw with single-target, color/depth, and multi-target overloads. |
| `DrawIndexed(...)` | Indexed graphics draw with an explicit index buffer and matching target overloads. |
| `Close()` | Ends recording; idempotent. |
| `Reset()` | Clears commands and resumes recording. |

`GpuQueue.Submit` snapshots every closed list before executing any command, replays the snapshot as one ordered managed queue operation, and returns a fence for that exact native submission. Other command-list submissions, immediate dispatch/draw calls, and managed buffer/texture transfers cannot interleave its replay. Compute bindings, copy buffers, graphics pipelines, vertex/index buffers, shader resources, and render targets are retained by the fence until completion. Resetting or disposing a command list after `Submit` does not change an in-flight submission.

The next submitted fence also adopts resource leases from earlier immediate non-waiting dispatches or draws on the same context. Waiting that fence therefore both proves queue completion and releases the resources retained by all work ordered before it; `Signal()` is the explicit way to obtain such a fence when no command list needs to be replayed.

`GpuFence` provides:

| API | Semantics |
| --- | --- |
| `IsCompleted` | Polls the native submission without waiting for GPU completion. |
| `Wait()` | Waits indefinitely. |
| `Wait(timeout)` | Returns `false` on timeout and `true` on completion. |
| `WaitAsync(token)` | Shared non-blocking completion observation with per-caller cancellation. |
| `WaitAsync(timeout, token)` | Shared asynchronous observation with per-caller timeout and cancellation. |
| `TryGetGpuElapsedNanoseconds(out elapsed)` | Resolves the real aggregate GPU interval owned by a timestamped submission without waiting. Returns `false` for pending, unsupported, or invalid results; never substitutes CPU or fence latency. |
| `IsDisposed` | Reports whether the native submission marker has been released. |
| `Dispose()` / `DisposeAsync()` | Waits for completion before releasing retained resources and the native fence. |

Concurrent waiters and concurrent synchronous/asynchronous disposal are supported. One caller performs each native wait or release operation while the others observe the same result. A failed deterministic dispose remains retryable. If a fence is abandoned, its finalizer schedules a non-blocking reaper that waits for submission completion before releasing retained resources and the native marker; it does not block the CLR finalizer thread or free in-flight resources early.

Use `GPU.Queue.WaitIdle()` only when all outstanding work must finish. Prefer a fence when waiting for one submission, since it does not turn a local dependency into a device-wide idle point.

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
| `[Callable]` | Method | Emits a helper method into the shader module. Instance methods on `[GpuStruct]` values lower the receiver explicitly. |
| `[ShaderLibrary]` | Class/struct | Enables source-available static `[Callable]` helpers to be imported by generated shaders. |
| `[GpuStruct]` | Struct | Requests deterministic GPU layout metadata and enables GPU-value instance callables. |
| `[Position]` | Field/property | Marks graphics position output. |
| `[Color(index)]` | Field/property | Marks a fragment color output. |
| `[Binding(index)]` | Parameter/field | Overrides a resource binding where supported. |

`[GpuStruct]` is a value-type GPU layout contract. It is also the supported surface for object-style shader code: instance `[Callable]` methods can read fields, and mutating methods lower their receiver as `inout` when the call site can write back.

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
| `GpuMemoryBarrier` | Explicit command-list memory dependency flags. |

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

### Explicit diagnostic captures

`GpuContext.BeginExecutionHeatCapture(...)` and `GpuContext.BeginLineValueCapture(...)` create
private instrumented compute variants for an explicitly bounded diagnostic operation. They do not
alter or replace the ordinary cached kernel. Only one diagnostic capture may be active per
context, and the caller must wait the exact submission fence before reading the result.

`BeginLineValueCapture` binds one typed FEIR statement, one matching dispatch index, and one
global compute invocation. Its fixed 64-byte record returns whether the statement executed, its
dynamic occurrence count, and the raw bits of the LAST bool/int/uint/float32 scalar or vector
value. Unsupported sites and aggregate types fail closed. This is source-value evidence from an
instrumented variant; it is not line timing, a PC sample, or a hardware Hot Path.

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
- `samples/GpuStructInterfaces`
- `samples/ProfilerSuite`
- `samples/SpirvOptInspection`
- `tests/Feather.Tests/PublicApiTests.cs`
- `tests/Feather.Integration.Tests/GeneratedComputeDispatchTests.cs`
