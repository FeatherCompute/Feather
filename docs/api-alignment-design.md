# Feather API Alignment With Luisa Runtime Granularity

## Status And Scope

This is a design for M8 and M9. M8.1 device discovery, M8.2 multi-context
ownership, and M8.3 explicit streams/fences are implemented; later milestones
in this document remain design-only.
LuisaCompute is the target runtime model. EasyGPU is scheduled for removal in
M9, so new public API must describe Luisa concepts directly rather than add
another backend-selection layer.

The terms below are deliberately distinct:

* A **runtime** owns Luisa module discovery and a Luisa `Context`.
* A **device context** owns exactly one selected Luisa `Device`, its default
  stream, and resources created through it.
* A **stream** owns submission order. Completion is represented by an event or
  fence, never by an implicit global wait.

`GpuContext` below means a Feather device context. `GpuRuntime` is introduced
to make the underlying Luisa `Context` lifetime visible without exposing Luisa
C++ types in the managed public contract.

## Current Surface And Evidence

### Runtime and device model

Feather has a process-wide static facade: `GPU.Context` lazily obtains only
`GpuContext.GetDefault()` ([src/Feather/Core/GPU.cs](../src/Feather/Core/GPU.cs):9-17).
`GpuContext.GetDefault` asks native code for a default handle and initializes it
([src/Feather/Core/GpuContext.cs](../src/Feather/Core/GpuContext.cs):49-54).
The native ABI defines that handle as `kDefaultContext = 1`, and rejects every
other handle during initialization
([native/feather_c_api.cpp](../native/feather_c_api.cpp):64,
[native/feather_c_api.cpp](../native/feather_c_api.cpp):12146-12164).

Its `BackendType` and `BackendCaps` query EasyGPU, rather than the Luisa device
that executes the Luisa route
([src/Feather/Core/GpuContext.cs](../src/Feather/Core/GpuContext.cs):17-46;
[native/feather_c_api.cpp](../native/feather_c_api.cpp):12218-12276).

Luisa separates these concerns. `Context` loads modules, creates a device with
`DeviceConfig`, and lists backend device names
([LuisaCompute/include/luisa/runtime/context.h](../LuisaCompute/include/luisa/runtime/context.h):48-62).
`DeviceConfig.device_index` is explicit and defaults to no selected index
([LuisaCompute/include/luisa/runtime/rhi/device_interface.h](../LuisaCompute/include/luisa/runtime/rhi/device_interface.h):81-89).
`Device` exposes backend identity, native handle, warp size, resource creation,
and stream/event creation
([LuisaCompute/include/luisa/runtime/device.h](../LuisaCompute/include/luisa/runtime/device.h):120-149,
:183-247).

| Area | Feather today | Luisa capability | Gap |
| --- | --- | --- | --- |
| Runtime discovery | Implicit static default | Explicit `Context`, installed backends and device-name enumeration | No runtime object or device list |
| Device selection | One hard-coded context handle | `DeviceConfig.device_index` passed to `Context::create_device` | Cannot select, name, or concurrently own devices |
| Capabilities | EasyGPU-specific `BackendCaps` | Device/backend properties plus backend-specific extensions | Reports the wrong implementation after M9 |
| Context lifetime | `GpuContext.Dispose` shuts down the global native runtime | Device resources retain a device implementation | No per-device ownership or safe multi-context teardown |

### Dispatch and synchronization

The generated-kernel API accepts `wait`, but it is only a Boolean on each
dispatch. `GpuKernel.Dispatch` calculates groups and forwards it directly to
`fe_kernel_dispatch` ([src/Feather/Kernels/GpuKernel.cs](../src/Feather/Kernels/GpuKernel.cs):66-116).
`GPU.Dispatch` has the same Boolean and always uses the default context
([src/Feather/Core/GPU.cs](../src/Feather/Core/GPU.cs):193-243). Graphics draw
methods likewise take `wait` ([src/Feather/Graphics/GpuGraphicsPipeline.cs](../src/Feather/Graphics/GpuGraphicsPipeline.cs):342-418).
There is no public stream, fence, callback, dependency, or explicit
synchronization surface.

Luisa devices create tagged streams and events
([LuisaCompute/include/luisa/runtime/device.h](../LuisaCompute/include/luisa/runtime/device.h):140-147).
A stream accepts commands and callbacks, has explicit `commit` and
`synchronize`, and carries a `StreamTag`
([LuisaCompute/include/luisa/runtime/stream.h](../LuisaCompute/include/luisa/runtime/stream.h):39-50,
:85-111). Events support signal, wait, completion polling, and CPU
synchronization ([LuisaCompute/include/luisa/runtime/event.h](../LuisaCompute/include/luisa/runtime/event.h):48-86).

### Resources, graphics, and presentation

`GpuBuffer<T>` and `GpuTexture2D<TPixel, TValue>` retain their creating
`GpuContext`, but their interop handles are internal Feather ABI handles rather
than backend-native handles
([src/Feather/Resources/GpuBuffer.cs](../src/Feather/Resources/GpuBuffer.cs):15-24,
[src/Feather/Resources/GpuTexture2D.cs](../src/Feather/Resources/GpuTexture2D.cs):27-46).
Upload/read methods are synchronous native calls
([src/Feather/Resources/GpuBuffer.cs](../src/Feather/Resources/GpuBuffer.cs):94-169).
The native ABI exposes create/upload/download/destroy, but not external resource
import/export ([native/feather_c_api.h](../native/feather_c_api.h):293-319).

Graphics pipelines are created for one `GpuContext`; drawing binds Feather
handles and passes a per-draw wait flag
([src/Feather/Graphics/GpuGraphicsPipeline.cs](../src/Feather/Graphics/GpuGraphicsPipeline.cs):342-418).
The window presenter accepts only a Feather texture or CPU pixel buffer
([src/Feather/Windowing/GpuTexturePresenter.cs](../src/Feather/Windowing/GpuTexturePresenter.cs):19-39).

Luisa has native device handles and explicit import paths for images and
buffers ([LuisaCompute/include/luisa/runtime/device.h](../LuisaCompute/include/luisa/runtime/device.h):124-126,
:183-196,
:234-247). Whether a Metal, Vulkan, or DX native handle can be imported and
synchronized is backend-specific and must be verified before Feather exposes it.

### Ray tracing readiness

Current Feather exposes no acceleration-structure or ray-tracing pipeline type.
Luisa `Device` already creates meshes, curves, procedural primitives, an
`Accel`, and bindless arrays
([LuisaCompute/include/luisa/runtime/device.h](../LuisaCompute/include/luisa/runtime/device.h):150-181).
This is evidence for the target object graph, not evidence that every pinned
backend supports every RT feature. Per-device RT capability remains **pending
verification**.

## Proposed Public Model

### Runtime, devices, and compatibility

Introduce this managed ownership tree:

```text
GpuRuntime
  +- GpuDeviceInfo[]                 discovery only
  `- GpuContext (one selected device)
       +- DefaultStream : GpuStream
       +- GpuStream[]
       +- GpuBuffer / GpuTexture / Sampler / Pipeline
       `- future GpuAccel / GpuRayTracingPipeline
```

Proposed shape, subject to managed API review:

```csharp
var runtime = GpuRuntime.Create();
IReadOnlyList<GpuDeviceInfo> devices = runtime.EnumerateDevices();
using var context = runtime.CreateContext(new GpuContextOptions {
    DeviceIndex = 0, Backend = GpuBackend.Auto
});
using var copy = context.CreateStream(GpuStreamKind.Copy);
```

`GpuDeviceInfo` includes a runtime-local ordinal, backend identifier, display
name, and structured `GpuDeviceCapabilities`. It must not promise a globally
stable PCI or OS identity until LC backends provide one uniformly.
`GpuContextOptions.DeviceIndex` maps directly to LC `DeviceConfig.device_index`.
`GpuBackend.Auto` is a Feather policy with a documented preference order.
Passing an explicit backend makes selection deterministic. Device creation
failure, unavailable features, and invalid indices are errors; M8 must not fall
back to another device silently.

Keep `GPU.Context` and static resource/dispatch helpers as a source-compatible
facade over lazily-created `GpuRuntime.Default` and `GpuContext.Default`. Mark
them as convenience APIs. New examples and library APIs take a `GpuContext`.
Do not introduce another public `GpuExecutionBackend` selector: M8 makes Luisa
the only execution runtime, and M9 removes `EasyGpu` from that enum and its
diagnostics ([src/Feather/Core/Enums.cs](../src/Feather/Core/Enums.cs):97-116).

`GpuRuntime.Dispose` rejects disposal while contexts exist. `GpuContext.Dispose`
first synchronizes and disposes its streams, then releases only resources it
owns. Resources, pipelines, streams, and fences validate context identity; a
resource cannot bind to a kernel or pipeline from another context. This makes
multi-device transfer explicit rather than accidentally sharing a native handle.

### Stream and completion model

Add `GpuStream`, `GpuFence`, and optionally `GpuCommandList` only after native
stream ownership is implemented. The minimum M8 surface is:

```csharp
GpuFence GpuKernel.Dispatch<T>(GpuStream stream, T kernel, GpuDispatchSize size);
GpuFence GpuGraphicsPipeline<...>.Draw(GpuStream stream, ...);
GpuFence GpuBuffer<T>.UploadAsync(GpuStream stream, ReadOnlyMemory<T> source);
ValueTask GpuFence.WaitAsync(CancellationToken cancellationToken = default);
bool GpuFence.IsCompleted { get; }
void GpuStream.Synchronize();
```

Present synchronous overloads remain and mean submit on `DefaultStream` then
wait. Existing `wait:false` overloads become compatibility shims: they submit
on the default stream and return before completion, but cannot expose ordering
to callers. New code uses the returned fence. A fence is tied to its stream and
context. Cross-stream dependencies use `stream.Wait(fence)`; cross-context
dependencies and resource sharing are rejected until explicitly supported.

Native implementation must retain dispatch bindings, staging allocations, and
the compiled Luisa shader until the corresponding LC event completes. A C# task
or callback alone is not proof that GPU resources are safe to release. Exact
callback-thread affinity and LC host-callback guarantees on all targeted
backends are **pending verification**; `WaitAsync` must not require a particular
callback thread.

### Resource and interop model

All resources become explicitly context-owned and expose metadata required for
validation:

```csharp
GpuResourceInfo { Context, Size, Format, Usage, DebugName }
GpuExternalHandle TryExport(GpuExternalHandleKind kind)
GpuTexture2D<...> GpuContext.ImportTexture2D(GpuExternalTextureDesc desc)
```

`GpuExternalHandle` is an opaque, disposable capability carrying backend,
handle kind, ownership rule, and synchronization requirements. It must not be a
raw `IntPtr` property. Export/import is opt-in and initially limited to
backend/OS pairs proven by tests. It requires a fence or documented ownership
transfer; it never makes cross-context aliasing implicit.

The first intended consumer is a swapchain/presenter path, but the abstraction
also covers external compute and media APIs. The M8 baseline remains host-staged
for unsupported interop pairs. M8 may add asynchronous host transfer with fences
before zero-copy. Texture usage should separate sampled, storage, render target,
depth/stencil, transfer source, and transfer destination; current
`TextureAccess` is too coarse for that contract.

### Graphics and future RT

Graphics pipeline creation, draw, texture presentation, and future swapchains
take an owning `GpuContext` and optional `GpuStream`. Existing graphics methods
route to `DefaultStream` for source compatibility. Presentation is specified as
an acquire/render/release operation associated with a stream and fence, even
where the first backend implementation stages through host memory.

Reserve, but do not implement, this RT namespace:

```csharp
namespace Feather.RayTracing;
GpuMesh, GpuCurve, GpuProceduralPrimitive, GpuAccel,
GpuRayTracingPipeline, GpuShaderBindingTable, RayTracingDispatchDesc
```

All are context-owned resources. `GpuAccel.Build` and ray dispatch submit to a
`GpuStream` and return `GpuFence`. `GpuDeviceCapabilities.RayTracing` gates
construction before FEIR grows RT records. This mirrors LC's resource graph
without claiming that XIR, generated FEIR, Metal, Vulkan, and DX have current
parity. Their support matrix is **pending verification**.

### Cross-platform contract

The managed surface is identical on Metal, Vulkan, and DX. Semantics are:

| Contract | Required behavior |
| --- | --- |
| Device index | Index is scoped to `(runtime, backend)` enumeration and is never silently remapped |
| Stream order | Commands on one stream are ordered; different streams require an explicit fence dependency |
| Fence | Completion means all prior covered work is available to its declared consumer |
| Resource affinity | Binding a resource from another context fails before submission |
| Interop | Export/import is capability-gated; handle type, ownership, layout/state, and synchronization are explicit |
| Unsupported capability | Creation/dispatch fails with a diagnostic; it never routes to CPU or another GPU backend |

Backend-specific limitations belong in `GpuDeviceCapabilities` and test
matrices, not in behavior hidden behind the same API. M8.1 exposes the one
uniform numeric query LC 35a06cb provides after device creation, compute warp
size. Bindless-capacity sufficiency, subgroup, and quad support remain
`Unknown`. Stream classes, external memory and semaphore support,
presentability, timestamp/profiling, raster, ray tracing, and texture-format
queries remain **pending verification** for later milestones.

## Execution Plan And Acceptance Gates

### M8.1 implementation status

M8.1 is implemented by `GpuRuntime.EnumerateDevices`, `DefaultDevice`, and
`CreateContext(GpuContextOptions)`. Discovery is backed by LC
`Context::installed_backends` and `backend_device_names`; explicit selection is
validated by constructing an LC `Device` with `DeviceConfig.device_index`.
`compute_warp_size` becomes known after that construction. LC 35a06cb has no
uniform device query for bindless capacity sufficiency, subgroup operations, or
quad operations, so those three fields deliberately report `Unknown`; requiring
one fails before device creation instead of guessing support.

The existing `GPU.Context` remains the compatibility default and reports the
same platform-default Luisa device. M8.2 now gives non-default contexts native
resource and kernel ownership; stream ownership remains an M8.3 deliverable.

### M8.2 implementation status

M8.2 implements independent logical context ownership. `GPU.WithContext(context)`
returns a non-ambient `GpuContextOperations` facade for context-owned buffer
creation and 1D/2D/3D dispatch. It never replaces `GPU.Context`, so existing
static calls retain their default EasyGPU behavior while explicit contexts
execute through the Luisa backend and device selected by `GpuContextOptions`.
`GpuContext.Backend` and `GpuContext.Device` expose that immutable selection;
the legacy `BackendType`/`Caps` properties remain EasyGPU compatibility queries.

Native buffers, textures, samplers, kernels, and graphics pipelines record their
owner context. Kernel/pipeline bindings, render targets, index buffers, and AD
gradient destinations reject a different owner with `FE_ERROR_INVALID_ARGUMENT`.
Destroying an explicit context synchronizes and removes its Luisa runtime state,
kernels, pipelines, and resources; managed operations retain the owner and throw
`ObjectDisposedException` after context disposal. A `GpuKernel` also retains its
creator context and rejects dispatch through a different context before native
binding begins.

Luisa residency is stored in a `RuntimeRegistry` keyed by the Feather context
handle. Metal contexts hold independent LC `Context`/`Device`/`Stream` states.
The pinned Vulkan backend asserts that only one `Device` can be live because
Volk dispatch tables are process-global
([LuisaCompute/src/backends/vk/device.cpp](../LuisaCompute/src/backends/vk/device.cpp):513-528).
Feather therefore preserves multiple logical Vulkan contexts but synchronizes
and releases the previously active Vulkan state before reconstructing the next
context's selected device. Compute resources use their context-owned host state
when submitted again. This is correct isolation, not concurrent multi-device
Vulkan execution; removing that serialization requires an upstream LC change.

Local Apple M5 coverage creates two contexts selecting Metal device 0, alternates
real FEIR-to-XIR dispatches between them, verifies independent buffer results and
`DispatchPath.Luisa`, exercises managed and native cross-context rejection, and
checks disposal without changing `GPU.Context`. `GpuStream` and `GpuFence`,
including true concurrent submission, are implemented by M8.3.

### M8.3 implementation status

M8.3 adds `GpuContext.CreateStream`, `GpuStream.Dispatch`, `Synchronize`, and
`Wait`, plus `GpuFence.IsCompleted`, `Wait`, and `WaitAsync`. Explicit stream
dispatch always selects the context's Luisa device and returns immediately after
submission. Existing `GpuKernel.Dispatch(..., wait: true)` remains synchronous;
the compatibility `wait: false` route now retains native staging, readback,
uncached shader, and repacking state until the default stream completes.

The native runtime owns a stream set and LC events per Feather context. LC
35a06cb creates streams and events through `Device::create_stream` and
`create_event`
([LuisaCompute/include/luisa/runtime/device.h](../LuisaCompute/include/luisa/runtime/device.h):141-145).
`Event` provides signal, device wait, completion polling, and host synchronization
([LuisaCompute/include/luisa/runtime/event.h](../LuisaCompute/include/luisa/runtime/event.h):48-52),
while `Stream` accepts commands/events and exposes synchronization
([LuisaCompute/include/luisa/runtime/stream.h](../LuisaCompute/include/luisa/runtime/stream.h):85-95).
Feather uses a real event for each fence and `Event::is_completed` for polling.

Host-visible output repacking is part of Feather's fence completion contract.
For deterministic behavior across the pinned backends, `GpuFence.Wait` and
`GpuStream.Wait(fence)` synchronize the producer stream before releasing its
retained submission. Consequently, same-context cross-stream dependencies are
correct but `GpuStream.Wait` is currently host-blocking; independent streams
still submit and execute concurrently. Cross-context waits are rejected because
LC events belong to one device. Context, stream, buffer, and texture teardown or
host access synchronizes affected work before native storage is released, so an
in-flight dispatch cannot retain dangling host pointers. Autodiff remains
host-synchronous because gradient retrieval has no asynchronous managed contract.

| Milestone | Deliverable | Acceptance |
| --- | --- | --- |
| M8.1 Runtime discovery (implemented) | `GpuRuntime`, device enumeration, `GpuContextOptions`, LC-backed capability report | Device list/default semantics and invalid selection are covered by managed tests; a GPU test creates the selected LC device; multi-device hardware coverage remains conditional |
| M8.2 Context-native ownership (implemented) | `GPU.WithContext`; native per-context LC state; resources, pipelines, and kernels carry owner context | Two same-device Metal contexts dispatch independently; cross-context binding and post-disposal use are rejected; pinned Vulkan contexts are serialized because LC permits one live device |
| M8.3 Streams and fences (implemented) | Explicit streams, LC events, retained async staging, `GpuFence` | `wait:true` remains synchronous; `wait:false` retains staging safely; GPU tests cover completion polling/wait, two streams, cross-stream ordering, cross-context rejection, teardown, and Vulkan context activation switches |
| M8.4 Resource interop and presentation | Capability-gated external resource contract; native presenter/swapchain route where supported | Host staging fallback remains correct; every enabled zero-copy path has ownership/state/fence tests and a non-black multi-frame presentation test |
| M8.5 API migration | Static `GPU` facade delegates to default context/stream; samples migrate to explicit contexts where appropriate | Existing public examples compile unchanged; new multi-context/stream samples pass; no public API selects EasyGPU |
| M8.6 RT readiness | Capability schema and inert RT type/descriptor design review, then a separate implementation proposal | No fake RT success path; construction is capability-gated; FEIR/XIR/backend coverage proposal approved before implementation |
| M9 EasyGPU removal | Remove EasyGPU ABI, source, assets, enum branch, GLSL inspection APIs, and CI setup after M8 gates | Repository has no EasyGPU runtime dependency; supported compute/graphics/window paths use LC; package and three-platform build gates pass |

M9 depends on M8.1 through M8.5. It must not begin deletion while window
presentation or asynchronous resource ownership still relies on EasyGPU. M8.6
is a design dependency for a later RT milestone, not a prerequisite for M9's
compute and graphics migration.

## Open Questions Requiring Validation

1. Which backend-specific LC extensions can promote M8.1's `Unknown` capability
   fields to reliable values without changing their cross-platform semantics?
2. Can the pinned Metal, Vulkan, and DX backends all import/export the native
   images/buffers required for zero-copy presentation, with an explicit
   synchronization primitive usable by Feather?
3. What host-callback and event-lifetime guarantees apply to all targeted LC
   backends, particularly during runtime/context shutdown?
4. Which RT, raster, external-memory, multi-stream, and profiling features are
   uniformly available through the FEIR-to-XIR route rather than LC DSL-only
   APIs?
5. Does managed API review prefer `GpuRuntime`/`GpuContext` terminology above,
   or a renamed `GpuDevice` for the object currently called `GpuContext`? The
   selected names must be settled before M8.1 ships.
