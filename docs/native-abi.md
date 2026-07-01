# Native ABI

Feather's managed API talks to native code through a C ABI declared in `native/feather_c_api.h`. Normal users usually interact with this through `GPU`, resources, windows, graphics pipelines, and AD wrappers. You should read this page when:

- `DllNotFoundException` or native loading fails.
- A backend feature is rejected at runtime.
- You need to package native assets.
- You are debugging dispatch paths, graphics draw paths, AD gradients, or profiler output.
- You want to contribute to the managed/native bridge.

## Boundary Model

```text
Feather managed API
  -> Feather.Native P/Invoke layer
  -> native Feather C ABI
  -> EasyGPU runtime and backends
```

The ABI intentionally avoids C++ classes, templates, STL containers, exceptions, and ownership transfer across the boundary. Managed code owns `SafeHandle` wrappers; native code returns `FeResult` values and records a per-thread last-error string.

## Handles And Errors

Native handles are represented as integer-backed objects:

- Context
- Buffer
- Texture
- Sampler
- Kernel
- Graphics pipeline
- AD kernel state
- Window
- Texture presenter

The managed layer wraps these as disposable objects. Native calls return `FeResult`; failures become `FeatherNativeException` with the native last-error string.

## Library Loading

The native loader searches packaged runtime assets and development paths. Override it with:

```bash
export FEATHER_NATIVE_LIBRARY=/absolute/path/to/libfeather.dylib
```

Use the appropriate extension for your platform:

| Platform | Typical native name |
| --- | --- |
| macOS | `libfeather.dylib` |
| Linux | `libfeather.so` |
| Windows | `feather_native.dll` |

See [Packaging](packaging.md) for project and asset guidance.

## Backend Initialization

`fe_context_initialize`, `fe_context_get_backend_type`, and `fe_context_get_caps` initialize/query the EasyGPU runtime. If the backend cannot initialize, Feather returns `FE_ERROR_BACKEND_UNAVAILABLE` instead of reporting placeholder capabilities.

`BackendCaps` exposes:

- Backend type.
- Max workgroup dimensions.
- Graphics support flags.
- AD/NN capability flags.
- Optional raster features such as depth clamp and non-fill polygon modes.

## Compute Dispatch

The core path is typed FEIR -> EasyGPU module lowering:

1. Managed generated kernel provides serialized FEIR and resource descriptors.
2. Native `fe_kernel_create_from_ir` validates and stores the payload.
3. Managed binding code binds buffers, textures, samplers, and push constants.
4. `fe_kernel_dispatch` receives logical dispatch size and backend workgroup counts.
5. Native lowering registers resources with EasyGPU `ModuleBuilder`.
6. EasyGPU lowers and dispatches the kernel.

`[Kernel(BoundsCheck = true)]` adds hidden logical-dispatch-size push constants and a synthetic early-return guard. `[Kernel(BoundsCheck = false)]` omits that guard.

Dispatch records one of:

| Path | Meaning |
| --- | --- |
| `FE_DISPATCH_PATH_TYPED_EASYGPU` | Supported typed EasyGPU route succeeded. |
| `FE_DISPATCH_PATH_CPU_REFERENCE_FALLBACK` | Narrow compatibility/reference fallback. |
| `FE_DISPATCH_PATH_REJECTED` | Validation or backend lowering rejected the payload. |

Managed code exposes this through `DispatchPath`.

## Resource ABI

Buffers carry:

- Byte size.
- Access mode.
- GPU-side element stride.

Texture descriptors carry:

- Width, height, and optional depth.
- Mip level count.
- Pixel format.
- Access mode.

Sampler descriptors carry filter, address, mip, comparison, anisotropy, and border-color settings.

The typed EasyGPU texture bridge supports the runtime formats documented in [Support Status](support-status.md). Unsupported formats return explicit unsupported-format errors instead of falling back silently.

## Layout Bridge

Managed `GpuBuffer<T>` computes GPU element stride through `GpuValueLayout<T>.BufferElementStride`, mirroring EasyGPU std430 layout rules:

- `float`, `int`, `uint`: 4-byte stride.
- `float2`, `int2`: 8-byte stride.
- `float3`, `float4`, `int3`, `int4`: 16-byte stride.

A `float3` buffer therefore stores a 12-byte CPU payload in each 16-byte GPU slot. Generated push constants and `[GpuStruct]` fields use std430-like field alignment and explicit generated layout metadata.

## Graphics ABI

Graphics pipeline creation passes:

- Combined pipeline IR.
- Explicit vertex and fragment FEIR blobs.
- Topology and sample count.
- Depth/stencil state.
- Blend state and MRT blend attachments.
- Raster state.
- Color attachment count.

Graphics draws render to offscreen `FeTextureHandle` color attachments, not window swapchains. Supported paths include:

- Rgba8/Rgba32Float color targets.
- Optional matching depth/stencil textures.
- MSAA with resolve back into the color target.
- MRT arrays.
- Explicit per-draw `uint`/`ushort` index buffers.
- Texture sampling and sampler binding.

Unsupported draw shapes return `FE_ERROR_UNSUPPORTED` or `FE_ERROR_INVALID_ARGUMENT` and record `FE_DISPATCH_PATH_REJECTED`.

## Window ABI

Window support includes:

- `fe_window_create`, destroy, open/close, poll/wait events.
- Size/title/vsync/input queries.
- CPU pixel presentation.
- Texture presenter creation and texture presentation.

Native window support is controlled by `FEATHER_BUILD_WINDOW`, enabled by default:

```bash
cmake -S native -B native/build -DFEATHER_BUILD_WINDOW=OFF
```

Use this for headless/core-only builds. On macOS, create and poll windows on the main thread.

## AD Gradient ABI

FEIR AD metadata is authoritative. Native AD handling:

- Registers buffer parameters with EasyGPU `GradientTape`.
- Marks scalar losses by generator-emitted identity.
- Generates merged forward/backward GLSL.
- Dispatches with separate gradient SSBOs.
- Exposes gradient count, metadata, readback, reduction-to-buffer, and backward GLSL inspection.

Gradient query/readback entry points:

- `fe_kernel_get_ad_gradient_count`
- `fe_kernel_get_ad_gradient_info`
- `fe_kernel_read_ad_gradient`
- `fe_kernel_reduce_ad_gradient_to_buffer`
- `fe_kernel_get_ad_backward_glsl`

Gradient buffers never alias parameter value buffers.

## Profiler ABI

Profiler functions mirror EasyGPU's global profiler shape:

- Enable/disable.
- Clear.
- Query by name.
- Total time.
- Formatted report.

Managed wrappers live in `GpuProfiler`.

## Debugging Checklist

1. Confirm the native library loaded from the path you expect.
2. Print `GPU.Context.Caps`.
3. For compute, inspect `GPU.DispatchAndGetPath(...)`.
4. For graphics, inspect `pipeline.LastDispatchPath`.
5. For AD, inspect `ad.LastDispatchPath` and `ad.GetBackwardGLSL()`.
6. Read the native exception message; unsupported features are meant to name the rejected shape.
7. Cross-check [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md) and [Support Status](support-status.md).
