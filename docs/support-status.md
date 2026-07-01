# Support Status

Feather is experimental. This page describes the intended stability of each public area and the current source-tree packaging reality.

## Capability Levels

| Area | Status | Notes |
| --- | --- | --- |
| C# compute kernels over buffers | Preview-stable | Main supported path; generated typed IR dispatches through EasyGPU when the kernel is in the supported subset. |
| Vector, matrix, scalar math | Preview-stable | Includes `float2/3/4`, `int2/3/4`, square float matrices, swizzles, and `ShaderMath`/`Hlsl` helpers. |
| 2D and 3D textures | Preview | Load/store, sampling, mipmap requests, and TGA IO are available for supported formats. |
| Windows and texture presentation | Preview | Native window support depends on EasyGPU window support and platform libraries. |
| Raster graphics pipeline | Preview | Vertex and fragment shaders written in C#; API may still evolve. |
| Automatic differentiation | Preview | 1D generated kernels, differentiable float and whole-vector buffer values, one scalar loss. |
| Neural-network helpers | Preview | Useful for small models and samples; not a drop-in replacement for mature ML frameworks. |
| Profiler and shader inspection | Preview | Useful for diagnostics; exact optimized output depends on backend. |
| NuGet packaging | Preview | Native assets are staged under `artifacts/native-assets` and packed through `Feather.NativeAssets`; publish only packages built from a full RID matrix. |

## Summary

| Area | Level | Notes |
| --- | --- | --- |
| Buffer compute kernels | Most mature | Core path, samples assert `TypedEasyGpu`. |
| 2D texture compute | Mature preview | Core formats are proven; some formats are backend-dependent. |
| 3D texture compute | Preview | API and native bridge exist; coverage is narrower than 2D. |
| Windowing | Preview | Presentation shell; platform-thread caveats apply. |
| Graphics pipeline | Preview | Real graphics path, but state surface is still evolving. |
| Automatic differentiation | Preview | Real EasyGPU AD bridge with explicit failures; source subset is narrower. |
| `Feather.NN` | Preview | Explicit small-model helper layer, not a full ML framework. |
| FEIR/native ABI | Advanced/internal | Documented for debugging and contributors, not a stable external file format. |

## Platform And Native Assets

The native bridge is built from `native/` and links EasyGPU. Source checkouts
build `libfeather` or `feather.dll` locally; published packages supply the
matching RID native asset under `runtimes/<rid>/native`.

| Platform | Current use path | Notes |
| --- | --- | --- |
| macOS arm64 | Local CMake build or packaged asset | Vulkan path uses MoltenVK through EasyGPU. |
| macOS x64 | Local CMake build | Requires matching native library in resolver path or `FEATHER_NATIVE_LIBRARY`. |
| Windows x64/arm64 | Local CMake build | Requires C++ toolchain and backend dependencies. |
| Linux x64/arm64 | Local CMake build | Window support needs X11 development libraries. |

## Backend Notes

Feather does not choose a backend at runtime. The native EasyGPU build is configured at CMake time:

```bash
cmake -S native -B native/build -DEASYGPU_BACKEND=Vulkan
cmake -S native -B native/build-gl -DEASYGPU_BACKEND=OpenGL
```

The active backend can be queried:

```csharp
Console.WriteLine(GPU.Context.BackendType);
Console.WriteLine(GPU.Context.Caps);
```

## Dispatch Paths

`DispatchPath` reports how the latest dispatch or draw ran:

| Path | Meaning |
| --- | --- |
| `TypedEasyGpu` | Generated typed IR was accepted by the native EasyGPU path. |
| `CpuReferenceFallback` | Compatibility/reference path used for legacy or test payloads. This should not be the normal path for new generated kernels. |
| `GraphicsFallback` | Graphics fallback route used for unsupported or compatibility draw behavior. |
| `Rejected` | Native bridge or backend rejected the operation. |
| `None` | Nothing has run yet. |

For samples and tests, prefer:

```csharp
var path = GPU.DispatchAndGetPath(new MyKernel(...), count);
```

Samples that call `SampleProof.AssertTypedEasyGpu(path)` are intended to prove the real typed backend route. When adapting a sample, keep the dispatch-path check until your custom kernel is stable.

## Qualification Guidance

Before shipping a workload, qualify the exact OS, GPU, driver, backend, and kernel subset you use. A successful run on one backend or RID does not prove support on another.

Practical checklist:

- Run the matching sample or test on the target platform.
- Confirm the dispatch/draw path is `TypedEasyGpu`.
- Check generated GLSL or optimized GLSL when debugging shader behavior.
- Read the capability notes for preview surfaces.

## Related Docs

- [Examples](examples.md)
- [FEIR Compiler Pipeline](feir.md)
- [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md)
- [Native ABI](native-abi.md)
