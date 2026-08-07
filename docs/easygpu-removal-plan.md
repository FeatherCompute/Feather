# M9 EasyGPU Removal Plan

> **Status: COMPLETED** — EasyGPU has been removed as source, build, runtime,
> public API, test-oracle, and packaging dependency (commits `95839c3` through
> `618b45b`). LuisaCompute is Feather's only execution runtime. This document
> is retained as the historical plan and recovery record.

## Scope, Decisions, And Recovery

M9 removes EasyGPU as a source, build, runtime, public API, test-oracle, and
packaging dependency. LuisaCompute becomes Feather's only execution runtime.
The automatic backend policy remains Metal on macOS and Vulkan elsewhere;
explicitly selected LC backends continue to follow `GpuContextOptions`.

The pre-removal recovery point is commit `a710b1666ac6392da63ba9c0305beb141b10397d`.
The work is split into reviewable commits in this order: plan, native, managed,
authorized tests, CI/engineering, and documentation. No stage is pushed before
leader review. The EasyGPU gitlink is removed only in the native commit, so the
entire deletion can also be recovered atomically from the recovery point.

These decisions constrain the implementation:

* CPU execution is not introduced. Compute, AD, NN, graphics, and presentation
  use LC devices and streams.
* Compute rasterization becomes the only graphics draw implementation. The
  `FEATHER_GRAPHICS_COMPUTE` switch and fixed-function EasyGPU fallback are
  removed rather than repurposed.
* The existing GLFW event host is retained as Feather-owned code. GLFW becomes
  a direct, pinned CMake dependency instead of arriving through EasyGPU.
* Same-device LC swapchain presentation is the preferred window path. The R11
  asynchronous host-staging ring remains only as a capability-gated LC
  fallback; its destination/upload path must also be LC-owned.
* EasyGPU GLSL and optimization inspection are breaking-change removals. FEIR
  metadata and backend-neutral layout APIs are retained; no fake XIR/SPIR-V
  source is substituted.
* Test authorization is limited to backend-specific assertions and fixtures.
  Numerical, resource, control-flow, AD/NN, graphics, ownership, stream, and
  presentation behavior remains tested.

## Native Dependency Inventory

| Dependency point | Evidence | Disposition |
| --- | --- | --- |
| Git dependency | `.gitmodules` and the root `EasyGPU` gitlink | **Delete.** Preserve LuisaCompute at its current pin and pristine state. |
| Native build | `native/CMakeLists.txt` sets `EASYGPU_*`, adds the subdirectory, links `EasyGPU::EasyGPU` and `EasyGPU::Window`, and links probes to EasyGPU | **Delete/migrate.** Require LC, link only LC plus direct pinned GLFW when windows are enabled, and remove obsolete EasyGPU benchmark options and targets. |
| Backend state and diagnostics | `native/feather_c_api.cpp` owns EasyGPU backend objects, guards, handles, caches, and error translation | **Delete.** Context state retains only per-context LC runtime/device/stream/resource ownership and reports LC errors. |
| Typed IR bridge | `native/feather_typed_ir_lowerer.*` exposes `GPU::IR::Module` and `TryLowerToEasyGpuModule`; `feather_ir_bridge.cpp` includes legacy bridge support | **Delete EasyGPU output only.** Preserve typed FEIR parsing, validation, and the inputs consumed by `feather_luisa_xir.cpp`/`feather_luisa_backend.cpp`. |
| Compute dispatch | EasyGPU module/source construction, pipeline compilation, binding, dispatch, and dirty-state helpers in `feather_c_api.cpp` | **Migrate to LC.** Kernel creation has one compiled LC form, static dispatch uses the owning context's default stream, explicit streams retain M8.3 behavior, and `wait` semantics do not change. |
| Resource residency | Buffer/texture states contain EasyGPU handles and EasyGPU upload/download/mapping/mipmap branches | **Migrate to LC.** Keep host shadows only where the public mapping or presentation fallback requires them; all device upload, download, copy, mip generation, and dirty tracking use the owning LC context. |
| Automatic differentiation | EasyGPU tape/module/merged GLSL/reduction paths coexist with LC AD in `feather_c_api.cpp` | **Delete EasyGPU AD.** Keep FEIR-to-XIR reverse-mode AD, LC gradient buffers, reduction, readback, and named-gradient metadata. |
| NN execution | Managed NN calls static dispatch; native execution consequently enters the default EasyGPU route today | **Migrate transitively and verify explicitly.** Default dispatch becomes LC and the complete NN training/inference suite is a native-stage gate. No NN algorithm is removed. |
| Graphics compilation and draw | EasyGPU pixel-format mappings, vertex/fragment GLSL, fixed raster pipeline caches, and `draw_graphics_pipeline_easygpu` | **Delete/migrate.** Compute raster is unconditional, retains depth/stencil/blend/MRT/MSAA/mipmap/index/instance behavior, and returns the LC dispatch path. |
| Window and input | `feather_window_host.*` uses EasyGPU `AppWindow`, configuration, input/event enums, and presenter fallback; its native path already uses GLFW | **Migrate.** Define Feather-owned native window/event/input types matching the C ABI, use direct GLFW on all desktop platforms, remove `EasyGpuWindow`, and keep native display/window extraction for LC swapchains. |
| Presentation | `feather_c_api.cpp` can still download/upload through EasyGPU when direct LC presentation is unavailable | **Migrate.** Prefer LC swapchain; retain the three-slot async ring only with LC readback/staging upload. Unsupported backend/window pairs fail with a capability error, never silently select another runtime. |
| Optimization tooling | `native/optimization_benchmark.cpp` and its CMake target compare EasyGPU shader optimization | **Delete.** Historical measurements remain documentation; LC profiling and Sponza tools remain. |
| ABI enums and structs | `native/feather_c_api.h` contains execution/dispatch enums and EasyGPU-named gradient fields | **Migrate in lockstep with managed ABI.** Remove EasyGPU enum values, rename backend-specific metadata, preserve struct layout/version checks where required, and reject stale callers clearly. |

## Managed Dependency Inventory

| Dependency point | Evidence | Disposition |
| --- | --- | --- |
| Backend selection | `GpuExecutionBackend.EasyGpu` in `Core/Enums.cs`; explicit backend overloads and dual caches in `GPU.cs`/`GpuKernel.cs` | **Remove EasyGPU selection.** A sole `Luisa` value may remain for source migration, but default and explicit creation both use the context-selected LC device; delete the EasyGPU cache. |
| Dispatch diagnostics | `DispatchPath.TypedEasyGpu` plus native mirrors in `Feather.Native/NativeStructs.cs` | **Remove/renumber in coordinated ABI change.** `DispatchPath.Luisa` identifies compute, AD, NN, and compute-raster execution. Do not use `GraphicsFallback` to disguise LC work. |
| Context compatibility | `GpuContext.BackendType`/`BackendCaps` expose legacy EasyGPU data | **Remove or replace with M8 `GpuDeviceInfo`/`GpuDeviceCapabilities`.** `GPU.Context` remains the default LC context and `GPU.WithContext` remains source-compatible. |
| Kernel and AD defaults | `GpuKernel.Create` and `GpuADKernel` default to EasyGPU; static `GPU` has backend-selecting overloads | **Migrate to LC.** Preserve synchronous and M8.3 stream/fence overload semantics while removing obsolete backend choice. |
| Shader inspection | `ShaderInspection.GetGLSL`, `GpuKernel.GetGLSL`, optimized GLSL calls, and related native entry points | **Delete.** Keep generated FEIR metadata APIs. Add a future LC inspection API only when LC exposes a truthful stable contract. |
| Value layouts | `GpuValueLayout` and generated std430 metadata use EasyGPU terminology in comments/test names | **Preserve behavior, rename terminology.** Layout is part of Feather's FEIR/native ABI and must be checked against the LC bindings before deleting only EasyGPU-specific helpers. |
| NN trace | `NnDispatchTrace` currently records `TypedEasyGpu` through static dispatch | **Migrate to `Luisa`.** Retain every NN result, optimizer, checkpoint, and training assertion. |
| Render host | Render results and protocol fixtures serialize/expect `TypedEasyGpu` | **Migrate protocol diagnostics to `Luisa`.** Preserve frame payload, graph, camera, timing, and process behavior. Document the protocol enum change. |
| Package metadata | `Feather.Native`, `Feather.NativeAssets`, and main project descriptions name EasyGPU | **Update in the documentation/package stage.** Package IDs and LC runtime asset layout stay unchanged. |

## Test Treatment Boundary

The following edits are authorized because their only subject is the deleted
backend. They will be made in a dedicated commit so review can distinguish them
from implementation changes.

| Test area | Authorized change |
| --- | --- |
| `ProjectShapeTests` | Remove EasyGPU submodule/module-lowering shape tests. Replace the native-link assertion with LC-only build/link and absence checks. Preserve all unrelated repository-shape tests. |
| `GeneratedComputeDispatchTests` and `GeneratedGraphicsPipelineTests` | Remove only `GetGLSL`/EasyGPU-source assertions and rewrite `TypedEasyGpu` route assertions to `Luisa`. Preserve generated metadata, numeric output, bindings, dimensions, resource order, control-flow, and graphics-state assertions. |
| `GraphicsDrawFallbackTests` | Rename/reframe as compute-raster default tests and expect `Luisa`; preserve every image/depth/stencil/blend/MRT/index/instance result assertion. |
| `ADSurfaceTests`, `AutoDiffNativeBridgeTests`, `MlpTrainingGradientTests`, and `NNTrainingIntegrationTests` | Expect `Luisa`, migrate the EasyGPU-named gradient metadata field, and preserve all gradient values, training loss, reductions, and lifecycle assertions. |
| `NNSurfaceTests` | Rewrite path assertions and the EasyGPU-named initialization test to backend-neutral/LC terminology. Preserve all model, tensor, optimizer, checkpoint, and expected-value assertions. |
| Render-host tests | Change `TypedEasyGpu` result/protocol fixtures to `Luisa`; preserve frame pixels, pass ordering, IPC, and lifecycle assertions. |
| GLSL inspection tests | Delete individual tests whose sole contract is EasyGPU GLSL text in `ShaderDslCoverageTests`, `ShaderCameraTests`, `ShaderNoiseTests`, `MlpLoweringBoundaryTests`, and `MlpInferenceSmokeTests`. Where a test also validates FEIR or runtime behavior, remove only the GLSL assertion and retain the functional test. |
| `NativeResourceRoundTripTests` and `RenderGraphBufferGpuTests` | Change backend-name/path expectations to LC while preserving resource round-trip and graph behavior. |
| Generator/graphics surface tests | Rename EasyGPU-specific test names/comments. Preserve std430 byte offsets, generator diagnostics, topology values, and sample-count assertions unless the removed public API makes an assertion impossible. |

The `LuisaBackend*Tests` and the 23-case Metal parity collection are retained.
Their `MatchEasyGpu` names and dual-runtime oracle helpers are changed to
deterministic expected-output or LC-only functional checks; the Luisa dispatch,
expected values, and feature coverage remain. No case may be dropped merely
because its former oracle was EasyGPU. Existing ownership, stream/fence,
device-discovery, compute-raster, graphics, capture, and window tests are not
authorized for weakening.

A new assertion verifies that a kernel created through the static/default API
dispatches via `DispatchPath.Luisa`. The test matrix must also exercise explicit
LC contexts and the static facade in one process, as M8.5 already requires.

## Samples, Engineering, CI, And Documentation

| Area | Dependency | Disposition |
| --- | --- | --- |
| Compute samples | `HelloWorld`, `HelloBuffer`, `ColorFilter`, `Histogram`, `ParallelReduction`, `TextureCopy`, `Mandelbrot`, `JuliaSet`, `SdfRenderer`, `VolumetricFog`, and related samples call `AssertEasyGpuGlsl` | **Migrate.** Remove GLSL proof helpers and assert/print LC dispatch plus existing output/PASS evidence. |
| Backend-selecting samples | `RayTracing` accepts `easygpu|luisa`; `ProfilerSuite` expects `TypedEasyGpu` | **Migrate.** Remove the EasyGPU option, make LC unconditional, and preserve numerical/image/path checks. |
| Inspection sample | `SpirvOptInspection` exists to inspect EasyGPU GLSL/optimization | **Retire or convert to truthful generated FEIR inspection.** It must not claim LC shader source availability. |
| Graphics/window samples | Raster and presentation previously documented an EasyGPU fallback | **Migrate.** Compute raster and LC swapchain/LC staging are the only paths; keep input/window behavior. |
| Native build script | `eng/build-native.sh` accepts `FEATHER_EASYGPU_BACKEND` | **Delete.** LC backend/configuration variables remain. |
| CI workflows | `ci.yml` and `release.yml` pass `EASYGPU_*` CMake arguments; native smoke implicitly links EasyGPU | **Delete/migrate.** Add direct GLFW prerequisites if the pinned dependency requires them; keep Linux/macOS/Windows native-smoke jobs equivalent. |
| Native staging/package | Staging currently copies the Feather library that links EasyGPU and validates mixed runtime assets | **Rebuild and audit.** Stage only Feather plus complete LC runtime assets; inspect `otool -L`, `ldd`, or `dumpbin` and unpack all NuGet packages to prove no EasyGPU binary/runtime dependency remains. |
| Coverage gates | AD/NN scripts and diagnostics may match EasyGPU route names | **Migrate patterns to LC** without reducing case counts or numerical gates. |
| Documentation | README, packaging, examples, compute-raster, AD/NN, FEIR, and API alignment pages describe EasyGPU | **Rewrite current behavior.** Keep a concise breaking-change migration note and historical benchmark attribution where relevant. Remove links to the deleted gitlink/license. |

## Commit And Verification Sequence

1. `docs(m9): plan EasyGPU removal`
   Adds only this inventory and plan.
2. `refactor(native): remove EasyGPU runtime`
   Introduces the direct window dependency and Feather-owned event types,
   migrates every native execution/resource/AD/graphics/presentation path to
   LC, removes obsolete native tooling, and deletes the gitlink/CMake wiring.
   Gate: clean Release native build, LC submodule pristine, native dependency
   inspection, compute/AD/NN/graphics/window smoke tests.
3. `refactor(api): make Luisa the sole backend`
   Removes managed selection, dual caches, legacy capabilities, shader
   inspection, and ABI mirrors; updates samples enough for a solution build.
   Gate: managed build plus API/ABI tests, default and explicit context smoke.
4. `test(m9): migrate EasyGPU-specific coverage to Luisa`
   Applies only the authorized table above and adds the default-LC assertion.
   Gate: all non-GPU tests, all GPU suites, 23 Metal cases, AD 66, NN 78,
   compute-raster 21, graphics, integration, and render-host suites.
5. `ci(m9): remove EasyGPU build and packaging inputs`
   Cleans scripts/workflows/staging/package validation. Gate: fresh native
   configure/build/stage/pack, five-package archive audit, and YAML validation.
6. `docs(m9): document Luisa-only runtime`
   Updates capability matrices, examples, packaging, and migration guidance.
   Gate: sample solution build/run matrix and markdown-link validation.

After the six reviewed local commits, the final release gate is a clean clone
with recursive submodules, three-platform native-smoke, managed pack, and the
full test matrix. `git submodule status` must list only LuisaCompute, and LC must
remain pristine. A repository search may retain EasyGPU only in the M9 migration
record and clearly labeled historical benchmark context, never in source,
project files, scripts, tests, samples, current capability docs, or package
metadata.
