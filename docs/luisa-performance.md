# Luisa Performance

This document records the reproducible baseline and each M7 runtime/code-generation
experiment. EasyGPU remains Feather's default backend until the separate default-switch
milestone.

## Benchmark method

- Scene: `samples/RayTracing`, the Cornell-box compute path tracer.
- Fixed workload: 1024 x 1024 pixels, 10,240 samples per pixel. Backend selection does not
  change the kernel, buffers, random seeds, output checks, resolution, or sample count.
- Command: `dotnet run --no-build --project samples/RayTracing/RayTracing.csproj -- 1024 1024 10240 <easygpu|luisa>`.
- Metric: the sample's `Render time`, starting immediately before dispatch and ending after
  the requested synchronous dispatch returns. Each backend runs in a fresh process three
  times; the reported result is the median. Every measured run must print its expected
  dispatch path and `PASS`.
- Host: Apple M5 (10 GPU cores), macOS 26.5.2, Vulkan SDK 1.4.350, MoltenVK 1.4.1,
  arm64 Release native build. `FEATHER_SHADER_OPTIMIZATION_LEVEL=Ultra` is the EasyGPU
  configuration default.

The sample accepts an optional fourth argument, `easygpu` or `luisa`, solely to make the
same generated kernel directly comparable. Its default remains `luisa`.

## M7 baseline

Measured on 2026-08-05 at commit `709c55e`, before M7 runtime or code-generation changes:

| Backend | Run 1 | Run 2 | Run 3 | Median | Relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| EasyGPU | 30.077 s | 30.849 s | 30.076 s | 30.077 s | 1.00x |
| Luisa Vulkan, XIR-SPIR-V | 55.609 s | 56.449 s | 57.609 s | 56.449 s | 1.88x |

All six runs passed. The image statistics were 1,041,954 lit pixels and 60,788 or
60,789 distinct quantized colors; the one-count variation is a backend floating-point
rounding difference, not a quality-setting difference.

## Baseline cost decomposition

The pre-M7 Luisa implementation constructed a `Context`, Vulkan `Device`, and compute
`Stream`, created and uploaded device buffers, translated and compiled the shader, dispatched,
downloaded writable buffers, and destroyed everything inside every `Dispatch` call.

A 1024 x 1024 construction experiment at one sample per pixel measured 33 ms on EasyGPU
and 4.956 s on Luisa. Luisa's timestamps placed context/plugin/device startup at about 4 ms
and its SPIR-V optimization/compilation completion 4.758 s after device creation. The
remaining roughly 0.19 s contains FEIR-to-XIR/AST translation, allocation, about 20 MiB of
buffer upload/download staging, synchronization, and the very small one-sample GPU workload;
these components are not separately instrumented in the baseline.

Comparing the 10,240-sample median with the one-sample construction run shows that fixed
startup cannot explain the 26.372 s median gap: after subtracting the 4.923 s one-sample
Luisa/EasyGPU delta, about 21.45 s remains attributable primarily to generated shader quality
and GPU execution. The resident-runtime and staging results below confirm that the single-
dispatch Cornell benchmark does not amortize startup.

## Vulkan code-generation survey

Pinned LuisaCompute 0.9.0 exposes mutually exclusive compute paths through
`LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV` and `LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV`:

| Feather option | LC route | Build/smoke result | Decision |
| --- | --- | --- | --- |
| `XirSpirv` | XIR → native SPIR-V (`XIR_TO_SPIRV`) | Release build succeeded; Cornell 64×36@4 spp passed in 2.915 s | Retain for M7 |
| `HlslSpirv` | AST → HLSL → DXC → SPIR-V (both flags OFF) | Build succeeded after supplying Vulkan SDK `libdxcompiler.dylib`, but DXC rejected generated code: illegal 4-element `float3` initializer and repeated `--0x...` lvalues | Not stable; do not select |
| `LlvmSpirv` | AST → LLVM → SPIR-V (`AST_LLVM_TO_SPIRV`) | Configuration stopped because no `LLVMConfig.cmake`/LLVM development package is installed on this host; no runtime result | Not selectable here |

The LC source confirms that the official HLSL+Vulkan recommendation is the first row's
alternative (`HlslSpirv`), not a separate FEIR route. Our FEIR→XIR→AST bridge can reach
that entry point, but the pinned HLSL generator is not compatible with the generated AST
for this kernel. The new `FEATHER_LUISA_VULKAN_CODEGEN` CMake cache option makes all three
choices explicit without changing the default (`XirSpirv`).

## SPIR-V optimization alignment

EasyGPU `Ultra` is SPIRV-Tools' maintained performance recipe plus LICM, strength
reduction, redundancy elimination, code sinking, and cleanup. LC 0.9.0 exposes no
`Ultra` enum: its `full` preset is `RegisterPerformancePasses` plus private-to-local
and copy-propagation, while `compute` is a smaller XIR-oriented pass list.

On the same 64×36@4 spp smoke (fresh process; all runs passed), Luisa dispatch times
were: `none` 4.377s (11,026 words), `lightweight` 4.394s (10,385), `compute` 4.310s
(10,384), and `full` 4.256s (10,271). The strongest stable LC choice is therefore
`full`. Feather now defaults the Luisa process to `LUISA_SPIRV_OPT_PASSES=full` when
the caller has not supplied an override; an explicit environment value remains
authoritative. This is the closest available LC equivalent to EasyGPU Ultra, not a
claim of pass-for-pass identity.

## Resident runtime and staging result

M7 now owns one Luisa `Context`/Vulkan `Device`/compute `Stream` per native Feather context.
The owner is explicitly reset during context shutdown and intentionally abandoned on process
exit, before any dynamically loaded backend can be unloaded. A shader cache keyed by the
generated kernel handle and bound resource handles retains the compiled shader and its device
buffers/images; repeated dispatches therefore reuse shader compilation and device allocation.
The `WindowCompute` sample was run for multiple frames: its log contained one context/device
creation and one SPIR-V compilation, followed by successful continuous dispatches.

The retained device buffers/images are the device-resident staging pool. Host packing and the
required upload/download copies remain because Feather and Luisa do not expose a safe shared
Vulkan allocation/import contract (queue-family ownership, image layout, and synchronization
would otherwise be undefined). No cross-runtime Vulkan import was attempted.

`wait:false` remains rejected by the native Luisa path; keeping synchronous readback semantics is
required by the current API. Multi-frame asynchronous submission is deferred to M8+.

The Cornell benchmark is one synchronous dispatch per fresh process, so resident runtime does
not amortize its startup in that measurement. The post-M7 fixed-workload rerun was:

| Backend | Run 1 | Run 2 | Run 3 | Median | Relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| EasyGPU | 28.517 s | 29.108 s | 30.032 s | 29.108 s | 1.00x |
| Luisa Vulkan, XIR-SPIR-V | 62.267 s | 60.093 s | 61.092 s | 61.092 s | 2.10x |

All six post-M7 runs printed the expected dispatch path and `PASS`. The gap is not yet within
the M7 target; generated shader/GPU execution dominates after startup and needs a separate M8
investigation.

## TRACK-F: FEIR to multiple AST backends

The experimental backend selector preserves one front end for every target:
`FEIR -> XIR -> XIR2AST -> AST -> Context::create_device(backend)`.
The default remains `vk` (the XIR-to-SPIR-V route). Set `FEATHER_LUISA_BACKEND=metal`
to opt into the Apple Metal AST compiler; `cuda` and `hip` are accepted selector values
and fail with an explicit unavailable-backend error when their native backend was not built.
The existing callable inlining and XIR verification happen before XIR2AST and are shared
by all selected backends.

### Backend matrix

| Backend | Build/configuration | Runtime result | Decision |
| --- | --- | --- | --- |
| Vulkan (`vk`) | ON by default; `XirSpirv` | Cornell and default HelloWorld PASS | Current default |
| Metal (`metal`) | macOS only; `-DFEATHER_LUISA_ENABLE_METAL=ON` | Builds and Cornell PASS, but 19/23 Luisa tests pass; 4 test hosts crash in LC Metal compilation | Experimental, default OFF |
| CUDA (`cuda`) | Requires `CUDAToolkit 12.1` (`cuda_driver`, `nvrtc_static`) | `nvcc`/toolkit absent on this host; explicit selector reports backend not built | Build-ready, unverified |
| HIP (`hip`) | Requires HIP/ROCm, `hiprtc`, and HIPRT toolchain | `hipcc`/ROCm absent on this host; explicit selector reports backend not built | Build-ready, unverified |

Metal smoke created `Metal device 'Apple M5' at index 0`; Cornell at 64x36@4 spp
completed in 3,094 ms. The fixed full workload (2026-08-05, Apple M5, macOS 26.5.2,
Release, three fresh processes per backend) produced:

| Backend | Run 1 | Run 2 | Run 3 | Median | Relative to EasyGPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| EasyGPU | 28.854 s | 28.500 s | 29.092 s | 28.854 s | 1.00x |
| Luisa Vulkan | 57.292 s | 56.710 s | 59.174 s | 57.292 s | 1.98x |
| Luisa Metal | 35.519 s | 33.598 s | 31.206 s | 33.598 s | 1.16x |

All nine Cornell runs printed `PASS`, with 1,041,954 lit pixels. Metal is faster than
Vulkan but does not meet the <=15% target and is not stable enough for default use.
The four Metal failures are deterministic compiler failures, not test assertion failures:

- `TextureSamplingAndMixedResourceOrderMatchEasyGpu`: MSL rejects `.sample` on the generated
  `texture2d<..., access::read>`; LC aborts at `metal_compiler.cpp:402`.
- `TextureSampleGradExecutesThroughLuisaXir`: the same sampled-texture access mismatch.
- `ShaderLibraryTextureAndSamplerCallablesMatchEasyGpu`: the same sampled-texture mismatch.
- `VectorConstructionAndSwizzlesMatchEasyGpu`: MSL rejects non-const `vector_element_ref`
  on swizzle temporaries such as `(v3).yx`; LC aborts at `metal_compiler.cpp:402`.

Because these failures terminate the test host, the fallback contract is active: Metal is
disabled by default in `native/CMakeLists.txt`; macOS CI passes
`-DFEATHER_LUISA_ENABLE_METAL=ON` to retain compile coverage, while all runtime calls remain
on Vulkan unless explicitly opted in. Fixing the pinned LC Metal compiler is an M8+ item.
