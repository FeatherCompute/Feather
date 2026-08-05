# Luisa Vulkan Performance

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

The current Luisa implementation constructs a `Context`, Vulkan `Device`, and compute
`Stream`, creates and uploads device buffers, translates and compiles the shader, dispatches,
downloads writable buffers, and destroys everything inside every `Dispatch` call.

A 1024 x 1024 construction experiment at one sample per pixel measured 33 ms on EasyGPU
and 4.956 s on Luisa. Luisa's timestamps placed context/plugin/device startup at about 4 ms
and its SPIR-V optimization/compilation completion 4.758 s after device creation. The
remaining roughly 0.19 s contains FEIR-to-XIR/AST translation, allocation, about 20 MiB of
buffer upload/download staging, synchronization, and the very small one-sample GPU workload;
these components are not separately instrumented in the baseline.

Comparing the 10,240-sample median with the one-sample construction run shows that fixed
startup cannot explain the 26.372 s median gap: after subtracting the 4.923 s one-sample
Luisa/EasyGPU delta, about 21.45 s remains attributable primarily to generated shader quality
and GPU execution. Later sections will replace this structural estimate with measurements
after code-generation, runtime residency, and staging experiments.

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
