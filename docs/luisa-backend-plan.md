# LuisaCompute Vulkan XIR Backend Plan

## M1 Integration

Feather vendors LuisaCompute as the `LuisaCompute/` git submodule, pinned by
the Feather gitlink rather than a floating branch. M1 uses
`0c84a1aa7ad4181373040fc018f20d735f73c626` from LuisaCompute's `stable`
branch (LuisaCompute 0.9.0, Apache-2.0). Recursive checkout is required:
LuisaCompute has nested source dependencies under `src/ext/`.

`native/CMakeLists.txt` adds LuisaCompute after all Feather targets are
created. LuisaCompute sets global CMake output directories, so this ordering
preserves Feather's established `native/build/libfeather.*` layout used by the
native asset staging scripts. The `feather` target depends on upstream target
`luisa-compute-backend-vk`, making the normal native build compile Luisa's
Vulkan XIR-to-SPIR-V path without changing Feather's public ABI or linking the
Luisa runtime into `libfeather` before M2 owns its loading and packaging.

The M1 configuration enables only `LUISA_COMPUTE_ENABLE_VULKAN` and
`LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV`. It disables DSL, tensor, DX, Metal, HIP,
CUDA, fallback, remote, GUI, tests, OIDN, Rust, and the alternative LLVM
SPIR-V code generator. Rust is not a dependency of the upstream Vulkan XIR
target, so M1 intentionally has no Cargo/toolchain setup.

## Backend Matrix

| Luisa backend | M1 policy | linux-x64 | osx-arm64 | win-x64 | Notes |
| --- | --- | --- | --- | --- | --- |
| Vulkan | Enabled | Yes | Yes | Yes | The only M1 backend; macOS uses MoltenVK. |
| CPU | Prohibited | No | No | No | Product decision: never build, link, run, or use it as a fallback. |
| DirectX 12 | Disabled | No | No | Future suite | Future optional NuGet compilation suite on Windows. |
| CUDA | Disabled | Future suite | No | Future suite | Future optional NuGet compilation suite for NVIDIA toolchains. |
| Metal | Disabled | No | No | No | Not selected; Vulkan/MoltenVK is the macOS route. |
| HIP, fallback, remote | Disabled | No | No | No | Outside M1. |

Luisa's Vulkan backend compiles its XIR-to-SPIR-V generator and uses bundled
`volk`. Linux CI already installs Vulkan headers/loader, glslang, and SPIR-V
tools; Windows CI provisions a Vulkan SDK. macOS CI installs `vulkan-headers`,
`vulkan-loader`, and `molten-vk`, then verifies the headers, loader, and
`libMoltenVK.dylib` before the native build.

## CI And Risks

The first recursive checkout downloads Luisa's nested submodules, including
large graphics dependencies. The Linux x64 Vulkan backend may download the
Luisa-provided DXC runtime when no `LUISA_COMPUTE_VK_SDK_DIR` is supplied.
Future CI should cache submodules and Luisa's DXC download, while retaining
the fixed gitlink and checksums supplied upstream.

LuisaCompute 0.9.0's Vulkan SPIR-V argument-analysis source dereferences
`luisa::compute::Type` without including its defining `luisa/ast/type.h`.
Feather adds that header as a target-local forced include for
`luisa-compute-spirv` (using `/FI` on MSVC and `-include` elsewhere). Remove
this compatibility workaround when a pinned upstream revision includes the
header directly; the Luisa submodule itself remains unmodified.

M1 also uses Luisa's supported `LUISA_COMPUTE_USE_SYSTEM_STL` option. This
avoids bundled EASTL alias-template CTAD that macOS-14's Apple Clang cannot
compile in Luisa's Vulkan dependencies. It is a project-wide upstream build
mode, not a backend fallback; the enabled execution backend remains Vulkan.

Xcode 15.4 additionally cannot construct Luisa's aggregate-only `SharedVar`
and local `LoopSite` types through `emplace`. CMake writes narrowly patched
copies of those two sources to the build directory and compiles the copies;
the pinned submodule is never modified. Configuration fails if the expected
upstream text changes, forcing an explicit compatibility review on upgrades.

M1 deliberately does not package Luisa dylibs or expose a Luisa API through
the Feather C ABI. Loading those artifacts before a backend-selection contract
exists would risk changing the current EasyGPU runtime behavior. M2 must add
runtime packaging, rpath/install-name handling, and third-party notice review.

## M2 Plan

M2 adds a dedicated native Luisa Vulkan layer beside the current FEIR/EasyGPU
path. It will translate Feather FEIR modules into Luisa XIR using the upstream
XIR builder, preserving typed scalar/vector/aggregate semantics, control flow,
resource arguments, memory operations, atomics, and kernel entry metadata.
Luisa's XIR verifier and Vulkan SPIR-V pass pipeline then produce the module
executed by a Luisa Vulkan device and stream.

That layer will map Feather buffers, textures, samplers, dispatch dimensions,
and synchronization to Vulkan-backed Luisa resources. Backend selection must
be explicit and keep EasyGPU as the existing default until the Luisa path has
equivalent coverage. Once ABI and resource lifetimes are defined, managed
bindings, cache keys, staged Luisa runtime assets, and Vulkan device-backed
coverage can be introduced without weakening existing test assertions.
