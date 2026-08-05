# Compute Rasterizer Design

## Goal and Constraints

Feather.Graphics will keep its managed surface (`GpuGraphicsPipeline`, generated
vertex/fragment structs, resource binding, draw descriptors, render textures,
and window presenter) while moving raster execution to compute kernels. The
implementation must run through the existing FEIR -> XIR -> Luisa compute path
on Vulkan and Metal. EasyGPU graphics remains available during migration and is
removed only after the compute capability matrix is complete.

No CPU raster backend is part of this design. The small CPU triangle helper in
`feather_c_api.cpp` is legacy fallback code and is not a correctness oracle or
an execution route for the new pipeline.

## Existing Surface and Evidence

`GpuGraphicsPipeline.Create` passes separate generated vertex and fragment FEIR
modules to `fe_graphics_pipeline_create_from_ir`. Generated bind methods already
map shader constructor values to buffers, sampled textures, samplers, and packed
push constants. Draw supports indexed/non-indexed geometry, one to eight color
targets, optional depth/stencil, per-draw load/clear state, blending, culling,
polygon mode, and synchronous completion.

The current native path parses typed FEIR, discovers varying/resource/push-
constant layouts, lowers both stages to GLSL, and creates an EasyGPU fixed-
function graphics pipeline. Existing integration coverage exercises generated
vertex/fragment control flow, structured varyings, perspective interpolation,
indexed draws, depth/stencil, culling, blending/write masks, MRT, texture
sampling (level/gradients/mips/filter/address), and load/clear behavior. Window
samples render offscreen and then use `GpuTexturePresenter`; that presenter is
currently implemented by EasyGPU window infrastructure.

The reference CUDA rasterizer uses separate vertex transform, primitive
assembly, raster/depth, fragment shading, and framebuffer stages. Its scanline
path uses one thread per primitive and an atomic depth/mutex; its tile path bins
primitive bounding boxes before per-pixel barycentric tests. Feather adopts the
stage split and tile concept, but not the fixed-capacity triangle lists or
spin-lock depth protocol.

## Pipeline Architecture

### Compilation

Pipeline creation parses the two typed FEIR modules once and builds a shared
`ComputeRasterProgram`:

1. Lower the vertex entry to an XIR callable with explicit vertex/instance IDs,
   declared resources, and push constants.
2. Lower the fragment entry to an XIR callable with an explicit interpolated
   varying value, fragment coordinate, declared resources, and push constants.
3. Flatten the generated varying type into aligned scalar/vector slots. Record
   the `[Position]` slot and preserve integer/bool fields as flat values;
   floating-point fields are perspective-correct interpolants.
4. Generate and cache compute kernels by FEIR hashes, resource formats, target
   count, depth format, and pipeline state. Pipeline creation remains lazy so a
   target-dependent variant is compiled at first draw.

This extends the native XIR lowerer instead of round-tripping through GLSL. The
existing managed generator and C ABI remain unchanged. A new dispatch-path enum
value identifies compute rasterization; it must not masquerade as EasyGPU.

### GPU Data

Per pipeline/draw storage is device-resident and capacity-grown:

- `RasterVertex[]`: clip position, inverse W, and packed varying slots.
- `RasterPrimitive[]`: three vertex indices, screen-space bounds, signed area,
  reciprocal area, facing, and validity/clip flags.
- `TileCounts[]`, prefix offsets, and `TilePrimitiveIndices[]`: compact lists
  for 8x8 or 16x16 pixel tiles. No fixed per-tile triangle limit.
- Optional internal `uint` ownership/depth keys for deterministic arbitration.
- Existing color/depth textures remain the public render targets. Internal
  multisample color/depth storage is allocated only for MSAA variants.

The vertical slice intentionally uses a simpler deterministic pixel kernel that
loops over assembled primitives. It is O(pixels * triangles), but establishes
coverage, interpolation, depth, and FEIR invocation without atomics. Tile
binning replaces that loop after correctness is locked down.

### Dispatch Sequence

1. **Load/clear:** honor color and depth load operations with a 2D clear/copy
   kernel. `DontCare` permits initialization to be skipped.
2. **Vertex:** one invocation per logical vertex executes the vertex FEIR entry
   and writes clip position/varyings.
3. **Assembly/clip/bin:** one invocation per primitive fetches indices, assembles
   topology, rejects invalid W and fully clipped primitives, clips triangles
   against the homogeneous view volume, performs perspective divide, viewport
   transform, facing/cull selection, bounding-box generation, and tile binning.
4. **Raster/fragment:** one invocation per sample/pixel traverses only its tile's
   primitives, evaluates top-left edge rules at sample locations, computes
   barycentrics, performs perspective-correct interpolation, selects the winning
   depth/stencil candidate, invokes the fragment FEIR callable, applies blending
   and write masks, then stores color/depth/stencil.
5. **Resolve:** multisample variants resolve into the public color targets.

Pixel ownership avoids cross-primitive races: each raster invocation owns one
pixel/sample and evaluates all candidates serially. The first tiled version
keeps this model; atomics are needed only while constructing compact tile lists.
Primitive order breaks equal-depth ties, preserving deterministic results.

### Raster Rules

- Viewport is the full target until an explicit viewport/scissor is added to
  `GraphicsDrawDesc`; pixel centers use `(x + 0.5, y + 0.5)`.
- Homogeneous clipping uses `-w <= x,y <= w` and Vulkan/Metal-compatible depth
  convention selected by the target backend. Generated clip vertices interpolate
  all floating varyings before perspective divide.
- Triangle coverage uses signed edge functions and the top-left rule. Front face
  is evaluated after the Y-flipped framebuffer viewport transform. Back/front
  culling follows `RasterState`.
- Perspective interpolation uses `(sum(lambda_i * value_i / w_i)) /
  sum(lambda_i / w_i)`. Integer and bool fields are flat from the provoking
  vertex. `FragmentIds.Coord` receives pixel-center X/Y, final depth, and `1/w`.
- Depth compare/write, stencil operations, blend factors/ops, MRT write masks,
  and load/clear semantics are implemented as compute code generated from the
  immutable pipeline state.
- SampledTexture2D bindings are forwarded to the fragment callable. Explicit
  level and gradient sampling are direct; implicit LOD derives finite
  differences from neighboring interpolants in the owned 2x2 pixel quad.

## API Mapping

The public generic pipeline, shader interfaces, descriptors, draw overloads,
resource binding, and render texture types do not change. Native pipeline state
gains a compute-program cache and scratch allocations. `Draw` and `DrawIndexed`
select the compute route explicitly during migration; EasyGPU remains the
fallback only for capability gaps and is reported honestly through
`LastDispatchPath`.

The default transition cannot silently change `LastDispatchPath` from
`TypedEasyGpu` while tests still assert it. The migration therefore introduces
an opt-in compute route and dedicated GPU tests first. Flipping the default and
updating the dispatch-path contract is a separately reviewable compatibility
change once feature parity is reached; existing pixel behavior tests themselves
remain unchanged.

## Delivery Stages

1. **Vertical slice:** add the compute dispatch route, vertex/fragment callable
   lowering for `float4`, triangle-list assembly, deterministic pixel raster,
   perspective barycentrics, and RGBA target writes. Add GPU tests for coverage,
   interpolated color, and outside pixels.
2. **Core parity:** structured varyings, indexed topology, depth, culling,
   viewport/clipping, sampled textures/samplers, push constants, load/clear,
   MRT, blending, and stencil. Run all existing Graphics tests unchanged on the
   legacy default and the same pixel cases through compute.
3. **Performance:** compact tile binning, persistent scratch buffers, pipeline
   caching, MSAA, line/point modes, timestamp benchmarks, and memory bounds.
4. **Presentation and cutover:** present compute-produced textures through the
   current readback presenter as a transition; replace the presenter with a
   backend-neutral shared texture/swapchain path before deleting EasyGPU.

## Implemented Slice (M3.5)

The implementation is available behind `FEATHER_GRAPHICS_COMPUTE=1`; the
default remains EasyGPU. It executes generated vertex FEIR as a one-dimensional
Luisa compute dispatch, rasterizes a triangle list with a pixel-owned Luisa DSL
kernel, and executes generated fragment FEIR as a two-dimensional compute
dispatch. Stage resources and packed push constants use the same generated
bindings as the public graphics API.

The raster stage currently provides:

- non-indexed and `ushort`/`uint` indexed triangle lists, including
  `FirstVertex`, `FirstIndex`, `VertexOffset`, instancing, and first-instance
  vertex builtins;
- position-first all-float structured varyings, perspective-correct
  interpolation, top-left shared-edge ownership, viewport, scissor, front-face
  selection, culling, depth clipping/clamp, and fill/line/point modes;
- D32Float compare/write and packed D24S8 stencil compare/operations/masks with
  clear/load behavior;
- one to eight dense `[Color(n)] float4` outputs, per-attachment blending and
  write masks, and RGBA8/RGBA32Float targets;
- generated vertex/fragment push constants using their shared aligned layout;
- sampled `Texture2D` with nearest/linear filtering, repeat/clamp addressing,
  explicit level/gradient sampling, mip chains, and finite-difference Ddx/Ddy.

Twenty-one dedicated GPU tests pass on both native Metal and Vulkan/MoltenVK.
They prove coverage/interpolation, depth and stencil across draws, viewport,
scissor, cull and polygon modes, generated stage FEIR, sampling/derivatives,
aligned cross-stage constants, blend/write masks, MRT, instancing, shared-edge
ownership, and indexed assembly. The optimized path retains transformed
vertices, raster scratch, and color targets on the device and submits the
no-depth stages to one ordered stream. See [Compute Rasterizer Performance](rasterizer-perf.md)
for the fresh-process benchmark and stage profile.

## Presentation Status

`WindowGraphicsTriangle`, `WindowGraphicsTexturedQuad`, and
`GpuTexturePresenter` still present through EasyGPU window infrastructure. A
compute-rendered Feather texture can be synchronized back into the shared host
texture state and then presented by that existing path, but this incurs
readback/re-upload and does not remove the EasyGPU dependency. No window code
was redirected in this milestone. A backend-neutral Luisa swapchain/shared-image
presenter remains a hard requirement before EasyGPU deletion.

## Benchmark

`samples/GraphicsRasterBenchmark` is the reproducible microbenchmark. It draws
one triangle to a 512x512 RGBA32Float target, performs one warmup and five
synchronous measured draws in a fresh process, validates 105,800 visible
pixels, and reports the median host wall time. Both routes use the same managed
shader and workload. The following runs were collected serially in fresh
processes on Apple M5; EasyGPU used its default route and compute raster used
the native Metal Luisa backend:

| Route | Dispatch path | Runs (ms) | Median (ms) | Relative |
| --- | --- | --- | ---: | ---: |
| EasyGPU | `TypedEasyGpu` | 2.892, 0.359, 0.345, 0.394, 0.732 | 0.394 | 1.0x |
| Compute raster (Metal) | `Luisa` | 8.965, 8.190, 9.330, 7.763, 8.368 | 8.368 | 21.2x |

These rejected-round numbers are retained as the optimization baseline. The R1
implementation removes intermediate host round trips and reaches the acceptance
target; current raw runs and the complete evolution table live in
[Compute Rasterizer Performance](rasterizer-perf.md).

## Risks and Controls

- **Stage lowering:** FEIR currently lowers compute entries, not graphics
  callables. Keep callable lowering isolated and test generated XIR validation
  before dispatch.
- **Cross-backend texture formats:** storage writes and depth formats differ.
  Normalize internal color/depth representations and resolve through explicit
  format conversion kernels.
- **Ordering and races:** triangle-parallel raster needs atomic depth plus
  payload ownership. Start pixel-owned, then optimize without changing results.
- **Clipping/interpolation:** incorrect W handling produces large bounds and NaNs.
  Validate finite clip coordinates, clip before divide, and test near-plane and
  shared-edge cases.
- **Unbounded binning:** use count/prefix/fill passes with checked capacities;
  never use a fixed list like the reference implementation.
- **Implicit texture LOD:** quad derivatives are subtle at primitive edges.
  Land explicit level/gradient sampling first and retain a documented temporary
  limitation until derivative conformance tests pass.
- **Presentation:** window creation/presentation still depends on EasyGPU. This
  is an explicit transition state and does not block offscreen compute raster.

## Current Capability Matrix

| Capability | M3.5 status | Cutover requirement |
| --- | --- | --- |
| Triangle-list RGBA offscreen | Supported; indexed/non-indexed and instanced | Optimize/tile |
| Generated vertex/fragment FEIR | Float varyings; float4/MRT color | Add flat integer/bool varyings and remaining fragment builtins |
| Barycentric interpolation | Perspective-correct, top-left rule | Full homogeneous clipping and edge helper lanes |
| Depth/cull/viewport/scissor | D32Float plus depth clamp/clip | Full homogeneous clipping |
| SampledTexture2D | Sample/level/grad/mips/filter/address proven | Different U/V address modes; exact edge derivatives |
| Blend/MRT/stencil | Supported; 8 MRT, D24S8 stencil | Broader conformance matrix |
| Polygon modes | Fill/line/point supported | Conformance for API rasterization rules |
| MSAA | Unsupported | Per-sample color/depth storage and resolve |
| Instancing | Supported | Performance qualification |
| Vulkan | 21/21 GPU tests on MoltenVK | Prove native Vulkan GPU host |
| Metal | 21/21 GPU tests on Apple M5 | Performance qualification |
| Window presentation | EasyGPU transition only | Backend-neutral presenter required |

EasyGPU raster cannot be retired at this point. The blocking sequence is:
device-resident stage handoff and tile binning; full clipping/flat interpolation;
MSAA; neutral presentation; then cross-backend conformance and performance
qualification. The default route remains EasyGPU because changing
`LastDispatchPath` is a public contract change covered by existing tests.

## Verification

Local verification used `native/build-raster/libfeather.dylib`. The native
`feather` target and benchmark sample build cleanly. With
`FEATHER_GRAPHICS_COMPUTE=1`, all 21 dedicated compute-raster tests pass on both
`FEATHER_LUISA_BACKEND=metal` and `vk`. Existing generated graphics pixel tests
also exercise the compute route, but their hard-coded `TypedEasyGpu` dispatch
assertions intentionally prevent treating that run as the default contract.
On the normal EasyGPU route, the unchanged `GeneratedGraphicsPipelineTests`
pass 44/44 and `Feather.Graphics.Tests` pass 7 with 3 opt-in window tests
skipped. The complete integration project reports 420 passed, 21 opt-in compute
tests skipped, and 4 failures: all four are `LuisaBackendMetalTests` that still
expect compiler failures fixed by the pinned LC 35a06cb0 Metal
sample/swizzle/vector-element changes. No existing test was modified to hide
that baseline mismatch. Raw audit and benchmark output is stored under ignored
`artifacts/compute-rasterizer/`.
