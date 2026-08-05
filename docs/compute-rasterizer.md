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
  `FirstVertex`, `FirstIndex`, and `VertexOffset`;
- position-first all-float structured varyings, perspective-correct
  interpolation, viewport, scissor, front-face selection, and culling;
- optional D32Float compare/write with clear/load behavior;
- one RGBA8, BGRA8, or RGBA32Float color target and clear/load behavior;
- generated fragment push constants and sampled `Texture2D`/sampler bindings.

Dedicated Vulkan/MoltenVK GPU tests prove coverage, interpolation, depth across
draws, viewport/scissor/cull, actual vertex and fragment FEIR execution,
sampled-texture execution, structured varyings, multiple triangles, and indexed
assembly. The implementation still stages intermediate buffers through host
memory and synchronizes each stage. This is a correctness architecture slice,
not the device-resident tiled design described above.

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
shader and workload. Results on Apple M5 via Vulkan SDK 1.4.350/MoltenVK:

| Route | Dispatch path | Runs (ms) | Median (ms) | Relative |
| --- | --- | --- | ---: | ---: |
| EasyGPU | `TypedEasyGpu` | 0.304, 0.436, 0.219, 0.214, 0.256 | 0.256 | 1.0x |
| Compute raster | `Luisa` | 23.303, 15.946, 16.959, 14.764, 14.347 | 15.946 | 62.3x |

These are end-to-end synchronous draw times, not GPU timestamp queries. The
compute number includes three dispatches, repeated resource staging, host
readback of transformed/interpolated values, and synchronization between
stages. It establishes the current product cost but does not attribute the
slowdown to raster ALU. Device-resident scratch, cached kernels, fused
raster/fragment execution, and tile binning are required before a useful GPU
throughput comparison.

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
| Triangle-list RGBA offscreen | Supported; indexed/non-indexed | Optimize/tile |
| Generated vertex/fragment FEIR | Supported for float varyings and float4 color | Add flat integer/bool fields and fragment builtins |
| Barycentric interpolation | Perspective-correct float lanes | Add clipping-generated vertices and top-left tie rule |
| Depth/cull/viewport/scissor | Supported for D32Float | Add depth clamp and full homogeneous clipping |
| SampledTexture2D | `Sample` proven | Prove level/grad/mips and implement implicit derivatives |
| Blend/MRT/stencil | Unsupported | Required |
| MSAA, line/point/polygon modes | Unsupported | Required if API remains unchanged |
| Instancing | Unsupported | Required |
| Vulkan | Proven locally on MoltenVK | Prove native Vulkan CI/GPU host |
| Metal | Not built in the tested configuration | Build and run the same GPU suite |
| Window presentation | EasyGPU transition only | Backend-neutral presenter required |

EasyGPU raster cannot be retired at this point. The blocking sequence is:
device-resident stage handoff and kernel caching; clipping/flat interpolation
and fragment builtins; blend/MRT/stencil/MSAA parity; neutral presentation;
then cross-backend correctness and performance qualification.

## Verification

Local verification used `native/build-compute/libfeather.dylib` and its Vulkan
runtime directory. The native `feather` target and benchmark sample build
cleanly. With the compute route explicitly enabled, all 9 compute-raster GPU
tests pass. With the default EasyGPU route, the full solution reports 401
integration tests passed and the 9 opt-in compute GPU tests skipped; Graphics
reports 7 passed and 3 opt-in window tests skipped. Existing generated graphics
coverage separately reports 44/44 passed. Raw benchmark commands and output are
stored under `artifacts/compute-rasterizer/`.
