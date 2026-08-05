# Compute Rasterizer Performance Drill

## Reference and Scope

This drill reviews Aman Sachan's CUDA Rasterizer at commit
`870d4cef64a2179eb44479f9441d660405dd03ca`. The reference is useful as a
concrete comparison between primitive-parallel scanline and tiled rasterization,
but it is an educational CUDA implementation rather than a blueprint to copy
unchanged. In particular, its fixed-capacity bins, one launch per tile, and
spin-lock depth protocol are deliberately excluded from Feather.

Feather's pre-optimization compute path dispatches vertex FEIR, copies transformed
vertices to the host, assembles triangles on the CPU, dispatches one pixel thread
that scans every triangle, copies full-frame varyings and coverage to the host,
then reuploads them for a separate fragment FEIR dispatch. The 512x512 benchmark
therefore measures synchronization and transfer costs in addition to raster work.

Status labels below mean:

- **Applicable:** expressible with the current FEIR/XIR/Luisa compute surface.
- **Requires extension:** architecturally useful, but needs a new persistent
  resource, kernel composition, lowering feature, or profiling hook.
- **Not applicable:** an implementation detail that is unsafe or counterproductive
  for Feather's portable Metal/Vulkan path.

## Extracted Techniques

| Area | CUDA reference evidence | Assessment for Feather |
| --- | --- | --- |
| Tile/block rasterization | The output is divided into a fixed 32x32 tile grid (`src/rasterize.cu:23-26`). Primitive bounds are binned before a per-tile pixel kernel (`src/rasterize.cu:1168-1217`, `1238-1290`). The README reports nearly 4x improvement for spatially distributed triangles and warns that concentrated geometry removes the benefit (`README.md:30-41`). | **Applicable.** Use a fixed pixel tile size, initially 8x8 or 16x16, and dispatch all tiles in one 2D grid. Each pixel examines only its compact tile list. Do not launch a separate kernel from the CPU for every tile. |
| Edge-function math | Coverage evaluates three signed 2D cross products (`src/rasterizeTools.h:87-108`), while the reference barycentric path recomputes signed areas/divisions per candidate (`src/rasterizeTools.h:44-68`, `src/rasterize.cu:1150-1155`). | **Applicable.** Precompute edge coefficients and reciprocal area once per primitive, evaluate edges at pixel centers with fused multiply-add style expressions, and derive barycentrics from those edge values. Integer fixed-point edges would improve exactness but **require extension** to validate overflow/top-left behavior across Metal and Vulkan; float edges remain the first portable implementation. Incremental edge stepping is useful only if one lane owns several adjacent pixels, so it is deferred until profiling justifies that mapping. |
| Parallel work assignment | Scanline launches one thread per primitive, then serially walks its bounding box (`src/rasterize.cu:1093-1118`). The tile path launches pixel threads with a 2D block and each thread walks that tile's triangle list (`src/rasterize.cu:1120-1166`). | **Applicable.** Keep deterministic pixel ownership and use a fixed 8x8/16x16 threadgroup per tile. This avoids depth/payload races and maps to Metal's SIMD/threadgroup model. Dynamic per-tile work queues are **not initially applicable**: FEIR has atomics, but no portable subgroup scheduler, and queue overhead is excessive for small bins. |
| Depth, early-Z, and hierarchy | The reference converts depth to `int`, spins on a per-pixel `atomicCAS` mutex, and updates depth plus payload in a critical section (`src/rasterize.cu:933-960`). Its README notes the serialization cost (`README.md:80-86`). | Pixel ownership makes the mutex **not applicable**. **Applicable:** maintain the best depth/stencil candidate in registers while traversing a tile, reject candidates before interpolation/fragment work, and commit once. Hierarchical-Z **requires extension** because persistent per-tile depth summaries and conservative invalidation across load/store, blending, and multiple draws are needed. |
| Pixel coverage loop | Primitive-parallel scanline restricts work to each triangle AABB but visits every enclosed pixel (`src/rasterize.cu:1015-1048`). Tile pixels instead loop only over triangles binned to that tile (`src/rasterize.cu:1131-1164`). | **Applicable.** Replace the current full `pixel x primitive` loop with `pixel x tilePrimitive`, reject tiles using primitive bounds, and place cull/scissor/bounds checks before edge/interpolation work. Preserve top-left ownership, fill/line/point rules, depth clamp, and stencil semantics. |
| Memory layout and locality | The reference stores assembled vertices inside array-of-struct `Primitive` objects and fragment payloads in `Fragment` objects; its README identifies vertex/assembly global-memory indirection and reassignment as expensive (`README.md:43-55`). It does not use shared-memory tile buffers. | **Applicable:** compact primitive records and coalesced linear pixel/depth/color buffers; keep stage data device-resident and capacity-grown. **Requires extension:** a fully SoA varying representation and shared-memory tile cache, because the current dynamic varying layout and Luisa-generated kernel signatures need specialization. Texture accesses naturally gain locality once adjacent tile lanes shade together. |
| Scanline versus per-pixel | The README models scanline as `pixels x primitives` and observes about half the tiled throughput (`README.md:43-58`). The implementation assigns one primitive per CUDA thread (`src/rasterize.cu:1093-1118`). | Primitive-owned scanline is **not applicable** to the full Feather state matrix: overlapping primitives require atomic arbitration of depth, stencil, blend, and MRT payloads. Pixel-owned tiled traversal is **applicable** and deterministic. Scanline remains useful only for line-mode edge traversal after separate profiling. |
| Triangle binning | One thread per primitive computes its AABB and inserts its index into overlapping tiles under a tile mutex (`src/rasterize.cu:1168-1217`). Each `Tile` has a fixed `triIndices[1000]` (`src/rasterize.cu:84-87`). | Binning is **applicable**, but the fixed list and mutex insertion are **not applicable**. Feather should use checked compact storage: count overlaps, prefix-sum offsets, then fill indices. A first bounded CPU-built bin is acceptable only as an instrumentation step; the performance path must build and consume bins on-device without a synchronization round trip. |
| Load balancing | Tiling reduces work when primitives are distributed, but concentrated geometry leaves a few heavy tiles (`README.md:38-41`). The reference uses the same pixel launch shape regardless of occupancy and a hard bin capacity. | **Applicable:** compact empty tiles, keep small fixed threadgroups, and size bins from actual overlap counts. **Requires extension:** splitting pathological high-occupancy tiles or large triangles into multiple work items while retaining deterministic per-pixel ordering. Measure bin histograms before adding this complexity. |

## Feather Optimization Order

The reference's strongest transferable result is reducing candidate triangles per
pixel. It cannot by itself close Feather's measured 21.2x gap, because Feather
also performs two full intermediate readback/reupload cycles and three synchronous
dispatches. Optimization proceeds in this order:

1. Add per-stage timing and preserve the current output as a correctness baseline.
2. Eliminate allocation churn and host round trips by retaining raster resources
   on the device; combine raster coverage/interpolation and fragment invocation
   where the XIR callable interface permits it.
3. Precompute compact primitive edge/bounds data and reduce coverage arithmetic.
4. Add compact tile binning and fixed threadgroup pixel ownership.
5. Add register early-Z/stencil rejection and shade only the winning candidate.
6. Profile memory traffic and tile occupancy before considering SoA/shared-memory
   specialization, hierarchical-Z, or dynamic load balancing.

Every step must retain depth/stencil, blending, MRT, sampling, fill/line/point,
viewport/scissor, culling, and indexed/instanced behavior. The benchmark remains
five serial fresh-process runs at 512x512 with 105,800 visible pixels, compared
against an identically collected EasyGPU median.
