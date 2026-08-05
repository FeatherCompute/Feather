# Compute Rasterizer Performance

## Method

`samples/GraphicsRasterBenchmark` renders the same single triangle through the
EasyGPU graphics route and the Luisa Metal compute route. Each process creates a
512x512 RGBA32Float target, performs one warmup draw, measures five synchronous
draws, validates exactly 105,800 visible pixels, and reports the median. The
comparison below uses five serial fresh processes for each route on Apple M5.
No resolution, shader, pipeline state, or validation threshold differs between
the routes.

## Evolution

| Revision | Change | EasyGPU median | Compute median | Ratio |
| --- | --- | ---: | ---: | ---: |
| Rejected M3.5 baseline | Cached stage shaders, but three synchronizations and two full intermediate host round trips | 0.394 ms | 8.368 ms | 21.2x |
| R1 local baseline | Reproduced the rejected architecture before optimization | 0.583 ms | 8.151 ms | 14.0x |
| Resident-stream iteration | Resident vertex/varying/coverage/color resources, GPU index assembly, deferred target readback, one ordered stream synchronization | 0.381 ms | 0.492 ms | 1.29x |
| Final rebuild verification | Same optimized code after profiling instrumentation and full rebuild | 0.336 ms | 0.495 ms | 1.47x |

The final five compute process medians were `0.495, 1.954, 0.489, 0.530,
0.489 ms`; their median is `0.495 ms`. The matching EasyGPU medians were
`0.249, 0.464, 0.461, 0.300, 0.336 ms`; their median is `0.336 ms`. One compute
process experienced system scheduling noise, but the prescribed median remains
below both acceptance limits: `0.495 ms <= 0.590 ms` and `1.47x <= 1.5x`.

## Implemented Optimizations

- Generic Luisa dispatch batches uploads, execution, and requested downloads
  behind one synchronization instead of synchronizing every resource copy.
- Stable resource keys retain transformed vertices, raster varyings, coverage,
  and color targets on the device. Color data is downloaded only when the
  texture API requests host-visible bytes.
- Raster assembly reads the draw's index list on the GPU rather than expanding
  the complete transformed vertex stream on the CPU.
- Vertex, raster, and fragment shaders and their bound resources are cached for
  compatible layouts. The no-depth path submits all three stages to one ordered
  Luisa stream and synchronizes at the final target readback.
- The 16x16 two-dimensional launch keeps adjacent pixels together for coherent
  coverage and texture access. Pixel ownership retains deterministic depth,
  stencil, blend, MRT, fill/line/point, and shared-edge behavior without an
  atomic payload protocol.

The reference drill's compact triangle bins, SoA varyings, shared-memory tile
cache, and hierarchical Z remain justified for complex scenes, but the measured
single-triangle gap was dominated by synchronization and transfers. Adding
those structures to this workload would increase instruction and setup cost.

## Stage Profile

`FEATHER_RASTER_PROFILE_STAGES=1 FEATHER_GRAPHICS_TRACE=1` forces a synchronization
at each stage boundary. Across the five measured steady-state draws, median
host-wall upper bounds were:

| Stage | Median | Share of measured stage total |
| --- | ---: | ---: |
| Setup | 0.022 ms | 1.3% |
| Vertex | 0.499 ms | 28.9% |
| Assembly | 0.002 ms | 0.1% |
| Raster | 0.691 ms | 40.0% |
| Fragment | 0.512 ms | 29.7% |

The pinned Luisa runtime exposes synchronization and timeline events but no
public portable Metal/Vulkan GPU timestamp-query API. These values therefore
include command submission and synchronization overhead and must not be labeled
hardware-only GPU timestamps. Metal's private backend can read command-buffer
GPU start/end times, but consuming that internal API would couple Feather to an
unstable backend implementation and violate the pristine-submodule boundary.

## Correctness Gate

The optimized path retains all 21 compute-raster GPU parity tests, covering
depth/stencil across draws, blending and write masks, MRT, texture sampling,
viewport/scissor/culling, fill/line/point modes, indexed and instanced assembly,
interpolation, and shared-edge ownership. The separate Graphics suite remains
green; its three window tests are intentionally opt-in.
