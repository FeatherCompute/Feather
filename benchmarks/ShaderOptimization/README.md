# Shader Optimization Benchmark

This benchmark compares Feather-generated compute GLSL with equivalent handwritten GLSL across every EasyGPU Vulkan SPIR-V optimization level.

## Run

From the repository root:

```bash
python3 scripts/optimization-benchmark.py
```

For a fast functional run:

```bash
python3 scripts/optimization-benchmark.py --quick
```

Reports and emitted shader sources are written under `artifacts/optimization-benchmark/`.

## Measurements

- FEIR-to-GLSL source lowering median from repeated `ShaderInspection.GetGLSL<TKernel>()` calls.
- Cold Vulkan inspection with a forced persistent-cache miss.
- Warm Vulkan inspection from a validated SPIR-V disk-cache hit.
- Optimized GLSL line and byte counts after SPIRV-Cross.
- Steady-state dispatch median, p95, mean, min, and max.

The runner uses Vulkan timestamp queries when the device exposes them. Otherwise it batches several dispatches behind one `Finish()` and reports synchronized host time per dispatch. The report records which timing path was used.

All source and optimization variants execute identical input sizes and resource layouts. Results are validated against the handwritten Ultra output before a report is produced.

## Workloads

- `fused-mlp`: arithmetic-heavy elementwise NN work with 64 affine, activation, polynomial, and residual steps over 1,048,576 elements.
- `particle-sim`: branch-heavy physical simulation with 128 fixed integration steps over 524,288 particles.

The benchmark intentionally reports cold optimization cost alongside cache and runtime results. A smaller shader is not presented as a GPU speedup unless the measured dispatch time also improves.
