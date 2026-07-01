# Typed IR Compute Support Matrix

This page describes what the typed FEIR -> EasyGPU compute path supports today. Use it when a kernel generates successfully but native lowering returns `Rejected` or `FE_ERROR_UNSUPPORTED`.

For the conceptual overview, read [FEIR Compiler Pipeline](feir.md). For byte-level records, read [FEIR Binary Format](ir-format.md).

## Dispatch Contract

Supported generated compute kernels should execute through:

```text
DispatchPath.TypedEasyGpu
```

New completed DSL features should not prove behavior through `CpuReferenceFallback`. Samples and tests commonly assert `TypedEasyGpu` for this reason.

## Feature Matrix

| Feature | Typed EasyGPU status | Notes |
| --- | --- | --- |
| 1D/2D/3D compute dispatch | Supported | Logical dispatch size is passed separately from backend group count. |
| Bounds checks | Supported | Hidden guard emitted unless `[Kernel(BoundsCheck = false)]`. |
| Buffers | Supported | Read-only, write-only, read-write views. |
| 2D textures | Supported | Load/store/sample/sample-level for supported formats. |
| 3D textures | Supported | Load/store baseline. |
| Samplers | Supported | Descriptor sampler binding. |
| Uniforms / push constants | Supported | Scalar, vector, matrix, supported GPU structs. |
| `float`, `int`, `uint` arithmetic | Supported | `uint` supported for buffers, locals, casts, push constants. |
| Vectors | Supported | `float2/3/4`, `int2/3/4`, `bool2/3/4`. |
| Matrices | Supported | Explicit Feather matrix types. |
| `[GpuStruct]` | Supported | Deterministic layout metadata required. |
| Fixed GPU arrays | Supported | Use `GpuArrayN<T>` wrappers. |
| `if/else` | Supported | Structured block records validated natively. |
| `for` | Supported | Canonical typed lowering. |
| `while` / `do` | Supported for compute | AD has narrower restrictions. |
| `break` / `continue` | Supported for compute | AD currently rejects these. |
| `[Callable]` helpers | Supported | Overloads bind by generated mangled identity. |
| Shader math / HLSL aliases | Supported subset | Calls resolve by Roslyn symbol identity. |
| Shared memory | Supported | Dedicated declaration records. |
| Barriers | Supported | Workgroup, memory, and full barrier markers. |
| Integer atomics | Supported | L-value based section 7 representation. |
| AD markers | Supported in AD kernels | See AD contract below. |

## Texture And Format Notes

The typed texture bridge maps Feather 2D texture formats to EasyGPU runtime formats. Proven baseline formats include:

- `R8`
- `Rg8`
- `Rgba8`
- `R32Float`
- `Rgba32Float`

Additional formats may have host-side storage and descriptor support but still be rejected by a backend route. For example, `Bgra8` has host byte storage but no completed EasyGPU runtime/backend format in the current bridge.

Mip generation is supported for 2D color textures. Depth and 3D mipmap generation are rejected.

## Type Mapping Contract

| Feather type | GPU notes |
| --- | --- |
| `float`, `int`, `uint` | 32-bit scalar types. |
| `float2`, `int2` | 8-byte vector storage. |
| `float3`, `float4`, `int3`, `int4` | 16-byte buffer array stride. |
| `float2x2`, `float3x3`, `float4x4` | Explicit Feather matrix layout. |
| `[GpuStruct]` | Generated layout metadata required. |
| `GpuArrayN<T>` | Fixed array field inside GPU structs. |

Byte- and ushort-sized unsigned records can appear as layout metadata for host pixel/struct fields, but they are not a standalone shader arithmetic surface.

## AD Coverage Contract

AD uses the same typed FEIR foundation but has a narrower accepted source shape:

| AD feature | Status |
| --- | --- |
| Generated 1D kernels | Supported |
| `AD.Parameter(float/float2/float3/float4)` | Supported with buffer-backed values |
| `AD.Loss(float)` | Supported |
| Structured `if/else` | Supported |
| Canonical counted `for` | Supported |
| `while`, `do-while`, `break`, `continue` | Rejected in current AD bridge |
| Missing parameter/loss | Rejected |
| Forward-only fallback as AD success | Not allowed |

Run the AD gate when changing AD lowering:

```bash
python3 scripts/ad-industrial-gate.py
```

## Common Rejections

| Symptom | Check |
| --- | --- |
| Native unsupported call | Is the call a known `ShaderMath`, `Hlsl`, AD marker, texture sample, or `[Callable]`? |
| Invalid l-value | Is the assignment target a supported buffer/texture/shared-memory l-value? |
| Unsupported format | Is the texture format in the supported backend set? |
| AD rejected control flow | Did the AD kernel use `while`, `do`, `break`, or `continue`? |
| Fallback instead of `TypedEasyGpu` | Does the generated payload include section 7 typed IR? |

## Related Docs

- [C# Shader Subset](csharp-subset.md)
- [FEIR Compiler Pipeline](feir.md)
- [Native ABI](native-abi.md)
- [Diagnostics](diagnostics.md)
