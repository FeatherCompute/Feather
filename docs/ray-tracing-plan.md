# Feather Ray Tracing Pipeline Plan

## Status

**Implemented and verified on Metal (MetalRT).** The full hardware ray
tracing pipeline is live in Feather:

- `GpuMesh` / `GpuAccel` / `ReadOnlyAccel` + `GPU.CreateMesh` /
  `GPU.CreateAccel` (managed API).
- `Ray` / `SurfaceHit` GPU structs; `Ray` supports both the four-argument
  constructor and the two-argument form `new Ray(origin, direction)` which
  defaults to the `[0, +inf)` range.
- `TraceClosest` lowers through FEIR -> XIR -> LC `accel_trace_closest`
  (MetalRT), with correct closest-hit semantics (verified against a
  two-triangle scene in both traversal directions).
- RayTracing sample: Cornell-box hardware path (`-- hw`, 1024x1024 in ~4 s)
  and the software path tracer (both PASS the image-variation proof).
- Integration tests: `RayTracingHardwareTests` (3 tests).

Two Feather bugs were found and fixed along the way: the accel vertex count
unit mismatch (floats vs float3) and the `float3` 16-byte alignment mismatch
when repacking packed vertex data. One LC bug was fixed upstream
(`xir2ast` conditional-branch merge detection inside loops with backedges,
commit `c50f71ca9` on `perf/linearize-xir-analyses`).

Remaining known limitation: kernels whose loop bodies contain conditional
`break`s cannot be restructured by the pinned LC `restructure_cfg` pass
(see `xir-reverse-loop-autodiff-plan.md`); the samples avoid `break` in
favor of bound termination.

## Goal

Expose LuisaCompute's hardware ray tracing through Feather's C# DSL so that
generated kernels can trace rays against GPU-accelerated structures:

```csharp
using var mesh = GPU.CreateMesh(vertices, indices);          // vertices: GpuBuffer<float3>, indices: GpuBuffer<uint>
using var accel = GPU.CreateAccel(mesh);                      // builds TLAS immediately
GPU.Dispatch(new TraceKernel(accel.AsReadOnly(), image.AsReadWrite(), ...), size);

[Kernel]
public readonly partial struct TraceKernel(ReadOnlyAccel accel, ReadWriteBuffer<float4> image, ...) : IKernel2D
{
    public void Execute()
    {
        var ray = new Ray(origin, direction, 0.0f, 1e30f);
        var hit = accel.TraceClosest(ray);                    // SurfaceHit { Inst, Prim, Bary, T }
        ...
    }
}
```

## Design

### 1. Managed API (`src/Feather/Resources/GpuAccel.cs`, `src/Feather/Core/GPU.cs`)

- `GpuMesh` — owns the native mesh built from a vertex buffer
  (`GpuBuffer<TVertex>` where `TVertex` is a `float3`-compatible struct) and an
  index buffer (`GpuBuffer<uint>`); `GPU.CreateMesh<TVertex>(GpuBuffer<TVertex>, GpuBuffer<uint>)`.
- `GpuAccel` — owns the native TLAS; `GPU.CreateAccel(params GpuMesh[])`;
  immediately builds on the context stream; `Dispose` releases.
- `ReadOnlyAccel` — kernel binding view (like `ReadOnlyBuffer<T>`).
- Kernel DSL structs: `Ray` (`Float3 Origin`, `Float3 Direction`, `float TMin`, `float TMax`)
  and `SurfaceHit` (`uint Inst`, `uint Prim`, `float2 Bary`, `float T`).

### 2. FEIR protocol extension

- **Type**: new resource type kind `accel` (mirror the buffer resource kind).
- **Instruction**: `TraceClosest(accel, ray) -> SurfaceHit` emitted by the
  generator when the kernel calls `ReadOnlyAccel.TraceClosest`.
- The generator (`Feather.Generators`) resolves the `TraceClosest` call into
  an FEIR expression node; the native parser (`feather_typed_ir.cpp`) accepts
  the new kind/opcode.

### 3. Native side

- **C API**: `fe_accel_create(context, mesh_count, mesh_handles, out handle)`
  creates a LuisaCompute `Accel` from meshes; accel handle registry mirrors the
  buffer registry (`feather_c_api.cpp`).
- **Lowering** (`feather_luisa_xir.cpp`): map the FEIR `accel` type to LC
  `Type::accel()`; bind the accel as an `AccelVar` function argument;
  translate `TraceClosest` to `accel.intersect(ray, {})`; lower `Ray`/`SurfaceHit`
  structs like ordinary FEIR structs.
- **Dispatch** (`feather_luisa_backend.cpp`): include accel bindings in the
  `bound_arguments` span (mirroring buffer bindings).

### 4. Ray tracing sample upgrade

`samples/RayTracing` gains a hardware path: Cornell-box walls as triangle
meshes, `TraceClosest` in the kernel, same image-writer proof. The software
fallback kernel stays available behind an option.

## Stage 2 entry points (investigated, ready to implement)

The FEIR protocol is shared between the generator and the native parser.
Stage 2 needs these exact touch points:

1. **Managed kernel type**: `ReadOnlyAccel` in `src/Feather/Resources/` (like
   `ReadOnlyBuffer<T>`), holding the accel binding for kernel parameters.
2. **Generator resource recognition** (`ShaderModelFactory.cs`):
   - `ResourceKindModel.Accel` in `ShaderModels.cs`;
   - map `"global::Feather.Resources.ReadOnlyAccel"` to it (near line 1477);
   - `ToIrResourceKind` in `FeatherIrWriter.cs` writes the new kind.
3. **Generator call lowering**: recognize `ReadOnlyAccel.TraceClosest(Ray)`
   in `ValidateCall` and the elementwise expression lowerer (near the
   ShaderMath builtins), emitting a new FEIR call opcode; `Ray` and
   `SurfaceHit` are plain FEIR structs (float3/float/float and uint/uint/float2/float).
4. **Native parser** (`feather_typed_ir.cpp`): accept the accel resource kind
   and the TraceClosest opcode (validate operand/result types).
5. **Lowerer** (`feather_luisa_xir.cpp`): accel resource -> LC `Type::accel()`
   function argument (`AccelVar`); TraceClosest -> `accel.intersect(ray, {})`;
   Ray/SurfaceHit struct lowering like existing FEIR structs.
6. **Dispatch** (`feather_luisa_backend.cpp`): `HostAccelBinding` span added
   to `Dispatch`; bind `Function::AccelBinding{accel->handle()}` in
   `bound_arguments`; RuntimeState keeps accels alive (already added in
   stage 1).

## Implementation phases

1. Native accel handle + C API + dispatch binding (no DSL yet).
2. FEIR type/opcode + parser + lowerer (`TraceClosest`).
3. Generator: `Ray`/`SurfaceHit` structs + `TraceClosest` call lowering.
4. Managed `GpuMesh`/`GpuAccel`/`ReadOnlyAccel` + `GPU.CreateMesh/CreateAccel`.
5. Integration test (single triangle, like the verified LC smoke test) +
   RayTracing sample hardware path.

## Verification

- New integration test: dispatch a `TraceClosest` kernel against a two-triangle
  accel; assert `SurfaceHit` fields (instance/primitive/t) — the exact scenario
  already proven on LC directly.
- RayTracing sample renders the Cornell box through MetalRT and passes the
  existing image-variation proof.
- Full suite regression (AD 66, NN 78, Integration 301) stays green.

## Open questions

- `TVertex` layout rules: reuse `GpuValueLayout<T>`; require a 3-float vertex
  (float3 or 12-byte struct) and `uint` indices initially.
- Motion blur / curves: out of scope for the first milestone.
