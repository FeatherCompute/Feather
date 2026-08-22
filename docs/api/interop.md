# API Reference: Interop And Inspection

## Purpose

Interop APIs expose generated contracts, FEIR readers, shader inspection helpers, GPU layout metadata, and fixed GPU array wrappers.

Most users only need `ShaderInspection`. Contributors and advanced users may also inspect `FeatherIr` models and generated descriptors.

## Shader Inspection

```csharp
using Feather.Interop;

string ir = ShaderInspection.GetIR<MyKernel>();
string glsl = ShaderInspection.GetGLSL<MyKernel>();
string optimized = ShaderInspection.GetOptimizedGLSL<MyKernel>();
string optimizationReport = ShaderInspection.GetOptimizationReport<MyKernel>();
ResourceDescriptor[] resources = ShaderInspection.GetResources<MyKernel>();
```

| API | Purpose |
| --- | --- |
| `GetIR<TKernel>()` | Returns serialized FEIR as uppercase hex. |
| `GetGLSL<TKernel>()` | Builds through EasyGPU and returns unoptimized GLSL. |
| `GetOptimizedGLSL<TKernel>()` | Returns backend-optimized GLSL inspection text. |
| `GetOptimizedIR<TKernel>()` | Returns optimized target IR, such as readable SPIR-V assembly on Vulkan. |
| `GetOptimizationReport<TKernel>()` | Returns the active backend's versioned structured optimization report. |
| `GetResources<TKernel>()` | Returns generated resource descriptors. |
| `GetGraphicsSource<TVS,TFS,TVaryings>()` | Returns graphics FEIR source payloads. |

`GetOptimizationReport<TKernel>()` currently exposes the backend-owned
`EasyGPU.ShaderOptimizationReport` JSON contract. Vulkan reports stable reason codes and
cost inputs for inline, loop-unroll, barrier, vectorization, and specialization decisions,
plus actual whole-shader code-size metrics. Driver register allocation and occupancy are
marked unavailable when the backend cannot measure them; static instruction pressure is
not relabeled as a physical-register count.

For already-loaded work, `GetLoadedShaders(assembly)` returns the exact backend-input GLSL,
optimized GLSL and target IR when available, and `OptimizationReportJson` for each compute,
vertex, or fragment stage. It inspects cached live handles without compiling a second shader
or incrementing shader cache counters. Optional optimized fields are empty on unsupported
backends.

## Generated Contracts

Generated shaders implement internal/public interop contracts such as `IGeneratedKernel<T>` and `IGeneratedGraphicsPipeline<TVS,TFS,TVaryings>`. Application code usually does not implement these manually; the source generator emits them.

Generated descriptors include:

- Kernel dimension.
- Thread-group size.
- Resource descriptors.
- AutoDiff flag.
- Bounds-check flag.
- Serialized FEIR bytes.
- Callable metadata in the serialized typed IR, including mangled names and parameter directions.

## FEIR Reader

`FeatherIr.Read(ReadOnlySpan<byte>)` parses a FEIR payload into managed records:

- `FeatherIrModule`
- `FeatherIrResource`
- `FeatherIrInstruction`
- `FeatherIrElementwiseAssignment`
- `FeatherIrElementwiseExpressionAssignment`
- `FeatherIrAdAnnotation`
- `FeatherIrExpressionNode`

Use this in tests, diagnostics, and tooling that needs structured access to generated FEIR.

The public reader exposes the stable outer payload and legacy structured sections. Section 7 typed IR is the canonical native route and is documented in [FEIR Binary Format](../ir-format.md) for contributors, but it is not currently exposed as a complete public object model.

## Layout Metadata

`GpuValueLayout<T>` and generated `[GpuStruct]` metadata describe CPU/GPU size, alignment, buffer stride, and fixed-array layout.

`GpuArrayN<T>` wrappers represent fixed arrays in GPU structs. Supported sizes include common small sizes and larger fixed sizes such as `GpuArray16<T>`, `GpuArray32<T>`, `GpuArray64<T>`, `GpuArray128<T>`, and `GpuArray256<T>`.

## Native Asset Override

Interop with native code is normally automatic. To force a specific native library:

```bash
export FEATHER_NATIVE_LIBRARY=/absolute/path/to/libfeather.dylib
```

See [Native ABI](../native-abi.md) and [Packaging](../packaging.md).

## Dispatch Paths

`DispatchPath` is exposed from core APIs but commonly used with inspection:

```csharp
DispatchPath path = GPU.DispatchAndGetPath(kernel, count);
Console.WriteLine(path);
```

`TypedEasyGpu` is the expected route for supported modern kernels.

## Host Vs Shader

Interop and inspection APIs are host-side. They expose generated shader metadata, FEIR bytes, native route information, and layout data; they are not called from shader entry bodies.

## Lifetime And Errors

- `ShaderInspection.GetGLSL<TKernel>()` creates native kernel state internally and can throw native/backend exceptions.
- Direct optimized-output and optimization-report getters throw when the active backend cannot expose that representation; loaded-shader snapshots leave optional fields empty instead.
- `FeatherIr.Read(...)` validates binary payload shape and can throw for malformed data.
- Native library override is process/environment configuration, not a per-kernel option.

## Guide

See [FEIR Compiler Pipeline](../feir.md), [FEIR Binary Format](../ir-format.md), and [Diagnostics](../diagnostics.md).

## Samples And Tests

- `samples/SpirvOptInspection`
- `samples/GpuStructInterfaces`
- `samples/ProfilerSuite`
- `tests/Feather.Integration.Tests/GeneratedComputeDispatchTests.cs`
- `tests/Feather.Integration.Tests/NativeResourceRoundTripTests.cs`
