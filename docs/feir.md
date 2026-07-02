# FEIR Compiler Pipeline

FEIR is Feather's serialized intermediate representation. It is the contract between generated C# shader code and the native EasyGPU bridge.

Most users do not need to read raw FEIR bytes. You should know FEIR exists because it explains:

- Why Feather accepts a GPU-safe C# subset rather than arbitrary .NET code.
- How `ShaderInspection.GetIR<T>()` and `ShaderInspection.GetGLSL<T>()` relate.
- Why resource names, bindings, access modes, and thread-group sizes are deterministic.
- How AD markers and graphics shader stages travel from C# into native code.
- Why a dispatch can report `TypedEasyGpu`, `CpuReferenceFallback`, or `Rejected`.

## Pipeline At A Glance

```text
C# source
  -> Roslyn syntax and semantic model
  -> Roslyn IOperation trees
  -> Feather typed shader model
  -> FEIR binary payload
  -> native Feather ABI
  -> EasyGPU IR ModuleBuilder
  -> GLSL / SPIR-V / backend execution
```

The important design choice is that the native bridge receives typed semantic records, not arbitrary source strings. Calls are identified by Roslyn symbol identity, resources by generated binding metadata, and statements by typed IR nodes.

## What The Generator Emits

For a generated compute kernel, Feather emits:

- Kernel descriptor metadata.
- Resource table: buffers, textures, samplers, uniforms/push constants.
- Thread-group size.
- Entry-point information.
- Serialized FEIR bytes.
- Typed IR section data for statements, expressions, l-values, callables, struct layout, atomics, texture operations, and AD annotations.

For graphics, the generator emits combined graphics metadata plus separate vertex and fragment stage FEIR payloads.

For AD, the generator emits parameter/loss annotations so the native bridge can register them with EasyGPU's gradient tape.

## Why FEIR Uses Typed Sections

Early compatibility payloads could represent simple elementwise assignments such as:

```text
output[i] = input[i] * 2.0f
```

Modern Feather uses section 7 typed IR as the canonical representation. That lets the native bridge validate and lower:

- Structured control flow.
- Resource l-values.
- Push constants.
- Known math intrinsics.
- Kernel-local callables, source-available shader-library callables, and overloads.
- GPU structs and fixed arrays.
- Atomics.
- Texture loads, stores, samples, and sample-level operations.
- AD parameter/loss metadata.

This is why generated kernels can use `ShaderMath.Sqrt`, `Hlsl.Dot`, a kernel-local `[Callable]`, or a source-available `[ShaderLibrary]` helper without the native bridge guessing from source text.

## Inspecting FEIR And GLSL

```csharp
using Feather.Interop;

string irHex = ShaderInspection.GetIR<MyKernel>();
string glsl = ShaderInspection.GetGLSL<MyKernel>();
string optimized = ShaderInspection.GetOptimizedGLSL<MyKernel>();
ResourceDescriptor[] resources = ShaderInspection.GetResources<MyKernel>();
```

Use `GetIR` when you need to confirm that the generator produced metadata. Use `GetGLSL` when you need to inspect the native EasyGPU lowering. Use `GetOptimizedGLSL` when backend optimization is involved.

## Dispatch Paths

`DispatchPath` reports the route used by native execution:

| Path | Meaning |
| --- | --- |
| `TypedEasyGpu` | The FEIR typed path lowered into EasyGPU and executed. This is the expected path for supported features. |
| `CpuReferenceFallback` | Compatibility/reference fallback for older or narrow payloads. New completed DSL features should not rely on this. |
| `Rejected` | Native validation rejected the payload or backend route. Check the exception message and support matrix. |
| `None` | No dispatch has been recorded. |

Samples commonly assert `TypedEasyGpu` so a passing demo proves the real backend path.

## Relationship To EasyGPU

Feather does not generate final backend code directly. The native bridge translates FEIR typed records into EasyGPU's language-neutral IR builder. EasyGPU then lowers that module through the selected backend path.

That separation matters:

- Feather owns C# source analysis and public .NET API shape.
- EasyGPU owns the backend runtime, shader generation, and AD machinery.
- The native Feather ABI owns the stable interop layer between managed code and EasyGPU.

## Where To Go Deeper

- [FEIR Binary Format](ir-format.md): byte layout and section definitions.
- [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md): accepted compute features.
- [Native ABI](native-abi.md): managed/native boundary and handles.
- [AD Internals And Coverage](ad-implementation-note.md): how AD annotations become gradient tape work.
- [API: Interop and Inspection](api/interop.md): public inspection helpers.
