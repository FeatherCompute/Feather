# FEIR Binary Format

This page is the detailed format reference for Feather IR payloads. Start with [FEIR Compiler Pipeline](feir.md) if you want the user-level explanation.

Generated Feather kernels emit a versioned binary IR payload beginning with `FEIR`. The binary format is primarily useful when debugging the generator, native bridge, or compatibility between managed and native code.

## Baseline Header

| Offset | Field |
| --- | --- |
| 0 | Magic `FEIR` |
| 4 | `uint16` major version |
| 6 | `uint16` minor version |
| 8 | Endian marker, `1` for little endian |
| 9 | Shader kind |
| 10 | `uint16` section count; `0` for v1.0 payloads |
| 12 | Thread group X |
| 16 | Thread group Y |
| 20 | Thread group Z |
| 24 | Entry point string id |
| 28 | Resource count |
| 32 | Push constant count, matching resource records whose kind is `5` |
| 36 | Instruction count |
| 40 | String table byte length |

## Resource Records

Each resource record is 15 bytes:

| Field | Type |
| --- | --- |
| Binding | `uint32` |
| Kind | `uint8` |
| Access | `uint8` |
| Reserved | `uint8` |
| Name string id | `uint32` |
| Element type string id | `uint32` |

Resource kind values:

| Value | Meaning |
| --- | --- |
| 1 | Buffer |
| 2 | 2D texture |
| 3 | Sampler |
| 4 | Uniform constructor resource |
| 5 | Generated push constant |
| 6 | 3D texture |

## Instruction Records

Instruction records follow the resources and are 8 bytes:

| Field | Type |
| --- | --- |
| Opcode | `uint8` |
| Operand kind | `uint8` |
| Reserved | `uint16` |
| Operand string id | `uint32` |

Current instruction opcodes:

| Value | Meaning |
| --- | --- |
| 1 | Local declaration |
| 2 | Assignment |
| 3 | Return |
| 4 | If |
| 5 | For |
| 6 | While |
| 7 | Do |
| 8 | Break |
| 9 | Continue |
| 10 | Invocation |
| 11 | Resource access |
| 12 | Expression |
| 13 | Begin structured block |
| 14 | Else separator |
| 15 | End structured block |
| 16 | Workgroup barrier |
| 17 | Memory barrier |
| 18 | Full barrier |
| 19 | Atomic add |
| 20 | Atomic subtract |
| 21 | Atomic minimum |
| 22 | Atomic maximum |
| 23 | Atomic bitwise AND |
| 24 | Atomic bitwise OR |
| 25 | Atomic bitwise XOR |
| 26 | Atomic exchange |
| 27 | Atomic compare-exchange |
| 28 | Shared-memory declaration |

Operand kind `0` means no operand, `1` means normalized source text, `2` means a canonical elementwise assignment emitted by Roslyn semantic lowering, and `3` means a fully qualified Roslyn symbol identity.

## Section Table

IR v1.1 stores an optional section table after instruction records and before the string table. Each record is:

| Field | Type |
| --- | --- |
| Kind | `uint32` |
| Byte length | `uint32` |

Section payloads immediately follow the table in the same order. The string table remains last and stores a `uint32` count followed by length-prefixed UTF-8 strings.

## Section 1: Structured Assignment

Section kind `1` stores structured elementwise assignment records. The payload starts with:

| Field | Type |
| --- | --- |
| Assignment count | `uint32` |

Each assignment record is 28 bytes:

| Field | Type |
| --- | --- |
| Instruction index | `uint32` |
| Destination binding | `uint32` |
| Left binding | `uint32` |
| Right binding, or `uint.MaxValue` when absent | `uint32` |
| Operation: `1` copy, `2` add, `3` subtract, `4` multiply, `5` divide | `uint8` |
| Right operand kind: `0` none, `1` resource, `2` literal | `uint8` |
| Reserved | `uint16` |
| Index string id | `uint32` |
| Right literal string id, or `uint.MaxValue` when absent | `uint32` |

This section remains useful for compatibility and reference validation. Current generated kernels also carry canonical section 7 typed IR.

## Section 2: Typed Elementwise Expressions

Section kind `2` stores typed elementwise expression assignment records for expression trees such as `(input[i] * 2.0f) + bias[i]`.

Payload header:

| Field | Type |
| --- | --- |
| Assignment count | `uint32` |
| Expression node count | `uint32` |
| Argument index count, present for invocation-capable payloads | `uint32` |

Each expression assignment record is 16 bytes:

| Field | Type |
| --- | --- |
| Instruction index | `uint32` |
| Destination binding | `uint32` |
| Index string id | `uint32` |
| Root node index | `uint32` |

Each invocation-capable expression node record is 40 bytes:

| Field | Type |
| --- | --- |
| Node kind: `1` resource, `2` literal, `3` binary, `4` invocation, `5` push constant | `uint8` |
| Operation: `0` none, `1` add, `2` subtract, `3` multiply, `4` divide | `uint8` |
| Reserved | `uint16` |
| Resource binding, or `uint.MaxValue` when absent | `uint32` |
| Index string id, or `uint.MaxValue` when absent | `uint32` |
| Literal string id, or `uint.MaxValue` when absent | `uint32` |
| Type string id, or `uint.MaxValue` when absent | `uint32` |
| Left node index, or `uint.MaxValue` when absent | `uint32` |
| Right node index, or `uint.MaxValue` when absent | `uint32` |
| Symbol string id for invocation nodes, or `uint.MaxValue` when absent | `uint32` |
| First argument index in the argument table, or `uint.MaxValue` when absent | `uint32` |
| Argument count | `uint32` |

The argument table follows node records. Invocation nodes store fully qualified Roslyn method symbols such as `global::Feather.Math.ShaderMath.Sqrt` or `global::Feather.Math.Hlsl.Dot`.

## Section 7: Canonical Typed IR

Section 7 is the production model for modern generated shaders. It represents:

- Function bodies.
- Typed statements and expressions.
- L-values and resources.
- Callables, overload binding, and parameter directions.
- Struct metadata and fixed arrays.
- Shared memory and barriers.
- Integer atomics.
- Texture load/store/sample/sample-level operations.
- Table ranges.
- AD parameter/loss annotations.

Native compute inspection and dispatch feed section 7 into EasyGPU `GPU::IR::ModuleBuilder`. Older structured assignment sections remain compatibility/reference data for fallback-only payloads.

Important section 7 conventions:

- Primitive type records use kind `1` with primitive ids `0` bool, `1` int, `2` uint, and `3` float plus bit width.
- Array type records use kind `5`; field `A` references the element type and field `B` stores the fixed element count.
- Callable function records store both display name and mangled identity.
- Callable parameter records store direction values: `0` `in`, `1` `out`, and `2` `inout`.
- `[GpuStruct]` instance callables store the lowered receiver as the first parameter named `this`. Mutating receiver methods use direction `2`.
- Generic callable monomorphizations are represented as ordinary callable function records with constructed mangled identities.
- Atomic expression kind `22` stores operation and l-value references.
- Texture-sample expression kind `23` stores sample/sample-level operation and argument ranges.
- Known math, HLSL aliases, and AD markers are resolved by Roslyn symbol identity.

### Section 7 Callable Tables

The section 7 function table stores one entry point plus any reachable callables. Each function record contains:

| Field | Meaning |
| --- | --- |
| Kind | Entry stage or callable function kind. |
| Name string id | Source/display name. |
| Mangled name string id | Stable generated identity used for overloads and constructed generics. |
| Return type id | Type table reference. |
| First parameter | Parameter table index, or `uint.MaxValue` when absent. |
| Parameter count | Number of parameter records. |
| Body statement id | Root statement/block reference. |

Each parameter record contains:

| Field | Meaning |
| --- | --- |
| Direction | `0` `in`, `1` `out`, `2` `inout`. |
| Name string id | Source parameter name, or lowered `this`. |
| Type id | Type table reference. |

Callable call expressions reference the mangled callable name and an argument range. For generic interface monomorphization, the call expression targets the constructed callable or concrete `[GpuStruct]` implementation; no interface-dispatch table is serialized.

## Compatibility Payloads

The transitional string shape for simple assignments is:

```text
ASSIGN1|destinationResource|indexSymbol|operation|leftOperand|rightOperand
```

Resource accesses historically used:

```text
RESOURCE1|resourceName|indexSymbol
```

These forms are retained for compatibility/reference validation. New completed compute features should use the section 7 typed EasyGPU path.

## Validation Entry Points

- Native bridge validation: `native/feather_ir_bridge.cpp`.
- Typed IR lowering: `native/feather_typed_ir_lowerer.cpp`.
- Managed reader model: `src/Feather/Interop/FeatherIrModule.cs`.

## Planned Format Additions

- Richer debug metadata.
- Broader shader-stage metadata.
- More explicit AD transform annotations as the AD surface grows.
- Additional texture/volume sampling metadata as those APIs graduate.
