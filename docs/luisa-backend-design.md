# Luisa Vulkan Backend Design

## Existing FEIR Execution

The source generator serializes each generated shader as FEIR. The native
bridge validates the envelope in `native/feather_ir_bridge.cpp`; section 7 is
parsed by `native/feather_typed_ir.cpp` into `Feather::TypedIR::Module`.
`TryLowerToEasyGpuModule` in `native/feather_typed_ir_lowerer.cpp` registers
resources and callables, lowers typed statements and expressions through
`GPU::IR::ModuleBuilder`, and returns an EasyGPU IR module. During dispatch,
`native/feather_c_api.cpp` turns that module into a
`GPU::Kernel::KernelBuildContext`, obtains GLSL, asks the active EasyGPU Vulkan
backend to compile it to SPIR-V, binds Feather resources, and submits work.

The Luisa path branches after section 7 parsing and before
`TryLowerToEasyGpuModule`. Parsing, validation, generated bindings, logical
dispatch dimensions, and Feather resource ownership remain shared. EasyGPU is
unchanged and remains the default.

## FEIR To XIR

The Feather-owned translator in `native/feather_luisa_xir.cpp` creates a Luisa
`xir::Module` and translates each typed FEIR compute entry and callable into a
`xir::KernelFunction` or `xir::CallableFunction`. It maintains maps from FEIR
types, functions, locals, l-values, and resources to XIR values. Callable
symbols and parameter lists are staged before bodies, so nested call graphs and
resource arguments do not depend on declaration order.

The primary Luisa 0.9.0 API is:

- `luisa/xir/module.h`: `xir::Module`, kernels, callables, constants, and
  dispatch/thread special registers.
- `luisa/xir/function.h`: function definitions, basic blocks, value/reference/
  resource arguments, and kernel block size.
- `luisa/xir/builder.h`: `xir::XIRBuilder` instruction insertion, arithmetic,
  control flow, memory, atomics, and resource operations.
- `luisa/xir/instructions/resource.h` and `luisa/xir/op.h`: typed buffer,
  texture, bindless, and atomic operation contracts.
- `luisa/xir/verifier.h`: structural and semantic verification before codegen.
- `luisa/xir/translators/xir2ast.h`: the public 0.9.0 execution bridge.

Luisa 0.9.0 `Device` does not accept XIR directly. Feather therefore verifies
and normalizes XIR, calls `xir_to_ast_translate`, wraps the returned
`detail::FunctionBuilder` as a Luisa `Function`, and invokes
`Device::compile`. The selected device is always `vk`; the resulting AST is an
internal handoff to Luisa's Vulkan XIR/SPIR-V backend, not a Feather DSL or CPU
fallback.

M2.2 covers every forward-compute statement, expression, l-value, and resource
kind emitted by the current FEIR generator. Invalid FEIR and unsupported future
record kinds fail explicitly on the Luisa route and never silently switch
backend.

## Resources And Scheduling

Buffer resource arguments use `Type::buffer(elementType)` and XIR
`BUFFER_READ`/`BUFFER_WRITE`; textures use `Type::texture` with
`TEXTURE2D/3D_READ/WRITE`; samplers become the filter/address operands required
by Luisa sampling calls. Push constants become value arguments with the same
FEIR packing and alignment contract. Read/write access is retained for
validation and dirty-state tracking.

Correctness currently uses host staging: synchronize dirty EasyGPU
buffers to Feather host bytes, upload those bytes to typed Luisa buffers,
dispatch on a Luisa compute stream, download writable buffers, then mark the
Feather host copy authoritative. Aggregate buffers are repacked recursively
between Feather's declared offsets and XIR's device layout. Textures use the
same staging contract and validate their Vulkan pixel storage before creation.
This avoids assuming cross-runtime Vulkan ownership. Zero-copy Vulkan sharing
remains a performance milestone: it requires explicit buffer/image import,
queue-family ownership transfers, layout transitions, timeline synchronization,
and lifetime tracking, but does not change forward semantics.

Luisa dispatch receives Feather's logical extent, while XIR kernel block size
comes from FEIR's thread-group metadata. Luisa 0.9.0 requires 32--1024 physical
threads per XIR block in multiples of 32. Feather packs an integral number of
logical groups into each physical block, reconstructs exact 1D/2D/3D local and
group IDs, offsets each logical group's shared allocation, and guards padded
dispatch lanes. Barriers remain physical-block barriers, which is correct
because every packed logical group reaches each barrier together. `wait: true`
synchronizes before returning. Asynchronous dispatch remains disabled until
staging allocations can be retained by completion objects.

The native asset staging step places Luisa's runtime, XIR libraries, and Vulkan
backend module beside the Feather library. Feather resolves that directory from
its own loaded module; `FEATHER_LUISA_RUNTIME_DIR` is an explicit override for
development and diagnostics, not a requirement for packaged execution.

## Backend Selection

Managed code exposes `GpuExecutionBackend.EasyGpu` and
`GpuExecutionBackend.Luisa`. `GpuKernel.Create<TKernel>(context)` continues to
select EasyGPU. An overload accepting the enum records the choice on the native
kernel before bindings and dispatch. Existing `GPU.Dispatch` and cached-kernel
behavior are unchanged. A Luisa selection is explicit, per kernel, observable
through `LastDispatchPath`, and never falls back to EasyGPU or CPU when Luisa
translation or execution fails.

## Coverage Plan

| FEIR feature | M2.2 status | Local GPU evidence |
| --- | --- | --- |
| Scalar types/constants/casts | Complete: bool/int/uint/float, numeric casts, unary/binary/bitwise/comparison/logical/select, and supported intrinsics | `LuisaBackendTypeFeatureTests` (5 tests) |
| Vectors/matrices | Complete: 2--4 component constructors, extraction/swizzles/arithmetic, and square 2--4 matrix construction/component and linear-algebra operations | `VectorConstructionAndSwizzlesMatchEasyGpu`; `MatrixConstructionAndLinearAlgebraMatchEasyGpu` |
| Arrays/struct aggregates | Complete: recursive ABI repacking, constants, indexed/field addressing, loads, stores, and nested writeback | `StructAggregateLoadsAndNestedFieldExtractionMatchEasyGpu`; `StructArraysAndNestedWritebackMatchEasyGpu` |
| Control flow | Complete: if/else, for/while/do-while, break/continue/return and expression statements; mutable locals remove FEIR phi requirements before XIR CFG normalization/verification | `StructuredControlFlowAndMutableLocalsMatchEasyGpu` (four kernels) |
| Local/shared memory | Complete: local/shared allocas, logical-group shared offsets, barriers, and 32--1024 physical block validation | `SharedMemoryLocalIdsAndBarrierMatchEasyGpu` |
| Buffer memory | Complete for FEIR: typed reads/writes, aggregate field/index writes, push constants, and access validation | Type, control/memory, and resource/dispatch suites |
| Atomics | Complete: add/sub/min/max/and/or/xor/exchange/compare-exchange on FEIR int targets | `FullIntegerAtomicMatrixMatchesEasyGpu` |
| Callables/shader libraries | Complete: staged/de-duplicated symbols, nested calls, value/reference parameters, aggregate writeback, and buffer/texture/sampler resource parameters | four callable parity tests in `LuisaBackendResourceDispatchTests` |
| Textures/samplers | Complete for emitted FEIR: 2D/3D load/store; 2D Sample/SampleLevel/SampleGrad; decoded struct-pixel components; format and sampler validation | four texture tests in `LuisaBackendResourceDispatchTests` |
| Dispatch/bounds checks | Complete: 1D/2D/3D global/local/group/size builtins, logical bounds guards, and padded-lane suppression | `TwoAndThreeDimensionalDispatchIdsMatchEasyGpu`; `NonDivisibleLogicalBoundsMatchEasyGpu`; shared-memory test |
| Automatic differentiation | M3 | Not a forward-compute M2.2 feature |
| Graphics/ray tracing | M3 | Separate non-compute pipelines |

The local M2.2 suite contains 19 `[Category=Gpu]` tests; every parity-capable
kernel runs once through EasyGPU and once through Luisa Vulkan and compares GPU
readback element-by-element. `SampleGrad` is executed and checked against its
known texel result on Luisa because EasyGPU's current `ModuleBuilder` has no
`TextureSampleGrad` operation; adding parity there would require changing the
pristine EasyGPU submodule.

The current FEIR schema has no switch statement, recursive callable graph,
byte-buffer or volatile-access node, sampled-3D expression, or indirect-dispatch
record. Those entries are N/A rather than rejected M2.2 features: the generator
rejects switch and recursion, `GpuTexture3D` exposes load/store but no sampling
API, and dispatch is direct. The residual rejection list for generator-emitted
forward compute FEIR is therefore empty. AD, vertex/fragment graphics, and ray
tracing remain explicitly outside M2.2 and are the only translation families
left for M3.

## Risks And Gates

The remaining risks are cross-runtime zero-copy ownership, asynchronous staging
lifetimes, Luisa's single filter/address sampler representation versus Feather's
richer descriptor, and the M3 AD/graphics lowering model. The Luisa route
validates descriptors it cannot represent rather than changing their meaning.
LuisaCompute remains a pristine pinned submodule; compatibility is handled only
through supported build options and Feather-owned integration code.
