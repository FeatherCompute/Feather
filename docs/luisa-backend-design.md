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

`Feather::Luisa::TryLowerToXir` owns a Luisa `xir::Module` and translates each
typed FEIR function into a `xir::KernelFunction` or `xir::CallableFunction`.
It maintains maps from FEIR type, function, value, lvalue, and resource IDs to
their XIR counterparts. Translation is staged so recursive callable symbols
and resource arguments exist before bodies are emitted.

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

M2.1 implements the first general subset: one-dimensional scalar buffer
expressions composed of typed resource reads, dispatch index, constants, and
arithmetic, followed by a buffer write. Unsupported constructs fail explicitly
on the Luisa route and never silently switch backend.

## Resources And Scheduling

Buffer resource arguments use `Type::buffer(elementType)` and XIR
`BUFFER_READ`/`BUFFER_WRITE`; textures use `Type::texture` with
`TEXTURE2D/3D_READ/WRITE`; samplers become the filter/address operands required
by Luisa sampling calls. Push constants become value arguments with the same
FEIR packing and alignment contract. Read/write access is retained for
validation and dirty-state tracking.

M2.1 establishes correctness with host staging: synchronize dirty EasyGPU
buffers to Feather host bytes, upload those bytes to typed Luisa buffers,
dispatch on a Luisa compute stream, download writable buffers, then mark the
Feather host copy authoritative. This avoids assuming cross-runtime Vulkan
ownership. M2.2 will replace staging with explicitly negotiated Vulkan buffer
and image import, queue-family ownership transfers, layout transitions,
timeline synchronization, and lifetime tracking.

Luisa dispatch receives Feather's logical extent, while XIR kernel block size
comes from FEIR's thread-group metadata. `wait: true` synchronizes the Luisa
stream before returning. Asynchronous dispatch will require retained staging
and completion objects; until that is implemented, the Luisa route rejects
`wait: false` rather than weakening lifetime guarantees.

## Backend Selection

Managed code exposes `GpuExecutionBackend.EasyGpu` and
`GpuExecutionBackend.Luisa`. `GpuKernel.Create<TKernel>(context)` continues to
select EasyGPU. An overload accepting the enum records the choice on the native
kernel before bindings and dispatch. Existing `GPU.Dispatch` and cached-kernel
behavior are unchanged. A Luisa selection is explicit, per kernel, observable
through `LastDispatchPath`, and never falls back to EasyGPU or CPU when Luisa
translation or execution fails.

## Coverage Plan

| FEIR feature | M2.1 | M2.2 plan |
| --- | --- | --- |
| Scalar types/constants/casts | int/uint/float slice | Complete scalar and bitcast matrix |
| Vectors/matrices | Types mapped, not executed | Constructors, swizzles, arithmetic, matrix ops |
| Arrays/struct aggregates | Design only | Layout-checked constants, GEP, load/store |
| Control flow | Straight-line slice | If/switch/loops, break/continue, phi normalization |
| Local/shared memory | Design only | Alloca address spaces, barriers, block-size validation |
| Buffer memory | Typed read/write | Byte buffers, volatile access, atomics, zero-copy interop |
| Atomics | API mapping identified | Full FEIR atomic operation/type matrix |
| Callables/shader libraries | Symbol staging designed | Recursive call graph, captures, deduplication |
| Textures/samplers | Mapping designed | 2D/3D read/write/sample, mip/grad, format validation |
| Dispatch/bounds checks | 1D dispatch index | 2D/3D IDs, logical bounds, indirect dispatch |
| Automatic differentiation | Not routed | XIR autodiff parity only after forward coverage |
| Graphics/ray tracing | Not routed | Separate milestones after compute parity |

## Risks And Gates

The highest risks are exact aggregate layout parity, Vulkan resource sharing
between two runtimes, callable/resource capture ordering, and differences
between FEIR structured control flow and XIR CFG verification. Every expansion
must add backend-parity tests and preserve explicit rejection for unsupported
features. LuisaCompute remains a pristine pinned submodule; compatibility is
handled only through supported build options and Feather-owned integration
code.
