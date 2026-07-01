# AD Internals And Coverage

This page explains how Feather's AD user API maps to the native EasyGPU AD bridge. Read [Automatic Differentiation](autodiff.md) first if you only want to write AD kernels.

Read this page when:

- `GpuADKernel<T>.Backward(...)` throws.
- A gradient name cannot be matched to a parameter.
- You need to inspect merged backward GLSL.
- You are changing generator AD metadata, native typed lowering, or EasyGPU AD code.
- You need to run the AD coverage gate before a release.

## User API To Native Bridge

```text
[AutoDiff] C# kernel
  -> AD.Parameter / AD.Loss markers
  -> FEIR section 7 typed IR + AD metadata
  -> native Feather validation
  -> EasyGPU GradientTape registration
  -> forward module lowering
  -> adjoint generation
  -> merged forward/backward GLSL
  -> native gradient buffers
  -> managed GradientSet or device buffer handoff
```

The AD path is no longer a managed-only placeholder or forward-only success marker. Unsupported AD shapes fail through generator diagnostics or native `FE_ERROR_UNSUPPORTED`.

## Current Bridge

- Generated kernels emit AD parameter/loss metadata plus section 7 typed IR.
- Native Feather registers buffer parameters and the scalar loss with EasyGPU `GradientTape` before forward lowering records differentiable operations.
- EasyGPU generates an adjoint body, merges forward/backward GLSL, and writes gradients into separate native gradient buffers.
- `GpuADKernel<T>.Backward(...)` returns only after native AD dispatch succeeds.
- Named gradients remain native until read back or copied/reduced into destination GPU buffers.
- `GradientSet` is lazy: `ReadBackGradients()` materializes arrays for debugging and tests; optimizers should prefer device handoff.

## Gradient Metadata

Each native gradient carries:

- Stable gradient name.
- Source resource binding.
- Element type.
- Logical element count.
- Element stride.
- Gradient byte size.
- Native gradient SSBO binding.
- Vector component count.

Managed matching accepts `Parameter<T>.GradientNames`, so callers can add aliases when AD resource names differ from model parameter names.

## Control Flow Contract

The native AD bridge currently accepts:

- Structured `if/else`.
- Canonical counted `for` loops.
- Supported scalar/vector float expressions.
- Supported callables and math intrinsics.

It rejects:

- `while`.
- `do-while`.
- `break`.
- `continue`.
- Missing parameter or loss metadata.
- Non-scalar loss.
- Unsupported l-values or source metadata.

This is stricter than ordinary compute support.

## Failure Semantics

AD failures are explicit. These conditions return a non-OK native result and become managed exceptions:

- Missing parameters.
- Missing loss.
- Unsupported AD source metadata.
- Unsupported AD control flow.
- Gradient-buffer allocation failure.
- Shader compile failure.
- Backend execution failure.
- Empty gradient result.

Forward-only dispatch does not count as AD success.

## Debugging Steps

1. Confirm the forward kernel lowers:

   ```csharp
   Console.WriteLine(ShaderInspection.GetGLSL<MyAdKernel>());
   ```

2. Run backward and inspect the dispatch path:

   ```csharp
   using var ad = GPU.CreateADKernel(kernel);
   ad.Backward(count);
   Console.WriteLine(ad.LastDispatchPath);
   ```

3. Inspect gradient names:

   ```csharp
   var gradients = ad.ReadBackGradients();
   Console.WriteLine(string.Join(", ", gradients.Names));
   ```

4. Inspect merged GLSL:

   ```csharp
   Console.WriteLine(ad.GetBackwardGLSL());
   ```

5. If optimizer handoff fails, compare native gradient names with `Parameter<T>.GradientNames`.

## Coverage Gate

Run the full AD proof with:

```bash
python3 scripts/ad-industrial-gate.py
```

The managed coverage gate enforces scoped aggregate line and branch coverage >= 90%, plus per-file scoped line coverage >= 90%.

The native coverage gate:

- Builds an LLVM-instrumented native library.
- Runs AD tests through `FEATHER_NATIVE_LIBRARY`.
- Runs the native EasyGPU AD coverage probe.
- Gates aggregate scoped native AD line coverage >= 80% across the Feather native bridge and EasyGPU AD implementation files.

High-risk native files must also have line coverage >= 80%. `EasyGPU/source/AD/GradientTape.cpp` and `EasyGPU/source/AD/AdjointGenerator.cpp` are mandatory high-risk files.

Managed-only or native aggregate-only results are not sufficient.

## Related Docs

- [Automatic Differentiation](autodiff.md)
- [API: Automatic Differentiation](api/autodiff.md)
- [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md)
- [Native ABI](native-abi.md)
