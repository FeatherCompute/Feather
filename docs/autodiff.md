# Automatic Differentiation

Feather's automatic differentiation path lets a generated 1D kernel describe a forward computation, mark differentiable parameters, mark a scalar loss, and ask EasyGPU to generate the reverse-mode adjoint work.

This is a preview API, but it is a real GPU path: unsupported AD shapes fail explicitly rather than pretending a forward-only dispatch is a successful backward pass.

## Why AD Exists In Feather

AD is useful when the thing you want is not just a value but the derivative of a value:

- Fit parameters in a small model.
- Solve inverse rendering or simulation problems.
- Train compact GPU-resident experiments without switching to Python.
- Use `Feather.NN` optimizers with gradients produced by generated C# kernels.

The user-facing model is:

1. Write a normal generated 1D kernel for the forward pass.
2. Add `[AutoDiff]`.
3. Mark parameter values with `AD.Parameter(...)`.
4. Mark one scalar loss value with `AD.Loss(...)`.
5. Run `GpuADKernel<T>.Backward(count)` or `TrainingStep<TKernel>.Run()`.
6. Copy gradients into parameter gradient buffers or read them back for debugging.

## Minimal AD Kernel

```csharp
using Feather;
using Feather.Resources;
using ADMarker = Feather.AD.AD;

[Kernel]
[AutoDiff]
[ThreadGroupSize(1, 1, 1)]
public readonly partial struct LinearLossKernel(
    ReadOnlyBuffer<float> x,
    ReadOnlyBuffer<float> y,
    ReadWriteBuffer<float> w,
    ReadWriteBuffer<float> b,
    ReadWriteBuffer<float> loss) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float pred = (w[0] * x[i]) + b[0];
        float error = pred - y[i];
        float l = error * error;

        loss[i] = l;
        ADMarker.Parameter(w[0]);
        ADMarker.Parameter(b[0]);
        ADMarker.Loss(l);
    }
}
```

Rules in this example:

- The kernel is 1D.
- `[AutoDiff]` asks the generator to emit AD metadata.
- `AD.Parameter` marks values that should receive gradients.
- `AD.Loss` marks the scalar value that seeds the backward pass.
- The loss is written to a buffer so host code can report or reduce it.

## Host API

Use `GPU.CreateADKernel(...)` when you want direct control:

```csharp
using var ad = GPU.CreateADKernel(new LinearLossKernel(
    x.AsReadOnly(),
    y.AsReadOnly(),
    w.AsReadWrite(),
    b.AsReadWrite(),
    loss.AsReadWrite()));

ad.Backward(count: 1);

ad.CopyGradientToBuffer("w", wGradient);
ad.CopyGradientToBuffer("b", bGradient);

float[] debugW = ad.ReadBackGradients().Get<float>("w");
string backwardGlsl = ad.GetBackwardGLSL();
```

Important members:

| API | Purpose |
| --- | --- |
| `GPU.CreateADKernel<TKernel>(kernel)` | Creates the managed AD wrapper. |
| `GpuADKernel<T>.Forward(count)` | Runs forward-only and invalidates previous gradient state. |
| `GpuADKernel<T>.Backward(count)` | Runs the AD path and leaves native gradient buffers available. |
| `GpuADKernel<T>.CopyGradientToBuffer(name, destination)` | Reduces/copies a named gradient into a GPU buffer. |
| `GpuADKernel<T>.ReadBackGradients()` | Materializes gradients into managed arrays for tests or debugging. |
| `GpuADKernel<T>.GetBackwardGLSL()` | Returns the merged forward/backward GLSL inspection dump. |
| `GpuADKernel<T>.LastDispatchPath` | Reports the route used by the most recent backward launch. |

## TrainingStep And Optimizers

For NN-style loops, prefer `TrainingStep<TKernel>`:

```csharp
var optimizer = new SGD([weight, bias], learningRate: 0.05f);
using var trainingStep = TrainingStep<LinearRegressionAdKernel>.Create(
    new LinearRegressionAdKernel(
        xBuffer.AsReadOnly(),
        yBuffer.AsReadOnly(),
        weight.Value.Buffer.AsReadWrite(),
        bias.Value.Buffer.AsReadWrite(),
        lossBuffer.AsReadWrite()),
    [weight, bias],
    optimizer,
    lossBuffer,
    count: 1);

float loss = trainingStep.Run();
```

`TrainingStep` runs backward, copies gradients into parameter gradient buffers, calls the optimizer, and reports the last loss. This is what `samples/AdLinearRegression` uses.

## Gradient Names

Gradient names come from the generated AD metadata. For simple resource parameters, names often match the resource name or an alias added to the `Parameter<T>`:

```csharp
weight.AddGradientAlias("w");
bias.AddGradientAlias("b");
```

Use aliases when the kernel resource name and the NN parameter name differ. `CopyGradientToBuffer` validates both the name and destination length.

## What Runs On The GPU

The AD path is:

1. Roslyn generator emits typed FEIR and AD annotations.
2. Native Feather registers parameter buffers and scalar loss markers with EasyGPU `GradientTape`.
3. EasyGPU records differentiable operations while lowering the forward module.
4. EasyGPU generates an adjoint body and merges forward/backward GLSL.
5. Native Feather dispatches the merged kernel and stores gradients in separate gradient buffers.
6. Managed Feather reads or copies those gradient buffers by stable name.

Parameter value buffers are not used as gradient output buffers. Gradients live in native gradient buffers until you copy or read them.

## Supported AD Contract

Current AD support is intentionally narrow and explicit:

| Supported | Notes |
| --- | --- |
| Generated 1D kernels | `GpuADKernel<T>` requires `IKernel1D`. |
| Scalar and vector float parameters | `float`, `float2`, `float3`, `float4` markers exist; vector gradients are flattened for native storage and rebuilt by managed readback. |
| Scalar loss | `AD.Loss(float)` is the supported loss marker. Non-scalar loss overloads reject in generated GPU code. |
| Structured `if/else` | Lowered through typed IR when the expression/body shape is supported. |
| Canonical counted `for` loops | Supported by the current native AD bridge. |
| Shader math intrinsics | Supported intrinsics lower by Roslyn symbol identity. |

Unsupported or rejected:

- `while`, `do-while`, `break`, and `continue` in current AD lowering.
- Reference type allocation, exceptions, async/await, virtual dispatch, and ordinary .NET collections.
- Missing `AD.Parameter` or missing `AD.Loss`.
- Loss shapes that are not scalar.
- Native backend failures, shader compile failures, and unsupported typed IR nodes.

## Loss Scaling

`AD.Loss(l)` seeds gradients from the scalar `l` value produced by each logical thread. If you dispatch multiple logical elements, think carefully about reduction semantics:

- Use one logical element when the kernel already computes one scalar loss.
- Write per-element losses to a buffer if you need host-side reporting.
- Prefer explicit averaging or scaling in the forward kernel when optimizer step size depends on batch size.

The current AD bridge exposes per-dispatch gradient storage and device-side reduction when copying a named gradient into a destination buffer.

## Debugging

Use these checks when an AD kernel fails:

```csharp
using var ad = GPU.CreateADKernel(kernel);
ad.Backward(count);

Console.WriteLine(ad.LastDispatchPath);
Console.WriteLine(ad.GetBackwardGLSL());
Console.WriteLine(string.Join(", ", ad.Gradients.Names));
```

If `Backward` throws:

1. Check generator diagnostics first.
2. Call `ShaderInspection.GetGLSL<TKernel>()` to confirm the forward kernel lowers.
3. Confirm that all `AD.Parameter` calls refer to buffer-backed float values.
4. Confirm that exactly the intended scalar loss reaches `AD.Loss`.
5. Read [AD Internals And Coverage](ad-implementation-note.md) for native bridge details.

## Samples

- `samples/AdLinearRegression`: smallest AD + optimizer loop.
- `samples/AutoDiffLinearRegression`: direct AD-focused regression sample.
- `samples/AdTransformer`: larger NN-style AD usage.
- `samples/ProfilerSuite`: measures AD dispatch and gradient materialization.

## Next Reading

- [Neural Networks](nn.md)
- [API: Automatic Differentiation](api/autodiff.md)
- [AD Internals And Coverage](ad-implementation-note.md)
- [FEIR Compiler Pipeline](feir.md)
