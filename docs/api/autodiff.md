# API Reference: Automatic Differentiation

## Purpose

AD APIs mark differentiable values in generated kernels, run native backward dispatch, and expose named gradients.

## Shader Markers

```csharp
AD.Parameter(w[0]);
AD.Loss(lossValue);
```

| API | Purpose |
| --- | --- |
| `AD.Parameter(float)` | Marks a scalar parameter. |
| `AD.Parameter(float2/float3/float4)` | Marks a vector parameter. |
| `AD.Loss(float)` | Marks the scalar loss. |
| `AD.Loss(float2/float3/float4)` | Present to reject non-scalar losses in generated code. |

Markers are shader-only and throw on the CPU.

## `GpuADKernel<TKernel>`

```csharp
using var ad = GPU.CreateADKernel(kernel);
ad.Backward(count);
ad.CopyGradientToBuffer("weight", gradientBuffer);
```

| API | Purpose |
| --- | --- |
| `Kernel` | Wrapped generated kernel struct. |
| `Gradients` | Lazy named gradient set. |
| `LastBackwardCount` | Count used by the last successful backward call. |
| `HasBackwardRun` | Whether backward has succeeded. |
| `LastDispatchPath` | Native route used by the last forward/backward operation. |
| `Backward(count)` | Runs native AD backward path. |
| `Forward(count)` | Runs forward-only and invalidates gradients. |
| `ReadBackGradients()` | Materializes gradients into managed arrays. |
| `CopyGradientToBuffer(name, destination)` | Reduces/copies a named gradient into a GPU buffer. |
| `GetBackwardGLSL()` | Returns merged forward/backward GLSL. |
| `Dispose()` | Releases native kernel objects. |

`Backward(count)` requires a positive count and a generated 1D kernel.

## `GradientSet`

| API | Purpose |
| --- | --- |
| `Names` | Materialized gradient names. |
| `HasMaterializedValues` | Whether arrays have been loaded. |
| `Contains(name)` | Checks for a named gradient. |
| `Register<T>(name, value/span)` | Adds values manually, mostly for tests. |
| `Clear()` | Clears materialized values. |
| `TryGet<T>(name, out value)` | Gets a scalar gradient. |
| `TryGetArray<T>(name, out values)` | Gets an array gradient. |
| `Get<T>(name)` / `GetArray<T>(name)` | Gets an array or throws. |
| `GetScalar<T>(name)` | Gets one scalar or throws. |

## Host Vs Shader

- `AD.Parameter` and `AD.Loss` only appear inside generated GPU code.
- `GpuADKernel<T>` and `GradientSet` are host/runtime APIs.
- Gradients are native buffers until copied or read back.

## Lifetime And Errors

- `GpuADKernel<T>` is disposable.
- `Backward(count)` requires a positive count and a generated 1D kernel.
- `ReadBackGradients()` and `CopyGradientToBuffer(...)` require a successful backward run.
- Unsupported AD source shapes fail explicitly; forward-only dispatch is not AD success.

## Samples And Tests

- `samples/AdLinearRegression`
- `samples/AutoDiffLinearRegression`
- `samples/AdTransformer`
- `samples/ProfilerSuite`
- `tests/Feather.AD.Tests/ADSurfaceTests.cs`
- `tests/Feather.Integration.Tests/AutoDiffNativeBridgeTests.cs`

## Guide

See [Automatic Differentiation](../autodiff.md).
