# API Reference

This reference maps Feather's public API by subsystem. It is written for application code first, with notes for shader-only markers and generated contracts where they matter.

Feather code usually uses these namespaces:

| Namespace | Purpose |
| --- | --- |
| `Feather` | Runtime entry point, contexts, kernel attributes, dispatch IDs, barriers, atomics, profiler. |
| `Feather.Resources` | Buffers, textures, samplers, uniforms, shader-facing resource views. |
| `Feather.Math` | Vectors, matrices, swizzles, shader math, HLSL-style aliases. |
| `Feather.Graphics` | Preview raster graphics pipeline and draw-state types. |
| `Feather.Windowing` | Native windows, events, pixel buffers, and texture presentation. |
| `Feather.AD` | Preview automatic differentiation markers, AD kernel wrapper, gradient set. |
| `Feather.NN` | Preview tensor, module, optimizer, training, and checkpoint APIs. |
| `Feather.Interop` | Generated contracts, FEIR readers, resource descriptors, shader inspection helpers. |

## Reference Pages

- [Core Runtime](api/core.md): `GPU`, `GpuContext`, capabilities, dispatch, profiler, attributes, diagnostics.
- [Resources](api/resources.md): buffers, textures, samplers, uniforms, access modes, layout and lifetime.
- [Kernels](api/kernels.md): kernel interfaces, thread IDs, group IDs, callables, shader libraries, barriers, shared memory, atomics.
- [Math](api/math.md): vector and matrix types, swizzles, `ShaderMath`, `Hlsl`.
- [Graphics](api/graphics.md): vertex/fragment shaders, pipelines, state objects, draw calls.
- [Windowing](api/windowing.md): `GpuWindow`, events, pixel buffers, texture presenters.
- [Automatic Differentiation](api/autodiff.md): `AD`, `GpuADKernel<T>`, `GradientSet`.
- [Neural Networks](api/nn.md): tensors, parameters, modules, tensor ops, losses, optimizers, training steps.
- [Interop and Inspection](api/interop.md): FEIR readers, shader inspection, dispatch paths, native asset override.

## Common Program Shape

```csharp
using Feather;
using Feather.Math;
using Feather.Resources;

using var input = GPU.CreateBuffer<float>([1, 2, 3, 4], BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(4, BufferAccess.ReadWrite);

GPU.Dispatch(new ScaleKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(2.0f)), 4);

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct ScaleKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<float> scale) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * scale.Value;
    }
}
```

## Host API Vs Shader API

Feather has two kinds of APIs:

| Kind | Examples | Where it runs |
| --- | --- | --- |
| Host/runtime API | `GPU.CreateBuffer`, `GPU.Dispatch`, `GpuWindow.Create`, `GpuADKernel.Backward` | Normal .NET code. |
| Shader-facing values | `ReadWriteBuffer<T>`, `Uniform<T>`, `ThreadIds.X`, `AD.Parameter` | Inside generated kernel/shader methods. |

Shader-facing marker methods throw if called on the CPU. They exist so the Roslyn generator can recognize intent and lower it into FEIR.

## Related Guides

- [Getting Started](getting-started.md)
- [Tutorial](tutorial.md)
- [C# Shader Subset](csharp-subset.md)
- [FEIR Compiler Pipeline](feir.md)
- [Diagnostics](diagnostics.md)
