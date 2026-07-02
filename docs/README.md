# Feather Documentation

Feather lets .NET applications write GPU work in C# and execute it through the native EasyGPU runtime. These docs are written for users first: start with a working kernel, then move into images, windows, graphics, AD, NN, inspection, and internals only when you need them.

## Learning Path

| Stage | Read | You will learn |
| --- | --- | --- |
| First kernel | [Getting Started](getting-started.md) | Build the native bridge, run `HelloBuffer`, write a 1D buffer kernel. |
| GPU mental model | [Tutorial](tutorial.md) | Thread IDs, dispatch sizes, buffers, uniforms, textures, callables, shader libraries, inspection. |
| Image compute | [Examples](examples.md) | How Mandelbrot, Julia, SDF, ray tracing, and volume samples are organized. |
| Window output | [Windowing](window.md) | Native windows, events, CPU pixels, GPU texture presentation. |
| Graphics | [Graphics Pipeline](graphics-pipeline.md) | C# vertex/fragment shaders, render targets, depth, sampling, draw state. |
| AD and NN | [Automatic Differentiation](autodiff.md), [Neural Networks](nn.md) | Reverse-mode AD, gradient readback/handoff, tensors, modules, optimizers. |
| Debugging | [Diagnostics](diagnostics.md), [FEIR](feir.md) | Generator errors, IR inspection, GLSL dumps, dispatch path checks. |
| API lookup | [API Reference](api.md) | Public types and methods grouped by subsystem. |

## Visual Tour

| Compute | Graphics | Advanced rendering |
| --- | --- | --- |
| ![Mandelbrot rendered by Feather](img/mandelbrot-feather.png) | ![Sponza renderer](img/sponza.png) | ![SDF renderer](img/sdf-renderer.png) |
| ![Julia set rendered by Feather](img/julia-set.png) | ![Cornell box](img/cornell-box.png) | ![Volumetric fog](img/volumetric-fog.png) |

## Core Guides

- [Getting Started](getting-started.md): prerequisites, native build, first kernel, first dispatch.
- [Tutorial](tutorial.md): progressive compute walkthrough from buffers to textures and inspection.
- [Shader Libraries](shader-libraries.md): reusable `[ShaderLibrary]` + `[Callable]` helpers for BRDFs, SDFs, sampling, and math.
- [Examples](examples.md): sample gallery with recommended learning order.
- [C# Shader Subset](csharp-subset.md): what is legal inside generated GPU code.
- [Windowing](window.md): window creation, event loops, pixels, and texture presentation.
- [Graphics Pipeline](graphics-pipeline.md): preview raster pipelines with C# vertex and fragment shaders.
- [Automatic Differentiation](autodiff.md): preview reverse-mode AD for generated 1D kernels.
- [Neural Networks](nn.md): tensors, parameters, modules, optimizers, trainers, and checkpoints.
- [Diagnostics](diagnostics.md): how to fix generator and runtime errors.
- [Packaging](packaging.md): project references, native asset resolution, local packages.
- [Support Status](support-status.md): current maturity and platform expectations.
- [FAQ](faq.md): short answers to common setup and usage questions.

## API Reference

The public API reference is split by subsystem:

- [Overview](api.md)
- [Core Runtime](api/core.md)
- [Resources](api/resources.md)
- [Kernels](api/kernels.md)
- [Math](api/math.md)
- [Graphics](api/graphics.md)
- [Windowing](api/windowing.md)
- [Automatic Differentiation](api/autodiff.md)
- [Neural Networks](api/nn.md)
- [Interop and Inspection](api/interop.md)

## Advanced And Internals

These pages are written for users who need to inspect generated output, debug a backend issue, contribute to Feather, or understand why a shader shape is accepted or rejected:

- [FEIR Compiler Pipeline](feir.md): the readable overview of Feather IR and how it reaches EasyGPU.
- [FEIR Binary Format](ir-format.md): the versioned `FEIR` payload and section layout.
- [Native ABI](native-abi.md): how managed Feather talks to the native bridge.
- [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md): what the typed EasyGPU route accepts today.
- [AD Internals And Coverage](ad-implementation-note.md): how AD metadata becomes EasyGPU gradient tape work.
- [Feather.NN Status](nn-status.md): which NN paths are GPU-native and which are intentionally explicit host boundaries.

## Recommended First Hour

1. Build the native bridge and run `samples/HelloBuffer`.
2. Read the [Tutorial](tutorial.md) through the texture section.
3. Run `samples/Mandelbrot` and compare the output with the screenshot above.
4. Run `samples/WindowCompute` if you want interactive output.
5. Read [Automatic Differentiation](autodiff.md) and run `samples/AdLinearRegression` if you care about gradients.
6. Keep [API Reference](api.md) open while writing your first custom kernel.
