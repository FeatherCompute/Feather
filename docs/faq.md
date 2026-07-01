# FAQ

## What is Feather?

Feather is a C# front end for EasyGPU. You write generated C# kernel structs, Feather lowers them into FEIR with a Roslyn source generator, and the native bridge sends the typed module to EasyGPU.

## Is Feather production-ready?

Not yet. The compute surface is the most usable part today. Graphics, automatic differentiation, and NN helpers are preview features and should be qualified carefully on your target machines. See [Support Status](support-status.md).

## How do I install it?

Install the preview package:

```bash
dotnet add package FeatherCompute --prerelease
```

The NuGet package ID is `FeatherCompute`; the C# namespaces remain `Feather` and
its subnamespaces. When working from source, reference `src/Feather/Feather.csproj`
and, for generated kernels or shaders, reference `src/Feather.Generators/Feather.Generators.csproj`
as an analyzer. See [Getting Started](getting-started.md).

## Why do I get `DllNotFoundException` for `feather`?

The managed package could not find the native library. Build it and make sure it is discoverable:

```bash
./eng/build-native.sh
./eng/stage-native-assets.sh
```

Or set:

```bash
export FEATHER_NATIVE_LIBRARY=/absolute/path/to/libfeather.dylib
```

Use `feather.dll` on Windows and `libfeather.so` on Linux.

## Do I need to know GLSL, HLSL, Vulkan, or OpenGL?

For basic Feather compute, no. You still need to understand GPU-style parallel execution, memory movement, and the supported C# shader subset. Backend setup depends on EasyGPU and your platform.

## Why must kernels be `readonly partial struct`?

The source generator needs a stable value type shape and emits additional partial members that implement generated binding and IR contracts.

## What is FEIR?

FEIR is Feather's serialized intermediate representation. It carries resources, typed statements, thread-group sizes, AD metadata, and graphics stage information from generated C# to the native EasyGPU bridge. Start with [FEIR Compiler Pipeline](feir.md).

## Can I use normal C# collections inside a kernel?

No. Kernel bodies execute as shader code. Use `GpuBuffer<T>`, texture views, `Uniform<T>`, `SharedMemory<T>`, scalar/vector values, and `[Callable]` helper methods in the supported subset.

## Can kernels allocate objects or throw exceptions?

No. Allocations, exceptions, async/await, reflection, and virtual/interface calls are not part of the GPU subset. The generator reports diagnostics for unsupported constructs.

## Why did my kernel read past the end?

By default `[Kernel]` enables logical bounds checking for lanes created by workgroup rounding. If you set `[Kernel(BoundsCheck = false)]`, your code must guard all resource accesses manually.

## What does `TypedEasyGpu` mean?

It means the generated typed FEIR was accepted by the native bridge and dispatched through EasyGPU. It is the expected path for supported compute kernels.

## Can I use Feather.NN like PyTorch?

No. Feather.NN is a small GPU-native helper layer for experiments and samples. It has tensors, modules, losses, optimizers, and a few trainers, but it is not a full dynamic autograd framework. Training is explicit through generated AD kernels or built-in trainers.

## How do I debug generated shader code?

Use:

```csharp
string glsl = Feather.Interop.ShaderInspection.GetGLSL<MyKernel>();
string optimized = Feather.Interop.ShaderInspection.GetOptimizedGLSL<MyKernel>();
```

For AD kernels, `GpuADKernel<T>.GetBackwardGLSL()` returns the merged backward shader after a successful backward run.

## Where are the examples?

Read [Examples](examples.md). Start with `HelloBuffer`, then `Mandelbrot`, `WindowCompute`, `WindowGraphicsTriangle`, and `AdLinearRegression`.
