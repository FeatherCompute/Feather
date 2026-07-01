# API Reference: Kernels

## Purpose

Kernel APIs define generated compute shader shapes and shader-only runtime markers such as thread IDs, barriers, shared memory, and atomics.

## Kernel Shape

```csharp
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct MyKernel(ReadWriteBuffer<float> data) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        data[i] = data[i] + 1.0f;
    }
}
```

| Interface | Dispatch size | Thread ID helper |
| --- | --- | --- |
| `IKernel1D` | `int` | `ThreadIds.X` |
| `IKernel2D` | `int2` | `ThreadIds.XY` |
| `IKernel3D` | `int3` | `ThreadIds.XYZ` |

## Thread And Group IDs

| API | Meaning |
| --- | --- |
| `ThreadIds.X/Y/Z` | Global invocation components. |
| `ThreadIds.XY/XYZ` | Global invocation vectors. |
| `LocalIds.X/Y/Z` | Local workgroup invocation components. |
| `GroupIds.X/Y/Z` | Workgroup ID components. |
| `DispatchSize.X/Y/Z` | Logical dispatch size. |
| `GroupSize.X/Y/Z` | Declared local group size. |

These properties are shader-only markers and throw on the CPU.

## Dispatch

```csharp
GPU.Dispatch(new Kernel1D(...), count);
GPU.Dispatch(new Kernel2D(...), new int2(width, height));
GPU.Dispatch(new Kernel3D(...), new int3(width, height, depth));

DispatchPath path = GPU.DispatchAndGetPath(new Kernel1D(...), count);
```

`wait` defaults to `true`. Use `DispatchAndGetPath` in samples and tests when you need to prove the native route.

## Callables

```csharp
[Callable]
private static float Smooth(float x)
{
    return x * x * (3.0f - (2.0f * x));
}
```

Callables must use supported shader value types. Overloads bind by generated symbol identity.

## Shared Memory

```csharp
var shared = new SharedMemory<float>(256);
shared[LocalIds.X] = input[ThreadIds.X];
GpuBarrier.Workgroup();
output[ThreadIds.X] = shared[LocalIds.X];
```

`SharedMemory<T>` is shader-only and intended for workgroup-local scratch storage.

## Barriers

| API | Purpose |
| --- | --- |
| `GpuBarrier.Workgroup()` | Workgroup execution/memory barrier. |
| `GpuBarrier.Memory()` | Memory barrier. |
| `GpuBarrier.Full()` | Combined barrier. |

Barriers are shader markers and throw if called on the CPU.

## Atomics

Integer atomics operate on supported l-values:

```csharp
GpuAtomic.Add(ref counters[0], 1);
GpuAtomic.CompareExchange(ref values[i], expected, replacement);
```

| API | Operation |
| --- | --- |
| `Add` | Atomic add. |
| `Sub` | Atomic subtract. |
| `Min` / `Max` | Atomic min/max. |
| `And` / `Or` / `Xor` | Atomic bitwise operations. |
| `Exchange` | Atomic exchange. |
| `CompareExchange` | Atomic compare-exchange. |

## Generated Kernel Objects

Most code uses `GPU.Dispatch`. `GpuKernel` is available for lower-level inspection and dispatch:

| API | Purpose |
| --- | --- |
| `GpuKernel.Create<TKernel>(context)` | Creates a native kernel object. |
| `GpuKernel.Dispatch(...)` | Dispatches with an explicit context/kernel object. |
| `GetGLSL()` | Returns unoptimized GLSL. |
| `GetOptimizedGLSL()` | Returns backend-optimized GLSL inspection text. |
| `LastDispatchPath` | Last native route. |

## Related Docs

- [C# Shader Subset](../csharp-subset.md)
- [Tutorial](../tutorial.md)
- [Diagnostics](../diagnostics.md)

## Host Vs Shader

- `GPU.Dispatch` and `GpuKernel` are host APIs.
- `ThreadIds`, `LocalIds`, `GroupIds`, `DispatchSize`, `GroupSize`, `GpuBarrier`, `GpuAtomic`, and `SharedMemory<T>` are shader-only markers.
- `[Callable]` methods are ordinary C# declarations at compile time, but their bodies must fit the shader subset.

## Lifetime And Errors

- `GpuKernel` is disposable when you create one explicitly.
- Most applications use `GPU.Dispatch`, which creates and releases native kernel state for that dispatch.
- Shader-only marker APIs throw when called on the CPU.
- Unsupported statements/calls produce generator diagnostics before dispatch.

## Samples And Tests

- `samples/HelloBuffer`
- `samples/ParallelReduction`
- `samples/Histogram`
- `tests/Feather.Integration.Tests/GeneratedComputeDispatchTests.cs`
- `tests/Feather.Integration.Tests/ShaderDslCoverageTests.cs`
