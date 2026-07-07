# Shader Libraries

Shader libraries let you share GPU helper functions across kernels and graphics shaders. They are intended for code such as BRDFs, color transforms, SDF primitives, sampling functions, hash/noise helpers, and small math utilities that should stay in C# instead of being copied into every kernel struct.

## Basic Pattern

Put reusable shader functions on a source-available type marked with `[ShaderLibrary]`, and mark each imported function with `[Callable]`:

```csharp
using Feather;
using Feather.Math;

[ShaderLibrary]
public static class Pbr
{
    [Callable]
    public static float3 Lambert(float3 albedo, float3 normal, float3 lightDir)
    {
        float nDotL = ShaderMath.Max(ShaderMath.Dot(ShaderMath.Normalize(normal), lightDir), 0.0f);
        return albedo * nDotL;
    }

    [Callable]
    public static float3 FresnelSchlick(float cosTheta, float3 f0)
    {
        float factor = ShaderMath.Pow(1.0f - ShaderMath.Clamp(cosTheta, 0.0f, 1.0f), 5.0f);
        return f0 + ((new float3(1.0f) - f0) * factor);
    }
}
```

Call the library from a generated shader exactly like ordinary C#:

```csharp
[Kernel]
[ThreadGroupSize(8, 8, 1)]
public readonly partial struct ShadeKernel(
    ReadOnlyBuffer<float3> albedo,
    ReadOnlyBuffer<float3> normal,
    ReadWriteBuffer<float3> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        float3 lightDir = ShaderMath.Normalize(new float3(0.4f, 0.7f, 0.2f));
        output[i] = Pbr.Lambert(albedo[i], normal[i], lightDir);
    }
}
```

The generator imports only the reachable `[Callable]` methods into each shader module. If two kernels call different overloads or different helper chains, each generated module receives only the functions it needs.

## Local Callables Vs Libraries

Use kernel-local `[Callable]` methods for helpers that are specific to one shader:

```csharp
[Callable]
private static float Smooth(float x) => x * x * (3.0f - (2.0f * x));
```

Use `[ShaderLibrary]` when the helper belongs to a reusable domain:

```csharp
[ShaderLibrary]
public static class ColorSpace
{
    [Callable]
    public static float3 LinearToSrgb(float3 value)
    {
        return new float3(
            ShaderMath.Pow(value.X, 1.0f / 2.2f),
            ShaderMath.Pow(value.Y, 1.0f / 2.2f),
            ShaderMath.Pow(value.Z, 1.0f / 2.2f));
    }
}
```

## Rules

- Library methods must be `static`.
- Library methods must be marked with `[Callable]`.
- The containing type must be marked with `[ShaderLibrary]`.
- The method body must be source-available to the generator. Methods from a normal compiled DLL or NuGet package cannot be imported unless their source is also compiled into the consuming project.
- Generic methods are supported only when they can be monomorphized from concrete GPU value type arguments. The main supported pattern is an interface constraint implemented by `[GpuStruct]` types.
- Recursion, runtime virtual/interface dispatch, exceptions, async/await, object allocation, managed collections, and LINQ are not supported.
- Parameters and return values must use supported shader types: scalars, Feather vectors/matrices, supported GPU structs, and supported shader resource wrappers where the backend allows them.
- Overloads are supported; Feather binds by Roslyn symbol identity and emits a generated mangled function name.

## Generic Interface Pattern

Generic shader-library callables can express a small object-style pattern without runtime dispatch:

```csharp
public interface ISdfShape
{
    float Sdf(float3 p);
}

[GpuStruct]
public readonly partial record struct Sphere(float Radius) : ISdfShape
{
    [Callable]
    public float Sdf(float3 p) => ShaderMath.Length(p) - Radius;
}

[ShaderLibrary]
public static class SdfOps
{
    [Callable]
    public static float Eval<TShape>(TShape shape, float3 p)
        where TShape : ISdfShape
    {
        return shape.Sdf(p);
    }
}
```

When a kernel calls `SdfOps.Eval(sphere, p)`, Feather emits a concrete `Eval<Sphere>` callable and resolves `shape.Sdf(p)` to `Sphere.Sdf`. A second concrete type gets a second generated callable. This is closer to C++ template or Rust-style monomorphization than to C# runtime interface dispatch.

The interface is used only as a compile-time constraint. Do not store interface-typed shader locals such as `ISdfShape shape = ...`; that would require runtime dispatch and is rejected.

## Source Availability

Feather is a source-generator-based shader compiler. It does not decompile DLLs. This works:

```csharp
// Pbr.cs is included in the same project as the kernel.
[ShaderLibrary]
public static class Pbr
{
    [Callable]
    public static float3 Lambert(...) { ... }
}
```

This does not work by itself:

```csharp
// Pbr is only referenced from a compiled binary package.
output[i] = Pbr.Lambert(...);
```

For shared libraries today, use a source package, shared project, linked source file, or project reference that lets the consuming compilation see the method bodies.

## Diagnostics

Common failures:

| Symptom | Fix |
| --- | --- |
| `FE0008` on a helper method | Add `[Callable]`, put the method inside the shader struct, or put it on a `[ShaderLibrary]` type. |
| `FE0008` says a library method must be static | Make the `[ShaderLibrary]` callable static. |
| `FE0008` says the method must be source-available | Include the library source in the consuming project. |
| `FE0010` | Use a concrete callable, or make the generic callable monomorphizable from supported GPU value type arguments. |
| `FE0015` | Break direct or mutual recursion between callables. |
| `FE0027` | Inspect the typed IR lowering message; the callable body contains a construct outside the shader subset. |

## Related Pages

- [C# Shader Subset](csharp-subset.md)
- [API Reference: Kernels](api/kernels.md)
- [Diagnostics](diagnostics.md)
- [FEIR Compiler Pipeline](feir.md)
