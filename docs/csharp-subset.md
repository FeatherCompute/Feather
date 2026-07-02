# C# Shader Subset

Feather kernels and graphics shaders are written in C#, but their entry-point bodies are compiled as GPU shader code. Treat that code as a shader DSL with C# syntax, not as arbitrary .NET code.

The generator analyzes Roslyn syntax and semantic `IOperation` trees, lowers supported constructs into typed FEIR, and the native bridge translates that FEIR into EasyGPU IR. Unsupported source shapes are rejected early with `FE0001`-style diagnostics.

## Shader Type Shape

Generated shader types must be `readonly partial struct` values:

```csharp
[Kernel]
public readonly partial struct AddOne(ReadWriteBuffer<float> values) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        values[i] = values[i] + 1.0f;
    }
}
```

Supported shader interfaces:

- `IKernel1D`
- `IKernel2D`
- `IKernel3D`
- `IVertexShader<TVaryings>`
- `IFragmentShader<TVaryings>`
- `IFragmentShader<TVaryings, TOutput>`

Compute kernels normally use one public `void Execute()` method. You can also mark exactly one compatible method with `[Entry]`.

## Resources

Pass resources through the shader struct constructor:

```csharp
public readonly partial struct MyKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output,
    Uniform<float> scale) : IKernel1D
```

Common resource types:

- `ReadOnlyBuffer<T>`, `WriteOnlyBuffer<T>`, `ReadWriteBuffer<T>`
- `ReadOnlyTexture2D<T>`, `WriteOnlyTexture2D<T>`, `ReadWriteTexture2D<T>`
- `SampledTexture2D<T>` plus `SamplerState`
- `ReadOnlyTexture3D<T>`, `WriteOnlyTexture3D<T>`, `ReadWriteTexture3D<T>`
- `Uniform<T>`

Access modes are enforced. Writing to a `ReadOnlyBuffer<T>` or reading from a `WriteOnlyTexture2D<T>` is a generator error.

Host-owned resource objects such as `GpuBuffer<T>` and `GpuTexture2D<TPixel,TValue>` should stay outside shader code. Pass shader-facing views such as `buffer.AsReadOnly()` or `texture.AsReadWrite()` into the generated struct.

## Built-In IDs

Use marker properties for dispatch indices:

```csharp
int i = ThreadIds.X;
int2 pixel = ThreadIds.XY;
int local = LocalIds.X;
int group = GroupIds.X;
int width = DispatchSize.X;
```

These properties are shader-only. Calling them on the CPU throws.

## Supported Values

Common supported value types:

- `bool`, `int`, `uint`, `float`
- `float2`, `float3`, `float4`
- `int2`, `int3`, `int4`
- `bool2`, `bool3`, `bool4`
- `float2x2`, `float3x3`, `float4x4`
- Supported `[GpuStruct]` values

`byte`, `sbyte`, `short`, and `ushort` may be used in host pixel structs such as `Rgba32`, but they are not general-purpose shader arithmetic or storage types today.

## Control Flow

Supported in the ordinary compute subset:

- `if` / `else`
- `for`
- `while`
- `do`
- `break`
- `continue`
- `return`

Automatic differentiation has a narrower control-flow subset: structured `if/else` and canonical counted `for` loops are supported, while `while`, `do`, `break`, and `continue` are currently rejected by the AD bridge. See [Automatic Differentiation](autodiff.md).

## Math

Use `Feather.Math` vector and matrix types and `ShaderMath` / `Hlsl` helpers:

```csharp
float3 n = ShaderMath.Normalize(input[i].XYZ);
float lighting = ShaderMath.Clamp(ShaderMath.Dot(n, lightDir), 0.0f, 1.0f);
```

Supported helpers include common trigonometry, exponent/log, power/square-root, abs/floor/ceil/round/fract, min/max/clamp/saturate, lerp/mix, dot/cross/length/normalize, and square-matrix operations.

Use `f` suffixes for float literals. A bare `2.0` is a C# `double`, which is not the same shader type as `2.0f`.

## Callable Helpers

Mark one-off helper methods inside a shader with `[Callable]`:

```csharp
[Callable]
private static float Smooth(float x)
{
    return x * x * (3.0f - (2.0f * x));
}
```

For helpers shared across kernels, use a source-available `[ShaderLibrary]` type:

```csharp
[ShaderLibrary]
public static class Sdf
{
    [Callable]
    public static float Sphere(float3 p, float radius)
    {
        return ShaderMath.Length(p) - radius;
    }
}
```

Rules:

- Local callables can be inside the shader struct.
- Shared callables must be static methods on a type marked with `[ShaderLibrary]`.
- Shared callables must have source available to the consuming compilation.
- Recursive callables are not supported.
- Callable parameters and return values must use supported shader types.
- Nested callable patterns are supported for ordinary compute where the generator can lower them, but AD callables are more restricted.

## GPU Structs

Use `[GpuStruct]` for vertex data, varyings, texture pixel records, and structured buffer elements:

```csharp
[GpuStruct]
public readonly partial record struct Vertex(float3 Position, float3 Normal, float2 Uv);
```

Supported members include explicit unmanaged fields and positional record primary-constructor storage. Invalid layouts produce `FE0019` and suppress generated layout source.

Fixed array fields must use Feather fixed-array wrappers such as `GpuArray4<float3>`. Ordinary managed arrays are not shader values.

## Shared Memory, Barriers, Atomics

```csharp
var shared = new SharedMemory<float>(256);
shared[LocalIds.X] = input[ThreadIds.X];
GpuBarrier.Workgroup();
output[ThreadIds.X] = shared[LocalIds.X];
```

Integer atomics are available through `GpuAtomic.Add`, `Sub`, `Min`, `Max`, `And`, `Or`, `Xor`, `Exchange`, and `CompareExchange` for supported buffer/shared-memory l-values.

## Not Supported In Shader Code

These are regular .NET features, but not GPU shader features:

- Reference type allocation with `new`.
- Managed arrays, `List<T>`, dictionaries, LINQ, spans inside shader entry bodies.
- `try`, `catch`, `throw`.
- `async` / `await`.
- Reflection and dynamic dispatch.
- Virtual or interface calls.
- Delegates and lambdas inside shader code.
- Unsupported generic methods.
- Arbitrary BCL calls.

Move host-side setup outside the kernel and pass data through buffers, textures, uniforms, and GPU structs.

## How To Unblock Unsupported Code

| If you wrote | Prefer |
| --- | --- |
| `List<T>` or managed arrays inside `Execute` | `GpuBuffer<T>` or `GpuArrayN<T>` in a `[GpuStruct]`. |
| `MathF.SomeCall` that does not lower | `ShaderMath.SomeCall` or `Hlsl.SomeCall` if supported. |
| A helper method without attributes | A static `[Callable]` method inside the shader struct. |
| Object construction for host data | Build host objects before dispatch and pass resources/uniforms. |
| A complex C# abstraction | Make the shader data flow explicit with buffers, textures, uniforms, structs. |
| A top-level local referenced by shader code | Move it to a static const/static readonly field on a named type. |

## Diagnostics

Unsupported code is reported as `FE0001`-style diagnostics. The most common are:

- `FE0001`: shader type is not a `readonly partial struct`.
- `FE0004`: constructor parameter type is not a supported shader resource/value.
- `FE0008`: method call is not supported in shader code.
- `FE0016`: resource access violates read/write mode.
- `FE0019`: GPU struct layout is not supported.
- `FE0027`: typed shader lowering rejected the construct.

See [Diagnostics](diagnostics.md) for the full catalog.

## Related Reference

- [API: Kernels](api/kernels.md)
- [API: Resources](api/resources.md)
- [API: Math](api/math.md)
- [FEIR Compiler Pipeline](feir.md)
