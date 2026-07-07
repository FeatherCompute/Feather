# Diagnostics

Feather generator diagnostics use `FE0001`-style ids. They mean the source generator rejected C# shader code before native dispatch. Runtime failures from the native bridge throw managed exceptions such as `FeatherNativeException`.

When debugging, first decide which layer rejected the program:

| Layer | Evidence | Where to look |
| --- | --- | --- |
| C# compiler | Ordinary C# compile errors | Your project source. |
| Feather generator | `FE0001`-style diagnostics | This page and [C# Shader Subset](csharp-subset.md). |
| Native FEIR bridge | `FeatherNativeException` or `DispatchPath.Rejected` | [FEIR](feir.md), [Typed IR Matrix](typed-ir-compute-support-matrix.md), [Native ABI](native-abi.md). |
| Backend execution | Backend-specific native error | Capabilities, support status, generated GLSL. |

## Common Fixes

### `FE0001`: Shader type shape

Shader types must be `readonly partial struct`.

```csharp
[Kernel]
public readonly partial struct MyKernel : IKernel1D
{
    public void Execute() { }
}
```

### `FE0002`: Missing shader interface

Use one supported interface: `IKernel1D`, `IKernel2D`, `IKernel3D`, `IVertexShader<T>`, or `IFragmentShader<T>`.

### `FE0003`: Invalid entry point

Declare exactly one compatible entry method. Most compute kernels use:

```csharp
public void Execute()
```

Alternatively, mark one method with `[Entry]`.

### `FE0004` / `FE0005`: Unsupported captured type

Constructor parameters and captured fields must be shader resources or supported unmanaged shader values. Move host-only objects outside the shader and pass data through `GpuBuffer<T>`, textures, or `Uniform<T>`.

### `FE0006` / `FE0007` / `FE0008`: Unsupported statement, expression, or call

The code is outside Feather's shader subset. Typical causes include LINQ, managed collections, unsupported BCL calls, reference allocation, or calling a helper method without `[Callable]`.

For shared helper functions:

- Put one-off helpers inside the shader struct and mark them `[Callable]`.
- Put reusable helpers on a source-available `[ShaderLibrary]` type and mark each imported method `[Callable]`.
- Make `[ShaderLibrary]` methods `static`; compiled binary-only helpers cannot be imported because the generator cannot see their method bodies.
- Generic helper methods must be monomorphizable from concrete GPU value types. Interface constraints implemented by `[GpuStruct]` values are supported; runtime interface dispatch is not.

### `FE0010`: Unsupported generic usage

The generic method or type parameter could not be resolved to a concrete shader value type at the call site. Use a concrete helper, or use a generic `[Callable]` whose type parameters are constrained by interfaces implemented by `[GpuStruct]` types and are called with concrete GPU value arguments.

### `FE0014`: Unsupported virtual/interface call

Runtime virtual dispatch and interface-typed shader values are not supported. For interface-style code, move the call into a generic callable:

```csharp
[Callable]
public static float Eval<TShape>(TShape shape, float3 p)
    where TShape : IShape
{
    return shape.Sdf(p);
}
```

Each concrete call is emitted as its own shader callable.

### `FE0016`: Resource access mode violation

The shader is writing to a read-only resource or reading from a write-only resource. Use the correct view:

```csharp
input.AsReadOnly();
output.AsReadWrite();
```

### `FE0019`: GPU struct layout

`[GpuStruct]` fields must have supported unmanaged GPU layout. Prefer partial record structs for compact immutable value shapes:

```csharp
[GpuStruct]
public readonly partial record struct Rgba32(byte R, byte G, byte B, byte A);
```

### `FE0021` / `FE0022`: AD marker problem

AD markers must be inside a generated `[AutoDiff]` 1D kernel. Parameters must trace to captured buffer elements, and the loss must be one scalar `float`.

### `FE0027`: Typed shader lowering failed

The source parsed as a shader shape but the typed IR lowerer intentionally rejected a construct. Common causes are unsupported texture formats, unsupported l-values, unsupported call targets, or unsupported AD/control-flow shapes.

Useful checks:

```csharp
Console.WriteLine(ShaderInspection.GetIR<MyKernel>());
Console.WriteLine(ShaderInspection.GetGLSL<MyKernel>());
Console.WriteLine(GPU.DispatchAndGetPath(new MyKernel(...), count));
```

## Diagnostic Catalog

| Id | Meaning |
| --- | --- |
| `FE0001` | Shader type must be a readonly partial struct. |
| `FE0002` | Shader type must implement a supported shader interface. |
| `FE0003` | Shader entry point is invalid. |
| `FE0004` | Shader constructor parameter type is not supported. |
| `FE0005` | Captured field type is not supported. |
| `FE0006` | Unsupported statement in shader body. |
| `FE0007` | Unsupported expression in shader body. |
| `FE0008` | Unsupported method call in shader body. |
| `FE0009` | Unsupported control flow. |
| `FE0010` | Unsupported generic usage. |
| `FE0011` | Unsupported allocation in shader body. |
| `FE0012` | Unsupported exception handling in shader body. |
| `FE0013` | Unsupported async/await in shader body. |
| `FE0014` | Unsupported virtual/interface call in shader body. |
| `FE0015` | Recursive shader function is not supported. |
| `FE0016` | Resource access violates declared access mode. |
| `FE0017` | Buffer index type must be int-compatible. |
| `FE0018` | Texture index type must be int2 or int3. |
| `FE0019` | Struct layout is not std430-compatible. |
| `FE0020` | Matrix layout requires explicit Feather matrix type. |
| `FE0021` | Automatic differentiation marker is not supported here. |
| `FE0022` | Automatic differentiation source is not supported. |
| `FE0023` | Graphics varying type is unsupported. |
| `FE0024` | Fragment shader output type is unsupported. |
| `FE0025` | Thread-group size is invalid for the kernel dimension. |
| `FE0026` | Elementwise expression intrinsic is unsupported. |
| `FE0027` | Typed shader IR lowering failed. |
| `FE0028` | Top-level local cannot be referenced from shader code. |

## Runtime Errors

If source generation succeeds but native dispatch fails, Feather throws a managed exception such as `FeatherNativeException`. Inspect:

- The exception message from the native bridge.
- `ShaderInspection.GetGLSL<TKernel>()`.
- `GPU.DispatchAndGetPath(...)`.
- `GpuADKernel<T>.GetBackwardGLSL()` after a successful AD backward build.

## AD-Specific Failures

AD failures should be explicit. If `Backward` throws:

- Ensure the kernel has `[AutoDiff]`.
- Ensure the kernel implements `IKernel1D`.
- Ensure at least one `AD.Parameter(...)` and one scalar `AD.Loss(...)` are reached.
- Avoid AD-rejected control flow: `while`, `do`, `break`, and `continue`.
- Check gradient aliases if optimizer handoff cannot find a name.

## Related Reference

- [API: Interop and Inspection](api/interop.md)
- [Automatic Differentiation](autodiff.md)
- [FEIR Compiler Pipeline](feir.md)
