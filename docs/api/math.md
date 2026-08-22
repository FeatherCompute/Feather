# API Reference: Math

## Purpose

`Feather.Math` provides shader-friendly scalar, vector, matrix, swizzle, and helper APIs that can run both in ordinary C# and inside generated GPU code when supported by the lowerer.

## Vector Types

Common vector types:

- `float2`, `float3`, `float4`
- `int2`, `int3`, `int4`
- `bool2`, `bool3`, `bool4`

Typical usage:

```csharp
float3 n = ShaderMath.Normalize(normal);
float lighting = ShaderMath.Clamp(ShaderMath.Dot(n, lightDir), 0.0f, 1.0f);
float4 color = new float4(lighting, lighting, lighting, 1.0f);
```

## Matrix Types

Common matrix types:

- `float2x2`
- `float3x3`
- `float4x4`

Use explicit Feather matrix types in GPU structs and uniforms.

## Swizzles

Vector swizzles expose HLSL-style component combinations:

```csharp
float2 xy = position.XY;
float3 rgb = color.RGB;
float4 rgba = new float4(rgb, 1.0f);
```

Swizzles are generated as strongly typed properties.

## `ShaderMath`

Important helpers:

| Group | Examples |
| --- | --- |
| Vector algebra | `Dot`, `Cross`, `Length`, `LengthSquared`, `Normalize` |
| Trigonometry | `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh` |
| Exponential | `Exp`, `Log`, `Pow`, `Sqrt`, `InverseSqrt` |
| Value ops | `Abs`, `Floor`, `Ceil`, `Round`, `Fract` |
| Bounds/interpolation | `Min`, `Max`, `Clamp`, `Saturate`, `Lerp`, `Mix`, `Smoothstep` |
| Matrix ops | Matrix multiplication and transform helpers where defined |

Use `f` suffixes for float literals in shader code.

### Component-wise vector overloads

The scalar-style functions below also accept `float2`, `float3`, and `float4` and return the same vector width. Each component is evaluated independently on the CPU and in generated GPU code:

- `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh`
- `Exp`, `Log`, `Sqrt`, `InverseSqrt`
- `Abs`, `Floor`, `Ceil`, `Round`, `Fract`, `Saturate`

`Pow` accepts matching base and exponent vectors. `Min` and `Max` accept either a matching vector or a scalar second operand. `Clamp` accepts matching vector bounds or two scalar bounds. `Lerp` and `Mix` accept either a scalar factor or a matching factor vector. `Smoothstep` accepts either three matching vectors or two scalar edges and a vector value. `Reflect` is available for `float2`, `float3`, and `float4`.

The integer bounds helpers follow the same shape rules: `Min`, `Max`, and `Clamp` accept `int2`, `int3`, and `int4`, with either matching vector operands/bounds or scalar operands/bounds where applicable.

```csharp
float3 phase = new float3(time, time + uv.X, time + uv.Y);
float3 wave = ShaderMath.Cos(phase);
float3 shaped = ShaderMath.Smoothstep(0.0f, 1.0f, wave * 0.5f + new float3(0.5f));
```

## `Hlsl`

`Hlsl` provides alias names for users who prefer HLSL-style spelling. Existing scalar aliases expose the same matching vector widths where the corresponding `ShaderMath` function is component-wise. Calls lower by Roslyn symbol identity, not by source text, so supported aliases are stable even if imported through `using static`.

## Host Vs Shader

Most math helpers have CPU implementations so you can use them in host setup code. Inside generated shaders, only the supported subset lowers to FEIR/EasyGPU. Unsupported calls are generator diagnostics.

## Lifetime And Errors

Math types are value types and do not own native resources. Unsupported math calls in shader code are generator diagnostics; invalid CPU-only operations such as inverting a singular matrix can still throw normal managed exceptions when run on the host.

## Samples And Tests

- `samples/Mandelbrot`
- `samples/JuliaSet`
- `samples/SdfRenderer`
- `samples/SponzaRenderer`
- `tests/Feather.Tests/MathSurfaceTests.cs`
- `tests/Feather.Integration.Tests/ShaderDslCoverageTests.cs`
