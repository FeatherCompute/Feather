# Tutorial

This tutorial explains Feather's compute model by building up from one thread per element to textures, windows, graphics, and inspection.

## 1. Think In Threads

A Feather compute dispatch launches a logical grid of threads. Each thread runs your `Execute` method once.

```csharp
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct AddOneKernel(ReadWriteBuffer<float> values) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        values[i] = values[i] + 1.0f;
    }
}
```

Host code chooses the logical size:

```csharp
GPU.Dispatch(new AddOneKernel(values.AsReadWrite()), values.Length);
```

With `[Kernel]` bounds checks enabled, Feather adds a guard against extra backend workgroup threads. If you set `[Kernel(BoundsCheck = false)]`, your code is responsible for staying inside the logical dispatch range.

## 2. Buffers And Access Modes

Host code owns `GpuBuffer<T>`. Kernel code receives access-mode views:

```csharp
using var input = GPU.CreateBuffer<float>([1, 2, 3], BufferAccess.ReadOnly);
using var output = GPU.CreateBuffer<float>(3, BufferAccess.ReadWrite);

GPU.Dispatch(new CopyKernel(input.AsReadOnly(), output.AsReadWrite()), 3);
```

```csharp
[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct CopyKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i];
    }
}
```

Use access modes as documentation and as validation:

| View | Shader operation |
| --- | --- |
| `ReadOnlyBuffer<T>` | Read only |
| `WriteOnlyBuffer<T>` | Write only |
| `ReadWriteBuffer<T>` | Read and write |

## 3. Dispatch Dimensions

Feather supports 1D, 2D, and 3D compute kernels:

```csharp
GPU.Dispatch(new Kernel1D(...), count);
GPU.Dispatch(new Kernel2D(...), new int2(width, height));
GPU.Dispatch(new Kernel3D(...), new int3(width, height, depth));
```

Inside the kernel:

```csharp
int x = ThreadIds.X;
int2 xy = ThreadIds.XY;
int3 xyz = ThreadIds.XYZ;
int localX = LocalIds.X;
int groupX = GroupIds.X;
int width = DispatchSize.X;
```

`DefaultThreadGroupSizes.X` and `DefaultThreadGroupSizes.XY` are convenient defaults. Use explicit `[ThreadGroupSize(x, y, z)]` when the algorithm needs a particular local shape.

## 4. Uniforms

`Uniform<T>` stores a small CPU value that Feather uploads as push-constant data when the kernel is bound.

```csharp
GPU.Dispatch(
    new ScaleKernel(input.AsReadOnly(), output.AsReadWrite(), new Uniform<float>(0.25f)),
    input.Length);
```

```csharp
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

Uniforms are best for scalars, vectors, matrices, and small GPU structs that change per dispatch.

## 5. Math And Callables

Use `Feather.Math` types and `ShaderMath`/`Hlsl` helpers inside GPU code. Put one-off helpers in the shader struct:

```csharp
[Callable]
private static float3 Shade(float3 normal, float3 light)
{
    float nDotL = ShaderMath.Max(ShaderMath.Dot(ShaderMath.Normalize(normal), light), 0.0f);
    return new float3(nDotL, nDotL, nDotL);
}
```

Use `[ShaderLibrary]` when the helper should be shared by many kernels:

```csharp
[ShaderLibrary]
public static class Pbr
{
    [Callable]
    public static float3 Lambert(float3 albedo, float3 normal, float3 light)
    {
        float nDotL = ShaderMath.Max(ShaderMath.Dot(ShaderMath.Normalize(normal), light), 0.0f);
        return albedo * nDotL;
    }
}
```

`[Callable]` methods are emitted into the generated shader module. They must stay inside the supported shader subset: no object allocation, no exceptions, no async, no virtual dispatch, and no ordinary .NET collections. Library callables must be static and source-available to the generator. See [Shader Libraries](shader-libraries.md) for the full set of rules.

## 6. Textures

Textures are created through `GPU` and passed to kernels as shader-facing views.

```csharp
using var color = GPU.CreateRenderTexture2D<float4, float4>(
    width,
    height,
    PixelFormat.Rgba32Float);

GPU.Dispatch(new FillTexture(color.AsReadWrite()), color.Size);
```

```csharp
[Kernel]
[ThreadGroupSize(8, 8, 1)]
public readonly partial struct FillTexture(ReadWriteTexture2D<float4> output) : IKernel2D
{
    public void Execute()
    {
        int2 p = ThreadIds.XY;
        float2 uv = new float2((float)p.X / output.Size.X, (float)p.Y / output.Size.Y);
        output[p] = new float4(uv.X, uv.Y, 0.5f, 1.0f);
    }
}
```

Sampled textures use a sampler:

```csharp
using var sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);
float4 color = texture.Sample(sampler, uv);
```

For a complete visual example, run [Mandelbrot](examples.md#mandelbrot), [Julia Set](examples.md#julia-set), or [WindowCompute](examples.md#recommended-order).

## 7. Windows

Window output is explicit: create a `GpuWindow`, render into a `GpuTexture2D`, and present that texture.

```csharp
using var window = GpuWindow.Create(new()
{
    Width = 800,
    Height = 450,
    Title = "Feather Compute Texture"
});
using var presenter = window.CreateTexturePresenter();
using var color = GPU.CreateRenderTexture2D<float4, float4>(
    window.Width,
    window.Height,
    PixelFormat.Rgba32Float);

while (window.IsOpen)
{
    window.PollEvents();
    GPU.Dispatch(new ComputePixels(color.AsReadWrite(), new Uniform<int>(frame)), color.Size);
    presenter.Present(color);
}
```

Read [Windowing](window.md) for event handling and platform notes.

## 8. Graphics

Graphics pipelines are separate from compute kernels. A vertex shader implements `IVertexShader<TVaryings>` and a fragment shader implements `IFragmentShader<TVaryings>`.

```csharp
using var pipeline = GPU.CreateGraphicsPipeline<TriangleVS, TriangleFS, float4>();
pipeline.Draw(new TriangleVS(vertices.AsReadOnly()), new TriangleFS(), color, vertexCount: 3);
```

Graphics render into Feather textures. Present them with a `GpuTexturePresenter`, save them with `GpuTexture2D.Save`, or read them back for tests. Read [Graphics Pipeline](graphics-pipeline.md) for depth, MSAA, MRT, indexed draws, and state objects.

## 9. Automatic Differentiation

AD kernels are generated 1D kernels marked with `[AutoDiff]`. Inside the kernel, mark parameters and a scalar loss:

```csharp
ADMarker.Parameter(w[0]);
ADMarker.Loss(l);
```

Host code drives the backward pass through `GPU.CreateADKernel(...)` or through `Feather.NN.TrainingStep<TKernel>`. Read [Automatic Differentiation](autodiff.md) before writing your first AD kernel.

## 10. Inspection And Debugging

When a kernel does not compile or does not run as expected:

```csharp
string ir = ShaderInspection.GetIR<MyKernel>();
string glsl = ShaderInspection.GetGLSL<MyKernel>();
DispatchPath path = GPU.DispatchAndGetPath(new MyKernel(...), count);
```

Use these checks:

- `DispatchPath.TypedEasyGpu`: the typed EasyGPU path succeeded.
- `DispatchPath.Rejected`: the native bridge rejected the shape.
- Generator diagnostics such as `FE0006`: the C# source shape cannot be lowered.

Next reads:

- [Diagnostics](diagnostics.md)
- [FEIR Compiler Pipeline](feir.md)
- [Typed IR Compute Support Matrix](typed-ir-compute-support-matrix.md)
