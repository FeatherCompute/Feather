# API Reference: Resources

## Purpose

Resource APIs own GPU memory on the host and expose lightweight shader-facing views to generated kernels and graphics shaders.

## Buffers

```csharp
using var buffer = GPU.CreateBuffer<float>(1024, BufferAccess.ReadWrite);
using var uploaded = GPU.CreateBuffer<float>(data, BufferAccess.ReadOnly);

buffer.Upload(values);
float[] values = buffer.ToArray();

ReadOnlyBuffer<float> ro = uploaded.AsReadOnly();
WriteOnlyBuffer<float> wo = buffer.AsWriteOnly();
ReadWriteBuffer<float> rw = buffer.AsReadWrite();
```

### `GpuBuffer<T>`

| API | Purpose |
| --- | --- |
| `Length` | Logical element count. |
| `SizeInBytes` | GPU-side byte size. |
| `Access` | Declared buffer access mode. |
| `Create(context, count, access)` | Allocates a buffer. |
| `Create(context, data, access)` | Allocates and uploads initial data. |
| `AsReadOnly()` | Creates `ReadOnlyBuffer<T>`. |
| `AsWriteOnly()` | Creates `WriteOnlyBuffer<T>`. |
| `AsReadWrite()` | Creates `ReadWriteBuffer<T>`. |
| `Upload(data)` / `Upload(start, data)` | Uploads CPU values. |
| `Read(destination)` | Downloads values into a span. |
| `ToArray()` | Downloads into a new array. |
| `Dispose()` | Releases the native buffer. |

Shader views expose `Length` and indexers. `ReadWriteBuffer<T>` returns `ref T` for supported l-value operations such as atomics.

## Buffer Layout

Feather follows EasyGPU/std430-style buffer strides. Important examples:

| Type | Buffer stride |
| --- | --- |
| `float`, `int`, `uint` | 4 bytes |
| `float2`, `int2` | 8 bytes |
| `float3`, `float4`, `int3`, `int4` | 16 bytes |

A `float3` has a 12-byte CPU payload but a 16-byte GPU buffer slot.

## 2D Textures

```csharp
using var tex = GPU.CreateTexture2D<Rgba32, Rgba32>(
    width,
    height,
    PixelFormat.Rgba8,
    TextureAccess.ReadWrite);

tex.Upload(pixels);
tex.Read(pixels);
tex.GenerateMipmaps();
tex.Save("output.tga");
```

### `GpuTexture2D<TPixel,TValue>`

| API | Purpose |
| --- | --- |
| `Width`, `Height`, `Size` | Base level dimensions. |
| `MipLevels` | Allocated mip count. |
| `Format` | Pixel format. |
| `Access` | Texture access mode. |
| `AsReadOnly()` | Creates `ReadOnlyTexture2D<TValue>`. |
| `AsWriteOnly()` | Creates `WriteOnlyTexture2D<TValue>`. |
| `AsReadWrite()` | Creates `ReadWriteTexture2D<TValue>`. |
| `AsReadWriteNormalized()` | Creates normalized read-write view. |
| `AsSampled()` | Creates `SampledTexture2D<TValue>`. |
| `Upload(pixels)` | Uploads base-level pixels. |
| `Read(pixels)` | Downloads base-level pixels. |
| `GenerateMipmaps()` | Generates mipmaps when supported. |
| `Save(path)` | Saves the base level as TGA. |

Shader texture views use `int2` coordinates. Sampled textures provide:

```csharp
T Sample(SamplerState sampler, float2 uv);
T SampleLevel(SamplerState sampler, float2 uv, float lod);
T SampleGrad(SamplerState sampler, float2 uv, float2 ddx, float2 ddy);
```

## 3D Textures

`GpuTexture3D<TPixel,TValue>` mirrors the 2D texture model with `Width`, `Height`, `Depth`, `Size`, access views, upload/read methods, and shader views indexed by `int3`. Use it for volume resources and 3D compute workloads.

## Samplers

```csharp
using var sampler = GPU.CreateSampler(SamplerDesc.LinearRepeat);
```

`SamplerDesc` includes min/mag filters, mip mode, address modes, LOD bias/range, anisotropy, comparison, and border color. Convenience descriptors:

- `SamplerDesc.NearestClamp`
- `SamplerDesc.LinearClamp`
- `SamplerDesc.NearestRepeat`
- `SamplerDesc.LinearRepeat`
- `SamplerDesc.LinearMirroredRepeat`

## Uniforms

```csharp
new Uniform<float>(0.25f)
new Uniform<int2>(new int2(width, height))
new Uniform<float4x4>(mvp)
```

`Uniform<T>.Value` is read inside shader code. Uniforms are best for small per-dispatch values that map to push constants.

## Lifetime And Errors

- Host-owned `GpuBuffer<T>`, `GpuTexture2D<,>`, `GpuTexture3D<,>`, and `SamplerState` are disposable.
- Shader-facing views are lightweight values and do not own native handles.
- Access-mode violations are generator diagnostics.
- Unsupported texture formats are native errors.

## Host Vs Shader

- `GpuBuffer<T>`, `GpuTexture2D<,>`, `GpuTexture3D<,>`, and `SamplerState` are host-owned resource objects.
- `ReadOnlyBuffer<T>`, `ReadWriteTexture2D<T>`, `SampledTexture2D<T>`, and similar view types are shader-facing values passed into generated structs.
- `Uniform<T>.Value` is read in shader code; the `Uniform<T>` value itself is created on the host.

## Samples And Tests

- `samples/HelloBuffer`
- `samples/TextureCopy`
- `samples/ColorFilter`
- `samples/WindowCompute`
- `tests/Feather.Integration.Tests/GeneratedComputeDispatchTests.cs`
- `tests/Feather.Integration.Tests/ShaderDslCoverageTests.cs`
