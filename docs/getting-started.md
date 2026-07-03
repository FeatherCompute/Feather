# Getting Started

This guide takes you from a fresh checkout to a working generated GPU kernel.

## Prerequisites

- .NET SDK `10.0.301` or a compatible SDK feature band.
- CMake 3.20+.
- A C++20 compiler.
- A GPU and driver supported by the selected EasyGPU backend.
- Vulkan SDK when building the Vulkan backend.
- Linux window samples need X11 development libraries.

Build the native Feather runtime locally when working from source. Published
packages carry native assets under NuGet's `runtimes/<rid>/native` layout; the
source tree does not keep generated native binaries checked in.

## Install From NuGet

Preview packages are published under the `FeatherCompute` package ID. The
package name is different from the C# namespace: application code still imports
`Feather`, `Feather.Math`, `Feather.Resources`, and related subnamespaces.

```bash
dotnet add package FeatherCompute --prerelease
```

The main package includes the runtime API, the source generator, the native
binding layer, and RID-specific native assets for the RIDs included in the
release.

## Build Feather

From the repository root:

```bash
git submodule update --init --recursive

./eng/build-native.sh
./eng/stage-native-assets.sh

dotnet build Feather.slnx
```

Useful variants:

```bash
cmake -S native -B native/build-opengl -DEASYGPU_BACKEND=OpenGL
cmake -S native -B native/build-headless -DFEATHER_BUILD_WINDOW=OFF
```

If you want managed code to load a specific native library, set:

```bash
export FEATHER_NATIVE_LIBRARY=/absolute/path/to/libfeather.dylib
```

Use the platform-appropriate filename on Windows or Linux.

## Run The First Sample

```bash
dotnet run --project samples/HelloBuffer/HelloBuffer.csproj
```

Expected output includes:

- The active backend and max workgroup size.
- `EasyGPU GLSL bridge: OK`.
- `Dispatch path: TypedEasyGpu`.
- `PASS`.

`TypedEasyGpu` matters because it proves the kernel went through the typed FEIR -> EasyGPU module route rather than an old compatibility fallback.

## Write A Kernel

Create or open a C# project that references the package:

```xml
<ItemGroup>
  <PackageReference Include="FeatherCompute" Version="0.1.0-preview.16" />
</ItemGroup>
```

When working from a source checkout instead of NuGet, reference both the runtime
and generator projects:

```xml
<ItemGroup>
  <ProjectReference Include="..\..\src\Feather\Feather.csproj" />
  <ProjectReference Include="..\..\src\Feather.Generators\Feather.Generators.csproj"
                    OutputItemType="Analyzer"
                    ReferenceOutputAssembly="false" />
</ItemGroup>
```

Then write a generated kernel:

```csharp
using Feather;
using Feather.Math;
using Feather.Resources;

float[] input = [1, 2, 3, 4, 5, 6, 7, 8];

using var src = GPU.CreateBuffer<float>(input, BufferAccess.ReadOnly);
using var dst = GPU.CreateBuffer<float>(input.Length, BufferAccess.ReadWrite);

GPU.Dispatch(new DoubleKernel(src.AsReadOnly(), dst.AsReadWrite()), input.Length);

Console.WriteLine(string.Join(", ", dst.ToArray()));

[Kernel]
[ThreadGroupSize(DefaultThreadGroupSizes.X)]
public readonly partial struct DoubleKernel(
    ReadOnlyBuffer<float> input,
    ReadWriteBuffer<float> output) : IKernel1D
{
    public void Execute()
    {
        int i = ThreadIds.X;
        output[i] = input[i] * 2.0f;
    }
}
```

The important pieces:

- `[Kernel]` marks the struct for source generation.
- `partial` lets the generator add generated contracts.
- `IKernel1D` tells Feather the dispatch shape.
- Constructor parameters are shader resources.
- `ThreadIds.X` maps to the global GPU invocation index.
- `ReadOnlyBuffer<T>` and `ReadWriteBuffer<T>` are shader-facing views over host-owned `GpuBuffer<T>` objects.

## Inspect Generated Output

You can inspect the generated resource table, FEIR, and GLSL:

```csharp
using Feather.Interop;

Console.WriteLine(ShaderInspection.GetIR<DoubleKernel>());
Console.WriteLine(ShaderInspection.GetGLSL<DoubleKernel>());
Console.WriteLine(ShaderInspection.GetOptimizedGLSL<DoubleKernel>());
```

Use this when a kernel compiles but behaves differently than expected. For a higher-level explanation of the compiler path, read [FEIR Compiler Pipeline](feir.md).

## Common First Errors

| Symptom | Likely fix |
| --- | --- |
| `DllNotFoundException` | Build `native/`, copy native assets, or set `FEATHER_NATIVE_LIBRARY`. |
| `FE0001` / `FE0002` | Ensure the shader type is a `readonly partial struct` implementing `IKernel1D`, `IKernel2D`, `IKernel3D`, `IVertexShader<T>`, or `IFragmentShader<T>`. |
| Unsupported call diagnostic | Move unsupported host code outside `Execute`, or replace it with `ShaderMath`/`Hlsl` helpers. |
| Unexpected fallback/rejection | Check `DispatchPath`, then inspect FEIR/GLSL and the [Typed IR support matrix](typed-ir-compute-support-matrix.md). |

## Next Steps

- [Tutorial](tutorial.md): learn the compute model in detail.
- [Examples](examples.md): pick the next sample to run.
- [C# Shader Subset](csharp-subset.md): learn exactly what is supported in GPU code.
- [API Reference](api.md): look up runtime and shader-facing types.
