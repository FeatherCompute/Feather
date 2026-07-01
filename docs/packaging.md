# Packaging And Native Assets

Feather publishes NuGet packages under the `FeatherCompute*` IDs. The package
name is intentionally different from the public C# namespace: application code
still uses `using Feather;`.

Feather separates managed packages from native runtime assets. EasyGPU is built
behind Feather's C ABI, then packaged as RID-specific `feather` native libraries
inside `FeatherCompute.NativeAssets`.

## Projects

| Project | NuGet package | Role |
| --- | --- | --- |
| `src/Feather` | `FeatherCompute` | Main managed API. Includes the generator analyzer in the package. |
| `src/Feather.Generators` | `FeatherCompute.Generators` | Roslyn analyzer/source generator for kernels and shaders. Published separately for advanced analyzer-only scenarios. |
| `src/Feather.Native` | `FeatherCompute.Native` | P/Invoke declarations and native library resolver. |
| `src/Feather.NativeAssets` | `FeatherCompute.NativeAssets` | RID-specific native runtime assets. |

## NuGet Consumption

Most users only need the main package:

```bash
dotnet add package FeatherCompute --prerelease
```

The main package depends on the native binding and native asset packages, and it
also carries `Feather.Generators.dll` under `analyzers/dotnet/cs` so generated
kernels work without a second package reference.

## Source Consumption

Generated kernels require the generator as an analyzer:

```xml
<ItemGroup>
  <ProjectReference Include="../Feather/src/Feather/Feather.csproj" />
  <ProjectReference Include="../Feather/src/Feather.Generators/Feather.Generators.csproj"
                    OutputItemType="Analyzer"
                    ReferenceOutputAssembly="false" />
</ItemGroup>
```

Window-only projects that do not define generated shader types can reference only `Feather.csproj`.

## Native Library Resolution

The managed resolver looks for `feather` in this order:

- `FEATHER_NATIVE_LIBRARY`, if set.
- The application base directory.
- A `native/` subdirectory under the application base directory.
- The managed assembly directory.
- Ancestor repository folders such as `native/build`.
- `artifacts/native-assets/runtimes/<rid>/native`.
- `runtimes/<rid>/native`.

Override the resolver during development or CI with:

```bash
export FEATHER_NATIVE_LIBRARY=/absolute/path/to/libfeather.dylib
```

Use the platform-specific filename from the table below.

Platform file names:

| OS | Library name |
| --- | --- |
| Windows | `feather.dll` |
| Linux | `libfeather.so` |
| macOS | `libfeather.dylib` |

## Build Native Assets

```bash
./eng/build-native.sh
./eng/stage-native-assets.sh
```

For headless builds:

```bash
cmake -S native -B native/build -DFEATHER_BUILD_WINDOW=OFF
```

## Pack Locally

Pack all managed projects and the staged native asset for the current runtime
identifier:

```bash
./eng/pack.sh
```

`eng/stage-native-assets.sh` copies the native build output to
`artifacts/native-assets/runtimes/<rid>/native`. `Feather.NativeAssets.csproj`
packs files from that staging directory into `FeatherCompute.NativeAssets`. Do
not commit generated native binaries under `src/`.

Supported RID folders declared by the project:

- `win-x64`
- `win-arm64`
- `linux-x64`
- `linux-arm64`
- `osx-arm64`
- `osx-x64`

Build and package each RID on a compatible machine with the matching native binary.

Release CI should run the native build on each supported RID, upload the staged
`artifacts/native-assets/runtimes/<rid>/native` directory, then run a final pack
job that downloads every RID artifact before publishing `FeatherCompute.NativeAssets`.

## Release Workflow

The repository has a manual GitHub Actions workflow named `Release`. It accepts a
version and a `publish` switch:

- `publish=false` builds packages and uploads them as workflow artifacts.
- `publish=true` pushes the packages to nuget.org using the `NUGET_API_KEY`
  repository secret.

The workflow currently builds packaged native assets for:

- `linux-x64`
- `osx-arm64`
- `win-x64`

It validates that `FeatherCompute.NativeAssets` contains all three native files
and that `FeatherCompute` contains the analyzer before uploading or publishing.

For the first preview release, use a prerelease version such as:

```text
0.1.0-preview.1
```

NuGet package versions are immutable once published, so run the workflow once
with `publish=false`, download and inspect the package artifacts, then rerun the
same version with `publish=true`.

## Related Docs

- [Native ABI](native-abi.md)
- [Getting Started](getting-started.md)
- [Support Status](support-status.md)
