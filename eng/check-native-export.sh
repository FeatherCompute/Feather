#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <path-to-native-library>" >&2
    exit 2
fi

library="$1"
if [[ ! -f "$library" ]]; then
    echo "Native library does not exist: $library" >&2
    exit 1
fi

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

cat > "$workdir/FeatherNativeExportCheck.csproj" <<'EOF'
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net10.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
  </PropertyGroup>
</Project>
EOF

cat > "$workdir/Program.cs" <<'EOF'
using System.Runtime.InteropServices;

if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: FeatherNativeExportCheck <path-to-native-library>");
    return 2;
}

var libraryPath = Path.GetFullPath(args[0]);
var handle = NativeLibrary.Load(libraryPath);
try
{
    _ = NativeLibrary.GetExport(handle, "fe_runtime_flush_caches");
    _ = NativeLibrary.GetExport(handle, "fe_runtime_shutdown");

    var export = NativeLibrary.GetExport(handle, "fe_ir_bridge_contract_version");
    var contractVersion = Marshal.GetDelegateForFunctionPointer<FeIrBridgeContractVersion>(export)();
    if (contractVersion != 1)
    {
        Console.Error.WriteLine($"Unexpected Feather native contract version: {contractVersion}");
        return 1;
    }

    Console.WriteLine($"Feather native contract export OK: {libraryPath}");
    return 0;
}
finally
{
    NativeLibrary.Free(handle);
}

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate uint FeIrBridgeContractVersion();
EOF

dotnet run --project "$workdir/FeatherNativeExportCheck.csproj" -- "$library"
