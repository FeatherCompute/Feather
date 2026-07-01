#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${FEATHER_PACKAGE_OUTPUT:-$ROOT/artifacts/packages}"

"$ROOT/eng/build-native.sh"
"$ROOT/eng/stage-native-assets.sh"

dotnet pack "$ROOT/src/Feather.Generators/Feather.Generators.csproj" -c Release -o "$OUTPUT"
dotnet pack "$ROOT/src/Feather.Native/Feather.Native.csproj" -c Release -o "$OUTPUT"
dotnet pack "$ROOT/src/Feather.NativeAssets/Feather.NativeAssets.csproj" -c Release -o "$OUTPUT"
dotnet pack "$ROOT/src/Feather/Feather.csproj" -c Release -o "$OUTPUT"
