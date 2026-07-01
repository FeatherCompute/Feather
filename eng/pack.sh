#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${FEATHER_PACKAGE_OUTPUT:-$ROOT/artifacts/packages}"

pack_project() {
    local project="$1"

    if [[ -n "${FEATHER_PACKAGE_VERSION:-}" ]]; then
        dotnet pack "$project" -c Release -o "$OUTPUT" -p:Version="$FEATHER_PACKAGE_VERSION"
    else
        dotnet pack "$project" -c Release -o "$OUTPUT"
    fi
}

if [[ "${FEATHER_SKIP_NATIVE_BUILD:-}" != "1" ]]; then
    "$ROOT/eng/build-native.sh"
    "$ROOT/eng/stage-native-assets.sh"
fi

dotnet build "$ROOT/src/Feather.Generators/Feather.Generators.csproj" -c Release

pack_project "$ROOT/src/Feather.Generators/Feather.Generators.csproj"
pack_project "$ROOT/src/Feather.Native/Feather.Native.csproj"
pack_project "$ROOT/src/Feather.NativeAssets/Feather.NativeAssets.csproj"
pack_project "$ROOT/src/Feather/Feather.csproj"
