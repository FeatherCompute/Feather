#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

detect_rid() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin) os="osx" ;;
        Linux) os="linux" ;;
        MINGW*|MSYS*|CYGWIN*) os="win" ;;
        *) echo "Unsupported OS for RID detection: $os" >&2; return 1 ;;
    esac

    case "$arch" in
        arm64|aarch64) arch="arm64" ;;
        x86_64|amd64) arch="x64" ;;
        *) echo "Unsupported architecture for RID detection: $arch" >&2; return 1 ;;
    esac

    printf '%s-%s\n' "$os" "$arch"
}

RID="${FEATHER_RUNTIME_IDENTIFIER:-$(detect_rid)}"
case "$RID" in
    win-*) native_library="feather.dll" ;;
    osx-*) native_library="libfeather.dylib" ;;
    linux-*) native_library="libfeather.so" ;;
    *) echo "Unsupported runtime identifier: $RID" >&2; exit 1 ;;
esac

staged_native="$ROOT/artifacts/native-assets/runtimes/$RID/native/$native_library"
if [[ -z "${FEATHER_NATIVE_LIBRARY:-}" && -f "$staged_native" ]]; then
    export FEATHER_NATIVE_LIBRARY="$staged_native"
fi

dotnet test "$ROOT/tests/Feather.Native.Tests/Feather.Native.Tests.csproj" -v minimal
dotnet test "$ROOT/tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj" -v minimal
dotnet test "$ROOT/tests/Feather.Tests/Feather.Tests.csproj" -v minimal

if [[ "${FEATHER_RUN_GPU_TESTS:-0}" == "1" ]]; then
    dotnet test "$ROOT/tests/Feather.Gpu.Tests/Feather.Gpu.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.AD.Tests/Feather.AD.Tests.csproj" -v minimal
    dotnet test "$ROOT/tests/Feather.NN.Tests/Feather.NN.Tests.csproj" -v minimal
else
    echo "Skipping GPU/native integration tests. Set FEATHER_RUN_GPU_TESTS=1 to include them."
fi
