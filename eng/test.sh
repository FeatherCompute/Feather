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
    win-*) native_library="feather_native.dll"; luisa_backend="vk" ;;
    osx-*) native_library="libfeather.dylib"; luisa_backend="metal" ;;
    linux-*) native_library="libfeather.so"; luisa_backend="vk" ;;
    *) echo "Unsupported runtime identifier: $RID" >&2; exit 1 ;;
esac

# Keep the existing Vulkan parity contract on CI platforms. macOS runs Metal,
# where the local Luisa Vulkan path is unavailable with the pinned LC version.
export FEATHER_LUISA_BACKEND="$luisa_backend"

staged_native="$ROOT/artifacts/native-assets/runtimes/$RID/native/$native_library"
if [[ -z "${FEATHER_NATIVE_LIBRARY:-}" && -f "$staged_native" ]]; then
    export FEATHER_NATIVE_LIBRARY="$staged_native"
fi

test_project() {
    local project="$1"
    shift
    dotnet build "$project" -v minimal

    if [[ -f "$staged_native" ]]; then
        local target_dir
        target_dir="$(dotnet msbuild "$project" -getProperty:TargetDir)"
        mkdir -p "$target_dir"
        cp "$staged_native" "$target_dir/$native_library"
    fi

    dotnet test "$project" --no-build -v minimal "$@"
}

test_project "$ROOT/tests/Feather.Native.Tests/Feather.Native.Tests.csproj"
test_project "$ROOT/tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj"

if [[ "${FEATHER_RUN_GPU_TESTS:-0}" == "1" ]]; then
    test_project "$ROOT/tests/Feather.Tests/Feather.Tests.csproj"
    test_project "$ROOT/tests/Feather.Gpu.Tests/Feather.Gpu.Tests.csproj"
    test_project "$ROOT/tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj"
    if [[ "$luisa_backend" == "metal" ]]; then
        # Existing LuisaBackend tests execute directly and contain four known LC
        # Metal compiler aborts. LuisaBackendMetalTests isolates their expected
        # failures while preserving the remaining parity coverage.
        test_project "$ROOT/tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj" \
            --filter "FullyQualifiedName!~LuisaBackend"
        test_project "$ROOT/tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj" \
            --filter "FullyQualifiedName~LuisaBackendMetalTests"
    else
        test_project "$ROOT/tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj"
    fi
    test_project "$ROOT/tests/Feather.AD.Tests/Feather.AD.Tests.csproj"
    test_project "$ROOT/tests/Feather.NN.Tests/Feather.NN.Tests.csproj"
    test_project "$ROOT/tests/Feather.Blender.RenderHost.Tests/Feather.Blender.RenderHost.Tests.csproj"
else
    test_project "$ROOT/tests/Feather.Tests/Feather.Tests.csproj" --filter "Category!=Gpu"
    test_project "$ROOT/tests/Feather.Blender.RenderHost.Tests/Feather.Blender.RenderHost.Tests.csproj" --filter "Category!=Gpu"
    echo "Skipping GPU/native integration tests. Set FEATHER_RUN_GPU_TESTS=1 to include them."
fi
