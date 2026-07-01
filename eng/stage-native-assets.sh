#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${FEATHER_NATIVE_BUILD_DIR:-$ROOT/native/build}"
STAGING_ROOT="${FEATHER_NATIVE_ASSET_STAGING:-$ROOT/artifacts/native-assets}"

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
    win-*) library="feather.dll" ;;
    osx-*) library="libfeather.dylib" ;;
    linux-*) library="libfeather.so" ;;
    *) echo "Unsupported runtime identifier: $RID" >&2; exit 1 ;;
esac

source="$BUILD_DIR/$library"
if [[ ! -f "$source" && -n "${FEATHER_NATIVE_CONFIGURATION:-}" ]]; then
    source="$BUILD_DIR/$FEATHER_NATIVE_CONFIGURATION/$library"
fi
if [[ ! -f "$source" ]]; then
    source="$(find "$BUILD_DIR" -maxdepth 3 -type f -name "$library" | head -n 1 || true)"
fi
target="$STAGING_ROOT/runtimes/$RID/native/$library"

if [[ ! -f "$source" ]]; then
    echo "Native library was not found under: $BUILD_DIR" >&2
    echo "Run ./eng/build-native.sh first or set FEATHER_NATIVE_BUILD_DIR." >&2
    exit 1
fi

mkdir -p "$(dirname "$target")"
cp "$source" "$target"
echo "Staged $target"
