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
    win-*) library="feather_native.dll" ;;
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

is_macos_system_dependency() {
    local dependency="$1"

    case "$dependency" in
        /System/Library/*|/usr/lib/*)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

append_unique_line() {
    local file="$1"
    local value="$2"

    if [[ -z "$value" ]]; then
        return
    fi

    if [[ -f "$file" ]] && grep -Fqx "$value" "$file"; then
        return
    fi

    printf '%s\n' "$value" >> "$file"
}

print_macos_rpaths() {
    local binary="$1"

    otool -l "$binary" | awk '
        $1 == "cmd" && $2 == "LC_RPATH" {
            getline
            getline
            if ($1 == "path") {
                print $2
            }
        }'
}

print_macos_dependencies() {
    local binary="$1"

    otool -L "$binary" | awk 'NR > 1 { print $1 }'
}

print_macos_dependency_search_dirs() {
    printf '%s\n' \
        "$(dirname "$target")" \
        /opt/homebrew/lib \
        /usr/local/lib \
        /opt/homebrew/opt/glslang/lib \
        /opt/homebrew/opt/spirv-tools/lib \
        /opt/homebrew/opt/vulkan-loader/lib \
        /opt/homebrew/opt/openssl@3/lib \
        /usr/local/opt/glslang/lib \
        /usr/local/opt/spirv-tools/lib \
        /usr/local/opt/vulkan-loader/lib \
        /usr/local/opt/openssl@3/lib

    if command -v brew >/dev/null 2>&1; then
        local formula prefix
        for formula in glslang spirv-tools vulkan-loader openssl@3; do
            prefix="$(brew --prefix "$formula" 2>/dev/null || true)"
            if [[ -n "$prefix" && -d "$prefix/lib" ]]; then
                printf '%s\n' "$prefix/lib"
            fi
        done
    fi
}

resolve_macos_dependency() {
    local dependency="$1"
    local binary="$2"
    local suffix candidate rpath

    if [[ "$dependency" == /* ]]; then
        if [[ -f "$dependency" ]]; then
            printf '%s\n' "$dependency"
            return 0
        fi
        return 1
    fi

    if [[ "$dependency" == @loader_path/* ]]; then
        suffix="${dependency#@loader_path/}"
        candidate="$(dirname "$binary")/$suffix"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    elif [[ "$dependency" == @executable_path/* ]]; then
        suffix="${dependency#@executable_path/}"
        candidate="$(dirname "$target")/$suffix"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    elif [[ "$dependency" == @rpath/* ]]; then
        suffix="${dependency#@rpath/}"
        while IFS= read -r rpath; do
            case "$rpath" in
                @loader_path*)
                    candidate="$(dirname "$binary")/${rpath#@loader_path/}/$suffix"
                    ;;
                @executable_path*)
                    candidate="$(dirname "$target")/${rpath#@executable_path/}/$suffix"
                    ;;
                *)
                    candidate="$rpath/$suffix"
                    ;;
            esac

            if [[ -f "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done < <(print_macos_rpaths "$binary")
    else
        suffix="$dependency"
    fi

    suffix="${dependency##*/}"
    while IFS= read -r search_dir; do
        candidate="$search_dir/$suffix"
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done < <(print_macos_dependency_search_dirs)

    return 1
}

ensure_macos_loader_path_rpath() {
    local binary="$1"

    if print_macos_rpaths "$binary" | grep -Fqx "@loader_path"; then
        return
    fi

    install_name_tool -add_rpath "@loader_path" "$binary"
}

remove_macos_host_rpaths() {
    local binary="$1"
    local rpath

    while IFS= read -r rpath; do
        case "$rpath" in
            /opt/homebrew*|/usr/local*)
                install_name_tool -delete_rpath "$rpath" "$binary" 2>/dev/null || true
                ;;
        esac
    done < <(print_macos_rpaths "$binary")
}

normalize_macos_install_id() {
    local binary="$1"
    local install_id

    install_id="$(print_macos_dependencies "$binary" | sed -n '1p')"
    case "$install_id" in
        /opt/homebrew*|/usr/local*)
            install_name_tool -id "@loader_path/$(basename "$binary")" "$binary"
            ;;
    esac
}

stage_macos_dependencies() {
    local native_dir pending processed current dependency dependency_name resolved staged_dependency

    if ! command -v otool >/dev/null 2>&1 || ! command -v install_name_tool >/dev/null 2>&1; then
        echo "macOS native asset staging requires otool and install_name_tool." >&2
        exit 1
    fi

    native_dir="$(dirname "$target")"
    pending="$native_dir/.feather-native-pending"
    processed="$native_dir/.feather-native-processed"
    : > "$pending"
    : > "$processed"
    append_unique_line "$pending" "$target"

    while [[ -s "$pending" ]]; do
        current="$(sed -n '1p' "$pending")"
        sed '1d' "$pending" > "$pending.tmp"
        mv "$pending.tmp" "$pending"

        if grep -Fqx "$current" "$processed"; then
            continue
        fi
        append_unique_line "$processed" "$current"

        chmod u+w "$current"
        ensure_macos_loader_path_rpath "$current"
        normalize_macos_install_id "$current"

        while IFS= read -r dependency; do
            if [[ -z "$dependency" ]] || is_macos_system_dependency "$dependency"; then
                continue
            fi

            dependency_name="${dependency##*/}"
            if [[ "$dependency_name" == "$(basename "$current")" ]]; then
                continue
            fi

            if ! resolved="$(resolve_macos_dependency "$dependency" "$current")"; then
                echo "Could not resolve macOS native dependency '$dependency' referenced by '$current'." >&2
                exit 1
            fi

            staged_dependency="$native_dir/$dependency_name"
            if [[ "$resolved" != "$staged_dependency" ]]; then
                cp -f "$resolved" "$staged_dependency"
                chmod u+w "$staged_dependency"
            fi

            case "$dependency" in
                @rpath/*)
                    ;;
                *)
                    install_name_tool -change "$dependency" "@loader_path/$dependency_name" "$current"
                    ;;
            esac

            append_unique_line "$pending" "$staged_dependency"
        done < <(print_macos_dependencies "$current")

        remove_macos_host_rpaths "$current"
    done

    rm -f "$pending" "$processed"

    if command -v codesign >/dev/null 2>&1; then
        find "$native_dir" -maxdepth 1 -type f -name '*.dylib' -print0 |
            while IFS= read -r -d '' current; do
                codesign --force --sign - "$current" >/dev/null 2>&1 || true
            done
    fi

    if find "$native_dir" -maxdepth 1 -type f -name '*.dylib' -print0 |
        xargs -0 otool -L |
        grep -E '/opt/homebrew|/usr/local/(opt|Cellar|lib)' >/dev/null; then
        echo "macOS native asset contains host-local Homebrew install names:" >&2
        find "$native_dir" -maxdepth 1 -type f -name '*.dylib' -print0 | xargs -0 otool -L >&2
        exit 1
    fi

    if find "$native_dir" -maxdepth 1 -type f -name '*.dylib' -print0 |
        xargs -0 otool -l |
        grep -E '/opt/homebrew|/usr/local/(opt|Cellar|lib)' >/dev/null; then
        echo "macOS native asset contains host-local Homebrew rpaths:" >&2
        find "$native_dir" -maxdepth 1 -type f -name '*.dylib' -print0 | xargs -0 otool -l >&2
        exit 1
    fi
}

if [[ "$RID" == osx-* ]]; then
    stage_macos_dependencies
fi

echo "Staged $target"
