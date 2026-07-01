#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${FEATHER_NATIVE_BUILD_DIR:-$ROOT/native/build}"
JOBS="${FEATHER_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

cmake_args=()
if [[ -n "${FEATHER_EASYGPU_BACKEND:-}" ]]; then
    cmake_args+=("-DEASYGPU_BACKEND=${FEATHER_EASYGPU_BACKEND}")
fi

if [[ -n "${FEATHER_BUILD_WINDOW:-}" ]]; then
    cmake_args+=("-DFEATHER_BUILD_WINDOW=${FEATHER_BUILD_WINDOW}")
fi

if [[ ${#cmake_args[@]} -gt 0 ]]; then
    cmake -S "$ROOT/native" -B "$BUILD_DIR" "${cmake_args[@]}"
else
    cmake -S "$ROOT/native" -B "$BUILD_DIR"
fi
cmake --build "$BUILD_DIR" --target feather --parallel "$JOBS"
