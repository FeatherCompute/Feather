#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${FEATHER_CI_JOBS:-8}"

run() {
    echo
    echo "==> $*"
    "$@"
}

configure_native_if_needed() {
    if [[ ! -f "$ROOT/native/build/CMakeCache.txt" ]]; then
        run cmake -S "$ROOT/native" -B "$ROOT/native/build"
    fi
}

copy_native_asset_if_present() {
    local source="$ROOT/native/build/libfeather.dylib"
    local target="$ROOT/src/Feather.NativeAssets/runtimes/osx-arm64/native/libfeather.dylib"

    if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -f "$source" ]]; then
        mkdir -p "$(dirname "$target")"
        run cp "$source" "$target"
    fi
}

run_test_projects() {
    local projects=(
        "tests/Feather.Tests/Feather.Tests.csproj"
        "tests/Feather.Native.Tests/Feather.Native.Tests.csproj"
        "tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj"
        "tests/Feather.Gpu.Tests/Feather.Gpu.Tests.csproj"
        "tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj"
        "tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj"
        "tests/Feather.AD.Tests/Feather.AD.Tests.csproj"
        "tests/Feather.NN.Tests/Feather.NN.Tests.csproj"
    )

    for project in "${projects[@]}"; do
        run dotnet test "$ROOT/$project" --no-restore -v minimal
    done
}

run_window_opt_in_tests_if_requested() {
    if [[ "${FEATHER_CI_RUN_WINDOW_TESTS:-0}" == "1" ]]; then
        echo
        echo "==> FEATHER_WINDOW_TESTS=1 dotnet test tests/Feather.Graphics.Tests"
        FEATHER_WINDOW_TESTS=1 dotnet test "$ROOT/tests/Feather.Graphics.Tests/Feather.Graphics.Tests.csproj" --no-restore -v minimal
    else
        echo
        echo "==> skipping native window opt-in tests; set FEATHER_CI_RUN_WINDOW_TESTS=1 to include them"
    fi
}

build_sample_projects() {
    local projects=(
        "samples/HelloWorld/HelloWorld.csproj"
        "samples/HelloBuffer/HelloBuffer.csproj"
        "samples/TextureCopy/TextureCopy.csproj"
        "samples/ParallelReduction/ParallelReduction.csproj"
        "samples/JuliaSet/JuliaSet.csproj"
        "samples/SpirvOptInspection/SpirvOptInspection.csproj"
        "samples/Histogram/Histogram.csproj"
        "samples/ColorFilter/ColorFilter.csproj"
        "samples/RayTracing/RayTracing.csproj"
        "samples/AdLinearRegression/AdLinearRegression.csproj"
        "samples/AdTransformer/AdTransformer.csproj"
        "samples/AdGptDemo/AdGptDemo.csproj"
        "samples/AdGptPoetDemo/AdGptPoetDemo.csproj"
        "samples/WindowHello/WindowHello.csproj"
        "samples/WindowPixels/WindowPixels.csproj"
        "samples/WindowCompute/WindowCompute.csproj"
        "samples/WindowGraphicsTriangle/WindowGraphicsTriangle.csproj"
        "samples/WindowGraphicsTexturedQuad/WindowGraphicsTexturedQuad.csproj"
        "samples/SponzaRenderer/SponzaRenderer.csproj"
    )

    for project in "${projects[@]}"; do
        run dotnet build "$ROOT/$project" --no-restore -v minimal
    done
}

run_sample_smoke_tests() {
    run dotnet run --no-restore --project "$ROOT/samples/HelloWorld/HelloWorld.csproj" -- 1024
    run dotnet run --no-restore --project "$ROOT/samples/HelloBuffer/HelloBuffer.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/TextureCopy/TextureCopy.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/ParallelReduction/ParallelReduction.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/Histogram/Histogram.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/ColorFilter/ColorFilter.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/SpirvOptInspection/SpirvOptInspection.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/JuliaSet/JuliaSet.csproj" -- 32 32
    run dotnet run --no-restore --project "$ROOT/samples/RayTracing/RayTracing.csproj" -- 32 32 4
    run dotnet run --no-restore --project "$ROOT/samples/AdLinearRegression/AdLinearRegression.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/AdTransformer/AdTransformer.csproj"
    run dotnet run --no-restore --project "$ROOT/samples/AdGptDemo/AdGptDemo.csproj"

    if [[ "${FEATHER_CI_SKIP_LONG_SAMPLES:-0}" == "1" ]]; then
        echo
        echo "==> skipping AdGptPoetDemo; unset FEATHER_CI_SKIP_LONG_SAMPLES to include it"
    else
        run dotnet run --no-restore --project "$ROOT/samples/AdGptPoetDemo/AdGptPoetDemo.csproj"
    fi
}

run_sponza_capture_if_assets_exist() {
    if [[ -f "$ROOT/Sponza/sponza.obj" ]]; then
        run dotnet run --no-restore --project "$ROOT/samples/SponzaRenderer/SponzaRenderer.csproj" -- "$ROOT/Sponza" --capture "$ROOT/artifacts/images/sponza-ci.tga"
    else
        echo
        echo "==> skipping Sponza capture; Sponza/sponza.obj not found"
    fi
}

cd "$ROOT"

configure_native_if_needed
run cmake --build "$ROOT/native/build" --target feather --parallel "$JOBS"
copy_native_asset_if_present

run dotnet build "$ROOT/Feather.slnx" --no-restore -v minimal
run_test_projects
run_window_opt_in_tests_if_requested

run python3 "$ROOT/scripts/ad-industrial-gate.py" --native-clean
run python3 "$ROOT/scripts/nn-industrial-gate.py" --clean

build_sample_projects
run_sample_smoke_tests
run_sponza_capture_if_assets_exist

echo
echo "Feather CI gate passed."
