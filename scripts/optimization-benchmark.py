#!/usr/bin/env python3
"""Build and run Feather's reproducible Vulkan shader optimization benchmark."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"\n==> {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def captured(command: list[str]) -> str:
    try:
        result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def find_file(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if not matches:
        raise FileNotFoundError(f"{name} was not found under {root}")
    return matches[0]


def host_info() -> dict[str, str]:
    cpu = platform.processor()
    gpu = ""
    if sys.platform == "darwin":
        cpu = captured(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        display = captured(["system_profiler", "SPDisplaysDataType"])
        for line in display.splitlines():
            if "Chipset Model:" in line:
                gpu = line.split(":", 1)[1].strip()
                break
    elif sys.platform.startswith("linux"):
        lspci = captured(["lspci"])
        gpu = next(
            (line.split(": ", 1)[-1] for line in lspci.splitlines() if "VGA compatible controller" in line),
            "",
        )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu or "unknown",
        "gpu": gpu or "reported by Vulkan driver",
    }


def percentage_delta(value: float, reference: float) -> float:
    return ((value / reference) - 1.0) * 100.0 if reference else 0.0


def render_report(report: dict[str, Any]) -> str:
    native = report["native"]
    results = native["results"]
    by_key = {(item["scenario"], item["authoring"], item["level"]): item for item in results}
    scenarios = [item["Name"] for item in report["sourceLowering"]]
    timing_kind = (
        "GPU timestamp queries"
        if all(item["usedGpuTimestamps"] for item in results)
        else "synchronized host timing; each wait is amortized across "
             f"{native['dispatchesPerSample']} dispatches"
    )

    lines = [
        "# Feather Shader Optimization Benchmark",
        "",
        f"Generated UTC: `{report['generatedAtUtc']}`",
        "",
        "## Environment",
        "",
        f"- Platform: `{report['host']['platform']}`",
        f"- CPU: `{report['host']['cpu']}`",
        f"- GPU: `{report['host']['gpu']}`",
        f"- Backend: `{native['backend']} ({native['device']})`",
        f"- Execution timing: {timing_kind}",
        f"- Warmup rounds: `{native['warmupIterations']}`",
        f"- Measured rounds: `{native['measuredIterations']}`",
        f"- Dispatches per sample: `{native['dispatchesPerSample']}`",
        f"- Compile samples: `{native['compileSamples']}`",
        "",
        "## Production Summary",
        "",
        "| Scenario | Feather source lowering ms | Feather Ultra bytes | Handwritten Ultra bytes | Feather code reduction | Feather Ultra ms | Handwritten Ultra ms | Runtime delta | Ultra cold/warm cache speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    source_by_name = {item["Name"]: item for item in report["sourceLowering"]}
    for scenario in scenarios:
        feather = by_key[(scenario, "Feather", "Ultra")]
        handwritten = by_key[(scenario, "Handwritten", "Ultra")]
        source = source_by_name[scenario]
        code_reduction = (1.0 - feather["optimizedGlslBytes"] / handwritten["optimizedGlslBytes"]) * 100.0
        runtime_delta = percentage_delta(feather["dispatchMedianMs"], handwritten["dispatchMedianMs"])
        cache_speedup = feather["coldInspectionMedianMs"] / feather["warmInspectionMedianMs"]
        lines.append(
            f"| `{scenario}` | {source['SourceLoweringMedianMs']:.3f} | "
            f"{feather['optimizedGlslBytes']} | {handwritten['optimizedGlslBytes']} | "
            f"{code_reduction:.2f}% | {feather['dispatchMedianMs']:.4f} | "
            f"{handwritten['dispatchMedianMs']:.4f} | {runtime_delta:+.2f}% | {cache_speedup:.2f}x |"
        )

    lines.extend([
        "",
        "## Feather Level Sweep",
        "",
        "The same production Feather GLSL is used for every backend level so this table isolates SPIR-V optimization.",
        "",
        "| Scenario | Level | Optimized bytes | Cold inspection ms | Optimizer ms | Warm cache ms | Dispatch ms | P95 ms | Speedup vs None |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for scenario in scenarios:
        baseline = by_key[(scenario, "Feather", "None")]["dispatchMedianMs"]
        for level in ("None", "Size", "Aggressive", "Ultra", "Extreme"):
            item = by_key[(scenario, "Feather", level)]
            speedup = baseline / item["dispatchMedianMs"] if item["dispatchMedianMs"] else 0.0
            lines.append(
                f"| `{scenario}` | `{level}` | {item['optimizedGlslBytes']} | "
                f"{item['coldInspectionMedianMs']:.3f} | {item['optimizerMedianMs']:.3f} | "
                f"{item['warmInspectionMedianMs']:.3f} | {item['dispatchMedianMs']:.4f} | "
                f"{item['dispatchP95Ms']:.4f} | {speedup:.3f}x |"
            )

    lines.extend([
        "",
        "## Validation",
        "",
        "Every source/level result is checked against the handwritten Ultra output after execution.",
        "",
        "| Scenario | Authoring | Level | Max absolute error | Values outside tolerance | Checksum |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ])
    for item in results:
        lines.append(
            f"| `{item['scenario']}` | `{item['authoring']}` | `{item['level']}` | "
            f"{item['maxAbsoluteError']:.9g} | {item['mismatchedValues']} | {item['checksum']:.6f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Cold inspection includes glslang, SPIRV-Tools, cache write, and SPIRV-Cross. Optimized levels intentionally spend more cold-start CPU time.",
        "- Warm inspection reads and validates cached SPIR-V, then runs SPIRV-Cross. It demonstrates persistent shader-cache startup behavior, not GPU execution.",
        "- FEIR-to-GLSL source lowering is measured separately. It does not include Roslyn/C# build time.",
        "- Similar dispatch medians across levels mean the active Vulkan driver normalized the variants. Code size and cache gains remain real, but this run does not support claiming a steady-state GPU speedup.",
        "- Compare results only within the same report. Power state, driver, GPU, and timestamp-query support materially affect sub-millisecond kernels.",
        "",
        "## Workloads",
        "",
        "- `fused-mlp`: 1,048,576 elements, 64 fused affine/activation/polynomial/residual steps per element.",
        "- `particle-sim`: 524,288 particles, 128 fixed integration steps with collision branches per particle.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "optimization-benchmark"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-samples", type=int, default=9)
    parser.add_argument("--dispatches-per-sample", type=int, default=8)
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    build_directory = output / "native-build"
    source_directory = output / "sources"
    raw_result_path = output / "native-results.json"
    output.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.warmup = 2
        args.iterations = 8
        args.compile_samples = 3
        args.dispatches_per_sample = 3

    if not args.skip_build:
        run([
            "cmake", "-S", "native", "-B", str(build_directory),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DFEATHER_BUILD_WINDOW=OFF",
            "-DFEATHER_BUILD_OPTIMIZATION_BENCHMARK=ON",
            "-DFEATHER_SHADER_OPTIMIZATION_LEVEL=Ultra",
            "-DEASYGPU_BACKEND=Vulkan",
            "-DEASYGPU_ENABLE_SPIRV_OPT=ON",
            "-DEASYGPU_ENABLE_SPIRV_CROSS_GLSL=ON",
        ])
        run([
            "cmake", "--build", str(build_directory), "--target", "feather",
            "feather_optimization_benchmark", "--config", "Release", "--parallel", str(os.cpu_count() or 4),
        ])
        run(["dotnet", "build", "benchmarks/ShaderOptimization/ShaderOptimization.csproj", "-c", "Release", "-v", "minimal"])

    native_library_name = {
        "darwin": "libfeather.dylib",
        "win32": "feather_native.dll",
    }.get(sys.platform, "libfeather.so")
    runner_name = "feather_optimization_benchmark.exe" if sys.platform == "win32" else "feather_optimization_benchmark"
    native_library = find_file(build_directory, native_library_name)
    runner = find_file(build_directory, runner_name)

    exporter_env = os.environ.copy()
    exporter_env["FEATHER_NATIVE_LIBRARY"] = str(native_library)
    run([
        "dotnet", "run", "--no-build", "-c", "Release", "--project",
        "benchmarks/ShaderOptimization/ShaderOptimization.csproj", "--",
        "--output", str(source_directory), "--samples", str(args.compile_samples),
    ], env=exporter_env)

    run([
        str(runner), "--sources", str(source_directory), "--output", str(raw_result_path),
        "--warmup", str(args.warmup), "--iterations", str(args.iterations),
        "--compile-samples", str(args.compile_samples),
        "--dispatches-per-sample", str(args.dispatches_per_sample),
    ])

    native = json.loads(raw_result_path.read_text())
    source_lowering = json.loads((source_directory / "feather-source-metadata.json").read_text())
    report = {
        "generatedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": host_info(),
        "sourceLowering": source_lowering,
        "native": native,
    }
    json_path = output / "optimization-benchmark.json"
    markdown_path = output / "optimization-benchmark.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(render_report(report))

    print(f"\nJSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
