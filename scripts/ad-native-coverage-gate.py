#!/usr/bin/env python3
"""Run and enforce native LLVM coverage for Feather's AD bridge.

The gate builds an instrumented native library, runs AD integration tests
against that exact library through FEATHER_NATIVE_LIBRARY, merges LLVM profile
data, and reports scoped native line coverage.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


LINE_THRESHOLD = 80.0
PER_FILE_LINE_THRESHOLD = 80.0
RESULTS_DIR = Path("artifacts/coverage/ad-native")
BUILD_DIR = Path("native/build-ad-coverage")


@dataclass(frozen=True)
class NativeScopeEntry:
    path: str
    reason: str
    ranges: tuple[tuple[int, int], ...]
    important_functions: tuple[str, ...]

    def includes(self, line: int) -> bool:
        return any(start <= line <= end for start, end in self.ranges)


NATIVE_SCOPE: tuple[NativeScopeEntry, ...] = (
    NativeScopeEntry(
        "native/feather_c_api.cpp",
        "AD bridge validation, dispatch, gradient readback, and gradient reduction",
        (
            (390, 419),
            (2066, 2113),
            (4806, 5262),
            (5320, 5455),
            (10508, 10653),
        ),
        (
            "try_dispatch_easygpu_ad_kernel",
            "typed_ir_contains_unsupported_ad_control_flow",
            "release_ad_gradient_buffers",
            "fe_kernel_get_ad_gradient_count",
            "fe_kernel_get_ad_gradient_info",
            "fe_kernel_read_ad_gradient",
            "fe_kernel_reduce_ad_gradient_to_buffer",
            "fe_kernel_get_ad_backward_glsl",
        ),
    ),
    NativeScopeEntry(
        "native/feather_typed_ir_lowerer.cpp",
        "typed IR to EasyGPU callable/control-flow lowering used by AD",
        (
            (203, 220),
            (323, 413),
            (659, 690),
            (895, 1024),
            (1050, 1110),
            (1194, 1269),
            (1655, 1677),
            (2639, 2643),
        ),
        (
            "TryLowerToEasyGpuModule",
            "RegisterCallables",
            "LowerStatement",
            "LowerIf",
            "LowerFor",
            "BuildCallableCall",
        ),
    ),
    NativeScopeEntry(
        "EasyGPU/source/AD/GradientTape.cpp",
        "forward tape recording including callable sub-tapes",
        (
            (88, 120),
            (141, 156),
            (172, 218),
            (229, 307),
            (647, 704),
            (718, 822),
            (828, 855),
        ),
        (
            "GradientTape::RegisterBufferParameter",
            "GradientTape::MarkLoss",
            "GradientTape::RecordCall",
            "GradientTape::PushSubTape",
            "GradientTape::FindSubTapeByCallableName",
        ),
    ),
    NativeScopeEntry(
        "EasyGPU/source/AD/AdjointGenerator.cpp",
        "reverse-mode adjoint generation",
        (
            (429, 590),
            (592, 603),
            (606, 647),
            (649, 756),
            (857, 930),
            (1076, 1295),
        ),
        (
            "AdjointGenerator::Generate",
            "AdjointGenerator::GenerateBody",
            "AdjointGenerator::ProcessEntry",
            "AdjointGenerator::ProcessCall",
            "AdjointGenerator::ProcessControlFlowBegin",
            "AdjointGenerator::PopControlFrameAndWrap",
        ),
    ),
    NativeScopeEntry(
        "EasyGPU/source/Kernel/KernelBuildContext.cpp",
        "callable body capture and AD sub-tape naming",
        (
            (70, 212),
            (364, 425),
            (431, 448),
        ),
        (
            "KernelBuildContext::PushCallableBody",
            "KernelBuildContext::PopCallableBody",
            "KernelBuildContext::GenerateCallableBodies",
            "KernelBuildContext::GetCompleteCode",
        ),
    ),
    NativeScopeEntry(
        "EasyGPU/source/IR/Module.cpp",
        "EasyGPU module/callable/control-flow lowering used by AD",
        (
            (199, 245),
            (301, 330),
            (421, 460),
            (1803, 1825),
            (1872, 1888),
            (1964, 1978),
        ),
        (
            "ModuleBuilder::AddCallable",
            "ModuleBuilder::If",
            "ModuleBuilder::For",
            "BuildKernelBuildContext",
        ),
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run(command: list[str], root: Path, *, env: dict[str, str] | None = None) -> None:
    print(f"\n==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=root, env=env, check=True)


def output(command: list[str], root: Path) -> str:
    return subprocess.check_output(command, cwd=root, text=True, stderr=subprocess.STDOUT).strip()


def llvm_cov_warning_failure(command: list[str], stderr: str) -> str | None:
    warnings = [line for line in stderr.splitlines() if line.strip()]
    if not warnings:
        return None

    # Coverage from llvm-cov is an industrial gate input. Treat warnings as
    # unreliable coverage unless a future warning is deliberately allowlisted here.
    allowlisted: tuple[str, ...] = ()
    unexpected = [line for line in warnings if not any(token in line for token in allowlisted)]
    if not unexpected:
        return None

    return (
        "Native AD coverage gate failed: llvm-cov emitted warnings while running "
        f"`{' '.join(command)}`:\n" + "\n".join(f"  {line}" for line in unexpected)
    )


def find_tool(name: str, root: Path) -> list[str]:
    direct = shutil.which(name)
    if direct:
        return [direct]

    xcrun = shutil.which("xcrun")
    if xcrun:
        try:
            output([xcrun, "--find", name], root)
            return [xcrun, name]
        except subprocess.CalledProcessError:
            pass

    raise SystemExit(
        f"Native AD coverage gate failed: required LLVM tool '{name}' was not found on PATH or through xcrun."
    )


def native_library_name() -> str:
    if sys.platform == "darwin":
        return "libfeather.dylib"
    if os.name == "nt":
        return "feather.dll"
    return "libfeather.so"


def configure_and_build(root: Path, build_dir: Path, *, clean: bool) -> Path:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise SystemExit("Native AD coverage gate failed: cmake was not found on PATH.")

    clang = find_tool("clang", root)
    clangxx = find_tool("clang++", root)
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    coverage_flags = "-O0 -g -fprofile-instr-generate -fcoverage-mapping"
    configure = [
        cmake,
        "-S",
        "native",
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Debug",
        f"-DCMAKE_C_COMPILER={clang[-1] if clang[0].endswith('xcrun') else clang[0]}",
        f"-DCMAKE_CXX_COMPILER={clangxx[-1] if clangxx[0].endswith('xcrun') else clangxx[0]}",
        f"-DCMAKE_C_FLAGS={coverage_flags}",
        f"-DCMAKE_CXX_FLAGS={coverage_flags}",
        f"-DCMAKE_SHARED_LINKER_FLAGS=-fprofile-instr-generate",
        f"-DCMAKE_EXE_LINKER_FLAGS=-fprofile-instr-generate",
        "-DEASYGPU_BUILD_EXAMPLES=OFF",
        "-DEASYGPU_BUILD_TESTS=OFF",
        "-DEASYGPU_BUILD_WINDOW=OFF",
        "-DEASYGPU_BUILD_WINDOW_EXAMPLES=OFF",
        "-DFEATHER_BUILD_AD_NATIVE_COVERAGE_PROBE=ON",
    ]

    env = os.environ.copy()
    if clang[0].endswith("xcrun"):
        env["CC"] = "clang"
    if clangxx[0].endswith("xcrun"):
        env["CXX"] = "clang++"

    run(configure, root, env=env)
    run([cmake, "--build", str(build_dir), "--target", "feather", "--parallel"], root, env=env)
    run([cmake, "--build", str(build_dir), "--target", "feather_ad_native_coverage_probe", "--parallel"], root, env=env)

    library = root / build_dir / native_library_name()
    if not library.exists():
        raise SystemExit(f"Native AD coverage gate failed: instrumented native library was not produced at {library}.")
    return library


def run_ad_tests(root: Path, library: Path, profile_dir: Path) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_pattern = profile_dir / "feather-ad-%p.profraw"
    env = os.environ.copy()
    env["FEATHER_NATIVE_LIBRARY"] = str(library)
    env["LLVM_PROFILE_FILE"] = str(profile_pattern)
    commands = (
        [
            "dotnet",
            "test",
            "tests/Feather.AD.Tests/Feather.AD.Tests.csproj",
            "--no-restore",
        ],
        [
            "dotnet",
            "test",
            "tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj",
            "--no-restore",
            "--filter",
            "FullyQualifiedName~AutoDiff|FullyQualifiedName~AD",
        ],
    )
    for command in commands:
        run(command, root, env=env)

    probe = library.parent / "feather_ad_native_coverage_probe"
    if sys.platform == "win32":
        probe = probe.with_suffix(".exe")
    if not probe.exists():
        raise SystemExit(f"Native AD coverage gate failed: native coverage probe was not produced at {probe}.")

    probe_env = env.copy()
    probe_env["LLVM_PROFILE_FILE"] = str(profile_dir / "feather-ad-native-probe-%p.profraw")
    run([str(probe)], root, env=probe_env)


def merge_profiles(root: Path, profile_dir: Path, profdata: Path) -> None:
    profraws = sorted(profile_dir.glob("*.profraw"))
    if not profraws:
        raise SystemExit(f"Native AD coverage gate failed: no .profraw files were produced in {profile_dir}.")

    llvm_profdata = find_tool("llvm-profdata", root)
    command = [*llvm_profdata, "merge", "-sparse", *map(str, profraws), "-o", str(profdata)]
    run(command, root)


def coverage_report(root: Path, library: Path, profdata: Path, report_path: Path) -> str:
    llvm_cov = find_tool("llvm-cov", root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    source_args = [str(root / entry.path) for entry in NATIVE_SCOPE]
    extra_objects = coverage_objects(library)
    command = [
        *llvm_cov,
        "report",
        str(library),
        *extra_objects,
        f"-instr-profile={profdata}",
        "-show-region-summary=false",
        *source_args,
    ]
    print(f"\n==> {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
    report_path.write_text(result.stdout, encoding="utf-8")
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if failure := llvm_cov_warning_failure(command, result.stderr):
        raise SystemExit(failure)
    return result.stdout


def coverage_export(root: Path, library: Path, profdata: Path, export_path: Path) -> dict:
    llvm_cov = find_tool("llvm-cov", root)
    source_args = [str(root / entry.path) for entry in NATIVE_SCOPE]
    extra_objects = coverage_objects(library)
    command = [
        *llvm_cov,
        "export",
        str(library),
        *extra_objects,
        f"-instr-profile={profdata}",
        "-format=text",
        *source_args,
    ]
    print(f"\n==> {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=root, text=True, capture_output=True, check=True)
    export_path.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if failure := llvm_cov_warning_failure(command, result.stderr):
        raise SystemExit(failure)
    return json.loads(result.stdout)


def coverage_objects(library: Path) -> list[str]:
    # The native probe deliberately executes EasyGPU AD paths, but EasyGPU is
    # linked statically into both the probe and libfeather in this build. Passing
    # both binaries to llvm-cov gives duplicate coverage mappings for the same
    # EasyGPU functions and can produce unreliable "mismatched data" warnings.
    # Profile counters from the probe are still merged; libfeather provides the
    # single authoritative mapping for the scoped Feather/EasyGPU source files.
    return []


def scoped_line_coverage(export: dict, root: Path) -> tuple[dict[str, dict[str, int]], float]:
    scope_by_abs = {(root / entry.path).resolve().as_posix(): entry for entry in NATIVE_SCOPE}
    stats_by_path: dict[str, dict[str, int]] = {}

    for data in export.get("data", []):
        for file_data in data.get("files", []):
            filename = Path(file_data.get("filename", "")).resolve().as_posix()
            scope = scope_by_abs.get(filename)
            if scope is None:
                continue

            line_hits: dict[int, int] = {}
            for segment in file_data.get("segments", []):
                if len(segment) < 4:
                    continue
                line = int(segment[0])
                count = int(segment[2])
                has_count = bool(segment[3])
                if has_count and scope.includes(line):
                    line_hits[line] = max(line_hits.get(line, 0), count)

            stats = {"lines": 0, "covered_lines": 0}
            for line in sorted(line_hits):
                stats["lines"] += 1
                if line_hits[line] > 0:
                    stats["covered_lines"] += 1
            stats_by_path[scope.path] = stats

    total_lines = sum(stats["lines"] for stats in stats_by_path.values())
    total_covered = sum(stats["covered_lines"] for stats in stats_by_path.values())
    total_pct = 100.0 if total_lines == 0 else total_covered * 100.0 / total_lines
    return stats_by_path, total_pct


def function_region_coverage(export: dict) -> dict[str, int]:
    coverage: dict[str, int] = {}
    for data in export.get("data", []):
        for function in data.get("functions", []):
            name = function.get("name", "")
            covered_regions = 0
            for region in function.get("regions", []):
                if len(region) >= 5 and int(region[4]) > 0:
                    covered_regions += 1
            coverage[name] = max(coverage.get(name, 0), covered_regions)
    return coverage


def show_missing_functions(function_coverage: dict[str, int]) -> list[str]:
    failures: list[str] = []

    print("\nImportant native AD functions:")
    for entry in NATIVE_SCOPE:
        for function in entry.important_functions:
            tokens = [token for token in re.split(r"::|\\W+", function) if token]
            matches = [
                covered
                for name, covered in function_coverage.items()
                if name == function or function in name or all(token in name for token in tokens)
            ]
            if not matches:
                print(f"  - {function}: not found in coverage report")
                failures.append(f"{function} was not found in native coverage data")
            else:
                covered_regions = max(matches)
                status = "covered" if covered_regions > 0 else "uncovered"
                print(f"  - {function}: {status}, covered regions {covered_regions}")
                if covered_regions <= 0:
                    failures.append(f"{function} has no covered native regions")

    return failures


def parse_total_line_percent(report: str) -> float | None:
    for line in report.splitlines():
        if line.strip().startswith("TOTAL"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    return float(parts[3].rstrip("%"))
                except ValueError:
                    return None
    return None


def parse_file_line_percentages(report: str) -> dict[str, float]:
    percentages: dict[str, float] = {}
    scope_by_name = {Path(entry.path).name: entry.path for entry in NATIVE_SCOPE}
    for line in report.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0] in {"Filename", "TOTAL"}:
            continue

        filename = Path(parts[0]).name
        scoped = scope_by_name.get(filename)
        if scoped is None:
            continue

        try:
            percentages[scoped] = float(parts[3].rstrip("%"))
        except ValueError:
            continue
    return percentages


def percent(covered: int, total: int) -> float:
    return 100.0 if total == 0 else covered * 100.0 / total


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-build", action="store_true", help="reuse the existing instrumented native build")
    parser.add_argument("--no-test", action="store_true", help="reuse existing native .profraw files")
    parser.add_argument("--clean", action="store_true", help="delete the native coverage build before configuring")
    parser.add_argument("--line-threshold", type=float, default=LINE_THRESHOLD)
    parser.add_argument("--per-file-line-threshold", type=float, default=PER_FILE_LINE_THRESHOLD)
    args = parser.parse_args(argv)

    root = repo_root()
    os.chdir(root)

    results = root / RESULTS_DIR
    build_dir = root / BUILD_DIR
    library = build_dir / native_library_name()
    profile_dir = results / "profraw"
    profdata = results / "feather-ad.profdata"
    report_path = results / "native-coverage-report.txt"
    export_path = results / "native-coverage-export.json"

    if not args.no_build:
        library = configure_and_build(root, build_dir, clean=args.clean)
    elif not library.exists():
        raise SystemExit(f"Native AD coverage gate failed: --no-build requested but {library} does not exist.")

    if not args.no_test:
        if profile_dir.exists():
            shutil.rmtree(profile_dir)
        run_ad_tests(root, library, profile_dir)

    merge_profiles(root, profile_dir, profdata)
    report = coverage_report(root, library, profdata, report_path)
    export = coverage_export(root, library, profdata, export_path)
    scoped_stats, scoped_total_pct = scoped_line_coverage(export, root)
    function_coverage = function_region_coverage(export)

    print("Native AD scope:")
    for entry in NATIVE_SCOPE:
        ranges = ", ".join(f"{start}-{end}" for start, end in entry.ranges)
        print(f"  - {entry.path} ({ranges}): {entry.reason}")

    print(
        "\nPer-file native AD scoped coverage "
        f"(threshold: lines >= {args.per_file_line_threshold:.0f}%):"
    )
    for entry in NATIVE_SCOPE:
        stats = scoped_stats.get(entry.path, {"lines": 0, "covered_lines": 0})
        print(
            f"  - {entry.path}: lines {percent(stats['covered_lines'], stats['lines']):5.1f}% "
            f"({stats['covered_lines']}/{stats['lines']})"
        )

    failures: list[str] = []
    if scoped_total_pct < args.line_threshold:
        failures.append(f"native scoped line coverage {scoped_total_pct:.2f}% < {args.line_threshold:.0f}%")

    for entry in NATIVE_SCOPE:
        stats = scoped_stats.get(entry.path)
        if stats is None or stats["lines"] == 0:
            failures.append(f"{entry.path} was missing from native coverage report")
            continue

        file_pct = percent(stats["covered_lines"], stats["lines"])
        if stats["covered_lines"] == 0:
            failures.append(f"{entry.path} has 0.00% native line coverage")
        elif file_pct < args.per_file_line_threshold:
            failures.append(
                f"{entry.path} line coverage {file_pct:.2f}% < {args.per_file_line_threshold:.0f}% "
                f"({stats['covered_lines']}/{stats['lines']})"
            )

    failures.extend(show_missing_functions(function_coverage))

    if failures:
        print("\nNative AD coverage gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    branch_note = "LLVM llvm-cov line/region coverage was gated; branch counters are reported by llvm-cov when available."
    print(
        f"\nNative AD coverage gate passed: scoped lines {scoped_total_pct:.2f}% >= {args.line_threshold:.0f}%. "
        f"Each scoped native AD file has lines >= {args.per_file_line_threshold:.0f}%. "
        f"{branch_note}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
