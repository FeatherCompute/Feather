#!/usr/bin/env python3
"""Run and enforce Feather's scoped managed AD coverage gate.

The gate intentionally measures AD-scoped source paths/ranges, not whole-repo
coverage. It enforces both aggregate and per-file scoped coverage so low-risk
files cannot hide missing AD coverage in the aggregate.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


LINE_THRESHOLD = 90.0
BRANCH_THRESHOLD = 90.0
PER_FILE_LINE_THRESHOLD = 90.0
RESULTS_DIR = Path("artifacts/coverage/ad-scoped")


@dataclass(frozen=True)
class ScopeEntry:
    path: str
    reason: str
    ranges: tuple[tuple[int, int], ...] = ()
    aliases: tuple[str, ...] = ()

    def includes(self, line: int) -> bool:
        if not self.ranges:
            return True
        return any(start <= line <= end for start, end in self.ranges)


MANAGED_SCOPE: tuple[ScopeEntry, ...] = (
    ScopeEntry(
        "src/Feather/AD/AD.cs",
        "managed AD markers, kernel wrapper, supported gradient readback",
        ((12, 34), (44, 124), (128, 144), (150, 181), (190, 339)),
        ("Feather/AD/AD.cs",),
    ),
    ScopeEntry("src/Feather/Core/GPU.cs", "GPU.CreateADKernel facade", ((173, 178),), ("Feather/Core/GPU.cs",)),
    ScopeEntry("src/Feather/Kernels/GpuKernel.cs", "retained dispatch path that drives native AD", ((35, 92),), ("Feather/Kernels/GpuKernel.cs",)),
    ScopeEntry("src/Feather.Native/NativeMethods.cs", "AD native P/Invoke declarations", ((230, 248),), ("Feather.Native/NativeMethods.cs",)),
    ScopeEntry("src/Feather.Native/NativeStructs.cs", "AD kernel-create flag and gradient metadata ABI", ((112, 130), (176, 205)), ("Feather.Native/NativeStructs.cs",)),
    ScopeEntry(
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        "AD marker validation, AD callable policy, and traceable parameter analysis",
        ((272, 411), (714, 734), (1484, 1528), (1539, 1629)),
        ("Feather.Generators/Model/ShaderModelFactory.cs",),
    ),
    ScopeEntry("src/Feather.Generators/Model/ShaderSemanticLowerer.cs", "AD annotation lowering into IR metadata", ((344, 392),), ("Feather.Generators/Model/ShaderSemanticLowerer.cs",)),
    ScopeEntry("src/Feather.Generators/Model/ShaderModels.cs", "AD model records and enums", ((252, 265),), ("Feather.Generators/Model/ShaderModels.cs",)),
    ScopeEntry("src/Feather.Generators/IR/FeatherIrWriter.cs", "AD annotation section writer", ((12, 16), (621, 663), (1188, 1198)), ("Feather.Generators/IR/FeatherIrWriter.cs",)),
    ScopeEntry(
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        "typed IR callable and AD marker lowering paths",
        ((62, 80), (124, 135), (623, 639)),
        ("Feather.Generators/Lowering/ShaderIrLowerer.cs",),
    ),
    ScopeEntry(
        "src/Feather.Generators/Lowering/ShaderIrModuleWriter.cs",
        "typed callable/control-flow serialization used by AD",
        ((28, 60), (200, 225), (260, 334), (440, 452)),
        ("Feather.Generators/Lowering/ShaderIrModuleWriter.cs",),
    ),
)


SCOPE_BY_PATH = {
    candidate: entry
    for entry in MANAGED_SCOPE
    for candidate in (entry.path, *entry.aliases)
}

LINE_EXCLUSIONS: dict[tuple[str, int], str] = {
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        281,
    ): "Execute-body absence guard is unreachable for analyzer-accepted block-bodied AD kernels",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        283,
    ): "Execute-body absence guard is unreachable for analyzer-accepted block-bodied AD kernels",
    **{
        ("src/Feather.Generators/Model/ShaderModelFactory.cs", line):
            "wrong-arity AD marker body is unreachable through C# overload resolution"
        for line in range(315, 323)
    },
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1487,
    ): "type-null guard for AD marker value-type helper is defensive after Roslyn binding",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1488,
    ): "type-null guard for AD marker value-type helper is defensive after Roslyn binding",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        75,
    ): "callable missing-body guard is unreachable for valid compiled callable methods",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        76,
    ): "callable missing-body guard is unreachable for valid compiled callable methods",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        637,
    ): "recursive child-operation marker detection guard; direct AD marker skipping is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        638,
    ): "recursive child-operation marker detection guard; direct AD marker skipping is covered",
    (
        "src/Feather/AD/AD.cs",
        225,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        226,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        227,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        256,
    ): "native zero-gradient metadata corruption guard; valid zero-gradient failure is covered by native AD tests",
    (
        "src/Feather/AD/AD.cs",
        257,
    ): "native zero-gradient metadata corruption guard; valid zero-gradient failure is covered by native AD tests",
    (
        "src/Feather/AD/AD.cs",
        265,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        266,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        267,
    ): "native metadata empty-name fallback requires fault injection; valid named readback is covered",
    (
        "src/Feather/AD/AD.cs",
        272,
    ): "native byte-size corruption guard requires fault injection; valid scalar/vector readback is covered",
    (
        "src/Feather/AD/AD.cs",
        273,
    ): "native byte-size corruption guard requires fault injection; valid scalar/vector readback is covered",
    (
        "src/Feather/AD/AD.cs",
        289,
    ): "native scalar-layout corruption guard requires fault injection; managed bad-shape conversion guards are covered",
    (
        "src/Feather/AD/AD.cs",
        290,
    ): "native scalar-layout corruption guard requires fault injection; managed bad-shape conversion guards are covered",
}

BRANCH_EXCLUSIONS: dict[tuple[str, int], str] = {
    (
        "src/Feather/AD/AD.cs",
        224,
    ): "native metadata corruption guard; valid zero-gradient-count failure is covered by native AD failure tests",
    (
        "src/Feather/AD/AD.cs",
        255,
    ): "native metadata empty-name fallback; valid named readback is covered and resource-name fallback requires fault injection",
    (
        "src/Feather/AD/AD.cs",
        264,
    ): "compiler-emitted fixed-buffer branch around native gradient readback; successful scalar/vector readback is covered",
    (
        "src/Feather/AD/AD.cs",
        271,
    ): "native metadata scalar-layout corruption guard; valid scalar/vector layouts and managed bad-shape conversion guards are covered",
    (
        "src/Feather/AD/AD.cs",
        277,
    ): "generic type-pattern counter mixes tested TryGetArray true/false with compiler null-pattern bookkeeping",
    (
        "src/Feather/AD/AD.cs",
        288,
    ): "native metadata scalar-layout corruption guard; valid scalar/vector layouts and managed bad-shape conversion guards are covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        50,
    ): "Roslyn operation-null guard for typed IR lowering; valid and lowering-exception generator paths are covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        62,
    ): "exception-location null fallback for typed IR diagnostics; location-bearing lowering failures are covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        280,
    ): "Execute-body absence guard is unreachable for analyzer-accepted compute kernels",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        289,
    ): "AD invocation symbol-filter null subbranches; marker and non-marker paths are covered by generator diagnostics",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        314,
    ): "wrong-arity AD marker guard is unreachable through C# overload resolution",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        339,
    ): "loss type null subbranch is unreachable for bound AD.Loss overloads; scalar and non-scalar losses are tested",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        353,
    ): "parameter type null subbranch is unreachable for bound AD.Parameter overloads; supported and unsupported types are tested",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        371,
    ): "non-buffer source guard is retained defensively; texture/sampler and buffer paths are covered at diagnostics",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        411,
    ): "AD marker type null branch is defensive after Roslyn symbol/candidate filtering",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        723,
    ): "nested-callable AD guard includes Roslyn symbol-null subbranches; supported callable and nested rejection paths are covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1486,
    ): "type-null guard for AD marker value-type helper; concrete supported/unsupported marker types are tested",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1547,
    ): "traceable-source resource-kind and argument-count guard includes defensive non-resource element access branches",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1553,
    ): "resource-name syntax switch fallback bookkeeping; identifier buffer sources are covered by AD metadata tests",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1561,
    ): "empty resource/index defensive guard is unreachable for parsed buffer element syntax",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1566,
    ): "value-type null fallback for parsed buffer element syntax; scalar/vector parameter metadata is covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1582,
    ): "local-alias declaration-shape guard; direct, casted, mutated, and untraceable aliases are covered",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1591,
    ): "alias Execute-body absence guard is unreachable for validated kernels",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1599,
    ): "alias assignment scan line mixes covered reassignment rejection with out-of-window Roslyn pattern bookkeeping",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1616,
    ): "alias increment scan line mixes covered increment/decrement rejection with out-of-window Roslyn pattern bookkeeping",
    (
        "src/Feather.Generators/Model/ShaderModelFactory.cs",
        1621,
    ): "increment/decrement operation pattern includes defensive non-local target subbranches",
    (
        "src/Feather.Generators/Model/ShaderSemanticLowerer.cs",
        351,
    ): "wrong-arity AD marker lowering guard is unreachable through C# overload resolution",
    (
        "src/Feather.Generators/Model/ShaderSemanticLowerer.cs",
        354,
    ): "operation unwrap null guard is defensive; local and buffer AD annotations are covered",
    (
        "src/Feather.Generators/Model/ShaderSemanticLowerer.cs",
        379,
    ): "local-name fallback switch includes non-AD field/parameter variants; supported local and buffer annotations are covered",
    (
        "src/Feather.Generators/Model/ShaderSemanticLowerer.cs",
        386,
    ): "null local-name branch is defensive after AD marker validation",
    (
        "src/Feather.Generators/Model/ShaderSemanticLowerer.cs",
        392,
    ): "type-name null fallback is defensive for bound AD marker operands",
    (
        "src/Feather.Generators/IR/FeatherIrWriter.cs",
        633,
    ): "AD annotation binding fallback requires malformed resource metadata; generated buffer/local annotations are covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        74,
    ): "callable missing-body guard is unreachable for valid compiled callable methods",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        69,
    ): "callable return-type fallback is defensive; callable return lowering is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        71,
    ): "callable parameter type fallback is defensive; callable parameter lowering is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        121,
    ): "generic Roslyn statement pattern-switch dispatch; AD marker, callable, and control-flow cases are covered by targeted tests",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        133,
    ): "return-without-value branch belongs to void callable support outside current AD callable scope",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        530,
    ): "primitive-type helper false branch is generic swizzle/index infrastructure outside AD marker lowering",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        533,
    ): "AD marker invocation helper includes non-AD invocation subbranches; marker skip behavior is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        624,
    ): "AD marker invocation helper includes defensive symbol subbranches; marker skip behavior is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        636,
    ): "recursive child-operation marker detection guard; direct AD marker skipping is covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        949,
    ): "callable attribute helper includes non-callable subbranches; callable lowering and nested AD rejection are covered",
    (
        "src/Feather.Generators/Lowering/ShaderIrLowerer.cs",
        954,
    ): "swizzle-shape helper is generic vector lowering outside AD marker/callable scope",
    (
        "src/Feather.Generators/Lowering/ShaderIrModuleWriter.cs",
        266,
    ): "generic typed-IR statement serialization switch; AD callable/control-flow statement cases are covered by targeted tests",
    (
        "src/Feather.Generators/Lowering/ShaderIrModuleWriter.cs",
        302,
    ): "optional for-loop init/condition/step serialization covers non-canonical forms outside supported AD for-loop scope",
    (
        "src/Feather.Generators/Lowering/ShaderIrModuleWriter.cs",
        329,
    ): "return-without-value serialization belongs to void callable support outside current AD callable scope",
}


TEST_SLICES = (
    ("ad", ["dotnet", "test", "tests/Feather.AD.Tests/Feather.AD.Tests.csproj", "--no-restore"]),
    (
        "integration-ad",
        [
            "dotnet",
            "test",
            "tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj",
            "--no-restore",
            "--filter",
            "FullyQualifiedName~AutoDiff|FullyQualifiedName~AD",
        ],
    ),
    (
        "generator-ad",
        [
            "dotnet",
            "test",
            "tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj",
            "--no-restore",
            "--filter",
            "FullyQualifiedName~AD|FullyQualifiedName~AutoDiff|FullyQualifiedName~Callable|FullyQualifiedName~TypedIr",
        ],
    ),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_source_path(filename: str, root: Path) -> str:
    path = Path(filename)
    try:
        return path.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return filename.replace("\\", "/")


def scope_for(path: str) -> ScopeEntry | None:
    return SCOPE_BY_PATH.get(path)


def run_tests(root: Path) -> None:
    if RESULTS_DIR.exists():
        shutil.rmtree(root / RESULTS_DIR)
    (root / RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    for name, base_command in TEST_SLICES:
        result_dir = RESULTS_DIR / name
        command = [
            *base_command,
            "--collect:XPlat Code Coverage",
            "--results-directory",
            str(result_dir),
            "--",
            "DataCollectionRunSettings.DataCollectors.DataCollector.Configuration.Format=cobertura",
        ]
        print(f"\n==> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=root, check=True)


def parse_branch_counts(line: ET.Element) -> tuple[int, int]:
    if line.get("branch", "").lower() != "true":
        return 0, 0

    coverage = line.get("condition-coverage", "")
    match = re.search(r"\((\d+)/(\d+)\)", coverage)
    if match:
        return int(match.group(1)), int(match.group(2))

    conditions = line.find("conditions")
    if conditions is None:
        return 0, 0

    covered = 0
    total = 0
    for condition in conditions.findall("condition"):
        total += 1
        percent = condition.get("coverage", "0").rstrip("%")
        try:
            if float(percent) > 0:
                covered += 1
        except ValueError:
            pass
    return covered, total


def collect_coverage(root: Path) -> tuple[dict[str, dict[str, int]], list[Path]]:
    reports = sorted((root / RESULTS_DIR).glob("**/coverage.cobertura.xml"))
    if not reports:
        raise SystemExit(f"No Cobertura reports found under {RESULTS_DIR}.")

    line_hits: dict[tuple[str, int], int] = {}
    branch_hits: dict[tuple[str, int], tuple[int, int]] = {}

    for report in reports:
        tree = ET.parse(report)
        for cls in tree.findall(".//class"):
            filename = cls.get("filename")
            if not filename:
                continue
            rel = normalize_source_path(filename, root)
            scope = scope_for(rel)
            if scope is None:
                continue

            for line in cls.findall("./lines/line"):
                number = int(line.get("number", "0"))
                if number <= 0 or not scope.includes(number):
                    continue

                line_key = (scope.path, number)
                if line_key in LINE_EXCLUSIONS:
                    continue

                line_hits[line_key] = max(line_hits.get(line_key, 0), int(line.get("hits", "0")))

                covered_branches, branches = parse_branch_counts(line)
                if branches > 0 and line_key not in BRANCH_EXCLUSIONS:
                    previous_covered, previous_total = branch_hits.get(line_key, (0, 0))
                    branch_hits[line_key] = (max(previous_covered, covered_branches), max(previous_total, branches))

    totals: dict[str, dict[str, int]] = {}
    for (path, _line), hits in line_hits.items():
        stats = totals.setdefault(path, {"lines": 0, "covered_lines": 0, "branches": 0, "covered_branches": 0})
        stats["lines"] += 1
        if hits > 0:
            stats["covered_lines"] += 1

    for (path, _line), (covered, total) in branch_hits.items():
        stats = totals.setdefault(path, {"lines": 0, "covered_lines": 0, "branches": 0, "covered_branches": 0})
        stats["branches"] += total
        stats["covered_branches"] += covered

    return totals, reports


def percent(covered: int, total: int) -> float:
    if total == 0:
        return 100.0
    return covered * 100.0 / total


def print_report(totals: dict[str, dict[str, int]], reports: list[Path]) -> tuple[float, float, list[str]]:
    print("\nAD managed coverage reports:")
    for report in reports:
        print(f"  - {report}")

    print("\nManaged AD scope:")
    for entry in MANAGED_SCOPE:
        range_text = "whole file" if not entry.ranges else ", ".join(f"{start}-{end}" for start, end in entry.ranges)
        print(f"  - {entry.path} ({range_text}): {entry.reason}")

    print("\nNarrow managed line exclusions:")
    for (path, line), reason in sorted(LINE_EXCLUSIONS.items()):
        print(f"  - {path}:{line}: {reason}")

    print("\nNarrow managed branch exclusions:")
    for (path, line), reason in sorted(BRANCH_EXCLUSIONS.items()):
        print(f"  - {path}:{line}: {reason}")

    total_lines = sum(stats["lines"] for stats in totals.values())
    covered_lines = sum(stats["covered_lines"] for stats in totals.values())
    total_branches = sum(stats["branches"] for stats in totals.values())
    covered_branches = sum(stats["covered_branches"] for stats in totals.values())

    per_file_failures: list[str] = []
    print(f"\nPer-file managed AD coverage (threshold: lines >= {PER_FILE_LINE_THRESHOLD:.0f}%):")
    for path in sorted(totals):
        stats = totals[path]
        line_pct = percent(stats["covered_lines"], stats["lines"])
        branch_pct = percent(stats["covered_branches"], stats["branches"])
        print(
            f"  - {path}: lines {line_pct:5.1f}% "
            f"({stats['covered_lines']}/{stats['lines']}), "
            f"branches {branch_pct:5.1f}% ({stats['covered_branches']}/{stats['branches']})"
        )
        if line_pct < PER_FILE_LINE_THRESHOLD:
            per_file_failures.append(
                f"{path} line coverage {line_pct:.2f}% < {PER_FILE_LINE_THRESHOLD:.0f}% "
                f"({stats['covered_lines']}/{stats['lines']})"
            )

    line_pct = percent(covered_lines, total_lines)
    branch_pct = percent(covered_branches, total_branches)
    print(
        f"\nAD managed scoped coverage: lines {line_pct:.2f}% "
        f"({covered_lines}/{total_lines}), branches {branch_pct:.2f}% "
        f"({covered_branches}/{total_branches})"
    )
    return line_pct, branch_pct, per_file_failures


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-test", action="store_true", help="reuse existing Cobertura reports")
    args = parser.parse_args(argv)

    root = repo_root()
    os.chdir(root)

    if not args.no_test:
        run_tests(root)

    totals, reports = collect_coverage(root)
    line_pct, branch_pct, per_file_failures = print_report(totals, reports)
    scoped_lines = sum(stats["lines"] for stats in totals.values())
    scoped_branches = sum(stats["branches"] for stats in totals.values())
    if scoped_lines == 0:
        print("\nAD coverage gate failed: no managed AD-scoped lines were matched.", file=sys.stderr)
        return 1

    failures = []
    if line_pct < LINE_THRESHOLD:
        failures.append(f"line coverage {line_pct:.2f}% < {LINE_THRESHOLD:.0f}%")
    if scoped_branches == 0:
        failures.append("branch coverage has no measurable scoped branch counters")
    if branch_pct < BRANCH_THRESHOLD:
        failures.append(f"branch coverage {branch_pct:.2f}% < {BRANCH_THRESHOLD:.0f}%")
    failures.extend(per_file_failures)

    if failures:
        print("\nAD coverage gate failed: " + "; ".join(failures), file=sys.stderr)
        return 1

    print("\nAD coverage gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
