#!/usr/bin/env python3
"""Run the full Feather industrial AD validation gate."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StepResult:
    name: str
    passed: bool
    output: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_step(name: str, command: list[str], root: Path) -> StepResult:
    print(f"\n==> {name}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    return StepResult(name, result.returncode == 0, result.stdout)


def parse_test_count(output: str) -> str:
    matches = re.findall(r"Passed!\s+-\s+Failed:\s+\d+,\s+Passed:\s+(\d+),\s+Skipped:\s+(\d+),\s+Total:\s+(\d+)", output)
    if not matches:
        return "test counts unavailable"
    passed = sum(int(match[0]) for match in matches)
    skipped = sum(int(match[1]) for match in matches)
    total = sum(int(match[2]) for match in matches)
    return f"passed {passed}, skipped {skipped}, total {total}"


def parse_managed_coverage(output: str) -> str:
    match = re.search(r"AD managed scoped coverage: lines ([0-9.]+)% .* branches ([0-9.]+)%", output)
    if not match:
        return "managed coverage unavailable"
    return f"lines {match.group(1)}%, branches {match.group(2)}%"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    root = repo_root()
    os.chdir(root)

    steps = [
        (
            "managed AD tests",
            ["dotnet", "test", "tests/Feather.AD.Tests/Feather.AD.Tests.csproj", "--no-restore"],
        ),
        (
            "AD integration tests",
            [
                "dotnet",
                "test",
                "tests/Feather.Integration.Tests/Feather.Integration.Tests.csproj",
                "--no-restore",
                "--filter",
                "AutoDiff|AD",
            ],
        ),
        (
            "generator AD tests",
            [
                "dotnet",
                "test",
                "tests/Feather.Generator.Tests/Feather.Generator.Tests.csproj",
                "--no-restore",
                "--filter",
                "AD|AutoDiff|Callable|TypedIr",
            ],
        ),
        ("managed coverage gate", ["python3", "scripts/ad-coverage-gate.py"]),
        ("AD sample smoke", ["dotnet", "run", "--project", "samples/AdLinearRegression/AdLinearRegression.csproj"]),
    ]

    results: list[StepResult] = []
    for name, command in steps:
        result = run_step(name, command, root)
        results.append(result)
        if not result.passed:
            break

    print("\nIndustrial AD gate summary:")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"  - {result.name}: {status}")
        if result.name.endswith("tests"):
            print(f"    {parse_test_count(result.output)}")
        elif result.name == "managed coverage gate":
            print("    threshold: managed aggregate lines >= 90%, branches >= 90%, per-file lines >= 90%")
            print(f"    {parse_managed_coverage(result.output)}")

    if not all(result.passed for result in results) or len(results) != len(steps):
        print("\nIndustrial AD gate failed.", file=sys.stderr)
        return 1

    print("\nIndustrial AD gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
