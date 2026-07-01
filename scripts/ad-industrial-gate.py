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


def parse_native_coverage(output: str) -> str:
    match = re.search(r"Native AD coverage gate passed: scoped lines ([0-9.]+)% >= ([0-9.]+)%", output)
    if match:
        return f"lines {match.group(1)}% (threshold {match.group(2)}%)"

    total_line = None
    for line in output.splitlines():
        if line.strip().startswith("TOTAL"):
            parts = line.split()
            if len(parts) >= 4:
                total_line = parts[3]
    return f"lines {total_line}" if total_line else "native coverage unavailable"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-no-build", action="store_true", help="reuse existing native coverage build")
    parser.add_argument("--native-no-test", action="store_true", help="reuse existing native coverage profiles")
    parser.add_argument("--native-clean", action="store_true", help="clean native coverage build before running")
    args = parser.parse_args(argv)

    root = repo_root()
    os.chdir(root)

    native_command = ["python3", "scripts/ad-native-coverage-gate.py"]
    if args.native_no_build:
        native_command.append("--no-build")
    if args.native_no_test:
        native_command.append("--no-test")
    if args.native_clean:
        native_command.append("--clean")

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
        ("native coverage gate", native_command),
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
        elif result.name == "native coverage gate":
            print("    threshold: native aggregate scoped lines >= 80%, per-file scoped lines >= 80%")
            print(f"    {parse_native_coverage(result.output)}")

    if not all(result.passed for result in results) or len(results) != len(steps):
        print("\nIndustrial AD gate failed.", file=sys.stderr)
        return 1

    print("\nIndustrial AD gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
