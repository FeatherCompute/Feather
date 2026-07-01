#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def normalize_target(raw: str) -> str | None:
    target = raw.strip()
    if not target or target.startswith("#"):
        return None
    if any(ch in target for ch in [" ", "\t", "\n"]):
        return None
    if "://" in target or target.startswith("mailto:"):
        return None
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split("#", 1)[0]
    if not target:
        return None
    return target


def check_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    errors: list[str] = []
    for match in [*LINK_RE.finditer(text), *IMAGE_RE.finditer(text)]:
        target = normalize_target(match.group(1))
        if target is None:
            continue
        if target.startswith("/"):
            errors.append(f"{path.relative_to(ROOT)}: absolute local link is not portable: {target}")
            continue
        resolved = (path.parent / target).resolve()
        try:
            resolved.relative_to(ROOT)
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)}: link escapes repository: {target}")
            continue
        if not resolved.exists():
            errors.append(f"{path.relative_to(ROOT)}: missing link target: {target}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {"EasyGPU", "bin", "obj", "artifacts", ".git", ".idea", ".VSCodeCounter"} for part in path.parts):
            continue
        errors.extend(check_file(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    print("Markdown links OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
