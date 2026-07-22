"""Prepare the repository before each handoff.

Run this before committing or publishing. It keeps the ritual boring and safe:
source files present, Python syntax OK, D2 coverage explicit, heavy generated
assets ignored, and Git status visible.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PY_FILES = [
    "utils_V2.py",
    "dunun_viz_V2.py",
    "atome_vivant_V2.py",
    "explain_fourier_V2.py",
    "check_project.py",
    "qc_d2_coverage.py",
]
HEAVY_PATHS = [
    "corpus",
    "output",
    "archive/youtube",
]


def run(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(args))
    result = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def main() -> None:
    run(["python", "check_project.py"])
    run(["python", "-m", "py_compile", *PY_FILES])

    print("\nIgnored heavy paths:")
    for path in HEAVY_PATHS:
        result = run(["git", "check-ignore", "-q", path], check=False)
        status = "OK ignored" if result.returncode == 0 else "WARN not ignored"
        print(f"- {path}: {status}")

    print("\nGit status:")
    run(["git", "status", "--short", "--branch"], check=False)


if __name__ == "__main__":
    main()
