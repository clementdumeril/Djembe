"""Smoke checks for the DJEMBE_LLM V2 workspace.

This script deliberately stays light: it validates imports, key files, the main
CSV schema when generated, and the D1/D2 coverage rules. It does not download
the corpus or rerun the full notebooks.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import utils_V2 as u


ROOT = Path(__file__).resolve().parent
MAIN_CSV = ROOT / "output" / "dunun_frappes_V2.csv"
REQUIRED_COLUMNS = {
    "time",
    "centroid",
    "low_ratio",
    "flatness",
    "zcr",
    "rms",
    "SD",
    "rel",
    "dur",
    "piece",
    "take",
    "code",
}


def _ok(label: str) -> None:
    print(f"OK  {label}")


def _fail(label: str) -> None:
    raise SystemExit(f"FAIL {label}")


def check_source_files() -> None:
    for name in [
        "README.md",
        "requirements.txt",
        "utils_V2.py",
        "dunun_viz_V2.py",
        "atome_vivant_V2.py",
        "boussole_rythmes_V2.py",
    ]:
        if not (ROOT / name).exists():
            _fail(f"missing {name}")
    _ok("source files present")


def check_d2_coverage() -> None:
    cov = u.dunun_coverage()
    if cov.empty:
        print("SKIP D2 coverage: corpus/ is not present locally")
        return
    paired = u.paired_dunun_takes()
    if paired.empty:
        _fail("no paired D1/D2 takes found")
    print("D1/D2 paired takes:")
    print(paired[["piece", "take"]].to_string(index=False))
    _ok("D2 treated as partial coverage")


def check_main_csv() -> None:
    if not MAIN_CSV.exists():
        print("SKIP main CSV: run 02_dunun_extraction_V2.ipynb to generate it")
        return

    df = pd.read_csv(MAIN_CSV)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        _fail(f"{MAIN_CSV} missing columns: {sorted(missing)}")
    if df.empty:
        _fail(f"{MAIN_CSV} is empty")
    if df[["time", "SD", "rel", "dur", "piece", "take", "code"]].isna().any().any():
        _fail(f"{MAIN_CSV} has missing values in core columns")
    if not set(df["code"]).issubset({"D1", "D2"}):
        _fail(f"{MAIN_CSV} contains unexpected dunun code values")
    _ok(f"main CSV schema valid ({len(df)} rows)")


def main() -> None:
    check_source_files()
    check_d2_coverage()
    check_main_csv()


if __name__ == "__main__":
    main()
