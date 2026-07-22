"""Generate D1/D2 coverage reports for the V2 dunun analysis.

The important methodological rule is simple: D1 is the complete comparable
voice; D2 is partial and should only be compared on paired takes.
"""
from __future__ import annotations

from pathlib import Path

import utils_V2 as u


def build_reports(out_dir: str = "output") -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    coverage = u.dunun_coverage()
    metrics = u.dunun_pair_metrics()

    coverage_path = out / "d2_coverage_V2.csv"
    metrics_path = out / "d2_pair_metrics_V2.csv"
    coverage.to_csv(coverage_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    return coverage_path, metrics_path


def main() -> None:
    coverage_path, metrics_path = build_reports()
    print(f"Coverage -> {coverage_path}")
    print(f"Pair metrics -> {metrics_path}")
    print()
    print("Prises appariées D1/D2 :")
    print(u.paired_dunun_takes()[["piece", "take"]].to_string(index=False))


if __name__ == "__main__":
    main()
