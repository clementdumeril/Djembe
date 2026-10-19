"""Map a SLURM array index to one (test take, seed) unit of train_humanize.py.

Index i covers take TAKES[i % 15] and seed SEEDS[i // 15], so an array of
15 x len(seeds) tasks runs every fold of every seed exactly once.

    python experiments/gnn/array_task.py --count --seeds 0 1 2      # -> 45
    python experiments/gnn/array_task.py --index 17 --seeds 0 1 2   # -> --takes Manjanin_3 --seeds 1
"""
from __future__ import annotations

import argparse

from djembe.dataset import TAKES


def task(index: int, seeds: list[int]) -> tuple[str, int]:
    n = len(TAKES) * len(seeds)
    if not 0 <= index < n:
        raise IndexError(f"index {index} outside 0..{n - 1}")
    return TAKES[index % len(TAKES)], seeds[index // len(TAKES)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true", help="print the number of array tasks")
    args = ap.parse_args()
    if args.count:
        print(len(TAKES) * len(args.seeds))
    else:
        take, seed = task(args.index, args.seeds)
        print(f"--takes {take} --seeds {seed}")


if __name__ == "__main__":
    main()
