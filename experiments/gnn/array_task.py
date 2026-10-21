"""Map a SLURM array index to one unit of train_humanize.py or train_curve.py.

Index i covers take TAKES[i % 15], then seed SEEDS[(i // 15) % S], then, with
--n-train, training size N[i // (15 S)]. An array of 15 x S (x len(N)) tasks
runs every combination exactly once.

    python experiments/gnn/array_task.py --count --seeds 0 1 2                    # -> 45
    python experiments/gnn/array_task.py --index 17 --seeds 0 1 2                 # -> --takes Manjanin_3 --seeds 1
    python experiments/gnn/array_task.py --count --seeds 0 1 2 --n-train 2 4 8 13 # -> 180
"""
from __future__ import annotations

import argparse

from djembe.dataset import TAKES


def task(index: int, seeds: list[int], sizes: list[int] | None = None):
    """(take, seed) or, with sizes, (take, seed, n_train)."""
    n = len(TAKES) * len(seeds) * (len(sizes) if sizes else 1)
    if not 0 <= index < n:
        raise IndexError(f"index {index} outside 0..{n - 1}")
    rest = index // len(TAKES)
    take, seed = TAKES[index % len(TAKES)], seeds[rest % len(seeds)]
    return (take, seed) if not sizes else (take, seed, sizes[rest // len(seeds)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--n-train", nargs="+", type=int, default=None, help="training sizes (train_curve.py)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--index", type=int)
    group.add_argument("--count", action="store_true", help="print the number of array tasks")
    args = ap.parse_args()
    if args.count:
        print(len(TAKES) * len(args.seeds) * (len(args.n_train) if args.n_train else 1))
    elif args.n_train:
        take, seed, n = task(args.index, args.seeds, args.n_train)
        print(f"--takes {take} --seeds {seed} --n-train {n}")
    else:
        take, seed = task(args.index, args.seeds)
        print(f"--takes {take} --seeds {seed}")


if __name__ == "__main__":
    main()
