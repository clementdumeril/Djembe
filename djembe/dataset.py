"""Single event table for the IEMP Malian Jembe corpus, plus frozen take-level splits.

One row = one annotated stroke: which take, which instrument and player, which
cycle and subdivision, when it was played, and how far it sits from the
isochronous grid. Every learning experiment in the project reads this table and
the folds in ``splits.json``; nothing else.

Source files (per take, from OSF node M652X):
- ``*_Onsets_Selected_Peak.csv``: one row per metric position (12 per cycle),
  with the onset of each instrument that plays there and its peak amplitude.
  We use this file rather than ``*_Onsets_Selected.csv`` because the OSF copy of
  ``MJ_Maraka_5_Onsets_Selected.csv`` actually contains Woloso_5 (checked against
  the raw dundun onsets: 99 % match with Woloso_5, 10 % with Maraka_5), and
  Woloso_5 has no ``_Selected`` file. The ``_Selected_Peak`` files are correct
  for all 15 takes.
- ``*_Metre.csv``: start time of each annotated cycle (the grid).
- ``*_Annotation.csv``: musical form sections and signal phrases.

Note on Woloso: its metre file labels half-cycles (``1.1``, ``1.2``...), each
holding 12 subdivisions. A "cycle" here is always one row of the metre file,
i.e. 12 subdivisions.

Known gaps, kept as they are: the last annotated cycle of each take has no end
time, so its strokes (43 in total) have NaN phase and offset; and 1,422 strokes
(3.5 %) sit more than half a pulse from their nominal subdivision.

Licence: the corpus is CC BY-NC-ND 4.0. The derived event table must not be
committed or redistributed; it is written to ``output/`` (git-ignored) or to a
Google Drive folder on Colab. ``splits.json`` holds only take identifiers.

Usage:
    python -m djembe.dataset download   # annotations only, from OSF
    python -m djembe.dataset download --audio --takes Woloso_2   # plus that take's WAVs
    python -m djembe.dataset build      # -> output/events.csv
    python -m djembe.dataset splits     # -> splits.json (deterministic)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OSF_NODE = "m652x"
PIECES = ("Manjanin", "Maraka", "Woloso")
INSTRUMENTS = ("J1", "J2", "D1", "D2")
PLAYER_COLUMN = {"J1": "Jembe 1", "J2": "Jembe 2", "D1": "Dundun 1", "D2": "Dundun 2"}
N_SUBDIVISIONS = 12

# The 15 recorded takes. Hard-coded so that splits.json never depends on what
# happens to be on disk.
TAKES = (
    [f"Manjanin_{t}" for t in range(1, 5)]
    + [f"Maraka_{t}" for t in range(1, 7)]
    + [f"Woloso_{t}" for t in range(1, 6)]
)
SPLITS_PATH = ROOT / "splits.json"
DEFAULT_CORPUS = ROOT / "corpus"
DEFAULT_EVENTS = ROOT / "output" / "events.csv"

ANNOTATION_PATTERN = re.compile(r"_(Onsets_Selected_Peak|Metre|Annotation|Onsets_Raw_\w+)\.csv$")


# ───────────────────────────── download ─────────────────────────────
def _osf_list(path: str) -> list[dict]:
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/dropbox/{path}/?page%5Bsize%5D=200"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["data"]


def download_corpus(corpus_dir: str | Path = DEFAULT_CORPUS, audio: bool = False, audio_takes=None,
                    verbose: bool = True) -> int:
    """Download annotations (and optionally the multitrack WAVs) from OSF.

    Idempotent: files already present are skipped. Returns the number of new
    files. Annotations are about 10 MB; all audio is about 900 MB. With
    ``audio_takes`` (e.g. ``["Woloso_2"]``), only those takes' WAVs are fetched.
    """
    corpus_dir = Path(corpus_dir)
    new = 0
    for piece in PIECES:
        target = corpus_dir / f"MJ_{piece}" / "annotations"
        target.mkdir(parents=True, exist_ok=True)
        listings = [("Annotations", target, ANNOTATION_PATTERN)]
        if audio:
            listings.append(("Media", target.parent, re.compile(r"_[JD][12]\.wav$")))
        for folder, dest_dir, pattern in listings:
            for item in _osf_list(f"{folder}/MJ_{piece}"):
                name = item["attributes"]["name"]
                if item["attributes"]["kind"] != "file" or not pattern.search(name):
                    continue
                if folder == "Media" and audio_takes and not any(name.startswith(f"MJ_{t}_") for t in audio_takes):
                    continue
                dest = dest_dir / name
                if dest.exists():
                    continue
                request = urllib.request.Request(item["links"]["download"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=300) as response:
                    dest.write_bytes(response.read())
                new += 1
        if verbose:
            print(f"{piece:9s} ok")
    if verbose:
        print(f"{new} new files in {corpus_dir}")
    return new


# ───────────────────────────── build ─────────────────────────────
def _annotation(corpus_dir: Path, take_id: str, suffix: str) -> Path:
    piece, _ = take_id.split("_")
    return corpus_dir / f"MJ_{piece}" / "annotations" / f"MJ_{take_id}_{suffix}.csv"


def _sections(path: Path) -> tuple[list[tuple[float, float, str]], list[tuple[float, float]]]:
    if not path.exists():
        return [], []
    table = pd.read_csv(path, header=None, names=["kind", "start", "end", "dur", "label"])
    form = table[table["kind"] == "MUSIC_FORM"]
    signal = table[table["kind"] == "SIGNAL_PHRASES"]
    return (
        list(zip(form["start"], form["end"], form["label"].str.strip())),
        list(zip(signal["start"], signal["end"])),
    )


def load_take(take_id: str, corpus_dir: str | Path = DEFAULT_CORPUS) -> pd.DataFrame:
    """Event rows for one take (see module docstring for the columns)."""
    corpus_dir = Path(corpus_dir)
    piece, take = take_id.split("_")
    positions = pd.read_csv(_annotation(corpus_dir, take_id, "Onsets_Selected_Peak"), dtype={"Label SD": str})
    labels = set(positions["Take"].unique())
    if labels != {take_id}:
        raise ValueError(f"{take_id}: file is labelled {sorted(labels)}")

    metre = pd.read_csv(_annotation(corpus_dir, take_id, "Metre"), dtype={"Cycle": str})
    metre["cycle_dur_s"] = metre["Time"].shift(-1) - metre["Time"]
    cycle_start = dict(zip(metre["Cycle"], metre["Time"]))
    cycle_dur = dict(zip(metre["Cycle"], metre["cycle_dur_s"]))
    form, signal = _sections(_annotation(corpus_dir, take_id, "Annotation"))

    positions["cycle_label"] = positions["Label SD"].str.split("|").str[0]
    order = {label: i for i, label in enumerate(dict.fromkeys(positions["cycle_label"]))}

    rows = []
    for instrument in INSTRUMENTS:
        if instrument not in positions or positions[instrument].notna().sum() == 0:
            continue
        hits = positions[positions[instrument].notna()]
        part = pd.DataFrame({
            "take_id": take_id,
            "piece": piece,
            "take": int(take),
            "n_players": hits["N"].astype(int),
            "cycle_label": hits["cycle_label"],
            "cycle_idx": hits["cycle_label"].map(order),
            "sd": hits["SD"].astype(int),
            "instrument": instrument,
            "player": hits[PLAYER_COLUMN[instrument]],
            "onset_s": hits[instrument].astype(float),
            "peak": hits.get(f"{instrument} Peak"),
            "dens_raw": hits.get(f"{instrument} Dens"),
        })
        rows.append(part)
    events = pd.concat(rows, ignore_index=True)

    events["cycle_start_s"] = events["cycle_label"].map(cycle_start)
    events["cycle_dur_s"] = events["cycle_label"].map(cycle_dur)
    events["phase"] = (events["onset_s"] - events["cycle_start_s"]) / events["cycle_dur_s"]
    events["dev_cycle"] = events["phase"] - (events["sd"] - 1) / N_SUBDIVISIONS
    events["dev_sd"] = events["dev_cycle"] * N_SUBDIVISIONS
    events["dev_ms"] = events["dev_cycle"] * events["cycle_dur_s"] * 1000.0

    onset = events["onset_s"].to_numpy()
    section = np.full(len(events), None, dtype=object)
    for start, end, label in form:
        section[(onset >= start) & (onset <= end)] = label
    events["section"] = section
    in_signal = np.zeros(len(events), dtype=bool)
    for start, end in signal:
        in_signal |= (onset >= start) & (onset <= end)
    events["in_signal"] = in_signal

    return events.sort_values(["onset_s", "instrument"]).reset_index(drop=True)


def build_events(corpus_dir: str | Path = DEFAULT_CORPUS, takes=TAKES) -> pd.DataFrame:
    """Event table for all takes, in a fixed order."""
    return pd.concat([load_take(t, corpus_dir) for t in takes], ignore_index=True)


def load_events(path: str | Path = DEFAULT_EVENTS) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"cycle_label": str})


# ───────────────────────────── splits ─────────────────────────────
def make_splits(takes=TAKES) -> dict:
    """Leave-one-take-out folds. The take is the only honest grouping unit:
    the same dundun player (Madu Jakite) plays D1 on every take."""
    takes = list(takes)
    return {
        "scheme": "leave-one-take-out",
        "group_key": "take_id",
        "note": "Same D1 player on all takes: folds test generalisation to an unseen take, not to an unseen musician.",
        "takes": takes,
        "folds": [
            {"fold": i, "test": [t], "train": [x for x in takes if x != t]}
            for i, t in enumerate(takes)
        ],
    }


def load_splits(path: str | Path = SPLITS_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, key: str = "take_id") -> None:
    """Fail loudly if any take appears on both sides of a split."""
    shared = set(train[key]) & set(test[key])
    if shared:
        raise AssertionError(f"takes in both train and test: {sorted(shared)}")


def iter_folds(frame: pd.DataFrame, splits: dict | None = None, key: str = "take_id"):
    """Yield (fold, train_frame, test_frame), checked for leakage."""
    splits = splits or load_splits()
    for fold in splits["folds"]:
        train = frame[frame[key].isin(fold["train"])]
        test = frame[frame[key].isin(fold["test"])]
        assert_no_leakage(train, test, key)
        yield fold["fold"], train, test


# ───────────────────────────── features ─────────────────────────────
def subdivision_profile(events: pd.DataFrame, instrument: str = "D1") -> pd.DataFrame:
    """Per take: probability that `instrument` plays each of the 12 subdivisions.

    The denominator is the number of annotated cycles in the take (all
    instruments), so cycles where the instrument is silent count as zeros.
    """
    n_cycles = events.groupby("take_id")["cycle_label"].nunique()
    hits = events[events["instrument"] == instrument]
    counts = hits.groupby(["take_id", "sd"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, N_SUBDIVISIONS + 1), fill_value=0)
    profile = counts.div(n_cycles.reindex(counts.index), axis=0)
    profile.columns = [f"sd{c}" for c in profile.columns]
    piece = events.drop_duplicates("take_id").set_index("take_id")["piece"]
    return profile.assign(piece=piece.reindex(profile.index)).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["download", "build", "splits"])
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--out", default=str(DEFAULT_EVENTS))
    parser.add_argument("--audio", action="store_true", help="also download the WAVs (about 900 MB)")
    parser.add_argument("--takes", nargs="+", help="with --audio: only these takes, e.g. Woloso_2 Maraka_6")
    args = parser.parse_args()

    if args.command == "download":
        download_corpus(args.corpus, audio=args.audio, audio_takes=args.takes)
    elif args.command == "build":
        events = build_events(args.corpus)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.out, index=False)
        summary = events.groupby(["piece", "instrument"]).size().unstack(fill_value=0)
        print(f"{len(events)} events, {events['take_id'].nunique()} takes -> {args.out}")
        print(summary.to_string())
    else:
        with open(SPLITS_PATH, "w", encoding="utf-8") as handle:
            json.dump(make_splits(), handle, indent=2)
            handle.write("\n")
        print(f"{len(TAKES)} folds -> {SPLITS_PATH.name}")


if __name__ == "__main__":
    main()
