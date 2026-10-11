"""Audio helpers for the demo notebook. Everything is computed in the user's
session from WAVs downloaded from OSF; nothing here writes audio to disk.

- ``ensemble_excerpt``: the multitrack recording of one take, mixed to mono.
- ``dundun_against_grid``: the D1 track with a click on an even 12-pulse grid
  built from the annotated cycle starts, so the offsets in the figures can be
  heard (strokes landing before or after the click).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import djembe_dataset as d

SR_OUT = 22050


def _track(corpus_dir, take_id: str, voice: str) -> tuple[np.ndarray, int] | None:
    piece = take_id.split("_")[0]
    path = Path(corpus_dir) / f"MJ_{piece}" / f"MJ_{take_id}_{voice}.wav"
    if not path.exists():
        return None
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def _slice_resample(y: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    seg = y[int(start_s * sr): int((start_s + dur_s) * sr)]
    return resample_poly(seg, SR_OUT, sr).astype(np.float32) if sr != SR_OUT else seg


def _normalise(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = np.max(np.abs(y)) or 1.0
    return y * (peak / m)


def ensemble_excerpt(corpus_dir, take_id: str, start_s: float = 30.0, dur_s: float = 15.0) -> np.ndarray:
    """Mono mix of every available track of a take."""
    parts = [t for v in d.INSTRUMENTS if (t := _track(corpus_dir, take_id, v)) is not None]
    if not parts:
        raise FileNotFoundError(f"no WAV for {take_id}: download it with audio_takes=['{take_id}']")
    segs = [_slice_resample(y, sr, start_s, dur_s) for y, sr in parts]
    n = min(len(s) for s in segs)
    return _normalise(sum(s[:n] for s in segs))


def _click(sr: int, freq: float, dur_s: float = 0.025) -> np.ndarray:
    t = np.arange(int(dur_s * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t * 180)).astype(np.float32)


def dundun_against_grid(corpus_dir, take_id: str, start_s: float = 30.0, dur_s: float = 15.0,
                        click_gain: float = 0.35) -> np.ndarray:
    """D1 track plus clicks on the isochronous 12-pulse grid of each annotated cycle.

    Cycle downbeats get a higher click. The grid comes from the metre file, the
    same reference the offsets in the event table are measured against.
    """
    track = _track(corpus_dir, take_id, "D1")
    if track is None:
        raise FileNotFoundError(f"no D1 WAV for {take_id}")
    y = _normalise(_slice_resample(*track, start_s, dur_s), 0.8)
    piece = take_id.split("_")[0]
    metre = pd.read_csv(Path(corpus_dir) / f"MJ_{piece}" / "annotations" / f"MJ_{take_id}_Metre.csv")
    starts = metre["Time"].to_numpy()
    hi, lo = _click(SR_OUT, 1800), _click(SR_OUT, 1100)
    for a, b in zip(starts[:-1], starts[1:]):
        for pulse in range(d.N_SUBDIVISIONS):
            t = a + pulse * (b - a) / d.N_SUBDIVISIONS - start_s
            if 0 <= t < dur_s:
                i = int(t * SR_OUT)
                c = hi if pulse == 0 else lo * 0.6
                j = min(len(y), i + len(c))
                y[i:j] += click_gain * c[: j - i]
    return _normalise(y)
