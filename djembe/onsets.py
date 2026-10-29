"""Detect every stroke on a drum's isolated track, including fast ornaments.

The corpus' raw onsets keep almost no two strokes closer than about 60 ms, and its
selected onsets keep one stroke per pulse, so fast phrases (flams, rolls, strokes
20-50 ms apart) collapse into their first stroke. This detector applies an explicit
rule to the audio itself:

1. onset strength: spectral flux, the summed increase of log-magnitude across
   frequencies between successive 2.5 ms frames (a stroke is a sudden rise of energy);
2. a frame is a stroke candidate if its strength is a local maximum (+/- 8 ms) and
   exceeds an adaptive threshold: the local median over +/- 40 ms plus ``k`` times the
   median absolute deviation of the whole track (loud passages raise the bar locally,
   quiet ones lower it);
3. the amplitude must actually rise: the envelope peak in the 15 ms after the candidate
   is at least ``rise_db`` above the envelope just before it, which rejects the
   decaying tail of the previous stroke and small bumps inside it;
4. two strokes are at least ``min_ioi`` apart (18 ms by default, below the fastest flams).

``compare_with`` measures agreement with reference onsets (e.g. the corpus' raw onsets).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter

HOP_S, NFFT = 0.0025, 1024


def onset_strength(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Spectral flux per frame and the frame period [s]."""
    hop = int(round(HOP_S * sr))
    n = 1 + (len(y) - NFFT) // hop
    idx = np.arange(NFFT)[None, :] + hop * np.arange(n)[:, None]
    frames = y[idx] * np.hanning(NFFT)[None, :]
    mag = np.log1p(1000 * np.abs(np.fft.rfft(frames, axis=1)))
    flux = np.concatenate([[0.0], np.maximum(np.diff(mag, axis=0), 0).sum(axis=1)])
    return flux, hop / sr


def envelope(y: np.ndarray, sr: int, win_s: float = 0.003) -> np.ndarray:
    w = max(int(win_s * sr), 1)
    return np.sqrt(np.convolve(y ** 2, np.ones(w) / w, mode="same")) + 1e-9


def detect(y: np.ndarray, sr: int, *, k: float = 3.0, rise_db: float = 3.0, min_ioi: float = 0.018) -> np.ndarray:
    """Stroke onset times [s] following the rule in the module docstring."""
    flux, dt = onset_strength(y, sr)
    local = median_filter(flux, size=int(0.080 / dt) | 1)
    mad = np.median(np.abs(flux - np.median(flux))) + 1e-9
    peak = (flux == maximum_filter1d(flux, size=int(0.016 / dt) | 1)) & (flux > local + k * mad)
    env = envelope(y, sr)
    times = []
    for f in np.flatnonzero(peak):
        t = f * dt + NFFT / (2 * sr)                     # frame centre
        i = int(t * sr)
        before = env[max(i - int(0.006 * sr), 0):max(i - int(0.002 * sr), 1)].mean()
        after = env[i:i + int(0.015 * sr)].max() if i < len(env) else 0.0
        if 20 * np.log10(after / before) < rise_db:
            continue
        if times and t - times[-1] < min_ioi:
            continue
        times.append(t)
    return np.array(times)


def compare_with(found: np.ndarray, reference: np.ndarray, tol: float = 0.015) -> dict:
    """Precision / recall of detected onsets against reference onsets (one-to-one within ``tol``)."""
    ref = np.sort(reference)
    used = np.zeros(len(ref), bool)
    hits = 0
    for t in np.sort(found):
        j = np.searchsorted(ref, t)
        cand = [c for c in (j - 1, j) if 0 <= c < len(ref) and not used[c] and abs(ref[c] - t) <= tol]
        if cand:
            used[min(cand, key=lambda c: abs(ref[c] - t))] = True
            hits += 1
    return {"precision": hits / max(len(found), 1), "recall": hits / max(len(ref), 1),
            "n_found": int(len(found)), "n_reference": int(len(ref))}
