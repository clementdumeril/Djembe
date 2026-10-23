"""Render a cycle grid to audio with synthetic drums (no sample from the corpus).

Each stroke is placed at ``start of cycle + (pulse + offset) / 12 x cycle duration``,
with the take's real cycle durations; intensity (z-scored log peak) sets the gain.
The sounds are plain synthesis: a pitched decaying sine with a noise attack, higher
and shorter for the jembes, lower and longer for the dundun.

    python -m djembe.render output/generated/Woloso_5_s0_GNN.npz output/generated/Woloso_5_s0_GNN.wav
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 22050
VOICES = {  # instrument: (pitch Hz, decay s, noise share, pan -1..1)
    0: (420.0, 0.09, 0.45, -0.35),
    1: (330.0, 0.10, 0.40, 0.35),
    2: (95.0, 0.32, 0.10, 0.0),
    3: (150.0, 0.25, 0.12, 0.0),
}


def _stroke(inst: int, gain: float, rng) -> np.ndarray:
    f0, decay, noise, _ = VOICES[inst]
    t = np.arange(int(SR * decay * 4)) / SR
    env = np.exp(-t / decay)
    pitch = f0 * (1 + 0.6 * np.exp(-t / 0.01))              # short downward chirp on the attack
    tone = np.sin(2 * np.pi * np.cumsum(pitch) / SR)
    burst = rng.standard_normal(t.size) * np.exp(-t / 0.008)
    return gain * env * ((1 - noise) * tone + noise * burst)


def render(x: np.ndarray, log_T: np.ndarray, present, start_cycle: int = 0, n_cycles: int | None = None,
           seed: int = 0) -> np.ndarray:
    """Stereo float array (samples, 2) for cycles [start_cycle, start_cycle + n_cycles)."""
    rng = np.random.default_rng(seed)
    stop = x.shape[0] if n_cycles is None else min(x.shape[0], start_cycle + n_cycles)
    T = np.exp(log_T[start_cycle:stop])
    starts = np.concatenate([[0.0], np.cumsum(T)[:-1]])
    out = np.zeros((int(SR * (T.sum() + 1.5)), 2))
    for c, (t0, dur) in enumerate(zip(starts, T)):
        for i in np.flatnonzero(present):
            for k in range(12):
                hit, inten, delta = x[start_cycle + c, k, i]
                if hit <= 0:
                    continue
                onset = t0 + (k + delta) / 12 * dur
                s = _stroke(i, float(np.exp(0.25 * np.clip(inten, -3, 3))) * 0.25, rng)
                n0 = max(int(onset * SR), 0)
                n1 = min(n0 + s.size, out.shape[0])
                pan = VOICES[i][3]
                out[n0:n1, 0] += s[:n1 - n0] * (1 - pan) / 2
                out[n0:n1, 1] += s[:n1 - n0] * (1 + pan) / 2
    return out / max(np.abs(out).max(), 1e-9) * 0.9


def write_wav(path, y: np.ndarray) -> None:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    z = np.load(src, allow_pickle=False)
    start = int(z["n_context"]) if "n_context" in z else 0
    write_wav(dst, render(z["x"], z["log_T"], z["present"], start_cycle=start, n_cycles=24))
    print("->", dst)


if __name__ == "__main__":
    main()
