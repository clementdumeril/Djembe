"""Render a cycle grid to audio (never with sounds from the corpus).

Each stroke is placed at ``start of cycle + (pulse + offset) / 12 x cycle duration``,
with the take's real cycle durations; intensity (z-scored log peak) sets the gain.

Two voices:
- ``"samples"`` (default when ``assets/samples`` is present): recorded CC0 one-shots,
  djembe strokes from one session (carlmartin, Freesound pack 9826) and Malian dunun
  (JIMMYJAMES112, pack 10586), see ``assets/samples/CREDITS.md``. The model does not
  predict the stroke type, so a jembe stroke is chosen by a fixed rule on its
  intensity: slap above ``SLAP_Z``, bass below ``BASS_Z``, tone in between. Dundun 1
  plays the dundunba sound, dundun 2 the sangban;
- ``"synth"``: a pitched decaying sine with a noise attack, higher and shorter for the
  jembes, lower and longer for the dundun (no file needed).

    python -m djembe.render output/j5_grids/Woloso_5_s0_GNN.npz output/Woloso_5_GNN.wav
    python -m djembe.render output/j5_grids/Woloso_5_s0_GNN.npz output/Woloso_5_GNN_synth.wav --voice synth
    python -m djembe.render output/j5b_grids/Woloso_5_s0_GNN_solo.npz output/Woloso_5_solo.wav --solo-emphasis 1.8
"""
from __future__ import annotations

import argparse
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 22050
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "assets" / "samples"
SLAP_Z, BASS_Z = 0.6, -0.6
PAN = {0: -0.35, 1: 0.35, 2: 0.0, 3: 0.1}
SAMPLE_FILES = {"bass": "djembe_bass_158966.mp3", "tone": "djembe_tone_158968.mp3",
                "slap": "djembe_slap_158969.mp3", "dundunba": "dundunba_170180.mp3", "sangban": "sangban_170177.mp3"}
SYNTH = {  # instrument: (pitch Hz, decay s, noise share)
    0: (420.0, 0.09, 0.45),
    1: (330.0, 0.10, 0.40),
    2: (95.0, 0.32, 0.10),
    3: (150.0, 0.25, 0.12),
}


def _synth_stroke(inst: int, gain: float, rng) -> np.ndarray:
    f0, decay, noise = SYNTH[inst]
    t = np.arange(int(SR * decay * 4)) / SR
    env = np.exp(-t / decay)
    pitch = f0 * (1 + 0.6 * np.exp(-t / 0.01))              # short downward chirp on the attack
    tone = np.sin(2 * np.pi * np.cumsum(pitch) / SR)
    burst = rng.standard_normal(t.size) * np.exp(-t / 0.008)
    return gain * env * ((1 - noise) * tone + noise * burst)


@lru_cache(maxsize=None)
def load_samples(directory: str = str(SAMPLES_DIR)) -> dict:
    """Mono float arrays at SR, peak-normalised, keyed by sound name."""
    import soundfile as sf
    from scipy.signal import resample_poly

    out = {}
    for name, fname in SAMPLE_FILES.items():
        y, sr = sf.read(Path(directory) / fname, always_2d=True)
        y = y.mean(axis=1)
        if sr != SR:
            g = np.gcd(sr, SR)
            y = resample_poly(y, SR // g, sr // g)
        out[name] = y / max(np.abs(y).max(), 1e-9)
    return out


def sample_name(inst: int, intensity: float) -> str:
    if inst == 2:
        return "dundunba"
    if inst == 3:
        return "sangban"
    return "slap" if intensity >= SLAP_Z else "bass" if intensity <= BASS_Z else "tone"


def render(x: np.ndarray, log_T: np.ndarray, present, start_cycle: int = 0, n_cycles: int | None = None,
           seed: int = 0, voice: str | None = None, solo_emphasis: float = 1.0) -> np.ndarray:
    """Stereo float array (samples, 2) for cycles [start_cycle, start_cycle + n_cycles).

    ``solo_emphasis`` > 1 puts the soloist (jembe 1) forward: that much louder and centred.
    """
    if voice is None:
        voice = "samples" if all((SAMPLES_DIR / f).exists() for f in SAMPLE_FILES.values()) else "synth"
    bank = load_samples() if voice == "samples" else None
    rng = np.random.default_rng(seed)
    stop = x.shape[0] if n_cycles is None else min(x.shape[0], start_cycle + n_cycles)
    T = np.exp(log_T[start_cycle:stop])
    starts = np.concatenate([[0.0], np.cumsum(T)[:-1]])
    out = np.zeros((int(SR * (T.sum() + 3.0)), 2))
    for c, (t0, dur) in enumerate(zip(starts, T)):
        for i in np.flatnonzero(present):
            for k in range(12):
                hit, inten, delta = x[start_cycle + c, k, i]
                if hit <= 0:
                    continue
                gain = float(np.exp(0.25 * np.clip(inten, -3, 3))) * 0.25 * (solo_emphasis if i == 0 else 1.0)
                pan = 0.0 if (i == 0 and solo_emphasis > 1) else PAN[i]
                s = gain * bank[sample_name(i, inten)] if bank else _synth_stroke(i, gain, rng)
                n0 = max(int((t0 + (k + delta) / 12 * dur) * SR), 0)
                n1 = min(n0 + s.size, out.shape[0])
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
    ap = argparse.ArgumentParser(description="Render a generated grid (.npz) to a WAV file.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--voice", choices=("samples", "synth"), default=None)
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--solo-emphasis", type=float, default=1.0, help="soloist gain factor (>1 also centres it)")
    args = ap.parse_args()
    z = np.load(args.src, allow_pickle=False)
    start = int(z["n_context"]) if "n_context" in z else 0
    write_wav(args.dst, render(z["x"], z["log_T"], z["present"], start_cycle=start, n_cycles=args.cycles,
                               voice=args.voice, solo_emphasis=args.solo_emphasis))
    print("->", args.dst)


if __name__ == "__main__":
    main()
