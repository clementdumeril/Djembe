"""Stroke type (bass, tone, slap) of jembe strokes, from the isolated audio track.

The corpus annotates when each drum strikes, not how. Each jembe has its own
microphone, so a short window of its track around every annotated onset shows the
stroke's spectrum:

- bass: centre of the skin, a quarter of the energy below 150 Hz (the drum's air
  resonance, roughly 70-100 Hz);
- tone: edge, flat fingers, a ringing note: the longest decay and the most energy
  above 700 Hz (centroid around 750 Hz);
- slap: edge, fingers whipping the skin: loud and short, energy packed in 150-700 Hz.

(Measured on this corpus' microphones; checked by ear, see below.)

Per stroke: energy shares below 150 Hz, 150-700 Hz and above 2 kHz, spectral centroid
and flatness, and the decay (energy after 30 ms over energy before). Features are
log-transformed and standardised per take (microphones and levels differ between
takes), then a 3-component Gaussian mixture groups the strokes; the groups are named
from their acoustics (highest share below 150 Hz: bass; of the other two, the one that
rings shorter: slap; the remaining one: tone). Checked by ear on 105 soloist strokes
(3 passages): 75 % agreement (bass 14/19, slap 27/35, tone 38/51). Strokes within 30 ms
of a dundun onset are flagged (``bleed``): the dundun leaks into the jembe microphone
and can pass for a bass.

The labels are derived from the corpus audio (CC BY-NC-ND): they stay in ``output/``
and are never published.

    python -m djembe.strokes --instrument J1      # -> output/strokes_J1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from djembe import dataset as d

PRE, POST = 0.005, 0.090                 # window around the stroke [s]
SEARCH = (-0.010, 0.015)                 # where to look for the amplitude peak around the annotated onset
BLEED_S = 0.030
FEATURES = ["low", "mid", "high", "centroid", "flatness", "decay"]


def _window_features(y: np.ndarray, sr: int, onset: float) -> dict:
    a, b = int((onset + SEARCH[0]) * sr), int((onset + SEARCH[1]) * sr)
    a = max(a, 0)
    peak = a + int(np.argmax(np.abs(y[a:b]))) if b > a else int(onset * sr)
    s, e = max(peak - int(PRE * sr), 0), peak + int(POST * sr)
    w = y[s:e]
    if len(w) < int(0.05 * sr):
        return {k: np.nan for k in FEATURES}
    spec = np.abs(np.fft.rfft(w * np.hanning(len(w)))) ** 2
    f = np.fft.rfftfreq(len(w), 1 / sr)
    tot = spec.sum() + 1e-12
    split = int(0.030 * sr) + int(PRE * sr)
    early, late = (w[:split] ** 2).sum() + 1e-12, (w[split:] ** 2).sum() + 1e-12
    return {"low": spec[f < 150].sum() / tot, "mid": spec[(f >= 150) & (f < 700)].sum() / tot,
            "high": spec[f >= 2000].sum() / tot, "centroid": (spec * f).sum() / tot,
            "flatness": np.exp(np.log(spec + 1e-12).mean()) / (spec.mean() + 1e-12), "decay": late / early}


def take_features(events: pd.DataFrame, take: str, instrument: str = "J1", corpus=d.DEFAULT_CORPUS) -> pd.DataFrame:
    import soundfile as sf

    piece = take.split("_")[0]
    y, sr = sf.read(Path(corpus) / f"MJ_{piece}" / f"MJ_{take}_{instrument}.wav", dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    ev = events[events["take_id"] == take]
    strokes = ev[ev["instrument"] == instrument][["take_id", "piece", "cycle_idx", "sd", "onset_s", "peak"]].copy()
    dundun = np.sort(ev[ev["instrument"].isin(["D1", "D2"])]["onset_s"].to_numpy())
    feats = [_window_features(y, sr, t) for t in strokes["onset_s"]]
    out = pd.concat([strokes.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    if len(dundun):
        idx = np.clip(np.searchsorted(dundun, out["onset_s"]), 1, len(dundun) - 1)
        near = np.minimum(np.abs(dundun[idx] - out["onset_s"]), np.abs(dundun[idx - 1] - out["onset_s"]))
        out["bleed"] = near <= BLEED_S
    else:
        out["bleed"] = False
    return out


def classify(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Add ``stroke`` (bass / tone / slap) and ``confidence`` (mixture posterior) to a feature table."""
    from sklearn.mixture import GaussianMixture

    df = df.dropna(subset=FEATURES).copy()
    z = np.log(df[FEATURES].clip(lower=1e-9))
    z = z.groupby(df["take_id"]).transform(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    fit_rows = ~df["bleed"].to_numpy()
    gmm = GaussianMixture(3, covariance_type="full", n_init=5, random_state=seed).fit(z[fit_rows])
    post = gmm.predict_proba(z)
    means = pd.DataFrame(gmm.means_, columns=FEATURES)
    bass = int(means["low"].idxmax())
    rest = [c for c in range(3) if c != bass]
    # Slap = the shorter-ringing group. Naming it the brighter group instead swapped tone and
    # slap: checked by ear on 105 strokes (3 passages), slaps are short, loud and mid-range,
    # tones ring longer with more energy above 700 Hz.
    slap = min(rest, key=lambda c: means.loc[c, "decay"])
    names = {bass: "bass", slap: "slap", next(c for c in rest if c != slap): "tone"}
    df["stroke"] = [names[c] for c in post.argmax(1)]
    df["confidence"] = post.max(1)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Stroke types of one jembe from its isolated track.")
    ap.add_argument("--instrument", default="J1", choices=["J1", "J2"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    events = d.load_events()
    takes = [t for t in d.TAKES if (Path(d.DEFAULT_CORPUS) / f"MJ_{t.split('_')[0]}" / f"MJ_{t}_{args.instrument}.wav").exists()]
    table = classify(pd.concat([take_features(events, t, args.instrument) for t in takes], ignore_index=True))
    out = Path(args.out or d.ROOT / "output" / f"strokes_{args.instrument}.csv")
    table.to_csv(out, index=False)
    print(table["stroke"].value_counts().to_string())
    print(table.groupby("stroke")[FEATURES].median().round(3).to_string())
    print(f"bleed-flagged: {table['bleed'].mean():.1%}   mean confidence: {table['confidence'].mean():.2f}")
    print("->", out)


if __name__ == "__main__":
    main()
