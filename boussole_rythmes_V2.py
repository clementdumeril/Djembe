"""Génère output/boussole_rythmes.html.

La Boussole des rythmes présente trois équations complexes, une par rythme.
Chaque équation combine les voix de jembé et de dunun disponibles sur les prises
appariées :

    R_piece(s) = [J1_piece(s), J2_piece(s), D1_piece(s), D2_piece(s)]

Les coefficients peuvent être lus en mode original ou modifiés en mode synthé.
L'objectif est de comparer les signatures complexes puis de sculpter une boucle
rythmique à partir des mêmes paramètres.

Pourquoi 6 coefficients ? Pour 12 subdivisions réelles, c7..c11 sont les
miroirs conjugués de c5..c1. Les ondes indépendantes non triviales sont donc
c1..c6, plus c0 qui porte la densité moyenne.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import utils_V2 as u


COLORS = {
    "Manjanin": "#355C9A",
    "Maraka": "#B46A2A",
    "Woloso": "#2E7D62",
    "J1": "#6B5CA5",
    "J2": "#8B6F47",
    "D1": "#214E8A",
    "D2": "#B46A2A",
}
ENSEMBLE_VOICES = ("J1", "J2", "D1", "D2")
DUNUN_VOICES = ("D1", "D2")
JEMBE_VOICES = ("J1", "J2")


def _paired_takes(piece: str) -> list[int]:
    paired = u.paired_dunun_takes()
    return paired.loc[paired["piece"] == piece, "take"].astype(int).tolist()


def _profile_from_takes(piece: str, code: str, takes: list[int]):
    vals = []
    used = []
    for take in sorted(set(takes)):
        profile = u.subdivision_profile(piece, int(take), code)
        if profile is not None:
            vals.append(profile.values.astype(float))
            used.append(int(take))

    if not vals:
        return np.zeros(12, dtype=float), []
    return np.mean(vals, axis=0), used


def _profile24_from_takes(piece: str, code: str, takes: list[int]):
    """Profil moyen sur deux cycles Polak successifs : A(12) + B(12)."""
    vals = []
    used = []
    for take in sorted(set(takes)):
        selected = u.load_selected(piece, int(take))
        if selected is None or code not in selected or not {"Label SD", "SD"}.issubset(selected.columns):
            continue

        selected = selected.copy()
        selected["cycle_id"] = (
            selected["Label SD"].astype(str).str.split("|", regex=False).str[0]
        )
        cycles = [cycle for _, cycle in selected.groupby("cycle_id", sort=False)]
        take_vals = []
        for idx in range(0, len(cycles) - 1, 2):
            phrase = np.zeros(24, dtype=float)
            for half, cycle in enumerate((cycles[idx], cycles[idx + 1])):
                for row in cycle.itertuples(index=False):
                    sd = int(getattr(row, "SD"))
                    if not 1 <= sd <= 12:
                        continue
                    value = getattr(row, code)
                    if pd.notna(value):
                        phrase[half * 12 + sd - 1] = 1.0
            if phrase.sum() > 0:
                take_vals.append(phrase)

        if take_vals:
            vals.extend(take_vals)
            used.append(int(take))

    if not vals:
        return np.zeros(24, dtype=float), []
    return np.mean(vals, axis=0), used


def _coefficients(motif: np.ndarray):
    """Coefficients Ck = Σ x_s exp(-i k θ_s), k=0..N/2, x normalisé."""
    motif = np.asarray(motif, dtype=float)
    n = int(len(motif))
    density = float(motif.sum())
    x = motif / density if density > 0 else motif.copy()
    coeffs = []
    for k in range(n // 2 + 1):
        z = sum(x[s] * np.exp(-2j * np.pi * k * s / n) for s in range(n))
        coeffs.append({
            "k": k,
            "re": round(float(z.real), 8),
            "im": round(float(z.imag), 8),
            "mag": round(float(abs(z)), 8),
            "phase": round(float(np.angle(z)), 8),
        })
    return coeffs, density


def _pca_map(rhythms):
    vectors = []
    for rhythm in rhythms:
        row = []
        for voice in ENSEMBLE_VOICES:
            coeffs = rhythm["voices"][voice]["coeffs"]
            row.extend(v for k in range(1, 7) for v in (coeffs[k]["re"], coeffs[k]["im"]))
        vectors.append(row)

    X = np.asarray(vectors, dtype=float)
    if len(X) == 0:
        return []

    X = X - X.mean(axis=0, keepdims=True)
    if len(X) == 1 or np.allclose(X, 0):
        xy = np.zeros((len(X), 2))
    else:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        xy = X @ vt[:2].T
        if xy.shape[1] == 1:
            xy = np.c_[xy[:, 0], np.zeros(len(xy))]

    scale = float(np.max(np.abs(xy))) or 1.0
    xy = xy / scale
    return [
        {"piece": rhythm["piece"], "x": round(float(x), 4), "y": round(float(y), 4)}
        for rhythm, (x, y) in zip(rhythms, xy)
    ]


def _microtiming_profiles(rhythms):
    """Résume l'écart à la grille et sa dispersion sur les voix de l'ensemble."""
    out = {}
    for rhythm in rhythms:
        piece = rhythm["piece"]
        rows_by_voice = {voice: [] for voice in ENSEMBLE_VOICES}
        for take in rhythm["takes"]:
            selected = u.load_selected(piece, int(take))
            if selected is None or "Label SD" not in selected:
                continue
            selected = selected.copy()
            selected["cycle_id"] = (
                selected["Label SD"].astype(str).str.split("|", regex=False).str[0]
            )
            time_cols = [c for c in ENSEMBLE_VOICES if c in selected.columns]
            if not time_cols:
                continue
            selected["reference"] = selected[time_cols].mean(axis=1, skipna=True)
            cycles = list(selected.groupby("cycle_id", sort=False))
            for idx in range(len(cycles) - 1):
                cycle = cycles[idx][1]
                next_cycle = cycles[idx + 1][1]
                start_values = cycle.loc[cycle["SD"] == 1, "reference"].dropna()
                next_values = next_cycle.loc[
                    next_cycle["SD"] == 1, "reference"
                ].dropna()
                if start_values.empty or next_values.empty:
                    continue
                start = float(start_values.iloc[0])
                duration = float(next_values.iloc[0]) - start
                if not 0.5 < duration < 5.0:
                    continue
                for row in cycle.itertuples(index=False):
                    sd = int(getattr(row, "SD"))
                    if not 1 <= sd <= 12:
                        continue
                    for voice in ENSEMBLE_VOICES:
                        if voice not in cycle.columns:
                            continue
                        value = getattr(row, voice)
                        if pd.isna(value):
                            continue
                        rel = (float(value) - start) / duration
                        if -0.05 <= rel < 1.05:
                            rows_by_voice[voice].append((sd, rel, duration))

        piece_data = {}
        for voice, rows in rows_by_voice.items():
            if not rows:
                continue
            s = pd.DataFrame(rows, columns=["SD", "rel", "dur"])

            s["dev_ms"] = (
                s["rel"] - (s["SD"].astype(float) - 1.0) / 12.0
            ) * s["dur"] * 1000.0
            mean_rel = s.groupby("SD")["rel"].transform("mean")
            precision = ((s["rel"] - mean_rel) * s["dur"] * 1000.0).std()
            grouped = s.groupby("SD")["dev_ms"]

            def value_at(series, subdivision, digits):
                value = series.get(subdivision, np.nan)
                return None if not np.isfinite(value) else round(float(value), digits)

            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.size()
            piece_data[voice] = {
                "mean": [value_at(means, sd, 2) for sd in range(1, 13)],
                "std": [value_at(stds, sd, 2) for sd in range(1, 13)],
                "n": [int(counts.get(sd, 0)) for sd in range(1, 13)],
                "precision": (
                    None if not np.isfinite(precision) else round(float(precision), 2)
                ),
            }
        if piece_data:
            out[piece] = piece_data
    return out


def _microtiming_phrase_profiles(rhythms):
    """Écart à la grille séparé sur deux cycles successifs A(12)+B(12)."""
    out = {}
    for rhythm in rhythms:
        piece = rhythm["piece"]
        rows_by_voice = {voice: [] for voice in ENSEMBLE_VOICES}
        for take in rhythm["takes"]:
            selected = u.load_selected(piece, int(take))
            if selected is None or "Label SD" not in selected:
                continue
            selected = selected.copy()
            selected["cycle_id"] = (
                selected["Label SD"].astype(str).str.split("|", regex=False).str[0]
            )
            time_cols = [c for c in ENSEMBLE_VOICES if c in selected.columns]
            if not time_cols:
                continue
            selected["reference"] = selected[time_cols].mean(axis=1, skipna=True)
            cycles = list(selected.groupby("cycle_id", sort=False))
            for idx in range(len(cycles) - 1):
                cycle = cycles[idx][1]
                next_cycle = cycles[idx + 1][1]
                start_values = cycle.loc[cycle["SD"] == 1, "reference"].dropna()
                next_values = next_cycle.loc[
                    next_cycle["SD"] == 1, "reference"
                ].dropna()
                if start_values.empty or next_values.empty:
                    continue
                start = float(start_values.iloc[0])
                duration = float(next_values.iloc[0]) - start
                if not 0.5 < duration < 5.0:
                    continue
                half = idx % 2
                for row in cycle.itertuples(index=False):
                    sd = int(getattr(row, "SD"))
                    if not 1 <= sd <= 12:
                        continue
                    phrase_sd = half * 12 + sd
                    for voice in ENSEMBLE_VOICES:
                        if voice not in cycle.columns:
                            continue
                        value = getattr(row, voice)
                        if pd.isna(value):
                            continue
                        rel = (float(value) - start) / duration
                        if -0.05 <= rel < 1.05:
                            dev_ms = (rel - (sd - 1.0) / 12.0) * duration * 1000.0
                            rows_by_voice[voice].append((phrase_sd, dev_ms))

        piece_data = {}
        for voice, rows in rows_by_voice.items():
            if not rows:
                continue
            s = pd.DataFrame(rows, columns=["SD", "dev_ms"])
            grouped = s.groupby("SD")["dev_ms"]

            def value_at(series, subdivision, digits):
                value = series.get(subdivision, np.nan)
                return None if not np.isfinite(value) else round(float(value), digits)

            means = grouped.mean()
            stds = grouped.std()
            counts = grouped.size()
            piece_data[voice] = {
                "mean": [value_at(means, sd, 2) for sd in range(1, 25)],
                "std": [value_at(stds, sd, 2) for sd in range(1, 25)],
                "n": [int(counts.get(sd, 0)) for sd in range(1, 25)],
                "precision": (
                    None
                    if s.empty or not np.isfinite(s["dev_ms"].std())
                    else round(float(s["dev_ms"].std()), 2)
                ),
            }
        if piece_data:
            out[piece] = piece_data
    return out


def _metric_timing_profiles(rhythms):
    """Positions métriques moyennes des 12 subdivisions, estimées sur l'ensemble."""
    out = {}
    for rhythm in rhythms:
        piece = rhythm["piece"]
        rows = []
        for take in rhythm["takes"]:
            selected = u.load_selected(piece, int(take))
            if selected is None or "Label SD" not in selected:
                continue
            selected = selected.copy()
            selected["cycle_id"] = (
                selected["Label SD"].astype(str).str.split("|", regex=False).str[0]
            )
            time_cols = [c for c in ENSEMBLE_VOICES if c in selected.columns]
            if not time_cols:
                continue
            selected["reference"] = selected[time_cols].mean(axis=1, skipna=True)
            cycles = list(selected.groupby("cycle_id", sort=False))
            for idx in range(len(cycles) - 1):
                cycle = cycles[idx][1]
                next_cycle = cycles[idx + 1][1]
                start_values = cycle.loc[
                    cycle["SD"] == 1, "reference"
                ].dropna()
                next_values = next_cycle.loc[
                    next_cycle["SD"] == 1, "reference"
                ].dropna()
                if start_values.empty or next_values.empty:
                    continue
                start = float(start_values.iloc[0])
                duration = float(next_values.iloc[0]) - start
                if not 0.5 < duration < 5.0:
                    continue
                for row in cycle.dropna(subset=["reference"]).itertuples(index=False):
                    relative = (float(row.reference) - start) / duration
                    if -0.02 <= relative < 1.02:
                        rows.append((int(row.SD), relative))

        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=["SD", "relative"])
        grouped = frame.groupby("SD")["relative"]
        means = grouped.mean().reindex(range(1, 13))
        counts = grouped.size().reindex(range(1, 13), fill_value=0)
        if means.isna().any():
            continue

        positions = means.to_numpy(dtype=float)
        positions[0] = 0.0
        positions = np.maximum.accumulate(positions)
        durations = np.diff(np.r_[positions, 1.0])
        durations = np.maximum(durations, 1e-5)
        durations = durations / durations.sum()
        positions = np.r_[0.0, np.cumsum(durations[:-1])]
        triplet = np.asarray(
            [durations[q::3].mean() for q in range(3)], dtype=float
        )
        triplet = triplet / triplet.sum()
        out[piece] = {
            "positions": [round(float(x), 6) for x in positions],
            "durations": [round(float(x), 6) for x in durations],
            "triplet": [round(float(x), 5) for x in triplet],
            "counts": [int(x) for x in counts],
        }
    return out


def _take_cycles(piece: str, take: int):
    """Cycles annotes, avec frappes D1/D2 exactes dans le temps de la prise."""
    selected = u.load_selected(piece, int(take))
    if selected is None or not {"Label SD", "SD"}.issubset(selected.columns):
        return []

    selected = selected.copy()
    time_cols = [c for c in ENSEMBLE_VOICES if c in selected.columns]
    if not time_cols:
        return []

    selected["cycle_id"] = (
        selected["Label SD"].astype(str).str.split("|", regex=False).str[0]
    )
    selected["reference"] = selected[time_cols].mean(axis=1, skipna=True)
    grouped_cycles = list(selected.groupby("cycle_id", sort=False))
    out = []

    for idx in range(len(grouped_cycles) - 1):
        label, cycle = grouped_cycles[idx]
        next_cycle = grouped_cycles[idx + 1][1]
        start_values = cycle.loc[cycle["SD"] == 1, "reference"].dropna()
        next_values = next_cycle.loc[next_cycle["SD"] == 1, "reference"].dropna()
        if start_values.empty or next_values.empty:
            continue

        start = float(start_values.iloc[0])
        end = float(next_values.iloc[0])
        duration = end - start
        if not 0.5 < duration < 5.0:
            continue

        positions = [None] * 12
        events = []
        for row in cycle.itertuples(index=False):
            sd = int(getattr(row, "SD"))
            if not 1 <= sd <= 12:
                continue

            ref = getattr(row, "reference")
            if pd.notna(ref):
                relative = (float(ref) - start) / duration
                positions[sd - 1] = min(0.999999, max(0.0, relative))

            for voice in ENSEMBLE_VOICES:
                if voice not in cycle.columns:
                    continue
                value = getattr(row, voice)
                if pd.isna(value):
                    continue
                event_time = float(value) - start
                if -0.06 <= event_time <= duration + 0.06:
                    events.append({
                        "voice": voice,
                        "sd": sd - 1,
                        "time": round(float(min(duration, max(0.0, event_time))), 5),
                    })

        if not events:
            continue
        for sd in range(12):
            if positions[sd] is None:
                positions[sd] = sd / 12
        positions = np.asarray(positions, dtype=float)
        positions[0] = 0.0
        positions = np.maximum.accumulate(positions)
        positions = np.minimum(positions, 0.999999)
        events = sorted(events, key=lambda event: (event["time"], event["voice"]))
        out.append({
            "index": len(out),
            "label": str(label),
            "start": round(start, 5),
            "end": round(end, 5),
            "duration": round(duration, 5),
            "positions": [round(float(x), 6) for x in positions],
            "events": events,
            "eventCounts": {
                voice: int(sum(1 for event in events if event["voice"] == voice))
                for voice in ENSEMBLE_VOICES
            },
        })

    return out


def _corpus_recordings(rhythms):
    """Pistes réelles et départs de cycles pour les prises appariées."""
    out = {}
    for rhythm in rhythms:
        piece = rhythm["piece"]
        entries = []
        for take in rhythm["takes"]:
            paths = {
                voice: f"{u.CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{voice}.wav"
                for voice in ENSEMBLE_VOICES
            }
            if not all(os.path.exists(paths[voice]) for voice in DUNUN_VOICES):
                continue

            cycles = _take_cycles(piece, int(take))
            cycle_starts = [cycle["start"] for cycle in cycles]
            if cycles:
                cycle_starts.append(cycles[-1]["end"])

            entry = {
                "take": int(take),
                "cycleStarts": cycle_starts,
                "cycles": cycles,
            }
            for voice, path in paths.items():
                if os.path.exists(path):
                    entry[voice] = "../" + path.replace("\\", "/")
            entries.append(entry)
        if entries:
            out[piece] = entries
    return out


def _load_djembeloops_reference(path="data/djembeloops_reference.json"):
    if not os.path.exists(path):
        return {
            "source": "https://www.djembeloops.com",
            "patterns": [],
            "matrices": {},
            "samples": [],
        }
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_data():
    rhythms = []

    for piece in u.PIECES:
        takes = _paired_takes(piece)
        if not takes:
            continue

        voices = {}
        for code in ENSEMBLE_VOICES:
            motif, used = _profile_from_takes(piece, code, takes)
            if not used:
                continue
            coeffs, density = _coefficients(motif)
            voices[code] = {
                "voice": code,
                "motif": [round(float(x), 4) for x in motif],
                "takes": used,
                "density": round(density, 6),
                "coeffs": coeffs,
            }

        if set(ENSEMBLE_VOICES).issubset(voices):
            comparable_takes = sorted(
                set.intersection(*[set(voices[voice]["takes"]) for voice in ENSEMBLE_VOICES])
            )
            phrase24 = {}
            for code in ENSEMBLE_VOICES:
                motif24, used24 = _profile24_from_takes(piece, code, comparable_takes)
                coeffs24, density24 = _coefficients(motif24)
                phrase24[code] = {
                    "voice": code,
                    "motif": [round(float(x), 4) for x in motif24],
                    "takes": used24,
                    "density": round(density24, 6),
                    "coeffs": coeffs24,
                }
            rhythms.append({
                "piece": piece,
                "takes": comparable_takes,
                "voices": voices,
                "phrase24": phrase24,
            })

    return {
        "pieces": [r["piece"] for r in rhythms],
        "colors": COLORS,
        "rhythms": rhythms,
        "map": _pca_map(rhythms),
        "microtiming": _microtiming_profiles(rhythms),
        "microtimingPhrase": _microtiming_phrase_profiles(rhythms),
        "metricTiming": _metric_timing_profiles(rhythms),
        "recordings": _corpus_recordings(rhythms),
        "djembeloops": _load_djembeloops_reference(),
        "voices": list(ENSEMBLE_VOICES),
        "dununVoices": list(DUNUN_VOICES),
        "jembeVoices": list(JEMBE_VOICES),
        "coefNames": {
            "1": "1 oscillation par cycle",
            "2": "2 oscillations par cycle",
            "3": "3 oscillations par cycle",
            "4": "4 oscillations par cycle",
            "5": "5 oscillations par cycle",
            "6": "alternance maximale",
        },
    }


HTML = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boussole des rythmes</title>
<script>
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true
  },
  svg: {fontCache: "global"}
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
:root{
  --bg:#f2f1ed;--paper:#fbfaf7;--ink:#20211f;--muted:#77766f;
  --line:#d8d7d1;--line2:#e9e7e1;--soft:#f4f3ef;--accent:#b45d2c;
  --j1:#6b5ca5;--j2:#8b6f47;--d1:#245d9b;--d2:#c17a2b;--live:#26815a;--shadow:0 8px 28px rgba(34,34,28,.055)
}
@media(prefers-color-scheme:dark){
  :root{--bg:#111315;--paper:#191c1e;--ink:#f0eee8;--muted:#aaa9a2;--line:#34383a;--line2:#292d2f;--soft:#202426;--accent:#df8b50;--j1:#a799ea;--j2:#c7a06e;--d1:#6ca4df;--d2:#e0a458;--live:#52bc88;--shadow:0 12px 35px rgba(0,0,0,.22)}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}
button,input,select{font:inherit}
button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 35%,transparent);outline-offset:2px}
.page{max-width:1440px;margin:0 auto;padding:26px 24px 46px}
.masthead{display:flex;justify-content:space-between;gap:32px;align-items:flex-end;margin:0 2px 20px}
.titleBlock{min-width:max-content}
h1{font-family:Georgia,Times,serif;font-size:clamp(2.2rem,4vw,3.65rem);line-height:.95;margin:0;letter-spacing:-.045em}
.abstract{max-width:660px;color:var(--muted);font-size:.9rem;margin:0}
.abstract b{color:var(--ink)}
.surface{background:var(--paper);border:1px solid var(--line);border-radius:20px;box-shadow:var(--shadow)}
.toolbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 12px;margin-bottom:16px}
.tabs{display:flex;gap:6px;flex-wrap:wrap}
.tab{appearance:none;border:1px solid transparent;background:transparent;color:var(--muted);border-radius:10px;padding:8px 13px;cursor:pointer;font-weight:800}
.tab:hover{background:var(--soft);color:var(--ink)}
.tab.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.transport{display:flex;gap:9px;align-items:center;flex-wrap:wrap;justify-content:flex-end}
.play{min-width:168px;appearance:none;border:0;background:var(--accent);color:#fff;border-radius:11px;padding:10px 15px;cursor:pointer;font-weight:900}
.play.playing{background:var(--ink)}
.voicePicker,.scopePicker{display:flex;gap:3px;align-items:center;border:1px solid var(--line);background:var(--soft);border-radius:11px;padding:3px}
.voiceBtn{appearance:none;border:0;background:transparent;color:var(--muted);border-radius:8px;padding:6px 9px;cursor:pointer;font-weight:800;font-size:.78rem}
.voiceBtn.on{background:var(--ink);color:var(--paper)}
.scopePicker .voiceBtn{font-size:.74rem}
.volumeControl,.mixControl{display:grid;grid-template-columns:auto 82px 44px;align-items:center;gap:6px;color:var(--muted);font-size:.72rem;font-weight:750}
.mixControl{grid-template-columns:auto 88px 54px}
.volumeControl input,.mixControl input{width:82px;accent-color:var(--accent)}
.mixControl input{width:88px}
.volumeControl output,.mixControl output{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.takePicker,.cyclePicker{display:flex;align-items:center;gap:5px;color:var(--muted);font-size:.72rem;font-weight:750}
.takePicker select,.cyclePicker select{border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:8px;padding:5px 7px;font-size:.75rem}
.takePicker[hidden],.cyclePicker[hidden],.mixControl[hidden]{display:none}
.liveStatus{display:inline-flex;align-items:center;gap:7px;color:var(--muted);font-size:.78rem;min-width:82px}
.liveStatus i{width:8px;height:8px;border-radius:50%;background:var(--line)}
.liveStatus.on{color:var(--live);font-weight:850}.liveStatus.on i{background:var(--live);box-shadow:0 0 0 5px color-mix(in srgb,var(--live) 14%,transparent)}
.playCaption{width:100%;text-align:right;color:var(--muted);font-size:.75rem}
.studioGrid{display:grid;grid-template-columns:minmax(520px,1.06fr) minmax(490px,.94fr);gap:16px;align-items:stretch}
.wheelPanel,.synthPanel,.curvePanel,.microPanel,.mapPanel,.referencePanel{padding:18px}
.panelHead{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:4px}
.panelHead h2,.synthPanel h2,.curvePanel h2,.microPanel h2,.mapPanel h2,.referencePanel h2{font-family:Georgia,Times,serif;font-size:1.35rem;margin:0;letter-spacing:-.025em}
.modeTag{border:1px solid var(--line);background:var(--soft);border-radius:999px;color:var(--muted);padding:5px 9px;font-size:.72rem;font-weight:800}
.modeTag.live{border-color:color-mix(in srgb,var(--live) 45%,var(--line));color:var(--live);background:color-mix(in srgb,var(--live) 7%,var(--paper))}
.metricControlGroup{display:flex;align-items:center;justify-content:flex-end;gap:7px;flex-wrap:wrap}
.metricPicker,.phrasePicker{display:flex;align-items:center;gap:3px;padding:3px;border:1px solid var(--line);border-radius:10px;background:var(--soft)}
.metricPicker .smallBtn,.phrasePicker .smallBtn{border:0;background:transparent;padding:5px 8px;color:var(--muted)}
.metricPicker .smallBtn.on,.phrasePicker .smallBtn.on{background:var(--ink);color:var(--paper)}
canvas{display:block;width:100%;height:auto}
.wheelBody{display:grid;grid-template-columns:minmax(0,1fr) 118px;gap:14px;align-items:center}
.wheelPanel #circle{max-width:590px;margin:0 auto}
.wheelVoiceRail{align-self:stretch;display:flex;flex-direction:column;justify-content:center;gap:9px;border-left:1px solid var(--line2);padding-left:12px}
.wheelVoiceRail h3{margin:0;color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.12em}
.voicePicker.vertical{display:grid;gap:7px;border:0;background:transparent;padding:0}
.voicePicker.vertical .voiceBtn{border:1px solid var(--line);background:var(--paper);color:var(--ink);padding:9px 10px;text-align:left}
.voicePicker.vertical .voiceBtn.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.voicePicker.vertical .voiceBtn.off{opacity:.45}
.caption,.note{font-size:.85rem;color:var(--muted)}
.caption b,.note b{color:var(--ink)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin:6px 0 2px;color:var(--muted);font-size:.8rem}
.colorLegend{border:1px solid var(--line2);background:var(--soft);border-radius:12px;padding:8px 10px;margin-top:10px}
.colorLegend b{color:var(--ink)}
.swatch{display:inline-flex;align-items:center;gap:7px}.swatch i{width:13px;height:13px;border-radius:99px;display:inline-block;border:2px solid currentColor}
.j1{color:var(--j1)}.j2{color:var(--j2)}.d1{color:var(--d1)}.d2{color:var(--d2)}
.metricKey{display:inline-flex;align-items:center;gap:7px}
.metricKey i{width:16px;height:0;border-top:2px dashed var(--muted);opacity:.65}
.listeningNote{margin:9px 0 0;border-left:3px solid var(--accent);padding:4px 0 4px 11px;color:var(--muted);font-size:.78rem}
.listeningNote b{color:var(--ink)}
.synthPanel{display:block;align-self:start}
.synthIntro{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:12px;cursor:pointer;list-style:none}
.synthIntro::-webkit-details-marker{display:none}
.synthPanel:not([open]) .synthIntro{margin-bottom:0}
.synthIntro .caption{margin:4px 0 0;max-width:420px}
.synthSummaryState{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.foldState{color:var(--muted);font-size:.68rem;font-weight:850;text-transform:uppercase;letter-spacing:.08em}
.foldState::before{content:"ouvrir"}
.synthPanel[open] .foldState::before{content:"réduire"}
.synthPipeline{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin:0 0 11px}
.pipelineStep{border:1px solid var(--line2);background:var(--soft);border-radius:11px;padding:8px 9px;min-height:58px}
.pipelineStep span{display:block;color:var(--muted);font-size:.62rem;text-transform:uppercase;letter-spacing:.09em;font-weight:850}
.pipelineStep b{display:block;color:var(--ink);font-size:.8rem;margin-top:2px}
.pipelineStep small{display:block;color:var(--muted);font-size:.66rem;line-height:1.25;margin-top:1px}
.synthTop{display:flex;gap:7px;align-items:center;flex-wrap:wrap;padding:10px 0 12px;border-top:1px solid var(--line2);border-bottom:1px solid var(--line2);margin-bottom:10px}
.smallBtn{appearance:none;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:9px;padding:7px 10px;cursor:pointer;font-weight:800;font-size:.78rem}
.smallBtn:hover{background:var(--soft)}
.smallBtn.on{background:var(--ink);border-color:var(--ink);color:var(--paper)}
.resetBtn{margin-left:auto}
.transformBar{display:grid;grid-template-columns:.82fr 1.18fr;gap:7px;margin:0 0 10px}
.transformAction{display:grid;align-content:start;gap:6px;background:var(--soft);border:1px solid var(--line2);border-radius:11px;padding:7px}
.transformActionHead{display:flex;align-items:center;gap:6px}
.transformAction .smallBtn{flex:1;background:var(--paper)}
.transformAction select{max-width:98px;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:8px;padding:7px 6px;font-size:.73rem}
.symmetryGuide{margin:0;color:var(--muted);font-size:.71rem;line-height:1.35;font-variant-numeric:tabular-nums}
.symmetryGuide b{color:var(--ink)}
.transformFeedback{margin:-2px 2px 8px;color:var(--muted);font-size:.74rem}
.transformFeedback:empty{display:none}
.transformFeedback b{color:var(--ink)}
.synthThresholdControl{display:grid;grid-template-columns:132px minmax(160px,1fr) 84px;align-items:center;gap:9px;border:1px solid var(--line2);background:var(--soft);border-radius:12px;padding:9px 10px;margin:0 0 8px;color:var(--muted);font-size:.74rem}
.synthThresholdControl b{display:block;color:var(--ink);font-size:.82rem}.synthThresholdControl small{display:block;color:var(--muted);font-size:.64rem;line-height:1.2}
.synthThresholdControl input{width:100%;accent-color:var(--accent)}
.synthThresholdControl output{color:var(--ink);text-align:right;font-variant-numeric:tabular-nums}
.quantizeBadge{display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--line);border-radius:999px;background:var(--paper);color:var(--ink);font-size:.72rem;font-weight:850;padding:6px 10px;text-align:center}
.partitionEditor{border:1px solid var(--line2);background:var(--soft);border-radius:14px;padding:10px 11px;margin:0 0 10px}
.partitionEditorHead{display:flex;align-items:flex-end;justify-content:space-between;gap:10px;margin-bottom:9px}
.partitionEditorHead h3{margin:0;font-size:.92rem;letter-spacing:-.01em}.partitionEditorHead p{margin:2px 0 0;color:var(--muted);font-size:.68rem;line-height:1.25}
.partitionRows{display:grid;gap:7px}.partitionRow{display:grid;grid-template-columns:36px repeat(12,minmax(18px,1fr));gap:4px;align-items:center}
.partitionRow b{font-size:.8rem;color:var(--ink)}.partitionStep{height:28px;border:1px solid var(--line);border-radius:8px;background:var(--paper);color:var(--muted);font-size:.68rem;font-weight:850;cursor:pointer}
.partitionStep.on{background:var(--ink);border-color:var(--ink);color:var(--paper);box-shadow:0 3px 9px rgba(32,29,24,.14)}
.partitionRow[data-voice="D1"] .partitionStep.on{background:var(--d1);border-color:var(--d1)}.partitionRow[data-voice="D2"] .partitionStep.on{background:var(--d2);border-color:var(--d2)}
.partitionActions{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}.partitionActions .smallBtn{padding:6px 8px;font-size:.68rem}
.coefControls{display:grid;gap:6px}
.coefRow{display:grid;grid-template-columns:92px 28px minmax(90px,1fr) 45px 24px minmax(90px,1fr) 43px;gap:7px;align-items:center;background:var(--soft);border:1px solid transparent;border-radius:11px;padding:8px 9px;font-size:.76rem;color:var(--muted)}
.coefRow:hover{border-color:var(--line)}
.coefRow b{color:var(--ink);font-size:.82rem}.coefRow small{display:block;font-size:.63rem;color:var(--muted);font-weight:600;line-height:1.15;margin-top:1px}
.coefRow input[type=range]{width:100%;height:4px;accent-color:var(--accent);cursor:ew-resize}
.coefRow output{font-variant-numeric:tabular-nums;color:var(--ink);font-size:.72rem;text-align:right}
.coefRow.density{grid-template-columns:92px 28px minmax(170px,1fr) 45px}
.curvePanel{margin-top:16px}
.curvePanel .panelHead{align-items:flex-end;margin-bottom:6px}
.curvePanel #curve{max-height:350px}
.variantLibrary{margin-top:16px;overflow:hidden}
.variantLibrary>summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:16px 18px;cursor:pointer;list-style:none}
.variantLibrary>summary::-webkit-details-marker{display:none}
.variantLibrary>summary::after{content:"ouvrir";color:var(--muted);font-size:.72rem;font-weight:850;text-transform:uppercase;letter-spacing:.08em}
.variantLibrary[open]>summary{border-bottom:1px solid var(--line2)}
.variantLibrary[open]>summary::after{content:"replier"}
.variantLibrary h2{font-family:Georgia,Times,serif;font-size:1.35rem;margin:0;letter-spacing:-.025em}
.variantLibrary summary p{margin:3px 0 0;color:var(--muted);font-size:.8rem}
.libraryBody{padding:14px 18px 18px}
.libraryToolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:12px}
.libraryTabs{display:flex;gap:5px;flex-wrap:wrap}
.libraryStatus{color:var(--muted);font-size:.8rem;margin:0 0 12px}
.libraryStatus b{color:var(--ink)}
.libraryDistance{border-left:3px solid var(--accent);padding:4px 0 4px 11px;color:var(--muted);font-size:.75rem;margin:0 0 13px}
.libraryDistance b{color:var(--ink)}
.voiceVariantColumns{display:grid;grid-template-columns:1fr 1fr;gap:12px;align-items:start}
.voiceVariantColumn{border:1px solid var(--line2);border-radius:15px;padding:10px;background:color-mix(in srgb,var(--soft) 62%,var(--paper))}
.voiceVariantHead{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:0 2px 8px}
.voiceVariantHead h3{margin:0;font-family:Georgia,Times,serif;font-size:1.05rem}
.voiceVariantHead span{font-size:.7rem;color:var(--muted)}
.voiceVariantGrid{display:grid;gap:7px}
.variantCard{appearance:none;display:grid;grid-template-columns:78px minmax(0,1fr);grid-template-rows:auto auto 1fr;width:100%;min-width:0;text-align:left;border:1px solid var(--line2);background:var(--soft);color:var(--ink);border-radius:12px;padding:7px 9px 7px 7px;cursor:pointer;align-items:center}
.variantCard:hover{border-color:var(--line);background:var(--paper)}
.variantCard.on{border-color:var(--accent);box-shadow:inset 0 0 0 1px var(--accent);background:color-mix(in srgb,var(--accent) 5%,var(--paper))}
.variantCard canvas{grid-row:1/4;width:72px;height:72px;margin:0}
.variantCard b{font-family:Georgia,Times,serif;font-size:.94rem}
.variantCard small{display:block;color:var(--muted);font-size:.67rem;line-height:1.3;margin-top:1px}
.variantCard .variantAction{color:var(--accent);font-size:.68rem;font-weight:850;margin-top:4px}
.lowerGrid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(360px,.75fr);gap:16px;margin-top:16px}
.microControls{display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.microScale{display:grid;grid-template-columns:auto 110px 38px;align-items:center;gap:6px;color:var(--muted);font-size:.72rem}
.microScale input{width:110px;accent-color:var(--accent)}
.microScale output{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.microPanel #microTiming,.mapPanel #map{max-height:330px}
.equationBox{border:1px solid var(--line);background:var(--soft);border-radius:14px;padding:14px;overflow:auto}
.math{font-size:.98rem}
.metaGrid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0 0}
.metric{border:1px solid var(--line2);background:var(--soft);border-radius:11px;padding:9px}
.metric span{display:block;color:var(--muted);font-size:.74rem;text-transform:uppercase;letter-spacing:.09em}
.metric b{font-size:1rem}
.coefTable{width:100%;border-collapse:separate;border-spacing:0;font-size:.82rem;overflow:hidden;border:1px solid var(--line);border-radius:12px}
.coefTable th,.coefTable td{padding:8px 9px;border-bottom:1px solid var(--line2);text-align:right;white-space:nowrap}
.coefTable th:first-child,.coefTable td:first-child{text-align:left}
.coefTable tr:last-child td{border-bottom:0}
.coefTable th{color:var(--muted);font-weight:800;background:var(--soft)}
.referencePanel{margin-top:16px}
.referencePanel .panelHead{align-items:flex-end}
.referenceTools{display:flex;align-items:center;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.referenceTools select{border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:9px;padding:7px 9px;font-size:.78rem;max-width:260px}
.referencePlayer{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;border:1px solid var(--line2);background:var(--soft);border-radius:13px;padding:10px 11px;margin:12px 0}
.referencePlayerLeft{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.referenceTempo{display:grid;grid-template-columns:auto 120px 44px;align-items:center;gap:7px;color:var(--muted);font-size:.72rem;font-weight:750}
.referenceTempo input{width:120px;accent-color:var(--accent)}
.referenceTempo output{color:var(--ink);font-variant-numeric:tabular-nums;text-align:right}
.referenceStatus{font-size:.78rem;color:var(--muted)}
.referenceStatus.live{color:var(--live);font-weight:850}
.referenceStats{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px;margin:12px 0}
.referenceStat{border:1px solid var(--line2);background:var(--soft);border-radius:11px;padding:9px;font-size:.82rem}
.referenceStat span{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase;letter-spacing:.08em}
.referenceStat b{font-size:.98rem}
.referencePanel #djembeloopsCompare{max-height:330px}
.theory{margin-top:16px;padding:0;overflow:hidden}
.theory summary{cursor:pointer;list-style:none;padding:16px 18px;font-family:Georgia,Times,serif;font-size:1.1rem;font-weight:700}
.theory summary::-webkit-details-marker{display:none}
.theory summary::after{content:"+";float:right;color:var(--muted);font-family:Inter,sans-serif}
.theory[open] summary{border-bottom:1px solid var(--line2)}.theory[open] summary::after{content:"−"}
.theoryBody{padding:18px}
.theoryGrid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(430px,.8fr);gap:16px;align-items:start}
.theoryGrid h3{font-size:.72rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);margin:0 0 8px}
.researchNote{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:12px}
.researchNote div{border-left:3px solid var(--accent);padding:4px 0 4px 12px;color:var(--muted);font-size:.86rem}
@media(max-width:1080px){
  .masthead{align-items:flex-start}.studioGrid,.lowerGrid,.theoryGrid{grid-template-columns:1fr}
  .synthPipeline{grid-template-columns:repeat(5,minmax(120px,1fr));overflow-x:auto;padding-bottom:2px}
  .synthPanel{min-height:0}.wheelPanel #circle{max-width:620px}.curvePanel #curve{max-height:none}
}
@media(max-width:720px){
  .page{padding:18px 12px 34px}.masthead{display:block}.titleBlock{min-width:0}.abstract{margin-top:12px}
  .toolbar{align-items:flex-start;flex-direction:column}.transport{justify-content:flex-start}.playCaption{text-align:left}
  .studioGrid{grid-template-columns:minmax(0,1fr)}.wheelPanel,.synthPanel,.curvePanel,.microPanel,.mapPanel,.referencePanel{padding:13px}
  .panelHead{display:block}.metricControlGroup{justify-content:flex-start;margin-top:10px}
  .wheelBody{grid-template-columns:1fr}.wheelVoiceRail{border-left:0;border-top:1px solid var(--line2);padding:10px 0 0}.voicePicker.vertical{grid-template-columns:repeat(4,1fr)}
  .transformBar{grid-template-columns:1fr}
  .synthPipeline{grid-template-columns:1fr;overflow:visible}
  .synthThresholdControl{grid-template-columns:1fr 1fr 84px}
  .partitionRow{grid-template-columns:28px repeat(12,minmax(14px,1fr));gap:3px}.partitionStep{height:24px;font-size:.62rem}
  .coefRow,.coefRow.density{grid-template-columns:78px 1fr 46px}.coefRow span{display:none}
  .coefRow input[data-kind="phase"]{grid-column:2}.coefRow output[data-out^="phase"]{grid-column:3}
  .metaGrid{grid-template-columns:1fr 1fr}.theoryBody{padding:13px}
  .variantLibrary>summary{display:block}.variantLibrary>summary::after{display:block;margin-top:8px}
  .libraryBody{padding:12px}.voiceVariantColumns{grid-template-columns:1fr}
}
</style></head><body><main class="page">
<header class="masthead">
  <div class="titleBlock">
    <h1>Boussole des rythmes</h1>
  </div>
  <p class="abstract">
    <b>Objet.</b> Trois rythmes, trois équations complexes. Chaque signature est
    calculée sur les prises où J1, J2, D1 et D2 sont présents ensemble,
    pour comparer les quatre voix sur la même base. Les coefficients sont des
    projections du motif sur les racines complexes du cycle.
  </p>
</header>

<section class="toolbar surface">
  <nav class="tabs" id="tabs"></nav>
  <div class="transport">
    <div class="scopePicker" id="scopePicker" aria-label="Vue de la boussole" hidden></div>
    <button class="play" id="playBtn">▶ Lancer la boucle</button>
    <label class="takePicker" id="recordingTakeWrap">prise
      <select id="recordingTake" aria-label="Prise réelle du corpus"></select>
    </label>
    <label class="cyclePicker" id="recordingCycleWrap">cycle
      <select id="recordingCycle" aria-label="Cycle annoté de la prise"></select>
    </label>
    <label class="mixControl" id="reconstructionMixWrap">A/B
      <input type="range" id="reconstructionMix" min="0" max="100" step="5" value="50">
      <output id="reconstructionMixOut">50/50</output>
    </label>
    <label class="volumeControl">volume
      <input type="range" id="volumeSlider" min="0" max="200" step="5" value="130">
      <output id="volumeOut">130 %</output>
    </label>
    <span class="liveStatus" id="liveStatus"><i></i><span>Prêt</span></span>
    <div class="playCaption" id="playCaption">Les modifications seront audibles en direct.</div>
  </div>
</section>

<section class="studioGrid">
  <article class="wheelPanel surface">
    <div class="panelHead">
      <div><h2 id="wheelTitle">Cercle métrique · 12 subdivisions</h2></div>
      <div class="metricControlGroup">
        <div class="phrasePicker" id="phraseModePicker" aria-label="Fenêtre d'analyse">
          <button class="smallBtn on" data-phrase-mode="mean12">cycle 12 Polak</button>
        </div>
        <div class="metricPicker" aria-label="Temporalité du cercle">
          <button class="smallBtn" id="circleRecordingBtn">Enregistrement</button>
          <button class="smallBtn" id="circleSynthesisBtn">Partition synthèse</button>
          <button class="smallBtn" id="circleReferenceBtn">Référence externe</button>
        </div>
      </div>
    </div>
    <div class="wheelBody">
      <canvas id="circle" width="620" height="620"></canvas>
      <aside class="wheelVoiceRail">
        <h3>Voix à écouter</h3>
        <div class="voicePicker vertical" id="voicePicker"></div>
      </aside>
    </div>
    <div class="legend">
      <span class="swatch j1" data-voice-legend="J1"><i></i>J1 · djembé 1</span>
      <span class="swatch j2" data-voice-legend="J2"><i></i>J2 · djembé 2</span>
      <span class="swatch d1" data-voice-legend="D1"><i></i>D1 · dunun grave</span>
      <span class="swatch d2" data-voice-legend="D2"><i></i>D2 · dunun réponse</span>
      <span class="metricKey" id="metricKey"><i></i>repères isochrones</span>
    </div>
    <p class="caption" id="circleCaption"></p>
    <p class="listeningNote" id="listeningNote"></p>
  </article>

  <details class="synthPanel surface" id="synthPanel" open>
    <summary class="synthIntro">
      <div>
        <h2>Synthèse partition + coefficients</h2>
        <p class="caption">La partition de travail D1/D2 est la base jouée. Les coefficients complexes sont recalculés dessus pour analyser et transformer le rythme.</p>
      </div>
      <div class="synthSummaryState">
        <span class="modeTag" id="synthLiveTag">Prêt</span>
        <span class="foldState" aria-hidden="true"></span>
      </div>
    </summary>
    <div class="synthBody">
    <div class="synthPipeline" aria-label="Chaîne de synthèse">
      <div class="pipelineStep"><span>1 · données</span><b>Polak</b><small>prises et cycles annotés</small></div>
      <div class="pipelineStep"><span>2 · partition</span><b>D1/D2 · 12</b><small>base visible et corrigeable</small></div>
      <div class="pipelineStep"><span>3 · calcul</span><b>\(c_k\)</b><small>signature complexe de la partition</small></div>
      <div class="pipelineStep"><span>4 · édition</span><b>\(\tilde c_k\)</b><small>variations puis nouvelle partition</small></div>
      <div class="pipelineStep"><span>5 · écoute</span><b>samples</b><small>DjembeLoops comme timbre</small></div>
    </div>
    <div class="partitionEditor" id="partitionEditor"></div>
    <div class="synthTop">
      <button class="smallBtn" id="originalModeBtn">Signal Polak</button>
      <button class="smallBtn" id="synthModeBtn">Modifier \(c_k\)</button>
      <div class="voicePicker" id="synthVoicePicker"></div>
      <button class="smallBtn resetBtn" id="resetSynthBtn">Réinitialiser</button>
    </div>
    <div class="transformBar">
      <div class="transformAction">
        <button class="smallBtn" id="orthogonalBtn">⊥ Orthogonal à l’autre voix</button>
      </div>
      <div class="transformAction symmetryAction">
        <div class="transformActionHead">
          <button class="smallBtn" id="mirrorBtn">Refléter la voix</button>
          <select id="symmetryAxis" aria-label="Axe de symétrie">
            <option value="0">axe 1–7</option>
            <option value="1">axe 2–8</option>
            <option value="2">axe 3–9</option>
            <option value="3">axe 4–10</option>
            <option value="4">axe 5–11</option>
            <option value="5">axe 6–12</option>
          </select>
        </div>
        <p class="symmetryGuide" id="symmetryGuide"></p>
      </div>
    </div>
    <div class="transformFeedback" id="transformFeedback"></div>
    <div class="synthThresholdControl" id="synthThresholdControl"></div>
    <div class="coefControls" id="synthControls"></div>
    </div>
  </details>
</section>

<details class="variantLibrary surface" id="variantLibrary" open>
  <summary>
    <div>
      <h2>Dossier des variantes symétriques</h2>
      <p>Original et six réflexions pour chaque rythme et chaque voix de dunun.</p>
    </div>
  </summary>
  <div class="libraryBody">
    <div class="libraryToolbar">
      <nav class="libraryTabs" id="libraryPieceTabs" aria-label="Rythme du catalogue"></nav>
    </div>
    <p class="libraryStatus" id="libraryStatus"></p>
    <p class="libraryDistance" id="libraryDistance"></p>
    <div class="voiceVariantColumns" id="variantColumns"></div>
  </div>
</details>

<section class="curvePanel surface">
  <div class="panelHead">
    <div><h2>Courbes de l’ensemble</h2></div>
    <span class="modeTag" id="curveMode">Points observés + modèle</span>
  </div>
  <canvas id="curve" width="1260" height="340"></canvas>
  <div class="legend colorLegend">
    <b>Couleurs des voix :</b>
    <span class="swatch j1"><i></i>violet J1 · djembé</span>
    <span class="swatch j2"><i></i>brun J2 · djembé</span>
    <span class="swatch d1"><i></i>bleu D1 · dunun grave</span>
    <span class="swatch d2"><i></i>orange D2 · dunun réponse</span>
  </div>
  <p class="caption">Abscisse : position cyclique. Ordonnée : présence/intensité estimée depuis Polak. En mode synthèse, les carrés indiquent la partition de travail réellement jouée ; la courbe montre sa signature complexe reconstruite.</p>
</section>

<section class="referencePanel surface">
  <div class="panelHead">
    <div>
      <h2>Référence externe</h2>
      <p class="caption">Table de rythme et player inspirés de DjembeLoops. Le rythme proposé par défaut est celui trouvé le plus proche.</p>
    </div>
    <div class="referenceTools">
      <select id="djembeloopsPattern" aria-label="Rythme DjembeLoops de référence"></select>
      <span class="modeTag" id="djembeloopsMode">référence</span>
    </div>
  </div>
  <div class="referencePlayer">
    <div class="referencePlayerLeft">
      <button class="smallBtn" id="referencePlayBtn">▶ Lire référence</button>
      <button class="smallBtn" id="referenceStopBtn">■ Stop</button>
      <label class="referenceTempo">tempo
        <input type="range" id="referenceTempo" min="60" max="180" step="1" value="120">
        <output id="referenceTempoOut">120</output>
      </label>
    </div>
    <span class="referenceStatus" id="referenceStatus">samples locaux prêts</span>
  </div>
  <div class="referenceStats" id="djembeloopsStats"></div>
  <canvas id="djembeloopsCompare" width="1260" height="330"></canvas>
  <p class="caption" id="djembeloopsCaption"></p>
</section>

<section class="lowerGrid">
  <article class="microPanel surface">
    <div class="panelHead">
      <div><h2>Temporalité observée</h2></div>
      <div class="microControls">
        <button class="smallBtn" id="recordingTimingBtn">Enregistrement</button>
        <button class="smallBtn" id="synthesisTimingBtn">Partition synthèse</button>
        <button class="smallBtn" id="referenceTimingBtn">Référence externe</button>
        <label class="microScale">impact
          <input type="range" id="timingStrength" min="0" max="2" step="0.05" value="1">
          <output id="timingStrengthOut">×1.00</output>
        </label>
      </div>
    </div>
    <canvas id="microTiming" width="760" height="310"></canvas>
    <p class="caption" id="microCaption">Écart à la grille en millisecondes : négatif = avance, positif = retard. Les barres verticales montrent la dispersion.</p>
  </article>
  <article class="mapPanel surface">
    <div class="panelHead"><div><h2>Espace des signatures</h2></div></div>
    <canvas id="map" width="620" height="310"></canvas>
    <p class="caption"><b>Légende PC.</b> PC1 est la direction qui explique le plus de différences entre signatures complexes ; PC2 est la deuxième direction indépendante. Deux points proches correspondent à deux rythmes dont les coefficients Fourier se ressemblent.</p>
  </article>
</section>

<details class="theory surface">
  <summary>Équation complexe et détail des coefficients</summary>
  <div class="theoryBody">
    <div class="theoryGrid">
      <section>
        <h3>Équation du rythme sélectionné</h3>
        <div class="equationBox"><div class="math" id="mainEquation"></div></div>
        <div class="metaGrid" id="metrics"></div>
      </section>
      <section>
        <h3>Table des coefficients</h3>
        <div id="coefTable"></div>
      </section>
    </div>
    <div class="researchNote">
      <div><b>Ensemble comparable.</b><br>Les quatre voix sont calculées sur les mêmes prises appariées.</div>
      <div><b>N = résolution du cercle.</b><br>Ici la boussole principale utilise \(N=12\), mais le calcul s’écrit pareil pour une autre résolution.</div>
      <div><b>Calcul complexe.</b><br>\(z_s=e^{2\pi i s/N}\) et \(c_k\) additionne les frappes après rotation par \(z_s^{-k}\).</div>
      <div><b>Une couche = une onde test.</b><br>Pour \(k=3\), l'onde fait trois tours pendant le cycle.</div>
    </div>
  </div>
</details>
</main>

<script>
const DATA=__DATA__;
const TAU=Math.PI*2;
const ALL_VOICES=DATA.voices||["J1","J2","D1","D2"];
const DUNUN_VOICES=DATA.dununVoices||["D1","D2"];
const JEMBE_VOICES=DATA.jembeVoices||["J1","J2"];
const DJEMBELOOPS_CANDIDATES={
  Manjanin:["mandiani","mendiani","mendianiII"],
  Maraka:["marakadonII","marakadonI"],
  Woloso:["wolesedon","wolesedonII"],
};
const DJEMBELOOPS_DUNUN=["Kenkeni","Sangban","Doundoun"];
const DJEMBELOOPS_JEMBE=["DjembeOne","DjembeTwo","DjembeThree","DjembeFive","DjembeSix","DjembeSeven","DjembeSolo"];
const state={
  piece:DATA.pieces[0],scope:"ensemble",playVoice:"ensemble",selectedVoiceSet:{J1:false,J2:false,D1:true,D2:true},synthEnabled:false,synthVoice:"D1",
  synthData:{},partitionData:{},listeningMode:"recording",timingMode:"observed",timingStrength:1,volume:1.3,reconstructionMix:.5,transformMessage:"",symmetryAxis:0,
  phraseMode:"mean12",djembeloopsPatternByPiece:{},referenceTempoByPattern:{},
  libraryPiece:DATA.pieces[0],libraryLoaded:{D1:null,D2:null},
  recordingCycleByKey:{},
  recordingTakeByPiece:Object.fromEntries(DATA.pieces.map(piece=>[
    piece,DATA.recordings?.[piece]?.[0]?.take??null
  ]))
};
const playback={
  active:false,kind:null,step:.155,lookaheadMs:25,scheduleAhead:.11,raf:null,
  ctx:null,master:null,compressor:null,timer:null,nextStepTime:0,stepIndex:0,
  startTime:0,timeline:[],media:{},voiceGains:{},reconstructionGain:null,
  cycle:null,nextEventIndex:0
};
const referencePlayback={
  active:false,loadingAudio:false,timer:null,stepIndex:0,nextStepAt:0,events:[],patternKey:null,tempo:120,
  pending:[],ctx:null,master:null,buffers:new Map(),loading:new Map(),sources:[],webAudioReady:false,fallbackReason:""
};
const sampleAudioCache=new Map();

function rhythm(piece=state.piece){return DATA.rhythms.find(r=>r.piece===piece) || DATA.rhythms[0];}
function recordingOptions(piece=state.piece){return DATA.recordings?.[piece]||[];}
function selectedRecording(piece=state.piece){
  const options=recordingOptions(piece);
  const take=state.recordingTakeByPiece[piece];
  return options.find(item=>item.take===take)||options[0]||null;
}
function recordingKey(piece=state.piece,recording=selectedRecording(piece)){
  return `${piece}:${recording?.take??"none"}`;
}
function recordingCycles(recording=selectedRecording()){
  return recording?.cycles||[];
}
function defaultCycleIndex(recording=selectedRecording()){
  const cycles=recordingCycles(recording);
  const voices=state.scope==="ensemble" ? ALL_VOICES : DUNUN_VOICES;
  const complete=cycles.findIndex(cycle=>
    voices.every(voice=>(cycle.eventCounts?.[voice]||0)>0)
  );
  return complete>=0 ? complete : 0;
}
function selectedCycleIndex(recording=selectedRecording(),piece=state.piece){
  const cycles=recordingCycles(recording);
  if(!cycles.length)return -1;
  const key=recordingKey(piece,recording);
  const stored=Object.prototype.hasOwnProperty.call(state.recordingCycleByKey,key)
    ? Number(state.recordingCycleByKey[key])
    : defaultCycleIndex(recording);
  return Math.max(0,Math.min(cycles.length-1,Number.isFinite(stored)?stored:0));
}
function selectedCycle(recording=selectedRecording(),piece=state.piece){
  const index=selectedCycleIndex(recording,piece);
  return index<0 ? null : recordingCycles(recording)[index];
}
function selectedCycleLabel(recording=selectedRecording(),piece=state.piece){
  const cycle=selectedCycle(recording,piece), index=selectedCycleIndex(recording,piece);
  return cycle ? `cycle ${index+1}` : "cycle —";
}
function validCycleDuration(value){
  const duration=Number(value);
  return Number.isFinite(duration)&&duration>.5&&duration<5 ? duration : null;
}
function typicalRecordingCycleDuration(piece=state.piece){
  const durations=(DATA.recordings?.[piece]||[])
    .flatMap(recording=>(recording.cycles||[]).map(cycle=>validCycleDuration(cycle.duration)))
    .filter(duration=>duration!==null)
    .sort((a,b)=>a-b);
  if(!durations.length)return 12*playback.step;
  const mid=durations.length/2;
  return durations.length%2 ? durations[Math.floor(mid)] : (durations[mid-1]+durations[mid])/2;
}
function synthesisCycleDuration(piece=state.piece){
  return typicalRecordingCycleDuration(piece);
}
function synthesisCycleLabel(piece=state.piece){
  const duration=synthesisCycleDuration(piece);
  return `${duration.toFixed(3)} s/cycle médian`;
}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function phaseDeg(rad){let d=rad*180/Math.PI;while(d>180)d-=360;while(d<=-180)d+=360;return d;}
function signed(x,digits=2){return (x>=0?"+":"") + x.toFixed(digits);}
function phaseTex(rad){return signed(rad,2);}
function texVector(it){
  return String.raw`\left(${it.coeffs.slice(1).map(c=>`${c.mag.toFixed(2)}e^{${phaseTex(c.phase)}i}`).join(String.raw`,\; `)}\right)`;
}
function coefficientName(k,n=displayStepCount()){
  if(n===12&&DATA.coefNames[String(k)])return DATA.coefNames[String(k)];
  if(k===Math.floor(n/2))return "alternance maximale";
  return `${k} oscillations / fenêtre`;
}
function color(voice){return css(`--${voice.toLowerCase()}`)||css("--accent");}
function visibleVoices(){
  return state.scope==="dunun" ? DUNUN_VOICES.slice() : ALL_VOICES.slice();
}
function selectedVoices(){
  const voices=ALL_VOICES.filter(voice=>state.selectedVoiceSet?.[voice]);
  return voices.length ? voices : ALL_VOICES.slice();
}
function playVoiceLabel(){
  return selectedVoices().join("+");
}
function applyVolume(){
  if(!playback.master||!playback.ctx)return;
  const now=playback.ctx.currentTime;
  playback.master.gain.cancelScheduledValues(now);
  playback.master.gain.setTargetAtTime(state.volume,now,.025);
}
function renderRecordingTakeControl(){
  const wrap=document.getElementById("recordingTakeWrap");
  const select=document.getElementById("recordingTake");
  const options=recordingOptions();
  wrap.hidden=!["recording","reconstruction"].includes(state.listeningMode)||!options.length;
  select.innerHTML="";
  options.forEach(recording=>{
    const option=document.createElement("option");
    option.value=String(recording.take);
    option.textContent=`${recording.take}`;
    option.selected=selectedRecording()?.take===recording.take;
    select.appendChild(option);
  });
  select.onchange=async()=>{
    const wasPlaying=playback.active;
    if(wasPlaying)stopPlayback();
    state.recordingTakeByPiece[state.piece]=Number(select.value);
    renderTimingControls();
    drawCircle();
    if(wasPlaying)await startPlayback();
  };
}
function renderRecordingCycleControl(){
  const wrap=document.getElementById("recordingCycleWrap");
  const select=document.getElementById("recordingCycle");
  const recording=selectedRecording();
  const cycles=recordingCycles(recording);
  wrap.hidden=!["recording","reconstruction"].includes(state.listeningMode)||!cycles.length;
  select.innerHTML="";
  cycles.forEach((cycle,index)=>{
    const option=document.createElement("option");
    option.value=String(index);
    const counts=visibleVoices().map(voice=>`${voice} ${cycle.eventCounts?.[voice]??0}`).join(" / ");
    option.textContent=`${index+1} · ${cycle.duration.toFixed(3)} s · ${counts}`;
    option.selected=index===selectedCycleIndex(recording);
    select.appendChild(option);
  });
  select.onchange=async()=>{
    const wasPlaying=playback.active;
    if(wasPlaying)stopPlayback();
    state.recordingCycleByKey[recordingKey()]=Number(select.value);
    renderTimingControls();
    drawCircle();
    drawCurve();
    drawMicroTiming();
    if(wasPlaying)await startPlayback();
  };
}
function applyReconstructionMix(){
  if(!playback.ctx)return;
  const now=playback.ctx.currentTime;
  if(playback.reconstructionGain){
    playback.reconstructionGain.gain.setTargetAtTime(state.reconstructionMix,now,.025);
  }
  syncCorpusVoiceGains();
}
function renderReconstructionMixControl(){
  const wrap=document.getElementById("reconstructionMixWrap");
  const slider=document.getElementById("reconstructionMix");
  const output=document.getElementById("reconstructionMixOut");
  wrap.hidden=state.listeningMode!=="reconstruction";
  slider.value=Math.round(state.reconstructionMix*100);
  output.textContent=`${Math.round((1-state.reconstructionMix)*100)}/${Math.round(state.reconstructionMix*100)}`;
  slider.oninput=()=>{
    state.reconstructionMix=Number(slider.value)/100;
    output.textContent=`${Math.round((1-state.reconstructionMix)*100)}/${Math.round(state.reconstructionMix*100)}`;
    applyReconstructionMix();
    syncPlaybackUI();
  };
}
function renderVolumeControl(){
  const slider=document.getElementById("volumeSlider");
  const output=document.getElementById("volumeOut");
  slider.value=Math.round(state.volume*100);
  output.textContent=`${Math.round(state.volume*100)} %`;
  slider.oninput=()=>{
    state.volume=Number(slider.value)/100;
    output.textContent=`${slider.value} %`;
    applyVolume();
  };
}
function timingAmount(){
  if(state.listeningMode==="recording")return 1;
  if(state.listeningMode==="synthesis"&&state.timingMode==="observed")return state.timingStrength;
  return 0;
}
function metricTiming(piece=state.piece){
  return DATA.metricTiming?.[piece]||null;
}
function metricDurations(piece=state.piece){
  const observed=metricTiming(piece)?.durations;
  const regular=1/12, amount=timingAmount();
  const raw=Array.from({length:12},(_,s)=>{
    const target=Number.isFinite(observed?.[s])?observed[s]:regular;
    return Math.max(.015,regular+amount*(target-regular));
  });
  const total=raw.reduce((sum,value)=>sum+value,0)||1;
  return raw.map(value=>value/total);
}
function metricPositions(piece=state.piece){
  const durations=metricDurations(piece),positions=[0];
  for(let s=1;s<12;s++)positions.push(positions[s-1]+durations[s-1]);
  return positions;
}
function metricPhaseAtSubdivision(position,piece=state.piece){
  const step=((Math.floor(position)%12)+12)%12;
  const fraction=position-Math.floor(position);
  const positions=metricPositions(piece),durations=metricDurations(piece);
  return positions[step]+fraction*durations[step];
}
function metricOffsetMs(stepIndex,piece=state.piece){
  const cycleDuration=12*playback.step;
  return (metricPositions(piece)[stepIndex]-stepIndex/12)*cycleDuration*1000;
}

function synthKey(piece=state.piece,voice=state.synthVoice){return `${piece}:${voice}`;}
function partitionKey(piece=state.piece,voice=state.synthVoice){return `${piece}:${voice}`;}
function partitionFromMotif(motif,target=null){
  const values=motif.map(value=>Math.max(0,Number(value)||0));
  const count=Math.max(0,Math.min(values.length,Math.round(Number.isFinite(target)?target:values.reduce((sum,value)=>sum+value,0))));
  if(count<=0)return values.map(()=>0);
  const ranked=values
    .map((value,index)=>({value,index}))
    .sort((a,b)=>b.value-a.value||a.index-b.index)
    .slice(0,count);
  const keep=new Set(ranked.map(item=>item.index));
  return values.map((_,index)=>keep.has(index)?1:0);
}
function ensurePartition(piece=state.piece,voice=state.synthVoice){
  const key=partitionKey(piece,voice);
  if(!state.partitionData[key]){
    state.partitionData[key]=partitionFromMotif(rhythm(piece).voices[voice].motif);
  }
  return state.partitionData[key];
}
function partitionHitCount(voice=state.synthVoice,piece=state.piece){
  return ensurePartition(piece,voice).reduce((sum,value)=>sum+(value>.5?1:0),0);
}
function setPartitionFromMotif(piece=state.piece,voice=state.synthVoice,motif,target=null){
  const partition=partitionFromMotif(motif,target);
  state.partitionData[partitionKey(piece,voice)]=partition;
  state.synthData[synthKey(piece,voice)]=synthDataFromMotif(partition);
  return partition;
}
function resetPartition(piece=state.piece,voice=state.synthVoice){
  return setPartitionFromMotif(piece,voice,rhythm(piece).voices[voice].motif);
}
function cloneSynthData(it){
  const d={density:it.density, coeffs:{}};
  for(let k=1;k<=5;k++) d.coeffs[k]={mag:it.coeffs[k].mag, phase:it.coeffs[k].phase};
  d.coeffs[6]={real:it.coeffs[6].re};
  return d;
}
function ensureSynthData(piece=state.piece,voice=state.synthVoice){
  const key=synthKey(piece,voice);
  if(!state.synthData[key]) state.synthData[key]=synthDataFromMotif(ensurePartition(piece,voice));
  return state.synthData[key];
}
function synthDataFromMotif(motif){
  const clean=motif.map(v=>Math.max(0,Number(v)||0));
  const density=clean.reduce((a,b)=>a+b,0);
  const x=density>1e-10 ? clean.map(v=>v/density) : clean.map(()=>0);
  const coeffs={};
  for(let k=1;k<=5;k++){
    let re=0,im=0;
    x.forEach((v,s)=>{
      const angle=TAU*k*s/12;
      re+=v*Math.cos(angle);
      im-=v*Math.sin(angle);
    });
    coeffs[k]={mag:Math.hypot(re,im),phase:Math.atan2(im,re)};
  }
  let real6=0;
  x.forEach((v,s)=>{real6+=v*Math.cos(Math.PI*s);});
  coeffs[6]={real:real6};
  return {density,coeffs};
}
function motifForTransform(voice){
  return state.synthEnabled ? ensurePartition(state.piece,voice).slice() : partitionFromMotif(rhythm().voices[voice].motif);
}
function centered(values){
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  return values.map(v=>v-mean);
}
function centeredCosine(a,b){
  const ac=centered(a),bc=centered(b);
  const dot=ac.reduce((sum,v,i)=>sum+v*bc[i],0);
  const na=Math.sqrt(ac.reduce((sum,v)=>sum+v*v,0));
  const nb=Math.sqrt(bc.reduce((sum,v)=>sum+v*v,0));
  return na*nb>1e-10 ? dot/(na*nb) : 0;
}
function orthogonalComplement(source,reference){
  const a=centered(source),b=centered(reference);
  const bb=b.reduce((sum,v)=>sum+v*v,0);
  const ab=a.reduce((sum,v,i)=>sum+v*b[i],0);
  let q=bb>1e-10 ? a.map((v,i)=>v-ab/bb*b[i]) : a.slice();
  let norm=Math.sqrt(q.reduce((sum,v)=>sum+v*v,0));
  if(norm<1e-8){
    const candidate=Array.from({length:12},(_,s)=>Math.sin(TAU*s/12));
    const cb=candidate.reduce((sum,v,i)=>sum+v*b[i],0);
    q=bb>1e-10 ? candidate.map((v,i)=>v-cb/bb*b[i]) : candidate;
    norm=Math.sqrt(q.reduce((sum,v)=>sum+v*v,0))||1;
  }
  const sourceDensity=source.reduce((a,b)=>a+b,0);
  const floor=-Math.min(...q)+.012*norm;
  const playable=q.map(v=>v+floor);
  const total=playable.reduce((a,b)=>a+b,0)||1;
  return playable.map(v=>v*sourceDensity/total);
}
function mirroredMotif(source,axis){
  return Array.from({length:12},(_,s)=>source[(2*axis-s+120)%12]);
}
function libraryVariantKey(piece,voice,axis){
  return `${piece}:${voice}:${axis===null?"original":`axis-${axis}`}`;
}
function normalizedMotifDistance(a,b){
  const sumA=a.reduce((sum,value)=>sum+value,0)||1;
  const sumB=b.reduce((sum,value)=>sum+value,0)||1;
  const squared=a.reduce((sum,value,index)=>{
    const delta=value/sumA-b[index]/sumB;
    return sum+delta*delta;
  },0);
  return Math.min(1,Math.sqrt(squared/2));
}
function libraryVariants(piece=state.libraryPiece,voice="D1"){
  const base=partitionFromMotif(rhythm(piece).voices[voice].motif);
  const variants=[
    {axis:null,label:"Original",motif:base,distance:0},
    ...Array.from({length:6},(_,axis)=>({
      axis,
      label:`Axe ${axis+1}–${axis+7}`,
      motif:mirroredMotif(base,axis)
    }))
  ];
  const seen=new Map();
  variants.forEach(variant=>{
    variant.distance=normalizedMotifDistance(base,variant.motif);
    const signature=variant.motif.map(value=>value.toFixed(6)).join("|");
    variant.duplicateOf=seen.get(signature)||null;
    if(!seen.has(signature))seen.set(signature,variant.label);
  });
  return variants;
}
function clearLibrarySelection(voice=null){
  const voices=voice?[voice]:["D1","D2"];
  voices.forEach(target=>{
    state.libraryLoaded[target]=null;
    document.querySelectorAll(`.variantCard.on[data-voice="${target}"]`).forEach(card=>{
      card.classList.remove("on");
      card.setAttribute("aria-pressed","false");
      const action=card.querySelector(".variantAction");
      if(action)action.textContent=`charger pour ${target}`;
    });
  });
}
function drawVariantPreview(canvas,motif,voice,axis){
  const g=canvas.getContext("2d"),W=canvas.width,cx=W/2,cy=W/2,radius=48;
  const max=Math.max(...motif,.001);
  g.clearRect(0,0,W,W);
  g.save();
  g.translate(cx,cy);
  g.strokeStyle=css("--line");
  g.lineWidth=1.2;
  g.beginPath();g.arc(0,0,radius,0,TAU);g.stroke();
  for(let s=0;s<12;s++){
    const angle=s/12*TAU-Math.PI/2,ca=Math.cos(angle),sa=Math.sin(angle);
    g.fillStyle=css("--muted");g.globalAlpha=.42;
    g.beginPath();g.arc((radius+8)*ca,(radius+8)*sa,1.5,0,TAU);g.fill();
  }
  if(axis!==null){
    const angle=axis/12*TAU-Math.PI/2,ca=Math.cos(angle),sa=Math.sin(angle);
    g.setLineDash([4,4]);g.strokeStyle=css("--accent");g.globalAlpha=.5;g.lineWidth=1.3;
    g.beginPath();g.moveTo(-(radius+13)*ca,-(radius+13)*sa);g.lineTo((radius+13)*ca,(radius+13)*sa);g.stroke();
    g.setLineDash([]);
  }
  motif.forEach((value,s)=>{
    const angle=s/12*TAU-Math.PI/2,ca=Math.cos(angle),sa=Math.sin(angle);
    g.globalAlpha=.92;g.fillStyle=color(voice);
    g.beginPath();g.arc(radius*ca,radius*sa,2.8+7*Math.sqrt(value/max),0,TAU);g.fill();
    g.strokeStyle=css("--paper");g.lineWidth=1.1;g.stroke();
  });
  g.restore();
}
async function loadLibraryVariant(piece,voice,axis){
  const restartPlayback=playback.active&&piece!==state.piece;
  if(restartPlayback)stopPlayback();
  const base=partitionFromMotif(rhythm(piece).voices[voice].motif);
  const motif=axis===null?base:mirroredMotif(base,axis);
  state.piece=piece;
  state.libraryPiece=piece;
  state.synthVoice=voice;
  state.symmetryAxis=axis===null?0:axis;
  setPartitionFromMotif(piece,voice,motif);
  state.synthEnabled=true;
  state.listeningMode="synthesis";
  state.timingMode="observed";
  state.libraryLoaded[voice]=libraryVariantKey(piece,voice,axis);
  state.transformMessage=axis===null
    ? `<b>${piece} · ${voice}</b> original chargé · l’autre voix est conservée`
    : `<b>${piece} · ${voice}</b> réflexion ${axis+1}–${axis+7} chargée · l’autre voix est conservée`;
  render();
  if(restartPlayback)await startPlayback();
}
function renderSymmetryLibrary(){
  if(!DATA.pieces.includes(state.libraryPiece))state.libraryPiece=DATA.pieces[0];
  const pieceTabs=document.getElementById("libraryPieceTabs");
  const columns=document.getElementById("variantColumns");
  pieceTabs.innerHTML="";
  DATA.pieces.forEach(piece=>{
    const button=document.createElement("button");
    button.className="tab"+(piece===state.libraryPiece?" on":"");
    button.textContent=piece;
    button.onclick=()=>{
      state.libraryPiece=piece;
      state.libraryLoaded={D1:null,D2:null};
      renderSymmetryLibrary();
    };
    pieceTabs.appendChild(button);
  });

  const counts={};
  columns.innerHTML="";
  ["D1","D2"].forEach(voice=>{
    const variants=libraryVariants(state.libraryPiece,voice);
    counts[voice]=new Set(variants.map(variant=>
      variant.motif.map(value=>value.toFixed(6)).join("|")
    )).size;
    const column=document.createElement("section");
    column.className="voiceVariantColumn";
    column.innerHTML=
      `<div class="voiceVariantHead"><h3>${voice}</h3><span>modifie uniquement ${voice}</span></div>`+
      `<div class="voiceVariantGrid"></div>`;
    const grid=column.querySelector(".voiceVariantGrid");
    variants.forEach(variant=>{
      const key=libraryVariantKey(state.libraryPiece,voice,variant.axis);
      const loaded=state.libraryLoaded[voice]===key;
      const card=document.createElement("button");
      card.className="variantCard"+(loaded?" on":"");
      card.dataset.voice=voice;
      card.setAttribute("aria-pressed",loaded?"true":"false");
      const detail=variant.axis===null
        ? "motif mesuré · densité conservée"
        : variant.duplicateOf
          ? `identique à ${variant.duplicateOf}`
          : `écart de forme ${(variant.distance*100).toFixed(1)} %`;
      card.innerHTML=
        `<canvas width="136" height="136" aria-hidden="true"></canvas>`+
        `<b>${variant.label}</b><small>${detail}</small>`+
        `<span class="variantAction">${loaded?`chargé pour ${voice}`:`charger pour ${voice}`}</span>`;
      card.onclick=()=>loadLibraryVariant(state.libraryPiece,voice,variant.axis);
      grid.appendChild(card);
      drawVariantPreview(card.querySelector("canvas"),variant.motif,voice,variant.axis);
    });
    columns.appendChild(column);
  });
  document.getElementById("libraryStatus").innerHTML=
    `<b>${state.libraryPiece}</b> — choisis une variante dans chaque colonne : D1 et D2 se combinent sans s’écraser. `+
    `D1 : ${counts.D1} dispositions distinctes ; D2 : ${counts.D2} dispositions distinctes.`;
  document.getElementById("libraryDistance").innerHTML=
    `<b>Écart de forme.</b> Δ = 100 × √[½ Σ<sub>s</sub>(x′<sub>s</sub> − x<sub>s</sub>)²], `+
    `avec x<sub>s</sub> = D<sub>s</sub>/ρ. La densité est retirée : 0 % signifie une forme identique.`;
}
function symmetryPairs(axis){
  const fixed=[],pairs=[],seen=new Set();
  for(let s=0;s<12;s++){
    if(seen.has(s))continue;
    const target=(2*axis-s+120)%12;
    seen.add(s);seen.add(target);
    if(s===target)fixed.push(s+1);
    else pairs.push([s+1,target+1]);
  }
  return {fixed,pairs};
}
function renderSymmetryGuide(){
  const axis=state.symmetryAxis;
  const select=document.getElementById("symmetryAxis");
  const guide=document.getElementById("symmetryGuide");
  const mirrorBtn=document.getElementById("mirrorBtn");
  const {fixed,pairs}=symmetryPairs(axis);
  select.value=String(axis);
  mirrorBtn.textContent=`Refléter ${state.synthVoice}`;
  guide.innerHTML=
    `<b>A<sub>${axis+1}–${axis+7}</sub></b> : fixes {${fixed.join(", ")}} ; `+
    `permutation ${pairs.map(([a,b])=>`${a}↔${b}`).join(", ")} ; ρ′ = ρ.`;
  select.onchange=()=>{
    state.symmetryAxis=Number(select.value);
    state.transformMessage="";
    document.getElementById("transformFeedback").textContent="";
    renderSymmetryGuide();
    drawCircle();
  };
}
function applySynthMotif(motif,message){
  clearLibrarySelection(state.synthVoice);
  setPartitionFromMotif(state.piece,state.synthVoice,motif);
  state.synthEnabled=true;
  state.listeningMode="synthesis";
  state.timingMode="observed";
  state.transformMessage=message;
  renderSynthControls();
  refreshSynthViews();
}
function applyOrthogonalProjection(){
  const voice=state.synthVoice,other=voice==="D1"?"D2":"D1";
  const projected=orthogonalComplement(motifForTransform(voice),motifForTransform(other));
  const cos=centeredCosine(projected,motifForTransform(other));
  applySynthMotif(
    projected,
    `<b>${voice}</b> projeté dans le complément orthogonal de ${other} · cosinus centré ${cos.toFixed(3)}`
  );
}
function applyMirrorSymmetry(){
  const axis=state.symmetryAxis;
  const voice=state.synthVoice;
  applySynthMotif(
    mirroredMotif(motifForTransform(voice),axis),
    `<b>${voice}</b> réfléchi autour de l’axe ${axis+1}–${axis+7} · densité conservée, aucune frappe dupliquée`
  );
}
function synthCoeffArray(data){
  const out=[{k:0,re:1,im:0,mag:1,phase:0}];
  for(let k=1;k<=5;k++){
    const c=data.coeffs[k], re=c.mag*Math.cos(c.phase), im=c.mag*Math.sin(c.phase);
    out[k]={k,re,im,mag:c.mag,phase:c.phase};
  }
  const r=data.coeffs[6].real;
  out[6]={k:6,re:r,im:0,mag:Math.abs(r),phase:r>=0?0:Math.PI};
  return out;
}
function curveValueFromCoeffs(coeffs,density,t,clamp=true){
  const theta=TAU*t/12, N=12;
  let y=1;
  for(let k=1;k<=5;k++){
    const c=coeffs[k];
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  y += coeffs[6].re*Math.cos(6*theta);
  const value=y/N*(density??1);
  return clamp?Math.max(0,value):value;
}
function curveValue(it,t){
  const C=it.coeffs, theta=TAU*t/12, N=12;
  let y=C[0].re;
  for(let k=1;k<=5;k++){
    const c=C[k];
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  const c6=C[6];
  y += c6.re*Math.cos(6*theta)-c6.im*Math.sin(6*theta);
  return Math.max(0,y/N*(it.density??1));
}
function synthMotif(voice,piece=state.piece){
  const data=ensureSynthData(piece,voice), coeffs=synthCoeffArray(data);
  return Array.from({length:12},(_,s)=>curveValueFromCoeffs(coeffs,data.density,s,true));
}
function synthTargetHitCount(voice=state.synthVoice,piece=state.piece){
  const data=ensureSynthData(piece,voice);
  return Math.max(0,Math.min(12,Math.round(Number(data.density)||0)));
}
function quantizedSynthMotif(voice,piece=state.piece){
  const values=synthMotif(voice,piece);
  const target=synthTargetHitCount(voice,piece);
  return partitionFromMotif(values,target);
}
function voiceView(voice,piece=state.piece){
  if(!state.synthEnabled) return rhythm(piece).voices[voice];
  const data=ensureSynthData(piece,voice);
  return {voice,density:ensurePartition(piece,voice).reduce((sum,value)=>sum+value,0),motif:ensurePartition(piece,voice).slice(),takes:rhythm(piece).voices[voice].takes,coeffs:synthCoeffArray(data),synthetic:true};
}
function curveValueForVoice(voice,t){
  if(!state.synthEnabled) return curveValue(rhythm().voices[voice],t);
  const data=ensureSynthData(state.piece,voice), coeffs=synthCoeffArray(data);
  return curveValueFromCoeffs(coeffs,data.density,t,true);
}
function motifForPlayback(voice){
  if(state.synthEnabled)return ensurePartition(state.piece,voice).slice();
  const it=rhythm().voices[voice];
  const motif=it.motif.slice();
  const target=Math.max(1,Math.min(12,Math.round(it.density||motif.reduce((a,b)=>a+b,0))));
  const ranked=motif
    .map((value,index)=>({value,index}))
    .filter(item=>item.value>=.18)
    .sort((a,b)=>b.value-a.value)
    .slice(0,target);
  const keep=new Set(ranked.map(item=>item.index));
  return motif.map((value,index)=>keep.has(index)?Math.max(.82,value):0);
}
function phraseModeLabel(mode=state.phraseMode){
  return {
    mean12:"cycle 12 moyen",
    cycleA:"cycle A · 12",
    cycleB:"cycle B · 12",
    phrase24:"phrase A+B · 24"
  }[mode]||"cycle 12 moyen";
}
function displayStepCount(){
  return state.phraseMode==="phrase24" ? 24 : 12;
}
function phrase24Motif(voice,piece=state.piece){
  const motif=rhythm(piece).phrase24?.[voice]?.motif;
  return Array.isArray(motif)&&motif.length>=24 ? motif.slice(0,24) : null;
}
function displayMotif(voice,piece=state.piece){
  if(state.phraseMode==="mean12")return voiceView(voice,piece).motif.slice();
  const phrase=phrase24Motif(voice,piece);
  if(!phrase)return rhythm(piece).voices[voice].motif.slice();
  if(state.phraseMode==="cycleA")return phrase.slice(0,12);
  if(state.phraseMode==="cycleB")return phrase.slice(12,24);
  return phrase;
}
function coefficientsFromMotif(motif){
  const clean=motif.map(value=>Math.max(0,Number(value)||0));
  const n=clean.length, density=clean.reduce((sum,value)=>sum+value,0);
  const x=density>1e-10 ? clean.map(value=>value/density) : clean.map(()=>0);
  const coeffs=[];
  for(let k=0;k<=Math.floor(n/2);k++){
    let re=0,im=0;
    x.forEach((value,s)=>{
      const angle=TAU*k*s/n;
      re+=value*Math.cos(angle);
      im-=value*Math.sin(angle);
    });
    coeffs.push({k,re,im,mag:Math.hypot(re,im),phase:Math.atan2(im,re)});
  }
  return {coeffs,density};
}
function displayVoiceView(voice,piece=state.piece){
  if(state.phraseMode==="mean12")return voiceView(voice,piece);
  const motif=displayMotif(voice,piece);
  const {coeffs,density}=coefficientsFromMotif(motif);
  return {
    voice,
    motif,
    density,
    coeffs,
    takes:rhythm(piece).phrase24?.[voice]?.takes||rhythm(piece).voices[voice].takes,
    phrase:true
  };
}
function displayDurations(piece=state.piece){
  const base=metricDurations(piece);
  if(displayStepCount()===12)return base;
  return base.map(value=>value/2).concat(base.map(value=>value/2));
}
function displayPositions(piece=state.piece){
  const base=metricPositions(piece);
  if(displayStepCount()===12)return base;
  return base.map(value=>value/2).concat(base.map(value=>.5+value/2));
}
function displayPhaseAtSubdivision(position,piece=state.piece){
  const steps=displayStepCount();
  if(steps===12)return metricPhaseAtSubdivision(position,piece);
  const step=((Math.floor(position)%steps)+steps)%steps;
  const fraction=position-Math.floor(position);
  const positions=displayPositions(piece),durations=displayDurations(piece);
  return positions[step]+fraction*durations[step];
}
function curveValueFromCoeffArray(coeffs,density,t,n){
  const theta=TAU*t/n;
  let y=coeffs[0]?.re??1;
  const half=Math.floor(n/2);
  for(let k=1;k<half;k++){
    const c=coeffs[k]||{re:0,im:0};
    y += 2*(c.re*Math.cos(k*theta)-c.im*Math.sin(k*theta));
  }
  if(n%2===0){
    const c=coeffs[half]||{re:0,im:0};
    y += c.re*Math.cos(half*theta)-c.im*Math.sin(half*theta);
  }
  return Math.max(0,y/n*(density??1));
}
function displayCurveValue(voice,t){
  const view=displayVoiceView(voice);
  return curveValueFromCoeffArray(view.coeffs,view.density,t,view.motif.length);
}
function subdivisionAtMetricPhase(phase){
  return subdivisionAtPositions(phase,metricPositions(),metricDurations());
}
function subdivisionAtPositions(phase,positions,durations=null){
  const resolvedDurations=durations||positions.map((position,index)=>{
    const next=positions[index+1]??1;
    return Math.max(1e-6,next-position);
  });
  const wrapped=((phase%1)+1)%1;
  const n=positions.length;
  let step=n-1;
  for(let s=0;s<n;s++){
    if(wrapped>=positions[s]&&wrapped<positions[s]+resolvedDurations[s]){
      step=s;
      break;
    }
  }
  return step+(wrapped-positions[step])/(resolvedDurations[step]||1/12);
}
function subdivisionAtCyclePhase(phase,cycle=selectedCycle()){
  if(!cycle?.positions?.length)return subdivisionAtMetricPhase(phase);
  return subdivisionAtPositions(phase,cycle.positions);
}
function primaryMedia(){
  const preferred=selectedVoices().map(voice=>playback.media[voice]).find(Boolean);
  return preferred||playback.media.D1||Object.values(playback.media)[0]||null;
}
function corpusPlaybackPhase(){
  if(!["recording","reconstruction"].includes(playback.kind))return null;
  const audio=primaryMedia();
  const cycle=playback.cycle||selectedCycle();
  if(!audio||!cycle||!Number.isFinite(audio.currentTime))return null;
  const duration=cycle.duration||(cycle.end-cycle.start)||1;
  return Math.max(0,Math.min(.999999,(audio.currentTime-cycle.start)/duration));
}
function corpusPlayheadPosition(){
  const phase=corpusPlaybackPhase();
  if(phase===null)return null;
  return subdivisionAtCyclePhase(phase,playback.cycle||selectedCycle());
}
function playheadPosition(){
  if(!playback.active)return null;
  if(playback.kind==="recording"||playback.kind==="reconstruction")return corpusPlayheadPosition();
  if(!playback.ctx)return null;
  const now=playback.ctx.currentTime;
  if(now<playback.startTime)return 0;
  let event=null;
  for(const candidate of playback.timeline){
    if(candidate.time<=now&&now<candidate.end){event=candidate;break;}
  }
  if(!event){
    const past=playback.timeline.filter(candidate=>candidate.time<=now);
    event=past[past.length-1]||playback.timeline[0];
  }
  if(!event)return 0;
  const fraction=Math.max(0,Math.min(1,(now-event.time)/(event.end-event.time||1)));
  return event.step+fraction;
}
function playheadCirclePhase(position){
  const phase=corpusPlaybackPhase();
  if(phase!==null)return phase;
  return position===null ? null : metricPhaseAtSubdivision(position);
}
function playheadDisplayPosition(){
  const position=playheadPosition();
  if(position===null)return null;
  if(displayStepCount()===12)return position;
  if((playback.kind==="recording"||playback.kind==="reconstruction")&&corpusPlaybackPhase()!==null){
    const cycle=playback.cycle||selectedCycle();
    const half=((cycle?.index??selectedCycleIndex())%2+2)%2;
    return half*12+position;
  }
  if(playback.kind==="synthetic"&&playback.ctx){
    const cycleDuration=12*playback.step;
    const elapsed=Math.max(0,playback.ctx.currentTime-playback.startTime);
    const half=Math.floor(elapsed/cycleDuration)%2;
    return half*12+(position%12);
  }
  return position%24;
}
function playheadDisplayPhase(){
  const position=playheadDisplayPosition();
  return position===null ? null : displayPhaseAtSubdivision(position);
}

function latexForRhythm(r){
  const n=displayStepCount();
  const maxK=Math.floor(n/2);
  const voices=visibleVoices();
  const vectorRows=voices.map(voice=>String.raw`\operatorname{${voice}}_{\mathrm{${r.piece}}}(s)`).join(String.raw`\\`);
  const coefficientRows=voices.map(voice=>String.raw`\mathbf{c}_{${voice}}^{(${n})} &= ${texVector(displayVoiceView(voice,r.piece))}`).join(String.raw`\\`);
  return String.raw`
\[
\begin{aligned}
\text{fenêtre:}\quad
N &= ${n}\quad(\text{${phraseModeLabel()}})\\
\text{points du cercle:}\quad
z_s &= e^{2\pi i s/N}\\
\text{motif normalisé:}\quad
x_{v,s} &= \frac{\operatorname{V}_{v}(s)}{\rho_v},
\qquad \rho_v=\sum_{s=0}^{N-1}\operatorname{V}_{v}(s)\\
\text{coefficients:}\quad
c_{v,k}^{(N)} &= \sum_{s=0}^{N-1}x_{v,s}\,z_s^{-k},
\qquad k=0,\ldots,${maxK}\\
\operatorname{Rythme}_{\mathrm{${r.piece}}}(s)
  &= \begin{bmatrix}${vectorRows}\end{bmatrix},
  \qquad s\in\{0,\ldots,${n-1}\}\\
\operatorname{V}_{v,\mathrm{${r.piece}}}^{(${n})}(s)
  &= \frac{\rho_v}{N}\left[
      1+2\,\Re\!\left(\sum_{k=1}^{${maxK-1}}c_{v,k}^{(N)}z_s^k\right)
      + c_{v,${maxK}}^{(N)}(-1)^s
    \right]\\
${coefficientRows}
\end{aligned}
\]`;
}

function queueMath(){
  if(window.MathJax?.typesetPromise){
    MathJax.typesetClear?.();
    MathJax.typesetPromise().catch(()=>{});
  }else{
    setTimeout(queueMath,120);
  }
}

function renderPhraseModeControls(){
  document.querySelectorAll("[data-phrase-mode]").forEach(button=>{
    const mode=button.dataset.phraseMode;
    button.className="smallBtn"+(state.phraseMode===mode?" on":"");
    button.onclick=()=>{
      state.phraseMode=mode;
      renderPhraseModeControls();
      renderSynthControls();
      syncSynthModeButtons();
      drawCircle();
      drawCurve();
      drawMicroTiming();
      renderEquation();
      renderCoefTable();
      renderDjembeLoopsReference();
      queueMath();
    };
  });
}

function syncSynthModeButtons(){
  const original=document.getElementById("originalModeBtn"), synth=document.getElementById("synthModeBtn");
  if(!original||!synth)return;
  original.className="smallBtn"+(!state.synthEnabled?" on":"");
  synth.className="smallBtn"+(state.synthEnabled?" on":"");
  const circleMode=document.getElementById("circleMode");
  const curveMode=document.getElementById("curveMode");
  const liveTag=document.getElementById("synthLiveTag");
  if(circleMode){
    circleMode.textContent=state.synthEnabled?"Partition synthèse":"Signal Polak";
    circleMode.className="modeTag"+(state.synthEnabled?" live":"");
  }
  if(curveMode){
    curveMode.textContent=state.phraseMode==="mean12"
      ? (state.synthEnabled?"Partition jouée + signature":"Signal Polak observé")
      : phraseModeLabel();
    curveMode.className="modeTag"+(state.synthEnabled&&state.phraseMode==="mean12"?" live":"");
  }
  if(liveTag){
    liveTag.textContent=state.synthEnabled
      ? `${playback.active&&playback.kind==="synthetic"?"LIVE":"partition"} · ${state.synthVoice}`
      : "coefficients Polak";
    liveTag.className="modeTag"+(state.synthEnabled?" live":"");
  }
}

let equationRefreshTimer=null;
function scheduleEquationRefresh(){
  clearTimeout(equationRefreshTimer);
  equationRefreshTimer=setTimeout(()=>{
    renderEquation();
    renderCoefTable();
    queueMath();
  },140);
}

function refreshSynthViews(){
  syncSynthModeButtons();
  drawCircle();
  drawCurve();
  renderDjembeLoopsReference();
  scheduleEquationRefresh();
}

function renderSynthVoicePicker(){
  const box=document.getElementById("synthVoicePicker");
  box.innerHTML="";
  ["D1","D2"].forEach(voice=>{
    const b=document.createElement("button");
    b.className="voiceBtn"+(state.synthVoice===voice?" on":"");
    b.textContent=`éditer ${voice}`;
    b.onclick=()=>{
      state.synthVoice=voice;
      state.transformMessage="";
      renderSynthControls();
      syncSynthModeButtons();
    };
    box.appendChild(b);
  });
}

function renderPartitionEditor(){
  const box=document.getElementById("partitionEditor");
  if(!box)return;
  ["D1","D2"].forEach(voice=>{
    ensurePartition(state.piece,voice);
    ensureSynthData(state.piece,voice);
  });
  const rows=["D1","D2"].map(voice=>{
    const motif=ensurePartition(state.piece,voice);
    return `<div class="partitionRow" data-voice="${voice}"><b>${voice}</b>${
      motif.map((value,index)=>
        `<button class="partitionStep${value>.5?" on":""}" data-partition-voice="${voice}" data-step="${index}" aria-pressed="${value>.5?"true":"false"}">${index+1}</button>`
      ).join("")
    }</div>`;
  }).join("");
  box.innerHTML=
    `<div class="partitionEditorHead"><div><h3>Partition de travail</h3><p>Base jouée par le synthé : clique une case pour ajouter/enlever une frappe.</p></div>`+
    `<div class="partitionActions"><button class="smallBtn" data-reset-partition="D1">réinit D1</button><button class="smallBtn" data-reset-partition="D2">réinit D2</button></div></div>`+
    `<div class="partitionRows">${rows}</div>`;
  box.querySelectorAll("[data-partition-voice]").forEach(button=>{
    button.onclick=()=>{
      const voice=button.dataset.partitionVoice;
      const step=Number(button.dataset.step);
      const motif=ensurePartition(state.piece,voice).slice();
      motif[step]=motif[step]>.5?0:1;
      state.partitionData[partitionKey(state.piece,voice)]=motif;
      state.synthData[synthKey(state.piece,voice)]=synthDataFromMotif(motif);
      state.synthEnabled=true;
      state.listeningMode="synthesis";
      state.timingMode="observed";
      state.synthVoice=voice;
      state.transformMessage=`<b>${voice}</b> modifié dans la partition de travail`;
      renderSynthControls();
      renderTimingControls();
      refreshSynthViews();
      syncPlaybackUI();
    };
  });
  box.querySelectorAll("[data-reset-partition]").forEach(button=>{
    button.onclick=()=>{
      const voice=button.dataset.resetPartition;
      resetPartition(state.piece,voice);
      state.synthEnabled=true;
      state.listeningMode="synthesis";
      state.timingMode="observed";
      state.synthVoice=voice;
      state.transformMessage=`<b>${voice}</b> réinitialisé depuis les positions stables Polak`;
      renderSynthControls();
      renderTimingControls();
      refreshSynthViews();
      syncPlaybackUI();
    };
  });
}

function renderSynthControls(){
  renderSynthVoicePicker();
  renderPartitionEditor();
  syncSynthModeButtons();
  document.getElementById("originalModeBtn").onclick=()=>{
    state.synthEnabled=false;
    clearLibrarySelection();
    refreshSynthViews();
  };
  document.getElementById("synthModeBtn").onclick=()=>{
    state.synthEnabled=true;
    if(state.listeningMode!=="synthesis"){
      if(playback.active)stopPlayback();
      if(referencePlayback.active)stopReferencePlayback();
      state.listeningMode="synthesis";
      state.timingMode="observed";
    }
    renderTimingControls();
    refreshSynthViews();
    syncPlaybackUI();
  };
  document.getElementById("resetSynthBtn").onclick=()=>{
    resetPartition(state.piece,state.synthVoice);
    state.synthEnabled=true;
    state.listeningMode="synthesis";
    state.timingMode="observed";
    state.libraryPiece=state.piece;
    state.libraryLoaded[state.synthVoice]=libraryVariantKey(state.piece,state.synthVoice,null);
    state.transformMessage=`<b>${state.synthVoice}</b> revenu à la partition Polak de travail`;
    renderSynthControls();
    renderTimingControls();
    renderSymmetryLibrary();
    refreshSynthViews();
  };
  const other=state.synthVoice==="D1"?"D2":"D1";
  const orthogonalBtn=document.getElementById("orthogonalBtn");
  orthogonalBtn.textContent=`⊥ ${state.synthVoice} orthogonal à ${other}`;
  orthogonalBtn.onclick=applyOrthogonalProjection;
  document.getElementById("mirrorBtn").onclick=applyMirrorSymmetry;
  renderSymmetryGuide();
  const synthScopeNote=state.phraseMode==="mean12"
    ? ""
    : `<b>Note.</b> Les contrôles modifient le cycle 12 moyen. La fenêtre ${phraseModeLabel()} affiche les données observées A/B/24.`;
  document.getElementById("transformFeedback").innerHTML=
    state.transformMessage||synthScopeNote;

  const quantizeBox=document.getElementById("synthThresholdControl");
  quantizeBox.innerHTML=
    `<div><b>source jouée</b><small>partition de travail</small></div>`+
    `<span class="quantizeBadge">cases D1/D2</span>`+
    `<output>${partitionHitCount()} frappes</output>`;

  const data=ensureSynthData();
  const rows=[];
  rows.push(`<div class="coefRow density"><b>ρ<small>densité</small></b><span>val</span><input type="range" min="0" max="8" step="0.05" value="${data.density}" data-kind="density"><output data-out="density">${data.density.toFixed(2)}</output></div>`);
  for(let k=1;k<=5;k++){
    const c=data.coeffs[k], deg=phaseDeg(c.phase);
    rows.push(`<div class="coefRow"><b>c${k}<small>${DATA.coefNames[String(k)]}</small></b><span>|c|</span><input type="range" min="0" max="1.2" step="0.001" value="${c.mag}" data-kind="mag" data-k="${k}"><output data-out="mag-${k}">${c.mag.toFixed(3)}</output><span>φ</span><input type="range" min="-180" max="180" step="1" value="${deg}" data-kind="phase" data-k="${k}"><output data-out="phase-${k}">${deg.toFixed(0)}°</output></div>`);
  }
  rows.push(`<div class="coefRow density"><b>c6<small>alternance max.</small></b><span>val</span><input type="range" min="-1" max="1" step="0.001" value="${data.coeffs[6].real}" data-kind="real6"><output data-out="real6">${data.coeffs[6].real.toFixed(3)}</output></div>`);
  document.getElementById("synthControls").innerHTML=rows.join("");

  document.querySelectorAll("#synthControls input").forEach(input=>{
    input.oninput=()=>{
      const d=ensureSynthData(), kind=input.dataset.kind, k=Number(input.dataset.k), value=Number(input.value);
      state.synthEnabled=true;
      state.listeningMode="synthesis";
      state.timingMode="observed";
      clearLibrarySelection(state.synthVoice);
      state.transformMessage="";
      if(kind==="density"){
        d.density=value;
        document.querySelector('[data-out="density"]').textContent=value.toFixed(2);
        document.querySelector("#synthThresholdControl output").textContent=`${partitionHitCount()} frappes`;
      }else if(kind==="mag"){
        d.coeffs[k].mag=value;
        document.querySelector(`[data-out="mag-${k}"]`).textContent=value.toFixed(3);
      }else if(kind==="phase"){
        d.coeffs[k].phase=value*Math.PI/180;
        document.querySelector(`[data-out="phase-${k}"]`).textContent=`${value.toFixed(0)}°`;
      }else if(kind==="real6"){
        d.coeffs[6].real=value;
        document.querySelector('[data-out="real6"]').textContent=value.toFixed(3);
      }
      state.partitionData[partitionKey(state.piece,state.synthVoice)]=quantizedSynthMotif(state.synthVoice,state.piece);
      document.querySelector("#synthThresholdControl output").textContent=`${partitionHitCount()} frappes`;
      renderPartitionEditor();
      renderTimingControls();
      refreshSynthViews();
    };
  });
}

function renderScopePicker(){
  const box=document.getElementById("scopePicker");
  box.innerHTML="";
  [
    ["ensemble","Ensemble"],
    ["dunun","Dunun"],
  ].forEach(([value,label])=>{
    const b=document.createElement("button");
    b.className="voiceBtn"+(state.scope===value?" on":"");
    b.textContent=label;
    b.onclick=()=>{
      state.scope=value;
      if(value==="dunun"&&["ensemble","jembe","J1","J2"].includes(state.playVoice)){
        state.playVoice="dunun";
      }else if(value==="ensemble"&&state.playVoice==="dunun"){
        state.playVoice="ensemble";
      }
      renderScopePicker();
      renderVoicePicker();
      renderRecordingCycleControl();
      drawCircle();
      drawCurve();
      renderDjembeLoopsReference();
      drawMicroTiming();
      renderEquation();
      renderCoefTable();
      queueMath();
      syncPlaybackUI();
    };
    box.appendChild(b);
  });
}

function renderVoicePicker(){
  const box=document.getElementById("voicePicker");
  box.innerHTML="";
  const labels={
    J1:"J1 · djembé 1",
    J2:"J2 · djembé 2",
    D1:"D1 · dunun grave",
    D2:"D2 · dunun réponse",
  };
  ALL_VOICES.forEach(voice=>{
    const b=document.createElement("button");
    const selected=!!state.selectedVoiceSet[voice];
    b.className="voiceBtn "+(selected?"on":"off");
    b.textContent=labels[voice]||voice;
    b.setAttribute("aria-pressed",selected?"true":"false");
    b.onclick=()=>{
      const active=ALL_VOICES.filter(item=>state.selectedVoiceSet[item]);
      if(selected&&active.length===1)return;
      state.selectedVoiceSet[voice]=!selected;
      renderVoicePicker();
      syncCorpusVoiceGains();
      drawCircle();
      drawCurve();
      drawMicroTiming();
      syncPlaybackUI();
    };
    box.appendChild(b);
  });
  document.getElementById("playCaption").textContent =
    playback.active||referencePlayback.active
      ? `Lecture · ${playVoiceLabel()} · ${timingModeLabel()}.`
      : `Voix actives : ${playVoiceLabel()}.`;
}

async function selectPiece(piece){
  if(piece===state.piece)return;
  const wasPlaying=playback.active;
  if(wasPlaying)stopPlayback();
  if(referencePlayback.active)stopReferencePlayback();
  state.piece=piece;
  state.libraryPiece=piece;
  state.libraryLoaded={D1:null,D2:null};
  state.transformMessage="";
  render();
  if(wasPlaying)await startPlayback();
}
function renderTabs(){
  const nav=document.getElementById("tabs");
  nav.innerHTML="";
  DATA.rhythms.forEach(r=>{
    const b=document.createElement("button");
    b.className="tab"+(r.piece===state.piece?" on":"");
    b.textContent=r.piece;
    b.onclick=()=>selectPiece(r.piece);
    nav.appendChild(b);
  });
}

function drawCircle(){
  const r=rhythm(), cv=document.getElementById("circle"), g=cv.getContext("2d");
  const W=cv.width, cx=W/2, cy=W/2;
  const voices=visibleVoices();
  const outer=voices.length>2?238:218;
  const gap=voices.length>2?45:72;
  const ringMap=new Map(voices.map((voice,index)=>[voice,outer-index*gap]));
  const radii=[...ringMap.values()];
  const inner=Math.min(...radii);
  const steps=displayStepCount();
  const p=playheadDisplayPosition(), currentStep=p===null?null:Math.floor(p)%steps;
  const positions=displayPositions(),durations=displayDurations();
  const observedMode=state.timingMode==="observed";
  const audibleVoices=new Set(selectedVoices());
  document.querySelectorAll("[data-voice-legend]").forEach(item=>{
    item.style.display=voices.includes(item.dataset.voiceLegend)?"inline-flex":"none";
  });
  g.clearRect(0,0,W,W);
  g.save();
  g.translate(cx,cy);

  g.strokeStyle=css("--line");
  g.lineWidth=1.2;
  radii.forEach(rad=>{g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();});

  if(observedMode){
    g.save();
    g.strokeStyle=css("--muted");
    g.globalAlpha=.32;
    g.lineWidth=1.2;
    g.setLineDash([3,4]);
    for(let s=0;s<steps;s++){
      const a=s/steps*TAU-Math.PI/2,ca=Math.cos(a),sa=Math.sin(a);
      g.beginPath();
      g.moveTo((inner-23)*ca,(inner-23)*sa);
      g.lineTo((outer+17)*ca,(outer+17)*sa);
      g.stroke();
    }
    g.restore();
  }

  g.save();
  g.strokeStyle=css("--line2");
  g.globalAlpha=.85;
  g.lineWidth=1;
  for(let s=0;s<steps;s++){
    const a=positions[s]*TAU-Math.PI/2,ca=Math.cos(a),sa=Math.sin(a);
    g.beginPath();
    g.moveTo((inner-20)*ca,(inner-20)*sa);
    g.lineTo((outer+12)*ca,(outer+12)*sa);
    g.stroke();
  }
  g.restore();

  if(steps===12){
    const axisAngle=positions[state.symmetryAxis]*TAU-Math.PI/2;
    const axisX=Math.cos(axisAngle),axisY=Math.sin(axisAngle),axisRadius=outer+34;
    g.save();
    g.setLineDash([7,6]);
    g.strokeStyle=css("--accent");
    g.globalAlpha=.48;
    g.lineWidth=1.7;
    g.beginPath();
    g.moveTo(-axisRadius*axisX,-axisRadius*axisY);
    g.lineTo(axisRadius*axisX,axisRadius*axisY);
    g.stroke();
    g.restore();
    g.fillStyle=css("--accent");
    g.globalAlpha=.78;
    g.font="700 11px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(`A ${state.symmetryAxis+1}–${state.symmetryAxis+7}`,0,24);
  }

  for(let s=0;s<steps;s++){
    const a=positions[s]*TAU-Math.PI/2;
    const ca=Math.cos(a), sa=Math.sin(a);
    const fixed=steps===12&&(s===state.symmetryAxis||s===state.symmetryAxis+6);
    g.fillStyle=fixed?css("--accent"):css("--muted");
    g.globalAlpha=fixed ? .9 : .55;
    g.beginPath();g.arc((outer+30)*ca,(outer+30)*sa,fixed?4:2.2,0,TAU);g.fill();
    g.globalAlpha=1;
    g.fillStyle=fixed?css("--accent"):css("--muted");
    g.font=fixed?"800 12px Inter, system-ui, sans-serif":"12px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(String(s+1),(outer+49)*ca,(outer+49)*sa);
  }

  voices.forEach(voice=>{
    const rad=ringMap.get(voice);
    const it=displayVoiceView(voice,r.piece), values=it.motif, mx=Math.max(...values,0.001);
    const audible=audibleVoices.has(voice);
    g.strokeStyle=color(voice);
    g.globalAlpha=audible ? .72 : .2;
    g.lineWidth=2.2;
    g.beginPath();g.arc(0,0,rad,0,TAU);g.stroke();
    values.forEach((v,s)=>{
      const a=positions[s]*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
      const size=5+16*Math.sqrt(v/mx);
      if(audible&&s===currentStep&&v>.045){
        g.globalAlpha=.16;
        g.fillStyle=color(voice);
        g.beginPath();g.arc(rad*ca,rad*sa,size+10,0,TAU);g.fill();
      }
      g.globalAlpha=audible ? .96 : .28;
      g.fillStyle=color(voice);
      g.beginPath();g.arc(rad*ca,rad*sa,size,0,TAU);g.fill();
      g.globalAlpha=audible ? .95 : .3;
      g.strokeStyle=css("--paper");
      g.lineWidth=1.7;
      g.stroke();
    });
    g.globalAlpha=1;
    g.fillStyle=color(voice);
    g.font="700 15px Inter, system-ui, sans-serif";
    g.textAlign="left";g.textBaseline="middle";
    g.fillText(voice,rad+18,0);
  });

  if(p!==null){
    const a=playheadDisplayPhase()*TAU-Math.PI/2, ca=Math.cos(a), sa=Math.sin(a);
    let from=inner-42, to=outer+28;
    const activeSelection=selectedVoices();
    if(activeSelection.length===1&&ringMap.has(activeSelection[0])){
      const rad=ringMap.get(activeSelection[0]);
      from=rad-35;to=rad+30;
    }
    g.globalAlpha=1;
    g.strokeStyle=css("--ink");
    g.lineWidth=2.8;
    g.beginPath();g.moveTo(from*ca,from*sa);g.lineTo(to*ca,to*sa);g.stroke();
    g.fillStyle=css("--ink");
    g.beginPath();g.arc(to*ca,to*sa,5.2,0,TAU);g.fill();
    g.font="700 12px Inter, system-ui, sans-serif";
    g.textAlign="center";g.textBaseline="middle";
    g.fillText(playVoiceLabel(),0,0);
  }

  g.restore();
  const triplet=[0,1,2].map(q=>durations.filter((_,s)=>s%3===q).reduce((sum,value)=>sum+value,0));
  const tripletTotal=triplet.reduce((sum,value)=>sum+value,0)||1;
  const tripletText=triplet.map(value=>(value/tripletTotal*100).toFixed(1)+" %").join(" / ");
  const key=document.getElementById("metricKey");
  key.style.display=observedMode?"inline-flex":"none";
  const title=document.getElementById("wheelTitle");
  if(title)title.textContent=`Cercle métrique · ${steps} subdivisions`;
  const temporalCaption=["recording","reconstruction"].includes(state.listeningMode)
    ? `La barre noire suit ${selectedCycleLabel()} de la prise ${selectedRecording()?.take??"—"} ; les points montrent ${phraseModeLabel()}.`
    : observedMode
      ? "Les traits pointillés conservent la grille isochrone ; les points suivent les positions moyennes mesurées dans l’ensemble."
      : `Les ${steps} positions sont espacées régulièrement dans la fenêtre.`;
  const phraseCaption=state.phraseMode==="mean12"
    ? "Polak 12 : un cycle moyen."
    : state.phraseMode==="phrase24"
      ? "Phrase 24 : deux cycles Polak successifs A+B."
      : `${phraseModeLabel()} : moitié extraite de la phrase A+B.`;
  const synthCaption=state.synthEnabled&&state.phraseMode!=="mean12"
    ? " Le synthé 12 reste audible, mais cette lecture affiche les profils A/B observés."
    : "";
  document.getElementById("circleCaption").innerHTML =
    `<b>${r.piece}</b> — prises appariées ensemble : ${r.takes.join(", ")}. `+
    `${phraseCaption} ${temporalCaption}${synthCaption} `+
    `Proportions temporelles moyennes : ${tripletText}. `+
    `${state.synthEnabled&&state.phraseMode==="mean12"?"Densités synthétisées":"Densités affichées"} : ${
      voices.map(voice=>`${voice} ${displayVoiceView(voice).density.toFixed(2)}`).join(", ")
    } frappes/fenêtre.`;
}

function drawCurve(){
  const cv=document.getElementById("curve"), g=cv.getContext("2d");
  const W=cv.width,H=cv.height,left=50,right=20,top=18,bottom=42,span=W-left-right;
  const steps=displayStepCount();
  g.clearRect(0,0,W,H);
  const samples=[];
  visibleVoices().forEach(voice=>{
    const it=displayVoiceView(voice);
    for(let i=0;i<=240;i++) samples.push(curveValueFromCoeffArray(it.coeffs,it.density,i/240*steps,steps));
    samples.push(...it.motif);
    if(state.phraseMode==="mean12")samples.push(...motifForPlayback(voice));
  });
  const ymax=Math.max(...samples,0.05)*1.14;
  const xOf=t=>left+(t/steps)*span;
  const yOf=v=>H-bottom-(v/ymax)*(H-top-bottom);

  g.strokeStyle=css("--line");g.lineWidth=1;
  g.beginPath();g.moveTo(left,top);g.lineTo(left,H-bottom);g.lineTo(W-right,H-bottom);g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="center";g.textBaseline="top";
  for(let s=0;s<steps;s++){
    const x=xOf(s);
    g.globalAlpha=.45;g.strokeStyle=css("--line2");g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
    g.globalAlpha=1;g.fillStyle=css("--muted");g.fillText(String(s+1),x,H-bottom+10);
  }
  g.save();
  g.translate(13,H/2);g.rotate(-Math.PI/2);
  g.textAlign="center";g.textBaseline="middle";g.fillStyle=css("--muted");
  g.fillText("probabilité de frappe",0,0);
  g.restore();
  g.textAlign="center";g.textBaseline="bottom";g.fillStyle=css("--muted");
  g.fillText(steps===24?"subdivision de la phrase A+B (1–24)":"subdivision du cycle (1–12)",left+span/2,H-5);

  visibleVoices().forEach(voice=>{
    const it=displayVoiceView(voice), col=color(voice);
    const audible=selectedVoices().includes(voice);
    g.strokeStyle=col;g.lineWidth=2.6;g.globalAlpha=audible ? .92 : .25;
    if(state.synthEnabled&&state.phraseMode==="mean12"){g.setLineDash([8,5]);}else{g.setLineDash([]);}
    g.beginPath();
    for(let i=0;i<=240;i++){
      const t=i/240*steps, x=xOf(t), y=yOf(curveValueFromCoeffArray(it.coeffs,it.density,t,steps));
      if(i===0)g.moveTo(x,y);else g.lineTo(x,y);
    }
    g.stroke();
    g.setLineDash([]);
    it.motif.forEach((v,s)=>{
      g.globalAlpha=audible ? .95 : .25;g.fillStyle=col;
      g.beginPath();g.arc(xOf(s),yOf(v),5.2,0,TAU);g.fill();
      g.strokeStyle=css("--paper");g.lineWidth=1.4;g.stroke();
    });
    if(state.phraseMode==="mean12"){
      motifForPlayback(voice).forEach((v,s)=>{
        if(v<=.035)return;
        g.globalAlpha=audible ? .72 : .2;g.strokeStyle=col;g.lineWidth=2;
        g.beginPath();g.rect(xOf(s)-4,yOf(v)-4,8,8);g.stroke();
      });
    }
  });

  const p=playheadDisplayPosition();
  if(p!==null){
    const x=xOf(p%steps);
    g.globalAlpha=1;
    g.strokeStyle=css("--ink");
    g.lineWidth=2;
    g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
    g.fillStyle=css("--ink");
    g.beginPath();g.arc(x,top+7,4.5,0,TAU);g.fill();
  }
  g.globalAlpha=1;
}

function djembeloopsData(){
  return DATA.djembeloops||{patterns:[],matrices:{},samples:[],source:"https://www.djembeloops.com"};
}
function djembeloopsPatternByKey(key){
  return (djembeloopsData().patterns||[]).find(pattern=>pattern.key===key)||null;
}
function djembeloopsMatrix(pattern){
  return pattern ? djembeloopsData().matrices?.[pattern.matrix]||null : null;
}
function candidateDjembeloopsPatterns(piece=state.piece){
  const patterns=djembeloopsData().patterns||[];
  const keys=DJEMBELOOPS_CANDIDATES[piece]||[];
  return keys.map(key=>djembeloopsPatternByKey(key)).filter(Boolean);
}
function selectedDjembeloopsPattern(piece=state.piece){
  const patterns=djembeloopsData().patterns||[];
  if(!patterns.length)return null;
  const candidates=candidateDjembeloopsPatterns(piece);
  const stored=state.djembeloopsPatternByPiece[piece];
  const selected=patterns.find(pattern=>pattern.key===stored)
    || candidates[0]
    || patterns[0];
  state.djembeloopsPatternByPiece[piece]=selected.key;
  return selected;
}
function familyOfInstrument(instrument){
  return String(instrument||"").split("_")[0];
}
function referenceColor(instrument){
  const family=familyOfInstrument(instrument);
  if(family==="Doundoun")return color("D1");
  if(family==="Sangban")return color("D2");
  if(family==="Kenkeni")return css("--live");
  if(family==="DjembeOne"||family==="DjembeTwo"||family==="DjembeThree")return color("J1");
  if(family.startsWith("Djembe"))return color("J2");
  return css("--accent");
}
function resampleReference(values,target){
  if(!values.length)return Array.from({length:target},()=>0);
  return Array.from({length:target},(_,s)=>values[Math.min(values.length-1,Math.floor(s*values.length/target))]||0);
}
function normalizeReferenceWindow(values){
  const target=displayStepCount();
  const source=values.length;
  if(source===target)return values.slice();
  if(target===12&&source>=24&&source%12===0){
    if(state.phraseMode==="cycleA")return values.slice(0,12);
    if(state.phraseMode==="cycleB")return values.slice(12,24);
    const cycles=Math.floor(source/12);
    return Array.from({length:12},(_,s)=>{
      let sum=0,count=0;
      for(let cycle=0;cycle<cycles;cycle++){
        const index=cycle*12+s;
        if(index<source){sum+=values[index];count++;}
      }
      return count ? sum/count : 0;
    });
  }
  if(target===24){
    if(source===12)return values.concat(values);
    if(source>24&&source%24===0){
      const phrases=Math.floor(source/24);
      return Array.from({length:24},(_,s)=>{
        let sum=0,count=0;
        for(let phrase=0;phrase<phrases;phrase++){
          const index=phrase*24+s;
          if(index<source){sum+=values[index];count++;}
        }
        return count ? sum/count : 0;
      });
    }
    if(source>24&&source%12===0)return values.slice(0,24);
  }
  return resampleReference(values,target);
}
function referenceRows(pattern){
  const matrix=djembeloopsMatrix(pattern);
  if(!matrix?.rows)return [];
  return matrix.rows.map(row=>{
    const raw=(row.notes||[]).map(value=>value ? 1 : 0);
    return {
      instrument:row.instrument,
      family:familyOfInstrument(row.instrument),
      values:normalizeReferenceWindow(raw),
      rawSteps:raw.length,
    };
  });
}
function compatibleRowsForVoice(rows,voice){
  const isDunun=DUNUN_VOICES.includes(voice);
  const filtered=rows.filter(row=>{
    if(String(row.instrument).includes("Bell"))return false;
    if(isDunun)return DJEMBELOOPS_DUNUN.includes(row.family);
    return DJEMBELOOPS_JEMBE.includes(row.family);
  });
  return filtered.length ? filtered : rows;
}
function referenceDistance(a,b){
  const n=Math.min(a.length,b.length);
  if(!n)return 1;
  let sum=0;
  for(let i=0;i<n;i++){
    sum+=Math.abs(Math.max(0,Math.min(1,a[i]||0))-Math.max(0,Math.min(1,b[i]||0)));
  }
  return sum/n;
}
function bestReferenceMatches(pattern){
  const rows=referenceRows(pattern);
  return visibleVoices().map(voice=>{
    const motif=displayMotif(voice).map(value=>Math.max(0,Math.min(1,value||0)));
    let best=null,bestDistance=Infinity;
    compatibleRowsForVoice(rows,voice).forEach(row=>{
      const distance=referenceDistance(motif,row.values);
      if(distance<bestDistance){bestDistance=distance;best=row;}
    });
    return {voice,best,distance:Number.isFinite(bestDistance)?bestDistance:null};
  });
}
function renderDjembeLoopsPatternControl(){
  const select=document.getElementById("djembeloopsPattern");
  if(!select)return;
  const data=djembeloopsData();
  const all=(data.patterns||[]).slice().sort((a,b)=>a.name.localeCompare(b.name));
  const candidates=candidateDjembeloopsPatterns();
  const selected=selectedDjembeloopsPattern();
  select.innerHTML="";
  const addOption=(parent,pattern,prefix="")=>{
    const option=document.createElement("option");
    option.value=pattern.key;
    option.textContent=`${prefix}${pattern.name} · ${pattern.matrixSteps||pattern.steps} pas`;
    option.selected=selected?.key===pattern.key;
    parent.appendChild(option);
  };
  if(candidates.length){
    const group=document.createElement("optgroup");
    group.label=`Proches de ${state.piece}`;
    candidates.forEach(pattern=>addOption(group,pattern));
    select.appendChild(group);
  }
  const allGroup=document.createElement("optgroup");
  allGroup.label="Tous les rythmes extraits";
  all.forEach(pattern=>addOption(allGroup,pattern));
  select.appendChild(allGroup);
  select.onchange=()=>{
    stopReferencePlayback();
    state.djembeloopsPatternByPiece[state.piece]=select.value;
    renderDjembeLoopsReference();
  };
}
function sampleStats(){
  const samples=djembeloopsData().samples||[];
  const counts=samples.reduce((acc,sample)=>{
    acc[sample.family]=(acc[sample.family]||0)+1;
    return acc;
  },{});
  const dunun=DJEMBELOOPS_DUNUN.reduce((sum,family)=>sum+(counts[family]||0),0);
  const djembe=Object.entries(counts)
    .filter(([family])=>family.startsWith("Djembe"))
    .reduce((sum,[,count])=>sum+count,0);
  return {total:samples.length,dunun,djembe,counts};
}
function referenceTempo(pattern=selectedDjembeloopsPattern()){
  if(!pattern)return 120;
  const stored=state.referenceTempoByPattern[pattern.key];
  return Number.isFinite(stored) ? stored : (pattern.tempo||120);
}
function renderReferencePlayerControls(){
  const pattern=selectedDjembeloopsPattern();
  const playBtn=document.getElementById("referencePlayBtn");
  const stopBtn=document.getElementById("referenceStopBtn");
  const slider=document.getElementById("referenceTempo");
  const output=document.getElementById("referenceTempoOut");
  const status=document.getElementById("referenceStatus");
  if(!playBtn||!stopBtn||!slider||!output||!status)return;
  const tempo=referenceTempo(pattern);
  slider.value=Math.round(tempo);
  output.textContent=String(Math.round(tempo));
  playBtn.textContent=referencePlayback.active ? "■ Référence en cours" : "▶ Lire référence";
  playBtn.className="smallBtn"+(referencePlayback.active?" on":"");
  status.className="referenceStatus"+(referencePlayback.active?" live":"");
  if(referencePlayback.active&&referencePlayback.loadingAudio){
    status.textContent=`chargement audio · ${pattern?.name||"—"}`;
  }else if(referencePlayback.active){
    const engine=referencePlayback.webAudioReady ? "WebAudio local" : `fallback HTMLAudio${referencePlayback.fallbackReason ? " · "+referencePlayback.fallbackReason : ""}`;
    status.textContent=`lecture externe · ${pattern?.name||"—"} · ${engine}`;
  }else{
    status.textContent=`samples locaux prêts · ${(djembeloopsData().samples||[]).length} sons`;
  }
  slider.oninput=()=>{
    const value=Number(slider.value);
    if(pattern)state.referenceTempoByPattern[pattern.key]=value;
    output.textContent=String(value);
    referencePlayback.tempo=value;
  };
  playBtn.onclick=()=>toggleReferencePlayback();
  stopBtn.onclick=()=>stopReferencePlayback();
}
function referencePlayableRows(pattern=selectedDjembeloopsPattern()){
  const matrix=djembeloopsMatrix(pattern);
  if(!matrix?.rows)return [];
  return matrix.rows.filter(row=>{
    const family=familyOfInstrument(row.instrument);
    if(state.scope==="dunun")return DJEMBELOOPS_DUNUN.includes(family)||String(row.instrument).includes("Bell");
    return DJEMBELOOPS_DUNUN.includes(family)||DJEMBELOOPS_JEMBE.includes(family)||String(row.instrument).includes("Bell");
  });
}
function articulationForNote(instrument,note){
  const text=String(note||"");
  if(String(instrument||"").includes("Bell"))return "Bell_Open";
  if(text.includes("Muffled"))return "Muffled";
  if(text.includes("Mute"))return "Mute";
  if(text.includes("Bass"))return "Bass";
  if(text.includes("Slap"))return "Slap";
  if(text.includes("Open"))return "Open";
  return "Open";
}
function articulationCandidates(family,articulation){
  const values=[articulation];
  if(articulation==="Muffled")values.push("Mute");
  if(articulation==="Mute")values.push("Muffled");
  if(articulation==="Bell_Open")values.push("Open");
  if(!values.includes("Open"))values.push("Open");
  return values;
}
function sampleForArticulation(instrument,articulation,variant=""){
  const family=familyOfInstrument(instrument);
  const samples=djembeloopsData().samples||[];
  const variants=variant ? [variant,""] : [""];
  for(const art of articulationCandidates(family,articulation)){
    for(const candidateVariant of variants){
      const sample=samples.find(item=>item.family===family&&item.articulation===art&&item.variant===candidateVariant);
      if(sample)return sample;
    }
    const sample=samples.find(item=>item.family===family&&item.articulation===art);
    if(sample)return sample;
  }
  return samples.find(sample=>sample.family===family&&sample.articulation==="Open"&&sample.variant==="")
    || samples.find(sample=>sample.family===family&&sample.articulation==="Open")
    || samples.find(sample=>sample.family===family)
    || null;
}
function sampleForNote(instrument,note,rowIndex=99,step=0){
  const family=familyOfInstrument(instrument);
  const variant=family.startsWith("Djembe")&&rowIndex<3&&((step+1)%2===0) ? "L" : "";
  return sampleForArticulation(instrument,articulationForNote(instrument,note),variant);
}
function sampleForPolakVoice(voice){
  const samples=djembeloopsData().samples||[];
  const target={
    J1:["DjembeOne","Open"],
    J2:["DjembeTwo","Slap"],
    D1:["Doundoun","Open"],
    D2:["Sangban","Open"],
  }[voice]||["DjembeOne","Open"];
  return samples.find(sample=>sample.family===target[0]&&sample.articulation===target[1]&&sample.variant==="")
    || samples.find(sample=>sample.family===target[0]&&sample.articulation===target[1])
    || samples.find(sample=>sample.family===target[0])
    || samples[0];
}
function sampleHref(sample){
  if(!sample)return null;
  if(sample.localPath)return new URL(`../${sample.localPath}`,location.href).href;
  return sample.url;
}
function cachedAudioForHref(href){
  if(!href)return null;
  if(!sampleAudioCache.has(href)){
    const audio=new Audio(href);
    audio.preload="auto";
    audio.load();
    sampleAudioCache.set(href,audio);
  }
  return sampleAudioCache.get(href);
}
function preloadDjembeLoopSamples(){
  (djembeloopsData().samples||[]).forEach(sample=>cachedAudioForHref(sampleHref(sample)));
}
function ensureReferenceAudioContext(){
  const AudioCtor=window.AudioContext||window.webkitAudioContext;
  if(!AudioCtor)return null;
  if(!referencePlayback.ctx){
    referencePlayback.ctx=new AudioCtor();
    referencePlayback.master=referencePlayback.ctx.createGain();
    referencePlayback.master.gain.value=.95;
    referencePlayback.master.connect(referencePlayback.ctx.destination);
  }
  return referencePlayback.ctx;
}
function loadReferenceBuffer(sample){
  const href=sampleHref(sample);
  const ctx=ensureReferenceAudioContext();
  if(!href||!ctx)return Promise.resolve(null);
  if(referencePlayback.buffers.has(href))return Promise.resolve(referencePlayback.buffers.get(href));
  if(referencePlayback.loading.has(href))return referencePlayback.loading.get(href);
  const promise=new Promise(resolve=>{
    const request=new XMLHttpRequest();
    request.open("GET",href,true);
    request.responseType="arraybuffer";
    request.timeout=5000;
    request.onload=()=>{
      if(request.status&&request.status>=400){resolve(null);return;}
      const data=request.response;
      const done=buffer=>{referencePlayback.buffers.set(href,buffer);resolve(buffer);};
      const fail=()=>resolve(null);
      try{
        const decodeResult=ctx.decodeAudioData(data,done,fail);
        if(decodeResult&&typeof decodeResult.then==="function")decodeResult.then(done).catch(fail);
      }catch(error){fail();}
    };
    request.onerror=()=>resolve(null);
    request.ontimeout=()=>resolve(null);
    request.send();
  }).finally(()=>referencePlayback.loading.delete(href));
  referencePlayback.loading.set(href,promise);
  return promise;
}
function playSampleHref(href,delayMs,volume=.82,bag=null){
  if(!href)return;
  const template=cachedAudioForHref(href);
  const timeout=setTimeout(()=>{
    if(bag){
      const index=bag.indexOf(timeout);
      if(index>=0)bag.splice(index,1);
    }
    const audio=template?.cloneNode ? template.cloneNode(true) : new Audio(href);
    audio.volume=Math.max(0,Math.min(1,volume));
    audio.play().catch(()=>{});
  },Math.max(0,delayMs));
  if(bag)bag.push(timeout);
  return timeout;
}
function referenceGroupMetronome(pattern=selectedDjembeloopsPattern()){
  const raw=String(pattern?.rawSteps??pattern?.declaredSteps??"");
  if(raw==="4812")return 3;
  const declared=Number(raw)||(djembeloopsMatrix(pattern)?.steps||pattern?.steps||24);
  if([12,24,36,60,72,84].includes(declared))return 3;
  return 4;
}
function referenceStepMs(pattern=selectedDjembeloopsPattern()){
  const tempo=Math.max(40,referencePlayback.tempo||referenceTempo(pattern));
  return 60000/tempo/referenceGroupMetronome(pattern);
}
function referenceSwingAltValues(pattern=selectedDjembeloopsPattern()){
  const tempo=Math.max(40,referencePlayback.tempo||referenceTempo(pattern));
  const reference=Number(pattern?.tempo)||tempo;
  const swing=pattern?.swing||{};
  const factor=Number.isFinite(Number(swing.swingFactor)) ? Number(swing.swingFactor) : 10;
  const raw=["swingA","swingB","swingC","swingD"].map(key=>Number(swing[key])||0);
  return raw.map(value=>{
    if(value===0)return 0;
    if(Number(tempo)<reference+1)return value;
    const sign=value>0 ? -1 : 1;
    const slope=(factor*sign)/250;
    const intercept=value-slope*reference;
    let adjusted=slope*Number(tempo)+intercept;
    if((adjusted<0)===(value>0))adjusted=0;
    return adjusted;
  });
}
function referenceSwingIndex(step,pattern=selectedDjembeloopsPattern()){
  const group=referenceGroupMetronome(pattern);
  const local=((step%group)+group)%group;
  if(group===3)return local===0 ? 0 : (local===2 ? 2 : 1);
  return local===0 ? 0 : (local===3 ? 3 : (local===2 ? 2 : 1));
}
function referenceSwingValuesMs(pattern=selectedDjembeloopsPattern()){
  const stepMs=referenceStepMs(pattern);
  return referenceSwingAltValues(pattern).map(value=>stepMs*(value/100));
}
function referenceRollValuesMs(pattern=selectedDjembeloopsPattern()){
  const stepMs=referenceStepMs(pattern);
  const swing=referenceSwingValuesMs(pattern);
  return [
    (stepMs+swing[1]-swing[0])/2,
    (stepMs+swing[2]-swing[1])/2,
    (stepMs+swing[3]-swing[2])/2,
    (stepMs+swing[0]-swing[3])/2,
  ];
}
function referenceTripletValuesMs(pattern=selectedDjembeloopsPattern()){
  const stepMs=referenceStepMs(pattern);
  const swing=referenceSwingValuesMs(pattern);
  return [
    ((2*stepMs)+swing[2]-swing[0])/3,
    ((2*stepMs)+swing[3]-swing[1])/3,
    ((2*stepMs)+swing[0]-swing[2])/3,
    ((2*stepMs)+swing[1]-swing[3])/3,
  ];
}
function referenceTimingTerms(step,pattern=selectedDjembeloopsPattern()){
  const index=referenceSwingIndex(step,pattern);
  const stepMs=referenceStepMs(pattern);
  const swingMs=referenceSwingValuesMs(pattern);
  const rollMs=referenceRollValuesMs(pattern);
  const tripletMs=referenceTripletValuesMs(pattern);
  return {
    stepMs,
    swingOffsetMs:swingMs[index]||0,
    rollDelayMs:Math.max(18,rollMs[index]||stepMs/2),
    tripletDelayMs:Math.max(18,tripletMs[index]||(2*stepMs/3)),
    flamDelayMs:Math.max(18,stepMs*.25),
  };
}
function referenceSwingOffsetMs(step,pattern=selectedDjembeloopsPattern()){
  return referenceTimingTerms(step,pattern).swingOffsetMs;
}
function referenceStopDurationMs(row,step,pattern=selectedDjembeloopsPattern()){
  const notes=row?.notes||[];
  const steps=notes.length||(djembeloopsMatrix(pattern)?.steps||pattern?.steps||24);
  let noteToStop=1;
  let nextStep=(step+1)%steps;
  for(let t=1;t<7;t++){
    const index=(step+t)%steps;
    nextStep=index;
    if(!notes[index])noteToStop+=1;
    else break;
  }
  const currentSwing=referenceSwingOffsetMs(step,pattern);
  const nextSwing=referenceSwingOffsetMs(nextStep,pattern);
  return Math.max(24,noteToStop*referenceStepMs(pattern)+nextSwing-currentSwing);
}
function noteTokens(note){
  const matches=String(note||"").match(/Bass|Open|Slap|Muffled|Mute/g);
  return matches&&matches.length ? matches : ["Open"];
}
function planHit(instrument,articulation,offsetMs=0,variant="",durationMs=null){
  const sample=sampleForArticulation(instrument,articulation,variant);
  return sample ? {sample,offsetMs,durationMs,articulation,variant} : null;
}
function referenceNotePlan(row,note,rowIndex,step,pattern=selectedDjembeloopsPattern()){
  const instrument=row.instrument;
  const family=familyOfInstrument(instrument);
  const text=String(note||"");
  const terms=referenceTimingTerms(step,pattern);
  const stopMs=referenceStopDurationMs(row,step,pattern);
  const simpleVariant=family.startsWith("Djembe")&&rowIndex<3&&((step+1)%2===0) ? "L" : "";
  if(String(instrument||"").includes("Bell")){
    return [planHit(instrument,"Bell_Open",0,"",stopMs)].filter(Boolean);
  }
  if(text.endsWith("Triplet")){
    const tokens=noteTokens(text.replace("Triplet",""));
    const sequence=tokens.length>=3 ? tokens.slice(0,3) : [tokens[0]||"Open",tokens[1]||tokens[0]||"Open",tokens[2]||tokens[0]||"Open"];
    return [
      planHit(instrument,sequence[0],0,"",terms.tripletDelayMs),
      planHit(instrument,sequence[1],terms.tripletDelayMs,"L",terms.tripletDelayMs),
      planHit(instrument,sequence[2],2*terms.tripletDelayMs,"",Math.max(24,stopMs-2*terms.tripletDelayMs)),
    ].filter(Boolean);
  }
  if(text.endsWith("Roll")){
    const tokens=noteTokens(text.replace("Roll",""));
    const first=tokens[0]||"Open";
    const second=tokens[1]||first;
    return [
      planHit(instrument,first,0,"",terms.rollDelayMs),
      planHit(instrument,second,terms.rollDelayMs,"L",Math.max(24,stopMs-terms.rollDelayMs)),
    ].filter(Boolean);
  }
  if(text.endsWith("Flam")){
    const tokens=noteTokens(text.replace("Flam",""));
    const first=tokens[0]||"Open";
    const second=tokens[1]||first;
    return [
      planHit(instrument,first,0,"",terms.flamDelayMs),
      planHit(instrument,second,terms.flamDelayMs,"L",Math.max(24,stopMs-terms.flamDelayMs)),
    ].filter(Boolean);
  }
  return [planHit(instrument,articulationForNote(instrument,note),0,simpleVariant,stopMs)].filter(Boolean);
}
function referenceSamplesForPattern(pattern=selectedDjembeloopsPattern()){
  const matrix=djembeloopsMatrix(pattern);
  if(!matrix)return [];
  const byHref=new Map();
  const rows=referencePlayableRows(pattern);
  for(let step=0;step<matrix.steps;step++){
    rows.forEach((row,rowIndex)=>{
      const note=row.notes?.[step];
      if(!note)return;
      referenceNotePlan(row,note,rowIndex,step,pattern).forEach(part=>{
        const href=sampleHref(part.sample);
        if(href&&!byHref.has(href))byHref.set(href,part.sample);
      });
    });
  }
  return Array.from(byHref.values());
}
async function warmReferenceBuffers(pattern=selectedDjembeloopsPattern()){
  const ctx=ensureReferenceAudioContext();
  if(!ctx){referencePlayback.webAudioReady=false;referencePlayback.fallbackReason="AudioContext indisponible";return false;}
  try{if(ctx.state==="suspended")await ctx.resume();}catch(error){}
  const samples=referenceSamplesForPattern(pattern);
  const results=await Promise.allSettled(samples.map(sample=>loadReferenceBuffer(sample)));
  const loaded=results.filter(result=>result.status==="fulfilled"&&result.value).length;
  referencePlayback.webAudioReady=loaded>0;
  referencePlayback.fallbackReason=loaded>0 ? "" : "décodage MP3 bloqué en file://";
  return referencePlayback.webAudioReady;
}
function scheduleReferenceEvent(event,targetMs,volume=.82){
  const href=sampleHref(event.sample);
  const ctx=referencePlayback.ctx;
  const buffer=href ? referencePlayback.buffers.get(href) : null;
  if(ctx&&referencePlayback.master&&buffer){
    const when=Math.max(ctx.currentTime+.004,ctx.currentTime+((targetMs-performance.now())/1000));
    const source=ctx.createBufferSource();
    const gain=ctx.createGain();
    const safeVolume=Math.max(0,Math.min(1.35,volume));
    source.buffer=buffer;
    gain.gain.setValueAtTime(safeVolume,when);
    source.connect(gain);
    gain.connect(referencePlayback.master);
    try{
      source.start(when);
      if(Number.isFinite(event.durationMs)&&event.durationMs>20)source.stop(when+event.durationMs/1000);
      referencePlayback.sources.push(source);
      source.onended=()=>{
        const index=referencePlayback.sources.indexOf(source);
        if(index>=0)referencePlayback.sources.splice(index,1);
        try{source.disconnect();gain.disconnect();}catch(error){}
      };
    }catch(error){
      try{source.disconnect();gain.disconnect();}catch(disconnectError){}
      playSampleHref(href,targetMs-performance.now(),Math.min(1,safeVolume),referencePlayback.pending);
    }
    return;
  }
  playSampleHref(href,targetMs-performance.now(),Math.min(1,volume),referencePlayback.pending);
}
function referenceEventsForStep(step,pattern=selectedDjembeloopsPattern()){
  const rows=referencePlayableRows(pattern);
  const events=[];
  rows.forEach((row,rowIndex)=>{
    const note=row.notes?.[step];
    if(!note)return;
    referenceNotePlan(row,note,rowIndex,step,pattern).forEach((part,partIndex)=>{
      events.push({instrument:row.instrument,note,rowIndex,partIndex,...part});
    });
  });
  return events;
}
function referenceVolumeForInstrument(instrument){
  const family=familyOfInstrument(instrument);
  const master=Math.max(.1,Math.min(1.6,Number(state.volume)||1));
  if(String(instrument||"").includes("Bell"))return .15*master;
  if(family==="DjembeSolo")return 1*master;
  if(family.startsWith("Djembe"))return .5*master;
  if(DJEMBELOOPS_DUNUN.includes(family))return .5*master;
  return .5*master;
}
function referenceScheduler(){
  if(!referencePlayback.active)return;
  const pattern=selectedDjembeloopsPattern();
  const matrix=djembeloopsMatrix(pattern);
  if(!pattern||!matrix){stopReferencePlayback();return;}
  const now=performance.now();
  const stepMs=referenceStepMs(pattern);
  while(referencePlayback.nextStepAt<now+140){
    const step=referencePlayback.stepIndex%matrix.steps;
    const swingOffset=referenceSwingOffsetMs(step,pattern);
    referenceEventsForStep(step,pattern).forEach(event=>{
      scheduleReferenceEvent(event,referencePlayback.nextStepAt+swingOffset+(event.offsetMs||0),referenceVolumeForInstrument(event.instrument));
    });
    referencePlayback.stepIndex=(referencePlayback.stepIndex+1)%matrix.steps;
    referencePlayback.nextStepAt+=stepMs;
  }
}
async function startReferencePlayback(){
  const pattern=selectedDjembeloopsPattern();
  const matrix=djembeloopsMatrix(pattern);
  if(!pattern||!matrix){
    alert("Aucune référence externe jouable.");
    return;
  }
  stopReferencePlayback();
  referencePlayback.active=true;
  referencePlayback.loadingAudio=true;
  referencePlayback.patternKey=pattern.key;
  referencePlayback.tempo=referenceTempo(pattern);
  referencePlayback.stepIndex=0;
  referencePlayback.webAudioReady=false;
  renderReferencePlayerControls();
  await warmReferenceBuffers(pattern);
  if(!referencePlayback.active||referencePlayback.patternKey!==pattern.key)return;
  referencePlayback.loadingAudio=false;
  referencePlayback.nextStepAt=performance.now()+90;
  referenceScheduler();
  referencePlayback.timer=setInterval(referenceScheduler,35);
  renderReferencePlayerControls();
  syncPlaybackUI();
}
function stopReferencePlayback(){
  if(referencePlayback.timer)clearInterval(referencePlayback.timer);
  referencePlayback.pending.forEach(timeout=>clearTimeout(timeout));
  referencePlayback.sources.forEach(source=>{try{source.stop(0);}catch(error){}});
  referencePlayback.active=false;
  referencePlayback.loadingAudio=false;
  referencePlayback.timer=null;
  referencePlayback.pending=[];
  referencePlayback.sources=[];
  referencePlayback.stepIndex=0;
  renderReferencePlayerControls();
  syncPlaybackUI();
}
function toggleReferencePlayback(){
  if(referencePlayback.active)stopReferencePlayback();
  else startReferencePlayback();
}
function drawDjembeLoopsComparison(pattern,matches){
  const cv=document.getElementById("djembeloopsCompare");
  if(!cv)return;
  const g=cv.getContext("2d"),W=cv.width,H=cv.height;
  const matrix=djembeloopsMatrix(pattern);
  const left=150,right=18,top=24,bottom=38,steps=matrix?.steps||12,span=W-left-right;
  g.clearRect(0,0,W,H);
  if(!pattern||!matrix){
    g.fillStyle=css("--muted");g.font="13px Inter, system-ui";g.textAlign="center";
    g.fillText("Référence externe indisponible. Lance import_djembeloops_reference.py pour régénérer le JSON.",W/2,H/2);
    return;
  }
  const rows=matrix.rows.map(row=>({
    label:row.instrument,
    notes:row.notes||[],
    color:referenceColor(row.instrument),
  }));
  const rowGap=Math.min(30,Math.max(18,(H-top-bottom)/(rows.length||1)));
  const yOf=index=>top+index*rowGap;
  const xOf=s=>left+(steps===1?0:s/(steps-1))*span;
  g.strokeStyle=css("--line2");g.lineWidth=1;g.globalAlpha=.55;
  for(let s=0;s<steps;s++){
    const x=xOf(s);
    g.beginPath();g.moveTo(x,top-8);g.lineTo(x,Math.min(H-bottom,yOf(rows.length-1)+10));g.stroke();
  }
  g.globalAlpha=1;
  g.fillStyle=css("--muted");g.font="10px Inter, system-ui";g.textAlign="center";g.textBaseline="top";
  for(let s=0;s<steps;s++)g.fillText(String(s+1),xOf(s),H-bottom+10);
  rows.forEach((row,index)=>{
    const y=yOf(index);
    g.fillStyle=css("--muted");
    g.font="12px Inter, system-ui";
    g.textAlign="right";g.textBaseline="middle";
    g.fillText(row.label,left-12,y);
    row.notes.forEach((note,s)=>{
      if(!note)return;
      const x=xOf(s), size=5.8;
      g.fillStyle=row.color;g.globalAlpha=.78;
      g.beginPath();g.arc(x,y,size,0,TAU);g.fill();
      g.strokeStyle=css("--paper");g.lineWidth=1.1;g.stroke();
      g.fillStyle=css("--paper");g.globalAlpha=.95;g.font="700 7px Inter, system-ui";g.textAlign="center";g.textBaseline="middle";
      g.fillText(String(note)[0],x,y+.2);
    });
  });
  g.globalAlpha=1;
  if(referencePlayback.active&&referencePlayback.patternKey===pattern.key){
    const x=xOf((referencePlayback.stepIndex%steps+steps)%steps);
    g.strokeStyle=css("--ink");g.lineWidth=2;g.globalAlpha=.72;
    g.beginPath();g.moveTo(x,top-12);g.lineTo(x,H-bottom);g.stroke();
    g.globalAlpha=1;
  }
}
function renderDjembeLoopsReference(){
  renderDjembeLoopsPatternControl();
  const pattern=selectedDjembeloopsPattern();
  const matrix=djembeloopsMatrix(pattern);
  renderReferencePlayerControls();
  const modeTag=document.getElementById("djembeloopsMode");
  if(modeTag)modeTag.textContent="rythme similaire";
  const statsBox=document.getElementById("djembeloopsStats");
  if(statsBox)statsBox.innerHTML="";
  drawDjembeLoopsComparison(pattern,[]);
  const caption=document.getElementById("djembeloopsCaption");
  if(caption){
    if(!pattern){
      caption.textContent="Aucune donnée DjembeLoops chargée.";
      return;
    }
    const swingValues=pattern.swing
      ? ["swingA","swingB","swingC","swingD"].map(key=>pattern.swing[key]).filter(value=>Number.isFinite(value))
      : [];
    const swingText=swingValues.length ? ` · swing ${swingValues.join("/")}` : "";
    caption.innerHTML=
      `<b>${pattern.name}</b> · ${matrix?.steps||pattern.steps} pas · tempo ${referenceTempo(pattern)} bpm${swingText}. `+
      `Choisis un autre rythme dans le menu pour changer la table et le player.`;
  }
}

function microProfile(voice,piece=state.piece){
  return DATA.microtiming?.[piece]?.[voice]||null;
}
function microPhraseProfile(voice,piece=state.piece){
  return DATA.microtimingPhrase?.[piece]?.[voice]||null;
}
function slicedProfile(profile,start,count){
  if(!profile)return null;
  return {
    ...profile,
    mean:(profile.mean||[]).slice(start,start+count),
    std:(profile.std||[]).slice(start,start+count),
    n:(profile.n||[]).slice(start,start+count),
  };
}
function displayMicroProfile(voice,piece=state.piece){
  if(state.phraseMode==="mean12")return microProfile(voice,piece);
  const phrase=microPhraseProfile(voice,piece);
  if(!phrase)return microProfile(voice,piece);
  if(state.phraseMode==="cycleA")return slicedProfile(phrase,0,12);
  if(state.phraseMode==="cycleB")return slicedProfile(phrase,12,12);
  return phrase;
}
function displayTimingOffsetMs(voice,stepIndex){
  if(state.timingMode!=="observed")return 0;
  const value=displayMicroProfile(voice)?.mean?.[stepIndex];
  if(!Number.isFinite(value))return 0;
  return value*state.timingStrength;
}
function timingOffsetMs(voice,stepIndex){
  if(state.timingMode!=="observed")return 0;
  const value=microProfile(voice)?.mean?.[stepIndex];
  if(!Number.isFinite(value))return 0;
  const strength=state.listeningMode==="recording"?1:state.timingStrength;
  return Math.max(-100,Math.min(100,value*strength));
}
function voiceResidualOffsetMs(voice,stepIndex){
  const value=microProfile(voice)?.mean?.[stepIndex];
  if(state.timingMode!=="observed"||!Number.isFinite(value))return 0;
  return timingOffsetMs(voice,stepIndex)-metricOffsetMs(stepIndex);
}
function timingModeLabel(){
  if(state.listeningMode==="recording"){
    return `enregistrement · prise ${selectedRecording()?.take??"—"} · ${selectedCycleLabel()}`;
  }
  if(state.listeningMode==="synthesis"){
    return `partition synthèse · ${synthesisCycleLabel()} · ${partitionHitCount()} frappes`;
  }
  if(state.listeningMode==="model"){
    return `référence externe · ${selectedDjembeloopsPattern()?.name||"—"}`;
  }
  if(state.listeningMode==="reference"){
    return `référence externe · ${selectedDjembeloopsPattern()?.name||"—"}`;
  }
  return "modèle 12";
}
async function setListeningMode(mode){
  if(mode==="model")mode="reference";
  const wasPlaying=playback.active;
  if(wasPlaying)stopPlayback();
  if(referencePlayback.active)stopReferencePlayback();
  state.listeningMode=mode;
  if(mode==="synthesis"){
    state.synthEnabled=true;
    state.timingMode="observed";
  }else{
    state.timingMode=mode==="recording"?"observed":"grid";
  }
  renderSynthControls();
  renderTimingControls();
  drawCircle();
  drawCurve();
  drawMicroTiming();
  syncPlaybackUI();
  if(wasPlaying)await startPlayback();
}
function renderTimingControls(){
  const recording=document.getElementById("recordingTimingBtn");
  const synthesis=document.getElementById("synthesisTimingBtn");
  const reference=document.getElementById("referenceTimingBtn");
  const circleRecording=document.getElementById("circleRecordingBtn");
  const circleSynthesis=document.getElementById("circleSynthesisBtn");
  const circleReference=document.getElementById("circleReferenceBtn");
  const slider=document.getElementById("timingStrength");
  const output=document.getElementById("timingStrengthOut");
  [
    ["recording",[recording,circleRecording]],
    ["synthesis",[synthesis,circleSynthesis]],
    ["reference",[reference,circleReference]]
  ].forEach(([mode,buttons])=>{
    buttons.forEach(button=>{
      if(!button)return;
      button.className="smallBtn"+(state.listeningMode===mode?" on":"");
      button.onclick=()=>setListeningMode(mode);
    });
  });
  slider.value=state.timingStrength;
  slider.disabled=state.listeningMode!=="synthesis";
  output.textContent=state.listeningMode==="recording"?"×1 réel":state.listeningMode==="synthesis"?`×${state.timingStrength.toFixed(2)}`:"grille";
  slider.oninput=()=>{
    state.timingStrength=Number(slider.value);
    output.textContent=`×${state.timingStrength.toFixed(2)}`;
    drawCircle();
    drawCurve();
    drawMicroTiming();
    syncPlaybackUI();
  };
  renderRecordingTakeControl();
  renderRecordingCycleControl();
  renderReconstructionMixControl();
  const recordingInfo=selectedRecording();
  const cycleInfo=selectedCycle(recordingInfo);
  const note=document.getElementById("listeningNote");
  if(state.listeningMode==="recording"){
    note.innerHTML=
      `<b>Enregistrement réel.</b> Prise ${recordingInfo?.take??"—"}, ${selectedCycleLabel(recordingInfo)} : vrais timbres, `+
      `non-isochronie et asynchronie de l’ensemble. La boucle suit les bornes annotées du cycle sélectionné.`;
  }else if(state.listeningMode==="synthesis"){
    note.innerHTML=
      `<b>Partition synthèse.</b> La boucle joue la partition de travail D1/D2 visible dans le synthétiseur ; `+
      `les coefficients c_k sont calculés depuis cette partition et servent à proposer des transformations. `+
      `La boucle est calée sur ${synthesisCycleLabel()} afin d’éviter les pauses trop longues des cycles atypiques. `+
      `Le curseur d’impact règle la quantité de micro-timing observé appliquée à la boucle synthétique.`;
  }else if(state.listeningMode==="reference"){
    note.innerHTML=
      `<b>Référence externe.</b> Le player lit le rythme DjembeLoops jugé proche, dans la table de référence externe.`;
  }else{
    note.innerHTML=
      `<b>Référence externe.</b> Le player lit le rythme DjembeLoops sélectionné.`;
  }
}
function drawMicroTiming(){
  const cv=document.getElementById("microTiming"),g=cv.getContext("2d");
  const W=cv.width,H=cv.height,left=54,right=18,top=30,bottom=42,span=W-left-right;
  const profiles=visibleVoices().map(voice=>[voice,displayMicroProfile(voice)]).filter(([,p])=>p);
  const steps=displayStepCount();
  g.clearRect(0,0,W,H);
  if(!profiles.length){
    g.fillStyle=css("--muted");g.font="13px Inter, system-ui";g.textAlign="center";
    g.fillText("Données de micro-timing indisponibles pour ce rythme.",W/2,H/2);
    return;
  }
  const observedValues=profiles.flatMap(([,p])=>p.mean.flatMap((m,i)=>{
    if(!Number.isFinite(m)||p.n[i]<3)return [];
    const sd=Number.isFinite(p.std[i])?p.std[i]:0;
    return [m-sd,m+sd];
  }));
  const appliedValues=profiles.flatMap(([voice,p])=>p.mean.flatMap((m,s)=>
    Number.isFinite(m)&&p.n[s]>=3 ? [displayTimingOffsetMs(voice,s)] : []
  ));
  const maxAbs=Math.max(20,...observedValues.concat(appliedValues).map(v=>Math.abs(v)));
  const limit=Math.min(100,Math.ceil(maxAbs/10)*10);
  const xOf=s=>left+(steps<=1?0:s/(steps-1))*span;
  const yOf=v=>top+(limit-v)/(2*limit)*(H-top-bottom);
  const p=playheadDisplayPosition(),activeStep=p===null?null:Math.floor(p)%steps;

  if(activeStep!==null){
    const x=xOf(activeStep),cell=span/Math.max(1,steps-1);
    g.fillStyle="rgba(127,127,127,.07)";
    g.fillRect(x-cell*.42,top,cell*.84,H-top-bottom);
  }
  g.strokeStyle=css("--line2");g.lineWidth=1;
  for(let s=0;s<steps;s++){
    const x=xOf(s);
    g.beginPath();g.moveTo(x,top);g.lineTo(x,H-bottom);g.stroke();
  }
  g.strokeStyle=css("--line");g.lineWidth=1.2;
  g.beginPath();g.moveTo(left,yOf(0));g.lineTo(W-right,yOf(0));g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="center";g.textBaseline="top";
  for(let s=0;s<steps;s++)g.fillText(String(s+1),xOf(s),H-bottom+10);
  g.textAlign="right";g.textBaseline="middle";
  [-limit,0,limit].forEach(v=>g.fillText(`${v>0?"+":""}${v} ms`,left-8,yOf(v)));
  g.save();g.translate(13,H/2);g.rotate(-Math.PI/2);g.textAlign="center";
  g.fillText("avance ← écart à la grille → retard",0,0);g.restore();

  profiles.forEach(([voice,profile],voiceIndex)=>{
    const col=color(voice),audible=selectedVoices().includes(voice);
    const points=[];
    profile.mean.forEach((mean,s)=>{
      if(!Number.isFinite(mean)||profile.n[s]<3)return;
      const sd=Number.isFinite(profile.std[s])?profile.std[s]:0;
      const x=xOf(s)+(voiceIndex-(profiles.length-1)/2)*3;
      const observedY=yOf(mean),applied=displayTimingOffsetMs(voice,s);
      const appliedY=yOf(applied);
      g.globalAlpha=audible ? .34 : .12;g.strokeStyle=col;g.lineWidth=1.3;
      g.beginPath();g.moveTo(x,yOf(Math.min(limit,mean+sd)));g.lineTo(x,yOf(Math.max(-limit,mean-sd)));g.stroke();
      g.beginPath();g.moveTo(x-4,yOf(Math.min(limit,mean+sd)));g.lineTo(x+4,yOf(Math.min(limit,mean+sd)));g.stroke();
      g.beginPath();g.moveTo(x-4,yOf(Math.max(-limit,mean-sd)));g.lineTo(x+4,yOf(Math.max(-limit,mean-sd)));g.stroke();
      g.globalAlpha=audible ? .55 : .18;g.fillStyle=css("--paper");g.strokeStyle=col;g.lineWidth=1.5;
      g.beginPath();g.arc(x,observedY,4,0,TAU);g.fill();g.stroke();
      points.push([x,appliedY]);
    });
    g.globalAlpha=audible ? .82 : .2;g.strokeStyle=col;g.lineWidth=2;
    g.beginPath();points.forEach(([x,y],i)=>{if(i===0)g.moveTo(x,y);else g.lineTo(x,y);});g.stroke();
    points.forEach(([x,y])=>{g.fillStyle=col;g.beginPath();g.arc(x,y,4.5,0,TAU);g.fill();});
  });
  g.globalAlpha=1;
  const precisionText=profiles
    .map(([voice,profile])=>`${voice} ${Number.isFinite(profile?.precision)?profile.precision.toFixed(1):"—"} ms`)
    .join(", ");
  const heardLabel=state.listeningMode==="recording"
    ? `L’enregistrement réel varie à l’intérieur de cette distribution ; le point plein reste sa moyenne de référence.`
    : state.listeningMode==="reconstruction"
      ? `La reconstruction suit le cycle annoté sélectionné ; le graphe reste la distribution moyenne du corpus.`
      : `Point plein = placement entendu (${timingModeLabel()}).`;
  document.getElementById("microCaption").innerHTML=
    `Fenêtre active : <b>${phraseModeLabel()}</b>. Ce graphe sépare les écarts propres aux voix visibles. `+
    `Cercle vide + barre = moyenne et dispersion observées. ${heardLabel} `+
    `Précision résiduelle : ${precisionText}.`;
}

function drawMap(){
  const cv=document.getElementById("map"),g=cv.getContext("2d"),W=cv.width,H=cv.height,pad=36;
  g.clearRect(0,0,W,H);
  g.strokeStyle=css("--line");g.lineWidth=1;
  g.beginPath();g.moveTo(pad,H/2);g.lineTo(W-pad,H/2);g.moveTo(W/2,pad);g.lineTo(W/2,H-pad);g.stroke();
  g.fillStyle=css("--muted");g.font="11px Inter, system-ui";g.textAlign="right";g.textBaseline="top";
  g.fillText("axe x : PC1",W-pad,H/2+6);
  g.save();g.translate(W/2+8,pad);g.rotate(-Math.PI/2);g.textAlign="right";g.fillText("axe y : PC2",0,0);g.restore();
  DATA.map.forEach(pt=>{
    const selected=pt.piece===state.piece;
    const x=W/2+pt.x*(W/2-pad*1.6), y=H/2-pt.y*(H/2-pad*1.4);
    const col=DATA.colors[pt.piece]||css("--accent");
    g.globalAlpha=selected?1:.72;
    g.fillStyle=col;g.strokeStyle=css("--paper");g.lineWidth=selected?3:2;
    g.beginPath();g.arc(x,y,selected?13:9,0,TAU);g.fill();g.stroke();
    g.globalAlpha=selected?1:.78;
    g.fillStyle=css("--ink");
    g.font=selected?"700 12px Inter, system-ui":"12px Inter, system-ui";
    g.textAlign="center";g.textBaseline="bottom";
    g.fillText(pt.piece,x,y-15);
  });
  cv.onclick=(ev)=>{
    const rect=cv.getBoundingClientRect();
    const mx=(ev.clientX-rect.left)*cv.width/rect.width, my=(ev.clientY-rect.top)*cv.height/rect.height;
    let best=null, bd=1e9;
    DATA.map.forEach(pt=>{
      const x=W/2+pt.x*(W/2-pad*1.6), y=H/2-pt.y*(H/2-pad*1.4);
      const d=(x-mx)**2+(y-my)**2;
      if(d<bd){bd=d;best=pt;}
    });
    if(best&&bd<1100)selectPiece(best.piece);
  };
}

function renderEquation(){
  const r=rhythm();
  document.getElementById("mainEquation").textContent=latexForRhythm(r);
  const densityText=visibleVoices()
    .map(voice=>`${voice} ${displayVoiceView(voice).density.toFixed(2)}`)
    .join(" · ");
  document.getElementById("metrics").innerHTML=[
    ["prises",r.takes.join(", ")],
    ["vue",state.scope==="ensemble"?"ensemble complet":"dunun"],
    ["fenêtre",`${phraseModeLabel()}${state.synthEnabled&&state.phraseMode==="mean12"?" · partition synthèse":""}`],
    ["densités",densityText]
  ].map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join("");
}

function renderCoefTable(){
  const rows=[];
  const voices=visibleVoices();
  const maxK=Math.floor(displayStepCount()/2);
  for(let k=1;k<=maxK;k++){
    const name=coefficientName(k);
    rows.push(`<tr><td>c${k}<br><small>${name}</small></td>${
      voices.map(voice=>{
        const c=displayVoiceView(voice).coeffs[k];
        return `<td>${c.mag.toFixed(3)}<br><small>${phaseDeg(c.phase).toFixed(0)}°</small></td>`;
      }).join("")
    }</tr>`);
  }
  document.getElementById("coefTable").innerHTML=`<table class="coefTable">
    <thead><tr><th>onde</th>${voices.map(voice=>`<th>${voice}<br><small>|c| / arg</small></th>`).join("")}</tr></thead>
    <tbody>${rows.join("")}</tbody></table>`;
}

function hit(ctx,time,voice,amp){
  const sample=sampleForPolakVoice(voice);
  const href=sampleHref(sample);
  if(href){
    const sampleVolume=Math.min(1,Math.max(.08,amp)*(DUNUN_VOICES.includes(voice) ? .92 : .68)*(Number(state.volume)||1));
    playSampleHref(href,(time-ctx.currentTime)*1000,sampleVolume);
    return;
  }
  const osc=ctx.createOscillator(), gain=ctx.createGain(), filter=ctx.createBiquadFilter();
  const params={
    J1:{type:"square",f0:310,f1:180,filter:1450,decay:.105,gain:.18},
    J2:{type:"triangle",f0:255,f1:145,filter:1200,decay:.12,gain:.18},
    D1:{type:"triangle",f0:118,f1:74,filter:520,decay:.18,gain:.27},
    D2:{type:"sine",f0:235,f1:155,filter:900,decay:.18,gain:.25},
  }[voice]||{type:"sine",f0:180,f1:110,filter:900,decay:.16,gain:.22};
  osc.type=params.type;
  osc.frequency.setValueAtTime(params.f0,time);
  osc.frequency.exponentialRampToValueAtTime(params.f1,time+Math.min(.14,params.decay));
  filter.type="lowpass";filter.frequency.setValueAtTime(params.filter,time);
  gain.gain.setValueAtTime(0.0001,time);
  gain.gain.exponentialRampToValueAtTime(Math.max(0.018,amp)*params.gain,time+.006);
  gain.gain.exponentialRampToValueAtTime(0.0001,time+params.decay);
  const destination=playback.reconstructionGain||playback.master||ctx.destination;
  osc.connect(filter);filter.connect(gain);gain.connect(destination);
  osc.start(time);osc.stop(time+params.decay+.03);
}

function scheduleSubdivision(stepIndex,time){
  selectedVoices().forEach(voice=>{
    const value=motifForPlayback(voice)[stepIndex]||0;
    if(value<.055)return;
    const amp=Math.min(1,Math.sqrt(value));
    hit(playback.ctx,time+voiceResidualOffsetMs(voice,stepIndex)/1000,voice,amp);
  });
}

function audioScheduler(){
  if(!playback.active||playback.kind!=="synthetic"||!playback.ctx)return;
  while(playback.nextStepTime<playback.ctx.currentTime+playback.scheduleAhead){
    const step=playback.stepIndex;
    const duration=metricDurations()[step]*12*playback.step;
    const time=playback.nextStepTime;
    scheduleSubdivision(step,time);
    playback.timeline.push({step,time,end:time+duration});
    playback.nextStepTime+=duration;
    playback.stepIndex=(playback.stepIndex+1)%12;
  }
  const cutoff=playback.ctx.currentTime-.5;
  playback.timeline=playback.timeline.filter(event=>event.end>=cutoff).slice(-48);
}

function syncCorpusVoiceGains(){
  if(!["recording","reconstruction"].includes(playback.kind)||!playback.ctx)return;
  const audible=new Set(selectedVoices());
  const now=playback.ctx.currentTime;
  const realLevel=playback.kind==="reconstruction" ? 1-state.reconstructionMix : 1;
  ALL_VOICES.forEach(voice=>{
    const gain=playback.voiceGains[voice];
    if(gain)gain.gain.setTargetAtTime(audible.has(voice)?realLevel:0,now,.015);
  });
  if(playback.reconstructionGain){
    playback.reconstructionGain.gain.setTargetAtTime(state.reconstructionMix,now,.02);
  }
}
function recordingBounds(recording=selectedRecording()){
  const cycle=selectedCycle(recording);
  if(cycle)return {start:cycle.start,end:cycle.end,duration:cycle.duration};
  const starts=recording?.cycleStarts||[];
  if(starts.length<2)return {start:0,end:Infinity};
  const durations=starts.slice(1).map((value,index)=>value-starts[index]);
  const sorted=durations.slice().sort((a,b)=>a-b);
  const median=sorted[Math.floor(sorted.length/2)]||durations[0];
  return {start:starts[0],end:starts[starts.length-1]+median};
}
function syncCorpusPlayback(){
  if(!playback.active||!["recording","reconstruction"].includes(playback.kind))return;
  const primary=primaryMedia();
  const media=Object.values(playback.media);
  if(!primary||!media.length)return;
  const bounds=recordingBounds();
  if(media.some(audio=>audio.currentTime>=bounds.end)){
    media.forEach(audio=>{audio.currentTime=bounds.start;});
    playback.nextEventIndex=0;
    playback.timeline=[];
  }else{
    media.forEach(audio=>{
      if(audio!==primary&&Math.abs(audio.currentTime-primary.currentTime)>.025){
        audio.currentTime=primary.currentTime;
      }
    });
  }
}
function reconstructionScheduler(){
  if(!playback.active||playback.kind!=="reconstruction"||!playback.ctx)return;
  syncCorpusPlayback();
  const audio=primaryMedia();
  const cycle=playback.cycle||selectedCycle();
  if(!audio||!cycle)return;
  const events=cycle.events||[];
  const phase=Math.max(0,audio.currentTime-cycle.start);
  while(playback.nextEventIndex<events.length&&events[playback.nextEventIndex].time<phase-.045){
    playback.nextEventIndex++;
  }
  while(playback.nextEventIndex<events.length&&events[playback.nextEventIndex].time<=phase+playback.scheduleAhead){
    const event=events[playback.nextEventIndex];
    if(selectedVoices().includes(event.voice)&&event.time>=phase-.045){
      const time=playback.ctx.currentTime+Math.max(.004,event.time-phase);
      hit(playback.ctx,time,event.voice,.9);
      playback.timeline.push({step:event.sd,time,end:time+.12});
    }
    playback.nextEventIndex++;
  }
  const cutoff=playback.ctx.currentTime-.5;
  playback.timeline=playback.timeline.filter(event=>event.end>=cutoff).slice(-64);
}

function syncPlaybackUI(){
  const btn=document.getElementById("playBtn");
  const status=document.getElementById("liveStatus");
  const active=playback.active||referencePlayback.active;
  if(btn){
    btn.textContent=active
      ?"■ Arrêter l’écoute"
      :state.listeningMode==="recording"
        ?"▶ Lire l’enregistrement"
        :state.listeningMode==="synthesis"
          ?"▶ Lire la partition synthèse"
        :["reference","model"].includes(state.listeningMode)
          ?"▶ Lire la référence externe"
        :"▶ Lire la partition synthèse";
    btn.className="play"+(active?" playing":"");
  }
  if(status){
    status.className="liveStatus"+(active?" on":"");
    status.querySelector("span").textContent=active?"LIVE":"Prêt";
  }
  const caption=document.getElementById("playCaption");
  if(caption){
    if(state.listeningMode==="recording"){
      caption.textContent=playback.active
        ? `Lecture réelle · ${playVoiceLabel()} · prise ${selectedRecording()?.take??"—"} · ${selectedCycleLabel()} · pistes synchronisées.`
        : `Référence réelle · ${playVoiceLabel()} · prise ${selectedRecording()?.take??"—"} · ${selectedCycleLabel()}.`;
    }else if(state.listeningMode==="synthesis"){
      caption.textContent=playback.active
        ? `Partition synthèse en lecture · ${playVoiceLabel()} · ${synthesisCycleLabel()} · samples DjembeLoops.`
        : `Partition synthèse prête · ${playVoiceLabel()} · ${synthesisCycleLabel()} · cases D1/D2 modifiables.`;
    }else if(["reference","model"].includes(state.listeningMode)){
      caption.textContent=referencePlayback.active
        ? `Référence externe en lecture · ${selectedDjembeloopsPattern()?.name||"—"}.`
        : `Référence externe similaire · ${selectedDjembeloopsPattern()?.name||"—"}.`;
    }else{
      caption.textContent=`Référence externe similaire · ${selectedDjembeloopsPattern()?.name||"—"}.`;
    }
  }
  syncSynthModeButtons();
}

function stopPlayback(){
  const ctx=playback.ctx;
  playback.active=false;
  if(playback.timer)clearInterval(playback.timer);
  if(playback.raf)cancelAnimationFrame(playback.raf);
  playback.timer=null;
  playback.raf=null;
  Object.values(playback.media).forEach(audio=>{
    try{audio.pause();}catch(_){}
  });
  playback.kind=null;
  playback.ctx=null;
  playback.master=null;
  playback.compressor=null;
  playback.reconstructionGain=null;
  playback.cycle=null;
  playback.nextEventIndex=0;
  playback.timeline=[];
  playback.media={};
  playback.voiceGains={};
  if(ctx)ctx.close().catch(()=>{});
  syncPlaybackUI();
  drawCircle();
  drawCurve();
  drawMicroTiming();
}

function configureMaster(ctx){
  playback.master=ctx.createGain();
  playback.compressor=ctx.createDynamicsCompressor();
  playback.compressor.threshold.setValueAtTime(-8,ctx.currentTime);
  playback.compressor.knee.setValueAtTime(8,ctx.currentTime);
  playback.compressor.ratio.setValueAtTime(6,ctx.currentTime);
  playback.compressor.attack.setValueAtTime(.003,ctx.currentTime);
  playback.compressor.release.setValueAtTime(.18,ctx.currentTime);
  playback.master.gain.setValueAtTime(state.volume,ctx.currentTime);
  playback.master.connect(playback.compressor);
  playback.compressor.connect(ctx.destination);
}
async function startCorpusPlayback(AudioCtx){
  const recording=selectedRecording();
  if(!recording){
    alert("Aucune prise réelle n’est disponible pour ce rythme.");
    return;
  }
  const ctx=new AudioCtx();
  playback.ctx=ctx;
  await ctx.resume();
  configureMaster(ctx);
  playback.kind="recording";
  playback.media={};
  playback.voiceGains={};
  playback.cycle=selectedCycle(recording);
  const bounds=recordingBounds(recording);
  for(const voice of ALL_VOICES){
    if(!recording[voice])continue;
    const audio=new Audio(new URL(recording[voice],location.href).href);
    audio.preload="auto";
    audio.currentTime=bounds.start;
    const source=ctx.createMediaElementSource(audio);
    const gain=ctx.createGain();
    source.connect(gain);gain.connect(playback.master);
    playback.media[voice]=audio;
    playback.voiceGains[voice]=gain;
  }
  if(!Object.keys(playback.media).length){
    stopPlayback();
    alert("Aucune piste audio exploitable pour cette prise.");
    return;
  }
  syncCorpusVoiceGains();
  playback.active=true;
  try{
    await Promise.all(Object.values(playback.media).map(audio=>audio.play()));
  }catch(error){
    stopPlayback();
    alert(`Lecture de l’enregistrement impossible : ${error.message}`);
    return;
  }
  playback.timer=setInterval(syncCorpusPlayback,120);
  syncPlaybackUI();
  animatePlayback();
}
async function startReconstructionPlayback(AudioCtx){
  const recording=selectedRecording();
  const cycle=selectedCycle(recording);
  if(!recording||!cycle){
    alert("Aucun cycle annoté n’est disponible pour reconstruire cette prise.");
    return;
  }
  const ctx=new AudioCtx();
  playback.ctx=ctx;
  await ctx.resume();
  configureMaster(ctx);
  playback.kind="reconstruction";
  playback.media={};
  playback.voiceGains={};
  playback.reconstructionGain=ctx.createGain();
  playback.reconstructionGain.gain.setValueAtTime(state.reconstructionMix,ctx.currentTime);
  playback.reconstructionGain.connect(playback.master);
  playback.cycle=cycle;
  playback.nextEventIndex=0;
  playback.timeline=[];
  for(const voice of ALL_VOICES){
    if(!recording[voice])continue;
    const audio=new Audio(new URL(recording[voice],location.href).href);
    audio.preload="auto";
    audio.currentTime=cycle.start;
    const source=ctx.createMediaElementSource(audio);
    const gain=ctx.createGain();
    source.connect(gain);gain.connect(playback.master);
    playback.media[voice]=audio;
    playback.voiceGains[voice]=gain;
  }
  if(!Object.keys(playback.media).length){
    stopPlayback();
    alert("Aucune piste audio exploitable pour cette prise.");
    return;
  }
  syncCorpusVoiceGains();
  playback.active=true;
  try{
    await Promise.all(Object.values(playback.media).map(audio=>audio.play()));
  }catch(error){
    stopPlayback();
    alert(`Lecture de la reconstruction impossible : ${error.message}`);
    return;
  }
  reconstructionScheduler();
  playback.timer=setInterval(reconstructionScheduler,playback.lookaheadMs);
  syncPlaybackUI();
  animatePlayback();
}
async function startSyntheticPlayback(AudioCtx){
  const ctx=new AudioCtx();
  playback.ctx=ctx;
  await ctx.resume();
  configureMaster(ctx);
  playback.kind="synthetic";
  playback.active=true;
  playback.cycle=selectedCycle(selectedRecording());
  playback.step=synthesisCycleDuration()/12;
  playback.stepIndex=0;
  playback.startTime=ctx.currentTime+.13;
  playback.nextStepTime=playback.startTime;
  playback.timeline=[];
  audioScheduler();
  playback.timer=setInterval(audioScheduler,playback.lookaheadMs);
  syncPlaybackUI();
  animatePlayback();
}
async function startPlayback(){
  const AudioCtx=window.AudioContext||window.webkitAudioContext;
  if(!AudioCtx){alert("Audio Web non disponible dans ce navigateur.");return;}
  if(referencePlayback.active)stopReferencePlayback();
  if(state.listeningMode==="recording")await startCorpusPlayback(AudioCtx);
  else if(state.listeningMode==="synthesis")await startSyntheticPlayback(AudioCtx);
  else if(state.listeningMode==="reconstruction")await startReconstructionPlayback(AudioCtx);
  else startReferencePlayback();
}

async function togglePlayback(){
  if(["reference","model"].includes(state.listeningMode)){
    if(referencePlayback.active){stopReferencePlayback();syncPlaybackUI();return;}
    if(playback.active)stopPlayback();
    startReferencePlayback();
    syncPlaybackUI();
    return;
  }
  if(playback.active){stopPlayback();return;}
  await startPlayback();
}
document.getElementById("playBtn").onclick=togglePlayback;

function animatePlayback(){
  drawCircle();
  drawCurve();
  drawMicroTiming();
  if(playback.active) playback.raf=requestAnimationFrame(animatePlayback);
  else playback.raf=null;
}

function render(){
  preloadDjembeLoopSamples();
  renderTabs();
  renderScopePicker();
  renderVoicePicker();
  renderVolumeControl();
  renderPhraseModeControls();
  renderSynthControls();
  renderSymmetryLibrary();
  renderTimingControls();
  drawCircle();
  drawCurve();
  renderDjembeLoopsReference();
  drawMicroTiming();
  drawMap();
  renderEquation();
  renderCoefTable();
  syncPlaybackUI();
  queueMath();
}
window.addEventListener("pagehide",()=>{
  stopPlayback();
  stopReferencePlayback();
});
render();
</script></body></html>"""


def build(out_html="output/boussole_rythmes.html"):
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(build_data(), ensure_ascii=False))
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    return out_html, len(html)


if __name__ == "__main__":
    path, n = build()
    print(f"OK -> {path} ({n // 1000} Ko)")
