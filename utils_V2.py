"""
utils_V2 — module du chantier « Analyse des dununs » (V2).
Téléchargement du corpus (pistes de dunun + annotations) depuis OSF, chargement,
et features. Importé par les notebooks : `import utils_V2 as u`.
"""
import os, re, glob, json, urllib.request
import numpy as np
import pandas as pd
import librosa

SR = 22050
CORPUS = "corpus"
PIECES = ["Manjanin", "Maraka", "Woloso"]
OSF_NODE = "m652x"                       # IEMP Malian Jembe (Polak), accès libre
CODE2INST = {"D1": "Dundun1", "D2": "Dundun2"}

FAST_FEATS = ["centroid", "low_ratio", "flatness", "zcr", "rms"]


# ─────────────────────────── Téléchargement (OSF) ───────────────────────────
def _osf_list(path):
    url = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/dropbox/{path}/?page[size]=200"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req))["data"]


def _grab(url, dest):
    if os.path.exists(dest):
        return "déjà là"
    urllib.request.urlretrieve(url, dest)
    return "téléchargé"


def download_dunun_corpus(pieces=PIECES, verbose=True):
    """Télécharge les pistes de dunun (D1/D2) + annotations (onsets dunun,
    grille métrique, positions) des pièces demandées. Idempotent."""
    report = []
    for piece in pieces:
        base = f"{CORPUS}/MJ_{piece}"
        os.makedirs(f"{base}/annotations", exist_ok=True)
        nw = nc = 0
        for it in _osf_list(f"Media/MJ_{piece}"):
            a = it["attributes"]
            if a["kind"] == "file" and re.search(r"_D[12]\.wav$", a["name"]):
                if _grab(it["links"]["download"], f"{base}/{a['name']}") == "téléchargé":
                    nw += 1
        for it in _osf_list(f"Annotations/MJ_{piece}"):
            a = it["attributes"]
            if a["kind"] == "file" and re.search(r"(Onsets_Raw_Dundun|Metre|Onsets_Selected)", a["name"]) and a["name"].endswith(".csv"):
                if _grab(it["links"]["download"], f"{base}/annotations/{a['name']}") == "téléchargé":
                    nc += 1
        report.append({"pièce": piece, "wav_nouveaux": nw, "csv_nouveaux": nc})
        if verbose:
            print(f"  {piece:10s} : +{nw} WAV, +{nc} CSV")
    return pd.DataFrame(report)


# ─────────────────────────── Chargement ───────────────────────────
def list_dunun():
    """Une ligne par piste de dunun présente : piece, take, code, wav."""
    rows = []
    for wav in sorted(glob.glob(f"{CORPUS}/MJ_*/MJ_*_D[12].wav")):
        m = re.match(r"MJ_(\w+?)_(\d+)_(D[12])\.wav", os.path.basename(wav))
        if m:
            rows.append({"piece": m.group(1), "take": int(m.group(2)),
                         "code": m.group(3), "wav": wav})
    return pd.DataFrame(rows)


def dunun_coverage():
    """Table de couverture D1/D2 par pièce et prise.

    D2 n'est pas présent dans toutes les prises du corpus. Cette fonction rend
    cette asymétrie explicite afin que les analyses comparables entre rythmes
    utilisent D1 comme voix complète, et que les comparaisons D1/D2 soient
    limitées aux prises réellement appariées.
    """
    tracks = list_dunun()
    if tracks.empty:
        return pd.DataFrame(columns=[
            "piece", "take", "D1_wav", "D2_wav", "D1_selected", "D2_selected",
            "D1_raw_onsets", "D2_raw_onsets", "paired",
        ])

    rows = []
    for piece in PIECES:
        takes = sorted(tracks.loc[tracks["piece"] == piece, "take"].unique())
        for take in takes:
            row = {"piece": piece, "take": int(take)}
            sel = load_selected(piece, int(take))
            for code in ("D1", "D2"):
                has_wav = not tracks[
                    (tracks["piece"] == piece) & (tracks["take"] == take) & (tracks["code"] == code)
                ].empty
                on = load_onsets(piece, int(take), code) if has_wav else None
                row[f"{code}_wav"] = bool(has_wav)
                row[f"{code}_selected"] = bool(sel is not None and code in sel.columns and sel[code].notna().any())
                row[f"{code}_raw_onsets"] = 0 if on is None else int(len(on))
            row["paired"] = bool(row["D1_wav"] and row["D2_wav"] and row["D1_selected"] and row["D2_selected"])
            rows.append(row)
    return pd.DataFrame(rows)


def paired_dunun_takes():
    """Prises où D1 et D2 sont tous deux disponibles et annotés."""
    cov = dunun_coverage()
    return cov[cov["paired"]].reset_index(drop=True)


def representative_dunun_take(piece, prefer_paired=True):
    """Choisit une prise représentative pour les visualisations.

    Par défaut, on privilégie une prise appariée D1/D2. S'il n'y en a pas,
    on retombe sur la première prise D1 disponible. Retourne `None` si aucune
    piste de dunun n'est disponible pour la pièce.
    """
    if prefer_paired:
        paired = paired_dunun_takes()
        paired_piece = paired[paired["piece"] == piece]
        if not paired_piece.empty:
            return int(paired_piece["take"].min())

    tracks = list_dunun()
    d1 = tracks[(tracks["piece"] == piece) & (tracks["code"] == "D1")]
    if d1.empty:
        return None
    return int(d1["take"].min())


def dunun_pair_metrics():
    """Mesures D1/D2 uniquement sur les prises appariées.

    Retourne la densité par cycle de D1/D2 et, quand les annotations contiennent
    des frappes simultanément renseignées, le décalage D2-D1 en millisecondes.
    """
    rows = []
    for r in paired_dunun_takes().itertuples(index=False):
        p1 = subdivision_profile(r.piece, int(r.take), "D1")
        p2 = subdivision_profile(r.piece, int(r.take), "D2")
        sel = load_selected(r.piece, int(r.take))
        both = pd.DataFrame()
        if sel is not None and {"D1", "D2"}.issubset(sel.columns):
            both = sel[["D1", "D2"]].dropna()

        row = {
            "piece": r.piece,
            "take": int(r.take),
            "D1_density_per_cycle": np.nan if p1 is None else float(p1.sum()),
            "D2_density_per_cycle": np.nan if p2 is None else float(p2.sum()),
            "paired_onsets": int(len(both)),
            "D2_minus_D1_median_ms": np.nan,
            "D2_minus_D1_std_ms": np.nan,
        }
        if len(both) > 0:
            off = (both["D2"] - both["D1"]) * 1000
            row["D2_minus_D1_median_ms"] = float(off.median())
            row["D2_minus_D1_std_ms"] = float(off.std())
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["D2_to_D1_density_ratio"] = out["D2_density_per_cycle"] / out["D1_density_per_cycle"]
    return out


def load_track(piece, take, code, sr=SR):
    return librosa.load(f"{CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav", sr=sr, mono=True)[0]


def _ann(piece, take, suffix):
    fs = glob.glob(f"{CORPUS}/MJ_{piece}/annotations/*{piece}_{take}_{suffix}.csv")
    return pd.read_csv(fs[0]) if fs else None


def load_onsets(piece, take, code):
    """Onsets de référence d'un dunun -> colonnes Time, Peak."""
    return _ann(piece, take, f"Onsets_Raw_{CODE2INST[code]}")


def load_selected(piece, take):
    return _ann(piece, take, "Onsets_Selected")


def load_metre(piece, take):
    return _ann(piece, take, "Metre")


# ─────────────────────────── Analyse ───────────────────────────
def subdivision_profile(piece, take, code):
    """Proba que `code` frappe chaque subdivision (1..12) du cycle."""
    df = load_selected(piece, take)
    if df is None or code not in df.columns:
        return None
    return df.groupby("SD")[code].apply(lambda s: s.notna().mean()).reindex(range(1, 13), fill_value=0.0)


def cycle_positions(piece, take, code):
    """Position relative (0–1) de chaque frappe du dunun `code` dans son cycle."""
    sel = load_selected(piece, take)
    if sel is None or code not in sel.columns:
        return pd.DataFrame(columns=["SD", "rel", "dur", "start"])
    sel = sel.copy(); sel["cyc"] = sel["Label SD"].astype(str).str.split("|").str[0]
    tcols = [i for i in ["J1", "J2", "D1", "D2"] if i in sel.columns]
    sel["ref"] = sel[tcols].min(axis=1, skipna=True)         # onset de référence de la position
    out, cyc = [], list(sel.groupby("cyc", sort=False))
    for k in range(len(cyc) - 1):
        g, gn = cyc[k][1], cyc[k + 1][1]
        r1, r1n = g.loc[g["SD"] == 1, "ref"].dropna(), gn.loc[gn["SD"] == 1, "ref"].dropna()
        if r1.empty or r1n.empty:
            continue
        start = float(r1.iloc[0]); dur = float(r1n.iloc[0]) - start
        if not (0.5 < dur < 5):
            continue
        for _, r in g.iterrows():
            if pd.notna(r[code]):
                rel = (r[code] - start) / dur
                if -0.05 < rel < 1.05:
                    out.append((int(r["SD"]), rel, dur, start, float(r[code])))
    return pd.DataFrame(out, columns=["SD", "rel", "dur", "start", "time"])


def cycle_durations(piece, take):
    """(débuts, durées) de chaque cycle -> pour le tempo/accelerando."""
    met = load_metre(piece, take)
    if met is None:
        return np.array([]), np.array([])
    t = met["Time"].values; d = np.diff(t); ok = (d > 0.5) & (d < 5)
    return t[:-1][ok], d[ok]


def fast_stroke_features(y, times, sr=SR, win=0.090):
    """Features de frappe vectorisées (FFT batch) pour tous les onsets `times`."""
    wl = int(win * sr)
    idx = (np.asarray(times) * sr).astype(int)
    keep = idx + wl < len(y)
    idx, times = idx[keep], np.asarray(times)[keep]
    if len(idx) == 0:
        return pd.DataFrame()
    Wn = np.stack([y[i:i + wl] for i in idx]) * np.hanning(wl)
    Sp = np.abs(np.fft.rfft(Wn, axis=1)); fr = np.fft.rfftfreq(wl, 1 / sr)
    tot = Sp.sum(1) + 1e-9
    return pd.DataFrame({
        "time": times,
        "centroid":  (Sp * fr).sum(1) / tot,
        "low_ratio": Sp[:, fr < 200].sum(1) / tot,
        "flatness":  np.exp(np.log(Sp + 1e-12).mean(1)) / (Sp.mean(1) + 1e-12),
        "zcr":       np.mean(np.abs(np.diff(np.sign(Wn), axis=1)) > 0, axis=1),
        "rms":       np.sqrt((Wn ** 2).mean(1)),
    })
