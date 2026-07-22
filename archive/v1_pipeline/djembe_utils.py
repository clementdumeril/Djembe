"""
djembe_utils — outils partagés pour le projet DJEMBE_LLM.
Chargement du corpus IEMP Malian Jembe (pistes isolées + annotations) et features.
Importé par tous les notebooks : `import djembe_utils as dj`.
"""
import os, re, glob
import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import NMF

SR = 22050
CORPUS = "corpus"
PIECES = ["Manjanin", "Maraka", "Woloso"]

# Paramètres standards pour la NMF
NFFT = 2048
HOP = 512
KD = 6
KJ = 12

# code de piste -> (nom dans les CSV d'onsets, rôle)
CODE2INST = {"J1": "Jembe1", "J2": "Jembe2", "D1": "Dundun1", "D2": "Dundun2"}
ROLE      = {"J1": "jembe_solo", "J2": "jembe_accomp", "D1": "dunun", "D2": "dunun"}

FEAT_COLS = ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms", "low_ratio"] + \
            [f"mfcc{i}" for i in range(13)]


def list_corpus():
    """Une ligne par piste isolée : piece, take, code, role, wav."""
    rows = []
    for wav in sorted(glob.glob(f"{CORPUS}/MJ_*/MJ_*_*.wav")):
        m = re.match(r"MJ_(\w+?)_(\d+)_(J1|J2|D1|D2)\.wav", os.path.basename(wav))
        if not m:
            continue
        piece, take, code = m.group(1), int(m.group(2)), m.group(3)
        rows.append({"piece": piece, "take": take, "code": code,
                     "role": ROLE[code], "wav": wav})
    return pd.DataFrame(rows)


def load_track(piece, take, code, sr=SR):
    """Charge une piste isolée (mono)."""
    return librosa.load(f"{CORPUS}/MJ_{piece}/MJ_{piece}_{take}_{code}.wav",
                        sr=sr, mono=True)[0]


def _ann(piece, take, suffix):
    fs = glob.glob(f"{CORPUS}/MJ_{piece}/annotations/*{piece}_{take}_{suffix}.csv")
    return pd.read_csv(fs[0]) if fs else None


def load_onsets(piece, take, code):
    """Onsets de référence d'un instrument -> colonnes Time, Peak."""
    return _ann(piece, take, f"Onsets_Raw_{CODE2INST[code]}")


def load_selected(piece, take):
    """Onsets posés sur la grille métrique (cycle|subdivision, colonnes J1/J2/D1/D2 = temps)."""
    return _ann(piece, take, "Onsets_Selected")


def load_metre(piece, take):
    """Repères de cycle (Cycle, Time)."""
    return _ann(piece, take, "Metre")


def subdivision_profile(piece, take, code):
    """Probabilité que `code` frappe chaque subdivision (1..12) du cycle.
    = signature rythmique de l'instrument pour cette prise."""
    df = load_selected(piece, take)
    if df is None or code not in df.columns:
        return None
    occ = df.groupby("SD")[code].apply(lambda s: s.notna().mean())
    return occ.reindex(range(1, 13), fill_value=0.0)


def take_codes(piece, take):
    """Codes des instruments présents dans une prise, ex. ['D1','J1','J2']."""
    fs = glob.glob(f"{CORPUS}/MJ_{piece}/MJ_{piece}_{take}_*.wav")
    return sorted(re.match(r".*_(J1|J2|D1|D2)\.wav", os.path.basename(f)).group(1) for f in fs)


def build_mix(piece, take, sr=SR):
    """Mélange mono = somme des pistes isolées d'une prise. Renvoie (mix, codes)."""
    codes = take_codes(piece, take)
    tracks = [load_track(piece, take, cd, sr) for cd in codes]
    m = min(map(len, tracks))
    return sum(t[:m] for t in tracks), codes


def determinism(piece, take, code):
    """0 = motif varié (soliste), 1 = motif cyclique déterministe (accompagnement)."""
    p = subdivision_profile(piece, take, code)
    return float(np.mean(np.abs(p.values - 0.5) * 2)) if p is not None else np.nan


def track_timbre(y, sr=SR):
    """Timbre global d'une piste : (centroïde spectral moyen, ratio d'énergie <200 Hz)."""
    cen = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    S = np.abs(librosa.stft(y, n_fft=2048)); fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = float(S[fr < 200].sum() / (S.sum() + 1e-9))
    return cen, low


def cycle_positions(piece, take):
    """Position relative (0–1) de chaque onset dans son cycle. Base du micro-timing."""
    sel = load_selected(piece, take)
    if sel is None:
        return pd.DataFrame(columns=['SD', 'rel', 'dur', 'start'])
    sel = sel.copy(); sel['cyc'] = sel['Label SD'].astype(str).str.split('|').str[0]
    tcols = [i for i in ['J1', 'J2', 'D1', 'D2'] if i in sel.columns]
    sel['t'] = sel[tcols].min(axis=1, skipna=True)
    out, cyc = [], list(sel.groupby('cyc', sort=False))
    for k in range(len(cyc) - 1):
        g, gn = cyc[k][1], cyc[k + 1][1]
        t1, t1n = g.loc[g['SD'] == 1, 't'].dropna(), gn.loc[gn['SD'] == 1, 't'].dropna()
        if t1.empty or t1n.empty:
            continue
        start = float(t1.iloc[0]); dur = float(t1n.iloc[0]) - start
        if not (0.5 < dur < 5):
            continue
        for _, r in g.iterrows():
            if pd.notna(r['t']):
                rel = (r['t'] - start) / dur
                if -0.05 < rel < 1.05:
                    out.append((int(r['SD']), rel, dur, start))
    return pd.DataFrame(out, columns=['SD', 'rel', 'dur', 'start'])


def swing_ratio(piece, takes):
    """Ratio moyen court/moyen/long (%) des 3 subdivisions d'une pulsation."""
    d = pd.concat([cycle_positions(piece, int(t)) for t in takes], ignore_index=True)
    if d.empty:
        return np.array([np.nan] * 3)
    m = d.groupby('SD')['rel'].mean(); ratios = []
    for beat in range(4):
        s1, s2, s3 = beat*3+1, beat*3+2, beat*3+3
        end = m[beat*3+4] if beat*3+4 <= 12 else 1.0
        if all(k in m.index for k in (s1, s2, s3)):
            d1, d2, d3 = m[s2]-m[s1], m[s3]-m[s2], end-m[s3]; tot = d1+d2+d3
            ratios.append([d1/tot*100, d2/tot*100, d3/tot*100])
    return np.mean(ratios, axis=0) if ratios else np.array([np.nan]*3)


def intensity_curve(piece, take, win=4.0):
    """Courbe d'intensité musicale (0–100) = moyenne normalisée de densité, énergie, tempo."""
    mix, codes = build_mix(piece, take)
    on = np.sort(np.concatenate([load_onsets(piece, take, cd)['Time'].values for cd in codes]))
    dur = len(mix) / SR
    rms_all = librosa.feature.rms(y=mix, hop_length=512)[0]
    trms = librosa.frames_to_time(np.arange(len(rms_all)), sr=SR, hop_length=512)
    met = load_metre(piece, take)
    tg = np.arange(0, max(dur - win, win), 1.0)
    dens = np.array([np.sum((on >= t) & (on < t + win)) / win for t in tg])
    rms = np.array([rms_all[(trms >= t) & (trms < t + win)].mean() for t in tg])
    if met is not None:
        ct = met['Time'].values; d = np.diff(ct); ok = (d > 0.5) & (d < 5)
        tempo = np.interp(tg, ct[:-1][ok], 60 / (d[ok] / 4)) if ok.sum() > 1 else np.full_like(tg, np.nan)
    else:
        tempo = np.full_like(tg, np.nan)
    z = lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    inten = (z(dens) + z(rms) + z(tempo)) / 3 * 100
    return pd.DataFrame({'t': tg, 'density': dens, 'rms': rms, 'tempo': tempo, 'intensity': inten})


def stroke_features(seg, sr=SR):
    """Vecteur de timbre d'une frappe (centroïde, flatness, MFCC, ...)."""
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13).mean(axis=1)
    # ratio d'énergie grave (<200 Hz) : fort pour la basse, faible pour le slap
    w = seg * np.hanning(len(seg)) if len(seg) > 8 else seg
    sp = np.abs(np.fft.rfft(w)); fr = np.fft.rfftfreq(len(seg), 1/sr)
    low_ratio = float(sp[fr < 200].sum() / (sp.sum() + 1e-9))
    d = {"centroid":  librosa.feature.spectral_centroid(y=seg, sr=sr).mean(),
         "bandwidth": librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean(),
         "rolloff":   librosa.feature.spectral_rolloff(y=seg, sr=sr).mean(),
         "flatness":  librosa.feature.spectral_flatness(y=seg).mean(),
         "zcr":       librosa.feature.zero_crossing_rate(seg).mean(),
         "rms":       librosa.feature.rms(y=seg).mean(),
         "low_ratio": low_ratio}
    d.update({f"mfcc{i}": v for i, v in enumerate(mfcc)})
    return d


def strokes_dataframe(piece, take, code, win=0.080, sr=SR, max_per_track=None, seed=0):
    """Frappes d'une piste isolée : features + temps (via onsets de référence).
    max_per_track : si défini, échantillonne ce nb d'onsets (accélère l'extraction)."""
    y = load_track(piece, take, code, sr)
    on = load_onsets(piece, take, code)
    if on is None:
        return pd.DataFrame()
    times = on["Time"].values
    if max_per_track and len(times) > max_per_track:
        rng = np.random.RandomState(seed)
        times = np.sort(rng.choice(times, max_per_track, replace=False))
    wl = int(win * sr); rows = []
    for t in times:
        s = int(t * sr); seg = y[s:s + wl]
        if len(seg) < wl // 2:
            continue
        r = {"piece": piece, "take": take, "code": code, "role": ROLE[code], "time": float(t)}
        r.update(stroke_features(seg, sr))
        rows.append(r)
    return pd.DataFrame(rows)


def fast_stroke_features(y, times, sr=SR, win=0.080):
    """Features de frappe VECTORISÉES (une seule FFT batch) — ~1000x plus rapide
    que stroke_features. Renvoie un DataFrame (time, centroid, low_ratio,
    flatness, zcr, rms) pour tous les onsets `times`."""
    wl = int(win * sr)
    idx = (np.asarray(times) * sr).astype(int)
    keep = idx + wl < len(y)
    idx, times = idx[keep], np.asarray(times)[keep]
    if len(idx) == 0:
        return pd.DataFrame()
    Wn = np.stack([y[i:i + wl] for i in idx]) * np.hanning(wl)
    Sp = np.abs(np.fft.rfft(Wn, axis=1))
    fr = np.fft.rfftfreq(wl, 1 / sr)
    tot = Sp.sum(1) + 1e-9
    return pd.DataFrame({
        "time": times,
        "centroid":  (Sp * fr).sum(1) / tot,
        "low_ratio": Sp[:, fr < 200].sum(1) / tot,
        "flatness":  np.exp(np.log(Sp + 1e-12).mean(1)) / (Sp.mean(1) + 1e-12),
        "zcr":       np.mean(np.abs(np.diff(np.sign(Wn), axis=1)) > 0, axis=1),
        "rms":       np.sqrt((Wn ** 2).mean(1)),
    })


FAST_FEATS = ["centroid", "low_ratio", "flatness", "zcr", "rms"]


def label_stroke_clusters(df):
    """Étiquette 3 clusters en bass/tone/slap par critères physiques
    (bass = max low_ratio, slap = max flatness, tone = le reste).
    `df` doit avoir une colonne 'cl' (0/1/2)."""
    g = df.groupby("cl").agg(low_ratio=("low_ratio", "mean"),
                             flatness=("flatness", "mean"))
    bass = g["low_ratio"].idxmax()
    slap = g.drop(bass)["flatness"].idxmax()
    tone = [i for i in g.index if i not in (bass, slap)][0]
    return df["cl"].map({bass: "bass", tone: "tone", slap: "slap"})


def spec(y):
    """Calcule le spectrogramme d'amplitude (STFT) standard."""
    return np.abs(librosa.stft(y, n_fft=NFFT, hop_length=HOP))


def learn_spectral_dict(files, k):
    """Apprend un dictionnaire spectral (freq x k) sur des pistes isolées.
    files : liste de dictionnaires de pistes issues de list_corpus()."""
    specs = []
    for f in files:
        if os.path.exists(f["wav"]):
            y = load_track(f["piece"], f["take"], f["code"], sr=SR)
            specs.append(spec(y))
    if not specs:
        return None
    S = np.concatenate(specs, axis=1)
    m = NMF(n_components=k, init='nndsvda', max_iter=200, random_state=0)
    m.fit(S.T)
    return m.components_.T   # freq x k


def nmf_activations(S, W, iters=100):
    """Calcule les activations H (k x temps) pour un spectrogramme S
    en gardant le dictionnaire W (freq x k) FIXE."""
    H = np.random.RandomState(0).rand(W.shape[1], S.shape[1]) + 1e-3
    WtW = W.T @ W
    for _ in range(iters):
        H *= (W.T @ S) / (WtW @ H + 1e-9)
    return H
