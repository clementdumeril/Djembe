"""
DJEMBE_LLM - Pipeline d'analyse des frappes de djembe facon "oiseaux".

Etapes:
  1. Charger chaque rythme (mono 22050 Hz)
  2. Detecter les onsets (chaque frappe)
  3. Extraire un vecteur de features par frappe (timbre)
  4. Clustering non-supervise (KMeans k=3) -> bass / tone / slap
  5. Projection 2D (UMAP) facon visualisation chant d'oiseaux
  6. Sauvegarder CSV + figures dans output/

Usage:  python analyze.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

warnings.filterwarnings("ignore")

SR = 22050
WIN = 0.080            # fenetre d'analyse apres l'onset (s)
AUDIO_DIR = "audio"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# couleurs facon "oiseaux" : un ton par type de frappe
STROKE_COLORS = {"bass": "#2b6cff", "tone": "#d63bd6", "slap": "#2ec45f"}


def extract_strokes(path, rhythm):
    """Detecte les onsets et renvoie un DataFrame de features par frappe."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    # detection des frappes
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=SR, backtrack=True, units="frames",
        pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=10,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=SR)

    rows = []
    win_len = int(WIN * SR)
    for t in onset_times:
        start = int(t * SR)
        seg = y[start:start + win_len]
        if len(seg) < win_len // 2:
            continue
        # --- features de timbre ---
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13).mean(axis=1)
        cent = librosa.feature.spectral_centroid(y=seg, sr=SR).mean()
        bw = librosa.feature.spectral_bandwidth(y=seg, sr=SR).mean()
        roll = librosa.feature.spectral_rolloff(y=seg, sr=SR).mean()
        flat = librosa.feature.spectral_flatness(y=seg).mean()
        zcr = librosa.feature.zero_crossing_rate(seg).mean()
        rms = librosa.feature.rms(y=seg).mean()
        row = {"rhythm": rhythm, "time": float(t),
               "centroid": cent, "bandwidth": bw, "rolloff": roll,
               "flatness": flat, "zcr": zcr, "rms": rms}
        row.update({f"mfcc{i}": v for i, v in enumerate(mfcc)})
        rows.append(row)
    print(f"  {rhythm:14s} : {len(rows):5d} frappes")
    return pd.DataFrame(rows)


def label_clusters(df, labels, km_features):
    """Nomme les clusters bass/tone/slap selon le centroide spectral moyen."""
    df = df.copy()
    df["cluster"] = labels
    order = df.groupby("cluster")["centroid"].mean().sort_values().index.tolist()
    names = {order[0]: "bass", order[1]: "tone", order[2]: "slap"}
    df["stroke"] = df["cluster"].map(names)
    return df["stroke"].values


def main():
    print("1) Detection des onsets + features par fichier")
    wavs = sorted(f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav"))
    parts = [extract_strokes(os.path.join(AUDIO_DIR, w), os.path.splitext(w)[0])
             for w in wavs]
    df = pd.concat(parts, ignore_index=True)
    print(f"   -> total {len(df)} frappes sur {len(wavs)} rythmes\n")

    feat_cols = ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"] + \
                [f"mfcc{i}" for i in range(13)]
    X = StandardScaler().fit_transform(df[feat_cols].values)

    print("2) Clustering KMeans (k=3) -> bass / tone / slap")
    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    df["stroke"] = label_clusters(df, km.labels_, X)
    print(df["stroke"].value_counts().to_string(), "\n")

    print("3) Projection UMAP 2D")
    emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)
    df["umap_x"], df["umap_y"] = emb[:, 0], emb[:, 1]

    # --- figure 1 : colore par type de frappe ---
    fig, ax = plt.subplots(figsize=(11, 9), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    for stroke, color in STROKE_COLORS.items():
        m = df["stroke"] == stroke
        ax.scatter(df.loc[m, "umap_x"], df.loc[m, "umap_y"], s=6, c=color,
                   alpha=0.6, label=stroke, edgecolors="none")
    ax.set_title("Frappes de djembe (bass / tone / slap) - projection UMAP",
                 color="white")
    leg = ax.legend(markerscale=3, facecolor="#1a1a1a", labelcolor="white")
    ax.tick_params(colors="#666"); [s.set_color("#333") for s in ax.spines.values()]
    fig.savefig(os.path.join(OUT_DIR, "umap_strokes.png"), dpi=140,
                facecolor="#0d0d0d", bbox_inches="tight")

    # --- figure 2 : colore par rythme ---
    fig2, ax2 = plt.subplots(figsize=(11, 9), facecolor="#0d0d0d")
    ax2.set_facecolor("#0d0d0d")
    cmap = plt.get_cmap("turbo")
    rhythms = sorted(df["rhythm"].unique())
    for i, r in enumerate(rhythms):
        m = df["rhythm"] == r
        ax2.scatter(df.loc[m, "umap_x"], df.loc[m, "umap_y"], s=6,
                    c=[cmap(i / max(1, len(rhythms) - 1))], alpha=0.6,
                    label=r, edgecolors="none")
    ax2.set_title("Frappes colorees par rythme - projection UMAP", color="white")
    ax2.legend(markerscale=3, facecolor="#1a1a1a", labelcolor="white")
    ax2.tick_params(colors="#666"); [s.set_color("#333") for s in ax2.spines.values()]
    fig2.savefig(os.path.join(OUT_DIR, "umap_rhythms.png"), dpi=140,
                 facecolor="#0d0d0d", bbox_inches="tight")

    df.to_csv(os.path.join(OUT_DIR, "strokes.csv"), index=False)
    print(f"\nOK -> {OUT_DIR}/umap_strokes.png, umap_rhythms.png, strokes.csv")


if __name__ == "__main__":
    main()
