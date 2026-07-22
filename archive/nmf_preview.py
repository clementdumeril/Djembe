"""
Test rapide : separation par NMF d'une video (mix mono) en voix instrumentales.
But : verifier que la NMF separe dunun (grave) / cloche (metallique) / djembe (large bande).
Sortie : figure des templates + activations, et un .wav resynthetise par composante (a ecouter).
"""
import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 22050
FILE = "audio/Woloso.wav"
DEBUT, DUREE = 30, 45          # extrait analyse (s)
K = 8                          # nombre de briques NMF
N_FFT, HOP = 2048, 512
OUT = "output/nmf"
os.makedirs(OUT, exist_ok=True)

# --- charge un extrait ---
y, _ = librosa.load(FILE, sr=SR, mono=True, offset=DEBUT, duration=DUREE)

# --- STFT (on garde la phase pour resynthetiser) ---
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
S = np.abs(D)
freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

# --- NMF : S ~= W (freq x K)  @  H (K x temps) ---
W, H = librosa.decompose.decompose(S, n_components=K, sort=False, random_state=0)

# trie les composantes du plus grave au plus aigu (centroide spectral du template)
centroids = (W * freqs[:, None]).sum(0) / (W.sum(0) + 1e-9)
order = np.argsort(centroids)
W, H, centroids = W[:, order], H[order], centroids[order]

# --- caracterise + resynthetise chaque composante ---
print(f"Extrait : {FILE}  {DEBUT}-{DEBUT+DUREE}s   |   {K} composantes NMF\n")
print(f"{'comp':>4} {'centroide':>10} {'pic Hz':>8}   famille probable")
for k in range(K):
    peak = freqs[np.argmax(W[:, k])]
    cen = centroids[k]
    if cen < 250:
        fam = "dunun (grave)"
    elif cen > 4000:
        fam = "cloche / slap (aigu metallique)"
    else:
        fam = "djembe (medium / large bande)"
    print(f"{k:>4} {cen:>9.0f}Hz {peak:>7.0f}   {fam}")
    # masque doux facon Wiener -> resynthese de la composante
    Vk = np.outer(W[:, k], H[k])
    mask = Vk / (W @ H + 1e-9)
    yk = librosa.istft(mask * D, hop_length=HOP, length=len(y))
    sf.write(f"{OUT}/comp{k}_{cen:.0f}Hz.wav", yk, SR)

# --- figure : templates spectraux + activations ---
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for k in range(K):
    ax[0].semilogx(freqs[1:], W[1:, k] / (W[1:, k].max() + 1e-9) + k,
                   label=f"c{k} ({centroids[k]:.0f}Hz)")
ax[0].set(title="Templates spectraux NMF (1 brique = 1 voix)",
          xlabel="Frequence (Hz, log)", ylabel="composante (decalee)")
ax[0].legend(fontsize=8, loc="upper right")

t = librosa.frames_to_time(np.arange(H.shape[1]), sr=SR, hop_length=HOP)
ax[1].imshow(H / (H.max(1, keepdims=True) + 1e-9), aspect="auto", origin="lower",
             extent=[t[0], t[-1], -0.5, K - 0.5], cmap="magma")
ax[1].set(title="Activations dans le temps (quand chaque voix joue)",
          xlabel="temps (s)", ylabel="composante")
plt.tight_layout()
plt.savefig("output/nmf_templates.png", dpi=140)
print(f"\nFigure -> output/nmf_templates.png")
print(f"Audio par composante -> {OUT}/comp*.wav  (a ecouter pour identifier l'instrument)")
