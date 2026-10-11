"""Step a3: small neural models on stroke sequences, same folds and metrics as a2.

Input: a window of k cycles becomes a sequence of 12k metric positions. Each
position carries, per voice, a presence flag and the offset from the even grid
(fraction of a pulse, 0 when absent). Voices:
- ``full``:  D1 and jembe accompaniment (J1/J2 averaged), 4 channels;
- ``jembe``: jembe accompaniment only, 2 channels (no dundun pattern).

Models (deliberately small for at most about 2,000 windows: 11k parameters
for the CNN, 68k for the transformer):
- ``cnn``: two 1D convolutions, global average pooling, linear head;
- ``transformer``: 2 encoder layers, d_model 64, learned embedding of the pulse
  position (1..12), mean pooling, linear head.

Training is fixed in advance (epochs, learning rate, weight decay). Nothing is
tuned on the test take. Runs on CPU (2 threads) or GPU.

    python sequence_models.py --windows 1 4 --seeds 0
"""
from __future__ import annotations

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "2")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import djembe_dataset as d
import features as f
from recognition_baselines import LABELS, summarise

ROOT = Path(__file__).resolve().parent
VOICES = {"full": ("D1", "J"), "jembe": ("J",)}
EPOCHS, BATCH, LR, WEIGHT_DECAY, NOISE_SD = 40, 64, 1e-3, 1e-2, 0.02


# ───────────────────────────── data ─────────────────────────────
def window_sequences(events: pd.DataFrame, k: int, voices=("D1", "J")):
    """Return X (n, 12k, 2*len(voices)) float32, window meta DataFrame."""
    win = f.assign_windows(events, k).copy()
    win["voice"] = np.where(win["instrument"].isin(["J1", "J2"]), "J", win["instrument"])
    win = win[win["voice"].isin(voices)]
    cyc_rank = win.groupby("window")["cycle_idx"].rank(method="dense").astype(int) - 1
    win["pos"] = cyc_rank * f.N_SD + (win["sd"] - 1)

    meta = win.groupby("window").agg(take_id=("take_id", "first"), piece=("piece", "first")).reset_index()
    row = {w: i for i, w in enumerate(meta["window"])}
    X = np.zeros((len(meta), f.N_SD * k, 2 * len(voices)), dtype=np.float32)
    dev = win.groupby(["window", "voice", "pos"])["dev_sd"].mean().reset_index()
    for v_i, voice in enumerate(voices):
        part = dev[dev["voice"] == voice]
        r = part["window"].map(row).to_numpy()
        p = part["pos"].to_numpy()
        X[r, p, 2 * v_i] = 1.0
        X[r, p, 2 * v_i + 1] = part["dev_sd"].fillna(0.0).to_numpy()
    return X, meta


# ───────────────────────────── models ─────────────────────────────
class CNN(nn.Module):
    def __init__(self, channels: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes),
        )

    def forward(self, x):  # x: (batch, length, channels)
        return self.net(x.transpose(1, 2))


class Transformer(nn.Module):
    def __init__(self, channels: int, n_classes: int = 3, d_model: int = 64):
        super().__init__()
        self.inp = nn.Linear(channels, d_model)
        self.pulse = nn.Embedding(f.N_SD, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=128, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_model, n_classes))

    def forward(self, x):
        pulse = torch.arange(x.shape[1], device=x.device) % f.N_SD
        h = self.encoder(self.inp(x) + self.pulse(pulse))
        return self.head(h.mean(dim=1))


MODELS = {"cnn": CNN, "transformer": Transformer}


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ───────────────────────────── training ─────────────────────────────
def fit_predict(model_name, X_train, y_train, X_test, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MODELS[model_name](X_train.shape[2]).to(device)
    counts = np.bincount(y_train, minlength=len(LABELS)).astype(np.float32)
    weight = torch.tensor(counts.sum() / (len(LABELS) * np.maximum(counts, 1)), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xt = torch.tensor(X_train, device=device)
    yt = torch.tensor(y_train, device=device)
    timing_channels = torch.arange(1, X_train.shape[2], 2, device=device)
    model.train()
    for _ in range(EPOCHS):
        order = torch.tensor(rng.permutation(len(Xt)), device=device)
        for start in range(0, len(order), BATCH):
            idx = order[start:start + BATCH]
            xb = Xt[idx].clone()
            # timing jitter on played positions only (augmentation)
            present = xb[:, :, timing_channels - 1]
            xb[:, :, timing_channels] += NOISE_SD * torch.randn_like(present) * present
            opt.zero_grad()
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X_test, device=device)).argmax(1).cpu().numpy()


def evaluate(events, k, voice, model_name, splits, device, seeds, rng):
    X, meta = window_sequences(events, k, VOICES[voice])
    y = meta["piece"].map({p: i for i, p in enumerate(LABELS)}).to_numpy()
    per_seed = []
    for seed in seeds:
        preds = []
        for fold, train, test in d.iter_folds(meta, splits):
            p = fit_predict(model_name, X[train.index], y[train.index], X[test.index], device, seed)
            preds.append(test[["window", "take_id", "piece"]].assign(pred=[LABELS[i] for i in p]))
        per_seed.append(summarise(pd.concat(preds, ignore_index=True), rng))
    return {
        "n_parameters": n_parameters(MODELS[model_name](X.shape[2])),
        "seeds": list(seeds),
        "balanced_accuracy_mean": float(np.mean([r["balanced_accuracy"] for r in per_seed])),
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default=str(d.DEFAULT_EVENTS))
    parser.add_argument("--out", default=str(ROOT / "results" / "a3_sequence.json"))
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--voices", nargs="+", default=list(VOICES))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    events = d.load_events(args.events)
    splits = d.load_splits()
    rng = np.random.default_rng(0)
    results = {}
    for k in args.windows:
        for voice in args.voices:
            for name in args.models:
                t0 = time.time()
                res = evaluate(events, k, voice, name, splits, device, args.seeds, rng)
                results.setdefault(str(k), {}).setdefault(voice, {})[name] = res
                ci = res["per_seed"][0]["ci95_bootstrap_takes"]
                print(f"k={k} {voice:6s} {name:12s} bal {res['balanced_accuracy_mean']:.3f} "
                      f"(seed0 CI [{ci[0]:.2f}, {ci[1]:.2f}]) {time.time() - t0:.0f}s on {device}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"device": device, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY,
              "timing_noise_sd": NOISE_SD, "splits": "splits.json (leave-one-take-out)", "results": results}
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
