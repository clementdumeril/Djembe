# Results and commands

All scores are leave-one-take-out over the 15 takes.

## Recognition (balanced accuracy)

| What | Result | Source |
|---|---|---|
| Piece from the whole-take dundun pattern | 15/15 takes for random forest and nearest centroid (exact 95 % CI 0.78 to 1.00) | [`step0_rf.json`](../results/step0_rf.json) |
| Piece from one cycle of dundun pattern | 0.99 (logistic regression) | [`a2_recognition.json`](../results/a2_recognition.json) |
| Piece from jembe timing only, pattern masked, 4 cycles | 0.86 (95 % CI 0.74 to 0.95) | [`a2_recognition.json`](../results/a2_recognition.json) |
| Same, restricted to the tempo range shared by all pieces | 0.87; tempo alone drops to 0.29 (logistic regression) and 0.44 (nearest centroid) | [`a2_tempo_control.json`](../results/a2_tempo_control.json) |
| Same, tempo regressed out of every feature | 0.78 (95 % CI 0.64 to 0.90) | [`a2_tempo_control.json`](../results/a2_tempo_control.json) |
| 1D CNN on jembe stroke sequences, 1 / 8 cycles, mean of 3 seeds | 0.78 / 0.74, against 0.77 / 0.89 for logistic regression on the jembe feel | [`a3_sequence_cnn_cpu.json`](../results/a3_sequence_cnn_cpu.json) |

![Balanced accuracy against window length per feature set](figures/a2_window_accuracy.png)

## Beat symmetry

Feel: spread of the beat-averaged offset profile over the three pulses. Breach: how much the four beats differ at the same position. Noise floor from random beat rotation: 0.5 to 1.0 ms.

| Piece, drum | Strokes (takes) | Feel | Breach | Breach / feel |
|---|---|---|---|---|
| Manjanin, jembe 1 | 5,215 (4) | 8.7 ms | 1.8 ms | 0.21 |
| Manjanin, jembe 2 | 4,440 (3) | 8.3 ms | 1.4 ms | 0.16 |
| Maraka, jembe 1 | 4,812 (6) | 6.5 ms | 1.5 ms | 0.23 |
| Maraka, jembe 2 | 4,092 (5) | 3.3 ms | 2.2 ms | 0.66 |
| Woloso, jembe 1 | 4,389 (5) | 15.3 ms | 0.8 ms | 0.05 |
| Woloso, jembe 2 | 3,098 (4) | 19.5 ms | 1.2 ms | 0.06 |
| Woloso, dundun 1 | 2,313 (5) | 21.6 ms | 5.4 ms | 0.25 |
| Maraka, dundun 1 | 3,495 (6) | one shared position | 3.8 ms | – |

![Mean offset at each pulse of each beat](figures/j1_beat_symmetry.png)

Source: [`j1_beat_symmetry.json`](../results/j1_beat_symmetry.json).

## Reference models (negative log-likelihood, nats)

Jembe 1, jembe 2, dundun 1. Histogram intersection and error of the mean as in Mishra et al. (2025). Each take's median offset is removed.

| Model | Strikes | Offsets | Intensity | Histogram intersection | Error of the mean |
|---|---|---|---|---|---|
| Global | 0.655 | −0.367 | 1.417 | 0.61 | 11.6 ms |
| B1 position | 0.311 | −0.577 | 1.300 | 0.73 | 4.6 ms |
| B2 Markov | 0.256 | −0.900 | 1.206 | 0.73 | 5.0 ms |

![Negative log-likelihood per held-out take](figures/j2_baselines.png)

Source: [`j2_baselines.json`](../results/j2_baselines.json).

## Humanizing a cycle, first run (J3)

| Model | Offsets, 3 seeds (Colab) | Offsets, seed 0 (laptop) | Intensity, 3 seeds |
|---|---|---|---|
| B1 position | −0.579 | −0.579 | 1.294 |
| B2 Markov, earlier cycles | −0.709 | −0.709 | 1.286 |
| GNN V0, no symmetry | −0.709 | −0.758 | 0.907 |
| GNN V1, beat symmetry (C4) | −0.674 | −0.670 | 0.904 |
| GNN V3, pulse rotation (C12) | −0.659 | −0.687 | 0.918 |

![Negative log-likelihood per held-out take, mean and spread over three seeds](figures/j3_humanize.png)

Sources: [`j3_humanize_3seeds_colab.json`](../results/j3_humanize_3seeds_colab.json) (rebuilt from [`j3_wandb_export.csv`](../results/j3_wandb_export.csv) with [`fetch_wandb.py`](../experiments/gnn/fetch_wandb.py)), [`j3_humanize_seed0_cpu.json`](../results/j3_humanize_seed0_cpu.json), Colab rerun [`j3_rerun_wandb_export.csv`](../results/j3_rerun_wandb_export.csv). The three fixes: [`j3b_humanize_seed0_cpu.json`](../results/j3b_humanize_seed0_cpu.json).

## Commands

```bash
pip install -e ".[dev]"
python -m djembe.dataset download     # annotations only, about 10 MB from OSF
python -m djembe.dataset build        # output/events.csv, about 7 s
python -m djembe.dataset splits       # regenerates splits.json (deterministic)
python -m djembe.dataset pack         # optional: output/djembe_corpus.zip, a private backup of the annotations
python -m pytest -q                   # a few seconds
python experiments/recognition/step0_rf_baseline.py      # results/step0_rf.json, about 20 s
python experiments/recognition/recognition_baselines.py  # results/a2_recognition.json and figure, several minutes
python experiments/recognition/tempo_control.py          # results/a2_tempo_control.json, a few minutes
python experiments/recognition/sequence_models.py --models cnn --out results/a3_sequence_cnn_cpu.json   # about 20 min on 2 threads
python experiments/recognition/make_figures.py           # docs/figures/
python experiments/symmetry/beat_symmetry.py             # results/j1_beat_symmetry.json and figure, a few minutes
python experiments/baselines/run_baselines.py            # results/j2_baselines.json and figure, a few minutes
python experiments/gnn/train_humanize.py --variants V0 V1 V3 --seeds 0 --out results/j3_humanize_seed0_cpu.json   # about 1.5 h on a laptop CPU
python experiments/gnn/train_humanize.py --variants V0 V4 V5 --seeds 0 --select last --out results/j3b_humanize_seed0_cpu.json   # about 2 h in 4 parallel parts (--takes)
```
