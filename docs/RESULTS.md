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

## Humanizing a cycle, three fixes and three seeds (J3b)

Fixes: keep the last epoch (`--select last`), helix edge (V5), correction to B1 (V4, dropped after seed 0). V0 and V5 then ran with three seeds on a Slurm cluster, one task per (take, seed).

| Model | Offsets, 3 seeds (cluster) | Offsets, seed 0 (laptop) | Intensity, 3 seeds | Offsets vs B2, per take |
|---|---|---|---|---|
| B1 position | −0.579 | −0.579 | 1.294 | |
| B2 Markov, earlier cycles | −0.709 | −0.709 | 1.286 | |
| GNN V0, last epoch | −0.780 | −0.784 | 0.886 | better on 13 of 15 (Wilcoxon p = 0.003) |
| GNN V5, last epoch + helix | −0.796 | −0.801 | 0.884 | better on 13 of 15 (Wilcoxon p = 0.005, sign test p = 0.007) |
| GNN V4, V5 + correction to B1 | | −0.652 | | better on 11 of 15 (seed 0) |

- Per-take tests on the seed mean of each take, two-sided. V5 against V0: better on 14 of 15, −0.020 (Wilcoxon p = 0.007).
- Seeds: V5 overall −0.791 / −0.805 / −0.793; each seed beats B2 on 13, 13 and 14 takes; the per-take standard deviation across seeds has a median of 0.013 nats (max 0.046).
- Exceptions: Woloso_1 ties B2 (−1.085 against −1.086); Manjanin_4 loses on all three seeds (V5 −0.421, V0 −0.494, B2 −0.590).
- Reproducibility: B1 and B2 are identical on the cluster and on the laptop; seed 0 of the network differs by 0.016 (V0) and 0.019 (V5) per take on average, 0.12 at most.

![Negative log-likelihood per held-out take over three seeds](figures/j3b_humanize_3seeds.png)

Sources: [`j3b_humanize_3seeds.json`](../results/j3b_humanize_3seeds.json) (45 parts merged with [`merge_results.py`](../experiments/gnn/merge_results.py)), [`j3b_humanize_seed0_cpu.json`](../results/j3b_humanize_seed0_cpu.json).

## Data-efficiency curve (J4)

Every variant trained on 2, 4, 8 and 13 takes, B1 and B2 fitted on the same takes; nested subsets (round robin across pieces, the test piece first), 3 repetitions varying subset and seed, `--select last`. 180 cluster tasks ([`train_curve.py`](../experiments/gnn/train_curve.py)).

| Training takes | B1 | B2 | V0 | V1 (C4) | V3 (C12) | V5 (helix) |
|---|---|---|---|---|---|---|
| 2 | −0.150 | −0.310 | −0.558 | −0.547 | −0.560 | −0.558 |
| 4 | −0.505 | −0.631 | −0.652 | −0.632 | −0.627 | −0.633 |
| 8 | −0.550 | −0.676 | −0.733 | −0.628 | −0.685 | −0.729 |
| 13 | −0.565 | −0.691 | −0.780 | −0.739 | −0.748 | −0.795 |

Per-take tests on the mean of the 3 repetitions, two-sided Wilcoxon; "wins" = takes where the first model scores lower:

| Training takes | V0 vs B2 | V1 vs V0 | V3 vs V0 | V5 vs V0 |
|---|---|---|---|---|
| 2 | 11/15, −0.276, p = 0.003 | 9/15, +0.005, p = 0.89 | 8/15, +0.010, p = 0.89 | 8/15, −0.002, p = 0.98 |
| 4 | 12/15, −0.039, p = 0.015 | 4/15, +0.016, p = 0.035 | 3/15, +0.031, p = 0.022 | 6/15, +0.009, p = 0.85 |
| 8 | 14/15, −0.066, p = 0.007 | 1/15, +0.076, p < 0.001 | 0/15, +0.047, p < 0.001 | 12/15, −0.002, p = 0.11 |
| 13 | 14/15, −0.094, p = 0.001 | 1/15, +0.041, p < 0.001 | 3/15, +0.037, p = 0.008 | 12/15, −0.017, p = 0.012 |

- Spread over repetitions (overall score, min to max): within 0.04 for V0 and V3 at every size; up to 0.085 for V1 and V5 with 2 and 4 takes; V1 at 8 takes spans −0.693 to −0.496 because one run diverged (Manjanin_4, repetition 2, +1.245 against about −0.4 for the other two).
- B2 is fitted per piece, so with 2 takes it sees one take of the test piece; the network pools all takes, which accounts for part of its lead at that size.

![Data-efficiency curve](figures/j4_curve.png)

Source: [`j4_curve.json`](../results/j4_curve.json) (180 parts merged with [`merge_results.py`](../experiments/gnn/merge_results.py)).

## Generation of the whole ensemble (J5)

Strike model (V5 graph, target strikes hidden) plus the timing model; every held-out take generated from 3 real cycles, all 48 strikes of a cycle drawn independently; Markov (B2) generator as reference. 15 takes x 3 seeds on the cluster.

| | GNN | Markov (B2) | Another real take |
|---|---|---|---|
| Next-cycle strike prediction, nats per pulse | 0.196 | 0.256 | |
| Strike-profile distance to the real take | 0.103 | 0.074 | 0.075 |
| Timing distance, ms | 6.8 | 6.3 | 8.4 |
| Cycles identical to a training cycle (real takes: 56 %) | 10 % | 24 % | 53 % |
| Windows recognised as the right piece | 96 % | 99 % | 100 % |

Per take: strike prediction better on 15/15 (Wilcoxon p < 0.001); in generation the strike profile is closer for Markov on 14/15 (p < 0.001), timing ties (6/15, p = 0.64). Source: [`j5_generation.json`](../results/j5_generation.json).

## Generation of the soloist (J5b)

Jembe 1 only, over the real ensemble (accompaniment known in the target cycle too); strikes drawn pulse by pulse by a strike model trained on random prefixes; temperature chosen on the validation take (0.8 in 44 of 45 runs). 15 takes x 3 seeds on the cluster.

| Prediction on the held-out soloist | GNN | B2 | GNN better on (Wilcoxon p) |
|---|---|---|---|
| Strikes, sequential NLL per pulse | 0.424 | 0.538 | 15/15 (0.0001) |
| Offsets, NLL | −0.783 | −0.659 | 12/15 (0.010) |
| Intensities, NLL | 1.240 | 1.398 | 15/15 (0.0001) |

| Generated soloist | GNN | Markov | Another real take | Real soloist |
|---|---|---|---|---|
| Strike-profile distance | 0.150 | 0.127 | 0.118 | 0 |
| Timing distance, ms | 7.3 | 6.2 | 7.9 | 0 |
| Distinct figures per cycle | 51 % | 77 % | 33 % | 36 % |
| Figure known from another take | 42 % | 34 % | | 68 % |
| One pulse from a known figure | 37 % | 47 % | | 22 % |
| Two or more pulses from any known figure | 21 % | 19 % | | 10 % |
| Returns to a figure played earlier in the take | 49 % | 23 % | | 64 % |
| In a run of 4+ cycles identical to another take | 10 % | 0.1 % | | 16 % |
| Of those runs, share with 2+ different figures | 11 % | | | 5 % |

GNN against Markov, per take, "closer to the real soloist": distinct-figure rate 14/15 (p = 0.0001), strike profile 5/15 (p = 0.055), timing 3/15 (p = 0.004), intensity 2/15 (p = 0.004). Figures reused from other takes, by soloist (share of copied figures): GNN 35 / 32 / 18 / 15 %, real soloists 47 / 26 / 16 / 11 % (the most present soloist plays 7 of the 15 takes).

![Generated and real soloists](figures/j5b_solo.png)

Sources: [`j5b_solo.json`](../results/j5b_solo.json), [`j5b_solo_creativity.json`](../results/j5b_solo_creativity.json) ([`solo_creativity.py`](../experiments/generation/solo_creativity.py)).

## Graph model against transformer, seen and unseen rhythms (J6)

Soloist stroke by stroke (strike, then offset and intensity given it strikes), 15 takes x 3 seeds, mean of seed means. Protocols: `take` (leave-one-take-out, piece given), `take_nopiece` (same, no piece), `piece` (leave-one-piece-out, no piece). Files: `results/j6_{take,take_nopiece,piece}.json`.

| Prediction (nats) | take: hit / offset / intensity | take_nopiece | piece |
|---|---|---|---|
| GNN_seq (56 k) | 0.407 / −0.945 / 1.220 | 0.407 / −0.954 / 1.215 | 0.436 / −0.565 / 1.233 |
| Transformer_matched (60 k) | 0.423 / −0.844 / 1.243 | 0.427 / −0.833 / 1.245 | 0.548 / 0.411 / 1.340 |
| Transformer (181 k) | 0.436 / −0.704 / 1.355 | 0.446 / −0.702 / 1.357 | 0.590 / 1.055 / 1.538 |
| B2 Markov (stroke memory) | 0.537 / −0.749 / 1.393 | same | 0.631 / 0.045 / 1.429 |
| B1 per position | 0.597 / −0.573 / 1.396 | same | 0.645 / −0.281 / 1.432 |

GNN_seq better than Transformer_matched (takes of 15, Wilcoxon p): take 12 (0.015) / 14 (0.0003) / 10 (0.15); piece 15 / 15 / 15 (0.0001 each).

| Generation (take protocol) | GNN_seq | Transformer_matched | Transformer | Markov | real |
|---|---|---|---|---|---|
| timing_lag1 | 0.44 | 0.56 | 0.59 | 0.54 | 0.56 |
| distinct figures | 0.69 | 0.69 | 0.63 | 0.77 | 0.36 |
| timing W1 (ms) | 7.9 | 7.4 | 7.7 | 6.3 | |

## Live steps and figure memory (L1, L2, J7, J8)

- L1, grid tracking: a Kalman grid tracker on a streamed jembe-2 track predicts the next stroke 100 ms ahead within 8.0 ms median on the 9 of 12 takes it tracks, against 18.2 ms for an isochronous grid ([`live_tracking.json`](../results/live_tracking.json)).
- L2, causal soloist: hearing only jembe 2, up to the previous pulse, it beats B2 on 12 of 12 takes and stays close to the offline model (0.435 / -0.955 / 1.238 against 0.429 / -0.969 / 1.223, [`live_take.json`](../results/live_take.json), leave-one-piece-out in [`live_piece.json`](../results/live_piece.json)).
- J7, figure memory and groove encoder: a memory of the soloist's own figures does not improve prediction; the base model is better on 12 of 15 takes for strikes and offsets ([`j7.json`](../results/j7.json)).
- J8, two-stage soloist (a figure planner picks each cycle's base among the figures already played, then the graph network plays the cycle): job written ([`slurm/j8_array.sbatch`](../slurm/j8_array.sbatch)), no results yet.
- Not done: real-time engine, calibration on a live player, blind listening test.

## Reproducing the README figures

```bash
python experiments/j6/plot.py --out docs/figures/j6_gnn_transformer.png
python experiments/gnn/plot_humanize.py results/j3b_humanize_3seeds.json docs/figures/j3b_humanize_3seeds.png
python experiments/gnn/plot_curve.py results/j4_curve.json docs/figures/j4_curve.png
python experiments/recognition/make_figures.py   # hero and step-0 figures (needs the corpus)
```

The overview figure is hand-written SVG ([`figures/overview.svg`](figures/overview.svg)) with crops of these figures (`figures/ov_*.png`).

## Commands

```bash
pip install -e ".[dev,train,audio]"   # train: torch, torch-geometric, wandb; audio: soundfile, for rendering with the samples
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
sbatch -A <account> --array=0-44 --time=00:30:00 slurm/j3b_array.sbatch   # on the cluster: V0 and V5, 15 takes x 3 seeds, about 5 min per task (slurm/README.md)
python experiments/gnn/merge_results.py <run dir>/part_*.json --out results/j3b_humanize_3seeds.json --expect-takes 15 --expect-seeds 3
python experiments/gnn/plot_humanize.py results/j3b_humanize_3seeds.json docs/figures/j3b_humanize_3seeds.png
sbatch -A <account> --array=0-179 slurm/curve_array.sbatch   # on the cluster: data-efficiency curve, V0 V1 V3 V5 on 2, 4, 8, 13 takes x 3 repetitions
python experiments/gnn/merge_results.py <run dir>/part_*.json --out results/j4_curve.json --expect-takes 15 --expect-seeds 3
python experiments/gnn/plot_curve.py results/j4_curve.json docs/figures/j4_curve.png
DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A <account> --array=0-44 slurm/gen_array.sbatch   # J5 generation, see slurm/README.md
python experiments/generation/generate_eval.py --takes Woloso_5 --seeds 0 --epochs 2   # quick local check, about a minute
DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A <account> --array=0-44 slurm/solo_array.sbatch   # J5b soloist over the real ensemble
python experiments/generation/solo_eval.py --takes Woloso_5 --seeds 0 --epochs 2   # quick local check
python experiments/generation/summarise_solo.py <run dir>/part_*.json --out results/j5b_solo.json --expect-takes 15 --expect-seeds 3
python experiments/generation/solo_creativity.py <run dir>/grids --out results/j5b_solo_creativity.json --figure docs/figures/j5b_solo.png
PROTOCOL=piece DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A <account> --array=0-8 slurm/j6_array.sbatch   # J6 (take and take_nopiece: --array=0-44)
python experiments/j6/run.py --protocol take --index 0 --epochs 2   # quick local check
python experiments/j6/summarise.py <run dir>/part_*.json --out results/j6_piece.json --expect-seeds 3
python experiments/j6/plot.py --out docs/figures/j6_gnn_transformer.png
python -m djembe.strokes --instrument J1 && python experiments/strokes/type_eval.py   # stroke types (needs the corpus audio)
```
