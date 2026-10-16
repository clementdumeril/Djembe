# Djembe: a generative model of Malian jembe micro-timing

This repository models how Malian jembe ensembles *play* their rhythms: which strokes, at which exact moment, how hard, cycle after cycle, and in response to the other drummers. The data is the IEMP Malian Jembe corpus (Polak, Tarsitani & Clayton, 2018): 15 multitrack takes of three pieces (Manjanin, Maraka, Woloso), about 51 minutes and 41,072 annotated strokes.

The question I'm working on: with this little data, does building the symmetries of the metric cycle into the model work better than pretraining on a large drum corpus? Tempo is a time dilation, and these pieces repeat a 12-pulse cycle made of four beats of three pulses. My plan is a graph neural network on the cycle grid, with the symmetry kept or broken on purpose, and a measurement of how far the trained model departs from it.

Status (October 2026): the dataset, the leakage-safe splits and the recognition results below are done. The generative model is in progress. Its first task, predicting how late or early each stroke of a cycle falls, now works: the graph model beats the best reference model on 13 of 15 held-out takes (one seed, [details below](#three-fixes-seed-0-laptop-three-seeds-on-colab-pending); three seeds pending). The earlier analysis stage (notebooks, the *Boussole des rythmes* interactive page, a Colab listening demo, a Markov generator) is kept at tag [`v0.2-analysis`](https://github.com/clementdumeril/Djembe/tree/v0.2-analysis).

## Recognition results

![Dundun 1 pattern and micro-timing in Manjanin, Maraka and Woloso](docs/figures/hero_d1_pattern_timing.png)

| What | Result (balanced accuracy, leave-one-take-out) | Source |
|---|---|---|
| Piece from the whole-take dundun pattern | 15/15 takes for random forest and nearest centroid (exact 95 % CI 0.78 to 1.00) | [`results/step0_rf.json`](results/step0_rf.json) |
| Piece from one cycle of dundun pattern | 0.99 (logistic regression) | [`results/a2_recognition.json`](results/a2_recognition.json) |
| Piece from jembe timing only, pattern masked, 4 cycles | 0.86 (95 % CI 0.74 to 0.95) | [`results/a2_recognition.json`](results/a2_recognition.json) |
| Same, restricted to the tempo range shared by all pieces | 0.87; tempo alone drops to 0.29 (logistic regression) and 0.44 (nearest centroid) | [`results/a2_tempo_control.json`](results/a2_tempo_control.json) |
| Same, tempo regressed out of every feature (main tempo control) | 0.78 (95 % CI 0.64 to 0.90) | [`results/a2_tempo_control.json`](results/a2_tempo_control.json) |
| 1D CNN on jembe stroke sequences (which pulses and their timing), 1 / 8 cycles, mean of 3 seeds | 0.78 / 0.74, against 0.77 / 0.89 for logistic regression on the jembe feel | [`results/a3_sequence_cnn_cpu.json`](results/a3_sequence_cnn_cpu.json) |

The whole-take result is a ceiling: the three dundun patterns are so different that a nearest-centroid rule already separates them. The jembe timing is more interesting. With the pattern masked, the timing of the jembe strokes on the three pulses of each beat still identifies the piece, with a balanced accuracy of 0.77 (1 cycle) to 0.89 (8 cycles).

The main tempo control regresses tempo out of every feature inside each training fold. The feel then keeps 0.70, 0.75 and 0.78 for 1, 2 and 4 cycles, about 7 to 9 points lower. In the tempo range shared by the three pieces, the feel keeps 0.75 to 0.87, while tempo alone drops to 0.29–0.35 with logistic regression and 0.44–0.48 with nearest centroid, so some tempo information remains there.

The intervals rest on 4 to 6 takes per piece and resample fixed predictions without retraining, so they are probably too narrow.

![Balanced accuracy against window length per feature set](docs/figures/a2_window_accuracy.png)

The jembe *pattern* (which pulses are played) is at or below chance for short windows. My working hypothesis is that takes differ more by who plays which part than pieces do, so the classifier learns take-specific habits that don't carry over to the held-out take. The worst take for the timing features is Manjanin_1 (about 25 % of its windows correct), and I haven't found out why yet.

## Symmetry check: do the four beats share one feel?

Before building a model, I wanted the data to tell me which symmetry is worth building in. If the feel were the same on every beat (the cycle unchanged when rotated by one beat, the group C4), the mean offset at a pulse would depend only on its position in the beat. [`experiments/symmetry/beat_symmetry.py`](experiments/symmetry/beat_symmetry.py) measures two things per piece and drum, in milliseconds: the *feel*, the size of the short-medium-long pattern (spread of the beat-averaged profile over the three pulses), and the *breach*, how much the four beats differ at the same position. Breaks, strokes more than half a pulse off, and positions with fewer than 30 strokes per beat are left out.

![Mean offset at each pulse of each beat](docs/figures/j1_beat_symmetry.png)

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

The jembe feel is close to beat-symmetric: the four beats differ by 1 to 2 ms, against a pattern of 3 to 20 ms. The exception is Maraka's second jembe, whose feel is itself almost even, as Polak, London & Jacoby (2016) found for Maraka.

The breach is not zero, though. Rotating every cycle by a random number of beats (which makes C4 exact) gives a noise floor of 0.5 to 1.0 ms, and for every pair except Woloso's first jembe, the bootstrap interval over takes lies above that floor. The dundun breaks the symmetry more. Woloso's dundun, for example, plays the last pulse of beat 4 about 16 ms earlier than the same pulse on beats 1 to 3, just before the next cycle starts.

So beat symmetry is a good prior but not an exact law, and the model should get a learnable, penalized way to break it rather than a strict constraint. Full rotation by one pulse (C12) is clearly wrong, because the feel is several times larger than the breach. One caveat: differences of 1 to 2 ms are close to the precision of onset annotation, and they rest on 3 to 6 takes per piece.

Numbers (in fractions of a pulse, with intervals and noise floors): [`results/j1_beat_symmetry.json`](results/j1_beat_symmetry.json).

## Reference models

[`djembe/baselines.py`](djembe/baselines.py) and [`experiments/baselines/run_baselines.py`](experiments/baselines/run_baselines.py) score two reference models on each held-out take (leave-one-take-out). For every pulse of every recorded drum, each model answers three questions: does it strike, with which offset, and how hard. The score is the negative log-likelihood (nats, lower is better), which is also what the graph model will be judged on.

- B1 (position) turns Polak's description of the feel into a model. Per piece, drum and pulse, it has a strike rate, plus a Gaussian for the offset and one for the intensity.
- B2 (Markov) adds memory along each drum's line. The strike depends on the two previous pulses (an order-2 chain, shrunk toward B1 when a context is rare). The offset and the intensity follow B1's mean plus a fraction of the previous stroke's residual.
- Global ignores the pulse altogether, to show what position alone is worth.

The table also scores the *distribution* of offsets each model draws at the take's real strokes. Per drum and pulse, it reports histogram intersection and error of the mean, as Mishra et al. (2025) did for candombe.

| Model (jembe 1, jembe 2, dundun 1) | Strikes | Offsets | Intensity | Histogram intersection | Error of the mean |
|---|---|---|---|---|---|
| Global | 0.655 | −0.367 | 1.417 | 0.61 | 11.6 ms |
| B1 position | 0.311 | −0.577 | 1.300 | 0.73 | 4.6 ms |
| B2 Markov | **0.256** | **−0.900** | **1.206** | 0.73 | 5.0 ms |

![Negative log-likelihood per held-out take](docs/figures/j2_baselines.png)

Position halves the uncertainty about strikes, and memory improves every likelihood: B2 beats B1 on 14 of 15 takes for strikes, 15/15 for offsets and 13/15 for intensity. The drawn distributions are as good with or without memory, though. A Gaussian per pulse already reproduces the per-pulse histograms to within about 5 ms. What memory adds is predicting the *next* stroke from the last ones, that is, the drift and persistence of a player's timing.

The graph model therefore has to lower B2's likelihoods, which needs the interplay between drums that neither baseline sees, without losing B1's distributions.

This step also turned up two problems in the data. First, takes do not share one timing reference: the median offset of a whole take ranges from −27 to +17 ms. Manjanin 1 sits about 30 ms earlier than the other Manjanin takes, on every drum, which points at where its cycle starts were annotated rather than at the playing. The grid therefore removes each take's median offset (one constant per take, stored in `delta_offset`). That alone brings the error of the mean from 8.4 to 4.6 ms; the unaligned scores are kept in the JSON. It also removes any genuine take-wide "ahead of the beat" playing, which this corpus can't separate from the annotation reference.

Second, the second dundun can't be learned per piece. It is recorded on 4 takes, and on only one take of Maraka and of Woloso, so I score it separately. There, B1 and B2 predict its strikes worse than the global model does.

Intensities are z-scored within each take, so a take's own loudness spread is used to score it (the graph model gets the same input).

## First graph model: humanizing a cycle

The network sees 4 consecutive cycles. The first 3 are fully known; for the last one, it only knows which drums strike where. It predicts the offset and the intensity of every stroke of that last cycle as Gaussians, and it is scored exactly like the reference models, on the same strokes. In each fold the network trains on 13 takes and stops early on a 14th, while B1 and B2 are fitted on all 14. B2 here only remembers earlier cycles, since no offset of the target cycle is known.

| Model (jembe 1, jembe 2, dundun 1) | Offsets, 3 seeds (Colab) | Offsets, seed 0 (laptop) | Intensity, 3 seeds |
|---|---|---|---|
| B1 position | −0.579 | −0.579 | 1.294 |
| B2 Markov, earlier cycles | **−0.709** | −0.709 | 1.286 |
| GNN V0, no symmetry | **−0.709** | −0.758 | 0.907 |
| GNN V1, beat symmetry (C4) | −0.674 | −0.670 | 0.904 |
| GNN V3, pulse rotation (C12) | −0.659 | −0.687 | 0.918 |

![Negative log-likelihood per held-out take, mean and spread over three seeds](docs/figures/j3_humanize.png)

On offsets, the network only ties the best reference. With one seed, the unconstrained network looked better than B2 on 12 of 15 takes. Over three seeds it is level with it (−0.709 both): its mean over seeds beats B2 on 8 of 15 takes, and the paired difference is +0.010 (Wilcoxon p = 0.60). It is clearly better on Maraka 1 and the Manjanin takes, and worse on Maraka 2 to 5. So the one-seed claim does not hold.

The weak point is the training recipe rather than the architecture. The spread between seeds is large, 0.064 nats on average and up to 0.195 on one take, and it comes mostly from early stopping on a single validation take. The 35 runs that stopped within 12 epochs are 0.175 nats worse than B2; the 100 that trained longer are 0.012 better. The correlation between epochs run and the gap to B2 is −0.68.

Symmetry doesn't help at this data size. Both symmetric variants are behind B2 (beat symmetry +0.041, p = 0.05; pulse rotation +0.058, p = 0.03) and behind V0. Whether symmetry pays off when data is scarce is the question the data-efficiency curve is meant to answer.

On intensity, every network variant is far ahead of the references (about 0.91 against 1.29), consistently across seeds. I don't count that as a musical result yet. A jembe stroke measures about half a standard deviation louder when two other drums strike on the same pulse. That fits microphone bleed as well as genuine accents on the pulses everyone plays, and the network sees who else strikes. A control that uses the small timing gaps between drums should separate the two.

Next: the three fixes below, the bleed control, then the data-efficiency curve.

### Three fixes (seed 0, laptop; three seeds on Colab pending)

1. Keep the last epoch. The learning rate follows a cosine schedule down to zero, so the last epochs are stable, while the best epoch on a single validation take is mostly chance: on two folds it was worse than the last epoch in 3 of 4 checks. All 40 epochs are now run and the final weights are scored (`--select last`).
2. Helix edge (V5). In J3, pulse 1 received its "previous pulse" message from pulse 12 of the *same* cycle, which comes later in time and is unknown in the target cycle. Now pulse 12 feeds pulse 1 of the *next* cycle.
3. Correction to B1 (V4 = V5 + this). Every node is given B1's Gaussian, and the network predicts a correction to it, zero before training.

| Offsets, seed 0, laptop | Score | vs B2, per take |
|---|---|---|
| B2 Markov, earlier cycles | −0.709 | |
| GNN V0, best validation epoch (J3) | −0.758 | |
| GNN V0, last epoch | −0.784 | better on 13 of 15 (p = 0.001) |
| **GNN V5, last epoch + helix** | **−0.801** | **better on 13 of 15 (p = 0.001)** |
| GNN V4, V5 + correction to B1 | −0.652 | better on 11 of 15 |

The helix helps a little and consistently: −0.021 against V0, better on 11 of 15 takes (p = 0.02). Keeping the last epoch is not proven by one seed: −0.028 against the J3 laptop run, better on 8 of 15 (p = 0.21). Its expected benefit is a smaller spread between seeds, which only the three-seed run can show.

The correction to B1 hurts. It is worse than V5 on 14 of 15 takes, and worse than B1 itself on Manjanin 4 (+0.25 against −0.42). Starting from B1 does not stop the network from becoming overconfident where the test take departs from B1's averages, so I've dropped it for now. Intensity is unchanged (V5 0.878, V0 0.890) and still waits on the bleed control.

The same J3 runs, repeated, also don't give the same score. I reran J3 on Colab with identical code, settings and seeds: V0 scored −0.736 instead of −0.709, and single runs moved by 0.046 nats on average, up to 0.48. GPU arithmetic is not bit-exact, and picking the best validation epoch turns those tiny differences into different models. That is one more reason to keep the last epoch. Runs: [`results/j3_rerun_wandb_export.csv`](results/j3_rerun_wandb_export.csv).

Numbers:
- [`results/j3b_humanize_seed0_cpu.json`](results/j3b_humanize_seed0_cpu.json): the three fixes, seed 0, on a laptop;
- [`results/j3_humanize_3seeds_colab.json`](results/j3_humanize_3seeds_colab.json): three seeds on Colab, rebuilt from the Weights & Biases export [`results/j3_wandb_export.csv`](results/j3_wandb_export.csv) with [`experiments/gnn/fetch_wandb.py`](experiments/gnn/fetch_wandb.py);
- [`results/j3_humanize_seed0_cpu.json`](results/j3_humanize_seed0_cpu.json): the first run, seed 0, on a laptop.

## What's in the repository

- [`djembe/dataset.py`](djembe/dataset.py) downloads the annotations from OSF and builds one event table. Each row is one stroke: take, instrument, player, cycle, subdivision, onset, peak amplitude, offset from the even grid (fraction of a pulse and ms), musical section. Every experiment reads only this table.
- [`splits.json`](splits.json) holds the frozen leave-one-take-out folds; the tests in [`tests/`](tests/) fail if a take ends up on both sides of a split, if a window spans two takes, if a file's take label disagrees with its name, or if any audio is committed. CI runs them on every push.
- [`djembe/grid.py`](djembe/grid.py) builds the cycle grid of a take (cycle × 12 pulses × 4 drums × hit, intensity, offset, with tempo as a separate log cycle duration) and the performance graph. The graph has three edge types: next pulse (a ring of 12), same pulse on another drum, and same pulse one cycle earlier. Tests check that rotating the cycle maps every edge type onto itself, and that each choice of positional input keeps exactly the intended rotations: C12, C4 or none.
- [`djembe/baselines.py`](djembe/baselines.py) has the B1 and B2 reference models, with synthetic tests that they recover the rates, the feel and the timing memory they are built for.
- [`djembe/gnn.py`](djembe/gnn.py) and [`experiments/gnn/train_humanize.py`](experiments/gnn/train_humanize.py) are the graph model: relational message passing over the three edge types in plain PyTorch, with Gaussian heads for offset and intensity, trained on windows of 4 cycles whose last cycle is to be "humanized". A test checks numerically that its output rotates with the cycle exactly for the rotations its positional input allows. [`colab/j3_humanize.ipynb`](colab/j3_humanize.ipynb) runs it on Colab and logs to Weights & Biases (the key is read from Colab's Secrets panel).
- [`djembe/features.py`](djembe/features.py) and [`experiments/recognition/`](experiments/recognition/) cover recognition: windows of 1 to 8 cycles, five feature sets (tempo, dundun pattern, jembe pattern, jembe timing per pulse, jembe feel per position in the beat), the classifiers, the two tempo controls, and a small CNN and transformer on stroke sequences.

Data notes: the second dundun (D2) is recorded on 4 takes only (Manjanin 1 and 4, Maraka 6, Woloso 2), and the second jembe is missing on some takes; missing parts are masked, never imputed. Each drum has its own close microphone, so every track also hears the others. The onsets come from the corpus authors' detection and selection, but the peak amplitudes are measured on tracks with that bleed, so intensity is the least reliable variable.

## How it works

The corpus annotates every metric position of every cycle, 12 pulses per row of the meter file. Each stroke is placed relative to its cycle:

```
phase = (onset - cycle_start) / cycle_duration
offset = phase - (pulse - 1) / 12          (fraction of a cycle; x12 for pulses, x duration for ms)
```

An offset of zero means an isochronous grid. Polak's work showed that this music is not isochronous: the three pulses of a beat are not equally long. The "feel" features summarize that directly. For each position in the beat (1st, 2nd, 3rd pulse), they take the mean and spread of the jembe offsets over the four beats of the window, so which pulses are played does not enter.

The last cycle of each take has no end time, so its 43 strokes have no offset, and 1,422 strokes (3.5 %) sit more than half a pulse from their nominal position, which I keep as annotated. Woloso's meter file labels half-cycles (`1.1`, `1.2`) whose dundun figures differ, which is why its averaged pattern shows probabilities around 0.5.

## How I checked it

Data audit: the OSF file `MJ_Maraka_5_Onsets_Selected.csv` actually contains take Woloso_5. Its own `Take` column says so, and its dundun onsets match the raw Woloso_5 onsets (98.6 % within 10 ms) but not the Maraka_5 ones (9.9 %, about chance). Woloso_5 has no file of its own. My earlier analyses used these files and reported 12/14 for the take-level classifier; both errors trace back to this mix-up. The event table reads the `_Onsets_Selected_Peak` files, which are correct for all 15 takes. Details and a reproducible check are in [`docs/CORPUS_ERRATUM.md`](docs/CORPUS_ERRATUM.md).

Leakage: the split unit is the take, never the stroke, cycle or window. The same dundun player (Madu Jakite) plays on every take, so a split by musician is impossible. An early experiment (tag `v0.2-analysis`) used stratified folds over strokes, which mixes strokes of one take between train and test. `iter_folds` now refuses any split that shares a take.

Reference points and intervals: every score sits next to tempo alone, a majority-class model and a nearest-centroid rule. Intervals come from a bootstrap over takes, stratified by piece, because windows of one take are not independent. For whole-take scores I use an exact binomial interval, since over 15 takes a bootstrap collapses to [1, 1] at perfect accuracy. The majority-class model is a poor reference under this split. At window level it scores 0.00 to 0.06 balanced accuracy, because removing one take usually flips the training majority away from the held-out piece; at take level it scores 6/15 (balanced 0.33).

No tuning on test takes: all models use fixed settings chosen before running. When I start tuning hyperparameters, a validation take will be held out inside each training fold.

## Limitations

- 15 takes, 3 pieces, one dundun player, four jembe players in total, all recorded in the same sessions. Results hold within this corpus and say nothing about other musicians or other rhythms.
- The CNN is below the feature baselines, most clearly on long windows, where there are 8 times fewer training windows for the same fixed 40 epochs. Its settings have not been tuned, and the transformer has not been evaluated.
- The corpus license (CC BY-NC-ND 4.0) forbids redistributing derived data. The event table, audio and any trained weights stay out of the repository. Only code, fold definitions, aggregate metrics and figures are versioned.

## Reproduce

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

The Colab notebook [`colab/j3_humanize.ipynb`](colab/j3_humanize.ipynb) runs the three-seed version. OSF sometimes refuses downloads (403). If it does, upload the `djembe_corpus.zip` made by `pack` to the root of your own Google Drive, and the notebook will use it without contacting OSF. Keep that file private: the corpus license (CC BY-NC-ND) does not allow redistributing it.

## Credits

- Corpus: Polak, R., Tarsitani, S., & Clayton, M. (2018). *IEMP Malian Jembe* [Data set]. OSF. https://doi.org/10.17605/OSF.IO/M652X. License CC BY-NC-ND 4.0. The recordings, annotated onsets, and meter and form annotations are theirs.
- Founding thesis: Polak, R. (2004). *Festmusik als Arbeit, Trommeln als Beruf: Jenbe-Spieler in einer westafrikanischen Großstadt*. Berlin: Dietrich Reimer. ISBN 978-3-496-02771-3. Doctoral thesis, Universität Bayreuth, 2002. https://d-nb.info/971098840
- Collection paper: Clayton, M., et al. (2021). The Interpersonal Entrainment in Music Performance Data Collection. *Empirical Musicology Review, 16*(1), 65–84. https://doi.org/10.18061/emr.v16i1.7555
- Micro-timing findings this work builds on: Polak, R. (2010). Rhythmic feel as meter. *Music Theory Online, 16*(4), https://doi.org/10.30535/mto.16.4.4. Polak, R., & London, J. (2014). Timing and meter in Mande drumming from Mali. *Music Theory Online, 20*(1), https://doi.org/10.30535/mto.20.1.1. Polak, R., London, J., & Jacoby, N. (2016). Both isochronous and non-isochronous metrical subdivision afford precise and stable ensemble entrainment: A corpus study of Malian jembe drumming. *Frontiers in Neuroscience, 10*, 285, https://doi.org/10.3389/fnins.2016.00285
- Tools: pandas, scikit-learn, PyTorch, matplotlib.

## License

The code in this repository is under the [MIT License](LICENSE). The IEMP Malian Jembe corpus is not part of it: it is licensed CC BY-NC-ND 4.0 by its authors, is not redistributed here, and is downloaded from OSF (https://osf.io/m652x/) by `djembe/dataset.py`.

Clément Dumeril · https://clement.dumeril.net · https://github.com/clementdumeril
