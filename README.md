# Djembe: a generative model of Malian jembe micro-timing

This repository models how Malian jembe ensembles *play* their rhythms: which strokes, at which exact moment, how hard, cycle after cycle, and in response to the other drummers. The data is the IEMP Malian Jembe corpus (Polak, Tarsitani & Clayton, 2018): 15 multitrack takes of three pieces (Manjanin, Maraka, Woloso), about 51 minutes and 41,072 annotated strokes.

The question I am working on: with this little data, does building the symmetries of the metric cycle into the model work better than pretraining on a large drum corpus? Tempo is a time dilation, and these pieces repeat a 12-pulse cycle of four beats of three pulses. The plan is a graph neural network on the cycle grid, with the symmetry kept or broken on purpose, and its departure from symmetry measured.

**Status (October 2026).** The dataset, the leakage-safe splits and the recognition results below are done; the generative model is in progress. The earlier analysis stage (notebooks, the *Boussole des rythmes* interactive page, Colab listening demo, a Markov generator) is preserved at tag [`v0.2-analysis`](https://github.com/clementdumeril/Djembe/tree/v0.2-analysis).

## Results

![Dundun 1 pattern and micro-timing in Manjanin, Maraka and Woloso](docs/figures/hero_d1_pattern_timing.png)

| What | Result (balanced accuracy, leave-one-take-out) | Source |
|---|---|---|
| Piece from the whole-take dundun pattern | 15/15 takes for random forest and nearest centroid (exact 95 % CI 0.78 to 1.00) | [`results/step0_rf.json`](results/step0_rf.json) |
| Piece from one cycle of dundun pattern | 0.99 (logistic regression) | [`results/a2_recognition.json`](results/a2_recognition.json) |
| Piece from jembe timing only, pattern masked, 4 cycles | 0.86 (95 % CI 0.74 to 0.95) | [`results/a2_recognition.json`](results/a2_recognition.json) |
| Same, restricted to the tempo range shared by all pieces | 0.87; tempo alone drops to 0.29 (logistic regression) and 0.44 (nearest centroid) | [`results/a2_tempo_control.json`](results/a2_tempo_control.json) |
| Same, tempo regressed out of every feature (main tempo control) | 0.78 (95 % CI 0.64 to 0.90) | [`results/a2_tempo_control.json`](results/a2_tempo_control.json) |
| 1D CNN on jembe stroke sequences (which pulses and their timing), 1 / 8 cycles, mean of 3 seeds | 0.78 / 0.74, against 0.77 / 0.89 for logistic regression on the jembe feel | [`results/a3_sequence_cnn_cpu.json`](results/a3_sequence_cnn_cpu.json) |

The whole-take result is a ceiling: the three dundun patterns differ so much that a nearest-centroid rule already separates them. The more informative result is the jembe timing. With the pattern masked, the timing of the jembe strokes on the three pulses of each beat still identifies the piece, with a balanced accuracy of 0.77 (1 cycle) to 0.89 (8 cycles). The main control regresses tempo out of every feature inside each training fold: the feel then keeps 0.70, 0.75 and 0.78 for 1, 2 and 4 cycles, about 7 to 9 points lower. In the tempo range shared by the three pieces, the feel keeps 0.75 to 0.87, while tempo alone drops to 0.29–0.35 with logistic regression and 0.44–0.48 with nearest centroid, so some tempo information remains there.

These intervals rest on 4 to 6 takes per piece and resample fixed predictions without retraining, so they are probably too narrow.

![Balanced accuracy against window length per feature set](docs/figures/a2_window_accuracy.png)

The jembe *pattern* (which pulses are played) is at or below chance for short windows. My working hypothesis: takes differ more by who plays which part than pieces do, so the classifier learns take-specific habits that do not transfer to the held-out take. The worst take for the timing features is Manjanin_1 (about 25 % of its windows correct); I have not found out why yet.

## Symmetry check: do the four beats share one feel?

Before any model, the data has to say which symmetry is worth building in. A 12-pulse cycle is four beats of three pulses. If the feel were the same on every beat (the cycle unchanged when rotated by one beat, the group C4), the mean offset at a pulse would depend only on its position in the beat. [`experiments/symmetry/beat_symmetry.py`](experiments/symmetry/beat_symmetry.py) measures two things per piece and drum, in milliseconds:
- **feel**: the size of the short-medium-long pattern (spread of the beat-averaged profile over the three pulses);
- **breach**: how much the four beats differ at the same position.

Breaks, strokes more than half a pulse off and positions with fewer than 30 strokes per beat are left out.

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

**What this says:**
- **The jembe feel is close to beat-symmetric.** The four beats differ by 1 to 2 ms, against a pattern of 3 to 20 ms. The exception is Maraka's second jembe, whose feel is itself almost even, as Polak, London & Jacoby (2016) found for Maraka.
- **The breach is not zero.** Rotating every cycle by a random number of beats (which makes C4 exact) gives a noise floor of 0.5 to 1.0 ms. For every pair except Woloso's first jembe, the bootstrap interval over takes lies above that floor.
- **The dundun breaks the symmetry more.** For example, Woloso's dundun plays the last pulse of beat 4 about 16 ms earlier than the same pulse on beats 1 to 3, just before the next cycle starts.
- **Consequence for the model:**
  - beat symmetry is a good prior but not an exact law, so it should be built in with a learnable, penalised way to break it, not imposed strictly;
  - full rotation by one pulse (C12) is clearly wrong, because the feel is several times larger than the breach.
- **Caveat:** differences of 1 to 2 ms are close to the precision of onset annotation, and they rest on 3 to 6 takes per piece.

Numbers: [`results/j1_beat_symmetry.json`](results/j1_beat_symmetry.json) (in fractions of a pulse, with intervals and noise floors).

## What is in the repository

- [`djembe/dataset.py`](djembe/dataset.py): downloads the annotations from OSF and builds one event table. Each row is one stroke: take, instrument, player, cycle, subdivision, onset, peak amplitude, offset from the even grid (fraction of a pulse and ms), musical section. Every experiment reads only this table.
- [`splits.json`](splits.json), [`tests/`](tests/): frozen leave-one-take-out folds and tests that fail if a take ends up on both sides of a split, if a window spans two takes, if a file's take label disagrees with its name, or if any audio is committed. CI runs them on every push.
- [`djembe/grid.py`](djembe/grid.py): the cycle grid of a take (cycle × 12 pulses × 4 drums × hit, intensity, offset, with tempo as a separate log cycle duration) and the performance graph. It has three edge types: next pulse (a ring of 12), same pulse on another drum, and same pulse one cycle earlier. Tests check that rotating the cycle maps every edge type onto itself, and that each choice of positional input keeps exactly the intended rotations: C12, C4 or none.
- [`djembe/features.py`](djembe/features.py), [`experiments/recognition/`](experiments/recognition/): windows of 1 to 8 cycles, five feature sets (tempo, dundun pattern, jembe pattern, jembe timing per pulse, jembe feel per position in the beat), the classifiers, the two tempo controls, and a small CNN and transformer on stroke sequences.

**Data notes.** The second dundun (D2) is recorded on 4 takes only (Manjanin 1 and 4, Maraka 6, Woloso 2), and the second jembe is missing on some takes; missing parts are masked, never imputed. Each drum has its own close microphone, so every track also hears the others. The onsets come from the corpus authors' detection and selection, but the peak amplitudes are measured on tracks with that bleed, so intensity is the least reliable variable.

## How it works

The corpus annotates every metric position of every cycle: 12 pulses per row of the metre file. Each stroke is placed relative to its cycle:

```
phase = (onset - cycle_start) / cycle_duration
offset = phase - (pulse - 1) / 12          (fraction of a cycle; x12 for pulses, x duration for ms)
```

An offset of zero means an isochronous grid. Polak's work showed that this music is not isochronous: the three pulses of a beat are not equally long. The "feel" features summarise that directly. For each position in the beat (1st, 2nd, 3rd pulse), they take the mean and spread of the jembe offsets over the four beats of the window, so which pulses are played does not enter.

Two details of the table: the last cycle of each take has no end time, so its 43 strokes have no offset; and 1,422 strokes (3.5 %) sit more than half a pulse from their nominal position, which I keep as annotated. Woloso's metre file labels half-cycles (`1.1`, `1.2`) whose dundun figures differ, which is why its averaged pattern shows probabilities around 0.5.

## How I checked it

- **Data audit.** The OSF file `MJ_Maraka_5_Onsets_Selected.csv` actually contains take Woloso_5: its own `Take` column says so, and its dundun onsets match the raw Woloso_5 onsets (98.6 % within 10 ms) but not the Maraka_5 ones (9.9 %, about chance). Woloso_5 has no file of its own. My earlier analyses used these files and reported 12/14 for the take-level classifier; both errors trace back to this mix-up. The event table reads the `_Onsets_Selected_Peak` files, which are correct for all 15 takes. Details and a reproducible check: [`docs/CORPUS_ERRATUM.md`](docs/CORPUS_ERRATUM.md).
- **Leakage.** The split unit is the take, never the stroke, cycle or window. The same dundun player (Madu Jakite) plays on every take, so a split by musician is impossible. An early experiment (tag `v0.2-analysis`) used stratified folds over strokes, which mixes strokes of one take between train and test; `iter_folds` now refuses any split that shares a take.
- **Reference points and intervals.** Every score sits next to tempo alone, a majority-class model and a nearest-centroid rule. Intervals come from a bootstrap over takes, stratified by piece, because windows of one take are not independent. For whole-take scores I use an exact binomial interval: over 15 takes, a bootstrap collapses to [1, 1] at perfect accuracy. The majority-class model is a poor reference under this split: at window level it scores 0.00 to 0.06 balanced accuracy, because removing one take usually flips the training majority away from the held-out piece; at take level it scores 6/15 (balanced 0.33).
- **No tuning on test takes.** All models use fixed settings chosen before running. When hyperparameters start to be tuned, a validation take will be held out inside each training fold.

## Limitations

- 15 takes, 3 pieces, one dundun player, four jembe players in total, all recorded in the same sessions. Results hold within this corpus. They say nothing about other musicians or other rhythms.
- The CNN is below the feature baselines, most clearly on long windows, where there are 8 times fewer training windows for the same fixed 40 epochs. Its settings have not been tuned, and the transformer has not been evaluated.
- The corpus licence (CC BY-NC-ND 4.0) forbids redistributing derived data. The event table, audio and any trained weights stay out of the repository. Only code, fold definitions, aggregate metrics and figures are versioned.

## Reproduce

```bash
pip install -e ".[dev]"
python -m djembe.dataset download     # annotations only, about 10 MB from OSF
python -m djembe.dataset build        # output/events.csv, about 7 s
python -m djembe.dataset splits       # regenerates splits.json (deterministic)
python -m pytest -q                   # a few seconds
python experiments/recognition/step0_rf_baseline.py      # results/step0_rf.json, about 20 s
python experiments/recognition/recognition_baselines.py  # results/a2_recognition.json and figure, several minutes
python experiments/recognition/tempo_control.py          # results/a2_tempo_control.json, a few minutes
python experiments/recognition/sequence_models.py --models cnn --out results/a3_sequence_cnn_cpu.json   # about 20 min on 2 threads
python experiments/recognition/make_figures.py           # docs/figures/
python experiments/symmetry/beat_symmetry.py             # results/j1_beat_symmetry.json and figure, a few minutes
```

## Credits

- Corpus: Polak, R., Tarsitani, S., & Clayton, M. (2018). *IEMP Malian Jembe* [Data set]. OSF. https://doi.org/10.17605/OSF.IO/M652X. Licence CC BY-NC-ND 4.0. Recordings, annotated onsets, metre and form annotations are theirs.
- Founding thesis: Polak, R. (2004). *Festmusik als Arbeit, Trommeln als Beruf: Jenbe-Spieler in einer westafrikanischen Großstadt*. Berlin: Dietrich Reimer. ISBN 978-3-496-02771-3. Doctoral thesis, Universität Bayreuth, 2002. https://d-nb.info/971098840
- Collection paper: Clayton, M., et al. (2021). The Interpersonal Entrainment in Music Performance Data Collection. *Empirical Musicology Review, 16*(1), 65–84. https://doi.org/10.18061/emr.v16i1.7555
- Micro-timing findings this work builds on: Polak, R. (2010). Rhythmic feel as meter. *Music Theory Online, 16*(4), https://doi.org/10.30535/mto.16.4.4. Polak, R., & London, J. (2014). Timing and meter in Mande drumming from Mali. *Music Theory Online, 20*(1), https://doi.org/10.30535/mto.20.1.1. Polak, R., London, J., & Jacoby, N. (2016). Both isochronous and non-isochronous metrical subdivision afford precise and stable ensemble entrainment: A corpus study of Malian jembe drumming. *Frontiers in Neuroscience, 10*, 285, https://doi.org/10.3389/fnins.2016.00285
- Tools: pandas, scikit-learn, PyTorch, matplotlib.

## License

The code in this repository is under the [MIT License](LICENSE). The IEMP Malian Jembe corpus is not part of it: it is licensed CC BY-NC-ND 4.0 by its authors, is not redistributed here, and is downloaded from OSF (https://osf.io/m652x/) by `djembe/dataset.py`.

Clément Dumeril · https://clement.dumeril.net · https://github.com/clementdumeril
