# Djembe: a generative model of Malian jembe micro-timing

I model how Malian jembe ensembles play: which strokes, exactly when, how hard, and in response to the other drummers. The data is the IEMP Malian Jembe corpus (Polak, Tarsitani & Clayton, 2018): 15 multitrack takes of three pieces (Manjanin, Maraka, Woloso), about 51 minutes and 41,072 annotated strokes.

The question: with this little data, does building the symmetries of the 12-pulse cycle into a graph neural network work better than pretraining on a large drum corpus?

Status (October 2026): the dataset and recognition results are done; the generative model is in progress. On its first task, predicting how late or early each stroke falls, the graph model beats the best reference model on 13 of 15 held-out takes (one seed, [details below](#three-fixes-seed-0-laptop-three-seeds-on-colab-pending); three seeds pending). The earlier analysis stage (notebooks, listening demo) is at tag [`v0.2-analysis`](https://github.com/clementdumeril/Djembe/tree/v0.2-analysis).

![Dundun 1 pattern and micro-timing in Manjanin, Maraka and Woloso](docs/figures/hero_d1_pattern_timing.png)

## Recognition

| Task (leave-one-take-out) | Balanced accuracy |
|---|---|
| Piece from the whole-take dundun pattern | 15/15 takes |
| Piece from jembe timing only, pattern masked, 4 cycles | 0.86 (95 % CI 0.74 to 0.95) |
| Same, tempo regressed out of every feature | 0.78 (95 % CI 0.64 to 0.90) |

The dundun result is a ceiling: a nearest-centroid rule already separates the three patterns. The intervals rest on 4 to 6 takes per piece, so they are probably too narrow.

## Humanizing a cycle

The graph model sees 4 cycles: 3 fully known, and a last one where it only knows which drums strike where. It predicts each stroke's offset and intensity as Gaussians, scored by negative log-likelihood on held-out takes (nats, lower is better). References: B1, a Gaussian per piece, drum and pulse; B2, which adds Markov memory of earlier cycles.

Over three seeds on Colab, the first network only ties B2 (−0.709 both, better on 8 of 15 takes, p = 0.60). So my one-seed claim that it beat B2 on 12 of 15 takes does not hold. Symmetry did not help at this data size, even though the jembe feel is close to beat-symmetric in the data (the four beats differ by 1 to 2 ms, against a feel of 3 to 20 ms).

### Three fixes (seed 0, laptop; three seeds on Colab pending)

1. Keep the last epoch instead of the best epoch on one validation take, which caused most of the seed spread (`--select last`).
2. Helix edge (V5): pulse 12 now feeds pulse 1 of the next cycle, not of the same cycle.
3. Correction to B1 (V4 = V5 + this): the network predicts a correction to B1's Gaussian.

| Offsets, seed 0, laptop | Score | vs B2, per take |
|---|---|---|
| B2 Markov, earlier cycles | −0.709 | |
| GNN V0, best validation epoch (J3) | −0.758 | |
| GNN V0, last epoch | −0.784 | better on 13 of 15 (p = 0.001) |
| GNN V5, last epoch + helix | −0.801 | better on 13 of 15 (p = 0.001) |
| GNN V4, V5 + correction to B1 | −0.652 | better on 11 of 15 |

The helix helps a little: −0.021 against V0, better on 11 of 15 takes (p = 0.02). The correction to B1 hurts (worse than V5 on 14 of 15 takes), so I dropped it. On intensity every network is far ahead (about 0.91 against 1.29), but that may be microphone bleed. A bleed control comes next, then the data-efficiency curve.

## Limitations

- 15 takes, 3 pieces, one dundun player, same sessions. Results hold within this corpus only.
- The fixes above are one seed. Runs are not bit-exact either: rerunning J3 on Colab with the same code and seeds gave V0 −0.736 instead of −0.709.

Corpus erratum: one published OSF file holds the wrong take; I use the correct ones ([details](docs/CORPUS_ERRATUM.md)).

## Install and run

```bash
pip install -e ".[dev]"
python -m djembe.dataset download     # annotations only, about 10 MB from OSF
python -m djembe.dataset build        # output/events.csv, about 7 s
python -m pytest -q                   # a few seconds
python experiments/gnn/train_humanize.py --variants V0 V4 V5 --seeds 0 --select last --out results/j3b_humanize_seed0_cpu.json   # about 2 h in 4 parallel parts (--takes)
```

Folds are frozen by take in [`splits.json`](splits.json); the tests fail if a take is on both sides. Full tables and commands: [`docs/RESULTS.md`](docs/RESULTS.md).

[`colab/j3_humanize.ipynb`](colab/j3_humanize.ipynb) runs the three-seed version. If OSF refuses the download (403), run `python -m djembe.dataset pack` and put `output/djembe_corpus.zip` at the root of your Google Drive; the notebook then skips OSF. Keep that file private.

## License and credits

Code: [MIT License](LICENSE). The IEMP Malian Jembe corpus is licensed CC BY-NC-ND 4.0 by its authors and is not redistributed here; `djembe/dataset.py` downloads it from OSF (https://osf.io/m652x/). The event table, audio and trained weights stay out of the repository.

- Corpus: Polak, R., Tarsitani, S., & Clayton, M. (2018). *IEMP Malian Jembe* [Data set]. OSF. https://doi.org/10.17605/OSF.IO/M652X. The recordings and annotations are theirs.
- Polak, R. (2004). *Festmusik als Arbeit, Trommeln als Beruf*. Dietrich Reimer. https://d-nb.info/971098840
- Clayton, M., et al. (2021). The Interpersonal Entrainment in Music Performance Data Collection. *Empirical Musicology Review, 16*(1), 65–84. https://doi.org/10.18061/emr.v16i1.7555
- Polak, R. (2010). Rhythmic feel as meter. *Music Theory Online, 16*(4). https://doi.org/10.30535/mto.16.4.4
- Polak, R., & London, J. (2014). Timing and meter in Mande drumming from Mali. *Music Theory Online, 20*(1). https://doi.org/10.30535/mto.20.1.1
- Polak, R., London, J., & Jacoby, N. (2016). Both isochronous and non-isochronous metrical subdivision afford precise and stable ensemble entrainment: A corpus study of Malian jembe drumming. *Frontiers in Neuroscience, 10*, 285. https://doi.org/10.3389/fnins.2016.00285
