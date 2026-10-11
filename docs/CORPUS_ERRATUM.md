# Corpus erratum: `MJ_Maraka_5_Onsets_Selected.csv` contains take Woloso_5

Corpus: IEMP Malian Jembe (Polak, Tarsitani & Clayton, 2018), OSF node `m652x`, folder `Annotations/MJ_Maraka/`. Checked on 2026-10-09 against the file served by the OSF API, not only against a local copy.

## What is wrong

| File on OSF | Expected content | Actual content |
|---|---|---|
| `Annotations/MJ_Maraka/MJ_Maraka_5_Onsets_Selected.csv` | take Maraka_5 | take Woloso_5 |
| `Annotations/MJ_Woloso/MJ_Woloso_5_Onsets_Selected.csv` | take Woloso_5 | file absent |

All other annotation files of both takes are consistent, including `MJ_Maraka_5_Onsets_Selected_Peak.csv` and `MJ_Woloso_5_Onsets_Selected_Peak.csv`.

## Evidence

1. **The file labels itself as Woloso_5.** Every row has `Take = Woloso_5` and `Piece = Woloso`, and the cycle labels use Woloso's half-cycle notation (`1.1|1`, `1.2|1`, ...). Maraka files use `1|1`. The first data row is

   ```
   Woloso_5,Woloso,3,1.1|1,1,4.471,4.452,4.47,,3,,,,Jeli Madi Kuyate,Drissa Kone,Madu Jakite,
   ```

2. **Its onsets belong to the Woloso_5 audio.** The D1 onsets of the file were matched to the raw dundun onsets of each take (`*_Onsets_Raw_Dundun1.csv`). The table gives the share of selected D1 onsets that fall within a given tolerance of a raw onset:

   | Raw onsets of | 10 ms | 20 ms | 50 ms |
   |---|---|---|---|
   | Woloso_5 | 98.6 % | 99.7 % | 99.7 % |
   | Maraka_5 | 9.9 % | 20.0 % | 43.2 % |
   | chance overlap for Maraka_5 (2 × tolerance × 3.8 raw onsets/s) | 7.6 % | 15 % | 38 % |

   Against Maraka_5, the match never rises clearly above chance. Against Woloso_5, it is near-complete from 10 ms. (Below 10 ms, the match drops for both takes, because the `_Selected` times are rounded and not snapped to the raw onsets.)

3. **The `_Selected_Peak` files are correct for all 15 takes.** Each one carries its own take label, and 99.9 % to 100 % of its D1 onsets coincide (within 5 ms) with the raw dundun onsets of the same take.

To reproduce (after `python djembe_dataset.py download`):

```python
import numpy as np, pandas as pd
ann = "corpus/MJ_{p}/annotations/MJ_{p}_{t}_{s}.csv"
sel = pd.read_csv(ann.format(p="Maraka", t=5, s="Onsets_Selected"))
print(sel["Take"].unique())                     # ['Woloso_5']
d1 = sel["D1"].dropna().to_numpy()
for p in ("Woloso", "Maraka"):
    raw = pd.read_csv(ann.format(p=p, t=5, s="Onsets_Raw_Dundun1"))["Time"].to_numpy()
    near = np.abs(d1[:, None] - raw[None, :]).min(axis=1) < 0.010
    print(p, f"{near.mean():.1%}")              # Woloso 98.6 %, Maraka 9.9 %
```

## Effect on this project

The analysis notebooks (V1 and V2) loaded the `_Onsets_Selected` files. As a result, one Woloso take was labelled Maraka, the real Maraka_5 annotations were never used, and Woloso_5 was missing. The take-level rhythm classifier scored 12/14 on that input. Its two errors were the mislabelled take, predicted "Woloso" (which is in fact correct), and Woloso_2, which lost to Maraka because Woloso had only three clean training takes.

## Workaround

`djembe_dataset.py` builds the event table from the `_Onsets_Selected_Peak` files. They hold the same metric positions, with onset times snapped to the raw onsets and a peak amplitude column. `load_take` raises an error if a file's `Take` column does not match its file name. `tests/test_splits.py` checks that guard on a mislabelled copy of a real file. With the clean input, the same classifier scores 15/15.

The notebooks in the repository root still read the `_Onsets_Selected` files through `utils_V2.py`. Their Maraka and Woloso results include this error until they are ported to the event table.
