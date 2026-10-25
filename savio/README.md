# Running the graph model on Savio

Savio is UC Berkeley's SLURM cluster. Here it runs every (test take, seed) unit of `experiments/gnn/train_humanize.py` as one task of a job array, then merges the parts into one result file. One task (2 variants, 40 epochs) takes about 15 minutes on a laptop core pair; the 45-task array for 3 seeds costs on the order of 50 to 100 service units on `savio2_htc`.

Partition, QoS and GPU names below come from the Research IT user guide (checked October 2026): https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/

Use an allocation that covers this project. A course (instructional) allocation is meant for that course's work, so check with the course staff or Research IT before using one here; a faculty allowance (`fc_...`) for the project is the usual route.

## 1. Once: code and environment

```bash
ssh $USER@hpc.brc.berkeley.edu
sacctmgr -p show associations user=$USER format=account,user,partition,qos   # your accounts
cd /global/scratch/users/$USER
git clone https://github.com/clementdumeril/Djembe.git && cd Djembe
bash savio/setup_env.sh        # venv in scratch on top of the ml/pytorch module
```

## 2. Once: the annotations (about 10 MB)

The corpus is CC BY-NC-ND 4.0: it stays in your scratch space and is never committed. The guide does not say whether compute nodes reach the internet, so prepare the data before submitting:

```bash
module load ml/pytorch/2.3.1-py3.11.7 && source /global/scratch/users/$USER/envs/djembe/bin/activate
python -m djembe.dataset download   # from OSF; if OSF refuses, copy your private djembe_corpus.zip through dtn.brc.berkeley.edu and unzip it into corpus/
python -m djembe.dataset build      # output/events.csv
python -m pytest -q                 # optional check
```

## 3. Submit, follow, merge

```bash
mkdir -p slurm
sbatch -A fc_YOURACCOUNT --array=0-44 savio/j3b_array.sbatch                       # V0 and V5, seeds 0 1 2
SEEDS="0 1 2 3 4" sbatch -A fc_YOURACCOUNT --array=0-74 savio/j3b_array.sbatch     # 5 seeds
squeue -u $USER                     # running tasks; module load sq; sq explains pending ones
python experiments/gnn/merge_results.py /global/scratch/users/$USER/djembe_runs/j3b/<JOBID>/part_*.json \
    --out results/j3b_humanize_3seeds_savio.json --expect-takes 15 --expect-seeds 3
```

`python experiments/gnn/array_task.py --count --seeds 0 1 2` prints the array size for any seed list. The merge refuses incomplete runs (`--expect-*`) and conflicting duplicates, so a failed task shows up instead of silently shrinking the average. Rerun only the missing indices with `--array=3,17`.

Before a long array, check one task on the debug QoS: `sbatch -A fc_YOURACCOUNT -q savio_debug --array=14 --time=00:30:00 savio/j3b_array.sbatch`.

## Data-efficiency curve (J4)

`savio/curve_array.sbatch` trains V0, V1 (beat symmetry), V3 (pulse rotation) and V5 on 2, 4, 8 and 13 training takes, with B1 and B2 fitted on the same takes: 15 takes x 3 repetitions x 4 sizes = 180 tasks, from about 2 minutes (2 takes) to about 10 minutes (13 takes) each.

```bash
sbatch -A fc_YOURACCOUNT --array=0-179 savio/curve_array.sbatch
python experiments/gnn/merge_results.py /global/scratch/users/$USER/djembe_runs/j4/<JOBID>/part_*.json --out results/j4_curve_savio.json --expect-takes 15 --expect-seeds 3
python experiments/gnn/plot_curve.py results/j4_curve_savio.json docs/figures/j4_curve.png
```

## Generation (J5)

`savio/gen_array.sbatch` trains the strike model and the timing model (V5) per fold, generates the held-out take from its first 3 real cycles, with the GNN and with the Markov reference, and compares both with the real take: 15 takes x 3 seeds = 45 tasks of about 5 to 10 minutes. Generated grids are saved next to the parts, for rendering.

```bash
DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-44 savio/gen_array.sbatch
python experiments/generation/summarise_generation.py $HOME/djembe_runs/j5/<JOBID>/part_*.json --out results/j5_generation_savio.json --expect-takes 15 --expect-seeds 3
python -m djembe.render $HOME/djembe_runs/j5/<JOBID>/grids/Woloso_5_s0_GNN.npz output/Woloso_5_GNN.wav
```

Generated audio and grids come from a model trained on the corpus: keep them out of the repository.

## Soloist generation (J5b)

Jembe 1 is the soloist of the corpus. `savio/solo_array.sbatch` trains the soloist models per fold, scores their predictions against B2, picks the sampling temperature on the validation take, generates the held-out take's soloist over its real ensemble (strikes pulse by pulse) and compares it, and a Markov soloist, with the real one: 15 takes x 3 seeds = 45 tasks of about 5 to 10 minutes.

```bash
DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-44 savio/solo_array.sbatch
python experiments/generation/summarise_solo.py $HOME/djembe_runs/j5b/<JOBID>/part_*.json --out results/j5b_solo_savio.json --expect-takes 15 --expect-seeds 3
python -m djembe.render $HOME/djembe_runs/j5b/<JOBID>/grids/Maraka_2_s0_GNN_solo.npz output/Maraka_2_solo.wav --solo-emphasis 1.8
```

## GPUs, for later

The current model does not need one. For pretraining on a large drum corpus, the A5000 nodes are the regular-priority option under a faculty allowance (at most 4 GPUs per user):

```bash
#SBATCH --partition=savio4_gpu
#SBATCH --gres=gpu:A5000:1
#SBATCH --cpus-per-task=4
#SBATCH --qos=a5k_gpu4_normal
```

Typed `--gres` and the matching CPU count are required; jobs that ask for too few CPUs per GPU are rejected.
