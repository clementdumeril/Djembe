#!/bin/bash
# Generation (roadmap J5): train the strike and timing models, generate the held-out take, judge it.
# One task per (test take, seed). Submit from the repository root on Savio (see savio/README.md):
#
#   DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-44 savio/gen_array.sbatch
#
#SBATCH --job-name=djembe-j5
#SBATCH --partition=savio2_htc
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%A_%a.out

# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail
module load ml/pytorch/2.3.1-py3.11.7
# DJEMBE_VENV / DJEMBE_OUT move the venv and the results off scratch (e.g. to $HOME) if scratch misbehaves
source "${DJEMBE_VENV:-/global/scratch/users/$USER/envs/djembe}/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=${SEEDS:-"0 1 2"}
RUN=${RUN:-j5}
OUT_DIR=${DJEMBE_OUT:-/global/scratch/users/$USER/djembe_runs}/$RUN/$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT_DIR/grids"

ARGS=$(python experiments/gnn/array_task.py --index "$SLURM_ARRAY_TASK_ID" --seeds $SEEDS)
echo "task $SLURM_ARRAY_TASK_ID: $ARGS"
python experiments/generation/generate_eval.py $ARGS --device cpu \
    --out "$OUT_DIR/part_${SLURM_ARRAY_TASK_ID}.json" --save-dir "$OUT_DIR/grids"
