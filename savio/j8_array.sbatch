#!/bin/bash
# Roadmap J8: two-stage soloist (figure planner + graph model) against the graph model alone, leave-one-take-out.
# 15 takes x 3 seeds = 45 tasks. Submit from the repository root on Savio:
#
#   DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-44 savio/j8_array.sbatch
#
#SBATCH --job-name=djembe-j8
#SBATCH --partition=savio2_htc
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%A_%a.out

# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail
module load ml/pytorch/2.3.1-py3.11.7
source "${DJEMBE_VENV:-/global/scratch/users/$USER/envs/djembe}/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=${SEEDS:-"0 1 2"}
OUT_DIR=${DJEMBE_OUT:-/global/scratch/users/$USER/djembe_runs}/j8/$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT_DIR/grids"

echo "task $SLURM_ARRAY_TASK_ID"
python experiments/j8/run.py --index "$SLURM_ARRAY_TASK_ID" --seeds $SEEDS \
    --out "$OUT_DIR/part_${SLURM_ARRAY_TASK_ID}.json" --save-dir "$OUT_DIR/grids"
