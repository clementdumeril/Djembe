#!/bin/bash
# Data-efficiency curve (roadmap J4): every (test take, repetition, training size) as one task.
# Submit from the repository root on Savio (see savio/README.md):
#
#   sbatch -A fc_YOURACCOUNT --array=0-179 savio/curve_array.sbatch     # 15 takes x 3 reps x sizes 2 4 8 13
#
# Each task trains the variants below on n training takes and fits B1/B2 on the same takes.
#SBATCH --job-name=djembe-j4
#SBATCH --partition=savio2_htc
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --output=slurm/%x_%A_%a.out

# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail
module load ml/pytorch/2.3.1-py3.11.7
source /global/scratch/users/$USER/envs/djembe/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=${SEEDS:-"0 1 2"}
SIZES=${SIZES:-"2 4 8 13"}
VARIANTS=${VARIANTS:-"V0 V1 V3 V5"}
RUN=${RUN:-j4}
OUT_DIR=/global/scratch/users/$USER/djembe_runs/$RUN/$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT_DIR"

ARGS=$(python experiments/gnn/array_task.py --index "$SLURM_ARRAY_TASK_ID" --seeds $SEEDS --n-train $SIZES)
echo "task $SLURM_ARRAY_TASK_ID: $ARGS, variants $VARIANTS"
python experiments/gnn/train_curve.py $ARGS --variants $VARIANTS --select last --device cpu \
    --out "$OUT_DIR/part_${SLURM_ARRAY_TASK_ID}.json"
