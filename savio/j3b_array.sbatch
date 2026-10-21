#!/bin/bash
# Graph model, all folds x seeds as one job array (one task = one test take, one seed).
# Submit from the repository root on Savio, with your account (see savio/README.md):
#
#   sbatch -A fc_YOURACCOUNT --array=0-44 savio/j3b_array.sbatch           # 15 takes x 3 seeds
#   SEEDS="0 1 2 3 4" sbatch -A fc_YOURACCOUNT --array=0-74 savio/j3b_array.sbatch
#
# CPU only: the model is small (64-dim, 3 layers), so per-core scheduling on an HTC
# partition is cheaper than a GPU. Memory follows the core count on HTC partitions.
#SBATCH --job-name=djembe-j3b
#SBATCH --partition=savio2_htc
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=slurm/%x_%A_%a.out

# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail
module load ml/pytorch/2.3.1-py3.11.7
# DJEMBE_VENV / DJEMBE_OUT move the venv and the results off scratch (e.g. to $HOME) if scratch misbehaves
source "${DJEMBE_VENV:-/global/scratch/users/$USER/envs/djembe}/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEEDS=${SEEDS:-"0 1 2"}
VARIANTS=${VARIANTS:-"V0 V5"}
RUN=${RUN:-j3b}
OUT_DIR=${DJEMBE_OUT:-/global/scratch/users/$USER/djembe_runs}/$RUN/$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT_DIR"

ARGS=$(python experiments/gnn/array_task.py --index "$SLURM_ARRAY_TASK_ID" --seeds $SEEDS)
echo "task $SLURM_ARRAY_TASK_ID: $ARGS, variants $VARIANTS"
python experiments/gnn/train_humanize.py $ARGS --variants $VARIANTS --select last --device cpu \
    --group-prefix "$RUN" --out "$OUT_DIR/part_${SLURM_ARRAY_TASK_ID}.json"
