#!/bin/bash
# Roadmap J6: graph model vs transformer, soloist stroke by stroke, seen and unseen rhythms.
# PROTOCOL = take (45 tasks), take_nopiece (45) or piece (9). Submit from the repository root on Savio:
#
#   PROTOCOL=take  DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-44 savio/j6_array.sbatch
#   PROTOCOL=piece DJEMBE_VENV=$HOME/envs/djembe DJEMBE_OUT=$HOME/djembe_runs sbatch -A fc_YOURACCOUNT --array=0-8  savio/j6_array.sbatch
#
#SBATCH --job-name=djembe-j6
#SBATCH --partition=savio2_htc
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=slurm/%x_%A_%a.out

# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail
module load ml/pytorch/2.3.1-py3.11.7
source "${DJEMBE_VENV:-/global/scratch/users/$USER/envs/djembe}/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

PROTOCOL=${PROTOCOL:?set PROTOCOL to take, take_nopiece or piece}
SEEDS=${SEEDS:-"0 1 2"}
OUT_DIR=${DJEMBE_OUT:-/global/scratch/users/$USER/djembe_runs}/j6_$PROTOCOL/$SLURM_ARRAY_JOB_ID
mkdir -p "$OUT_DIR/grids"

echo "task $SLURM_ARRAY_TASK_ID: protocol $PROTOCOL"
python experiments/j6/run.py --protocol "$PROTOCOL" --index "$SLURM_ARRAY_TASK_ID" --seeds $SEEDS --device cpu \
    --out "$OUT_DIR/part_${SLURM_ARRAY_TASK_ID}.json" --save-dir "$OUT_DIR/grids"
