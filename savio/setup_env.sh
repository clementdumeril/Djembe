#!/bin/bash
# One-time setup on a Savio login node: code and Python environment in scratch.
#   bash savio/setup_env.sh
# The graph model needs only PyTorch (no PyTorch Geometric), which the ml/pytorch
# module provides; the venv adds the project and its light dependencies on top.
# No -u: the ml/pytorch module's activation script reads unset variables.
set -eo pipefail

SCRATCH_DIR=/global/scratch/users/$USER
REPO=$SCRATCH_DIR/Djembe
VENV=$SCRATCH_DIR/envs/djembe

module load ml/pytorch/2.3.1-py3.11.7

if [ ! -d "$REPO" ]; then
    git clone https://github.com/clementdumeril/Djembe.git "$REPO"
fi
if [ ! -d "$VENV" ]; then
    python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet -e "$REPO"
python -c "import torch, djembe; print('torch', torch.__version__, '| djembe ok')"
