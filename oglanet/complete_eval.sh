#!/bin/bash
# FILENAME: complete_eval.sh
#
# SLURM job script — runs complete_eval.py for one checkpoint.
# Submitted by complete_eval_submit.sh via sbatch --export=...
#
# Required env vars (passed via --export):
#   CHECKPOINT_PATH            — path to best_model.pth
#   OUTPUT_DIR                 — experiment output dir (checkpoint grandparent)
#   BASE_DATA_ROOT             — path to Final_data_test/
#   COMPARISON_INFERENCE_DIR   — path to baseline inference dir
#   COMPARISON_DATA_ROOT       — path to holdout city dir for GT masks
#
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -A cis260282-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=0:29:59

# =====================================================================
# Server paths — Gilbreth (uncomment to use instead of Anvil)
# =====================================================================
##SBATCH -A sukkusur
##SBATCH --partition=a100-40gb
##SBATCH --gres=gpu:1
##SBATCH --qos=standby

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup — Anvil
# ─────────────────────────────────────────────────────────────────────────────
cd /anvil/projects/x-cis260282/ShadeMaps/python/oglanet
PYTHON_BIN=/anvil/projects/x-cis260282/satmae_cuda12/bin/python

module purge
module load modtree/gpu
module load cuda/12.6.1
module load anaconda
conda activate /anvil/projects/x-cis260282/satmae_cuda12

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup — Gilbreth (uncomment to swap)
# ─────────────────────────────────────────────────────────────────────────────
# cd /home/mittal53/ShadeMaps/python/oglanet
# PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12

# ─────────────────────────────────────────────────────────────────────────────
# Validate required variables
# ─────────────────────────────────────────────────────────────────────────────
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: CHECKPOINT_PATH not set."
    exit 1
fi
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "============================================="
echo "  OGLANet+SIB — Complete Interrupted Eval"
echo "============================================="
echo "  CHECKPOINT : ${CHECKPOINT_PATH}"
echo "  OUTPUT_DIR : ${OUTPUT_DIR}"
echo "  DATA_ROOT  : ${BASE_DATA_ROOT}"
echo "  COMP_INF   : ${COMPARISON_INFERENCE_DIR}"
echo "  COMP_DATA  : ${COMPARISON_DATA_ROOT}"
echo "============================================="
echo ""

export PYTHONUNBUFFERED=1

$PYTHON_BIN -u complete_eval.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --base_data_root "${BASE_DATA_ROOT}" \
    --comparison_inference_dir "${COMPARISON_INFERENCE_DIR}" \
    --comparison_data_root "${COMPARISON_DATA_ROOT}" \
    --num_workers 1 \
    --batch_size 8