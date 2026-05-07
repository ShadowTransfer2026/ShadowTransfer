#!/bin/bash
# FILENAME: compute_isw_masks_oglanet.sh
# Precompute ISW sensitivity masks for OGLANet (one-time, run before training).

# ---- SBATCH: Server-specific (uncomment the one you need) ----

# --- Gilbreth ---
# #SBATCH -A sukkusur
# #SBATCH --partition=a100-40gb
# #SBATCH --gres=gpu:1
# #SBATCH --qos=normal
# #SBATCH --constraint=a100
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# #SBATCH --mem=64G
# #SBATCH --time=1:59:59

# --- Anvil ---
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=1
# #SBATCH --exclude=g005
# #SBATCH --mem=64G
# #SBATCH --time=1:59:59

# --- NCSA Delta ---
#SBATCH --account=bgpi-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:59:59
#SBATCH --job-name=oglanet_isw_mask

# ---- Modules (uncomment the one you need) ----

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12
# cd /home/mittal53/ShadeMaps/python/oglanet
# PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate /anvil/projects/x-cis260282/satmae_cuda12
# cd /anvil/projects/x-cis260282/ShadeMaps/python/oglanet
# PYTHON_BIN=/anvil/projects/x-cis260282/satmae_cuda12/bin/python

# --- NCSA Delta ---
module purge
module load pytorch-conda/2.8
conda activate /projects/bgpi/smittal5/ShadeMaps/envs/satmae_cuda12
cd /projects/bgpi/smittal5/ShadeMaps/python/oglanet
PYTHON_BIN=/projects/bgpi/smittal5/ShadeMaps/envs/satmae_cuda12/bin/python

export PYTHONUNBUFFERED=1

# ---- Build optional flags ----
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

CHECKPOINT_FLAG=""
if [ -n "${CHECKPOINT}" ]; then
    CHECKPOINT_FLAG="--checkpoint ${CHECKPOINT}"
fi

NUM_SAMPLES_FLAG=""
if [ -n "${NUM_SAMPLES}" ] && [ "${NUM_SAMPLES}" != "0" ]; then
    NUM_SAMPLES_FLAG="--num_samples ${NUM_SAMPLES}"
fi

echo "Mode:           ${MODE}"
echo "Output dir:     ${ISW_MASK_OUTPUT_DIR}"
echo "Contrast:       ${CONTRAST_FLAG}"
echo "Checkpoint:     ${CHECKPOINT_FLAG}"
echo "Num samples:    ${NUM_SAMPLES_FLAG}"

# ---- Debug checks ----
echo ""
echo "=== DEBUG CHECKS ==="
echo "Working dir: $(pwd)"
echo "Python:      ${PYTHON_BIN}"
echo "Python version: $(${PYTHON_BIN} --version 2>&1)"

if [ ! -f "compute_isw_masks_oglanet.py" ]; then
    echo "ERROR: compute_isw_masks_oglanet.py not found in $(pwd)"
    exit 1
fi
echo "compute_isw_masks_oglanet.py: OK"

if [ ! -f "utils/isw_loss.py" ]; then
    echo "ERROR: utils/isw_loss.py not found (copy from mamnet/utils/)"
    exit 1
fi
echo "utils/isw_loss.py: OK"

${PYTHON_BIN} -c "
import sys; sys.path.insert(0, '.')
print('  importing models.oglanet...')
from models.oglanet import OGLANet
print('  importing utils.isw_loss...')
from utils.isw_loss import ISWLoss, EncoderFeatureHooks
print('  importing data.dataset...')
from data.dataset import get_dataloaders
print('ALL imports OK')
" 2>&1
IMPORT_RC=$?
if [ $IMPORT_RC -ne 0 ]; then
    echo "ERROR: Import check failed (exit code ${IMPORT_RC})"
    exit 1
fi
echo "=== END DEBUG CHECKS ==="
echo ""

# ---- Run precomputation ----
echo "Launching ISW mask precomputation for OGLANet..."

if [ "$MODE" == "single" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_oglanet.py \
        --mode single \
        --data_root "${DATA_ROOT}" \
        --img_size 384 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG} 2>&1
    RC=$?

elif [ "$MODE" == "loco" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_oglanet.py \
        --mode loco \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --fold_id "${FOLD_ID}" \
        --img_size 384 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG} 2>&1
    RC=$?

elif [ "$MODE" == "all" ]; then
    ${PYTHON_BIN} -u compute_isw_masks_oglanet.py \
        --mode all \
        --base_data_root "${BASE_DATA_ROOT}" \
        --resolution "${RESOLUTION}" \
        --img_size 384 \
        --output_dir "${ISW_MASK_OUTPUT_DIR}" \
        --num_workers 2 \
        ${CONTRAST_FLAG} \
        ${CHECKPOINT_FLAG} \
        ${NUM_SAMPLES_FLAG} 2>&1
    RC=$?

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi

echo ""
echo "=== Precompute exited with code: ${RC} ==="
exit ${RC}