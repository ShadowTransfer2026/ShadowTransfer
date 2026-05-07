#!/bin/bash
# FILENAME: mamnet_mcl.sh
# ---- SBATCH: Common settings ----
# --- Anvil ---
# #SBATCH -A cis260282-gpu
# #SBATCH -p gpu
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --cpus-per-task=4
# # #SBATCH --exclude=g005
# #SBATCH --mem=64G
# #SBATCH --time=0:50:59
# ---- SBATCH: Server-specific (uncomment the one you need) ----
# --- Gilbreth ---
#SBATCH -A sukkusur
#SBATCH --partition=a100-40gb
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --constraint=a100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:59:59

# ---- Server paths (uncomment the one you need) ----
# --- Gilbreth ---
cd /home/mittal53/ShadeMaps/python/mamnet
PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python
module load conda
module load cuda/12.1.1
module load cudnn/9.2.0.82-12
conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12

# --- Anvil ---
# module purge
# module load modtree/gpu
# module load cuda/12.6.1
# module load anaconda
# conda activate /anvil/projects/x-cis260282/satmae_cuda12
# cd /anvil/projects/x-cis260282/ShadeMaps/python/mamnet
# PYTHON_BIN=/anvil/projects/x-cis260282/satmae_cuda12/bin/python

# Distributed training env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ---- Build optional flags from env vars ----

# USE_CONTRAST and EVAL_TOLERANT are passed as the full flag strings
# (e.g. "--use_contrast") from the submit script, so they expand directly.

# CHANGE: BOUNDARY_TOLERANCE is now passed as an integer from the submit script.
# Build the --boundary_tolerance flag here so it can be conditionally omitted
# when the var is unset (falls back to Python's default of 2).
BOUNDARY_TOL_FLAG=""
if [ -n "${BOUNDARY_TOLERANCE}" ]; then
    BOUNDARY_TOL_FLAG="--boundary_tolerance ${BOUNDARY_TOLERANCE}"
fi

# Build the python command based on MODE
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN -u train_mcl.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 3 \
        --lr 0.001 \
        --weight_decay 1e-4 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${USE_CONTRAST} \
        ${EVAL_TOLERANT} \
        ${BOUNDARY_TOL_FLAG}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_mcl.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 4 \
        --epochs 15 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --use_mcl \
        --use_bane \
        --lambda_fl 0.001 \
        --lambda_sl 0.001 \
        --lambda_lc 0.0005 \
        --early_stopping_patience 15 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${USE_CONTRAST} \
        ${EVAL_TOLERANT} \
        ${BOUNDARY_TOL_FLAG}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN -u train_mcl.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 12 \
        --lr 0.0001 \
        --weight_decay 1e-4 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${USE_CONTRAST} \
        ${EVAL_TOLERANT} \
        ${BOUNDARY_TOL_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi