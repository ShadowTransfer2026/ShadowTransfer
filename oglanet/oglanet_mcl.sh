#!/bin/bash
# FILENAME: oglanet_mcl.sh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=g005
#SBATCH --mem=128G
#SBATCH --time=6:59:59

# ---- Server SBATCH directives (uncomment the one you need) ----

# --- Gilbreth ---
##SBATCH -A sukkusur
##SBATCH --partition=a100-40gb
##SBATCH --gres=gpu:1
##SBATCH --qos=standby
##SBATCH --constraint=a100
##SBATCH --exclude=gilbreth-n015

# --- Anvil ---
#SBATCH -A cis260282-gpu
#SBATCH -p gpu

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
# cd /home/mittal53/ShadeMaps/python/oglanet
# PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
cd /anvil/projects/x-cis260282/ShadeMaps/python/oglanet
PYTHON_BIN=/anvil/projects/x-cis260282/satmae_cuda12/bin/python

# ---- Module / conda setup (uncomment the one you need) ----

# --- Gilbreth ---
# module load conda
# module load cuda/12.1.1
# module load cudnn/9.2.0.82-12
# conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12

# --- Anvil ---
module purge
module load modtree/gpu
module load cuda/12.6.1
module load anaconda
conda activate /anvil/projects/x-cis260282/satmae_cuda12

# Distributed training env vars (single node)
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export PYTHONUNBUFFERED=1

# ---- Build optional flags from environment variables ----
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

TOLERANT_FLAG=""
if [ "${EVAL_TOLERANT}" == "1" ]; then
    TOLERANT_FLAG="--eval_boundary_tolerant"
fi

echo "Contrast flag:  ${CONTRAST_FLAG}"
echo "Tolerant flag:  ${TOLERANT_FLAG}"

# ---- Run training ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN train_mcl.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_mcl.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 4 \
        --epochs 120 \
        --lr 0.0005 \
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
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN train_mcl.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --early_stopping_patience 15 \
        --comparison_inference_dir ${COMPARISON_INFERENCE_DIR} \
        --comparison_data_root ${COMPARISON_DATA_ROOT} \
        ${CONTRAST_FLAG} \
        ${TOLERANT_FLAG}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi