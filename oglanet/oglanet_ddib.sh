#!/bin/bash
# FILENAME: oglanet_ddib.sh
#SBATCH -A cis260282-gpu
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --exclude=g005
#SBATCH --time=2:59:59

# ---- Server paths (uncomment the one you need) ----
# Gilbreth
##SBATCH -A sukkusur
##SBATCH --partition=a100-40gb
##SBATCH --gres=gpu:1
##SBATCH --qos=standby

# Anvil
##SBATCH -A cis260282-gpu
##SBATCH -p gpu


# --- Gilbreth ---
# cd /home/mittal53/ShadeMaps/python/oglanet
# PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python

# --- Anvil ---
cd /anvil/projects/x-cis260282/ShadeMaps/python/oglanet
PYTHON_BIN=/anvil/projects/x-cis260282/satmae_cuda12/bin/python

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

# ---- Build DDIB flags from environment variables ----
DDIB_FLAGS=""
if [ "${USE_C1}" == "1" ]; then
    DDIB_FLAGS="${DDIB_FLAGS} --use_disentangle"
fi
if [ "${USE_C2}" == "1" ]; then
    DDIB_FLAGS="${DDIB_FLAGS} --use_vib"
fi
if [ "${USE_C3}" == "1" ]; then
    DDIB_FLAGS="${DDIB_FLAGS} --use_feat_aug"
fi
if [ "${USE_SF}" == "1" ]; then
    DDIB_FLAGS="${DDIB_FLAGS} --use_skip_filter"
fi

LOSS_FLAGS=""
if [ -n "${LAMBDA_HSIC}" ]; then
    LOSS_FLAGS="${LOSS_FLAGS} --lambda_hsic ${LAMBDA_HSIC}"
fi
if [ -n "${LAMBDA_DOMAIN}" ]; then
    LOSS_FLAGS="${LOSS_FLAGS} --lambda_domain ${LAMBDA_DOMAIN}"
fi
if [ -n "${LAMBDA_KL}" ]; then
    LOSS_FLAGS="${LOSS_FLAGS} --lambda_kl ${LAMBDA_KL}"
fi

# Optional: contrast channel
CONTRAST_FLAG=""
if [ "${USE_CONTRAST}" == "1" ]; then
    CONTRAST_FLAG="--use_contrast"
fi

echo "DDIB flags:     ${DDIB_FLAGS}"
echo "Loss flags:     ${LOSS_FLAGS}"
echo "Contrast flag:  ${CONTRAST_FLAG}"

# ---- Run training ----
if [ "$MODE" == "single" ]; then
    $PYTHON_BIN train_oglanet_ddib.py \
        --mode single \
        --data_root ${DATA_ROOT} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 15 \
	--comparison_inference_dir ${COMPARISON_INFERENCE_DIR}\
	--comparison_data_root ${COMPARISON_DATA_ROOT}\
        ${CONTRAST_FLAG} \
        ${DDIB_FLAGS} \
        ${LOSS_FLAGS}

elif [ "$MODE" == "loco" ]; then
    $PYTHON_BIN -u train_oglanet_ddib.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 15 \
	--comparison_inference_dir ${COMPARISON_INFERENCE_DIR}\
	--comparison_data_root ${COMPARISON_DATA_ROOT}\
        ${CONTRAST_FLAG} \
        ${DDIB_FLAGS} \
        ${LOSS_FLAGS}

elif [ "$MODE" == "all" ]; then
    $PYTHON_BIN train_oglanet_ddib.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
        --eval_boundary_tolerant \
        --early_stopping_patience 15 \
	--comparison_inference_dir ${COMPARISON_INFERENCE_DIR}\
	--comparison_data_root ${COMPARISON_DATA_ROOT}\
        ${CONTRAST_FLAG} \
        ${DDIB_FLAGS} \
        ${LOSS_FLAGS}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi