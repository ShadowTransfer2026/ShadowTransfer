#!/bin/bash
# FILENAME: mamnet.sh
#SBATCH -A sukkusur
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=3:59:59

cd /home/mittal53/ShadeMaps/python/mamnet
module load conda
module load cuda/12.1.1  # or cuda/12.6.0 (which is the default)
module load cudnn/9.2.0.82-12
conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12

# Set environment variables for single-node distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# Build the python command based on MODE
PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/satmae_cuda12/bin/python

# Print configuration
echo "=================================="
echo "Fine-tuning Configuration:"
echo "=================================="
echo "Target City: ${TARGET_CITY}"
echo "Resolution: ${RESOLUTION}"
echo "N Samples: ${N_SAMPLES}"
echo "Strategy: ${STRATEGY}"
echo "Random Seed: ${RANDOM_SEED}"
echo "Base Data Root: ${BASE_DATA_ROOT}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=================================="
echo ""

# $PYTHON_BIN generate_split.py \
    # --city ${CITY} \
    # --resolution ${RESOLUTION} \
    # --n_samples ${N_SAMPLES} \
    # --strategy ${STRATEGY} \
    # --random_seed ${SEED} \
    # --base_data_root ${BASE_DATA_ROOT} \
    # --output_dir ${OUTPUT_DIR}

# echo "Split generation complete!"

# Run fine-tuning
$PYTHON_BIN train_finetuning.py \
    --target_city ${TARGET_CITY} \
    --resolution ${RESOLUTION} \
    --n_samples ${N_SAMPLES} \
    --strategy ${STRATEGY} \
    --random_seed ${RANDOM_SEED} \
    --base_data_root ${BASE_DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
	--split_file ${SPLIT_FILE}
    --batch_size 8 \
    --lr 0.0001 \
    --max_epochs 10 \
    --early_stop_patience 3 \
    --num_workers 1 \
    --img_size 384

echo ""
echo "Fine-tuning completed!"