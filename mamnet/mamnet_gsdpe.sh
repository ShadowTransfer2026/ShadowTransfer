#!/bin/bash
# FILENAME: mamnet.sh
#SBATCH -A sukkusur
#SBATCH --partition=a100-40gb
#SBATCH --exclude=gilbreth-n015
#SBATCH --gres=gpu:1
#SBATCH --qos=standby
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
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

if [ "$MODE" == "single" ]; then
    # Single city mode
    $PYTHON_BIN train_gsdpe.py \
		--data_root ${DATA_ROOT} \
		--cities ${CITY} \
		--train_resolution ${RES} \
		--batch_size 8 \
		--epochs 100 \
		--lr 0.001 \
		--weight_decay 0.005 \
		--output_dir ${OUTPUT_DIR} \
		--num_workers 1 \
		${USE_CONTRAST} \
		${EVAL_TOLERANT}

elif [ "$MODE" == "all" ]; then
    # All cities mode (bonus - in case you want it)
    $PYTHON_BIN train_gsdpe.py \
        --mode all \
        --data_root ${DATA_ROOT} \
        --resolution ${RES} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1 \
		${USE_CONTRAST} \
		${EVAL_TOLERANT}

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi