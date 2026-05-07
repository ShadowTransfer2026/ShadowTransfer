#!/bin/bash
# FILENAME: dinov3.sh
#SBATCH -A sukkusur
#SBATCH --partition=a100-40gb
#SBATCH --exclude=gilbreth-n015
#SBATCH --gres=gpu:1
#SBATCH --qos=standby
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --time=1:29:59

cd /home/mittal53/ShadeMaps/python/dinov3
module load conda
module load cuda/12.1.1  # or cuda/12.6.0 (which is the default)
module load cudnn/9.2.0.82-12
conda activate /scratch/gilbreth/mittal53/ShadeMaps/conda_envs/dinov3_env

# Set environment variables for single-node distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12355
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

# Build the python command based on MODE
PYTHON_BIN=/scratch/gilbreth/mittal53/ShadeMaps/conda_envs/dinov3_env/bin/python

if [ "$MODE" == "single" ]; then
    # Single city mode
    $PYTHON_BIN train_dinov3_gsdpe.py \
        --data_root ${DATA_ROOT} \
		--model_name dinov3_vits16 \
		--cities ${CITY} \
		--resolution ${RES} \
		--weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.0001 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1

elif [ "$MODE" == "loco" ]; then
    # LOCO mode
    $PYTHON_BIN train_dinov3.py \
        --mode loco \
        --base_data_root ${BASE_DATA_ROOT} \
        --resolution ${RESOLUTION} \
        --fold_id ${FOLD_ID} \
		--model_name dinov3_vits16 \
		--weights_path ${WEIGHT_DIR} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1

elif [ "$MODE" == "all" ]; then
    # All cities mode (bonus - in case you want it)
    $PYTHON_BIN train_dinov3.py \
        --mode all \
        --base_data_root ${BASE_DATA_ROOT} \
		--model_name dinov3_vits16 \
		--weights_path ${WEIGHT_DIR} \
        --resolution ${RESOLUTION} \
        --batch_size 8 \
        --epochs 100 \
        --lr 0.00005 \
        --img_size 384 \
        --output_dir ${OUTPUT_DIR} \
        --num_workers 1

else
    echo "ERROR: Unknown MODE=${MODE}"
    exit 1
fi