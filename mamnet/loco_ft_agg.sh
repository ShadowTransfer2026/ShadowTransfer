#!/bin/bash
# FILENAME: mamnet.sh
#SBATCH -A sukkusur
#SBATCH --partition=a100-40gb
#SBATCH --qos=normal
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
echo "Fine-tuning Aggregate Analysis:"
echo "=================================="
echo ""

# Run fine-tuning
$PYTHON_BIN analyze_finetune_results.py \
    --output_dir "/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs" --save_dir "/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/fine_tune/"
	
# $PYTHON_BIN check_job_status.py \
    # --output_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs \
    # --log_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/finetune \
    # --save_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/job_reports \
	# --splits_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/splits
	
# $PYTHON_BIN visualize_splits_spatial_metrics.py \
    # --split_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/splits/ \
    # --output_dir /scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/splits/spatial_validation_plots


echo ""
echo "Fine-tuning Analysis completed!"