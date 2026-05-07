#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"
LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/finetune"
LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/job_reports"

mkdir -p ${LOG_DIR}

# Create descriptive job name
name="ft_plots"
# name="visualize_splits_metrics"
# name="poor_jobs"
outputfile="${LOG_DIR}/${name}.out"

# Submit job
sbatch --output=${outputfile} \
	   --job-name=${name} \
	   loco_ft_agg.sh