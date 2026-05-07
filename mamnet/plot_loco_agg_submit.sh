#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"

# ============================================================
# PART 1: Train individual file
# ============================================================
echo "Queueing individual file..."

name="LOCO_agg_plots_file"
outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/${name}.out"
RESULTS_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/loco_aggregate_results"

echo "${name}"
sbatch --output=${outputfile} \
	   --job-name=${name} \
	   --export=RESULTS_DIR=${RESULTS_DIR},OUTPUT_DIR=${OUTPUT_DIR} \
	   plot_loco_agg.sh

echo "All jobs queued!"