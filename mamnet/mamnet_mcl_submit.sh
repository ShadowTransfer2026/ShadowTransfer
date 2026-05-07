#!/bin/bash
# FILENAME: mamnet_mcl_submit.sh
#
# Queues MAMNet + mCL-LC training jobs on SLURM.
# Uncomment the server block you need.

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
BASE_PATH="/scratch/gilbreth/mittal53/ShadeMaps"
BASE_PATH2="/home/mittal53/ShadeMaps"

# --- Anvil ---
# BASE_PATH="/anvil/projects/x-cis260282/ShadeMaps"
# BASE_PATH2="/anvil/projects/x-cis260282/ShadeMaps"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"

COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

FOLD_NAMES=("phoenix" "miami" "chicago")

# ============================================================
# LOCO models (6 models: 3 folds x 2 resolutions)
# ============================================================
echo "Queueing LOCO models..."

for fold_id in 0
do
    for res in highres
    do
        holdout_city="${FOLD_NAMES[$fold_id]}"
        name="mcl_mamnet__loco_holdout_${holdout_city}__${res}"
        outputfile="${BASE_PATH}/data/mamnet/${name}.out"
        
        echo "  - LOCO fold ${fold_id} (holdout: ${holdout_city}) ${res}"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},USE_CONTRAST="--use_contrast",EVAL_TOLERANT="--eval_boundary_tolerant" \
               mamnet_mcl.sh
    done
done

echo "All jobs queued!"