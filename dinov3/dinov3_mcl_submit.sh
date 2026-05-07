#!/bin/bash
# FILENAME: dinov3_mcl_submit.sh
#
# Queues DINOv3 + mCL-LC training jobs on SLURM.
#
# LOCO folds × resolutions = 6 jobs (3 folds × 2 res)
# (Uncomment/comment resolution loop entries to run subsets)

# ---- Cluster paths (uncomment ONE pair) ----
# --- Gilbreth ---
# BASE_PATH="/scratch/gilbreth/mittal53/ShadeMaps/"
# BASE_PATH2="/home/mittal53/ShadeMaps/"

# --- Anvil ---
BASE_PATH="/anvil/projects/x-cis260282/ShadeMaps/"
BASE_PATH2="/anvil/projects/x-cis260282/ShadeMaps/"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/dinov3/outputs"
WEIGHT_DIR="${BASE_PATH2}/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

FOLD_NAMES=("phoenix" "miami" "chicago")

# =====================================================================
# Helper: submit one LOCO configuration across all folds and resolutions
# =====================================================================
submit_loco() {
    echo ""
    echo "===== Queueing: DINOv3 mCL-LC (LOCO) ====="

    for fold_id in 0 1 2; do
        for res in midres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="mcl_dinov3__loco_holdout_${holdout}__${res}"

            # --- Gilbreth ---
            # outfile="/scratch/gilbreth/mittal53/ShadeMaps/data/dinov3/${name}.out"

            # --- Anvil ---
            outfile="${BASE_PATH}/data/dinov3/${name}.out"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"
            sbatch --output=${outfile} \
                   --job-name=${name} \
                   --export=MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT} \
                   dinov3_mcl.sh
        done
    done
}

# =====================================================================
# Queue jobs
# =====================================================================
submit_loco

echo ""
echo "All jobs queued!"