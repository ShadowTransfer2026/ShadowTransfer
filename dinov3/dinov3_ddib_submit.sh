#!/bin/bash
# FILENAME: dinov3_ddib_submit.sh
#
# Queues DINOv3+DDIB training jobs on SLURM.
#
# Run 1 — Full DDIB (C1+C2+C3):   6 jobs (3 folds x 2 res)
# Run 2 — Ablation C3-only:        6 jobs  (uncomment when needed)
# Run 3 — Ablation C1+C2:          6 jobs  (uncomment when needed)
# Run 4 — Ablation C2+C3:          6 jobs  (uncomment when needed)
# Run 5 — No-DDIB baseline:        6 jobs  (uncomment when needed)
#
# Total if all uncommented: 30 jobs

BASE_PATH="/scratch/gilbreth/mittal53/ShadeMaps/"
BASE_PATH2="/home/mittal53/ShadeMaps/"

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
#
# Args:
#   $1  TAG          — experiment tag for naming  (e.g. "ddib_C1C2C3")
#   $2  USE_C1       — 0 or 1
#   $3  USE_C2       — 0 or 1
#   $4  USE_C3       — 0 or 1
#   $5  LAMBDA_HSIC  — float  (only matters if C1=1, but always safe to pass)
#   $6  LAMBDA_DOMAIN— float
#   $7  LAMBDA_KL    — float  (only matters if C2=1)
# =====================================================================
submit_loco() {
    local TAG=$1
    local C1=$2
    local C2=$3
    local C3=$4
    local L_HSIC=$5
    local L_DOM=$6
    local L_KL=$7

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  C1=${C1}  C2=${C2}  C3=${C3}  hsic=${L_HSIC}  dom=${L_DOM}  kl=${L_KL}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="dinov3_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${BASE_PATH}/data/dinov3/${name}.out"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"
            sbatch --output=${outfile} \
                   --job-name=${name} \
                   --export=MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},WEIGHT_DIR=${WEIGHT_DIR},USE_C1=${C1},USE_C2=${C2},USE_C3=${C3},LAMBDA_HSIC=${L_HSIC},LAMBDA_DOMAIN=${L_DOM},LAMBDA_KL=${L_KL} \
                   dinov3_ddib.sh
        done
    done
}

# =====================================================================
# RUN 1: Full DDIB  (C1 + C2 + C3)
# =====================================================================
submit_loco  "ddib_C1C2C3"  1 1 1  0.1 0.01 0.001

# =====================================================================
# RUN 2: Ablation — C3 only  (Feature Augmentation)
# =====================================================================
submit_loco  "ddib_C3only"  0 0 1  0.1 0.01 0.001

# =====================================================================
# RUN 3: Ablation — C1 + C2  (Disentangle + VIB, no augmentation)
# =====================================================================
submit_loco  "ddib_C1C2"  1 1 0  0.1 0.01 0.001

# =====================================================================
# RUN 4: Ablation — C2 + C3  (VIB + Augmentation, no disentanglement)
# =====================================================================
submit_loco  "ddib_C2C3"  0 1 1  0.1 0.01 0.001

# =====================================================================
# RUN 5: No-DDIB baseline  (same decoder arch, no DDIB components)
#         Confirms DDIB decoder matches vanilla performance
# =====================================================================
submit_loco  "ddib_none"  0 0 0  0.1 0.01 0.001

echo ""
echo "All jobs queued!"