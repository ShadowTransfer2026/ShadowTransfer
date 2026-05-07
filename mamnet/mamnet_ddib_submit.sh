#!/bin/bash
# FILENAME: mamnet_ddib_submit.sh
#
# Queues MAMNet+DDIB training jobs on SLURM.
#
# Run 1 — Full DDIB (C1+C2+C3):   6 jobs (3 folds x 2 res)
# Run 2 — Ablation C3-only:        6 jobs
# Run 3 — Ablation C1+C2:          6 jobs
# Run 4 — Ablation C2+C3:          6 jobs
# Run 5 — No-DDIB baseline:        6 jobs
# Run 6 — Full DDIB, no skip filters (Option A):    6 jobs
# Run 7 — Skip filters only (no DDIB components):   6 jobs
#
# Uncomment the runs you need.

# ---- Server paths (uncomment the one you need) ----

# --- Gilbreth ---
BASE_PATH="/scratch/gilbreth/mittal53/ShadeMaps"
BASE_PATH2="/home/mittal53/ShadeMaps"

# --- Anvil ---
BASE_PATH="/anvil/projects/x-cis260282/ShadeMaps"
BASE_PATH2="/anvil/projects/x-cis260282/ShadeMaps"

BASE_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"
OUTPUT_DIR="${BASE_PATH}/data/mamnet/outputs"

COMPARISON_INFERENCE_DIR="${BASE_PATH}/data/Test_img_results/"
COMPARISON_DATA_ROOT="${BASE_PATH}/data/Final_data_test/"

FOLD_NAMES=("phoenix" "miami" "chicago")

# =====================================================================
# Helper: submit one LOCO configuration across folds and resolutions
#
# Args:
#   $1  TAG            — experiment tag (e.g. "ddib_C1C2C3_SF")
#   $2  USE_C1         — 0 or 1
#   $3  USE_C2         — 0 or 1
#   $4  USE_C3         — 0 or 1
#   $5  USE_SF         — 0 or 1  (skip filters)
#   $6  LAMBDA_HSIC    — float
#   $7  LAMBDA_DOMAIN  — float
#   $8  LAMBDA_KL      — float
#   $9  USE_CONTRAST   — 0 or 1
# =====================================================================
submit_loco() {
    local TAG=$1
    local C1=$2
    local C2=$3
    local C3=$4
    local SF=$5
    local L_HSIC=$6
    local L_DOM=$7
    local L_KL=$8
    local CONTRAST=$9

    echo ""
    echo "===== Queueing: ${TAG} (LOCO) ====="
    echo "  C1=${C1}  C2=${C2}  C3=${C3}  SF=${SF}  contrast=${CONTRAST}"
    echo "  hsic=${L_HSIC}  dom=${L_DOM}  kl=${L_KL}"

    for fold_id in 0 1 2; do
        for res in highres; do
            holdout="${FOLD_NAMES[$fold_id]}"
            name="mamnet_${TAG}__loco_holdout_${holdout}__${res}"
            outfile="${BASE_PATH}/data/mamnet/${name}.out"

            echo "  - fold=${fold_id} (holdout: ${holdout})  res=${res}"

            sbatch --output=${outfile} \
                   --job-name=${name} \
                   --export=MODE=loco,BASE_DATA_ROOT=${BASE_DATA_ROOT},RESOLUTION=${res},FOLD_ID=${fold_id},OUTPUT_DIR=${OUTPUT_DIR},COMPARISON_INFERENCE_DIR=${COMPARISON_INFERENCE_DIR},COMPARISON_DATA_ROOT=${COMPARISON_DATA_ROOT},USE_C1=${C1},USE_C2=${C2},USE_C3=${C3},USE_SF=${SF},LAMBDA_HSIC=${L_HSIC},LAMBDA_DOMAIN=${L_DOM},LAMBDA_KL=${L_KL},USE_CONTRAST=${CONTRAST} \
                   mamnet_ddib.sh
        done
    done
}

# =====================================================================
# RUN 1: Full DDIB + Skip Filters  (C1 + C2 + C3 + SF)
# =====================================================================
# submit_loco  "ddib_C1C2C3_SF"  1 1 1 1  0.1 0.01 0.001  1

# =====================================================================
# RUN 2: Ablation — C3 + SF only  (Feature Augmentation + Skip Filters)
# =====================================================================
# submit_loco  "ddib_C3_SF"  0 0 1 1  0.1 0.01 0.001  1

# =====================================================================
# RUN 3: Ablation — C1 + C2 + SF  (Disentangle + VIB + Skip Filters)
# =====================================================================
# submit_loco  "ddib_C1C2_SF"  1 1 0 1  0.1 0.01 0.001  1

# =====================================================================
# RUN 4: Ablation — C2 + C3 + SF  (VIB + Augmentation + Skip Filters)
# =====================================================================
submit_loco  "ddib_C2C3_SF"  0 1 1 1  0.1 0.01 0.001  1

# =====================================================================
# RUN 5: No-DDIB baseline  (no DDIB, no skip filters)
# =====================================================================
submit_loco  "ddib_none"  0 0 0 0  0.1 0.01 0.001  1

# =====================================================================
# RUN 6: Option A equivalent — Full DDIB, NO skip filters
# =====================================================================
submit_loco  "ddib_C1C2C3_noSF"  1 1 1 0  0.1 0.01 0.001  1

# =====================================================================
# RUN 7: Skip filters only — no DDIB components
#         (tests skip filtering in isolation)
# =====================================================================
submit_loco  "ddib_SFonly"  0 0 0 1  0.1 0.01 0.001  1

echo ""
echo "All jobs queued!"