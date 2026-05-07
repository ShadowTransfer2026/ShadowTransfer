#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/oglanet/outputs"

# ============================================================
# PART 1: Train individual city models (6 models)
# ============================================================
echo "Queueing individual city models..."
for city in chicago miami phoenix
do
    for res in highres midres
    do
        name="gsdpe_oglanet__${city}__${res}"
        outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/oglanet/${name}.out"
        data_root="${BASE_DATA_ROOT}${city}/${res}/"
        
        echo "  - ${city} ${res} (single mode)"
        sbatch --output=${outputfile} \
               --job-name=${name} \
               --export=MODE=single,DATA_ROOT=${data_root},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST="use_contrast",EVAL_TOLERANT="eval_boundary_tolerant" \
               oglanet_gsdpe.sh
    done
done

echo "All jobs queued!"