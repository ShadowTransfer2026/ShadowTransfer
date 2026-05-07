#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/dinov3/outputs"
WEIGHT_DIR="/home/mittal53/ShadeMaps/python/dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# ============================================================
# PART 1: Train individual city models (6 models)
# ============================================================
echo "Queueing individual city models..."
for city in chicago miami phoenix
do
    res="highres"
	name="hrda_dinov3__${city}__${res}"
	outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/dinov3/${name}.out"
	data_root="${BASE_DATA_ROOT}${city}/${res}/"
	
	echo "  - ${city} ${res} (single mode)"
	sbatch --output=${outputfile} \
		   --job-name=${name} \
		   --export=MODE=single,DATA_ROOT=${data_root},RES=${res},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR} \
		   dinov3_hrda.sh

	res="midres"
	name="hrda_dinov3__${city}__${res}"
	outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/dinov3/${name}.out"
	data_root="${BASE_DATA_ROOT}${city}/${res}/"
	
	echo "  - ${city} ${res} (single mode)"
	sbatch --output=${outputfile} \
		   --job-name=${name} \
		   --export=MODE=single,DATA_ROOT=${data_root},RES=${res},OUTPUT_DIR=${OUTPUT_DIR},WEIGHT_DIR=${WEIGHT_DIR} \
		   dinov3_hrda.sh

done

echo "All jobs queued!"