#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"

# ============================================================
# PART 1: Train individual city models (6 models)
# ============================================================
# echo "Queueing individual city models..."
# for city in chicago miami phoenix
# do
	# res="midres"
    # name="hrda_mamnet__${city}__${res}"
	# outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/${name}.out"
	# data_root="${BASE_DATA_ROOT}${city}/${res}/"
	
	# echo "  - ${city} ${res} (single mode)"
	# sbatch --output=${outputfile} \
		   # --job-name=${name} \
		   # --export=MODE=single,DATA_ROOT=${data_root},RES=${res},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST="--use_contrast",EVAL_TOLERANT="--eval_boundary_tolerant",AUTO_LOAD_PRETRAINED="--auto_load_pretrained" \
		   # mamnet_hrda.sh
		   
	# res="highres"
    # name="hrda_mamnet__${city}__${res}"
	# outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/${name}.out"
	# data_root="${BASE_DATA_ROOT}${city}/${res}/"
	
	# echo "  - ${city} ${res} (single mode)"
	# sbatch --output=${outputfile} \
		   # --job-name=${name} \
		   # --export=MODE=single,DATA_ROOT=${data_root},RES=${res},OUTPUT_DIR=${OUTPUT_DIR},USE_CONTRAST="--use_contrast",EVAL_TOLERANT="--eval_boundary_tolerant",AUTO_LOAD_PRETRAINED="--auto_load_pretrained" \
		   # mamnet_hrda.sh
# done

for city in chicago
do
	res="highres"
	
	for task_id in 6 8
	do
		name="hrda_mamnet__${city}__${res}__task${task_id}"
		outputfile="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/${name}.out"
		data_root="${BASE_DATA_ROOT}${city}/${res}/"
		
		echo "  - ${city} ${res} task_id=${task_id}"
		sbatch --output=${outputfile} \
			   --job-name=${name} \
			   --export=MODE=single,DATA_ROOT=${data_root},RES=${res},OUTPUT_DIR=${OUTPUT_DIR},TASK_ID=${task_id},USE_CONTRAST="--use_contrast",EVAL_TOLERANT="--eval_boundary_tolerant",AUTO_LOAD_PRETRAINED="--auto_load_pretrained" \
			   mamnet_hrda.sh
	done
done