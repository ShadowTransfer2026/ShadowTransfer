#!/bin/bash



BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/oglanet/outputs"
LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/oglanet/logs/attributes"

mkdir -p ${LOG_DIR}

# Create descriptive job name

name="inference"

# for city in chicago, miami, phoenix; do
  # for res in highres midres; do
    # echo "Running inference: $city / $res"
	# outputfile="${LOG_DIR}/${name}_${city}_${res}.out"

    # sbatch --output=${outputfile} \
	   # --job-name="${name}_${city}_${res}" \
	   # inference.sh
  # done
# done


# Submit jobs for all cities and resolutions
for city in chicago miami phoenix; do
  for target_res in highres midres; do
    # Determine source resolution (opposite of target)
    if [ "$target_res" == "highres" ]; then
      source_res="midres"
    else
      source_res="highres"
    fi
    
    echo "Submitting job: city=$city, target=$target_res, source=$source_res"
    outputfile="${LOG_DIR}/${name}_${city}_${target_res}.out"
    
    sbatch --output=${outputfile} \
           --job-name="${name}_${city}_${target_res}" \
           --export=ALL,EVAL_TYPE=all,CITY=$city,TARGET_RES=$target_res,SOURCE_RES=$source_res,PYTHON_BIN=$PYTHON_BIN \
           inference.sh
  done
done

echo "All jobs submitted! Check logs in ${LOG_DIR}/"