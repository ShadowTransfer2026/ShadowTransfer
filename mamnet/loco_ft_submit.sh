#!/bin/bash

BASE_DATA_ROOT="/scratch/gilbreth/mittal53/ShadeMaps/data/Final_data_test/"
OUTPUT_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/outputs"
SPLITS_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/splits"
LOG_DIR="/scratch/gilbreth/mittal53/ShadeMaps/data/mamnet/logs/finetune"

# Create log directory
mkdir -p ${SPLITS_DIR}
mkdir -p ${LOG_DIR}

# echo "============================================================"
# echo "Submitting Fine-tuning Experiments"
# echo "============================================================"
# echo "Cities: chicago, miami, phoenix"
# echo "Resolutions: highres, midres"
# echo "Strategies: random, clustered, dispersed"
# echo "N values: 0, 25, 50, 100, 200, 350, 450"
# echo "Seeds: 1, 2, 3, 4, 5"
# echo "Total: 3 × 2 × 3 × 7 × 5 = 630 jobs"
# echo "============================================================"
# echo ""
# chicago miami phoenix
# highres midres
# random clustered dispersed
# 0 25 50 100 200 350 450
# 1 2 3 4 5

job_count=0


# echo "============================================================"
# echo "Submitting Split Generation Jobs"
# echo "============================================================"
# echo ""

# for city in miami
# do
    # for res in highres
    # do
        
		# # N=600: Generate once per city/resolution (no seed/strategy variation)
        # # n=600
        # # strategy="original"
        # # name="split_${city}_${res}_${strategy}_N${n}"
        # # outputfile="${LOG_DIR}/${name}.out"
        
        # # sbatch --output=${outputfile} \
               # # --job-name=${name} \
               # # --export=CITY=${city},RESOLUTION=${res},N_SAMPLES=${n},STRATEGY=${strategy},SEED=1,BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${SPLITS_DIR} \
               # # loco_ft.sh
        
        # # ((job_count++))
		
		# for seed in 2
        # do
            
            # # Regular cases: N<600 - run for all strategies
            # for n in 350
            # do
                # for strategy in dispersed
                # do
					# # # Skip dispersed split generation where it uses random as fallback
                    # # # Miami: N>=350, Chicago/Phoenix: N=450
                    # # if [ "$strategy" == "dispersed" ]; then
                        # # if [ "$city" == "miami" ] && [ $n -ge 350 ]; then
                            # # continue
                        # # elif [ "$city" != "miami" ] && [ $n -eq 450 ]; then
                            # # continue
                        # # fi
                    # # fi
                    # name="split_${city}_${res}_${strategy}_N${n}_s${seed}"
                    # outputfile="${LOG_DIR}/${name}.out"
                    
                    # sbatch --output=${outputfile} \
                           # --job-name=${name} \
                           # --export=CITY=${city},RESOLUTION=${res},N_SAMPLES=${n},STRATEGY=${strategy},SEED=${seed},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${SPLITS_DIR} \
                           # loco_ft.sh
                    
                    # ((job_count++))
                    
                    # if [ $((job_count % 50)) -eq 0 ]; then
                        # echo "Queued ${job_count} jobs..."
                    # fi
                # done
            # done
        # done
    # done
# done


echo "============================================================"
echo "Submitting Fine-tuning Jobs (using pre-generated splits)"
echo "============================================================"

for city in phoenix
do
    for res in highres midres
    do
        for seed in 1 2 3 4 5
        do			
			# Special case: N=600 - use original split (shared across all strategies/seeds)
            # But create separate output directories for tracking
            n=600
            split_file="${SPLITS_DIR}/${city}_${res}_original_N$(printf "%03d" ${n}).json"
            
            for strategy in random clustered dispersed
            do
                name="ft_${city}_${res}_${strategy}_N${n}_s${seed}"
                outputfile="${LOG_DIR}/${name}.out"
                
                sbatch --output=${outputfile} \
                       --job-name=${name} \
                       --export=TARGET_CITY=${city},RESOLUTION=${res},N_SAMPLES=${n},STRATEGY=${strategy},RANDOM_SEED=${seed},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR},SPLIT_FILE=${split_file} \
                       loco_ft.sh
                
                ((job_count++))
            done
            
            # Regular cases: N<450 - each strategy has its own split
            for n in 0 25 50 100 200 350 450
            do
                for strategy in random clustered dispersed
                do
					split_file="${SPLITS_DIR}/${city}_${res}_${strategy}_N$(printf "%03d" ${n})_seed${seed}.json"

                    name="ft_${city}_${res}_${strategy}_N${n}_s${seed}"
                    outputfile="${LOG_DIR}/${name}.out"
					
					sbatch --output=${outputfile} \
                           --job-name=${name} \
                           --export=TARGET_CITY=${city},RESOLUTION=${res},N_SAMPLES=${n},STRATEGY=${strategy},RANDOM_SEED=${seed},BASE_DATA_ROOT=${BASE_DATA_ROOT},OUTPUT_DIR=${OUTPUT_DIR},SPLIT_FILE=${split_file} \
                           loco_ft.sh
                    
                    ((job_count++))
                    
                    if [ $((job_count % 50)) -eq 0 ]; then
                        echo "Queued ${job_count} jobs..."
                    fi
                done
            done
        done
    done
done




echo ""
echo "============================================================"
echo "All ${job_count} jobs queued successfully!"
echo "============================================================"