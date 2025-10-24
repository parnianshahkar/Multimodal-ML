#!/bin/bash

# Parameters

#EXTRACTOR_LIST=("PCA" "encoder")
EXTRACTOR_LIST=("encoder")
#EXTRACTOR_LIST=("PCA")
SEED_LIST=$(seq 1 100)
#SEED_LIST=$(seq 1 2)
#RAT_LIST=("Barat" "Buchanan" "Mitt" "Stella" "Superchris")
RAT_LIST=("Barat")
DATASET="Neuron"

# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-03:30:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --partition=biodatascience.p"

comp=0
incomp=0

for SEED in $SEED_LIST; do
    for RAT in "${RAT_LIST[@]}"; do
        for EXTRACTOR in "${EXTRACTOR_LIST[@]}"; do

            LOGS="logs/"$DATASET"/"$RAT
            OUT_DIR="results/"$DATASET"/"$RAT

            mkdir -p $LOGS
            mkdir -p $OUT_DIR

            JOBN=$EXTRACTOR"_seed"$SEED
            OUT_FILE=$OUT_DIR"/"$JOBN".txt"
            COMPLETE=0
            #ls $OUT_FILE
            if [[ -f $OUT_FILE ]]; then
            COMPLETE=1
            ((comp++))
            fi

            if [[ $COMPLETE -eq 0 ]]; then
            ((incomp++))
            # Script to be run
            SCRIPT="exp_neuron.sh $RAT $EXTRACTOR $SEED"
            # Define job name
            OUTF=$LOGS"/"$JOBN".out"
            ERRF=$LOGS"/"$JOBN".err"
            # Assemble slurm order for this job
            ORD=$ORDP" -J "$JOBN" -o "$OUTF" -e "$ERRF" "$SCRIPT
            # Print order
            echo $ORD
            # Submit order
            $ORD
            fi
        done
    done
done

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
