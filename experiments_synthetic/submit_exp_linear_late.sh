#!/bin/bash

# Parameters
SEED_LIST=$(seq 1 100)
#SEED_LIST=$(seq 1 2)
EXTRACTOR_LIST=("separate")
DATASET="regression"
EXPNAME="linear_late"

# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-01:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --partition=biodatascience.p"

# Create directory for log files
LOGS="logs/"$DATASET"_"$EXPNAME
OUT_DIR="results/"$DATASET"_"$EXPNAME

mkdir -p $LOGS

comp=0
incomp=0

mkdir -p $OUT_DIR

for SEED in $SEED_LIST; do
    for EXTRACTOR in "${EXTRACTOR_LIST[@]}"; do
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
        SCRIPT="exp_linear_late.sh $EXTRACTOR $SEED"
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

echo "Jobs already completed: $comp, submitted unfinished jobs: $incomp"
