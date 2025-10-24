#!/bin/bash

# Parameters
#EXTRACTOR_LIST=("PCA" "encoder" "separate")
EXTRACTOR_LIST=("encoder")
SEED_LIST=$(seq 1 100)
#SEED_LIST=$(seq 1 2)
DATASET="NACC"
FINE_GRAINED=1

# Slurm parameters
MEMO=1G                             # Memory required (1 GB)
TIME=00-03:00:00                    # Time required (2 h)
CORE=1                              # Cores required (1)

# Assemble order
ORDP="sbatch --mem="$MEMO" --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$TIME" --partition=biodatascience.p"

# Create directory for log files
if [ "$FINE_GRAINED" -eq 1 ]; then
    LOGS="logs/"$DATASET"_fine_grained"
    OUT_DIR="results/"$DATASET"_fine_grained"
else
    LOGS="logs/"$DATASET
    OUT_DIR="results/"$DATASET
fi

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
        SCRIPT="exp_nacc.sh $EXTRACTOR $FINE_GRAINED $SEED"
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
