#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Hardcoded directories
BASELINE_DIR="../../../data/testing_data/marine_mammels/data/codas"
DIR1="../../../data/testing_data/ablation/noise"
DIR2="../../../data/testing_data/ablation/denoised"

# List of embedding types to run
EMBEDDINGS=("clap-laion-audio" "vggish" "encodec-emb" "clap-laion-music")

# Function to run FAD comparisons
run_fad() {
    local eval_dir=$1
    local name=$2
    for embed in "${EMBEDDINGS[@]}"; do
        echo "Running FAD with $embed on $name"
        fadtk "$embed" "$BASELINE_DIR" "$eval_dir" --workers 1
        echo "----------------------------------------"
    done
}

# Run for both directories
run_fad "$DIR1" "Nose Only Audio"
run_fad "$DIR2" "Denoised Audio"
