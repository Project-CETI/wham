#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Hardcoded embedding directories
NOISE_DIRECTORY="../../../data/testing_data/embeddings/noise"
DENOISED_DIRECTORY="../../../data/testing_data/embeddings/denonised"

echo "Distance between noise profile and raw codas:"
python calculate_fad_metric.py --evaluation_dir "$NOISE_DIRECTORY"

echo "------------------------------"

echo "Distance between denoised-codas and raw codas:"

python calculate_fad_metric.py --evaluation_dir "$DENOISED_DIRECTORY"

