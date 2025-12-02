#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

BASELINE_DIR="../../../data/testing_data/marine_mammels/data/codas"
DESTINATION_DIR="../../../data/training_data/regenerated_codas"

python ../tokenizer_only_generation.py --args.load conf/interface.yml --Interface.device cuda 

python mse_error_by_freq.py