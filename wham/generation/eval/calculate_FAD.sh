#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"



# Path to baseline audio directory
BASELINE_PATH="../../../data/testing_data/marine_mammels/data/codas"

# Directories to iterate over
DIR1="../../../data/testing_data/marine_mammels/data"
DIR2="../../../data/testing_data/generated_marine_mammels"

# If an argument is passed, only run for that directory name
if [ $# -gt 0 ]; then
    # Generate Samples

    dir_name="$1"
    eval_dir1="$DIR1/$dir_name"
    eval_dir2="$DIR2/$dir_name"

    echo "Calculating FAD for: $dir_name"

    python ../generate_output_snippets.py --main.species "$dir_name" --args.load conf/interface.yml --Interface.device cuda 

    

    fadtk clap-laion-audio "$BASELINE_PATH" "$eval_dir1" --workers 1
    fadtk clap-laion-audio "$BASELINE_PATH" "$eval_dir2" --workers 1

    echo "----------------------------------------"
else
    # Generate Samples
    python ../generate_output_snippets.py --args.load conf/interface.yml --Interface.device cuda 
    # Loop over all subdirectories in DIR1
    for eval_dir1 in "$DIR1"/*; do
        eval_dir1=${eval_dir1%/}
        dir_name=$(basename "$eval_dir1")
        eval_dir2="$DIR2/$dir_name"

        echo "Calculating FAD for: $dir_name"

        fadtk clap-laion-audio "$BASELINE_PATH" "$eval_dir1" --workers 1
        fadtk clap-laion-audio "$BASELINE_PATH" "$eval_dir2" --workers 1

        echo "----------------------------------------"
    done
fi
