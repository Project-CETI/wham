#!/bin/bash

# Navigate to the script directory (optional, modify if needed)
SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

python testing_data/impulses/create_impulses.py