#!/bin/bash

# From https://stackoverflow.com/a/246128
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# As script 'test_full_pipeline.sh' is in base git root dir,
base_root_dir=$SCRIPT_DIR

# Fetch and tidy sample 100 regular season games from 2020-21 season
python ${base_root_dir}/ift6758/fetch_and_tidy.py -t 2 -y 2020 -g 1-100
# Generate augmented data from all downloaded games
python ${base_root_dir}/ift6758/feature_engineering/pipeline_complex_features.py
# Run simple model from augmented data and log to WandB
python ${base_root_dir}/ift6758/simple_model/base_model.py
# Run advanced model from augmented data and log to WandB
python ${base_root_dir}/ift6758/advanced_models/decisiontrees/decisiontrees.py
