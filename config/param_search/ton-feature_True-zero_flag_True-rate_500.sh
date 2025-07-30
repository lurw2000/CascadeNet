#!/bin/bash

date
cd ./demo
path=../config/param_search/ton-feature_True-zero_flag_True-rate_500.json

echo "Using config file: $path"
cat "$path"
echo "-----------------------------------"

echo "===== Starting training.py ====="
time python training.py --path="$path"
echo "===== Finished training.py ====="
echo "-----------------------------------"

echo "===== Starting evaluating.py ====="
time python evaluating.py --path="$path"
echo "===== Finished evaluating.py ====="
echo "-----------------------------------"

time python evaluating_timestamp_ratio.py --path=$path
time python evaluating-median_span.py --path=$path
time python evaluating-equal.py --path=$path
time python generate_copy.py --path=$path