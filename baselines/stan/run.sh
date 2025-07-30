#!/bin/bash

# Check if a dataset name is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

DATASET=$1

echo "Running sample_100.py..."
python sample_100.py --data "$DATASET"

echo "Running test_train.py..."
python test_train.py --data "$DATASET"

echo "Running test_sample.py..."
python test_sample.py --data "$DATASET"

echo "Running reorder.py..."
python reorder.py --data "$DATASET"

echo "All processes completed!"
