#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: bash run.sh <dataset>"
    echo "Allowed datasets: caida, ca, dc, ton_iot"
    exit 1
fi

python train.py --dataset "$1"
