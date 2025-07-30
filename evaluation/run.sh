#!/bin/bash

# Navigate to the directory where the script is located
cd "$(dirname "$0")"

cd burst_analysis/
bash run.sh

cd ../syn_real_div/
python run.py

cd ../scalability/
python run.py

cd ../throughput_prediction_steps/
bash run.sh

cd ../stats/
bash plot_figures_paper.sh
wait

cd ../anomly_detection_cross/
bash run.sh
