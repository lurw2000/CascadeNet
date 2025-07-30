python run_arima.py \
    --input=../../test_data/original.csv \
    --raw=../../test_data/original.csv \
    --output=../../test_result/evaluation/throughput_test-raw.csv 

python run_arima.py \
    --input=../../test_data/generated.csv \
    --raw=../../test_data/original.csv \
    --output=../../test_result/evaluation/throughput_test-Generated.csv 

########################## plot
python tput_plot.py --test