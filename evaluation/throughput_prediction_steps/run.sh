########################## CAIDA
python run_arima.py \
    --input=../../data/caida/raw.csv \
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-raw.csv 

python run_arima.py \
    --input=../../result/netshare/caida/post_processed_data/syn.csv \
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-NetShare.csv 

python run_arima.py \
    --input=../../result/e-wgan-gp/caida/syn.csv \
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-E-WGAN-GP.csv 

python run_arima.py \
    --input=../../result/stan/caida/syn.csv\
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-STAN.csv 

python run_arima.py \
    --input=../../result/realtabformer/caida/realtabformer.csv\
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-REaLTabFormer.csv 

python run_arima.py \
    --input=../../result/cascadenet/caida-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv \
    --raw=../../data/caida/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CAIDA-CascadeNet.csv 

########################## TON

python run_arima.py \
    --input=../../data/ton_iot/normal_1.csv \
    --raw=../../data/ton_iot/normal_1.csv \
    --output=../../result/evaluation/throughput_prediction_steps/TON-raw.csv 

python run_arima.py \
    --input=../../result/netshare/ton_iot/post_processed_data/syn.csv \
    --raw=../../data/ton_iot/normal_1.csv \
    --output=../../result/evaluation/throughput_prediction_steps/TON-NetShare.csv 

python run_arima.py \
    --input=../../result/e-wgan-gp/ton_iot/syn.csv \
    --raw=../../data/ton_iot/normal_1.csv \
    --output=../../result/evaluation/throughput_prediction_steps/TON-E-WGAN-GP.csv 

python run_arima.py \
    --input=../../result/realtabformer/ton/realtabformer.csv\
    --raw=../../data/ton_iot/normal_1.csv \
    --output=../../result/evaluation/throughput_prediction_steps/TON-REaLTabFormer.csv 

python run_arima.py \
    --input=../../result/cascadenet/ton-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv \
    --raw=../../data/ton_iot/normal_1.csv \
    --output=../../result/evaluation/throughput_prediction_steps/TON-CascadeNet.csv 


# ########################## CA
python run_arima.py \
    --input=../../data/ca/raw.csv \
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-raw.csv 

python run_arima.py \
    --input=../../result/netshare/ca/post_processed_data/syn.csv \
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-NetShare.csv 

python run_arima.py \
    --input=../../result/e-wgan-gp/ca/syn.csv \
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-E-WGAN-GP.csv 

python run_arima.py \
    --input=../../result/stan/ca/syn.csv\
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-STAN.csv 

python run_arima.py \
    --input=../../result/realtabformer/ca/realtabformer.csv\
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-REaLTabFormer.csv 

python run_arima.py \
    --input=../../result/cascadenet/ca-feature_True-zero_flag_True-rate_20/postprocess/syn_comp.csv  \
    --raw=../../data/ca/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/CA-CascadeNet.csv 


# ########################## DC
python run_arima.py \
    --input=../../data/dc/raw.csv \
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-raw.csv 

python run_arima.py \
    --input=../../result/netshare/dc/post_processed_data/syn.csv \
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-NetShare.csv 

python run_arima.py \
    --input=../../result/e-wgan-gp/dc/syn.csv \
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-E-WGAN-GP.csv 

python run_arima.py \
    --input=../../result/stan/dc/syn.csv \
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-STAN.csv 

python run_arima.py \
    --input=../../result/realtabformer/dc/realtabformer.csv\
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-REaLTabFormer.csv 

python run_arima.py \
    --input=../../result/cascadenet/dc-feature_True-zero_flag_True-rate_100/postprocess/syn_comp.csv \
    --raw=../../data/dc/raw.csv \
    --output=../../result/evaluation/throughput_prediction_steps/DC-CascadeNet.csv 

########################## plot
python tput_plot.py