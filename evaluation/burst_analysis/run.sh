# Set paths for ton_iot synthetic data
export TON_IOT_E_WGAN_GP="../../result/e-wgan-gp/ton_iot/syn.csv"
export TON_IOT_REALTABFORMER="../../result/realtabformer/ton/realtabformer.csv"
export TON_IOT_NETSHARE="../../result/netshare/ton_iot/post_processed_data/syn.csv"
export TON_IOT_CASCADENET="../../result/cascadenet/ton-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv"

# Set paths for caida synthetic data
export CAIDA_E_WGAN_GP="../../result/e-wgan-gp/caida/syn.csv"
export CAIDA_REALTABFORMER="../../result/realtabformer/caida/realtabformer.csv"
export CAIDA_STAN="../../result/stan/caida/syn.csv"
export CAIDA_NETSHARE="../../result/netshare/caida/post_processed_data/syn.csv"
export CAIDA_CASCADENET="../../result/cascadenet/caida-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv"

# Set paths for dc synthetic data
export DC_E_WGAN_GP="../../result/e-wgan-gp/dc/syn.csv"
export DC_REALTABFORMER="../../result/realtabformer/dc/realtabformer.csv"
export DC_STAN="../../result/stan/dc/syn.csv"
export DC_NETSHARE="../../result/netshare/dc/post_processed_data/syn.csv"
export DC_CASCADENET="../../result/cascadenet/dc-feature_True-zero_flag_True-rate_100/postprocess/syn_comp.csv"

# Set paths for ca synthetic data
export CA_E_WGAN_GP="../../result/e-wgan-gp/ca/syn.csv"
export CA_REALTABFORMER="../../result/realtabformer/ca/realtabformer.csv"
export CA_STAN="../../result/stan/ca/syn.csv"
export CA_NETSHARE="../../result/netshare/ca/post_processed_data/syn.csv"
export CA_CASCADENET="../../result/cascadenet/ca-feature_True-zero_flag_True-rate_20/postprocess/syn_comp.csv"

# Confirm environment variables are set
echo "Environment variables for synthetic data paths have been set."

# Run the Python script
python burst.py
