################################## CAIDA
for ndm in OCSVM IForest AE PCA KDE; do
    python run_netml.py \
        --raw ../../data/caida/raw.csv \
        --dataset caida \
        --out_dir ../../result/evaluation/anomly_detection_cross \
        --input ../../result/cascadenet/caida-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv \
        --generator CascadeNet \
        --input ../../result/netshare/caida/post_processed_data/syn.csv \
        --generator NetShare \
        --input ../../result/realtabformer/caida/realtabformer.csv \
        --generator REaLTabFormer \
        --ndm $ndm
        --iters 1
done
################################## CA
for ndm in OCSVM IForest AE PCA KDE; do
    python run_netml.py \
        --raw ../../data/ca/raw.csv \
        --dataset ca \
        --out_dir ../../result/evaluation/anomly_detection_cross \
        --input ../../result/cascadenet/ca-feature_True-zero_flag_True-rate_50/postprocess/syn_comp.csv\
        --generator CascadeNet \
        --input ../../result/netshare/ca/post_processed_data/syn.csv \
        --generator NetShare \
        --input ../../result/realtabformer/ca/realtabformer.csv \
        --generator REaLTabFormer \
        --ndm $ndm
        --iters 1
done

################################### DC
for ndm in OCSVM IForest AE PCA KDE; do
    python run_netml.py \
        --raw ../../data/dc/raw.csv \
        --dataset dc \
        --out_dir ../../result/evaluation/anomly_detection_cross \
        --input ../../result/cascadenet/dc-feature_True-zero_flag_True-rate_2000/postprocess/syn_comp.csv \
        --generator CascadeNet \
        --input ../../result/netshare/dc/post_processed_data/syn.csv \
        --generator NetShare \
        --input ../../result/realtabformer/dc/realtabformer.csv \
        --generator REaLTabFormer \
        --ndm $ndm
        --iters 1
done

################################### plot
python plot_netml.py \
  --datasets caida ca dc \