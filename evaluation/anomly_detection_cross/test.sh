python run_netml.py \
    --raw ../../test_data/original.csv \
    --dataset test \
    --out_dir ../../test_result/evaluation/ \
    --input ../../test_data/generated.csv \
    --generator Generated \
    --ndm OCSVM
    --iters 1

################################### plot
python plot_netml.py \
  --base_dir ../../test_result/evaluation\
  --datasets test \
  --generators Generated \
  --ndms OCSVM \
