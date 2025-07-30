################################ diff_syn -- figure 6, 7, 12, 13
python task/temporal_feature/pdf.py --config 'config/paper_plot/diff_syn/ton_iot.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/diff_syn/dc.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/diff_syn/ca.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/diff_syn/caida.yaml' &

python task/temporal_feature/dist.py --config 'config/paper_plot/diff_syn/ton_iot.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/diff_syn/dc.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/diff_syn/ca.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/diff_syn/caida.yaml' &


python task/statistical_feature/pdf.py --config 'config/paper_plot/diff_syn/caida.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/diff_syn/dc.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/diff_syn/ca.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/diff_syn/ton_iot.yaml' &

python task/statistical_feature/dist.py --config 'config/paper_plot/diff_syn/caida.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/diff_syn/dc.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/diff_syn/ca.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/diff_syn/ton_iot.yaml' &



################################# trace pkt_rate for diff_syn -- figure 2, 23
python task/trace/timeseries.py --config 'config/paper_plot/diff_syn/caida.yaml' &
python task/trace/timeseries.py --config 'config/paper_plot/diff_syn/ca.yaml' &
python task/trace/timeseries.py --config 'config/paper_plot/diff_syn/dc.yaml' &
python task/trace/timeseries.py --config 'config/paper_plot/diff_syn/ton_iot.yaml' &


wait
echo "diff_syn processing completed. normalization.csv should now be available."

echo "Starting remaining tasks..."


################################# ablation (of optimization) -- figure 18
python task/temporal_feature/pdf.py --config 'config/paper_plot/ablation/ton_iot.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/ablation/dc.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/ablation/ca.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/ablation/caida.yaml' &

python task/temporal_feature/dist.py --config 'config/paper_plot/ablation/ton_iot.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/ablation/dc.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/ablation/ca.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/ablation/caida.yaml' &



################################# time_series_length -- figure 21, 22
python task/temporal_feature/pdf.py --config 'config/paper_plot/time_series_length/ton_iot.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/time_series_length/dc.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/time_series_length/ca.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/time_series_length/caida.yaml' &

python task/temporal_feature/dist.py --config 'config/paper_plot/time_series_length/ton_iot.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/time_series_length/dc.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/time_series_length/ca.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/time_series_length/caida.yaml' &


python task/statistical_feature/pdf.py --config 'config/paper_plot/time_series_length/caida.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/time_series_length/dc.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/time_series_length/ca.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/time_series_length/ton_iot.yaml' &

python task/statistical_feature/dist.py --config 'config/paper_plot/time_series_length/caida.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/time_series_length/dc.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/time_series_length/ca.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/time_series_length/ton_iot.yaml' &




################################ timestamp_recover -- figure 17
python task/temporal_feature/pdf.py --config 'config/paper_plot/timestamp_recover/ton_iot.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/timestamp_recover/dc.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/timestamp_recover/ca.yaml' &
python task/temporal_feature/pdf.py --config 'config/paper_plot/timestamp_recover/caida.yaml' &

python task/temporal_feature/dist.py --config 'config/paper_plot/timestamp_recover/ton_iot.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/timestamp_recover/dc.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/timestamp_recover/ca.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/timestamp_recover/caida.yaml' &




############################### cascadenet_test_structure -- figure 16
python task/temporal_feature/pdf.py --config 'config/paper_plot/cascadenet_test_structure/caida.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/cascadenet_test_structure/caida.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/cascadenet_test_structure/caida.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/cascadenet_test_structure/caida.yaml' &
python task/trace/timeseries.py --config 'config/paper_plot/cascadenet_test_structure/caida.yaml' &




################################ cascadenet_test_cond (ablation of conditional input) -- figure 19
python task/temporal_feature/pdf.py --config 'config/paper_plot/cascadenet_test_cond/caida.yaml' &
python task/temporal_feature/dist.py --config 'config/paper_plot/cascadenet_test_cond/caida.yaml' &
python task/statistical_feature/pdf.py --config 'config/paper_plot/cascadenet_test_cond/caida.yaml' &
python task/statistical_feature/dist.py --config 'config/paper_plot/cascadenet_test_cond/caida.yaml' &
python task/trace/timeseries.py --config 'config/paper_plot/cascadenet_test_cond/caida.yaml' &



################################ time_series_length comparsion -- figure 20
python score_cal.py


wait
echo "All background jobs have completed."