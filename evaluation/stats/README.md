This folder is a collection of features we can extract from raw or synthetic network trace using simple computations. Below is the folder structure

* `task/`: various features we want to extract

	* `task/trace/`: Time series aggregated from network trace

	* `task/statisical_feature/`: Contains statistical analysis tools

		This includes the pdf of the distribution of statistical characteristics, and the distance between raw distribution and various synthetic distributions.

	* `task/temporal_feature/`: Focuses on temporal characteristics
	
		This includes the pdf of the distribution of temporal characteristics, and the distance between raw distribution and various synthetic distributions.

* `config/`

	Configurations for plotting figures. Config files are in yaml format, which is compatible with json and has comments. There are two types of configs:
	
	* dataset-specific config. For example, `config/test/test.yaml`

	* general config at `config/general.yaml`.

* `util/`

	Utilities for stats tasks. For now, it contains the argparser that are shared by all stats tasks. This argparser initialize the necessary global variables for the tasks.

## Tutorial

To run a provided stats task with different dataset, you need to

1. create a dataset-specific config file at `config/[folder_name]/[config_name].yaml`

2. run the stats task at `task/[folder_name]/[task_name].py` with the `--config` argument.

As an example, let's run `taks/trace/timeseries.py` for the `caida_mini` dataset (first 10k packets of the CAIDA trace). This stats task plots the trace-level packet rate time series. The config file is at `config/test/test.yaml`. You should modify the path in that config file accordingly.

Suppose the current directory is the same as this README file, then we can run

```bash
python task/trace/timeseries.py --config "config/test/test.yaml"
```

You could also run all tasks by running the bash script

```bash
bash ./plot_figures_paper.sh
```