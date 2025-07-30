
"""
This script plots the distribution of temporal features of different datasets.
"""
import matplotlib.pyplot as plt
import os 
import numpy as np
import pickle
import json

import nta.utils.vis as vis 
import nta.utils.stats as stats
import nta.utils.data as data

"""
Configuration
"""
from evaluation.stats.util.argparser import *

## We may want to use a perturbed raw trace as reference
use_perturb_ref = False
perturb_coeff = 0.05

bins_count = 100

general_field_configs = {
	# "flow_level_interarrival": {
	# 	"field_name": "flow_level_interarrival",
	# 	"extractor": lambda df, dfg, gks, flows: stats.dfg2flow_level_interarrival(dfg, gks),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 	# 	x, ax, 
	# 	# 	bins_count=100, 
	# 	# 	# bins_min=1e-8, bins_max=1e-2,
	# 	# 	density=True, **kwargs),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_hist(
	# 	"plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 		x, ax,
	# 		bins_count=bins_count,
	# 		# bins_min=1e-7, bins_max=1e0,
	# 		density=True, cumsum=True, 
	# 		**kwargs),
	# 	"format": {
	# 		# "xlabel": "flow-level interarrival",
	# 		# "ylabel": "density",
	# 		"title": "Flow-level Interarrival",
	# 	}
	# },
	# "max_packetrate": {
	# 	"field_name": "max packetrate", 
	# 	"extractor": lambda df, dfg, gks, flows: flows.max(axis=0),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_hist(
	# 	"plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 		x, ax,
	# 		bins_count=bins_count, 
	# 		density=True, cumsum=True, 
	# 		**kwargs),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 	# 	x, ax,
	# 		# density=True, cumsum=True, 
	# 		# **kwargs),
	# 	"format": {
	# 		# "xlabel": "max packetrate",
	# 		# "ylabel": "density",
	# 		"title": "Max Packetrate",
	# 		# "yscale": "log",
	# 	}
	# },

	"flowsize": {
		"field_name": "flowsize",
		"extractor": lambda df, dfg, gks, flows: flows.sum(axis=0),
		"plotter": lambda x, ax, **kwargs: vis.plot_hist(
		# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
			x, ax,
			bins_count=bins_count, 
			bins_max=50000,
			# bins_min=global_bins_min["flowsize"],
    		# bins_max=global_bins_max["flowsize"],
			density=True, 
			cumsum=True,
			log_bins=True,
			**kwargs),
		# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
		# 	x, ax,
		# 	bins_count=100, 
			# density=True, cumsum=True, 
			# **kwargs),
		"format": {
			"xlabel": "Flow Size (#records)",
			"ylabel": "CDF",
			"title": "Flow Size",
			"yscale": "log",
		}
	},

	# "flow_duration": {
	# 	"field_name": "flow_duration",
	# 	"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["time"].max() - dfg.get_group(gk)["time"].min() for gk in gks]),
	# 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 		x, ax,
	# 		bins_count=bins_count, 
	# 		density=True, cumsum=False, 
	# 		**kwargs),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 	# 	x, ax,
	# 	# 	bins_count=100, 
	# 		# density=True, cumsum=True, 
	# 		# **kwargs),
	# 	"format": {
	# 		"xlabel": "Flow Duration (s)",
	# 		"ylabel": "Density",
	# 		"title": "Flow Duration",
	# 		# "yscale": "log",
	# 	}
	# },

	"trace_level_interarrival": {
        "field_name": "trace_level_interarrival",
        "extractor": lambda df: np.diff(df["time"]),
        # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
        # 	x, ax, 
        # 	bins_count=100, bins_min=1e0, bins_max=1e7,
        # 	density=True, **kwargs), 
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax,
            bins_count=100, 
            # bins_min=0, bins_max=1e7,
            density=True, **kwargs),
        "format": {
            "xlabel": "trace-level interarrival",
            "ylabel": "density",
            "title": "Trace-level Interarrival",
        }
    },
}

pcap_field_configs = {
	"byte_count": {
		"field_name": "byte_count",
		"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["pkt_len"].sum() for gk in gks]),
		"plotter": lambda x, ax, **kwargs: vis.plot_hist(
		# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
			x, ax,
			bins_count=bins_count, 
			# bins_min=global_bins_min["byte_count"],
    		# bins_max=global_bins_max["byte_count"],
			density=True, cumsum=True,
			# xlim=(23, 5e6), 
			log_bins=True,
			**kwargs),
		# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
		# 	x, ax,
		# 	bins_count=100, 
			# density=True, cumsum=True, 
			# **kwargs),
		"format": {
			"xlabel": "byte count",
			"ylabel": "density",
			"title": "Byte Count",
			"yscale": "log",
		}
	},
}

netflow_field_configs = {
	# "byte_count": {
	# 	"field_name": "byte_count",
	# 	"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["td"].sum() for gk in gks]),
	# 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 		x, ax,
	# 		bins_count=bins_count, 
	# 		density=True, cumsum=True, 
	# 		**kwargs),
	# 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
	# 	# 	x, ax,
	# 	# 	bins_count=100, 
	# 		# density=True, cumsum=True, 
	# 		# **kwargs),
	# 	"format": {
	# 		# "xlabel": "byte count",
	# 		# "ylabel": "density",
	# 		"title": "Byte Count",
	# 		# "yscale": "log",
	# 	}
	# },
}


time_point_count = 400

flow_tuple = stats.five_tuple
# flow_tuple = [flow_tuple[0], flow_tuple[2]]

flow_filter = {
    "flowsize_range": None,
	# "flowsize_range": 0.01,
	# "flowsize_range": (5, np.inf),
    "flowDurationRatio_range": None,
	# "flowDurationRatio_range": 0.01,
    "nonzeroIntervalCount_range": None,
    "maxPacketrate_range": None,
    # "maxPacketrate_range": (100, np.inf),
}

# style
label_size = 28
title_size = 2
tick_size = 28
legend_size = 28
figsize = (36, 12)

# Get the absolute path to the script
script_path = os.path.abspath(__file__)

# Go up to ML-testing-dev directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path)))))

# Create the result directory path
if test:
    save_config = {
        "folder": os.path.join(project_root, "test_result", "evaluation"),
        "filename": "temporal_features_distribution_test",
        "format": "pdf",
    }
else:
	save_config = {
		"folder": os.path.join(project_root, "result", "evaluation", "stats", folder, dataset),
		"filename": "temporal_features_distribution",
		"format": "pdf",
	}

if use_perturb_ref:
    label2path["raw_perturb"] = label2path["raw"]

field_configs = { **general_field_configs }
if dataset_type == DATASET_TYPE.PCAP:
    field_configs = { **field_configs, **pcap_field_configs }
elif dataset_type == DATASET_TYPE.NETFLOW:
    field_configs = { **field_configs, **netflow_field_configs }
else:
    raise ValueError("Unknown type of dataset: '{}'. Not pcap or netflow".format(dataset))

labels = list(label2path.keys())
fields = list(field_configs.keys())

label_filter = {}

if dataset in ['ca', 'caida', 'dc'] and folder == 'diff_syn':
	# Define label_filter
	label_filter = {
		# For CA, CAIDA, DC
		"flowsize": ["raw", "E-WGAN-GP", "REaLTabFormer", "NetShare", "CascadeNet"],
		"byte_count": ["raw", "E-WGAN-GP", "REaLTabFormer", "NetShare", "CascadeNet"],
		"trace_level_interarrival": ["raw", "CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
	}
elif dataset == 'ton_iot' and folder == 'diff_syn':
	label_filter = {
		# For TON
		"flowsize": ["raw", "E-WGAN-GP", "REaLTabFormer", "NetDiffusion", "NetShare", "CascadeNet"],
		"byte_count": ["raw", "E-WGAN-GP", "REaLTabFormer", "NetDiffusion", "NetShare", "CascadeNet"],
		"trace_level_interarrival": ["raw", "CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetDiffusion", "NetShare", "CascadeNet"],
	}
elif folder == 'ablation':
    label_filter = {
        "flowsize": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "byte_count": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "trace_level_interarrival": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
    }
elif folder == 'time_series_length':
    label_filter = {
		"flowsize": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "byte_count": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "trace_level_interarrival": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
    }
elif folder == 'timestamp_recover':
    label_filter = {
        "flowsize": ["EQ", "ML", "SP"],
        "byte_count": ["EQ", "ML", "SP"],
		"trace_level_interarrival": ["EQ", "ML", "SP"],
    }

# Default label_filter to include all available labels for any fields without specific filters
for field in fields:
    if field not in label_filter:
        label_filter[field] = labels

field_label_data_path = os.path.join(
    save_config["folder"],
    "{}_{}.pkl".format(save_config["filename"], dataset)
)
if not os.path.exists(field_label_data_path) or overwrite_history_result:
    field_label_data = {}
    for field in field_configs:
        labels_for_field = label_filter.get(field, labels)
        field_label_data[field] = dict.fromkeys(labels_for_field, -1)

    for label in labels:
        df = data.load_csv(
            path=label2path[label],
            verbose=False,
            need_divide_1e6="auto",
            unify_timestamp_fieldname=True,
        )

        # Process "trace_level_interarrival" separately
        if "trace_level_interarrival" in field_configs and label in field_label_data["trace_level_interarrival"]:
            field_name = "trace_level_interarrival"
            field_config = field_configs[field_name]
            field = field_config["extractor"](df)
            field_label_data[field_name][label] = field

        # Adjust time
        df["time"] = df["time"] - df["time"].min()

        # Process flow-level features if applicable
        labels_for_flow_features = [field for field in field_configs if field != "trace_level_interarrival"]
        process_flow_features = any(label in field_label_data[field] for field in labels_for_flow_features)
        if process_flow_features:
            dfg, gks, flows = data.load_flow_from_df(
                df,
                mode=("time_point_count", time_point_count),
                flow_tuple=flow_tuple,
                **flow_filter,
                return_all=True,
            )
            flows = flows.astype(int)

            # Check if flows are empty
            if flows.size == 0:
                print(f"No flows found for label {label}, skipping flow-level features.")
                continue

            for field_name in labels_for_flow_features:
                if label not in field_label_data[field_name]:
                    continue
                field_config = field_configs[field_name]
                field = field_config["extractor"](df, dfg, gks, flows)
                if use_perturb_ref and label == "raw_perturb":
                    field = field + np.random.normal(0, perturb_coeff * field.std(), field.shape)
                field_label_data[field_name][label] = field

    # Save as pickle for later use
    os.makedirs(os.path.dirname(field_label_data_path), exist_ok=True)
    with open(field_label_data_path, "wb") as f:
        pickle.dump(field_label_data, f)
else:
    with open(field_label_data_path, "rb") as f:
        field_label_data = pickle.load(f)

fig, axes = vis.subplots(
	1, len(field_configs), 
	# figsize=figsize, 
	sharex=False, sharey=False)
fig.set_figwidth(figsize[0])
fig.set_figheight(figsize[1])

for label_idx, label in enumerate(label2path):
	df = data.load_csv(
		path=label2path[label],
		verbose=False,
		need_divide_1e6="auto",
	)
	df["time"] = df["time"] - df["time"].min()	

	dfg, gks, flows = data.load_flow_from_df(
		df,
		mode = ("time_point_count", time_point_count),
		flow_tuple = flow_tuple,
		**flow_filter,
		return_all = True,
		)
	flows = flows.astype(int)

	for i, field_name in enumerate(field_configs):
		# extract field
		if field_name == "trace_level_interarrival":
			field_config = field_configs[field_name]
			field = field_config["extractor"](df)
			continue
		field_config = field_configs[field_name]
		field = field_config["extractor"](df, dfg, gks, flows)

		# Debugging: Print the extracted flow sizes
		if field_name == "flowsize":
			print(f"Extracted flow sizes for {label}: {field}")
			print(f"Statistics for {label}: min={np.min(field)}, max={np.max(field)}, mean={np.mean(field)}, median={np.median(field)}")

		# plot field
		field_config["plotter"](
			field, axes[i], 
			label=label, 
			alpha=0.5,
			histtype="plot",
			**vis.generate_line_style(label_idx),
		)

		# format plot
		axes[i].set_xlabel(field_config["format"]["xlabel"], fontsize=label_size)
		axes[i].set_ylabel(field_config["format"]["ylabel"], fontsize=label_size)
		axes[i].legend(prop={'size': legend_size})
		axes[i].tick_params(axis='both', which='major', labelsize=tick_size)
		axes[i].xaxis.get_offset_text().set_fontsize(tick_size)
		axes[i].yaxis.get_offset_text().set_fontsize(tick_size)

plt.tight_layout()

