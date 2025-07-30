"""
This script plots the distribution of statistical features for different datasets.
"""

import matplotlib.pyplot as plt
import os 
import pickle
import numpy as np
import pandas as pd
import nta.utils.vis as vis 
import nta.utils.data as data
import nta.utils.stats as stats

"""
Configuration
"""

from evaluation.stats.util.argparser import *

bins_count = 100

## There are three types of record fields
## - fields that both pcap and netflow have (e.g., interarrival)
## - fields that only pcap has (e.g., packet length)
## - fields that only netflow has (e.g., flow duration)
general_field_configs = {
    # "trace_level_interarrival": {
    #     "field_name": "trace_level_interarrival",
    #     "extractor": lambda df: np.diff(df["time"]),
    #     # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
    #     # 	x, ax, 
    #     # 	bins_count=100, bins_min=1e0, bins_max=1e7,
    #     # 	density=True, **kwargs), 
    #     "plotter": lambda x, ax, **kwargs: vis.plot_hist(
    #         x, ax,
    #         bins_count=100, 
    #         # bins_min=0, bins_max=1e7,
    #         density=True, **kwargs),
    #     "format": {
    #         "xlabel": "trace-level interarrival",
    #         "ylabel": "density",
    #         "title": "Trace-level Interarrival",
    #     }
    # },
    # "flow_level_interarrival": {
    # 	"field_name": "flow_level_interarrival",
    # 	"extractor": lambda df: stats.df2flow_level_interarrival(df),	
    # 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
    # 	# 	x, ax, 
    # 	# 	bins_count=500, bins_min=1e0, bins_max=1e7, truncate_x=True,
    # 	# 	density=True, **kwargs),
    # 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
    # 		x, ax,
    # 		bins_count=100, 
    # 		# bins_min=1e0, bins_max=1e7, truncate_x=True,
    # 		density=True, **kwargs),
    # 	"format": {
    # 		"xlabel": "flow-level interarrival",
    # 		"ylabel": "density",
    # 		"title": "Flow-level Interarrival",
    # 	}
    # },
    # "flowstart": {
    # 	"field_name": "flowstart",
    # 	# group by five tuple,
    # 	"extractor": lambda df: stats.df2flowstart(df, flow_filter=flow_filter),
    # 	# "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
    # 	# 	x, ax, 
    # 	# 	bins_count=100, bins_min=1e-2, bins_max=1e2,
    # 	# 	density=True, **kwargs),
    # 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
    # 		x, ax,
    # 		bins_count=500, 
    # 		density=True, **kwargs),
    # 	"format": {
    # 		"xlabel": "flow start",
    # 		"ylabel": "density",
    # 		"title": "Flow Start",
    # 	}
    # },
    # "srcip": {
    # 	"field_name": "srcip",
    # 	"extractor": lambda df: df["srcip"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: ax.hist(x, bins=100, density=True, histtype="step", **kwargs),
    # 	"format": {
    # 		"xlabel": "source ip",
    # 		"ylabel": "density",
    # 		"title": "Source IP",
    # 	}
    # },
    # "dstip": {
    # 	"field_name": "dstip",
    # 	"extractor": lambda df: df["dstip"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: ax.hist(x, bins=100, density=True, histtype="step", **kwargs),
    # 	"format": {
    # 		"xlabel": "destination ip",
    # 		"ylabel": "density",
    # 		"title": "Destination IP",
    # 	}
    # },
    # "srcport": {
    # 	"field_name": "srcport",
    # 	"extractor": lambda df: df["srcport"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
    # 		x, ax,
    # 		bins_count=100, bins_min=0, bins_max=65535,
    # 		density=True, **kwargs),
    # 	"format": {
    # 		"xlabel": "source port",
    # 		"ylabel": "density",
    # 		"title": "Source Port",
    # 	}
    # },
    # "dstport": {
    # 	"field_name": "dstport",
    # 	"extractor": lambda df: df["dstport"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
    # 		x, ax,
    # 		bins_count=100, bins_min=0, bins_max=65535,
    # 		density=True, **kwargs),
    # 	"format": {
    # 		"xlabel": "destination port",
    # 		"ylabel": "density",
    # 		"title": "Destination Port",
    # 	}
    # },
    # "proto": {
    # 	"field_name": "proto",
    # 	"extractor": lambda df: df["proto"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: ax.hist(x, bins=100, density=True, histtype="step", **kwargs),
    # 	"format": {
    # 		"xlabel": "proto",
    # 		"ylabel": "density",
    # 		"title": "Protocol",
    # 	}
    # },
}

pcap_field_configs = {
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
    "pkt_len": {
        "field_name": "pkt_len",
        "extractor": lambda df: df["pkt_len"].astype(int),
        # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
        # 	x, ax, 
        # 	bins_count=100, bins_min=1, bins_max=2e3,
        # 	density=True, **kwargs), 
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax, 
            bins_count=100, 
            # bins_min=0, bins_max=2e3,
            density=True, **kwargs),
        "format": {
            "xlabel": "packet length",
            "ylabel": "density",
            "title": "Packet Length",
        }
    },
    # "tos": {
    #     "field_name": "tos",
    #     "extractor": lambda df: df["tos"].astype(int),
    #     "plotter": lambda x, ax, **kwargs: vis.plot_hist(
    #         x, ax, 
    #         bins_count=100, 
    #         # bins_min=0, bins_max=400,
    #         density=True, **kwargs),
    #     "format": {
    #         "xlabel": "ToS",
    #         "ylabel": "density",
    #         "title": "ToS",
    #     }
    # },
    # "off": {
    # 	"field_name": "off",
    # 	"extractor": lambda df: df["off"].astype(int),
    # 	"plotter": lambda x, ax, **kwargs: vis.plot_hist(
    # 		x, ax, 
    # 		bins_count=100, 
    # 		# bins_min=0, bins_max=100,
    # 		density=True, **kwargs),
    # 	"format": {
    # 		"xlabel": "off",
    # 		"ylabel": "density",
    # 		"title": "off",
    # 	}
    # },
    "ttl": {
        "field_name": "ttl",
        "extractor": lambda df: df["ttl"].astype(int),
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax, 
            bins_count=100, 
            # bins_min=0, bins_max=400,
            density=True, **kwargs),
        "format": {
            "xlabel": "ttl",
            "ylabel": "density",
            "title": "ttl"
        }
    },
}

netflow_field_configs = {
    "td": {
        "field_name": "td",
        "extractor": lambda df: df["td"].astype(int),
        # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
        # 	x, ax,
        # 	bins_count=100, bins_min=1, bins_max=1e7,
        # 	density=True, **kwargs),
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax,
            bins_count=100, bins_min=0, bins_max=1e7,
            density=True, **kwargs),
        "format": {
            "xlabel": "flow duration",
            "ylabel": "density",
            "title": "Flow Duration",
        }
    },
    "pkt": {
        "field_name": "pkt",
        "extractor": lambda df: df["pkt"].astype(int),
        # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
        # 	x, ax,
        # 	bins_count=100, bins_min=1, bins_max=1e7,
        # 	density=True, **kwargs),
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax,
            bins_count=100, bins_min=0, bins_max=1e7,
            density=True, **kwargs),
        "format": {
            "xlabel": "packet count",
            "ylabel": "density",
            "title": "Packet Count",
        }
    },
    "byt": {
        "field_name": "byt",
        "extractor": lambda df: df["byt"].astype(int),
        # "plotter": lambda x, ax, **kwargs: vis.plot_loghist(
        # 	x, ax,
        # 	bins_count=100, bins_min=1, bins_max=1e7,
        # 	density=True, **kwargs),
        "plotter": lambda x, ax, **kwargs: vis.plot_hist(
            x, ax,
            bins_count=100, bins_min=0, bins_max=1e7,
            density=True, **kwargs),
        "format": {
            "xlabel": "byte count",
            "ylabel": "density",
            "title": "Byte Count",
        }
    },
    "type": {
        "field_name": "type",
        "extractor": lambda df: data.categories2number(df["type"])[0],
        "plotter": lambda x, ax, **kwargs: ax.hist(x, bins=100, density=True, histtype="step", **kwargs),
        "format": {
            "xlabel": "type",
            "ylabel": "density",
            "title": "Type",
        }
    },
}

time_point_count = 200

flow_filter = {
    "flowsize_range": None,
    # "flowsize_range": (100, np.inf),
    # "flowsize_range": (1, 2),
    "flowDurationRatio_range": None,
    "nonzeroIntervalCount_range": None,
    "maxPacketrate_range": None,
    # "maxPacketrate_range": (100, np.inf),
}

# style
label_size = 2
title_size = 12
tick_size = 2
legend_size = 5

# Get the absolute path to the script
script_path = os.path.abspath(__file__)

# Go up to ML-testing-dev directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path)))))

# Create the result directory path
if test:
    save_config = {
        "folder": os.path.join(project_root, "test_result", "evaluation"),
        "filename": "statistical_features_distribution_test",
        "format": "svg",
    }
else:
    save_config = {
        "folder": os.path.join(project_root, "result", "evaluation", "stats", folder, dataset),
        "filename": "statistical_features_distribution",
        "format": "svg",
    }

"""
Visualization Code
"""

def safe_plot_hist(x, ax, **kwargs):
    # Remove inf and nan values
    x = x[np.isfinite(x)]
    
    if len(x) == 0:
        print(f"Warning: All values are infinite or NaN for this field.")
        return
    
    bins_count = kwargs.get('bins_count', 100)
    bins_max = kwargs.get('bins_max', None)
    density = kwargs.get('density', True)
    cumsum = kwargs.get('cumsum', True)
    log_bins = kwargs.get('log_bins', True)
    
    if bins_max is None:
        bins_max = np.max(x)
    
    bins_min = np.min(x)
    if bins_min == bins_max:
        bins_min = 0.99 * bins_min
        bins_max = 1.01 * bins_max
    
    try:
        vis.plot_hist(x, ax, bins_count=bins_count, bins_min=bins_min, bins_max=bins_max,
                      density=density, cumsum=cumsum, log_bins=log_bins, **kwargs)
    except ValueError as e:
        print(f"Warning: Error in plotting histogram - {str(e)}")
        print(f"Data summary: min={np.min(x)}, max={np.max(x)}, mean={np.mean(x)}")


# field_configs = { **general_field_configs }
# if dataset in pcap_datasets:
#     field_configs = { **field_configs, **pcap_field_configs }
# elif dataset in netflow_datasets:
#     field_configs = { **field_configs, **netflow_field_configs }
field_configs = { **general_field_configs }
if dataset_type == DATASET_TYPE.PCAP:
    field_configs = { **field_configs, **pcap_field_configs }
elif dataset_type == DATASET_TYPE.NETFLOW:
    field_configs = { **field_configs, **netflow_field_configs }
else:
    raise ValueError("Unknown type of dataset: '{}'. Not pcap or netflow".format(dataset))

flow_tuple = stats.five_tuple
labels = list(label2path.keys())
fields = list(field_configs.keys())

print("List of packet fields to visualize:")
for field_name in field_configs:
    print("- {}".format(field_name))
print()

field_label_data_path = os.path.join(
    save_config["folder"],
    "{}_{}.pkl".format(save_config["filename"], dataset)
    )
field_label_data = None
if not os.path.exists(field_label_data_path) or overwrite_history_result:
    # initialize field_label_data
    field_label_data = dict.fromkeys(field_configs)
    for field in field_label_data:
        field_label_data[field] = dict.fromkeys(label2path, -1)

    for label in label2path:
        df = data.load_csv(
            path=label2path[label],
            verbose=False,
            need_divide_1e6="auto",
            unify_timestamp_fieldname=True,
        )
            
        if "byte_count" in field_configs or "flowsize" in field_configs:
            if df["time"].min() < 0:  # Check for fake time column
                print(f"Warning: Fake time column detected in the dataset {label}")
                # Assign a very large number for metrics that require the 'time' column
                if "byte_count" in field_configs:
                    field_label_data['byte_count'][label] = np.array([float('inf')])
                if "flowsize" in field_configs:
                    field_label_data['flowsize'][label] = np.array([float('inf')])
            else:
                df["time"] = df["time"] - df["time"].min()  
                dfg, gks, flows = data.load_flow_from_df(
                    df,
                    mode=("time_point_count", time_point_count),
                    flow_tuple=flow_tuple,
                    **flow_filter,
                    return_all=True,
                )
                flows = flows.astype(int)
                if "byte_count" in field_configs:
                    field_config = field_configs["byte_count"]
                    field_label_data['byte_count'][label] = field_config["extractor"](df, dfg, gks, flows)
                if "flowsize" in field_configs:
                    field_config = field_configs["flowsize"]
                    field_label_data['flowsize'][label] = field_config["extractor"](df, dfg, gks, flows)

        for field_name in field_configs:
            if field_name not in ["byte_count", "flowsize"]:
                field_config = field_configs[field_name]
                try:
                    field = field_config["extractor"](df)
                    field_label_data[field_name][label] = field
                except KeyError as e:
                    print(f"Warning: Could not extract {field_name} for dataset {label}. Error: {e}")
                    field_label_data[field_name][label] = np.array([50000])
                    # field_label_data[field_name][label] = field

        # for i, field_name in enumerate(field_configs):
        #     field_config = field_configs[field_name]
        #     if field_name in ["byte_count", "flowsize"]:
        #         continue
        #     field = field_config["extractor"](df)
        #     field_label_data[field_name][label] = field
    
    # save as json for later use
    os.makedirs(os.path.dirname(field_label_data_path), exist_ok=True)
    f = open(field_label_data_path, "wb")
    pickle.dump(field_label_data, f)
    f.close()
else:
    # load from json
    f = open(field_label_data_path, "rb")
    field_label_data = pickle.load(f)
    f.close()
