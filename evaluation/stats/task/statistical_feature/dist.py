"""
This script plots the statistical feature differences between raw and synthetic traces.
"""

import matplotlib.pyplot as plt
import os 
import numpy as np
import json
import pandas as pd
import csv

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

general_field_configs = {
    # "trace_level_interarrival": {
    #     "field_name": "trace_level_interarrival",
    #     "extractor": lambda df: np.diff(df["time"]),
    #     "dist": stats.emd,
    #     "format": {
    #         # Trace-level Interarrival
    #         "title": "TIT"
    #     }
    # },
	# TODO: trace-level jitter
    "srcip": {
        "field_name": "srcip",
        "extractor": lambda df: df["srcip"].astype(int),
        "dist": stats.jsd,
        "format": {
            # "title": "Source IP (jsd)",
            "title": "SA"
        }
    },
    "dstip": {
        "field_name": "dstip",
        "extractor": lambda df: df["dstip"].astype(int),
        "dist": stats.jsd,
        "format": {
            # "title": "Destination IP (jsd)",
            "title": "DA"
        }
    },
    "srcport": {
        "field_name": "srcport",
        "extractor": lambda df: df["srcport"].astype(int),
        "dist": stats.jsd,
        "format": {
            # "title": "Source Port (jsd)",
            "title": "SP"
        }
    },
    "dstport": {
        "field_name": "dstport",
        "extractor": lambda df: df["dstport"].astype(int),
        "dist": stats.jsd,
        "format": {
            # "title": "Destination Port (jsd)",
            "title": "DP"
        }
    },
    "proto": {
    	"field_name": "proto",
    	"extractor": lambda df: df["proto"].astype(int),
    	"dist": stats.jsd,
    	"format": {
    		# "title": "Protocol (jsd)",
            "title": "PR"
    	}
    },
}

def clean_numeric_column(df, col):
    series = pd.to_numeric(df[col], errors="coerce")
    series = series.dropna()
    return series.astype(int)


pcap_field_configs = {
    "pkt_len": {
        "field_name": "pkt_len",
        # "extractor": lambda df: df["pkt_len"].astype(int),
        "extractor": lambda df: clean_numeric_column(df, "pkt_len"),
        "dist": stats.emd,
        "format": {
            # "title": "Packet Length (emd)",
            "title": "PL"
        }
    },
    # "tos": {
    #     "field_name": "tos",
    #     "extractor": lambda df: df["tos"].astype(int),
    #     "dist": stats.emd,
    #     "format": {
    #         # "title": "Type of Service (emd)",
    #         "title": "TOS"
    #     },
    # },
    # "off": {
    # 	"field_name": "off",
    # 	"extractor": lambda df: df["off"].astype(int),
    # 	"dist": stats.emd,
    # 	"format": {
    # 		"title": "Offset (emd)",
    # 	},
    # },
    "ttl": {
        "field_name": "ttl",
        "extractor": lambda df: df["ttl"].astype(int),
        "dist": stats.emd,
        "format": {
            # "title": "Time to Live (emd)",
            "title": "TTL"
        },
    },

    "record_count": {
		"field_name": "flowsize",
		"extractor": lambda df, dfg, gks, flows: flows.sum(axis=0),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# "title": "Record Count"
			"title": "NPF"
		}
	},
	"byte_count": {
		"field_name": "byte_count",
		"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["pkt_len"].sum() for gk in gks]),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# "title": "Byte Count"
			"title": "NBF"
		}
	},
}

netflow_field_configs = {
    "td": {
        "field_name": "td",
        "extractor": lambda df: df["td"].astype(int),
        "dist": stats.emd,
        "format": {
            "title": "Flow Duration (emd)",
        }
    },
    "pkt": {
        "field_name": "pkt",
        "extractor": lambda df: df["pkt"].astype(int),
        "dist": stats.emd,
        "format": {
            "title": "Packet Count (emd)",
        }
    },
    "byt": {
        "field_name": "byt",
        "extractor": lambda df: df["byt"].astype(int),
        "dist": stats.emd,
        "format": {
            "title": "Byte Count (emd)",
        }
    },
    "type": {
        "field_name": "type",
        "extractor": lambda df: data.categories2number(df["type"])[0],
        "dist": stats.jsd,
        "format": {
            "title": "Type (jsd)",
        }
    }
}

time_point_count = 400

flow_tuple = stats.five_tuple

label_filter = {}

if dataset in ['ca', 'caida', 'dc'] and folder == 'diff_syn':
    label_filter = {
    # for CA, CAIDA, DC
        "srcip": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "dstip": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "srcport": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "dstport": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "proto": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "record_count": ["NetShare", "REaLTabFormer", "CascadeNet"],
        "byte_count": ["NetShare", "REaLTabFormer", "CascadeNet"],
        "pkt_len": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "ttl": ["CTGAN", "E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
    }
elif dataset == 'ton_iot' and folder == 'diff_syn':
    label_filter = {
    # for TON
        "srcip": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion", "CascadeNet"],
        "dstip": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
        "srcport": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
        "dstport": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
        "proto": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
        "record_count": ["REaLTabFormer", "NetShare", "NetDiffusion", "CascadeNet"],
        "byte_count": ["REaLTabFormer", "NetShare", "NetDiffusion", "CascadeNet"],
        "pkt_len": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
        "ttl": ["CTGAN", "E-WGAN-GP", "REaLTabFormer", "NetShare", "NetDiffusion","CascadeNet"],
    }
elif folder == 'ablation':
    label_filter = {
        "srcip": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "dstip": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "srcport": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "dstport": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "proto": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "record_count": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "byte_count": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "pkt_len": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "ttl": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
    }
elif folder == 'time_series_length':
        label_filter = {
        "srcip": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "dstip": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "srcport": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "dstport": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "proto": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "record_count": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "byte_count": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "pkt_len": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "ttl": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
    }


flow_filter = {
    "flowsize_range": None,
    # "flowsize_range": (100, np.inf),
    # "flowsize_range": (0, 5),
    "flowDurationRatio_range": None,
    "nonzeroIntervalCount_range": None,
    "maxPacketrate_range": None,
    # "maxPacketrate_range": (100, np.inf),
}

label_size = 28
title_size = 2
tick_size = 24
legend_size = 28

if folder == 'time_series_length':
    legend_ncols = 3
    legend_size = 24
elif folder == 'cascadenet_test_cond':
    legend_ncols = 2
    legend_size = 22
else:
    legend_ncols = 2

# Get the absolute path to the script
script_path = os.path.abspath(__file__)

# Go up to ML-testing-dev directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path)))))

# Create the result directory path
if test:
    save_config = {
        "folder": os.path.join(project_root, "test_result", "evaluation"),
        "filename": "statistical_features_test",
        "format": "pdf",
    }
else:
    save_config = {
        "folder": os.path.join(project_root, "result", "evaluation", "stats", folder, dataset),
        "filename": f"statistical_features_{dataset}",
        "format": "pdf",
    }

# Define path for normalization CSV file 
normalization_csv_path = os.path.join(project_root, "evaluation", "stats", "normalization.csv")

"""
Functions for saving and loading normalization values
"""
def save_normalization_values(values, csv_path):
    """
    Save normalization values to a CSV file
    
    Args:
        values (dict): Dictionary of field -> max value
        csv_path (str): Path to save the CSV file
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Check if file exists to either update or create new
    if os.path.exists(csv_path):
        # Read existing data
        existing_data = {}
        with open(csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                if len(row) >= 3:
                    key = (row[0], row[1])
                    existing_data[key] = float(row[2])
        
        # Update with new values if larger
        for field, max_val in values.items():
            key = (dataset, field)
            if key in existing_data:
                existing_data[key] = max(existing_data[key], max_val)
            else:
                existing_data[key] = max_val
        
        # Write back
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'field', 'max_value'])
            for (ds, field), max_val in existing_data.items():
                writer.writerow([ds, field, max_val])
    else:
        # Create new file
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['dataset', 'field', 'max_value'])
            for field, max_val in values.items():
                writer.writerow([dataset, field, max_val])

def load_normalization_values(csv_path, dataset_name):
    """
    Load normalization values from a CSV file
    
    Args:
        csv_path (str): Path to the CSV file
        dataset_name (str): Name of the dataset to load values for
        
    Returns:
        dict: Dictionary of field -> max value
    """
    if not os.path.exists(csv_path):
        return {}
    
    values = {}
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in reader:
            if len(row) >= 3 and row[0] == dataset_name:
                values[row[1]] = float(row[2])
    
    return values

"""
Visualization Code
"""

if use_perturb_ref:
	label2path["raw_perturb"] = label2path["raw"]

field_configs = { **general_field_configs }
if dataset_type == DATASET_TYPE.PCAP:
    field_configs = { **field_configs, **pcap_field_configs }
elif dataset_type == DATASET_TYPE.NETFLOW:
    field_configs = { **field_configs, **netflow_field_configs }
else:
    raise ValueError("Unknown type of dataset: '{}'. Not pcap or netflow".format(dataset))
    
labels = list(label2path)
# move "raw" to first so that iterate label starting from "raw"
labels.remove("raw")
labels = ["raw"] + labels
fields = list(field_configs.keys())

# Create figure with two subplots
fig, axes = vis.subplots(1, 2, figsize=(32, 4.8)) 
labels_to_plot = [label for label in labels if label != "raw"]

# Default label_filter to include all available labels for any fields without specific filters
for field in field_configs.keys():
    if field not in label_filter:
        label_filter[field] = labels_to_plot

if normalization_csv:
    use_csv_normalization = True

# Load existing normalization values
norm_values = {}
if use_csv_normalization and normalization_csv and os.path.exists(normalization_csv_path):
    norm_values = load_normalization_values(normalization_csv_path, dataset)
    print(f"Loaded normalization values from {normalization_csv_path} for dataset {dataset}:")
    for field, value in norm_values.items():
        print(f"  {field}: {value}")

# Path to the cached distances
field_label_dists_path = os.path.join(
    save_config["folder"],
    "{}_{}.json".format(save_config["filename"], dataset)
)

# If the cached file doesn't exist or we're overwriting, compute the distances
if not os.path.exists(field_label_dists_path) or overwrite_history_result:
    field_label_samples = dict.fromkeys(field_configs)
    for field in field_label_samples:
        field_label_samples[field] = dict.fromkeys(label2path, -1)

    field_label_dists = dict.fromkeys(field_configs)
    for field in field_label_dists:
        field_label_dists[field] = dict.fromkeys(label2path, -1)
        field_label_dists[field].pop("raw")

    # Compute samples for all fields for all labels
    for label in labels:
        df = data.load_csv(
            path=label2path[label],
            is_ip2int=True,
            is_proto2int=True,
            verbose=False,
            need_divide_1e6="auto",
            unify_timestamp_fieldname=True,
        )
        # Process time-related fields first
        if any(field in field_configs for field in ["byte_count", "flowsize", "record_count"]):
            if df["time"].min() < 0:  # Check for fake time column
                print(f"Warning: Fake time column detected in the dataset {label}")
                for field in ["byte_count", "flowsize", "record_count"]:
                    if field in field_configs:
                        field_label_samples[field][label] = np.array([5000])    # randomly pick a large num
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
                for field in ["byte_count", "flowsize", "record_count"]:
                    if field in field_configs:
                        field_config = field_configs[field]
                        field_label_samples[field][label] = field_config["extractor"](df, dfg, gks, flows)

        # Process other fields
        for field in field_configs:
            if field not in ["byte_count", "flowsize", "record_count"]:
                field_config = field_configs[field]
                samples = field_config["extractor"](df)
                if use_perturb_ref and field_config["dist"] == stats.emd and label == "raw_perturb":
                    samples = samples + np.random.normal(0, perturb_coeff * samples.std(), samples.shape)
                field_label_samples[field][label] = samples

    # Compute distance for all fields for all labels
    for label in labels[1:]:
        for field in field_configs:
            samples_raw = field_label_samples[field]["raw"] 
            # extract field
            field_config = field_configs[field]
            samples_syn = field_label_samples[field][label]

            # compute distance
            dist = field_config["dist"](samples_raw, samples_syn)

            # save distance
            field_label_dists[field][label] = dist

    # Dictionary to store max EMD values for normalization
    max_emd_values = {}
    
    # Compute max EMD values before normalization
    for field in field_label_dists:
        dists = field_label_dists[field]
        if field_configs[field]["dist"] != stats.emd:
            continue
            
        filtered_labels = label_filter.get(field, labels_to_plot)
        
        # Calculate max EMD value for this field
        if filtered_labels and any(label in dists for label in filtered_labels):
            max_emd = max(dists[label] for label in filtered_labels if label in dists)
            max_emd_values[field] = max_emd

    # Save normalization values if enabled and in diff_syn folder
    if folder == 'diff_syn' and save_normalization and normalization_csv:
        save_normalization_values(max_emd_values, normalization_csv_path)
        print(f"Saved normalization values to {normalization_csv_path}")
        for field, value in max_emd_values.items():
            print(f"  {field}: {value}")
    
    # Now normalize based on either CSV values or computed max values
    for field in field_label_dists:
        if field_configs[field]["dist"] != stats.emd:
            continue
            
        dists = field_label_dists[field]
        
        # Normalize using values from CSV if provided, otherwise use calculated max
        if use_csv_normalization and field in norm_values:
            norm_value = norm_values[field]
            print(f"Using CSV normalization value for {field}: {norm_value}")
        else:
            norm_value = max_emd_values[field]
            print(f"Using local max for {field}: {norm_value}")
            
        # Apply normalization and scale to [0, 0.9] for visualization
        for label in dists:
            field_label_dists[field][label] = 0.9 * (dists[label] / norm_value)

    # Save as json for later use
    os.makedirs(os.path.dirname(field_label_dists_path), exist_ok=True)
    with open(field_label_dists_path, "w") as f:
        json.dump(field_label_dists, f)
else:
    # Load from json
    print('The json file exists, loading from it')
    with open(field_label_dists_path, "r") as f:
        field_label_dists = json.load(f)

    if folder == 'diff_syn':
        # Load normalization values if available and requested
        if use_csv_normalization and normalization_csv:
            norm_values = load_normalization_values(normalization_csv_path, dataset)
            print(f"Loaded normalization values from {normalization_csv_path} for dataset {dataset}:")
            for field, value in norm_values.items():
                print(f"  {field}: {value}")
        
        # Only normalize EMD fields (not JSD)
        for field in field_label_dists:
            if field_configs[field]["dist"] != stats.emd:
                continue
                
            dists = field_label_dists[field]
            
            # Normalize using values from CSV if provided
            if use_csv_normalization and field in norm_values:
                norm_value = norm_values[field]
                print(f"Normalizing {field} using CSV value: {norm_value}")
                
                # Apply normalization and scale to [0, 0.9] for visualization
                for label in dists:
                    field_label_dists[field][label] = 0.9 * (dists[label] / norm_value)

# For each dataset, calculate the average for JSD and EMD separately
jsd_fields = [field for field in fields if field_configs[field]["dist"] == stats.jsd]
emd_fields = [field for field in fields if field_configs[field]["dist"] == stats.emd]

# For each model, compute some averages
label_field_avg = dict.fromkeys(labels[1:])
for label in labels[1:]:
    label_field_avg[label] = dict.fromkeys(["avg_jsd", "avg_emd", "avg_metric"], -1)
for label in labels[1:]:
    avg_jsd = np.mean([field_label_dists[field][label] for field in jsd_fields if label in field_label_dists[field]])
    avg_emd = np.mean([field_label_dists[field][label] for field in emd_fields if label in field_label_dists[field]])
    avg_metric = np.mean([field_label_dists[field][label] for field in fields if label in field_label_dists[field]])
    label_field_avg[label]["avg_jsd"] = avg_jsd
    label_field_avg[label]["avg_emd"] = avg_emd
    label_field_avg[label]["avg_metric"] = avg_metric

# Prepare colors and patterns for different datasets
dataset_styles = {
    label: {
        'color': 'none',
        'edgecolor': vis.get_color(label),
        'hatch': vis.get_hatch(label),
        'linewidth': 2,
        'alpha': 0.99,
    } for label in labels if label != "raw"
}

# Distance between groups of bars within a metric
group_gap = 0.4
# Distance between bars within a group
bar_gap = 0.0
# Width of each bar
bar_width = 0.15

# Calculate positions for all bars and group centers before plotting
jsd_groups = []  # Store information about JSD groups
emd_groups = []  # Store information about EMD groups

# Pre-calculate positions for JSD metrics
jsd_current_pos = 0
for field in fields:
    if field_configs[field]["dist"] != stats.jsd:
        continue
        
    filtered_labels = label_filter.get(field, [l for l in labels[1:] if l != "raw"])
    num_bars = len(filtered_labels)
    group_width = num_bars * bar_width + (num_bars - 1) * bar_gap
    
    group_info = {
        'field': field,
        'start_pos': jsd_current_pos,
        'width': group_width,
        'center': jsd_current_pos + group_width / 2,
        'labels': filtered_labels
    }
    jsd_groups.append(group_info)
    
    jsd_current_pos += group_width + group_gap

# Pre-calculate positions for EMD metrics
emd_current_pos = 0
for field in fields:
    if field_configs[field]["dist"] != stats.emd:
        continue
        
    filtered_labels = label_filter.get(field, [l for l in labels[1:] if l != "raw"])
    num_bars = len(filtered_labels)
    group_width = num_bars * bar_width + (num_bars - 1) * bar_gap
    
    group_info = {
        'field': field,
        'start_pos': emd_current_pos,
        'width': group_width,
        'center': emd_current_pos + group_width / 2,
        'labels': filtered_labels
    }
    emd_groups.append(group_info)
    
    emd_current_pos += group_width + group_gap

# Now draw the bars for JSD metrics
for group in jsd_groups:
    field = group['field']
    start_pos = group['start_pos']
    filtered_labels = group['labels']
    
    dists = field_label_dists[field]
    
    for j, label in enumerate(filtered_labels):
        if label in dists:
            bar_pos = start_pos + j * (bar_width + bar_gap)
            axes[0].bar(
                bar_pos,
                dists[label],
                width=bar_width,
                align='edge',
                label=label if field == jsd_groups[0]['field'] else "",
                **dataset_styles[label]
            )

# Now draw the bars for EMD metrics
for group in emd_groups:
    field = group['field']
    start_pos = group['start_pos']
    filtered_labels = group['labels']
    
    dists = field_label_dists[field]
    
    for j, label in enumerate(filtered_labels):
        if label in dists:
            bar_pos = start_pos + j * (bar_width + bar_gap)
            axes[1].bar(
                bar_pos,
                dists[label],
                width=bar_width,
                align='edge',
                label="",  # No need to add labels in the second subplot
                **dataset_styles[label]
            )

# Customizing the plots
metrc2label = {
    stats.jsd: "JS Divergence",
    stats.emd: "Normalized EMD"
}

# Set x-ticks at calculated centers for JSD subplot
jsd_centers = [group['center'] for group in jsd_groups]
jsd_labels = [field_configs[group['field']]['format']['title'] for group in jsd_groups]
axes[0].set_xticks(jsd_centers)
axes[0].set_xticklabels(jsd_labels)

# Set x-ticks at calculated centers for EMD subplot
emd_centers = [group['center'] for group in emd_groups]
emd_labels = [field_configs[group['field']]['format']['title'] for group in emd_groups]
axes[1].set_xticks(emd_centers)
axes[1].set_xticklabels(emd_labels)

# Configure both subplots
for i, (ax, metric, groups) in enumerate(zip(axes, [stats.jsd, stats.emd], [jsd_groups, emd_groups])):
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.set_ylabel(metrc2label[metric], fontsize=label_size)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.3, 0.6, 0.9])
    ax.set_yticklabels([0, 0.3, 0.6, 0.9])

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Move bottom spine to y=0
    ax.spines['bottom'].set_position('zero')
    
    # Adjust tick parameters
    ax.tick_params(axis='both', which='major', labelsize=tick_size, direction='in')

# Set x-axis limits to include all groups with proper spacing
if groups:
    min_pos = min(group['start_pos'] for group in groups) - group_gap/2
    max_pos = max(group['start_pos'] + group['width'] for group in groups) + group_gap/2
    ax.set_xlim(min_pos, max_pos)

handles, labels_in_plot = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels_in_plot, handles))
ordered_labels = [label for label in labels_to_plot if label in by_label]
ordered_handles = [by_label[label] for label in ordered_labels]

for ax in axes:
    ax.legend(
    ordered_handles,
    ordered_labels,
    fontsize=legend_size-4,
    ncol=legend_ncols,
    loc='upper left',
    frameon=True,
    labelspacing=0.2,
    columnspacing=1.2,
    handletextpad=0.2,
    borderaxespad=0.1,
    borderpad=0.5,
    handlelength=1.5,
    )


# Adjust layout to fit labels
plt.tight_layout()

if save_figure:
    vis.savefig(fig, save_config)
if show_figure:
    fig.canvas.manager.set_window_title("{} - {}".format(
        dataset, 
        os.path.basename(__file__).split(".")[0]
    ))
    plt.show()