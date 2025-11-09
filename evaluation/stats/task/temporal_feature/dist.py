"""
This script plots the temporal feature differences between raw and synthetic traces.
"""
import matplotlib
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

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

use_perturb_ref = False
perturb_coeff = 0.05

bins_count = 100

general_field_configs = {
	# Static Features
	# "record_count": {
	# 	"field_name": "flowsize",
	# 	"extractor": lambda df, dfg, gks, flows: flows.sum(axis=0),
	# 	"dist": stats.emd,
	# 	# "dist": stats.logemd,
	# 	"format": {
	# 		# "title": "Record Count"
	# 		"title": "RC"
	# 	}
	# },
	# "byte_count": {
	# 	"field_name": "byte_count",
	# 	"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["pkt_len"].sum() for gk in gks]),
	# 	"dist": stats.emd,
	# 	# "dist": stats.logemd,
	# 	"format": {
	# 		# "title": "Byte Count"
	# 		"title": "BC"
	# 	}
	# },

	# # Temporal Features
	# This measures the flow-level jitter
	"trace_level_interarrival": {
        "field_name": "trace_level_interarrival",
        "extractor": lambda df: np.diff(df["time"]),
        "dist": stats.emd,
        "format": {
            # Trace-level Interarrival
            "title": "IT"
        }
    },
	"flow_level_interarrival": {
		"field_name": "flow_level_interarrival",
		"extractor": lambda df, dfg, gks, flows: stats.dfg2flow_level_interarrival(dfg, gks),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			"xlabel": "flow-level interarrival",
			"ylabel": "density",
			# "title": "Interarrival",
			"title": "FIT"
		}
	},
	"flow_duration": {
		"field_name": "flow_duration",
		"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["time"].max() - dfg.get_group(gk)["time"].min() for gk in gks]),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# "title": "Flow Duration"
			"title": "FD"
		}
	},


	# "max_packetrate": {
	# 	"field_name": "max_packetrate", 
	# 	"extractor": lambda df, dfg, gks, flows: flows.max(axis=0),
	# 	"dist": stats.emd,
	# 	# "dist": stats.logemd,
	# 	"format": {
	# 		# "title": "Max Packetrate",
	# 		"title": "MPR"
	# 	}
	# },
	# "mean_packetrate": {
	# 	"field_name": "mean_packetrate", 
	# 	"extractor": lambda df, dfg, gks, flows: flows.mean(axis=0),
	# 	"dist": stats.emd,
	# 	# "dist": stats.logemd,
	# 	"format": {
	# 		# "title": "Mean Packetrate",
	# 		"title": "MPR"
	# 	}
	# },
	# "std_recordrate": {
	# 	"field_name": "std_recordrate", 
	# 	"extractor": lambda df, dfg, gks, flows: flows.std(axis=0),
	# 	"dist": stats.emd,
	# 	# "dist": stats.logemd,
	# 	"format": {
	# 		# "title": "Standard Deviance of Record Rate",
	# 		"title": "SRR"
	# 	}
	# },
}

pcap_field_configs = {

	# Temporal features for large flows
	"avg_tput": {
		"field_name": "byte_count",
		"extractor": lambda df, dfg, gks, flows: np.array([
			# sum of packet len divided by duration. If only one packet, use 1
			# used only for large flow evaluation
			g["pkt_len"].sum() / g["time"].max() - g["time"].min()
			for g in [dfg.get_group(gk) for gk in gks] if g["time"].max() > g["time"].min()
		]),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# Average Throughput
			"title": "AT"
		}
	},
	"burstiness": {
		"field_name": "burstiness",
		# calculate the coefficient of variation
		"extractor": lambda df, dfg, gks, flows: flows.std(axis=0) / flows.mean(axis=0),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# Burstiness, i.e. coefficient of variation
			"title": "BU"
		}
	},
	"hurst": {
		"field_name": "hurst_exponent",
		"extractor": lambda df, dfg, gks, flows: stats.flows2hursts(flows),
		"dist": stats.emd,
		# "dist": stats.logemd,
		"format": {
			# "title": "Hurst Exponent"
			"title": "LD"
		}
	},

	# # "byte_mean": {
	# # 	"field_name": "byte_mean",
	# # 	"extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["pkt_len"].mean() for gk in gks]),
	# # 	"dist": stats.emd,
	# # 	# "dist": stats.logemd,
	# # 	"format": {
	# # 		# "title": "Byte Mean"
	# # 		"title": "BM"
	# # 	}
	# # },
	# # "byte_var": {
	# # 	"field_name": "byte_var",
	# # 	# "extractor": lambda df, dfg, gks, flows: np.array([dfg.get_group(gk)["pkt_len"].mean() for gk in gks]),
	# # 	"extractor": lambda df, dfg, gks, flows: np.nan_to_num(np.array([dfg.get_group(gk)["pkt_len"].std() for gk in gks]), nan=0),
	# # 	"dist": stats.emd,
	# # 	# "dist": stats.logemd,
	# # 	"format": {
	# # 		# "title": "Byte Variance"
	# # 		"title": "BV"
	# # 	}
	# # },
}

netflow_field_configs = {

}

time_point_count = 400

flow_tuple = stats.five_tuple
# flow_tuple = [flow_tuple[0], flow_tuple[2]]
label_filter = {}

if dataset in ['ca', 'caida', 'dc'] and folder == 'diff_syn':
    label_filter = {
    # for CA, CAIDA, DC
        "flow_level_interarrival": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "flow_duration": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "trace_level_interarrival": ["E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
        "avg_tput": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "burstiness": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "hurst": ["REaLTabFormer", "NetShare", "CascadeNet"],
    }
elif dataset == 'ton_iot' and folder == 'diff_syn':
    label_filter = {
    # for TON
        "flow_level_interarrival": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "flow_duration": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "trace_level_interarrival": ["E-WGAN-GP", "REaLTabFormer", "NetShare", "CascadeNet"],
        "avg_tput": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "burstiness": ["REaLTabFormer", "NetShare", "CascadeNet"],
        "hurst": ["REaLTabFormer", "NetShare", "CascadeNet"],
    }
elif folder == 'ablation':
    label_filter = {
        "flow_level_interarrival": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "flow_duration": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "trace_level_interarrival": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "avg_tput": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "burstiness": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
        "hurst": ["CN-w/o-ZI", "CN-w/o-Cond", "CascadeNet"],
    }
elif folder == 'time_series_length':
    label_filter = {
        "flow_level_interarrival": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "flow_duration": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "trace_level_interarrival": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "avg_tput": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "burstiness": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
        "hurst": ["CN-20", "CN-50", "CN-100", "CN-200", "CN-500", "CN-1000", "CN-2000"],
    }
elif folder == 'timestamp_recover':
    label_filter = {
        "flow_level_interarrival": ["EQ", "SP", "ML"],
        "flow_duration":  ["EQ", "SP", "ML"],
        "trace_level_interarrival":  ["EQ", "SP", "ML"],
        "avg_tput":  ["EQ", "SP", "ML"],
        "burstiness":  ["EQ", "SP", "ML"],
        "hurst":  ["EQ", "SP", "ML"],
    }


flow_filter = {
    "flowsize_range": None,
	# "flowsize_range": 0.01,

    "flowDurationRatio_range": None,
	# "flowDurationRatio_range": 0.01,
    "nonzeroIntervalCount_range": None,
    "maxPacketrate_range": None,
    # "maxPacketrate_range": (100, np.inf),
}

if folder == 'timestamp_recover':
    legend_size = 26
    legend_ncols = 3
elif folder == 'time_series_length':
    legend_size = 17
    legend_ncols = 3
elif folder == 'cascadenet_test_cond':
    legend_size = 12
    legend_ncols = 2
else:
    legend_size = 17
    legend_ncols = 2

label_size = 21
title_size = 2
tick_size = 21

# Get the absolute path to the script
script_path = os.path.abspath(__file__)

# Go up to ML-testing-dev directory
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(script_path)))))

# Create the result directory path
if test:
    save_config = {
        "folder": os.path.join(project_root, "test_result", "evaluation"),
        "filename": "temproal_features_test",
        "format": "pdf",
    }
else:
    save_config = {
        "folder": os.path.join(project_root, "result", "evaluation", "stats", folder, dataset),
        "filename": f"temproal_features_{dataset}",
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

field_configs = {**general_field_configs}
if dataset_type == DATASET_TYPE.PCAP:
    field_configs = {**field_configs, **pcap_field_configs}
elif dataset_type == DATASET_TYPE.NETFLOW:
    field_configs = {**field_configs, **netflow_field_configs}
else:
    raise ValueError("Unknown type of dataset: '{}'. Not pcap or netflow".format(dataset))

all_labels = list(label2path.keys())
if 'raw' not in all_labels:
    all_labels.insert(0, 'raw')
fields = list(field_configs.keys())

# Default label_filter to include all available labels for any fields without specific filters
non_raw_labels = [label for label in all_labels if label != 'raw']
for field in fields:
    if field not in label_filter:
        label_filter[field] = non_raw_labels

if normalization_csv:
    use_csv_normalization = True

# Load existing normalization values
norm_values = {}
if use_csv_normalization and normalization_csv and os.path.exists(normalization_csv_path):
    norm_values = load_normalization_values(normalization_csv_path, dataset)
    print(f"Loaded normalization values from {normalization_csv_path} for dataset {dataset}:")
    for field, value in norm_values.items():
        print(f"  {field}: {value}")

field_label_dists_path = os.path.join(
    save_config["folder"],
    "{}_{}.json".format(save_config["filename"], dataset)
)
if not os.path.exists(field_label_dists_path) or overwrite_history_result:
    field_label_samples = {}
    field_label_dists = {}
    for field in field_configs:
        # Initialize samples and distances dictionaries for each field
        labels_for_field = label_filter.get(field, all_labels)
        if 'raw' not in labels_for_field:
            labels_for_field.insert(0, 'raw')
        field_label_samples[field] = dict.fromkeys(labels_for_field, None)
        field_label_dists[field] = {}

    # Load raw data once
    df_raw = data.load_csv(
        path=label2path['raw'],
        verbose=False,
        need_divide_1e6="auto",
        unify_timestamp_fieldname=True,
    )
    df_raw["time"] = df_raw["time"] - df_raw["time"].min()
    dfg_raw, gks_raw, flows_raw = data.load_flow_from_df(
        df_raw,
        mode=("time_point_count", time_point_count),
        flow_tuple=flow_tuple,
        **flow_filter,
        return_all=True,
    )
    flows_raw = flows_raw.astype(int)

    # Extract features for raw data
    for field in field_configs:
        field_config = field_configs[field]
        if 'raw' in field_label_samples[field]:
            if field in ["trace_level_interarrival"]:
                field_value = field_config["extractor"](df_raw)
            else:
                field_value = field_config["extractor"](df_raw, dfg_raw, gks_raw, flows_raw)
            field_label_samples[field]['raw'] = field_value

    # Process other labels
    for label in all_labels:
        if label == 'raw':
            continue
        df = data.load_csv(
            path=label2path[label],
            verbose=False,
            need_divide_1e6="auto",
            unify_timestamp_fieldname=True,
        )
        df["time"] = df["time"] - df["time"].min()
        dfg, gks, flows = data.load_flow_from_df(
            df,
            mode=("time_point_count", time_point_count),
            flow_tuple=flow_tuple,
            **flow_filter,
            return_all=True,
        )
        flows = flows.astype(int)

        for field in field_configs:
            labels_for_field = field_label_samples[field].keys()
            if label not in labels_for_field:
                continue  # Skip this field for this label
            field_config = field_configs[field]
            if field in ["trace_level_interarrival"]:
                field_value = field_config["extractor"](df)
            else:
                field_value = field_config["extractor"](df, dfg, gks, flows)
            if use_perturb_ref and label == "raw_perturb":
                field_value = field_value + np.random.normal(0, perturb_coeff * field_value.std(), field_value.shape)
            field_label_samples[field][label] = field_value

    # Compute distances
    for field in field_configs:
        field_config = field_configs[field]
        labels_for_field = label_filter.get(field, all_labels)
        # Ensure 'raw' is excluded when computing distances
        labels_for_field = [label for label in labels_for_field if label != 'raw']
        samples_raw = field_label_samples[field].get('raw')
        if samples_raw is None:
            continue  # Cannot compute distances without raw data
        for label in labels_for_field:
            samples_syn = field_label_samples[field].get(label)
            if samples_syn is None:
                continue  # Skip if samples are missing
            # Debug: Check for empty distributions before computing distance
            if samples_raw.size == 0:
                print(f"Error: Empty distribution for raw samples in field '{field}'")
                continue
            if samples_syn.size == 0:
                print(f"Warning: Empty distribution for synthetic samples in field '{field}' and label '{label}'")
                continue
            # Compute distance
            dist = field_config['dist'](samples_raw, samples_syn)
            # Save distance
            field_label_dists[field][label] = dist
    
    # Dictionary to store max EMD values for normalization
    max_emd_values = {}
    
    # First compute max EMD values before normalization
    for field in field_label_dists:
        dists = field_label_dists[field]
        if field_configs[field]["dist"] != stats.emd:
            continue
            
        filtered_labels = label_filter.get(field, non_raw_labels)
        
        # Calculate max EMD value for this field
        if filtered_labels and any(label in dists for label in filtered_labels):
            # Get the maximum value among filtered labels
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
    print('The json file exists, loading from it')
    with open(field_label_dists_path, "r") as f:
        field_label_dists = json.load(f)

    if folder == 'diff_syn':
        for field in field_label_dists:
            if field_configs[field]["dist"] != stats.emd:
                continue


group_positions = [] 
group_widths = []   
group_centers = [] 

group_gap = 0.4
bar_gap = 0.0
bar_width = 0.15

# Pre-calculate all positions
current_pos = 0
for i, field in enumerate(fields):
    filtered_labels = label_filter.get(field, list(field_label_dists[field].keys()))
    
    # Calculate group width for this field
    num_bars = len(filtered_labels)
    group_width = num_bars * bar_width + (num_bars - 1) * bar_gap
    
    # Store positions
    group_positions.append(current_pos)
    group_widths.append(group_width)
    
    # Calculate the true visual center of the group
    # This is the position of the first bar + half the total visual width
    first_bar_pos = current_pos
    last_bar_pos = current_pos + (num_bars - 1) * (bar_width + bar_gap)
    visual_center = first_bar_pos + (last_bar_pos - first_bar_pos) / 2 + bar_width / 2
    
    group_centers.append(visual_center)
    
    # Move to next group position
    current_pos += group_width + group_gap

# Create the figure and axes
if folder == 'cascadenet_test_cond':
    fig, ax = plt.subplots(figsize=(16, 4.8))
else:
    fig, ax = plt.subplots(figsize=(8, 4.8))

dataset_styles = {
    label: {
        'color': 'none',
        'edgecolor': vis.get_color(label),
        'hatch': vis.get_hatch(label),
        'linewidth': 2,
        'alpha': 0.99,
    } for label in all_labels if label != "raw"
}

# Now draw the bars using the pre-calculated positions - NO ADDITIONAL NORMALIZATION!
for i, field in enumerate(fields):
    dists = field_label_dists[field]  # Use the values directly, they should already be normalized
    filtered_labels = label_filter.get(field, list(dists.keys()))
    
    # Get the starting position for this group
    group_start_pos = group_positions[i]
    
    # Draw each bar for this field
    for j, label in enumerate(filtered_labels):
        if label in dists:
            bar_pos = group_positions[i] + j * (bar_width + bar_gap)
            ax.bar(
                bar_pos, 
                dists[label],
                width=bar_width,
                align='center',
                label=label if i == 0 else "",
                **dataset_styles[label]
            )

# Set x-ticks at the pre-calculated group centers
ax.set_xticks(group_centers)
ax.set_xticklabels([field_configs[field]['format']['title'] for field in fields], rotation=0)

ax.set_ylabel("Normalized EMD", fontsize=label_size)
ax.set_ylim(0, 1.4)
ax.set_yticks([0, 0.3, 0.6, 0.9])
ax.tick_params(axis='both', which='major', labelsize=tick_size)

ax.yaxis.label.set_size(label_size - 2)

for spine in ax.spines.values():
    spine.set_linewidth(1)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grid
ax.grid(True, linestyle='--', alpha=0.7)

ax.spines['bottom'].set_position('zero')
ax.tick_params(axis='both', which='major', labelsize=tick_size, direction='in')

# Legend
handles, labels_in_plot = ax.get_legend_handles_labels()
by_label = dict(zip(labels_in_plot, handles))
ordered_labels = [label for label in all_labels if label != 'raw' and label in by_label]
ordered_handles = [by_label[label] for label in ordered_labels]

ax.legend(
    ordered_handles,
    ordered_labels,
    fontsize=legend_size+7,
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

ax.set_xlim(-group_gap / 2, current_pos - group_gap + group_gap / 2)

plt.tight_layout()


if save_figure:
    vis.savefig(fig, save_config)
if show_figure:
    fig.canvas.manager.set_window_title("{} - {}".format(
        dataset,
        os.path.basename(__file__).split(".")[0]
    ))
    plt.show()