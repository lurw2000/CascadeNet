import json
import matplotlib.pyplot as plt
import numpy as np
import re
import nta.utils.vis as vis

def calculate_average_scores(statistical_data, temporal_data):
    """
    Calculate average scores by adding all JSD and EMD distances and dividing by the number of fields.
    Separate calculations for JSD and EMD metrics.
    """
    scores = {}
    
    # Initialize counters and sums for different distance types
    jsd_metrics_count = 0
    statistical_emd_metrics_count = 0
    temporal_emd_metrics_count = 0
    
    # Initialize scores dictionary
    if statistical_data:
        for key in statistical_data[list(statistical_data.keys())[0]]:
            scores[key] = {'jsd': 0.0, 'statistical_emd': 0.0, 'temporal_emd': 0.0}
    elif temporal_data:
        for key in temporal_data[list(temporal_data.keys())[0]]:
            scores[key] = {'jsd': 0.0, 'statistical_emd': 0.0, 'temporal_emd': 0.0}
    
    # Process statistical data (contains both JSD and EMD)
    if statistical_data:
        for metric in statistical_data:
            # Determine if this is a JSD or EMD metric based on the field
            is_jsd = metric in ["srcip", "dstip", "srcport", "dstport", "proto"]
            
            for key, value in statistical_data[metric].items():
                if is_jsd:
                    scores[key]['jsd'] += value
                    jsd_metrics_count += 1
                else:
                    scores[key]['statistical_emd'] += value
                    statistical_emd_metrics_count += 1
    
    # Process temporal data (all EMD)
    if temporal_data:
        for metric in temporal_data:
            for key, value in temporal_data[metric].items():
                scores[key]['temporal_emd'] += value
                temporal_emd_metrics_count += 1
    
    # Calculate averages
    for key in scores:
        if jsd_metrics_count > 0:
            scores[key]['jsd'] = scores[key]['jsd'] / jsd_metrics_count
        if statistical_emd_metrics_count > 0:
            scores[key]['statistical_emd'] = scores[key]['statistical_emd'] / statistical_emd_metrics_count
        if temporal_emd_metrics_count > 0:
            scores[key]['temporal_emd'] = scores[key]['temporal_emd'] / temporal_emd_metrics_count
    
    return scores

# List of datasets to process
datasets = ['caida', 'dc', 'ton_iot', 'ca']
folder = 'time_series_length'

# Dictionary to store results for plotting
statistical_data = {
    'jsd': {},
    'emd': {}
}

temporal_data = {
    'emd': {}
}

# Font size parameters
LEGEND_FONT_SIZE = 20
TICK_FONT_SIZE = 20
LABEL_FONT_SIZE = 22
TITLE_FONT_SIZE = 20

for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")

    # Read the statistical level JSON data
    statistical_file_path = f'../../result/evaluation/stats/{folder}/{dataset}/statistical_features_{dataset}_{dataset}.json'
    try:
        with open(statistical_file_path, 'r') as f:
            stat_data = json.load(f)
    except FileNotFoundError:
        print(f"Statistical data file not found: {statistical_file_path}")
        stat_data = None
    
    # Read the temporal level JSON data
    temporal_file_path = f'../../result/evaluation/stats/{folder}/{dataset}/temproal_features_{dataset}_{dataset}.json'
    try:
        with open(temporal_file_path, 'r') as f:
            temp_data = json.load(f)
    except FileNotFoundError:
        print(f"Temporal data file not found: {temporal_file_path}")
        temp_data = None
    
    # Ensure at least one dataset is available
    if not stat_data and not temp_data:
        print("Both statistical and temporal data files are missing!")
        continue
    
    # Calculate the average scores with separate statistical and temporal EMD
    scores = calculate_average_scores(stat_data, temp_data)
    
    # Store results for plotting - now with separate statistical and temporal EMD
    statistical_data['jsd'][dataset] = {k: v['jsd'] for k, v in scores.items()}
    statistical_data['emd'][dataset] = {k: v['statistical_emd'] for k, v in scores.items()}  # Statistical EMD only
    temporal_data['emd'][dataset] = {k: v['temporal_emd'] for k, v in scores.items()}  # Temporal EMD only
    
    # Sort the scores in ascending order (smaller value is better)
    # For display, calculate a combined score
    combined_scores = [(k, (v['jsd'] + v['statistical_emd'] + v['temporal_emd'])/3) for k, v in scores.items()]
    sorted_scores = sorted(combined_scores, key=lambda item: item[1])
    
    # Print the scores for all pkt_rate
    print("Scores (lower is better):")
    for pkt_rate, _ in sorted_scores:
        jsd_score = scores[pkt_rate]['jsd']
        statistical_emd_score = scores[pkt_rate]['statistical_emd']
        temporal_emd_score = scores[pkt_rate]['temporal_emd']
        print(f" {pkt_rate}: JSD={jsd_score:.4f}, Statistical EMD={statistical_emd_score:.4f}, Temporal EMD={temporal_emd_score:.4f}")
    
    # best_option = sorted_scores[0][0]
    # print(f"Best option for {dataset}: {best_option}")
    # print("-" * 40)

# Define the packet rates in order with their labels
packet_rates_numbers = [20, 50, 100, 200, 500, 1000, 2000]
packet_rates = [f"CN-{num}" for num in packet_rates_numbers]

# Function to extract number from packet rate string (e.g., "CN-100" -> 100)
def extract_number(pkt_rate):
    match = re.search(r'CN-(\d+)', pkt_rate)
    if match:
        return int(match.group(1))
    return 0

# Function to get proper dataset display name for legend
def get_display_name(dataset):
    if dataset == 'caida':
        return 'CAIDA'
    elif dataset == 'dc':
        return 'DC'
    elif dataset == 'ton_iot':
        return 'TON_IoT'
    elif dataset == 'ca':
        return 'CA'
    return dataset.upper()

# FIGURE 1: Statistical JSD and EMD side by side
fig1, axs1 = plt.subplots(1, 2, figsize=(16, 4.8))

# Plot types for statistical data
plot_types = ['jsd', 'emd']
y_labels = ['Average JSD', 'Average Normalized EMD']  # Updated label for statistical EMD

for idx, plot_type in enumerate(plot_types):
    # Remove top and right spines
    axs1[idx].spines['top'].set_visible(False)
    axs1[idx].spines['right'].set_visible(False)
    
    # Add grid with consistent style
    axs1[idx].grid(True, linestyle='--', alpha=0.7)
    
    # For each dataset, plot scores for each pkt_rate
    for i, dataset in enumerate(datasets):
        if dataset in statistical_data[plot_type]:
            # Get all keys for this dataset
            dataset_keys = list(statistical_data[plot_type][dataset].keys())
            
            # Sort keys by their numeric value
            dataset_keys.sort(key=extract_number)
            
            # Extract x and y values
            x_values = []
            y_values = []
            
            for key in dataset_keys:
                x_number = extract_number(key)
                if x_number in packet_rates_numbers:
                    # Find the index of this number in our ordered list
                    index = packet_rates_numbers.index(x_number)
                    x_values.append(index)  # Use the index for positioning
                    y_values.append(statistical_data[plot_type][dataset][key])
            
            if x_values and y_values:  # Only plot if we have data
                # Use consistent color and styling from vis module
                axs1[idx].plot(
                    x_values, 
                    y_values, 
                    marker='o', 
                    linewidth=2, 
                    markersize=8, 
                    label=get_display_name(dataset),
                    color=vis.get_color(get_display_name(dataset)),
                    linestyle=vis.get_linestyle(get_display_name(dataset)) if hasattr(vis, 'get_linestyle') else '-'
                )
    
    axs1[idx].set_xlabel('Time Series Length', fontsize=LABEL_FONT_SIZE)
    axs1[idx].set_ylabel(y_labels[idx], fontsize=LABEL_FONT_SIZE)
    
    # Update legend style to match second code
    axs1[idx].legend(
        fontsize=LEGEND_FONT_SIZE,
        frameon=True,
        ncol=2,
        labelspacing=0.2,
        columnspacing=1.2,
        handletextpad=0.2,
        borderaxespad=0.1,
        borderpad=0.5,
        handlelength=1.5
    )
    
    # Set consistent tick parameters
    axs1[idx].tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE-5)
    
    # Set x-ticks to use the CN-XX format
    axs1[idx].set_xticks(range(len(packet_rates)))
    axs1[idx].set_xticklabels(packet_rates)

plt.tight_layout()

# Save the first plot
plt.savefig('../../result/evaluation/stats/time_series_length/statistical_comparison.pdf', bbox_inches='tight', pad_inches=0.1)

# FIGURE 2: Temporal EMD as a separate figure
fig2, ax2 = plt.subplots(figsize=(8, 4.8))

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add grid with consistent style
ax2.grid(True, linestyle='--', alpha=0.7)

# For each dataset, plot temporal EMD scores
for dataset in datasets:
    if dataset in temporal_data['emd']:
        # Get all keys for this dataset
        dataset_keys = list(temporal_data['emd'][dataset].keys())
        
        # Sort keys by their numeric value
        dataset_keys.sort(key=extract_number)
        
        # Extract x and y values
        x_values = []
        y_values = []
        
        for key in dataset_keys:
            x_number = extract_number(key)
            if x_number in packet_rates_numbers:
                # Find the index of this number in our ordered list
                index = packet_rates_numbers.index(x_number)
                x_values.append(index)  # Use the index for positioning
                y_values.append(temporal_data['emd'][dataset][key])
        
        if x_values and y_values:  # Only plot if we have data
            # Use consistent color and styling from vis module
            ax2.plot(
                x_values, 
                y_values, 
                marker='o', 
                linewidth=2, 
                markersize=8, 
                label=get_display_name(dataset),
                color=vis.get_color(get_display_name(dataset)),
                linestyle=vis.get_linestyle(get_display_name(dataset)) if hasattr(vis, 'get_linestyle') else '-'
            )

ax2.set_xlabel('Time Series Length', fontsize=LABEL_FONT_SIZE)
ax2.set_ylabel('Average Normalized EMD', fontsize=LABEL_FONT_SIZE)  # Updated label for temporal EMD

# Update legend style to match second code
ax2.legend(
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    ncol=2,
    labelspacing=0.2,
    columnspacing=1.2,
    handletextpad=0.2,
    borderaxespad=0.1,
    borderpad=0.5,
    handlelength=1.5
)

# Set consistent tick parameters
ax2.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE-5)

# Set x-ticks to use the CN-XX format
ax2.set_xticks(range(len(packet_rates)))
ax2.set_xticklabels(packet_rates)

plt.tight_layout()

# Save the second plot with consistent padding
plt.savefig('../../result/evaluation/stats/time_series_length/temporal_comparison.pdf', bbox_inches='tight', pad_inches=0.1)

# Show all plots
plt.show()

