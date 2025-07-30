import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import argparse

from scipy.stats import wasserstein_distance, entropy

import nta.utils.data as data
import nta.utils.stats as stats
import nta.utils.vis as vis

def get_dataset_info(test_mode=False, test_original_path='../../test_data/original.csv', 
                     test_generated_path='../../test_data/generated.csv'):
    """
    Get dataset configuration based on mode.
    
    Args:
        test_mode (bool): If True, use test data configuration
        test_original_path (str): Path to test original CSV file
        test_generated_path (str): Path to test generated CSV file
    
    Returns:
        dict: Dataset configuration
    """
    if test_mode:
        return {
            'test': {
                'raw': test_original_path,
                'synthetic': {
                    'Generated': test_generated_path,
                }
            }
        }
    else:
        return {
            'ton_iot': {
                'raw': '../../data/ton_iot/normal_1.csv',
                'synthetic': {
                    'E-WGAN-GP': os.getenv('TON_IOT_E_WGAN_GP', 'path_to_e_wgan_gp_synthetic_csv'),
                    'REaLTabFormer': os.getenv('TON_IOT_REALTABFORMER', 'path_to_realtabformer_synthetic_csv'),
                    'NetShare': os.getenv('TON_IOT_NETSHARE', 'path_to_netshare_synthetic_csv'),
                    'CascadeNet': os.getenv('TON_IOT_CASCADENET', 'path_to_cascadenet_synthetic_csv'),
                }
            },
            'caida': {
                'raw': '../../data/caida/raw.csv',
                'synthetic': {
                    'E-WGAN-GP': os.getenv('CAIDA_E_WGAN_GP', 'path_to_e_wgan_gp_synthetic_csv'),
                    'REaLTabFormer': os.getenv('CAIDA_REALTABFORMER', 'path_to_realtabformer_synthetic_csv'),
                    'STAN': os.getenv('CAIDA_STAN', 'path_to_stan_synthetic_csv'),
                    'NetShare': os.getenv('CAIDA_NETSHARE', 'path_to_netshare_synthetic_csv'),
                    'CascadeNet': os.getenv('CAIDA_CASCADENET', 'path_to_cascadenet_synthetic_csv'),
                }
            },
            'dc': {
                'raw': '../../data/dc/raw.csv',
                'synthetic': {
                    'E-WGAN-GP': os.getenv('DC_E_WGAN_GP', 'path_to_e_wgan_gp_synthetic_csv'),
                    'REaLTabFormer': os.getenv('DC_REALTABFORMER', 'path_to_realtabformer_synthetic_csv'),
                    'STAN': os.getenv('DC_STAN', 'path_to_stan_synthetic_csv'),
                    'NetShare': os.getenv('DC_NETSHARE', 'path_to_netshare_synthetic_csv'),
                    'CascadeNet': os.getenv('DC_CASCADENET', 'path_to_cascadenet_synthetic_csv'),
                }
            },
            'ca': {
                'raw': '../../data/ca/raw.csv',
                'synthetic': {
                    'E-WGAN-GP': os.getenv('CA_E_WGAN_GP', 'path_to_e_wgan_gp_synthetic_csv'),
                    'REaLTabFormer': os.getenv('CA_REALTABFORMER', 'path_to_realtabformer_synthetic_csv'),
                    'STAN': os.getenv('CA_STAN', 'path_to_stan_synthetic_csv'),
                    'NetShare': os.getenv('CA_NETSHARE', 'path_to_netshare_synthetic_csv'),
                    'CascadeNet': os.getenv('CA_CASCADENET', 'path_to_cascadenet_synthetic_csv'),
                }
            },
        }

def calculate_time_granularity(df, time_column, num_points):
    df[time_column] = pd.to_datetime(df[time_column], unit='us')

    start_time = df[time_column].min()
    end_time = df[time_column].max()
    total_time_span = end_time - start_time

    # for test use: Handle case where all timestamps are the same or very small dataset
    if total_time_span.total_seconds() == 0:
        # Return a default interval of 1 second in microseconds
        return 1000000
    
    time_granularity = total_time_span / num_points
    granularity_us = int(time_granularity.total_seconds() * 1e6)
    
    # Ensure minimum granularity of 1 microsecond
    return max(granularity_us, 1)


# Function to aggregate data
def aggregate_data(df, time_column, pkt_len_column, interval):
    df[time_column] = pd.to_datetime(df[time_column], unit='us')
    
    # Replace negative pkt_len values with 1
    df[pkt_len_column] = df[pkt_len_column].apply(lambda x: 1 if x < 0 else x)
    
    # Set the time_column as the index
    df = df.set_index(time_column)
    
    # Perform aggregation
    df_agg = df[pkt_len_column].resample(f'{interval}U').sum().reset_index()
    df_agg['index'] = df_agg.index

    return df_agg

# Function to calculate utilization
def calculate_utilization(df_agg, pkt_len_column):
    max_utilization = df_agg[pkt_len_column].quantile(1.0)
    df_agg['utilization'] = df_agg[pkt_len_column] / max_utilization
    return df_agg

# Function to identify burst and non-burst periods
def identify_bursts(df_agg, utilization_column, threshold=0.5):
    df_agg['burst'] = df_agg[utilization_column] > threshold
    # print(df_agg)
    return df_agg

# Function to calculate burst durations and time between bursts
def calculate_burst_metrics(df_agg, interval):
    burst_durations = []
    time_between_bursts = []

    in_burst = False
    burst_start = None

    # Calculate burst durations
    for i in range(len(df_agg)):
        if df_agg['burst'].iloc[i]:
            if not in_burst:
                burst_start = i
                in_burst = True
            if i == len(df_agg) - 1 or not df_agg['burst'].iloc[i + 1]:
                burst_durations.append((i - burst_start + 1) * interval)
                in_burst = False
        elif in_burst:
            burst_durations.append((i - burst_start) * interval)
            in_burst = False

    # Calculate time between bursts
    last_burst_end = None
    i = 0
    while i < len(df_agg):
        if df_agg['burst'].iloc[i]:
            if last_burst_end is not None and i != last_burst_end + 1:
                time_between_bursts.append((i - last_burst_end - 1) * interval)
            while i < len(df_agg) and df_agg['burst'].iloc[i]:
                last_burst_end = i
                i += 1
        i += 1

    # Ensure lists are not empty before comparison
    if not burst_durations:
        burst_durations = [0]  
    if not time_between_bursts:
        time_between_bursts = [0] 

    return burst_durations, time_between_bursts

def calculate_packet_sizes(df_agg):
    burst_pkt_sizes = []
    non_burst_pkt_sizes = []
    
    in_burst = False
    current_burst_size = 0
    current_non_burst_size = 0
    
    for i in range(len(df_agg)):
        if df_agg['burst'].iloc[i]:
            if not in_burst:
                if current_non_burst_size > 0:
                    non_burst_pkt_sizes.append(current_non_burst_size)
                    current_non_burst_size = 0
                in_burst = True
            current_burst_size += df_agg['pkt_len'].iloc[i]
        else:
            if in_burst:
                burst_pkt_sizes.append(current_burst_size)
                current_burst_size = 0
                in_burst = False
            current_non_burst_size += df_agg['pkt_len'].iloc[i]
    
    if current_burst_size > 0:
        burst_pkt_sizes.append(current_burst_size)
    if current_non_burst_size > 0:
        non_burst_pkt_sizes.append(current_non_burst_size)
    
    return burst_pkt_sizes, non_burst_pkt_sizes


# Function to compare distributions using Wasserstein Distance or JSD
def compare_distributions(dist1, dist2, method='EMD'):
    dist1 = np.array(dist1).flatten()
    dist2 = np.array(dist2).flatten()

    # Handle empty distributions
    if len(dist1) == 0 or len(dist2) == 0:
        # Mark dataset as having no burst by setting normalized EMD to 1.0
        return 1.0

    if method == 'EMD':
        return wasserstein_distance(dist1, dist2)

    elif method == 'JSD':
        p = np.array(dist1) / sum(dist1)
        q = np.array(dist2) / sum(dist2)
        m = (p + q) / 2
        return (entropy(p, m) + entropy(q, m)) / 2
    else:
        raise ValueError("Invalid method: choose 'EMD' or 'JSD'")

# Function to normalize comparison results
def normalize_comparisons(results):
    max_value = max(results, default=1)  # Set default to avoid division by zero

    # Avoid division by zero
    if max_value == 0:
        max_value = 1  # Set to 1 to avoid division by zero

    return [0.9*(r / max_value) for r in results]

# Main function to perform the analysis
def perform_burst_analysis(filepaths, method='EMD', test_mode=False):
    results = {
        'Burst Durations': [],
        'Time Between Bursts': [],
        'Network Utilization': [],
        'Burst Packet Sizes': [],
        'Non-Burst Packet Sizes': []
    }
    
    raw_df = pd.read_csv(filepaths['raw'])
    
    if test_mode:
        num_points = 5
        print(f"Running in test")
    else:
        num_points = 1000 
    
    interval = calculate_time_granularity(raw_df, 'time', num_points)
    raw_agg = aggregate_data(raw_df, 'time', 'pkt_len', interval)
    raw_agg = calculate_utilization(raw_agg, 'pkt_len')
    raw_agg = identify_bursts(raw_agg, 'utilization')
    raw_burst_durations, raw_time_between_bursts = calculate_burst_metrics(raw_agg, interval)
    raw_burst_pkt_sizes, raw_non_burst_pkt_sizes = calculate_packet_sizes(raw_agg)
    
    for key in filepaths['synthetic']:
        synthetic_df = pd.read_csv(filepaths['synthetic'][key])
        synthetic_agg = aggregate_data(synthetic_df, 'time', 'pkt_len', interval)
        synthetic_agg = calculate_utilization(synthetic_agg, 'pkt_len')
        synthetic_agg = identify_bursts(synthetic_agg, 'utilization')
        synthetic_burst_durations, synthetic_time_between_bursts = calculate_burst_metrics(synthetic_agg, interval)
        synthetic_burst_pkt_sizes, synthetic_non_burst_pkt_sizes = calculate_packet_sizes(synthetic_agg)
        
        results['Burst Durations'].append(compare_distributions(raw_burst_durations, synthetic_burst_durations, method))
        results['Time Between Bursts'].append(compare_distributions(raw_time_between_bursts, synthetic_time_between_bursts, method))
        results['Network Utilization'].append(compare_distributions(raw_agg['utilization'], synthetic_agg['utilization'], method))
        results['Burst Packet Sizes'].append(compare_distributions([raw_burst_pkt_sizes], [synthetic_burst_pkt_sizes], method))
        results['Non-Burst Packet Sizes'].append(compare_distributions([raw_non_burst_pkt_sizes], [synthetic_non_burst_pkt_sizes], method))
    
    normalized_results = {key: normalize_comparisons(results[key]) for key in results}

    # scores
    scores = [0] * len(next(iter(normalized_results.values())))

    for key in normalized_results:
        for i in range(len(normalized_results[key])):
            scores[i] += normalized_results[key][i]

    for i in range(len(scores)):
        scores[i] = 1 / scores[i]

    max_score = max(scores)

    # Normalize scores
    for i in range(len(scores)):
        scores[i] = scores[i] / max_score
    
    return normalized_results

def plot_results(results, synthetic_methods, name, test_mode=False, epoch=None):
    categories = ['BD', 'TB', 'NU', 'BPS', 'NPS']
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_width = 0.14
    index = np.arange(len(categories))

    for i, method in enumerate(synthetic_methods):
        values = [results[category][i] for category in results.keys()]
        ax.bar(index + i * bar_width, values, bar_width,
               label=method, color='none', 
               hatch=vis.get_hatch(method),
               edgecolor=vis.get_color(method), linewidth=2, alpha=0.99)

    # Adjust the font size and labels
    ax.set_ylabel('Normalized EMD', fontsize=22)
    ax.set_xticks(index + bar_width * (len(synthetic_methods) - 1) / 2)
    ax.set_xticklabels(categories, fontsize=22)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.3, 0.6, 0.9])
    ax.set_yticklabels([0, 0.3, 0.6, 0.9], fontsize=22)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    legend_size = 23

    ax.legend(
        fontsize=legend_size,
        ncol=2,
        loc='upper left',
        frameon=True,
        labelspacing=0.2,
        columnspacing=1.2,
        handletextpad=0.2, 
        borderaxespad=0.1, 
        borderpad=0.5, 
        handlelength=1.5, 
    )

    # Define the relative path to save the figures
    save_dir = '../../test_result/evaluation/' if test_mode else '../../result/evaluation/burst_analysis/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.savefig(f'{save_dir}burst_{name}.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def main(test_mode=False, test_original_path='../../test_data/original.csv', 
         test_generated_path='../../test_data/generated.csv', datasets=None):
    """
    Main function to run burst analysis.
    
    Args:
        test_mode (bool): If True, run in test mode with test data
        test_original_path (str): Path to test original CSV file
        test_generated_path (str): Path to test generated CSV file
        datasets (list): List of dataset names to process (if None, processes all)
    """
    dataset_info = get_dataset_info(test_mode, test_original_path, test_generated_path)
    
    if test_mode:
        dataset_names = ['test']
    else:
        dataset_names = datasets if datasets is not None else ['caida', 'ca', 'dc', 'ton_iot']

    for name in dataset_names:
        if name not in dataset_info:
            print(f"Warning: Dataset '{name}' not found in configuration. Skipping...")
            continue
            
        print(f"Processing dataset: {name}")
        dataset = dataset_info[name]
        results = perform_burst_analysis(dataset, method='EMD', test_mode=test_mode)
        synthetic_methods = list(dataset['synthetic'].keys())
        plot_results(results, synthetic_methods, name, test_mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run burst analysis on network data')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode using test data')
    parser.add_argument('--test-original', default='../../test_data/original.csv',
                       help='Path to test original CSV file (default: ../../test_data/original.csv)')
    parser.add_argument('--test-generated', default='../../test_data/generated.csv',
                       help='Path to test generated CSV file (default: ../../test_data/generated.csv)')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['caida', 'ca', 'dc', 'ton_iot'],
                       help='Specific datasets to process (default: all)')
    
    args = parser.parse_args()
    
    main(test_mode=args.test, 
         test_original_path=args.test_original,
         test_generated_path=args.test_generated,
         datasets=args.datasets)