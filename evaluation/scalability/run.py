import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import nta.utils.vis as vis
import matplotlib
from matplotlib.patches import Rectangle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def get_dataset_config(test_mode=False):
    """
    Get dataset and model configuration based on mode.
    
    Args:
        test_mode (bool): If True, use test configuration
    
    Returns:
        tuple: (datasets, models, dataset_model_sequence)
    """
    if test_mode:
        datasets = ['TEST']
        models = ['Generated']
        dataset_model_sequence = {
            'TEST': ['Generated'],
        }
    else:
        datasets = ['CAIDA', 'CA', 'DC', 'TON_IoT']
        models = ['CTGAN', 'E-WGAN-GP', 'STAN', 'REaLTabFormer', 'NetShare', 'NetDiffusion', 'CascadeNet']
        dataset_model_sequence = {
            'CAIDA': ['CTGAN', 'E-WGAN-GP', 'STAN', 'REaLTabFormer', 'NetShare', 'CascadeNet'],
            'CA': ['CTGAN', 'E-WGAN-GP', 'STAN', 'REaLTabFormer', 'NetShare', 'CascadeNet'],
            'DC': ['CTGAN', 'E-WGAN-GP', 'STAN', 'REaLTabFormer', 'NetShare', 'CascadeNet'],
            'TON_IoT': ['CTGAN', 'E-WGAN-GP', 'REaLTabFormer', 'NetShare', 'NetDiffusion', 'CascadeNet'],
        }
    
    return datasets, models, dataset_model_sequence

def create_bar_plot(data, metric_name, ylabel, save_filename, test_mode=False):
    datasets, models, dataset_model_sequence = get_dataset_config(test_mode)
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_width = 0.12
    index = np.arange(len(datasets))
    
    for dataset_idx, dataset in enumerate(datasets):
        model_sequence = dataset_model_sequence.get(dataset, [])
        print(f"Processing dataset: {dataset}")
        
        for model_idx, model in enumerate(model_sequence):
            subset = data[(data['Dataset'] == dataset) & (data['Trace Generator'] == model)]
            if not subset.empty:
                value = subset.iloc[0][metric_name]
                print(f"Model: {model}, Value: {value}")
                x_pos = dataset_idx + (model_idx - len(model_sequence)/2 + 0.5) * bar_width
                ax.bar(x_pos, value, bar_width,
                       label=model if dataset_idx == 0 else "",
                       edgecolor=vis.get_color(model),
                       facecolor='none',
                       hatch=vis.get_hatch(model),
                       linewidth=1.25
                )
            else:
                print(f"Model: {model} - No data found.")

    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xticks(index)
    ax.set_xticklabels(datasets, fontsize=22)
    
    if not test_mode:
        ax.set_yscale('log')
        # Set y-axis limits and ticks for log scale
        y_min = max(0.001, min(data[metric_name][data[metric_name] > 0]) * 0.8)
        y_max = max(data[metric_name]) * 50
        ax.set_ylim(y_min, y_max)
        
        # Generate logarithmic tick locations
        y_ticks = [10**i for i in range(int(np.log10(y_min)), int(np.log10(y_max))+1)]
        ax.set_yticks(y_ticks)
        
        # Format y-tick labels to hide 10^-1
        y_tick_labels = [f'$10^{{{int(np.log10(y))}}}$' if y >= 1 else '' for y in y_ticks]
        ax.set_yticklabels(y_tick_labels, fontsize=22)
    else:
        # Linear scale for test mode with simple values
        ax.ticklabel_format(style='plain', axis='y')
        ax.tick_params(axis='y', labelsize=22)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Create custom legend handles and labels for all models
    legend_handles = [Rectangle((0,0),1,1, facecolor='none',
                               edgecolor=vis.get_color(model),
                               hatch=vis.get_hatch(model),
                               linewidth=1.25) for model in models]
    legend_labels = models

    # Adjust legend
    ncol = 3
    ax.legend(legend_handles, legend_labels,
              fontsize=17,
              ncol=ncol,
              loc='upper left',
              frameon=True,
              labelspacing=0.2,
              columnspacing=1.2,
              handletextpad=0.2,
              borderaxespad=0.1,
              borderpad=0.5,
              handlelength=1.5,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    plt.savefig(save_filename, format='pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def main(test_mode=False, test_csv_path='test_runtime.csv'):
    """
    Main function to run scalability analysis.
    
    Args:
        test_mode (bool): If True, run in test mode with test data
        test_csv_path (str): Path to test CSV file
    """
    # Load data
    if test_mode:
        csv_data = test_csv_path
        result_dir = '../../test_result/evaluation/'
        print("Running in test mode")
        
        training_time_filename = 'training_time_test.pdf'
        generating_time_filename = 'generating_time_test.pdf'
    else:
        csv_data = 'runtime.csv'
        result_dir = '../../result/evaluation/scalability/'
        
        training_time_filename = 'training_time.pdf'
        generating_time_filename = 'generating_time.pdf'

    # Ensure result directory exists
    os.makedirs(result_dir, exist_ok=True)

    # Load data
    if not os.path.exists(csv_data):
        print(f"Error: CSV file not found: {csv_data}")
        return

    data = pd.read_csv(csv_data)
    print(f"Loaded data from {csv_data}")
    # print(f"Data shape: {data.shape}")

    # Create plots with appropriate file names
    training_time_path = os.path.join(result_dir, training_time_filename)
    create_bar_plot(data, 'Training Time (hours)', 'Training Time (hours)', 
                    training_time_path, test_mode)

    generating_time_path = os.path.join(result_dir, generating_time_filename)
    create_bar_plot(data, 'Generating Time (minutes)', 'Generating Time (minutes)', 
                    generating_time_path, test_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run scalability analysis on runtime data')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode using test data')
    parser.add_argument('--test-csv', default='test_runtime.csv',
                       help='Path to test CSV file (default: test_runtime.csv)')
    
    args = parser.parse_args()
    
    main(test_mode=args.test, test_csv_path=args.test_csv)