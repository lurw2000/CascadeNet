import nta.utils.vis as vis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from argparse import ArgumentParser
import glob
from pathlib import Path

def parse_args():
    parser = ArgumentParser(description='Plot NetML results')
    parser.add_argument('--base_dir', type=str, 
                      default='../../result/evaluation/anomly_detection_cross', 
                      help='Base directory containing results')
    parser.add_argument('--datasets', nargs='+', default=['caida'],
                      help='List of datasets')
    parser.add_argument('--generators', nargs='+', default=['REaLTabFormer', 'NetShare', 'CascadeNet'],
                      help='List of generators')
    parser.add_argument('--ndms', nargs='+', 
                      default=['OCSVM', 'IForest', 'GMM', 'AE', 'PCA', 'KDE'],
                      help='List of NDMs to analyze')
    return parser.parse_args()

def load_and_process_data(base_dir, datasets, generators, selected_ndms):
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for generator in generators:
            results[dataset][generator] = {}
            for ndm in selected_ndms:
                try:
                    # Construct path using new directory structure
                    csv_path = os.path.join(base_dir, generator, 
                                          f"{dataset}_{generator}_{ndm}.csv")
                    print(f"Looking for file: {csv_path}")
                    
                    if not os.path.exists(csv_path):
                        print(f"Warning: File not found: {csv_path}")
                        continue
                        
                    data = pd.read_csv(csv_path)
                    
                    # Group by model and calculate mean F1 score
                    model_means = data.groupby('model')['f1_score'].mean().reindex(
                        ['SAMP_NUM', 'SAMP_SIZE', 'IAT', 'SIZE', 'IAT_SIZE', 'STATS']
                    )
                    
                    results[dataset][generator][ndm] = {
                        "F1 score": model_means.values.tolist()
                    }
                except Exception as e:
                    print(f"Error processing {csv_path}: {e}")
    
    return results

def plot_results(results, generators, dataset, ndm, metric="F1 score", base_dir=None):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_width = 0.25
    index = np.arange(6)  # 6 NetML models
    
    for i, generator in enumerate(generators):
        if ndm not in results[dataset][generator]:
            print(f"Warning: No data for {generator} with {ndm}")
            continue
            
        values = results[dataset][generator][ndm][metric]
        ax.bar(index + i * bar_width, values, bar_width,
               label=generator, color='none',
               hatch=vis.get_hatch(generator),
               edgecolor=vis.get_color(generator),
               linewidth=2, alpha=0.99)
    
    ax.set_ylabel(metric, fontsize=22)
    ax.set_xlabel('')
    ax.set_xticks(index + bar_width * (len(generators) - 1) / 2)
    ax.set_xticklabels(['SN', 'SS', 'IAT', 'SIZE', 'IS', 'STATS'], fontsize=22)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1], fontsize=22)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    legend_size = 24
    ax.legend(fontsize=legend_size, ncol=2, loc='upper left',
             frameon=True, labelspacing=0.2, columnspacing=1.2,
             handletextpad=0.2, borderaxespad=0.1, borderpad=0.5,
             handlelength=1.5)
    
    plt.tight_layout()
    
    # Save plots in the same directory as the input files
    output_base = os.path.join(base_dir, f"{ndm}_{dataset.lower()}_{metric}")
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight', pad_inches=0.1)
    # plt.savefig(f"{output_base}.svg", format="svg", bbox_inches='tight', pad_inches=0.1)
    
    plt.close(fig)

def main():
    args = parse_args()
    
    # Convert dataset names to lowercase to match new structure
    datasets = [d.lower() for d in args.datasets]
    
    # Load and process data
    results = load_and_process_data(args.base_dir, datasets, args.generators, args.ndms)
    
    # Generate plots
    for dataset in datasets:
        for ndm in args.ndms:
            print(f"Generating plot for {dataset} - {ndm}")
            plot_results(results, args.generators, dataset, ndm, "F1 score", args.base_dir)

if __name__ == "__main__":
    main()