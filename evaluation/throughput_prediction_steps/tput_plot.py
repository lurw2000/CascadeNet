import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nta.utils.vis as vis
import math
import os
import argparse

def get_dataset_configs(test_mode=False):
    """
    Get dataset configuration based on mode.
    
    Args:
        test_mode (bool): If True, use test configuration
    
    Returns:
        dict: Dataset configurations
    """
    if test_mode:
        return {
            'test': {
                'datasets': ["test"],
                'generators': ["Generated"],
                'hatches': ["x"],
                'colors': ["tab:red"]
            }
        }
    else:
        return {
            'main': {
                'datasets': ["CAIDA", "CA", "DC"],
                'generators': ["E-WGAN-GP", "STAN", "REaLTabFormer", "NetShare", "CascadeNet"],
                'hatches': ["*", "o", ".", "-", "/"],
                'colors': ["tab:orange", "tab:brown", "tab:pink", "tab:blue", "tab:red"]
            },
            'ton': {
                'datasets': ["TON"],
                'generators': ["E-WGAN-GP", "REaLTabFormer", "NetShare", "CascadeNet"],
                'hatches': ["*", ".", "-", "/"],
                'colors': ["tab:orange", "tab:pink", "tab:blue", "tab:red"]
            }
        }

def load_and_process_data(input_folder, datasets, generators, stats, models, test_mode=False):
    results = {}
    for dataset in datasets:
        results[dataset] = {}
        for stat in stats:
            results[dataset][stat] = {}
            
            # Handle different file naming conventions
            if test_mode:
                raw_file = os.path.join(input_folder, f"throughput_{dataset}-raw.csv")
            else:
                raw_file = os.path.join(input_folder, f"{dataset}-raw.csv")
            
            if not os.path.exists(raw_file):
                print(f"Warning: Raw file not found: {raw_file}")
                continue
            
            ratio_raw = pd.read_csv(raw_file)
            ratio_raw = ratio_raw[ratio_raw["stat"] == stat]
            
            for generator in generators:
                # Handle different file naming conventions
                if test_mode:
                    generator_file = os.path.join(input_folder, f"throughput_{dataset}-{generator}.csv")
                else:
                    generator_file = os.path.join(input_folder, f"{dataset}-{generator}.csv")
                
                if not os.path.exists(generator_file):
                    print(f"Warning: Generator file not found: {generator_file}")
                    continue
                
                try:
                    ratio_g = pd.read_csv(generator_file)
                    ratio_g = ratio_g[ratio_g["stat"] == stat]
                    normalized_mae = {}
                    for model in models:
                        if model not in ratio_raw["model"].values or model not in ratio_g["model"].values:
                            print(f"Warning: Model {model} not found in {dataset}-{generator} for {stat}")
                            continue
                        
                        raw_mae = ratio_raw[ratio_raw["model"] == model]["MAE"].values
                        g_mae = ratio_g[ratio_g["model"] == model]["MAE"].values
                        
                        if len(raw_mae) == 0 or len(g_mae) == 0:
                            print(f"Warning: No data for model {model} in {dataset}-{generator} for {stat}")
                            continue
                        
                        # Handle division by zero and invalid values
                        with np.errstate(divide='ignore', invalid='ignore'):
                            normalized = g_mae / raw_mae
                        
                        # Filter out invalid values (inf, -inf, nan)
                        valid_normalized = normalized[np.isfinite(normalized)]
                        
                        if len(valid_normalized) == 0:
                            if test_mode:
                                print(f"Warning: All normalized values are invalid for model {model} in {dataset}-{generator} for {stat}")
                            # Use a default value of 1.0 when all values are invalid
                            normalized_mae[model] = {
                                'mean': 1.0,
                                'se': 0.0
                            }
                        else:
                            if test_mode:
                                capped_normalized = np.clip(valid_normalized, 0, 10) 
                                if not np.array_equal(valid_normalized, capped_normalized):
                                    print(f"  Warning: Capped large values for model {model}")
                                final_normalized = capped_normalized
                            else:
                                # Production mode - no capping
                                final_normalized = valid_normalized
                            
                            normalized_mae[model] = {
                                'mean': np.mean(final_normalized),
                                'se': np.std(final_normalized, ddof=1) / np.sqrt(len(final_normalized)) if len(final_normalized) > 1 else 0
                            }

                    results[dataset][stat][generator] = normalized_mae
                except Exception as e:
                    print(f"Error processing {generator_file}: {e}")
    return results

def plot_results(results, generators, models, dataset, stat, output_folder, config, test_mode=False):
    if dataset not in results or stat not in results[dataset]:
        print(f"No data available for dataset {dataset} and statistic {stat}")
        return
    
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bar_width = 0.3 if test_mode else 0.15
    index = np.arange(len(models))

    hatches = config['hatches']
    colors = config['colors']

    for i, generator in enumerate(generators):
        if generator not in results[dataset][stat]:
            print(f"No data available for generator {generator} in dataset {dataset} for statistic {stat}")
            continue
        
        means = []
        errors = []
        for model in models:
            if (model in results[dataset][stat][generator] and 
                np.isfinite(results[dataset][stat][generator][model]['mean']) and
                np.isfinite(results[dataset][stat][generator][model]['se'])):
                means.append(results[dataset][stat][generator][model]['mean'])
                errors.append(results[dataset][stat][generator][model]['se'])
            else:
                means.append(0)
                errors.append(0)
        
        # Plot the bars
        bars = ax.bar(index + i * bar_width, means, bar_width,
                      label=generator, color='none', hatch=hatches[i],
                      edgecolor=colors[i], linewidth=2, alpha=0.99)
        
        # Add error bars
        ax.errorbar(index + i * bar_width, means, yerr=errors, fmt='none', 
                    ecolor=colors[i], capsize=3, capthick=2, linewidth=2)

    ax.set_ylabel('Normalized MAE', fontsize=20)
    ax.set_xlabel('')
    ax.set_xticks(index + bar_width * (len(generators) - 1) / 2)
    ax.set_xticklabels(models, fontsize=20)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Calculate y-axis limits and ticks
    if any(generator in results[dataset][stat] for generator in generators):
        # Collect all valid y-values for determining limits
        all_y_values = []
        for generator in results[dataset][stat]:
            if generator in results[dataset][stat]:
                for model in results[dataset][stat][generator]:
                    if model in results[dataset][stat][generator]:
                        mean_val = results[dataset][stat][generator][model]['mean']
                        se_val = results[dataset][stat][generator][model]['se']
                        if np.isfinite(mean_val) and np.isfinite(se_val):
                            all_y_values.append(mean_val + se_val)
        
        if test_mode:
            print(f"All y-values for {dataset}-{stat}: {all_y_values}")
        
        if all_y_values:
            y_max = max(all_y_values)
            
            # Handle case where y_max is 0 or very small
            if y_max <= 0 or not np.isfinite(y_max):
                y_max = 10 
            
            # Only apply capping in test mode
            if test_mode and y_max > 15:
                y_max = 15
                
            def find_best_interval(y_max):
                potential_intervals = [1, 2, 5, 10, 20, 50, 100]
                for interval in potential_intervals:
                    if y_max / interval <= 5:
                        return interval
                return potential_intervals[-1]

            interval = find_best_interval(y_max)
            num_ticks = math.ceil(y_max / interval) + 1
            adjusted_y_max = interval * num_ticks
            yticks = [i * interval for i in range(num_ticks)]
            y_limit = adjusted_y_max

            ax.set_ylim(0, y_limit)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f'{int(x)}' for x in yticks], fontsize=20)
        else:
            # No valid data, use default limits
            if test_mode:
                print("No valid data, using default limits")
            ax.set_ylim(0, 10)
            ax.set_yticks([0, 2, 4, 6, 8, 10])
            ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=20)
    else:
        ax.set_ylim(0, 10)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_yticklabels(['0', '2', '4', '6', '8', '10'], fontsize=20)

    # Adjust legend for test mode
    ncol = 1 if test_mode else 2
    legend = ax.legend(
        fontsize=20,
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

    # Handle matplotlib deprecation warning
    try:
        legend_handles = legend.legend_handles
    except AttributeError:
        legend_handles = legend.legendHandles
    
    for i, handle in enumerate(legend_handles):
        handle.set_hatch(hatches[i])

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save plots
    if test_mode:
        plt.savefig(os.path.join(output_folder, f'tput_plot_test_{dataset}_{stat}.pdf'), 
                bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join(output_folder, f'tput_plot_test_{dataset}_{stat}.svg'), 
                format="svg", bbox_inches='tight', pad_inches=0.1)
    else:
        plt.savefig(os.path.join(output_folder, f'tput_plot_{dataset}_{stat}.pdf'), 
                    bbox_inches='tight', pad_inches=0.1)
        plt.savefig(os.path.join(output_folder, f'tput_plot_{dataset}_{stat}.svg'), 
                    format="svg", bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main(test_mode=False, test_input_folder=None, test_output_folder=None):
    """
    Main function to run throughput prediction analysis.
    
    Args:
        test_mode (bool): If True, run in test mode with test data
        test_input_folder (str): Path to test input folder
        test_output_folder (str): Path to test output folder
    """
    # Configuration
    if test_mode:
        input_folder = test_input_folder or "../../test_result/evaluation/"
        output_folder = test_output_folder or "../../test_result/evaluation/"
        print("Running in test mode")
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
    else:
        input_folder = "../../result/evaluation/throughput_prediction_steps/"
        output_folder = "../../result/evaluation/throughput_prediction_steps/"
    
    stats = ["throughput"]
    models = ["AB", "AR", "ARMA", "DT", "KNN", "RF"]

    # Get dataset configurations based on mode
    dataset_configs = get_dataset_configs(test_mode)

    # Process each dataset configuration
    for config_name, config in dataset_configs.items():
        print(f"Processing {config_name} datasets...")
        results = load_and_process_data(
            input_folder, 
            config['datasets'],
            config['generators'],
            stats,
            models,
            test_mode
        )

        # Generate plots for each dataset and stat
        for dataset in config['datasets']:
            for stat in stats:
                plot_results(
                    results,
                    config['generators'],
                    models,
                    dataset,
                    stat,
                    output_folder,
                    config,
                    test_mode
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run throughput prediction analysis')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode using test data')
    parser.add_argument('--test-input', default='../../test_result/evaluation/',
                       help='Path to test input folder (default: ../../test_result/evaluation/)')
    parser.add_argument('--test-output', default='../../test_result/evaluation/',
                       help='Path to test output folder (default: ../../test_result/evaluation/)')
    
    args = parser.parse_args()
    
    main(test_mode=args.test,
         test_input_folder=args.test_input,
         test_output_folder=args.test_output)