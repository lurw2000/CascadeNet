import os
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import nta.utils.vis as vis
import numpy as np
import argparse

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
            'TEST': {
                'raw': test_original_path,
                'synthetic': {
                    'Generated': test_generated_path,
                }
            }
        }
    else:
        return {
            'CAIDA': {
                'raw': '../../data/caida/raw.csv',
                'synthetic': {
                    'CTGAN': '../../result/ctgan/caida/syn.csv',
                    'E-WGAN-GP': '../../result/e-wgan-gp/caida/syn.csv',
                    'REaLTabFormer': '../../result/realtabformer/caida/realtabformer.csv',
                    'NetShare': '../../result/netshare/caida/post_processed_data/syn.csv',
                    'CascadeNet': '../../result/cascadenet/caida-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv',
                    'STAN': '../../result/stan/caida/syn.csv',
                    'NetDiffusion': None,
                }
            },
            'DC': {
                'raw': '../../data/dc/raw.csv',
                'synthetic': {
                    'CTGAN': '../../result/ctgan/dc/syn.csv',
                    'E-WGAN-GP': '../../result/e-wgan-gp/dc/syn.csv',
                    'REaLTabFormer': '../../result/realtabformer/dc/realtabformer.csv',
                    'NetShare': '../../result/netshare/dc/post_processed_data/syn.csv',
                    'CascadeNet': '../../result/cascadenet/dc-feature_True-zero_flag_True-rate_100/postprocess/syn_comp.csv',
                    'STAN': '../../result/stan/dc/syn.csv',
                    'NetDiffusion': None,
                }
            },
            'CA': {
                'raw': '../../data/ca/raw.csv',
                'synthetic': {
                    'CTGAN': '../../result/ctgan/ca/syn.csv',
                    'E-WGAN-GP': '../../result/e-wgan-gp/ca/syn.csv',
                    'REaLTabFormer': '../../result/realtabformer/ca/realtabformer.csv',
                    'NetShare': '../../result/netshare/ca/post_processed_data/syn.csv',
                    'CascadeNet': '../../result/cascadenet/ca-feature_True-zero_flag_True-rate_20/postprocess/syn_comp.csv',
                    'STAN': '../../result/stan/ca/syn.csv',
                    'NetDiffusion': None,
                }
            },
            'TON_IoT': {
                'raw': '../../data/ton_iot/normal_1.csv',
                'synthetic': {
                    'CTGAN': '../../result/ctgan/ton/syn.csv',
                    'E-WGAN-GP': '../../result/e-wgan-gp/ton_iot/syn.csv',
                    'REaLTabFormer': '../../result/realtabformer/ton/realtabformer.csv',
                    'NetShare': '../../result/netshare/ton_iot/post_processed_data/syn.csv',
                    'CascadeNet': '../../result/cascadenet/ton-feature_True-zero_flag_True-rate_200/postprocess/syn_comp.csv',
                    'STAN': None,
                    'NetDiffusion': '../../result/netdiffusion/reconstructed_ton.csv',
                }
            },
        }

def analyze(real_path, generated_path):
    # srcip,dstip,srcport,dstport,ttl,pkt_len,proto,time
    if not os.path.exists(generated_path):
        raise FileNotFoundError('File not found, maybe because the model collapsed when training:', generated_path)
    
    fields = ['srcip', 'dstip', 'srcport', 'dstport', 'proto', 'ttl', 'pkt_len']
    real = pd.read_csv(real_path)[fields].round()
    if real['proto'].dtype == object:
        real['proto'] = real['proto'].map({'TCP': 6, 'UDP': 17})
    real = real.astype(int).to_numpy()
    
    generated = pd.read_csv(generated_path)[fields].round()
    if generated['proto'].dtype == object:
        generated['proto'] = generated['proto'].map({'TCP': 6, 'UDP': 17})
    generated = generated.astype(int).to_numpy()

    real = [tuple(x) for x in real]
    generated = [tuple(x) for x in generated]

    set_real = set(real)
    leaked = [x for x in generated if x in set_real]

    return len(leaked) / len(generated)

def main(test_mode=False, test_original_path='../../test_data/original.csv', 
         test_generated_path='../../test_data/generated.csv', output_format='csv', 
         force_compute=False):
    """
    Main function to run syn_real_div analysis.
    
    Args:
        test_mode (bool): If True, run in test mode with test data
        test_original_path (str): Path to test original CSV file
        test_generated_path (str): Path to test generated CSV file
        output_format (str): Format to save data ('json', 'csv', or 'both')
        force_compute (bool): If True, force recomputation even if cache exists
    """
    # Get dataset configuration
    dataset_info = get_dataset_info(test_mode, test_original_path, test_generated_path)
    
    # Get the absolute path to the script
    script_path = os.path.abspath(__file__)
    
    # Go up to reach ML-testing-dev directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
    
    # Create the result directory based on mode
    if test_mode:
        result_dir = os.path.join(project_root, "test_result", "evaluation")
        print("Running in test mode")
        json_file = os.path.join(result_dir, 'syn_real_div_test.json')
        csv_file = os.path.join(result_dir, 'syn_real_div_test.csv')
    else:
        result_dir = os.path.join(project_root, "result", "evaluation", "syn_real_div")
        json_file = os.path.join(result_dir, 'syn_real_div.json')
        csv_file = os.path.join(result_dir, 'syn_real_div.csv')

    os.makedirs(result_dir, exist_ok=True)
    
    # Font size parameters
    LEGEND_FONT_SIZE = 20
    TICK_FONT_SIZE = 20
    LABEL_FONT_SIZE = 20
    TITLE_FONT_SIZE = 20
    
    # Compute or load from JSON if it exists
    div_info = defaultdict(dict)
    
    if output_format in ['json', 'both']:
        try:
            if os.path.exists(json_file) and os.path.getsize(json_file) > 0 and not force_compute:
                with open(json_file, 'r') as f:
                    div_info = json.load(f)
                print(f"Loaded div data from {json_file}")
            else:
                force_compute = True
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading JSON file: {e}")
            force_compute = True
    
    # If we need to compute the data (either because we're forced to or because the JSON doesn't exist)
    if force_compute or len(div_info) == 0:
        print("Computing div data...")
        for dataset in dataset_info:
            raw_path = dataset_info[dataset]['raw']
            for model in dataset_info[dataset]['synthetic']:
                syn_path = dataset_info[dataset]['synthetic'][model]
                if syn_path is None:
                    div = None
                else:
                    try:
                        div = 1 - analyze(raw_path, syn_path)
                        print(f"Computed div for {dataset} - {model}: {div:.5f}")
                    except Exception as e:
                        print(f"Error computing div for {dataset} - {model}: {e}")
                        div = None
                div_info[dataset][model] = div
    
        # Save div_info to JSON file if specified
        if output_format in ['json', 'both']:
            with open(json_file, 'w') as f:
                json.dump(div_info, f, indent=4)
            print(f"Saved div data to {json_file}")
    
    # Save to CSV if specified
    if output_format in ['csv', 'both']:
        # Convert the nested dictionary to a dataframe
        # Rows are generators, columns are datasets
        if test_mode:
            models = ['Generated']  # Only one model in test mode
        else:
            models = ['CTGAN', 'E-WGAN-GP', 'STAN', 'REaLTabFormer', 'NetShare', 'NetDiffusion', 'CascadeNet']
        
        datasets = list(div_info.keys())
        
        # Create a DataFrame with models as index
        df = pd.DataFrame(index=models, columns=datasets)
        
        # Fill the DataFrame with div values
        for dataset in datasets:
            for model in models:
                df.loc[model, dataset] = div_info[dataset].get(model)
        
        # Save to CSV
        df.to_csv(csv_file)
        print(f"Saved div data to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze divergence between real and synthetic data and generate output files')
    parser.add_argument('--test', action='store_true', 
                       help='Run in test mode using test data')
    parser.add_argument('--test-original', default='../../test_data/original.csv',
                       help='Path to test original CSV file (default: ../../test_data/original.csv)')
    parser.add_argument('--test-generated', default='../../test_data/generated.csv',
                       help='Path to test generated CSV file (default: ../../test_data/generated.csv)')
    parser.add_argument('--output_format', type=str, choices=['json', 'csv', 'both'], default='csv',
                        help='Format to save the data: json, csv, or both')
    parser.add_argument('--force-compute', action='store_true',
                        help='Force recomputation even if cached results exist')
    
    args = parser.parse_args()
    
    main(test_mode=args.test,
         test_original_path=args.test_original,
         test_generated_path=args.test_generated,
         output_format=args.output_format,
         force_compute=args.force_compute)