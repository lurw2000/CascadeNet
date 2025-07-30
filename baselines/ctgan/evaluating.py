import os
import pandas as pd
import torch
from ctgan import CTGAN

from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser(description='script to evaluate ctgan')
    parser.add_argument('--input', type=str, help='path of input csv')
    parser.add_argument('--output', type=str, help='path of output folder')
    parser.add_argument('--type', type=str, help='type of input data')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    real_data = pd.read_csv(args.input)
    ctgan = torch.load(os.path.join(args.output, "ctgan"))
    synthetic_data = ctgan.sample(len(real_data))
    
    synthetic_data.to_csv(os.path.join(args.output, "syn.csv"), index=False)
    
    