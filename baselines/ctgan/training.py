import os
import pandas as pd
import torch
from ctgan import CTGAN

from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser(description='script to train ctgan')
    parser.add_argument('--input', type=str, help='path of input csv')
    parser.add_argument('--output', type=str, help='path of output folder')
    parser.add_argument('--type', type=str, help='type of input data')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    real_data = pd.read_csv(args.input)
    real_data.dropna()
    ctgan = CTGAN(verbose=True, cuda=True)
    if args.type == "pcap":
        discrete_columns = [
            "proto",
            "version",
            "tos",
            "flag"
        ]
    elif args.type == "netflow":
        discrete_columns = [
            "proto",
            "type"
        ]
        if "label" in real_data.columns:
            discrete_columns.append("label")
    
    print(real_data.info())
    
    ctgan.fit(real_data, discrete_columns)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    torch.save(ctgan, os.path.join(args.output, "ctgan"))
    
    