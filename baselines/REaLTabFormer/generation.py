# This script is based on the REaLTabFormer repository.
# Source: https://github.com/worldbank/REaLTabFormer
# Original paper: Solatorio, A. V., & Dupriez, O. (2023). 
# "REaLTabFormer: Generating Realistic Relational and Tabular Data using Transformers." 
# arXiv:2302.02041. Available at: https://arxiv.org/abs/2302.02041


# pip install realtabformer
import pandas as pd
from realtabformer import REaLTabFormer

import argparse
import torch
import os

parser = argparse.ArgumentParser(description="Tabular data generation using REaLTabFormer")
parser.add_argument("--input_path", type=str, default="../../data/caida/raw.csv", help="Path to the input trace data")
parser.add_argument("--output_path", type=str, default="../../result/realtabformer/caida", help="Path to the output generated data")
args = parser.parse_args()

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

df = pd.read_csv(args.input_path)

if torch.cuda.is_available():
    no_cuda = False
else:
    no_cuda = True

rtf_model = REaLTabFormer(
    model_type="tabular",
    gradient_accumulation_steps=4,
    logging_steps=5000,
    save_steps=5000,
    save_total_limit=2,
    fp16=True,
    no_cuda=False,
    dataloader_num_workers=4,
    report_to="none"
)

rtf_model.fit(df)

rtf_model.save(os.path.join(args.output_path, "rtf_model"))

samples = rtf_model.sample(n_samples=len(df))
samples.to_csv(os.path.join(args.output_path, "realtabformer.csv"), index=False)