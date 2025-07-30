from stannetflow import PCAPformatTransformer, STANTemporalTransformer
from datetime import timedelta
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Run STANSynthesizer with a specified dataset.")
parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., 'caida')")
args = parser.parse_args()
name = args.data

raw = f'../../data/{name}/raw.csv'
sample = f'../../result/stan/{name}/sampled_marginal.csv'
preprocessed = f'../../result/stan/{name}/preprocessed.csv'

df = pd.read_csv(raw)
columns_to_drop = ['version', 'ihl', 'id', 'off']
df = df.drop(columns=columns_to_drop)

sampled_data = {}
n_samples = 101
for col in df.columns:
    sampled_data[col] = df[col].sample(n=n_samples, replace=True).values
sampled_df = pd.DataFrame(sampled_data)
sampled_df.to_csv(sample, index=False)
print(sampled_df)


def _prepare_data(input='', output=''):
  pcapt = PCAPformatTransformer()
  tft = STANTemporalTransformer(output)
  df = pd.read_csv(input)
  # tft.push_back(df, agg=6, transformer=pcapt)
  df_processed = pcapt.transfer(df, name=name)
  df_processed.to_csv(output, mode='a', header=False, index=False)

_prepare_data(sample, preprocessed)