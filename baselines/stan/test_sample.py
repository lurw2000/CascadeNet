from stannetflow import STANSynthesizer, STANCustomDataLoader, PCAPformatTransformer
from datetime import timedelta
import pandas as pd
import glob
import argparse
import os
import time


def test_data(data, load_checkpoint=False):

  preprocess=f'../../result/stan/{data}/preprocessed.csv'
  partial = f'../../result/stan/{data}/partial_results_{data}.csv'
  
  # pkt_len, time, proto_6, proto_17, flag_DF, flag_nan, tos, ttl, spo, dpo, sip, dip
  #     0      1    （2         3）    （4         5）      6   7    dp*3, sp_sig/sp_sys/sp_other, sa*4, da*4,
  #                                                                  8      9    10-13  14-17
  train_loader = STANCustomDataLoader(preprocess, 101).get_loader()
  start_train = time.time()
  n_col, n_agg, arch_mode = 18, 100, 'B'
  discrete_columns = [[2, 3], [4, 5]]
  categorical_columns = {8:1670, 9:1670, 
                         10:256, 11:256, 12:256, 13:256, 
                         14:256, 15:256, 16:256, 17:256}
  execute_order = [0,1,2,4,6,7,8,9,10,11,12,13,14,15,16,17]

  stan = STANSynthesizer(dim_in=n_col, dim_window=n_agg, 
          discrete_columns=discrete_columns,
          categorical_columns=categorical_columns,
          execute_order=execute_order,
          arch_mode=arch_mode
          )
  
  if load_checkpoint is False:
    stan.batch_fit(train_loader, epochs=10)
  else:
    # stan.load_model('ep998') # checkpoint name
    stan.load_model('ep59', data=data) # checkpoint name

  elapsed_train = (time.time() - start_train)
  print("Train:", str(timedelta(seconds=elapsed_train)))


  start_generate = time.time()
  pcap_transformer = PCAPformatTransformer()
  samples = stan.time_series_sample(930000000, max_rows=998912, name=data)
  # samples = pd.read_csv(partial, header=None)
  print(samples)
  df_rev = pcap_transformer.rev_transfer(samples, name=data)
  elapsed_generate = (time.time() - start_generate)
  print("Generate:", str(timedelta(seconds=elapsed_generate)))
  return df_rev
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run STANSynthesizer with a specified dataset.")
  parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., 'caida')")
  args = parser.parse_args()
  test_data(args.data, True)
