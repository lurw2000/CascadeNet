"""
This file is an example usage of pre_post_processor
"""

import nta.pre_post_processor.pre_post_processor as ppp 
import json 
import numpy as np
import pandas as pd
import os 
import nta.utils.data as data
import nta.utils.stats as stats
import time
import pickle
import matplotlib.pyplot as plt
import nta.model.data as mdata
import nta.model.waterfallgan as gan
import torch

config_path = "config/config-ppp.json"
print("Using config file:\n\t{}".format(config_path))
with open(config_path, "r") as config_file:
    config = json.load(config_file)

processor = ppp.PrePostProcessor(config=config["pre_post_processor"])    

# Preprocessing
print("\nppp_test: pre-processing\n")
print("Using prepostprocessor: {}".format(type(processor)))
processor.pre_process()
print("\nCheck if everything is saved correctly:")
# load raw_packet_rate.npz and raw_packet_field.npz
raw_packet_rate = np.load(os.path.join(processor.preprocess_folder, "raw_packetrate.npz"))
raw_packet_field = np.load(os.path.join(processor.preprocess_folder, "raw_packetfield.npz"))

raw_condition = raw_packet_rate["condition"]
raw_metadata = raw_packet_rate["metadata"]
# raw_extrainfo = raw_packet_field["extrainfo"]
raw_metaoutput = raw_packet_rate["metaoutput"]
raw_output = raw_packet_rate["output"]
raw_packetrate = raw_packet_rate["packetrate"]

raw_packetinfo = raw_packet_field["packetinfo"]
raw_packetfield = raw_packet_field["packetfield"]

r = processor.metadata_post_process(raw_metadata)
print(r[0])



if raw_condition is not None:
    print("raw condition shape: {}".format(raw_condition.shape))
else:
    print("raw condition shape: None")
print("\tmin: {}, max: {}".format(raw_condition.min(), raw_condition.max()))
print("raw metadata shape: {}".format(raw_metadata.shape))
print("\tmin: {}, max: {}".format(raw_metadata.min(), raw_metadata.max()))
# print("raw extrainfo shape: {}".format(raw_extrainfo.shape))
# print("\tmin: {}, max: {}".format(raw_extrainfo.min(), raw_extrainfo.max()))
print("raw metaoutput shape: {}".format(raw_metaoutput.shape))
print("\tmin: {}, max: {}".format(raw_metaoutput.min(), raw_metaoutput.max()))
print("raw output shape: {}".format(raw_output.shape))
print("\tmin: {}, max: {}".format(raw_output.min(), raw_output.max()))
print("raw packetrate shape: {}".format(raw_packetrate.shape))
print("\tmin: {}, max: {}".format(raw_packetrate.min(), raw_packetrate.max()))
print("raw packetinfo shape: {}".format(raw_packetinfo.shape))
print("\tmin: {}, max: {}".format(raw_packetinfo.min(), raw_packetinfo.max()))
print("raw packetfield shape: {}".format(raw_packetfield.shape))
print("\tmin: {}, max: {}".format(raw_packetfield.min(), raw_packetfield.max()))


exit()

# Train and Generate
# -----------------REAL MODEL GENERATE----------------------
dataset = mdata.ThroughputDataset(sample_len=config["sample_len"], path=config["pre_post_processor"]["output_folder"])

train_dataloader = mdata.build_train_dataloader(dataset=dataset, config=config["dataloader"])
generate_dataloader = mdata.build_generate_dataloader(dataset=dataset, config=config["dataloader"])

device = torch.device("cpu")
model = gan.WaterfallGAN(
    length=(dataset.max_len + dataset.remainder) // config["sample_len"],
    sample_len=config["sample_len"],
    fivetuple_dim=dataset.metadata_dim,
    extrainfo_dim=dataset.extrainfo_dim * config["sample_len"],
    packetrate_addi_dim=dataset.metaoutput_dim,
    packetrate_dim=dataset.output_dim * config["sample_len"],
    config=config["model"],
    device=device
).to(device)

packetrate_result = model.generate(generate_dataloader)
packetrate_result = dataset.unpack_result(packetrate_result)


# -----------------REAL MODEL GENERATE----------------------


# Postprocessing
print("\n\nppp_test: packet rate post-processing\n\n")

processor.packetrate_post_process(packetrate_result)

syn_packetrate_result = np.load(os.path.join(processor.postprocess_folder, "syn_packetrate.npz"))
syn_metadata = syn_packetrate_result["metadata"]
syn_metaoutput = syn_packetrate_result["metaoutput"]
syn_output = syn_packetrate_result["output"]
syn_packetrate = syn_packetrate_result["packetrate"]

print("syn metadata shape: {}".format(syn_metadata.shape))
print("\tmin: {}, max: {}".format(syn_metadata.min(), syn_metadata.max()))
print("syn metaoutput shape: {}".format(syn_metaoutput.shape))
print("\tmin: {}, max: {}".format(syn_metaoutput.min(), syn_metaoutput.max()))
print("syn output shape: {}".format(syn_output.shape))
print("\tmin: {}, max: {}".format(syn_output.min(), syn_output.max()))
print("syn packetrate shape: {}".format(syn_packetrate.shape))
print("\tmin: {}, max: {}".format(syn_packetrate.min(), syn_packetrate.max()))


print("\n\nppp_test: packet field pre-processing\n\n")

processor.trace_pre_process()

syn_packetfield_result = np.load(os.path.join(processor.postprocess_folder, "syn_packetfield.npz"))
syn_packetinfo = syn_packetfield_result["packetinfo"]

print("syn packetinfo shape: {}".format(syn_packetinfo.shape))
print("\tmin: {}, max: {}".format(syn_packetinfo.min(), syn_packetinfo.max()))

exit()

print("\n\nppp_test: packet field post-processing\n\n")

processor.trace_post_process()

"""
# extract raw trace from
# - packetrate (stored in raw_packetrate.npz)
# - packetinfo, packetfield (stored in raw_packetfield.npz)
[v] processor.pre_process()

result_123 = gen123()

[v] processor.packetrate_post_process(result_123)

# packet rate => timestamp
# NOTE: this should be done for both raw and syn trace (or just syn trace, and use real timestamp for raw trace (configurable option))
[v] processor.trace_pre_process()

result_4 = gen4()          # separable from gen123 training (similar to preprocess, train, and generate)

# => csv ( => pcap )
[ ] processor.trace_post_process(result_123, result_4)

"""

