""" 
This is an argparser that should be imported in the task script, so that we can pass arguments to task. 

See evaluation/stats/task/trace/timeseries.py for an example.
"""
import argparse
import yaml
import os

with open("config/general.yaml") as f:
	# Loader=yaml.SafeLoader for version of yaml
	general_config = yaml.load(f, Loader=yaml.SafeLoader)
save_figure = general_config["save_figure"]
show_figure = general_config["show_figure"]
overwrite_history_result = general_config["overwrite_history_result"]
normalization_csv = general_config["normalization_csv"]
save_normalization = general_config["save_normalization"]

# Create the parser
parser = argparse.ArgumentParser(description="Run statistical evaluation of netowrk trace")

# Add an argument
parser.add_argument('--config', type=str, help='Path to configuration file')
parser.add_argument('--folder', type=str, help='Folder name (e.g., diff_syn, ablation)', default=None)
parser.add_argument('--test', action='store_true', help='Run in test mode', default=False)

# Parse the arguments
args = parser.parse_args()

with open(args.config) as f:
	# Loader=yaml.SafeLoader for version of yaml
	dataset_config = yaml.load(f, Loader=yaml.SafeLoader)

class DATASET_TYPE:
	PCAP = "pcap"
	NETFLOW = "netflow"

dataset = dataset_config["dataset"]
dataset_type = dataset_config["type"]
label2path = dataset_config["label2path"]

# Get folder from either command line argument or config file
folder = args.folder
if folder is None:
    folder = dataset_config.get("folder", os.path.basename(os.path.dirname(args.config)))

test = args.test
# Ensure that each dataset has a raw trace
if "raw" not in label2path:
	raise ValueError("raw trace not found in label2path of dataset: {}".format(dataset))

# Now you can use args.input to access the input file name
print("Running config:\n{}".format(dataset_config))

