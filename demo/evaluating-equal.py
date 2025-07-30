import json
from nta.builder import CascadeGANCompBuilder
from argparse import ArgumentParser

import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def parse_args():

    parser = ArgumentParser(description='script to evaluate cascadegan')
    parser.add_argument('--path', type=str, help='path of configuration')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = json.load(open(args.path, "r"))

    config["pre_post_processor"]["timestamp_recovery"]["method"] = "equidistant"

    builder = CascadeGANCompBuilder(config)

    builder.build_all(
        device=torch.device("cuda"),
        preprocess=False
    )
    # builder.evaluate_all(ppf=".equal")
    builder.evaluate(ppf=".equal")