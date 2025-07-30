import json
import torch
from nta.builder import CascadeGANCompBuilder
from argparse import ArgumentParser

def parse_args():

    parser = ArgumentParser(description='script to evaluate cascadegan')
    parser.add_argument('--path', type=str, help='path of configuration')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config = json.load(open(args.path, "r"))

    builder = CascadeGANCompBuilder(config)

    builder.build_all(
        device=torch.device("cuda"),
        preprocess=True
    )
    builder.generate_copy()