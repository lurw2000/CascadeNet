import os
import itertools
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .net import MLP, RNN, RNNC, bRNNC, BasicModel

class Discriminator(BasicModel):
    def __init__(self):
        super().__init__()


class FlowlevelDiscriminator(Discriminator):
    def __init__(self, condition_dim, fivetuple_dim, config, device):

        super().__init__()

        self.device = device

        self.discriminator = MLP(
            input_size=condition_dim + fivetuple_dim,
            output_size=1,
            config=config["mlp"]
        )
    
    def forward(self, condition, fivetuple):

        return self.discriminator(
            torch.cat([condition, fivetuple], dim=-1)
        )


class PacketrateDiscriminator(Discriminator):
    def __init__(self, flowlevel_dim, packetrate_dim, length, config, device):

        super().__init__()

        self.device = device

        if "mlp" in config:
            self.mode = "mlp"
            self.discriminator = MLP(
                input_size=flowlevel_dim + packetrate_dim * length,
                output_size=1,
                config=config["mlp"]
            )
        elif "brnnc" in config:
            self.mode = "brnnc"
            self.sample_len = config["sample_len"]
            self.discriminator = bRNNC(
                input_size=flowlevel_dim + packetrate_dim * self.sample_len,
                output_size=self.sample_len,
                config=config["brnnc"]
            )
    
    def forward(self, flowlevel, packetrate):
        if self.mode == "mlp":
            return self.discriminator(
                torch.cat([
                    flowlevel,
                    torch.flatten(packetrate, start_dim=1)
                ], dim=-1)
            )
        elif self.mode == "brnnc":
            batch_size = packetrate.shape[0]
            length = packetrate.shape[1]
            
            if length % self.sample_len != 0:
                packetrate = torch.cat([
                    packetrate,
                    torch.zeros(packetrate.shape[0], self.sample_len - length % self.sample_len, packetrate.shape[2]).to(self.device)
                ], dim=1)
            
            to_return = self.discriminator(
                torch.cat([
                    flowlevel[:,None,:].expand(-1, (length - 1) // self.sample_len + 1, -1),
                    packetrate.view(batch_size, -1, packetrate.shape[2] * self.sample_len)
                ], dim=-1),
                1
            )

            return to_return.view(batch_size, -1, to_return.shape[2] // self.sample_len)[:, :length, :]


class PacketfieldDiscriminator(Discriminator):
    def __init__(self, packetinfo_dim, packetfield_dim, length, config, device):

        super().__init__()

        self.device = device

        if length is not None:
            self.mode = "sequence"
            self.discriminator = MLP(
                input_size=packetinfo_dim + packetfield_dim * length,
                output_size=1,
                config=config["mlp"]
            )
        
        else:
            self.mode = "single"
            self.discriminator = MLP(
                input_size=packetinfo_dim + packetfield_dim,
                output_size=1,
                config=config["mlp"]
            )
    
    def forward(self, packetinfo, packetfield):

        if self.mode == "sequence":
            return self.discriminator(
                torch.cat([
                    packetinfo,
                    torch.flatten(packetfield, start_dim=1)
                ], dim=-1)
            )
        else:
            return self.discriminator(
                torch.cat([
                    packetinfo,
                    packetfield
                ], dim=-1)
            )

      
class CascadeDiscriminator(Discriminator):
    def __init__(self, flowlevel_discriminator, packetrate_discriminator):
        
        super().__init__()

        self.flowlevel_discriminator = flowlevel_discriminator
        self.packetrate_discriminator = packetrate_discriminator
    
    def forward(self, condition, fivetuple, packetrate):

        prob_flowlevel = self.flowlevel_discriminator(condition, fivetuple)
        flowlevel = torch.cat([condition, fivetuple], dim=-1)
        prob_packetrate = self.packetrate_discriminator(flowlevel, packetrate)

        return prob_flowlevel, prob_packetrate
    
    def save(self, path, postfix=""):
        self.flowlevel_discriminator.save(path, postfix)
        self.packetrate_discriminator.save(path, postfix)

    def load(self, path, postfix=""):
        self.flowlevel_discriminator.load(path, postfix)
        self.packetrate_discriminator.load(path, postfix)


class CascadeCompDiscriminator(Discriminator):
    def __init__(self, flowlevel_discriminator, packetrate_discriminator, packetfield_discriminator):
        
        super().__init__()

        self.flowlevel_discriminator = flowlevel_discriminator
        self.packetrate_discriminator = packetrate_discriminator
        self.packetfield_discriminator = packetfield_discriminator
    
    def forward(self, condition, fivetuple, packetrate, packetinfo, packetfield):

        prob_flowlevel = self.flowlevel_discriminator(condition, fivetuple)
        flowlevel = torch.cat([condition, fivetuple], dim=-1)
        prob_packetrate = self.packetrate_discriminator(flowlevel, packetrate)
        prob_packetfield = self.packetfield_discriminator(packetinfo, packetfield)

        return prob_flowlevel, prob_packetrate, prob_packetfield
    
    def save(self, path, postfix=""):
        self.flowlevel_discriminator.save(path, postfix)
        self.packetrate_discriminator.save(path, postfix)
        self.packetfield_discriminator.save(path, postfix)

    def load(self, path, postfix=""):
        self.flowlevel_discriminator.load(path, postfix)
        self.packetrate_discriminator.load(path, postfix)
        self.packetfield_discriminator.load(path, postfix)
    
