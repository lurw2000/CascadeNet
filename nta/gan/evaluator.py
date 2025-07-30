import os
import itertools
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from .net import MLP, RNN, BasicModel

class Evaluator(BasicModel):
    def __init__(self, generator, device):
        super().__init__()

        self.generator = generator
        self.device = device

    def save(self, path, postfix=""):
        raise NotImplementedError

    def load(self, path, postfix=""):
        self.generator.load(path, postfix)

class ProcessorEvaluator(object):
    def __init__(self):
        return
    
    def generate(self, dataloader):

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        packetinfo_list = []
        packetfield_list = []
        with torch.no_grad():
            for condition, fivetuple, packetrate, packetinfo, packetfield in iter(dataloader):

                condition_list.append(condition.cpu())
                fivetuple_list.append(fivetuple.cpu())
                packetrate_list.append(packetrate.cpu())
                packetinfo_list.append(packetinfo.cpu())
                packetfield_list.append(packetfield.cpu())
        
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetrate_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetinfo_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetfield_list, dim=0).cpu().numpy().astype(np.float64)
        )

class FlowlevelEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader):
        self.eval()

        condition_list = []
        fivetuple_list = []
        with torch.no_grad():
            for condition, _ in iter(dataloader):

                batch_size = condition.shape[0]
                condition = condition.to(self.device)

                condition_hat, fivetuple_hat = self.generate_epoch(condition, batch_size)

                condition_list.append(condition_hat.cpu())
                fivetuple_list.append(fivetuple_hat.cpu())
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
        )

    def generate_epoch(self, condition, batch_size):
        
        condition_hat, fivetuple_hat = self.generator(condition, batch_size)

        return condition_hat, fivetuple_hat

class PacketrateEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader, length):
        self.eval()

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        with torch.no_grad():
            for condition, fivetuple, _ in iter(dataloader):

                batch_size = condition.shape[0]
                flowlevel = torch.cat([condition, fivetuple], dim=-1).to(self.device)

                packetrate_hat = self.generate_epoch(flowlevel, batch_size, length)

                condition_list.append(condition.cpu())
                fivetuple_list.append(fivetuple.cpu())
                packetrate_list.append(packetrate_hat.cpu())
        
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetrate_list, dim=0).cpu().numpy().astype(np.float64)
        )

    def generate_epoch(self, flowlevel, batch_size, length):
        
        packetrate_hat = self.generator(flowlevel, batch_size, length)

        return packetrate_hat

class PacketfieldEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader):
        self.eval()

        packetinfo_list = []
        packetfield_list = []
        with torch.no_grad():
            for packetinfo, _ in iter(dataloader):

                batch_size = packetinfo.shape[0]
                packetinfo = packetinfo.to(self.device)

                packetfield_hat = self.generate_epoch(packetinfo, batch_size)

                packetinfo_list.append(packetinfo.cpu())
                packetfield_list.append(packetfield_hat.cpu())
        
        return (
            torch.cat(packetinfo_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetfield_list, dim=0).cpu().numpy().astype(np.float64)
        )

    def generate_epoch(self, packetinfo, batch_size):
        
        packetfield_hat = self.generator(packetinfo, batch_size)

        return packetfield_hat

class CascadeEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader, length):
        self.eval()

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        with torch.no_grad():
            for condition, _, _ in iter(dataloader):

                batch_size = condition.shape[0]
                condition = condition.to(self.device)

                condition_hat, fivetuple_hat, packetrate_hat = self.generate_epoch(condition, batch_size, length)

                condition_list.append(condition_hat.cpu())
                fivetuple_list.append(fivetuple_hat.cpu())
                packetrate_list.append(packetrate_hat.cpu())
        
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetrate_list, dim=0).cpu().numpy().astype(np.float64),
        )

    def generate_epoch(self, condition, batch_size, length):
        
        condition_hat, fivetuple_hat, packetrate_hat = self.generator(condition, batch_size, length)

        return condition_hat, fivetuple_hat, packetrate_hat

class CascadeCompEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader, length):
        self.eval()

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        packetinfo_list = []
        packetfield_list = []
        with torch.no_grad():
            for condition in iter(dataloader):

                batch_size = condition.shape[0]
                condition = condition.to(self.device)

                condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat = \
                    self.generate_epoch(condition, batch_size, length)

                condition_list.append(condition_hat.cpu())
                fivetuple_list.append(fivetuple_hat.cpu())
                packetrate_list.append(packetrate_hat.cpu())
                packetinfo_list.append(packetinfo_hat.cpu())
                packetfield_list.append(packetfield_hat.cpu())
        
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetrate_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetinfo_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetfield_list, dim=0).cpu().numpy().astype(np.float64),
        )

    def generate_epoch(self, condition, batch_size, length):
        
        condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat = \
            self.generator(condition, batch_size, length)

        return condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat
    
    def generate_faster(self, processor, dataloader, length):
        self.eval()

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        
        with torch.no_grad():
            for condition in tqdm(iter(dataloader)):

                batch_size = condition.shape[0]
                condition = condition.to(self.device)

                condition_hat, fivetuple_hat, packetrate_hat = \
                    self.generate_flow_epoch(condition, batch_size, length)

                condition_list.append(condition_hat.cpu())
                fivetuple_list.append(fivetuple_hat.cpu())
                packetrate_list.append(packetrate_hat.cpu())
        
            all_condition = torch.cat(condition_list, dim=0).cpu()
            all_fivetuple = torch.cat(fivetuple_list, dim=0).cpu()
            all_packetrate = torch.cat(packetrate_list, dim=0).cpu()
            
            all_packetinfo = processor.to_packetinfo(all_condition, all_fivetuple, all_packetrate)
            packetinfo_list = torch.split(all_packetinfo, dataloader.batch_size)
            
            packetfield_list = []
            for packetinfo_hat in tqdm(packetinfo_list):
                batch_size = packetinfo_hat.shape[0]
                packetinfo_hat = packetinfo_hat.to(self.device)

                packetfield_hat = \
                    self.generate_packet_epoch(packetinfo_hat, batch_size)

                packetfield_list.append(packetfield_hat.cpu())
        
        return (
            all_condition.numpy().astype(np.float64),
            all_fivetuple.numpy().astype(np.float64),
            all_packetrate.numpy().astype(np.float64),
            all_packetinfo.numpy().astype(np.float64),
            torch.cat(packetfield_list, dim=0).cpu().numpy().astype(np.float64),
        )


    def generate_flow_epoch(self, condition, batch_size, length):
        
        condition_hat, fivetuple_hat, packetrate_hat = \
            self.generator.generate_flow(condition, batch_size, length)

        return condition_hat, fivetuple_hat, packetrate_hat
    
    def generate_packet_epoch(self, packetinfo_hat, batch_size):
        
        packetfield_hat = \
            self.generator.generate_packet(packetinfo_hat, batch_size)

        return packetfield_hat

class PacketrateFTEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader, length):
        self.eval()

        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        with torch.no_grad():
            for _, _, _, condition, fivetuple in iter(dataloader):

                batch_size = condition.shape[0]
                flowlevel = torch.cat([condition, fivetuple], dim=-1).to(self.device)

                packetrate_hat = self.generate_epoch(flowlevel, batch_size, length)

                condition_list.append(condition.cpu())
                fivetuple_list.append(fivetuple.cpu())
                packetrate_list.append(packetrate_hat.cpu())
        
        return (
            torch.cat(condition_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(fivetuple_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetrate_list, dim=0).cpu().numpy().astype(np.float64)
        )

    def generate_epoch(self, flowlevel, batch_size, length):
        
        packetrate_hat = self.generator(flowlevel, batch_size, length)

        return packetrate_hat

class PacketfieldFTEvaluator(Evaluator):
    def __init__(self, generator, device):

        super().__init__(generator, device)
    
    def generate(self, dataloader):
        self.eval()

        packetinfo_list = []
        packetfield_list = []
        with torch.no_grad():
            for _, _, packetinfo in iter(dataloader):

                batch_size = packetinfo.shape[0]

                packetfield_hat = self.generate_epoch(packetinfo, batch_size)

                packetinfo_list.append(packetinfo.cpu())
                packetfield_list.append(packetfield_hat.cpu())
        
        return (
            torch.cat(packetinfo_list, dim=0).cpu().numpy().astype(np.float64),
            torch.cat(packetfield_list, dim=0).cpu().numpy().astype(np.float64)
        )

    def generate_epoch(self, packetinfo, batch_size):
        
        packetfield_hat = self.generator(packetinfo, batch_size)

        return packetfield_hat
