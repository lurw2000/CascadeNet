import os
import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader 

class ThroughputDataset(Dataset):
    def __init__(self, sample_len, path): #TODO change name
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"))        
        self.metadata = raw_npz["metadata"]             # (num_flows, metadata_dim)
        self.extrainfo = raw_npz["extrainfo"]           # (max_len, extrainfo_dim)
        self.metaoutput = raw_npz["metaoutput"]         # (num_flows, metaoutput_dim)
        self.output = raw_npz["output"]                 # (num_flows, max_len, output_dim)
        self.output_mask = np.ones_like(self.extrainfo)
        
        #print(np.isnan(self.output).sum())
        #import pdb;pdb.set_trace()
        
        self.sample_len = sample_len

        assert self.metadata.shape[0] == self.metaoutput.shape[0] == self.output.shape[0]
        self.num_flows = self.metadata.shape[0]

        assert self.extrainfo.shape[0] == self.output.shape[1]
        self.max_len = self.extrainfo.shape[0]

        self.metadata_dim = self.metadata.shape[1]
        self.extrainfo_dim = self.extrainfo.shape[1]
        self.metaoutput_dim = self.metaoutput.shape[1]
        self.output_dim = self.output.shape[2]

        self.remainder = (-self.max_len) % sample_len
        if not self.remainder == 0:
            self.extrainfo = np.concatenate([
                    self.extrainfo,
                    np.zeros((self.remainder, self.extrainfo.shape[1]))
                ], axis=0)
            self.output = np.concatenate([
                    self.output,
                    np.zeros((self.num_flows, self.remainder, self.output.shape[2]))
                ], axis=1)
            self.output_mask = np.concatenate([
                    self.output_mask,
                    np.zeros((self.remainder, self.output_mask.shape[1]))
                ], axis=0)
        
        self.extrainfo = np.concatenate(np.split(self.extrainfo, sample_len, axis=0), axis=1)
        self.output = np.concatenate(np.split(self.output, sample_len, axis=1), axis=2)
        self.output_mask = np.concatenate(np.split(self.output_mask, sample_len, axis=0), axis=1)
    
    def __getitem__(self, index):

        fivetuple = torch.tensor(self.metadata[index], dtype=torch.float32)
        extrainfo = torch.tensor(self.extrainfo, dtype=torch.float32)
        packetrate_addi = torch.tensor(self.metaoutput[index], dtype=torch.float32)
        packetrate = torch.tensor(self.output[index], dtype=torch.float32)
        packetrate_mask = torch.tensor(self.output_mask, dtype=torch.float32)

        return fivetuple, extrainfo, packetrate_addi, packetrate, packetrate_mask
    
    def __len__(self):
        return self.num_flows
    
    def unpack_result(self, result):
        result1, result2, result3 = result
        result3 = np.concatenate(np.split(result3, self.sample_len, axis=2), axis=1)
        result3 = result3[:, :self.max_len, :]
        return result1, result2, result3 #TODO

def build_train_dataloader(dataset, config):
        
    return DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True
    )

def build_generate_dataloader(dataset, config):
    
    return DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False
    )

def build_generate_dataloader_comp(dataset, config):

    def collate_fn_comp(batch):
        
        condition_list = []
        fivetuple_list = []
        packetrate_list = []
        packetinfo_list = []
        packetfield_list = []
        for _condition, _fivetuple, _packetrate, _packetinfo, _packetfield in batch:
            condition_list.append(_condition[None, :])
            fivetuple_list.append(_fivetuple[None, :])
            packetrate_list.append(_packetrate[None, :, :])
            packetinfo_list.append(_packetinfo)
            packetfield_list.append(_packetfield)
        
        return (
            torch.cat(condition_list, dim=0),
            torch.cat(fivetuple_list, dim=0),
            torch.cat(packetrate_list, dim=0),
            torch.cat(packetinfo_list, dim=0),
            torch.cat(packetfield_list, dim=0)
        )


    return DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_comp
    )

class TraceDataset(Dataset):
    # TODO: 
    # - load raw_packetinfo and raw_packetfield from preprocess/raw_packetfield.npz
    # - load syn_packetinfo from postprocess/syn_packetfield.npz
    # - determine if shuffle is needed, since packets are stored according to order of metadata by default

    def __init__(self, path):
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetfield.npz"))
        syn_npz = np.load(os.path.join(path, "postprocess", "syn_packetfield.npz"))  
        self.packetinfo = raw_npz["packetinfo"]             # (num_packets, packetinfo_dim)
        self.packetfield = raw_npz["packetfield"]           # (num_packets, packetfield_dim)
        self.packetinfo_hat = syn_npz["packetinfo"]         # (num_packets_hat, packetinfo_dim)
        
        assert self.packetinfo.shape[0] == self.packetfield.shape[0]
        self.num_packets = self.packetinfo_hat.shape[0]

        assert self.packetinfo.shape[1] == self.packetinfo_hat.shape[1]
        self.packetinfo_dim = self.packetinfo.shape[1]
        self.packetfield_dim = self.packetfield.shape[-1] # uncertainty because of sample unit

        # resample packetinfo, packetfield into (num_packets_hat, xxx)
        random_index = list(range(self.packetinfo.shape[0]))
        random.shuffle(random_index)
        resample_index = list(range(self.packetinfo.shape[0])) * (self.num_packets // self.packetinfo.shape[0]) + random_index[:(self.num_packets % self.packetinfo.shape[0])]
        self.packetinfo = self.packetinfo[resample_index]
        self.packetfield = self.packetfield[resample_index]

    def __getitem__(self, index):

        packetinfo = torch.tensor(self.packetinfo[index], dtype=torch.float32)
        packetfield = torch.tensor(self.packetfield[index], dtype=torch.float32)
        packetinfo_hat = torch.tensor(self.packetinfo_hat[index], dtype=torch.float32)

        return packetinfo, packetfield, packetinfo_hat

    def __len__(self):
        return self.num_packets

class PacketrateDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"))        
        self.metadata = raw_npz["metadata"]             # (num_flows, metadata_dim)
        self.metaoutput = raw_npz["metaoutput"]         # (num_flows, metaoutput_dim)
        self.output = raw_npz["output"]                 # (num_flows, max_len, output_dim)

        assert self.metadata.shape[0] == self.metaoutput.shape[0] == self.output.shape[0]
        self.num_flows = self.metadata.shape[0]
        self.max_len = self.output.shape[1]

        self.metadata_dim = self.metadata.shape[1]
        self.metaoutput_dim = self.metaoutput.shape[1]
        self.output_dim = self.output.shape[2]
    
    def __getitem__(self, index):

        fivetuple = torch.tensor(self.metadata[index], dtype=torch.float32)
        packetrate_addi = torch.tensor(self.metaoutput[index], dtype=torch.float32)
        packetrate = torch.tensor(self.output[index], dtype=torch.float32)

        return fivetuple, packetrate_addi, packetrate
    
    def __len__(self):
        return self.num_flows

class ConditionDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"), allow_pickle=True)
        self.condition = raw_npz["condition"]           # (num_flows, condition_dim)   

        self.num_flows = self.condition.shape[0]
        self.max_len = raw_npz["output"].shape[1]

        self.condition_dim = self.condition.shape[1]
    
    def __getitem__(self, index):

        condition = torch.tensor(self.condition[index], dtype=torch.float32)

        return condition
    
    def __len__(self):
        return self.num_flows
class ConditionalFlowlevelDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"), allow_pickle=True)
        self.condition = raw_npz["condition"]           # (num_flows, condition_dim)   
        self.fivetuple = raw_npz["metadata"]            # (num_flows, fivetuple_dim)

        assert self.condition.shape[0] == self.fivetuple.shape[0]
        self.num_flows = self.condition.shape[0]

        self.condition_dim = self.condition.shape[1]
        self.fivetuple_dim = self.fivetuple.shape[1]
    
    def __getitem__(self, index):

        condition = torch.tensor(self.condition[index], dtype=torch.float32)
        fivetuple = torch.tensor(self.fivetuple[index], dtype=torch.float32)

        return condition, fivetuple
    
    def __len__(self):
        return self.num_flows
    

class ConditionalPacketrateDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"), allow_pickle=True)
        self.condition = raw_npz["condition"]           # (num_flows, condition_dim)   
        self.fivetuple = raw_npz["metadata"]            # (num_flows, fivetuple_dim)
        self.packetrate = raw_npz["output"]             # (num_flows, max_len, packetrate_dim)

        assert self.condition.shape[0] == self.fivetuple.shape[0] == self.packetrate.shape[0]
        self.num_flows = self.condition.shape[0]
        self.max_len = self.packetrate.shape[1]

        self.condition_dim = self.condition.shape[1]
        self.fivetuple_dim = self.fivetuple.shape[1]
        self.packetrate_dim = self.packetrate.shape[2]
    
    def __getitem__(self, index):

        condition = torch.tensor(self.condition[index], dtype=torch.float32)
        fivetuple = torch.tensor(self.fivetuple[index], dtype=torch.float32)
        packetrate = torch.tensor(self.packetrate[index], dtype=torch.float32)

        return condition, fivetuple, packetrate
    
    def __len__(self):
        return self.num_flows

class ConditionalPacketrateFTDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"), allow_pickle=True)
        syn_npz = np.load(os.path.join(path, "postprocess", "syn_flowlevel.npz"), allow_pickle=True)
        self.condition = raw_npz["condition"]           # (num_flows, condition_dim)   
        self.fivetuple = raw_npz["metadata"]            # (num_flows, fivetuple_dim)
        self.packetrate = raw_npz["output"]             # (num_flows, max_len, packetrate_dim)
        self.condition_hat = syn_npz["condition"]
        self.fivetuple_hat = syn_npz["metadata"]

        assert self.condition.shape[0] == self.fivetuple.shape[0] == self.packetrate.shape[0]
        assert self.condition_hat.shape[0] == self.fivetuple.shape[0]
        self.num_flows = self.condition.shape[0]
        self.num_flows_hat = self.condition_hat.shape[0]
        self.max_len = self.packetrate.shape[1]

        self.condition_dim = self.condition.shape[1]
        self.fivetuple_dim = self.fivetuple.shape[1]
        self.packetrate_dim = self.packetrate.shape[2]
    
    def __getitem__(self, index):

        condition = torch.tensor(self.condition[index % self.num_flows], dtype=torch.float32)
        fivetuple = torch.tensor(self.fivetuple[index % self.num_flows], dtype=torch.float32)
        packetrate = torch.tensor(self.packetrate[index % self.num_flows], dtype=torch.float32)
        condition_hat = torch.tensor(self.condition_hat[index % self.num_flows_hat], dtype=torch.float32)
        fivetuple_hat = torch.tensor(self.fivetuple_hat[index % self.num_flows_hat], dtype=torch.float32)

        return condition, fivetuple, packetrate, condition_hat, fivetuple_hat
    
    def __len__(self):
        return self.num_flows_hat if self._eval_mode else max(self.num_flows, self.num_flows_hat)

    def train_mode(self):
        self._eval_mode = False
    
    def eval_mode(self):
        self._eval_mode = True


class ConditionalPacketFieldDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetfield.npz"), allow_pickle=True)
        self.packetinfo = raw_npz["packetinfo"]           # (num_packets, packetinfo_dim)   
        self.packetfield = raw_npz["packetfield"]         # (num_packets, packetfield_dim)

        assert self.packetinfo.shape[0] == self.packetfield.shape[0]
        self.num_packets = self.packetinfo.shape[0]

        if self.packetfield.ndim == 3:
            self.field_max_len = self.packetfield.shape[1]
        else:
            self.field_max_len = None

        self.packetinfo_dim = self.packetinfo.shape[1]
        self.packetfield_dim = self.packetfield.shape[-1] # uncertainty because of sample unit
    
    def __getitem__(self, index):

        packetinfo = torch.tensor(self.packetinfo[index], dtype=torch.float32)
        packetfield = torch.tensor(self.packetfield[index], dtype=torch.float32)

        return packetinfo, packetfield
    
    def __len__(self):
        return self.num_packets

class ConditionalPacketFieldFTDataset(Dataset):
    def __init__(self, path):
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetfield.npz"), allow_pickle=True)
        syn_npz = np.load(os.path.join(path, "preprocess", "syn_packetrate.npz"), allow_pickle=True)
        self.packetinfo = raw_npz["packetinfo"]         # (num_packets, packetinfo_dim)   
        self.packetfield = raw_npz["packetfield"]       # (num_packets, packetfield_dim)
        self.packetinfo_hat = syn_npz["packetinfo"]     # (num_packets, packetinfo_dim)

        assert self.packetinfo.shape[0] == self.packetfield.shape[0]
        self.num_packets = self.packetinfo.shape[0]
        self.num_packets_hat = self.packetinfo_hat.shape[0]

        if self.packetfield.ndim == 3:
            self.field_max_len = self.packetfield.shape[1]
        else:
            self.field_max_len = None

        self.packetinfo_dim = self.packetinfo.shape[1]
        self.packetfield_dim = self.packetfield.shape[-1] # uncertainty because of sample unit
    
    def __getitem__(self, index):

        packetinfo = torch.tensor(self.packetinfo[index % self.num_packets], dtype=torch.float32)
        packetfield = torch.tensor(self.packetfield[index % self.num_packets], dtype=torch.float32)
        packetinfo_hat = torch.tensor(self.packetinfo[index % self.num_packets_hat], dtype=torch.float32)

        return packetinfo, packetfield, packetinfo_hat
    
    def __len__(self):
        return self.num_packets_hat if self._eval_mode else max(self.num_packets, self.num_packets_hat)

    def train_mode(self):
        self._eval_mode = False
    
    def eval_mode(self):
        self._eval_mode = True

class CascadeCompDataset(Dataset):
    def __init__(self, path):
        raw_npz1 = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"), allow_pickle=True)
        raw_npz2 = np.load(os.path.join(path, "preprocess", "raw_packetfield.npz"), allow_pickle=True)
        self.condition = raw_npz1["condition"]           # (num_flows, condition_dim)   
        self.fivetuple = raw_npz1["metadata"]            # (num_flows, fivetuple_dim)
        self.packetrate = raw_npz1["output"]             # (num_flows, max_len, packetrate_dim)
        self.packetinfo = raw_npz2["packetinfo"]         # (num_packets, packetinfo_dim)   
        self.packetfield = raw_npz2["packetfield"]       # (num_packets, packetfield_dim)
        self.packetindex = raw_npz2["packetindex"].astype(int)       # (num_flows+1)

        assert self.condition.shape[0] == self.fivetuple.shape[0] == self.packetrate.shape[0]
        self.num_flows = self.condition.shape[0]
        self.max_len = self.packetrate.shape[1]

        assert self.packetinfo.shape[0] == self.packetfield.shape[0]
        self.num_packets = self.packetinfo.shape[0]

        if self.packetfield.ndim == 3:
            self.field_max_len = self.packetfield.shape[1]
        else:
            self.field_max_len = None

        self.condition_dim = self.condition.shape[1]
        self.fivetuple_dim = self.fivetuple.shape[1]
        self.packetrate_dim = self.packetrate.shape[2]
        self.packetinfo_dim = self.packetinfo.shape[1]
        self.packetfield_dim = self.packetfield.shape[-1] # uncertainty because of sample unit
    
    def __getitem__(self, index):

        condition = torch.tensor(self.condition[index], dtype=torch.float32)
        fivetuple = torch.tensor(self.fivetuple[index], dtype=torch.float32)
        packetrate = torch.tensor(self.packetrate[index], dtype=torch.float32)

        if self._eval_mode:
            packetinfo = torch.tensor(self.packetinfo[self.packetindex[index]:self.packetindex[index+1]], dtype=torch.float32)
            packetfield = torch.tensor(self.packetfield[self.packetindex[index]:self.packetindex[index+1]], dtype=torch.float32)
        else:
            random_index = random.randint(self.packetindex[index], self.packetindex[index+1]-1)
            packetinfo = torch.tensor(self.packetinfo[random_index], dtype=torch.float32)
            packetfield = torch.tensor(self.packetfield[random_index], dtype=torch.float32)

        return condition, fivetuple, packetrate, packetinfo, packetfield
    
    def __len__(self):
        return self.num_flows

    def train_mode(self):
        self._eval_mode = False
    
    def eval_mode(self):
        self._eval_mode = True

    
class LatentDataset(Dataset):
    def __init__(self, sample_len, path): #TODO change name
        
        raw_npz = np.load(os.path.join(path, "preprocess", "raw_packetrate.npz"))        
        self.metadata = raw_npz["metadata"]             # (num_flows, metadata_dim)
        _extrainfo = raw_npz["extrainfo"]           # (max_len, extrainfo_dim)
        self.metaoutput = raw_npz["metaoutput"]         # (num_flows, metaoutput_dim)
        self.output = raw_npz["output"]                 # (num_flows, max_len, output_dim)
        # placeholder
        self.extrainfo = np.ones((self.output.shape[1], _extrainfo.shape[1]))
        # placeholder
        self.output_mask = np.ones_like(self.extrainfo)
        
        #print(np.isnan(self.output).sum())
        #import pdb;pdb.set_trace()
        
        self.sample_len = sample_len

        assert self.metadata.shape[0] == self.metaoutput.shape[0] == self.output.shape[0]
        self.num_flows = self.metadata.shape[0]

        assert self.extrainfo.shape[0] == self.output.shape[1]
        self.max_len = self.extrainfo.shape[0]

        self.metadata_dim = self.metadata.shape[1]
        self.extrainfo_dim = self.extrainfo.shape[1]
        self.metaoutput_dim = self.metaoutput.shape[1]
        self.output_dim = self.output.shape[2]

        self.remainder = (-self.max_len) % sample_len
        if not self.remainder == 0:
            self.extrainfo = np.concatenate([
                    self.extrainfo,
                    np.zeros((self.remainder, self.extrainfo.shape[1]))
                ], axis=0)
            self.output = np.concatenate([
                    self.output,
                    np.zeros((self.num_flows, self.remainder, self.output.shape[2]))
                ], axis=1)
            self.output_mask = np.concatenate([
                    self.output_mask,
                    np.zeros((self.remainder, self.output_mask.shape[1]))
                ], axis=0)
        
        self.extrainfo = np.concatenate(np.split(self.extrainfo, sample_len, axis=0), axis=1)
        self.output = np.concatenate(np.split(self.output, sample_len, axis=1), axis=2)
        self.output_mask = np.concatenate(np.split(self.output_mask, sample_len, axis=0), axis=1)
    
    def __getitem__(self, index):

        fivetuple = torch.tensor(self.metadata[index], dtype=torch.float32)
        extrainfo = torch.tensor(self.extrainfo, dtype=torch.float32)
        packetrate_addi = torch.tensor(self.metaoutput[index], dtype=torch.float32)
        packetrate = torch.tensor(self.output[index], dtype=torch.float32)
        packetrate_mask = torch.tensor(self.output_mask, dtype=torch.float32)

        return fivetuple, extrainfo, packetrate_addi, packetrate, packetrate_mask
    
    def __len__(self):
        return self.num_flows
    
    def unpack_result(self, result):
        result1, result2, result3 = result
        result3 = np.concatenate(np.split(result3, self.sample_len, axis=2), axis=1)
        result3 = result3[:, :self.max_len, :]
        return result1, result2, result3 #TODO
