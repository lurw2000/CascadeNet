"""
generator components
"""
import typing
import os
import itertools
import importlib

import torch
from torch import nn
import torch.nn.functional as F

from .net import MLP, RNN, RNNC, bRNNC, BasicModel

class Generator(BasicModel):
    """
    a basic model for generator
    """
    def __init__(self):
        super().__init__()


class FlowlevelGenerator(Generator):
    """
    IP 5-tuples Generator

    Parameters
    ----------
    condition_dim
        dimension of the input/output condition
    fivetuple_dim
        dimension of the output IP 5-tuples
    condition_gen_flag
        If true, the generator will generate IP 5-tuples and condition together, ignoring the given condition;
        If false, the generator will generate IP 5-tuples based on the given condition.
    condition_normalization
        the last layer of condition generator. For example, if `normalization` equals to `([nn.Sigmoid(), nn.Softmax(dim=-1)], [10, 3])`,
        the first 10 dimension will go through sigmoid on the last layer, and the last 3 dimension will go through softmax on the last layer.
    fivetuple_normalization
        the last layer of fivetuple generator. For example, if `normalization` equals to `([nn.Sigmoid(), nn.Softmax(dim=-1)], [10, 3])`,
        the first 10 dimension will go through sigmoid on the last layer, and the last 3 dimension will go through softmax on the last layer.
    config
        configuration dictionary
    device
        the device where the model is located

    """
    def __init__(self,
                 condition_dim: int,
                 fivetuple_dim: int,
                 condition_gen_flag: bool,
                 condition_normalization: typing.Tuple[typing.List[torch.nn.Module], typing.List[int]],
                 fivetuple_normalization: typing.Tuple[typing.List[torch.nn.Module], typing.List[int]],
                 config: typing.Mapping[str, typing.Any],
                 device: torch.device) -> None:

        super().__init__()

        self.condition_gen_flag = condition_gen_flag
        self.noise_dim = config["noise_dim"]
        self.device = device
        self.split_size = [condition_dim, fivetuple_dim]

        if self.condition_gen_flag:
            self.generator = MLP(
                input_size=self.noise_dim,
                output_size=condition_dim + fivetuple_dim,
                config=config["mlp"]
            )
            self.normalization = [
                condition_normalization[0] + fivetuple_normalization[0],
                condition_normalization[1] + fivetuple_normalization[1],
            ]
        else:
            self.generator = MLP(
                input_size=condition_dim + self.noise_dim,
                output_size=fivetuple_dim,
                config=config["mlp"]
            )
            self.normalization = fivetuple_normalization
    
    def forward(self, condition: torch.Tensor, batch_size: int) -> typing.Tuple[torch.Tensor]:
        """
        the forward process

        Parameters
        ----------
        condition
            the input condition. Only when `self.condition_gen_flag` is false, the condition will be used.
        batch_size
            the size of minibatch

        Returns
        -------
        If `self.condition_gen_flag` is false, return the input condition and the generated IP 5-tuples;
        If `self.condition_gen_flag` is true, return the generated condition and IP 5-tuples.

        """

        if self.condition_gen_flag:
            to_return = self.generator(
                torch.randn(batch_size, self.noise_dim).to(self.device)
            )
        else:
            to_return = self.generator(
                torch.cat([
                    condition,
                    torch.randn(batch_size, self.noise_dim).to(self.device)
                ], dim=-1)
            )

        to_return = torch.cat(
            [norm(x) for x, norm in zip(torch.split(to_return, self.normalization[1], dim=-1), self.normalization[0])], dim=-1
        )

        
        if self.condition_gen_flag:
            return torch.split(to_return, self.split_size, dim=-1)
        else:
            return condition, to_return
    

class PacketrateGenerator(Generator):
    """
    Aggregated Time Series Generator

    Parameters
    ----------
    flowlevel_dim
        dimension of the flowlevel input
    packetrate_dim
        dimension of the time series output
    packetrate_normalization
        the last layer of packetrate generator. For example, if `normalization` equals to `([nn.Sigmoid(), nn.Softmax(dim=-1)], [10, 3])`,
        the first 10 dimension will go through sigmoid on the last layer, and the last 3 dimension will go through softmax on the last layer.
    config
        configuration dictionary
    device
        the device where the model is located

    """
    def __init__(self,
                 flowlevel_dim: int,
                 packetrate_dim: int,
                 packetrate_normalization: typing.Tuple[typing.List[torch.nn.Module], typing.List[int]],
                 config: typing.Mapping[str, typing.Any],
                 device: torch.device) -> None:

        super().__init__()

        self.third_activation_label = config["third_activation_label"]
        self.flow_time_adjustment = config["flow_time_adjustment"]
        if "zero_flag" in config and not config["zero_flag"]:
            self.zero_flag = False
        else:
            self.zero_flag = True
        self.noise_dim = config["noise_dim"]
        self.sample_len = config["sample_len"]
        self.device = device

        if "rnn" in config:
            self.mode = "rnn"
            self.generator = RNN(
               input_size=flowlevel_dim + self.noise_dim,
                output_size=packetrate_dim * self.sample_len,
                config=config["rnn"]
            )
        elif "rnnc" in config:
            self.mode = "rnnc"
            self.generator = RNNC(
                input_size=flowlevel_dim + self.noise_dim,
                output_size=packetrate_dim * self.sample_len,
                config=config["rnnc"]
            )
        elif "mlp" in config:
            self.mode = "mlp"
            self.generator = MLP(
                input_size=flowlevel_dim + self.noise_dim,
                output_size=packetrate_dim * self.sample_len,
                config=config["mlp"]
            )

        self.normalization = packetrate_normalization
    
    def forward(self, flowlevel: torch.Tensor, batch_size: int, length: int) -> torch.Tensor:
        """
        the forward process

        Parameters
        ----------
        flowlevel
            the input flowlevel
        batch_size
            the size of minibatch
        length
            the length of the output time series

        Returns
        -------
        the generated time series

        """
        
        if self.mode == "mlp":

            to_return = self.generator(
                torch.cat([
                    flowlevel,
                    torch.randn(batch_size, self.noise_dim).to(self.device)
                ], dim=-1)
            )

            to_return = to_return.view(batch_size, self.sample_len, -1)[:, :length, :]
        
        else:

            to_return = self.generator(
                torch.cat([
                    flowlevel[:,None,:].expand(-1, (length - 1) // self.sample_len + 1, -1),
                    torch.randn(batch_size, (length - 1) // self.sample_len + 1, self.noise_dim).to(self.device)
                ], dim=-1),
                1
            )
            to_return = to_return.view(batch_size, -1, to_return.shape[2] // self.sample_len)[:, :length, :]


        to_return = torch.cat(
            [norm(x) for x, norm in zip(torch.split(to_return, self.normalization[1], dim=-1), self.normalization[0])], dim=-1
        )
        
        def continuous_mask(mask, reverse=False):
            return_mask = mask
            if reverse:
                tmp_mask = return_mask[:, -1]
                for i in range(return_mask.shape[1]):
                    return_mask[:, -(i+1)] = return_mask[:, -(i+1)] * tmp_mask
                    tmp_mask = return_mask[:, -(i+1)]
            else:
                tmp_mask = return_mask[:, 0]
                for i in range(return_mask.shape[1]):
                    return_mask[:, i] = return_mask[:, i] * tmp_mask
                    tmp_mask = return_mask[:, i]
            return return_mask
        
        import warnings
        warnings.warn(f"reached packetrate generator branch {self.third_activation_label} {self.zero_flag} {self.flow_time_adjustment} {to_return.requires_grad}")
        if self.third_activation_label:
            # output_zero_flag
            if self.zero_flag:
                zero_mask = (to_return[:, :, -5] > to_return[:, :, -4]).float()
                if self.flow_time_adjustment:
                    zero_mask[:, 0] = 1 #force
                to_return[:, :, :-5] = to_return[:, :, :-5] * zero_mask[:, :, None]

            if self.flow_time_adjustment:
                mask = ((to_return[:, :, -3] > to_return[:, :, -2]) & (to_return[:, :, -3] > to_return[:, :, -1])).float()
                mask[:, 0] = 1 # force
                mask = continuous_mask(mask)
                
                to_return[:, :, :-3] = to_return[:, :, :-3] * mask[:, :, None]
            else:
                start_mask = ((to_return[:, :, -3] > to_return[:, :, -2]) & (to_return[:, :, -3] > to_return[:, :, -1])).float()
                start_duration_mask = ((to_return[:, :, -2] > to_return[:, :, -1]) | (to_return[:, :, -3] > to_return[:, :, -1])).float()
                start_mask = continuous_mask(start_mask)
                start_duration_mask = continuous_mask(start_duration_mask)
                
                to_return[:, :, :-3] = to_return[:, :, :-3] * (1 - start_mask[:, :, None]) * start_duration_mask[:, :, None]
        
        else:
            # output_zero_flag
            if self.zero_flag:
                zero_mask = (to_return[:, :, -4] > to_return[:, :, -3]).float()
                zero_mask[:, 0] = 1 #force
                to_return[:, :, :-4] = to_return[:, :, :-4] * zero_mask[:, :, None]
            mask = ((to_return[:, :, -2] > to_return[:, :, -1])).float()
            mask[:, 0] = 1 # force
            mask = continuous_mask(mask)
            
            to_return[:, :, :-2] = to_return[:, :, :-2] * mask[:, :, None]
            
        return to_return


class PacketfieldGenerator(Generator):
    """
    Measurement Fields Generator

    Parameters
    ----------
    packetinfo_dim
        dimension of the information necessary to measurement field generation, which includes IP 5-tuples and information of corresponding time series step
    packetfield_dim
        dimension of the output measurement fields
    packetfield_normalization
        the last layer of packetfield generator. For example, if `normalization` equals to `([nn.Sigmoid(), nn.Softmax(dim=-1)], [10, 3])`,
        the first 10 dimension will go through sigmoid on the last layer, and the last 3 dimension will go through softmax on the last layer.
    config
        configuration dictionary
    device
        the device where the model is located

    """
    def __init__(self,
                 packetinfo_dim: int,
                 packetfield_dim: int, 
                 packetfield_normalization: typing.Tuple[typing.List[torch.nn.Module], typing.List[int]],
                 config: typing.Mapping[str, typing.Any],
                 device: torch.device) -> None:

        super().__init__()

        self.noise_dim = config["noise_dim"]
        self.device = device

        if "mlp" in config:
            self.mode = "mlp"
            self.generator = MLP(
                input_size=packetinfo_dim + self.noise_dim,
                output_size=packetfield_dim,
                config=config["mlp"]
            )
        elif "rnn" in config:
            self.mode = "rnn"
            self.max_sample_num = config["max_sample_num"]
            self.sample_len = config["sample_len"]
            self.generator = RNN(
                input_size=packetinfo_dim + self.noise_dim,
                output_size=packetfield_dim * self.sample_len,
                config=config["rnn"]
            )
            

        self.normalization = packetfield_normalization
    
    def forward(self, packetinfo: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        the forward process

        Parameters
        ----------
        packetinfo
            the input condition for measurement field generation, including IP 5-tuples and information of corresponding time series step
        batch_size
            the size of minibatch

        Returns
        -------
        the generated measurement fields

        """

        if self.mode == "mlp":
            to_return = self.generator(
                torch.cat([
                    packetinfo,
                    torch.randn(batch_size, self.noise_dim).to(self.device)
                ], dim=-1)
            )

            to_return = torch.cat(
                [norm(x) for x, norm in zip(torch.split(to_return, self.normalization[1], dim=-1), self.normalization[0])], dim=-1
            )

        else:
            to_return = self.generator(
                torch.cat([
                    packetinfo[:,None,:].expand(-1, (self.max_sample_num - 1) // self.sample_len + 1, -1),
                    torch.randn(batch_size, (self.max_sample_num - 1) // self.sample_len + 1, self.noise_dim).to(self.device)
                ], dim=-1),
                1
            )
            to_return = to_return.view(batch_size, -1, to_return.shape[2] // self.sample_len)[:, :self.max_sample_num, :]

            to_return = torch.cat(
                [norm(x) for x, norm in zip(torch.split(to_return, self.normalization[1], dim=-1), self.normalization[0])], dim=-1
            )

            def continuous_mask(mask, reverse=False):
                return_mask = mask
                if reverse:
                    tmp_mask = return_mask[:, -1]
                    for i in range(return_mask.shape[1]):
                        return_mask[:, -(i+1)] = return_mask[:, -(i+1)] * tmp_mask
                        tmp_mask = return_mask[:, -(i+1)]
                else:
                    tmp_mask = return_mask[:, 0]
                    for i in range(return_mask.shape[1]):
                        return_mask[:, i] = return_mask[:, i] * tmp_mask
                        tmp_mask = return_mask[:, i]
                return return_mask

            mask = ((to_return[:, :, -2] > to_return[:, :, -1])).float()
            mask[:, 0] = 1 # force
            mask = continuous_mask(mask)
            
            to_return[:, :, :-2] = to_return[:, :, :-2] * mask[:, :, None]
        
        return to_return


class CascadeGenerator(Generator):
    """
    Incomplete CascadeNet Generator, generating only IP 5-tuples and aggregated time series but not measurement fields

    Parameters
    ----------
    flowlevel_generator
        a FlowlevelGenerator object that generates IP 5-tuples
    packetrate_generator
        a PacketrateGenerator object that generates aggregated time series
    condition_gen_flag
        If true, the `flowlevel_generator` will generate IP 5-tuples and condition together, ignoring the given condition;
        If false, the `flowlevel_generator` will generate IP 5-tuples based on the given condition.

    """
    def __init__(self,
                 flowlevel_generator: FlowlevelGenerator,
                 packetrate_generator: PacketrateGenerator,
                 condition_gen_flag: bool) -> None:
        
        super().__init__()

        self.flowlevel_generator = flowlevel_generator
        self.packetrate_generator = packetrate_generator
        self.condition_gen_flag = condition_gen_flag
    
    def forward(self, condition: torch.Tensor, batch_size: int, length: int) -> typing.Tuple[torch.Tensor]:
        """
        the forward process

        Parameters
        ----------
        condition
            the input condition. Only when `self.flowlevel_generator.condition_gen_flag` is false, the condition will be used.
        batch_size
            the size of minibatch
        length
            the length of the output time series
        
        Returns
        -------
        If `self.flowlevel_generator.condition_gen_flag` is true, return the generated condition, IP 5-tuples and time series;
        If `self.flowlevel_generator.condition_gen_flag` is false, return the input condition and the generated IP 5-tupless and time series.
        
        """

        condition, fivetuple = self.flowlevel_generator(condition, batch_size)
        
        flowlevel = torch.cat([condition, fivetuple], dim=-1)

        packetrate = self.packetrate_generator(flowlevel, batch_size, length)
    
        return condition, fivetuple, packetrate

    def save(self, path: str, postfix: str = "") -> None:
        self.flowlevel_generator.save(path, postfix)
        self.packetrate_generator.save(path, postfix)

    def load(self, path: str, postfix: str = "") -> None:
        self.flowlevel_generator.load(path, postfix)
        self.packetrate_generator.load(path, postfix)


class CascadeCompGenerator(Generator):
    """
    Complete CascadeNet Generator, generating IP 5-tuples, aggregated time series and measurement fields

    Parameters
    ----------
    flowlevel_generator
        a FlowlevelGenerator object that generates IP 5-tuples
    packetrate_generator
        a PacketrateGenerator object that generates aggregated time series
    packetfield_generator
        a PacketfieldGenerator object that generates measurement fields
    condition_gen_flag
        If true, the `flowlevel_generator` will generate IP 5-tuples and condition together, ignoring the given condition;
        If false, the `flowlevel_generator` will generate IP 5-tuples based on the given condition.
    processor
        a PrePostProcessor object that manages the preprocessing and the postprocessing
        We need the function `processor.to_packetinfo` to covert IP 5-tuples and time series into the information for each packet's measurement fields.

    """
    def __init__(self,
                 flowlevel_generator: FlowlevelGenerator,
                 packetrate_generator: PacketrateGenerator,
                 packetfield_generator: PacketfieldGenerator,
                 condition_gen_flag: bool,
                 processor) -> None:
        
        super().__init__()

        self.flowlevel_generator = flowlevel_generator
        self.packetrate_generator = packetrate_generator
        self.packetfield_generator = packetfield_generator
        self.condition_gen_flag = condition_gen_flag
        self.processor = processor
    
    def forward(self, condition: torch.Tensor, batch_size: int, length: int, single_sample: int = False) -> typing.Tuple[torch.Tensor]:
        """
        the forward process

        Parameters
        ----------
        condition
            the input condition. Only when `self.flowlevel_generator.condition_gen_flag` is false, the condition will be used.
        batch_size
            the size of minibatch
        length
            the length of the output time series
        single_sample
            If true, the generator will only generate measurement fields of one packet for each flow each time step, used in the training process;
            If false, the generator will generate measurement fields of all packets for each flow each time step, used in the generating process.
        
        Returns
        -------
        If `self.flowlevel_generator.condition_gen_flag` is true, return the generated condition, IP 5-tuples, time series and measurement fields;
        If `self.flowlevel_generator.condition_gen_flag` is false, return the input condition and the generated IP 5-tuples, time series and measurement fields.
        
        """

        condition, fivetuple = self.flowlevel_generator(condition, batch_size)
        
        flowlevel = torch.cat([condition, fivetuple], dim=-1)

        packetrate = self.packetrate_generator(flowlevel, batch_size, length)
        
        packetinfo = self.processor.to_packetinfo(condition, fivetuple, packetrate, single_sample)

        packetfield = self.packetfield_generator(packetinfo, packetinfo.shape[0])
    
        return condition, fivetuple, packetrate, packetinfo, packetfield
    
    def generate_flow(self, condition: torch.Tensor, batch_size: int, length: int) -> typing.Tuple[torch.Tensor]:
        """
        A function generating IP 5-tuples and time series
        """

        condition, fivetuple = self.flowlevel_generator(condition, batch_size)
        
        flowlevel = torch.cat([condition, fivetuple], dim=-1)

        packetrate = self.packetrate_generator(flowlevel, batch_size, length)
    
        return condition, fivetuple, packetrate
    
    def generate_packet(self, packetinfo: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        A function generating measurement fields
        """

        packetfield = self.packetfield_generator(packetinfo, batch_size)
    
        return packetfield

    def save(self, path: str, postfix: str = ""):
        self.flowlevel_generator.save(path, postfix)
        self.packetrate_generator.save(path, postfix)
        self.packetfield_generator.save(path, postfix)

    def load(self, path: str, postfix: str = ""):
        self.flowlevel_generator.load(path, postfix)
        self.packetrate_generator.load(path, postfix)
        self.packetfield_generator.load(path, postfix)


