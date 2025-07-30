"""
Some basic neural network components
"""

import typing
import os
import torch
from torch import nn
import torch.nn.functional as F
import importlib

class BasicModel(nn.Module):
    """
    a basic class for neural network models
    """
    def __init__(self) -> None:
        super().__init__()
    
    def save(self, path: str, postfix: str = "") -> None:
        """
        save the model into some path

        Parameters
        ----------
        path
            the path to save model
        postfix
            the postfix of the file name. For example, the file name will be "checkpoint.abc" if the postfix is ".abc"

        """
        os.makedirs(os.path.join(path, "gan", self.__class__.__name__), exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "gan", self.__class__.__name__, "checkpoint"+postfix))

    def load(self, path, postfix="") -> None:
        """
        load the model from some path

        Parameters
        ----------
        path
            the path to load model
        postfix
            the postfix of the file name. For example, the file name will be "checkpoint.abc" if the postfix is ".abc"

        """
        self.load_state_dict(torch.load(os.path.join(path, "gan", self.__class__.__name__, "checkpoint"+postfix)))


class MLP(nn.Module):
    """
    a Multilayer Perceptron model
    """
    def __init__(self, input_size: int, output_size: int, config: typing.Mapping[str, typing.Any]) -> None: 

        super().__init__()

        activation = getattr(importlib.import_module("torch.nn"), config["activation"])()
        layers = []
        if config["num_layers"] > 0:
            layers.append(nn.Linear(input_size, config["hidden_dim"]))
            if config["batch_norm"]:
                layers.append(nn.BatchNorm1d(config["hidden_dim"]))
            layers.append(activation)
            
            
            for _ in range(config["num_layers"] - 1):
                layers.append(nn.Linear(config["hidden_dim"], config["hidden_dim"]))
                if config["batch_norm"]:
                    layers.append(nn.BatchNorm1d(config["hidden_dim"]))
                layers.append(activation)
            
            layers.append(nn.Linear(config["hidden_dim"], output_size))
        else:
            layers.append(nn.Linear(input_size, output_size))
        
        if config["normalization"] is not None:
            normalization = getattr(importlib.import_module("torch.nn"), config["normalization"])()
            layers.append(normalization)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.mlp(x)

class RNN(nn.Module):
    """
    a LSTM model using nn.LSTM backend
    """
    def __init__(self, input_size: int, output_size: int, config: typing.Mapping[str, typing.Any]) -> None:

        super().__init__()

        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_dim"]
        self.bidirectional = 2 if config["bidirectional"] else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            bidirectional=config["bidirectional"],
            batch_first=True
        )

        self.output = MLP(
            input_size=self.bidirectional * config["hidden_dim"],
            output_size=output_size,
            config=config["output"]
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h_0 = torch.randn(self.bidirectional * self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c_0 = torch.randn(self.bidirectional * self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        hidden, _ = self.lstm(x, (h_0, c_0))
        output = self.output(hidden)
        output = output * mask

        return output

class RNNC(nn.Module):
    """
    a LSTM model using nn.LSTMCell backend
    """
    def __init__(self, input_size: int, output_size: int, config: typing.Mapping[str, typing.Any]) -> None:

        super().__init__()

        self.hidden_size = config["hidden_dim"]

        self.lstm_cell = nn.LSTMCell(
            input_size=input_size,
            hidden_size=config["hidden_dim"],
        )

        self.output = MLP(
            input_size=config["hidden_dim"],
            output_size=output_size,
            config=config["output"]
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        cell = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        hidden_list = torch.zeros(x.shape[0], x.shape[1], self.hidden_size).to(x.device)
        for i in range(x.shape[1]):
            hidden, cell = self.lstm_cell(x[:, i], (hidden, cell))
            hidden_list[:, i] = hidden
        
        output = self.output(hidden_list)
        output = output * mask

        return output

class bRNNC(nn.Module):
    """
    a bidirectional LSTM model using nn.LSTMCell backend
    """
    def __init__(self, input_size: str, output_size: str, config: typing.Mapping[str, typing.Any]) -> None:

        super().__init__()

        self.hidden_size = config["hidden_dim"]

        self.lstm_cell_f = nn.LSTMCell(
            input_size=input_size,
            hidden_size=config["hidden_dim"],
        )
        self.lstm_cell_b = nn.LSTMCell(
            input_size=input_size,
            hidden_size=config["hidden_dim"],
        )

        self.output = MLP(
            input_size=config["hidden_dim"] * 2,
            output_size=output_size,
            config=config["output"]
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden_f = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        cell_f = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        hidden_b = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        cell_b = torch.randn(x.shape[0], self.hidden_size).to(x.device)
        hidden_f_list = torch.zeros(x.shape[0], x.shape[1], self.hidden_size).to(x.device)
        hidden_b_list = torch.zeros(x.shape[0], x.shape[1], self.hidden_size).to(x.device)
        for i in range(x.shape[1]):
            hidden_f, cell_f = self.lstm_cell_f(x[:, i], (hidden_f, cell_f))
            hidden_b, cell_b = self.lstm_cell_b(x[:, -i-1], (hidden_b, cell_b))
            hidden_f_list[:, i] = hidden_f
            hidden_b_list[:, -i-1] = hidden_b
        
        output = self.output(
            torch.cat([
                hidden_f_list,
                hidden_b_list
            ], dim=2)
        )
        output = output * mask

        return output