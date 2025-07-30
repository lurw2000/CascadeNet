"""
Use VAE to map time series to a latent space, and use the latent space as features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os 
import re
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from nta.utils.feature_extraction.soft_dtw_cuda import SoftDTW

class PacketrateDataset(torch.utils.data.Dataset):
    def __init__(self, flows, batch_size=64, device="cpu", swap_flow_dim=False):
        # flows: list of flows of size (n_interval, n_flow)
        if swap_flow_dim:
            flows = flows.T[:, :, np.newaxis]
        else:
            flows = flows[:, :, np.newaxis]
        flows = torch.Tensor(flows)
        self.flows = flows.to(device=device)
        print("Loading flows as an array of shape {} on device {}{}".format(self.flows.shape, device, self.flows.get_device()))
        self.batch_size = batch_size
    
    def __len__(self):
        return np.ceil(self.flows.shape[0] / self.batch_size).astype(int)

    def __getitem__(self, idx):
        # return a batch of flows of size (batch_size, n_interval)
        if idx*self.batch_size >= self.flows.shape[0]:
            raise StopIteration
        return self.flows[idx*self.batch_size: (idx+1)*self.batch_size, :, :]


class VAE_MLP(nn.Module):
    def __init__(self, input_dim=200, num_layers=2, hidden_dim=100, latent_dim=20, dropout=0.5, verbose=True):
        super(VAE_MLP, self).__init__()
        self.input_dim = input_dim
        self.num_layer = num_layers
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.verbose = verbose

        print("input_dim = {}".format(input_dim))
        print("num_layer = {}".format(num_layers))
        print("hidden_dim = {}".format(hidden_dim))
        print("latent_dim = {}".format(latent_dim))
        print("dropout = {}".format(dropout))

        # Encoder
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, latent_dim)  # mu layer
        # self.fc22 = nn.Linear(hidden_dim, latent_dim)  # logvar layer
        encoder_layers = []
        encoder_layers.append(nn.Linear(input_dim, hidden_dim))
        encoder_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            # with dropout
            encoder_layers.append(nn.Dropout(p=self.dropout))
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())


        self.encoder_layers = nn.Sequential(*encoder_layers)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        # self.fc3 = nn.Linear(latent_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, input_dim)
        decoder_layers = []
        decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
        decoder_layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            # with dropout
            decoder_layers.append(nn.Dropout(p=self.dropout))
            decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
        decoder_layers.append(nn.Linear(hidden_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder_layers = nn.Sequential(*decoder_layers)

        if self.verbose:
            print("Encoder:")
            print(self.encoder_layers)
            print("mu_layer:")
            print(self.mu_layer)
            print("logvar_layer:")
            print(self.logvar_layer)
            print("Decoder:")
            print(self.decoder_layers)


    def encode(self, x):
        # h1 = F.relu(self.fc1(x))
        h = self.encoder_layers(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        # FIXME: 
        # squeeze to [0, 1]. This is only temporarily so that later model can learn to generate latent vectors
        # without normalization
        # z = F.sigmoid(z)
        return z

    def decode(self, z):
        # h3 = F.relu(self.fc3(z))
        # h4 = self.fc4(h3)
        # return F.sigmoid(h4)
        return self.decoder_layers(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def load(self, ckpt_str):
        if not os.path.exists(ckpt_str):
            raise ValueError("Path does not exist:\n\t{}".format(ckpt_str))
        if os.path.isdir(ckpt_str):
            ckpt_folder = ckpt_str 
            ckpt_files = os.listdir(ckpt_folder)
            ckpt_files = [f for f in ckpt_files if re.match(r"ckpt_\d+\.pt", f)]
            ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
            if len(ckpt_files) == 0:
                print("No checkpoint found in {}".format(ckpt_folder))
                exit(0)
            ckpt_path = os.path.join(ckpt_folder, ckpt_files[-1])
        else:
            ckpt_path = ckpt_str 

        print("Loading checkpoint from {}".format(ckpt_path))
        self.load_state_dict(torch.load(ckpt_path))
    
    def save(self, ckpt_folder):
        pass 

class VAE_RNN(nn.Module):
    def __init__(self, sequence_length, sequence_dim, hidden_dim, latent_dim):
        super(VAE_RNN, self).__init__()

        # Encoder
        self.lstm1 = nn.LSTM(sequence_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, latent_dim)  # mean
        self.fc2 = nn.Linear(hidden_dim, latent_dim)  # log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        
        self.sequence_length = sequence_length

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h1, _ = self.lstm1(x)
        h1 = h1[:, -1, :]
        mu = self.fc1(h1)
        log_var = self.fc2(h1)
        return mu, log_var
    
    def decode(self, z):
        # Decoding
        # convert latent to hidden
        h3 = self.fc3(z)
        # convert hidden to sequence
        h3 = h3.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # convert sequence to output
        h4, _ = self.lstm2(h3)
        return h4


    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

class VAE_CNN(nn.Module):
    pass 


# Define the VAE loss
def loss_function(recon_x, x, input_dim, mu, logvar, model=None, loss_type="bce", reg=("l2", 1e-5), device="cpu"):
    if reg is not None and model is None:
        raise ValueError("model must be provided if reg is not None")
    if loss_type == "bce":
        bce_loss = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='mean')
        loss = bce_loss
    elif loss_type == "bce_kld":
        # Normal distribution is not a good fit for network flows b/c they are notoriouly heavy-tailed
        bce_loss = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss= bce_loss + kld_loss
    elif loss_type == "bce_kld_longtail":
        bce_loss = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='mean')
        kld_loss_longtail = 0.5 * (-1 + logvar + (mu.pow(2) + logvar.pow(2) - 2 * torch.log(logvar)))
        loss = bce_loss + kld_loss_longtail
    elif loss_type == "mse":
        mse_loss = F.mse_loss(recon_x, x.view(-1, input_dim), reduction='mean')
        # average of square of L2 norm of difference
        # torch.nn.MSELoss(reduction='mean')
        loss = mse_loss
    elif loss_type == "mse_kld":
        # Normal distribution is not a good fit for network flows b/c they are notoriouly heavy-tailed
        mse_loss = F.mse_loss(recon_x, x.view(-1, input_dim), reduction='mean')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss= mse_loss + kld_loss
    elif loss_type == "mse_kld_longtail":
        mse_loss = F.mse_loss(recon_x, x.view(-1, input_dim), reduction='mean')
        kld_loss_longtail = 0.5 * torch.sum(-1 + logvar + (mu.pow(2) + logvar.pow(2) - 2 * torch.log(logvar)))
        loss = mse_loss + kld_loss_longtail
    elif loss_type == "dtw":
        sdtw = SoftDTW(use_cuda=True, gamma=1.0, normalize=True)
        # input of shape batch_size x seq_len x dims
        stdw_loss = sdtw(recon_x.unsqueeze(-1), x).mean()
        # print("stdw_loss = {} of type {} and shape {}".format(
        #     stdw_loss,
        #     type(stdw_loss),
        #     stdw_loss.shape
        # ))
        loss = stdw_loss
    elif loss_type == "rel_err":
        # given a vector x and x' the relative error of x' is defined as
        # ||x - x'||_2 / ||x||_2
        # To make the loss differentiable, we use the following approximation
        # ||x - x' + 1e-12||_2^2 / ||x + 1e-12||_2
        x = x.view(-1, input_dim)
        rel_err_loss = torch.norm(x - recon_x + 1e-12, p=2, dim=1) / torch.norm(x + 1e-12, p=2, dim=1)
        rel_err_loss = rel_err_loss.mean()
        loss = rel_err_loss

    if reg is None:
        pass 
    elif reg[0] == "l2":
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += reg[1] * l2_reg
    elif reg[0] == "l1":
        raise NotImplementedError("Not implemented yet")
        
    return loss

class VAETrainer(nn.Module):
    def __init__(self, model, config, device="cpu"):
        super(VAETrainer, self).__init__()

        self.model = model
        self.config = config
        self.device = device

        optimizer_class = getattr(torch.optim, config["optimizer"]["class"])
        self.optimizer = optimizer_class(
            model.parameters(),
            **config["optimizer"]["args"])
        
        self.writer = None
        
        

    def fit(self, train_dataloader, dev_dataloader=None, epochs=100, ckpt_epoch=20, ckpt_folder="./ckpt"):
        progress_bar = tqdm(range(epochs))
        self.writer = SummaryWriter(log_dir=os.path.join(ckpt_folder, "runs"))
        train_losses = []

        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)

        print("Start training vae model...")
        for epoch in progress_bar:
            self.model.train()
            train_loss = 0

            for batch_idx, data in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                train_loss = loss_function(
                    recon_x=recon_batch, x=data,
                    input_dim=self.model.input_dim,
                    mu=mu, logvar=logvar, model=self.model, reg=self.config["loss"]["reg"],
                    loss_type=self.config["loss"]["loss_type"], device=self.device)
                train_loss.backward()
                train_loss += train_loss.item()
                self.optimizer.step()

            train_loss = train_loss / len(train_dataloader)
            train_losses.append(train_loss.cpu().detach().numpy())

            self.writer.add_scalar("train_loss", train_loss, epoch+1)

            if dev_dataloader is not None:
                # evaluate on dev set
                self.model.eval()
                dev_loss = 0
                with torch.no_grad():
                    for batch_idx, data in enumerate(dev_dataloader):
                        recon_batch, mu, logvar = self.model(data)
                        dev_loss_batch = loss_function(
                            recon_x=recon_batch, x=data,
                            input_dim=self.model.input_dim,
                            mu=mu, logvar=logvar, model=self.model, reg=self.config["loss"]["reg"],
                            loss_type=self.config["loss"]["loss_type"], device=self.device)
                        dev_loss += dev_loss_batch.item()
                dev_loss = dev_loss / len(dev_dataloader)
                self.writer.add_scalar("dev_loss", dev_loss, epoch+1)
                # plot two losses in one figure with labels
                self.writer.add_scalars("loss", {"train": train_loss, "dev": dev_loss}, epoch+1)
                progress_bar.set_description("Epoch: {}/{}, train_loss: {:.4e}, dev_loss: {:.4e}".format(
                    epoch+1, epochs, train_loss, dev_loss))
            else:
                progress_bar.set_description("Epoch: {}/{}, train_loss: {:.4e}".format(
                    epoch+1, epochs, train_loss))

            # save checkpoint if needed
            if (epoch+1) % ckpt_epoch == 0 or epoch+1 == epochs:
                ckpt_path = os.path.join(ckpt_folder, "ckpt_{}.pt".format(epoch+1))
                torch.save(self.model.state_dict(), ckpt_path)
                print("Checkpoint at {} epoch saved to {}".format(epoch+1, ckpt_path))

        return np.array(train_losses)

    def generate(self, sample):
        self.model.eval()
        with torch.no_grad():
            # use noise as latent vector
            z = torch.randn(sample, self.model.latent_dim).to(self.device)
            recon_batch = self.model.decode(z)
        return recon_batch


