import os
import itertools
import importlib
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .net import MLP, RNN, BasicModel

class Trainer(BasicModel):
    def __init__(self, generator, discriminator, config, device):

        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.g_optimizer = getattr(importlib.import_module("torch.optim"), config["optimizer"])(
            self.generator.parameters(),
            lr = config["learning_rate"]
        )
        self.d_optimizer = getattr(importlib.import_module("torch.optim"), config["optimizer"])(
            self.discriminator.parameters(),
            lr = config["learning_rate"]
        )

        self.d_iter = config["d_iter"]
        self.gp = config["gradient_penalty"]
        self.device = device
    
    def build_record(self, name):
        return {
            name: {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            }
        }
    
    def write_summary(self, writer, record, epoch):
        for k1 in record.keys():
            for k2 in record[k1].keys():
                for k3 in record[k1][k2].keys():
                    writer.add_scalar(os.path.join(k1,k2,k3), np.mean(record[k1][k2][k3]), epoch)
    
    def gan_d_loss(self,
                   prob, prob_hat, gp,
                   record):
        
        d_loss_real = prob.mean()

        record["d_loss_real"].append(d_loss_real.item())

        d_loss_fake = prob_hat.mean()

        record["d_loss_fake"].append(d_loss_fake.item())

        d_loss = d_loss_fake - d_loss_real + self.gp * gp

        record["d_loss"].append(d_loss.item())

        return d_loss

    def gan_g_loss(self,
                   prob_hat,
                   record):
        
        g_loss = - prob_hat.mean()

        record["g_loss"].append(g_loss.item())

        return g_loss

    def save(self, path, postfix=""):
        self.generator.save(path, postfix)
        self.discriminator.save(path, postfix)
        #os.makedirs(os.path.join(path, "gan", self.__class__.__name__), exist_ok=True)
        os.makedirs(os.path.join(path, "gan", self.__class__.__name__, "g_optimizer"), exist_ok=True)
        os.makedirs(os.path.join(path, "gan", self.__class__.__name__, "d_optimizer"), exist_ok=True)
        torch.save(self.g_optimizer.state_dict(), os.path.join(path, "gan", self.__class__.__name__, "g_optimizer", "checkpoint"+postfix))
        torch.save(self.d_optimizer.state_dict(), os.path.join(path, "gan", self.__class__.__name__, "d_optimizer", "checkpoint"+postfix))

    def load(self, path, postfix=""):
        self.generator.load(path, postfix)
        self.discriminator.load(path, postfix)
        self.g_optimizer.load_state_dict(torch.load(os.path.join(path, "gan", self.__class__.__name__, "g_optimizer", "checkpoint"+postfix)))
        self.d_optimizer.load_state_dict(torch.load(os.path.join(path, "gan", self.__class__.__name__, "d_optimizer", "checkpoint"+postfix)))


class FlowlevelTrainer(Trainer):
    def __init__(self, generator, discriminator, condition_gen_flag, config, device):

        super().__init__(generator, discriminator, config, device)
        self.condition_gen_flag = condition_gen_flag
    
    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):

        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record("flowlevel")
            for condition, fivetuple in iter(dataloader):
                
                batch_size = condition.shape[0]
                condition = condition.to(self.device)
                fivetuple = fivetuple.to(self.device)

                self.fit_epoch(condition, fivetuple, batch_size, record["flowlevel"])
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".pretrain"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, condition, fivetuple, batch_size, record):
        
        condition_hat, fivetuple_hat = self.generator(condition, batch_size)

        for _ in range(self.d_iter):
            
            prob = self.discriminator(condition, fivetuple)
            prob_hat = self.discriminator(condition_hat, fivetuple_hat)

            gp = self.calculate_gradient_penalty(
                condition, condition_hat,
                fivetuple, fivetuple_hat,
                batch_size, record["gan_d_loss"]
            )

            d_loss = self.gan_d_loss(
                prob, prob_hat, gp,
                record["gan_d_loss"]
            )

            gan_d_loss = d_loss

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_hat = self.discriminator(condition_hat, fivetuple_hat)

        g_loss = self.gan_g_loss(
            prob_hat,
            record["gan_g_loss"]
        )

        gan_g_loss = g_loss

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   condition, condition_hat,
                                   fivetuple, fivetuple_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)

        interpolated_condition = alpha_dim2 * condition + (1 - alpha_dim2) * condition_hat
        interpolated_fivetuple = alpha_dim2 * fivetuple + (1 - alpha_dim2) * fivetuple_hat

        prob_interpolated = self.discriminator(interpolated_condition, interpolated_fivetuple)

        if self.condition_gen_flag:
            gradients = torch.autograd.grad(
                outputs=prob_interpolated,
                inputs=[interpolated_condition, interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=1) +
                torch.sum(torch.square(gradients[1]), dim=1)
            )
        else:
            gradients = torch.autograd.grad(
                outputs=prob_interpolated,
                inputs=[interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=1)
            )

        gp = torch.mean(torch.square(slope - 1))

        record["gradient_penalty"].append(gp.item())

        return gp
    

class PacketrateTrainer(Trainer):
    def __init__(self, generator, discriminator, config, device):

        super().__init__(generator, discriminator, config, device)
    
    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):

        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record("packetrate")
            for condition, fivetuple, packetrate in iter(dataloader):
                
                batch_size = condition.shape[0]
                length = packetrate.shape[1]
                flowlevel = torch.cat([condition, fivetuple], dim=-1).to(self.device)
                packetrate = packetrate.to(self.device)

                self.fit_epoch(flowlevel, packetrate, batch_size, length, record["packetrate"])
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".pretrain"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, flowlevel, packetrate, batch_size, length, record):
        
        packetrate_hat = self.generator(flowlevel, batch_size, length)

        for _ in range(self.d_iter):
            
            prob = self.discriminator(flowlevel, packetrate)
            prob_hat = self.discriminator(flowlevel, packetrate_hat)

            gp = self.calculate_gradient_penalty(
                flowlevel,
                packetrate, packetrate_hat,
                batch_size, record["gan_d_loss"]
            )

            d_loss = self.gan_d_loss(
                prob, prob_hat, gp,
                record["gan_d_loss"]
            )

            gan_d_loss = d_loss

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_hat = self.discriminator(flowlevel, packetrate_hat)

        g_loss = self.gan_g_loss(
            prob_hat,
            record["gan_g_loss"]
        )

        gan_g_loss = g_loss

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   flowlevel,
                                   packetrate, packetrate_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        interpolated_packetrate = alpha_dim3 * packetrate + (1 - alpha_dim3) * packetrate_hat

        prob_interpolated = self.discriminator(
            flowlevel, interpolated_packetrate
        )

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=[interpolated_packetrate],
            grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
            create_graph=True,
            retain_graph=True
        )

        slope = torch.sqrt(
            torch.sum(torch.square(gradients[0]), dim=(1,2))
        )

        gp = torch.mean(torch.square(slope - 1))

        record["gradient_penalty"].append(gp.item())

        return gp


class PacketfieldTrainer(Trainer):
    def __init__(self, generator, discriminator, config, device):

        super().__init__(generator, discriminator, config, device)
    
    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None, tqdm_per_epoch=False):

        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record("packetfield")
            single_epoch_iterator = iter(dataloader)
            if tqdm_per_epoch:
                single_epoch_iterator = tqdm(single_epoch_iterator)
            for packetinfo, packetfield in single_epoch_iterator:
                
                batch_size = packetinfo.shape[0]
                packetinfo = packetinfo.to(self.device)
                packetfield = packetfield.to(self.device)

                self.fit_epoch(packetinfo, packetfield, batch_size, record["packetfield"])
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".pretrain"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, packetinfo, packetfield, batch_size, record):
        
        packetfield_hat = self.generator(packetinfo, batch_size)

        for _ in range(self.d_iter):
            
            prob = self.discriminator(packetinfo, packetfield)
            prob_hat = self.discriminator(packetinfo, packetfield_hat)

            gp = self.calculate_gradient_penalty(
                packetinfo,
                packetfield, packetfield_hat,
                batch_size, record["gan_d_loss"]
            )

            d_loss = self.gan_d_loss(
                prob, prob_hat, gp,
                record["gan_d_loss"]
            )

            gan_d_loss = d_loss

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_hat = self.discriminator(packetinfo, packetfield_hat)

        g_loss = self.gan_g_loss(
            prob_hat,
            record["gan_g_loss"]
        )

        gan_g_loss = g_loss

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   packetinfo,
                                   packetfield, packetfield_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        if packetfield_hat.ndim == 3:
            interpolated_packetfield = alpha_dim3 * packetfield + (1 - alpha_dim3) * packetfield_hat
        else:
            interpolated_packetfield = alpha_dim2 * packetfield + (1 - alpha_dim2) * packetfield_hat

        prob_interpolated = self.discriminator(
            packetinfo, interpolated_packetfield
        )

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=[interpolated_packetfield],
            grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
            create_graph=True,
            retain_graph=True
        )

        if packetfield_hat.ndim == 3:
            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=(1,2))
            )
        else:
            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=1)
            )

        gp = torch.mean(torch.square(slope - 1))

        record["gradient_penalty"].append(gp.item())

        return gp


class CascadeTrainer(Trainer):
    def __init__(self, generator, discriminator, condition_gen_flag, config, device):

        super().__init__(generator, discriminator, config, device)
        self.alpha = config["alpha"]
        self.condition_gen_flag = condition_gen_flag
    
    def build_record(self):
        return {
            "flowlevel": {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            },
            "packetrate": {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            },
        }

    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):
        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record()
            for condition, fivetuple, packetrate in iter(dataloader):
                
                batch_size = condition.shape[0]
                length = packetrate.shape[1]
                condition = condition.to(self.device)
                fivetuple = fivetuple.to(self.device)
                packetrate = packetrate.to(self.device)

                self.fit_epoch(condition, fivetuple, packetrate, batch_size, length, record)
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".finetune"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, condition, fivetuple, packetrate, batch_size, length, record):
        
        condition_hat, fivetuple_hat, packetrate_hat = self.generator(condition, batch_size, length)

        for _ in range(self.d_iter):
            
            prob_flowlevel, prob_packetrate = self.discriminator(condition, fivetuple, packetrate)
            prob_flowlevel_hat, prob_packetrate_hat = self.discriminator(condition_hat, fivetuple_hat, packetrate_hat)

            gp_flowlevel, gp_packetrate = self.calculate_gradient_penalty(
                condition, condition_hat,
                fivetuple, fivetuple_hat,
                packetrate, packetrate_hat,
                batch_size, record
            )

            d_loss_flowlevel = self.gan_d_loss(
                prob_flowlevel, prob_flowlevel_hat, gp_flowlevel,
                record["flowlevel"]["gan_d_loss"]
            )
            d_loss_packetrate = self.gan_d_loss(
                prob_packetrate, prob_packetrate_hat, gp_packetrate,
                record["packetrate"]["gan_d_loss"]
            )

            gan_d_loss = d_loss_flowlevel * self.alpha + d_loss_packetrate

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_flowlevel_hat, prob_packetrate_hat = self.discriminator(condition_hat, fivetuple_hat, packetrate_hat)

        g_loss_flowlevel = self.gan_g_loss(
            prob_flowlevel_hat,
            record["flowlevel"]["gan_g_loss"]
        )
        g_loss_packetrate = self.gan_g_loss(
            prob_packetrate_hat,
            record["packetrate"]["gan_g_loss"]
        )

        gan_g_loss = g_loss_flowlevel * self.alpha + g_loss_packetrate

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   condition, condition_hat,
                                   fivetuple, fivetuple_hat,
                                   packetrate, packetrate_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        interpolated_condition = alpha_dim2 * condition + (1 - alpha_dim2) * condition_hat
        interpolated_fivetuple = alpha_dim2 * fivetuple + (1 - alpha_dim2) * fivetuple_hat
        interpolated_packetrate = alpha_dim3 * packetrate + (1 - alpha_dim3) * packetrate_hat

        prob_interpolated_flowlevel, prob_interpolated_packetrate = \
            self.discriminator(interpolated_condition, interpolated_fivetuple, interpolated_packetrate)

        if self.condition_gen_flag:
            gradients_flowlevel = torch.autograd.grad(
                outputs=prob_interpolated_flowlevel,
                inputs=[interpolated_condition, interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated_flowlevel.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetrate = torch.autograd.grad(
                outputs=prob_interpolated_packetrate,
                inputs=[interpolated_condition, interpolated_fivetuple, interpolated_packetrate],
                grad_outputs=torch.ones(prob_interpolated_packetrate.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope_flowlevel = torch.sqrt(
                torch.sum(torch.square(gradients_flowlevel[0]), dim=1) +
                torch.sum(torch.square(gradients_flowlevel[1]), dim=1)
            )
            slope_packetrate = torch.sqrt(
                torch.sum(torch.square(gradients_packetrate[0]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[1]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[2]), dim=(1,2))
            )
        else:
            gradients_flowlevel = torch.autograd.grad(
                outputs=prob_interpolated_flowlevel,
                inputs=[interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated_flowlevel.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetrate = torch.autograd.grad(
                outputs=prob_interpolated_packetrate,
                inputs=[interpolated_fivetuple, interpolated_packetrate],
                grad_outputs=torch.ones(prob_interpolated_packetrate.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope_flowlevel = torch.sqrt(
                torch.sum(torch.square(gradients_flowlevel[0]), dim=1)
            )
            slope_packetrate = torch.sqrt(
                torch.sum(torch.square(gradients_packetrate[0]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[1]), dim=(1,2))
            )

        gp_flowlevel = torch.mean(torch.square(slope_flowlevel - 1))
        gp_packetrate = torch.mean(torch.square(slope_packetrate - 1))

        record["flowlevel"]["gan_d_loss"]["gradient_penalty"].append(gp_flowlevel.item())
        record["packetrate"]["gan_d_loss"]["gradient_penalty"].append(gp_packetrate.item())

        return gp_flowlevel, gp_packetrate


class CascadeCompTrainer(Trainer):
    def __init__(self, generator, discriminator, condition_gen_flag, config, device):

        super().__init__(generator, discriminator, config, device)
        self.alpha = config["alpha"]
        self.beta = config["beta"]
        self.condition_gen_flag = condition_gen_flag
    
    def build_record(self):
        return {
            "flowlevel": {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            },
            "packetrate": {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            },
            "packetfield": {
                "gan_d_loss":{
                    "d_loss_real":[],
                    "d_loss_fake":[],
                    "gradient_penalty":[],
                    "d_loss":[],
                },
                "gan_g_loss":{
                    "g_loss":[],
                },
            },
        }

    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):
        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record()
            for condition, fivetuple, packetrate, packetinfo, packetfield in iter(dataloader):
                
                batch_size = condition.shape[0]
                length = packetrate.shape[1]
                condition = condition.to(self.device)
                fivetuple = fivetuple.to(self.device)
                packetrate = packetrate.to(self.device)
                packetinfo = packetinfo.to(self.device)
                packetfield = packetfield.to(self.device)

                self.fit_epoch(condition, fivetuple, packetrate, packetinfo, packetfield, batch_size, length, record)
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".finetune"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, condition, fivetuple, packetrate, packetinfo, packetfield, batch_size, length, record):
        
        condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat = \
            self.generator(condition, batch_size, length, single_sample=True)

        for _ in range(self.d_iter):
            
            prob_flowlevel, prob_packetrate, prob_packetfield = \
                self.discriminator(condition, fivetuple, packetrate, packetinfo, packetfield)
            prob_flowlevel_hat, prob_packetrate_hat, prob_packetfield_hat = \
                self.discriminator(condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat)

            gp_flowlevel, gp_packetrate, gp_packetfield = self.calculate_gradient_penalty(
                condition, condition_hat,
                fivetuple, fivetuple_hat,
                packetrate, packetrate_hat,
                packetinfo, packetinfo_hat,
                packetfield, packetfield_hat,
                batch_size, record
            )

            d_loss_flowlevel = self.gan_d_loss(
                prob_flowlevel, prob_flowlevel_hat, gp_flowlevel,
                record["flowlevel"]["gan_d_loss"]
            )
            d_loss_packetrate = self.gan_d_loss(
                prob_packetrate, prob_packetrate_hat, gp_packetrate,
                record["packetrate"]["gan_d_loss"]
            )
            d_loss_packetfield = self.gan_d_loss(
                prob_packetfield, prob_packetfield_hat, gp_packetfield,
                record["packetfield"]["gan_d_loss"]
            )

            gan_d_loss = d_loss_flowlevel * self.alpha + d_loss_packetrate * self.beta + d_loss_packetfield

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_flowlevel_hat, prob_packetrate_hat, prob_packetfield_hat = \
            self.discriminator(condition_hat, fivetuple_hat, packetrate_hat, packetinfo_hat, packetfield_hat)

        g_loss_flowlevel = self.gan_g_loss(
            prob_flowlevel_hat,
            record["flowlevel"]["gan_g_loss"]
        )
        g_loss_packetrate = self.gan_g_loss(
            prob_packetrate_hat,
            record["packetrate"]["gan_g_loss"]
        )
        g_loss_packetfield = self.gan_g_loss(
            prob_packetfield_hat,
            record["packetfield"]["gan_g_loss"]
        )

        gan_g_loss = g_loss_flowlevel * self.alpha + g_loss_packetrate * self.beta + g_loss_packetfield

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   condition, condition_hat,
                                   fivetuple, fivetuple_hat,
                                   packetrate, packetrate_hat,
                                   packetinfo, packetinfo_hat,
                                   packetfield, packetfield_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        interpolated_condition = alpha_dim2 * condition + (1 - alpha_dim2) * condition_hat
        interpolated_fivetuple = alpha_dim2 * fivetuple + (1 - alpha_dim2) * fivetuple_hat
        interpolated_packetrate = alpha_dim3 * packetrate + (1 - alpha_dim3) * packetrate_hat
        # each flow sample one packet
        interpolated_packetinfo = alpha_dim2 * packetinfo + (1 - alpha_dim2) * packetinfo_hat
        if packetfield_hat.ndim == 3:
            interpolated_packetfield = alpha_dim3 * packetfield + (1 - alpha_dim3) * packetfield_hat
        else:
            interpolated_packetfield = alpha_dim2 * packetfield + (1 - alpha_dim2) * packetfield_hat

        prob_interpolated_flowlevel, prob_interpolated_packetrate, prob_interpolated_packetfield= \
            self.discriminator(interpolated_condition, interpolated_fivetuple, interpolated_packetrate, interpolated_packetinfo, interpolated_packetfield)

        if self.condition_gen_flag:
            gradients_flowlevel = torch.autograd.grad(
                outputs=prob_interpolated_flowlevel,
                inputs=[interpolated_condition, interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated_flowlevel.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetrate = torch.autograd.grad(
                outputs=prob_interpolated_packetrate,
                inputs=[interpolated_condition, interpolated_fivetuple, interpolated_packetrate],
                grad_outputs=torch.ones(prob_interpolated_packetrate.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetfield = torch.autograd.grad(
                outputs=prob_interpolated_packetfield,
                inputs=[interpolated_packetinfo, interpolated_packetfield],
                grad_outputs=torch.ones(prob_interpolated_packetfield.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope_flowlevel = torch.sqrt(
                torch.sum(torch.square(gradients_flowlevel[0]), dim=1) +
                torch.sum(torch.square(gradients_flowlevel[1]), dim=1)
            )
            slope_packetrate = torch.sqrt(
                torch.sum(torch.square(gradients_packetrate[0]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[1]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[2]), dim=(1,2))
            )
            if packetfield_hat.ndim == 3:
                slope_packetfield = torch.sqrt(
                    torch.sum(torch.square(gradients_packetfield[0]), dim=1) +
                    torch.sum(torch.square(gradients_packetfield[1]), dim=(1,2))
                )
            else:
                slope_packetfield = torch.sqrt(
                    torch.sum(torch.square(gradients_packetfield[0]), dim=1) +
                    torch.sum(torch.square(gradients_packetfield[1]), dim=1)
                )
        else:
            gradients_flowlevel = torch.autograd.grad(
                outputs=prob_interpolated_flowlevel,
                inputs=[interpolated_fivetuple],
                grad_outputs=torch.ones(prob_interpolated_flowlevel.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetrate = torch.autograd.grad(
                outputs=prob_interpolated_packetrate,
                inputs=[interpolated_fivetuple, interpolated_packetrate],
                grad_outputs=torch.ones(prob_interpolated_packetrate.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )
            gradients_packetfield = torch.autograd.grad(
                outputs=prob_interpolated_packetfield,
                inputs=[interpolated_packetinfo, interpolated_packetfield],
                grad_outputs=torch.ones(prob_interpolated_packetfield.shape).to(self.device),
                create_graph=True,
                retain_graph=True
            )

            slope_flowlevel = torch.sqrt(
                torch.sum(torch.square(gradients_flowlevel[0]), dim=1)
            )
            slope_packetrate = torch.sqrt(
                torch.sum(torch.square(gradients_packetrate[0]), dim=1) +
                torch.sum(torch.square(gradients_packetrate[1]), dim=(1,2))
            )
            if packetfield_hat.ndim == 3:
                slope_packetfield = torch.sqrt(
                    torch.sum(torch.square(gradients_packetfield[0]), dim=1) +
                    torch.sum(torch.square(gradients_packetfield[1]), dim=(1,2))
                )
            else:
                slope_packetfield = torch.sqrt(
                    torch.sum(torch.square(gradients_packetfield[0]), dim=1) +
                    torch.sum(torch.square(gradients_packetfield[1]), dim=1)
                )

        gp_flowlevel = torch.mean(torch.square(slope_flowlevel - 1))
        gp_packetrate = torch.mean(torch.square(slope_packetrate - 1))
        gp_packetfield = torch.mean(torch.square(slope_packetfield - 1))

        
        record["flowlevel"]["gan_d_loss"]["gradient_penalty"].append(gp_flowlevel.item())
        record["packetrate"]["gan_d_loss"]["gradient_penalty"].append(gp_packetrate.item())
        record["packetfield"]["gan_d_loss"]["gradient_penalty"].append(gp_packetfield.item())

        return gp_flowlevel, gp_packetrate, gp_packetfield


class PacketrateFTTrainer(Trainer):
    def __init__(self, generator, discriminator, config, device):

        super().__init__(generator, discriminator, config, device)
    
    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):
        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record("packetrate")
            for condition, fivetuple, packetrate, condition_hat, fivetuple_hat in iter(dataloader):
                
                batch_size = condition.shape[0]
                length = packetrate.shape[1]
                flowlevel = torch.cat([condition, fivetuple], dim=-1).to(self.device)
                flowlevel_hat = torch.cat([condition_hat, fivetuple_hat], dim=-1).to(self.device)
                packetrate = packetrate.to(self.device)

                self.fit_epoch(flowlevel, flowlevel_hat, packetrate, batch_size, length, record["packetrate"])
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if _epoch + 1 == epoch or (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".finetune"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, flowlevel, flowlevel_hat, packetrate, batch_size, length, record):
        
        packetrate_hat = self.generator(flowlevel_hat, batch_size, length)

        for _ in range(self.d_iter):
            
            prob = self.discriminator(flowlevel, packetrate)
            prob_hat = self.discriminator(flowlevel_hat, packetrate_hat)

            gp = self.calculate_gradient_penalty(
                flowlevel, flowlevel_hat,
                packetrate, packetrate_hat,
                batch_size, record["gan_d_loss"]
            )

            d_loss = self.gan_d_loss(
                prob, prob_hat, gp,
                record["gan_d_loss"]
            )

            gan_d_loss = d_loss

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_hat = self.discriminator(flowlevel_hat, packetrate_hat)

        g_loss = self.gan_g_loss(
            prob_hat,
            record["gan_g_loss"]
        )

        gan_g_loss = g_loss

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   flowlevel, flowlevel_hat,
                                   packetrate, packetrate_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        interpolated_flowlevel = alpha_dim2 * flowlevel + (1 - alpha_dim2) * flowlevel_hat
        interpolated_flowlevel.requires_grad_()
        interpolated_packetrate = alpha_dim3 * packetrate + (1 - alpha_dim3) * packetrate_hat

        prob_interpolated = self.discriminator(
            interpolated_flowlevel, interpolated_packetrate
        )

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=[interpolated_flowlevel, interpolated_packetrate],
            grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
            create_graph=True,
            retain_graph=True
        )

        slope = torch.sqrt(
            torch.sum(torch.square(gradients[0]), dim=1) +
            torch.sum(torch.square(gradients[1]), dim=(1,2))
        )

        gp = torch.mean(torch.square(slope - 1))

        record["gradient_penalty"].append(gp.item())

        return gp


class PacketfieldFTTrainer(Trainer):
    def __init__(self, generator, discriminator, config, device):

        super().__init__(generator, discriminator, config, device)
    
    def fit(self, dataloader, epoch, path, continue_epoch=0, save_epoch=None):

        if save_epoch is None:
            save_epoch = epoch // 10
            # if epoch less than 10, save only the last epoch
            if save_epoch == 0:
                save_epoch = 1

        writer = SummaryWriter(os.path.join(path, "summary", self.__class__.__name__,))
        self.train()

        for _epoch in tqdm(range(epoch)):
            record = self.build_record("packetfield")
            for packetinfo, packetfield, packetinfo_hat in iter(dataloader):
                
                batch_size = packetinfo.shape[0]
                packetinfo = packetinfo.to(self.device)
                packetfield = packetfield.to(self.device)
                packetinfo_hat = packetinfo_hat.to(self.device)

                self.fit_epoch(packetinfo, packetinfo_hat, packetfield, batch_size, record["packetfield"])
            
            self.write_summary(writer, record, _epoch + continue_epoch)
            if (_epoch + continue_epoch + 1) % save_epoch == 0:
                self.save(path, postfix=".pretrain"+str(_epoch + continue_epoch))
    
    def fit_epoch(self, packetinfo, packetinfo_hat, packetfield, batch_size, record):
        
        packetfield_hat = self.generator(packetinfo, batch_size)

        for _ in range(self.d_iter):
            
            prob = self.discriminator(packetinfo, packetfield)
            prob_hat = self.discriminator(packetinfo_hat, packetfield_hat)

            gp = self.calculate_gradient_penalty(
                packetinfo, packetinfo_hat,
                packetfield, packetfield_hat,
                batch_size, record["gan_d_loss"]
            )

            d_loss = self.gan_d_loss(
                prob, prob_hat, gp,
                record["gan_d_loss"]
            )

            gan_d_loss = d_loss

            self.d_optimizer.zero_grad()
            gan_d_loss.backward(retain_graph=True)
            self.d_optimizer.step()
        
        prob_hat = self.discriminator(packetinfo_hat, packetfield_hat)

        g_loss = self.gan_g_loss(
            prob_hat,
            record["gan_g_loss"]
        )

        gan_g_loss = g_loss

        self.g_optimizer.zero_grad()
        gan_g_loss.backward()
        self.g_optimizer.step()
    
    def calculate_gradient_penalty(self,
                                   packetinfo, packetinfo_hat,
                                   packetfield, packetfield_hat,
                                   batch_size, record):
        
        alpha_dim2 = torch.rand(batch_size, 1).to(self.device)
        alpha_dim3 = alpha_dim2[:,:,None]

        interpolated_packetinfo = alpha_dim2 * packetinfo + (1 - alpha_dim2) * packetinfo_hat
        interpolated_packetinfo.requires_grad_()
        if packetfield_hat.ndim == 3:
            interpolated_packetfield = alpha_dim3 * packetfield + (1 - alpha_dim3) * packetfield_hat
        else:
            interpolated_packetfield = alpha_dim2 * packetfield + (1 - alpha_dim2) * packetfield_hat

        prob_interpolated = self.discriminator(
            packetinfo, interpolated_packetfield
        )

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=[interpolated_packetinfo, interpolated_packetfield],
            grad_outputs=torch.ones(prob_interpolated.shape).to(self.device),
            create_graph=True,
            retain_graph=True
        )

        if packetfield_hat.ndim == 3:
            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=1) +
                torch.sum(torch.square(gradients[1]), dim=(1,2))
            )
        else:
            slope = torch.sqrt(
                torch.sum(torch.square(gradients[0]), dim=1) +
                torch.sum(torch.square(gradients[1]), dim=1)
            )
        

        gp = torch.mean(torch.square(slope - 1))

        record["gradient_penalty"].append(gp.item())

        return gp


