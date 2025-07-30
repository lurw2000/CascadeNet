# To build CascadeGAN

import torch
import numpy as np
import os
import random
from .gan import data, generator, discriminator, trainer, evaluator
from .pre_post_processor import pre_post_processor

class CascadeGANBuilder(object):
    def __init__(self, config):
        self.config = config
    
    def build_processor(self, preprocess=True):
        print("Building PrePostProcessor ...")

        self.processor = pre_post_processor.PrePostProcessor(config=self.config["pre_post_processor"])
        if preprocess:
            self.processor.pre_process()
        self.flowlevel_dataset = data.ConditionalFlowlevelDataset(path=self.processor.output_folder)
        self.packetrate_dataset = data.ConditionalPacketrateDataset(path=self.processor.output_folder)

    def build_generator(self, device):
        print("Building Generators ...")

        self.flowlevel_generator = generator.FlowlevelGenerator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["generator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_generator = generator.PacketrateGenerator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            config=self.config["gan"]["generator"]["packetrate"],
            device=device
        ).to(device)
        self.cascade_generator = generator.CascadeGenerator(
            flowlevel_generator=self.flowlevel_generator,
            packetrate_generator=self.packetrate_generator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"]
        )
    
    def build_discriminator(self, device):
        print("Building Discriminators ...")

        self.flowlevel_discriminator = discriminator.FlowlevelDiscriminator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            config=self.config["gan"]["discriminator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_discriminator = discriminator.PacketrateDiscriminator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            length=self.packetrate_dataset.max_len,
            config=self.config["gan"]["discriminator"]["packetrate"],
            device=device
        ).to(device)
        self.cascade_discriminator = discriminator.CascadeDiscriminator(
            flowlevel_discriminator=self.flowlevel_discriminator,
            packetrate_discriminator=self.packetrate_discriminator
        )
    
    def build_trainer(self, device):
        print("Building Trainers ...")

        self.flowlevel_trainer = trainer.FlowlevelTrainer(
            generator=self.flowlevel_generator,
            discriminator=self.flowlevel_discriminator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["trainer"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_trainer = trainer.PacketrateTrainer(
            generator=self.packetrate_generator,
            discriminator=self.packetrate_discriminator,
            config=self.config["gan"]["trainer"]["packetrate"],
            device=device
        ).to(device)
        self.cascade_trainer = trainer.CascadeTrainer(
            generator=self.cascade_generator,
            discriminator=self.cascade_discriminator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["trainer"]["cascade"],
            device=device
        )
    
    def build_evaluator(self, device):
        print("Building Evaluators ...")

        self.flowlevel_evaluator = evaluator.FlowlevelEvaluator(
            generator=self.flowlevel_generator,
            device=device
        ).to(device)
        self.packetrate_evaluator = evaluator.PacketrateEvaluator(
            generator=self.packetrate_generator,
            device=device
        ).to(device)
        self.cascade_evaluator = evaluator.CascadeEvaluator(
            generator=self.cascade_generator,
            device=device
        )
    
    def build_all(self, device, preprocess=True):
        self.build_processor(preprocess)
        self.build_generator(device)
        self.build_discriminator(device)
        self.build_trainer(device)
        self.build_evaluator(device)
    
    def build_lightweight(self, device, preprocess=False):
        self.build_processor(preprocess)
        self.build_generator(device)
        self.build_evaluator(device)
    
    def train_flowlevel(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Flowlevel ...")

        train_dataloader = data.build_train_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        self.flowlevel_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["flowlevel"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.flowlevel_generator.save(self.processor.output_folder)
        self.flowlevel_discriminator.save(self.processor.output_folder)
    
    def train_packetrate(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Packetrate ...")

        train_dataloader = data.build_train_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        self.packetrate_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["packetrate"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.packetrate_generator.save(self.processor.output_folder)
        self.packetrate_discriminator.save(self.processor.output_folder)
    
    def train_cascade(self, continue_epoch=0, save_epoch=None):
        print("Finetuning Flowlevel & Packetrate Together ...")

        train_dataloader = data.build_train_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["cascade"])
        self.cascade_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["cascade"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.cascade_generator.save(self.processor.output_folder)
        self.cascade_discriminator.save(self.processor.output_folder)
    
    def pretrain_finetune(self):
        self.train_flowlevel()
        self.train_packetrate()
        self.train_cascade()
    
    def find_all_checkpoints(self, generator, tag="pretrain"):
        for checkpoint in os.listdir(os.path.join(self.processor.output_folder, "gan", generator.__class__.__name__)):
            if checkpoint.startswith("checkpoint." + tag):
                yield checkpoint.lstrip("checkpoint")

    def generate_flowlevel(self, postfix=""):
        print("Generating Flowlevel ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.flowlevel_generator):
                self.flowlevel_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
                #raise NotImplementedError
        else:
            self.flowlevel_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
            #raise NotImplementedError
    
    def generate_packetrate(self, postfix=""):
        print("Generating Packetrate ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.packetrate_generator):
                self.packetrate_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
                self.processor.packetrate_post_process(
                    packetrate_result=(fivetuple, condition, packetrate),
                    result_filename="syn_packetrate" + _postfix
                )
        else:
            self.packetrate_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
            self.processor.packetrate_post_process(
                packetrate_result=(fivetuple, condition, packetrate),
                result_filename="syn_packetrate" + postfix
            )
    
    def generate_cascade(self, postfix=""):
        print("Generating Flowlevel & Packetrate Together ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["cascade"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.cascade_generator.flowlevel_generator, tag="finetune"):
                self.cascade_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate = self.cascade_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
                self.processor.packetrate_post_process(
                    packetrate_result=(fivetuple, condition, packetrate),
                    result_filename="syn_packetrate" + _postfix
                )
        else:
            self.cascade_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate = self.cascade_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
            self.processor.packetrate_post_process(
                packetrate_result=(fivetuple, condition, packetrate),
                result_filename="syn_packetrate" + postfix
            )
    
    def evaluate_all(self):

        self.generate_flowlevel("ALL")
        self.generate_packetrate("ALL")
        self.generate_cascade("ALL")
    
    def generate(self, postfix=""):

        self.generate_cascade(postfix)


class CascadeGANFTBuilder(object):
    def __init__(self, config):
        self.config = config
    
    def build_processor(self, preprocess=True):
        print("Building PrePostProcessor ...")

        self.processor = pre_post_processor.PrePostProcessor(config=self.config["pre_post_processor"])
        if preprocess:
            self.processor.pre_process()
        
        self.flowlevel_dataset = data.ConditionalFlowlevelDataset(path=self.processor.output_folder)
        self.packetrate_dataset = data.ConditionalPacketrateDataset(path=self.processor.output_folder)

    def build_generator(self, device):
        print("Building Generators ...")

        self.flowlevel_generator = generator.FlowlevelGenerator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["generator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_generator = generator.PacketrateGenerator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            config=self.config["gan"]["generator"]["packetrate"],
            device=device
        ).to(device)
    
    def build_discriminator(self, device):
        print("Building Discriminators ...")

        self.flowlevel_discriminator = discriminator.FlowlevelDiscriminator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            config=self.config["gan"]["discriminator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_discriminator = discriminator.PacketrateDiscriminator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            length=self.packetrate_dataset.max_len,
            config=self.config["gan"]["discriminator"]["packetrate"],
            device=device
        ).to(device)
    
    def build_trainer(self, device):
        print("Building Trainers ...")

        self.flowlevel_trainer = trainer.FlowlevelTrainer(
            generator=self.flowlevel_generator,
            discriminator=self.flowlevel_discriminator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["trainer"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_trainer = trainer.PacketrateTrainer(
            generator=self.packetrate_generator,
            discriminator=self.packetrate_discriminator,
            config=self.config["gan"]["trainer"]["packetrate"],
            device=device
        ).to(device)
    
    def build_evaluator(self, device):
        print("Building Evaluators ...")

        self.flowlevel_evaluator = evaluator.FlowlevelEvaluator(
            generator=self.flowlevel_generator,
            device=device
        ).to(device)
        self.packetrate_evaluator = evaluator.PacketrateEvaluator(
            generator=self.packetrate_generator,
            device=device
        ).to(device)
    
    def build_all(self, device, preprocess=True):
        self.build_processor(preprocess)
        self.build_generator(device)
        self.build_discriminator(device)
        self.build_trainer(device)
        self.build_evaluator(device)
    
    def build_lightweight(self, device, preprocess=False):
        raise NotImplementedError
    
    def pretrain_flowlevel(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Flowlevel ...")

        train_dataloader = data.build_train_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        self.flowlevel_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["flowlevel"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.flowlevel_generator.save(self.processor.output_folder)
        self.flowlevel_discriminator.save(self.processor.output_folder)
    
    def pretrain_packetrate(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Packetrate ...")

        train_dataloader = data.build_train_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        self.packetrate_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["packetrate"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.packetrate_generator.save(self.processor.output_folder)
        self.packetrate_discriminator.save(self.processor.output_folder)
    
    def finetune_packetrate(self, continue_epoch=0, save_epoch=None):
        print("Finetuning Packetrate ...")

        self.packetrate_ft_dataset.train_mode()
        train_dataloader = data.build_train_dataloader(dataset=self.packetrate_ft_dataset, config=self.config["dataloader"]["packetrate_ft"])
        self.packetrate_ft_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["packetrate_ft"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.packetrate_generator.save(self.processor.output_folder)
        self.packetrate_discriminator.save(self.processor.output_folder)
    
    def prepare_packetrate_ft(self, device):
        print("Preparing for Packetrate Finetune ...")
        
        self.generate_flowlevel()
        self.packetrate_ft_dataset = data.ConditionalPacketrateFTDataset(path=self.processor.output_folder)
        
        self.packetrate_ft_trainer = trainer.PacketrateFTTrainer(
            generator=self.packetrate_generator,
            discriminator=self.packetrate_discriminator,
            config=self.config["gan"]["trainer"]["packetrate_ft"],
            device=device
        ).to(device)
        
        self.packetrate_ft_evaluator = evaluator.PacketrateFTEvaluator(
            generator=self.packetrate_generator,
            device=device
        ).to(device)
    
    def pretrain_finetune(self, device):
        self.pretrain_flowlevel()
        self.pretrain_packetrate()
        self.prepare_packetrate_ft(device)
        self.finetune_packetrate()
    
    def find_all_checkpoints(self, generator, tag="pretrain"):
        for checkpoint in os.listdir(os.path.join(self.processor.output_folder, "gan", generator.__class__.__name__)):
            if checkpoint.startswith("checkpoint." + tag):
                yield checkpoint.lstrip("checkpoint")

    def generate_flowlevel(self, postfix=""):
        print("Generating Flowlevel ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.flowlevel_generator):
                self.flowlevel_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
                self.processor.flowlevel_post_process(
                    flowlevel_result=(condition, fivetuple),
                    result_filename="syn_flowlevel" + _postfix
                )
        else:
            self.flowlevel_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
            self.processor.flowlevel_post_process(
                flowlevel_result=(condition, fivetuple),
                result_filename="syn_flowlevel" + postfix
            )
    
    def generate_packetrate(self, postfix=""):
        print("Generating Packetrate ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.packetrate_generator):
                self.packetrate_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
                self.processor.packetrate_post_process(
                    packetrate_result=(fivetuple, condition, packetrate),
                    result_filename="syn_packetrate" + _postfix
                )
        else:
            self.packetrate_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
            self.processor.packetrate_post_process(
                packetrate_result=(fivetuple, condition, packetrate),
                result_filename="syn_packetrate" + postfix
            )
    
    def generate_packetrate_ft(self, postfix=""):
        print("Generating Finetuned Packetrate ...")

        self.packetrate_ft_dataset.eval_mode()
        generate_dataloader = data.build_generate_dataloader(dataset=self.packetrate_ft_dataset, config=self.config["dataloader"]["packetrate_ft"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.packetrate_generator, tag="finetune"):
                self.packetrate_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate = self.packetrate_ft_evaluator.generate(generate_dataloader, self.packetrate_ft_dataset.max_len)
                self.processor.packetrate_post_process(
                    packetrate_result=(fivetuple, condition, packetrate),
                    result_filename="syn_packetrate" + _postfix
                )
        else:
            self.packetrate_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate = self.packetrate_ft_evaluator.generate(generate_dataloader, self.packetrate_ft_dataset.max_len)
            self.processor.packetrate_post_process(
                packetrate_result=(fivetuple, condition, packetrate),
                result_filename="syn_packetrate" + postfix
            )
    
    def evaluate_all(self, device):

        self.generate_flowlevel("ALL")
        self.generate_packetrate("ALL")
        self.prepare_packetrate_ft(device)
        self.generate_packetrate_ft("ALL")
    
    def generate(self, postfix=""):

        raise NotImplementedError


class CascadeGANCompBuilder(object):
    def __init__(self, config, seed=6):
        self.config = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def build_processor(self, preprocess=True):
        print("Building PrePostProcessor ...")

        self.processor = pre_post_processor.PrePostProcessor(config=self.config["pre_post_processor"])
        if preprocess:
            self.processor.pre_process()
        else:
            self.processor.load()
        self.condition_dataset = data.ConditionDataset(path=self.processor.output_folder)
        self.flowlevel_dataset = data.ConditionalFlowlevelDataset(path=self.processor.output_folder)
        self.packetrate_dataset = data.ConditionalPacketrateDataset(path=self.processor.output_folder)
        self.packetfield_dataset = data.ConditionalPacketFieldDataset(path=self.processor.output_folder)
        self.cascade_comp_dataset = data.CascadeCompDataset(path=self.processor.output_folder)

    def build_generator(self, device):
        print("Building Generators ...")

        self.flowlevel_generator = generator.FlowlevelGenerator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            condition_normalization=self.processor.normalizations["condition"],
            fivetuple_normalization=self.processor.normalizations["fivetuple"],
            config=self.config["gan"]["generator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_generator = generator.PacketrateGenerator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            packetrate_normalization=self.processor.normalizations["packetrate"],
            config=self.config["gan"]["generator"]["packetrate"],
            device=device
        ).to(device)
        self.packetfield_generator = generator.PacketfieldGenerator(
            packetinfo_dim=self.packetfield_dataset.packetinfo_dim,
            packetfield_dim=self.packetfield_dataset.packetfield_dim,
            packetfield_normalization=self.processor.normalizations["packetfield"],
            config=self.config["gan"]["generator"]["packetfield"],
            device=device
        ).to(device)
        self.cascade_comp_generator = generator.CascadeCompGenerator(
            flowlevel_generator=self.flowlevel_generator,
            packetrate_generator=self.packetrate_generator,
            packetfield_generator=self.packetfield_generator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            processor=self.processor
        ).to(device)
    
    def build_discriminator(self, device):
        print("Building Discriminators ...")

        self.flowlevel_discriminator = discriminator.FlowlevelDiscriminator(
            condition_dim=self.flowlevel_dataset.condition_dim,
            fivetuple_dim=self.flowlevel_dataset.fivetuple_dim,
            config=self.config["gan"]["discriminator"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_discriminator = discriminator.PacketrateDiscriminator(
            flowlevel_dim=self.packetrate_dataset.condition_dim+self.packetrate_dataset.fivetuple_dim,
            packetrate_dim=self.packetrate_dataset.packetrate_dim,
            length=self.packetrate_dataset.max_len,
            config=self.config["gan"]["discriminator"]["packetrate"],
            device=device
        ).to(device)
        self.packetfield_discriminator = discriminator.PacketfieldDiscriminator(
            packetinfo_dim=self.packetfield_dataset.packetinfo_dim,
            packetfield_dim=self.packetfield_dataset.packetfield_dim,
            length=self.packetfield_dataset.field_max_len,
            config=self.config["gan"]["discriminator"]["packetfield"],
            device=device,
        ).to(device)
        self.cascade_comp_discriminator = discriminator.CascadeCompDiscriminator(
            flowlevel_discriminator=self.flowlevel_discriminator,
            packetrate_discriminator=self.packetrate_discriminator,
            packetfield_discriminator=self.packetfield_discriminator
        ).to(device)
    
    def build_trainer(self, device):
        print("Building Trainers ...")

        self.flowlevel_trainer = trainer.FlowlevelTrainer(
            generator=self.flowlevel_generator,
            discriminator=self.flowlevel_discriminator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["trainer"]["flowlevel"],
            device=device
        ).to(device)
        self.packetrate_trainer = trainer.PacketrateTrainer(
            generator=self.packetrate_generator,
            discriminator=self.packetrate_discriminator,
            config=self.config["gan"]["trainer"]["packetrate"],
            device=device
        ).to(device)
        self.packetfield_trainer = trainer.PacketfieldTrainer(
            generator=self.packetfield_generator,
            discriminator=self.packetfield_discriminator,
            config=self.config["gan"]["trainer"]["packetfield"],
            device=device
        ).to(device)
        self.cascade_comp_trainer = trainer.CascadeCompTrainer(
            generator=self.cascade_comp_generator,
            discriminator=self.cascade_comp_discriminator,
            condition_gen_flag=self.config["gan"]["condition_gen_flag"],
            config=self.config["gan"]["trainer"]["cascade_comp"],
            device=device
        ).to(device)
    
    def build_evaluator(self, device):
        print("Building Evaluators ...")

        self.flowlevel_evaluator = evaluator.FlowlevelEvaluator(
            generator=self.flowlevel_generator,
            device=device
        ).to(device)
        self.packetrate_evaluator = evaluator.PacketrateEvaluator(
            generator=self.packetrate_generator,
            device=device
        ).to(device)
        self.packetfield_evaluator = evaluator.PacketfieldEvaluator(
            generator=self.packetfield_generator,
            device=device
        ).to(device)
        self.cascade_comp_evaluator = evaluator.CascadeCompEvaluator(
            generator=self.cascade_comp_generator,
            device=device
        ).to(device)
        self.processor_evaluator = evaluator.ProcessorEvaluator()
    
    def build_all(self, device, preprocess=True):
        self.build_processor(preprocess)
        self.build_generator(device)
        self.build_discriminator(device)
        self.build_trainer(device)
        self.build_evaluator(device)
    
    def build_lightweight(self, device, preprocess=False):
        self.build_processor(preprocess)
        self.build_generator(device)
        self.build_evaluator(device)
    
    def train_flowlevel(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Flowlevel ...")

        train_dataloader = data.build_train_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        self.flowlevel_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["flowlevel"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.flowlevel_generator.save(self.processor.output_folder)
        self.flowlevel_discriminator.save(self.processor.output_folder)
    
    def train_packetrate(self, continue_epoch=0, save_epoch=None):
        print("Pretraining Packetrate ...")

        train_dataloader = data.build_train_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        self.packetrate_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["packetrate"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.packetrate_generator.save(self.processor.output_folder)
        self.packetrate_discriminator.save(self.processor.output_folder)
    
    def train_packetfield(self, continue_epoch=0, save_epoch=None, tqdm_per_epoch=False):
        print("Pretraining Packetfield ...")

        train_dataloader = data.build_train_dataloader(dataset=self.packetfield_dataset, config=self.config["dataloader"]["packetfield"])
        self.packetfield_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["packetfield"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch,
            tqdm_per_epoch=tqdm_per_epoch,
        )
        self.packetfield_generator.save(self.processor.output_folder)
        self.packetfield_discriminator.save(self.processor.output_folder)
    
    def train_cascade_comp(self, continue_epoch=0, save_epoch=None):
        print("Finetuning Flowlevel & Packetrate & Packetfield Together ...")

        self.cascade_comp_dataset.train_mode()
        train_dataloader = data.build_train_dataloader(dataset=self.cascade_comp_dataset, config=self.config["dataloader"]["cascade_comp"])
        self.cascade_comp_trainer.fit(
            dataloader=train_dataloader,
            epoch=self.config["epoch"]["cascade_comp"],
            path=self.processor.output_folder,
            continue_epoch=continue_epoch,
            save_epoch=save_epoch
        )
        self.cascade_comp_generator.save(self.processor.output_folder)
        self.cascade_comp_discriminator.save(self.processor.output_folder)
    
    def pretrain_finetune(self):
        save_epoch = None
        if "save_epoch" in self.config["epoch"]:
            save_epoch = self.config["epoch"]["save_epoch"]
        self.train_flowlevel(save_epoch=save_epoch)
        self.train_packetrate(save_epoch=save_epoch)
        self.train_packetfield(save_epoch=save_epoch)
        self.train_cascade_comp(save_epoch=save_epoch)
    
    def find_all_checkpoints(self, generator, tag="pretrain"):
        for checkpoint in os.listdir(os.path.join(self.processor.output_folder, "gan", generator.__class__.__name__)):
            if checkpoint.startswith("checkpoint." + tag):
                yield checkpoint.lstrip("checkpoint")

    def generate_flowlevel(self, postfix="", ppf=""):
        print("Generating Flowlevel ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.flowlevel_dataset, config=self.config["dataloader"]["flowlevel"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.flowlevel_generator):
                self.flowlevel_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
                self.processor.flowlevel_post_process(
                    flowlevel_result=(condition, fivetuple),
                    result_filename="syn_flowlevel" + _postfix + ppf
                )
        else:
            self.flowlevel_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple = self.flowlevel_evaluator.generate(generate_dataloader)
            self.processor.flowlevel_post_process(
                flowlevel_result=(condition, fivetuple),
                result_filename="syn_flowlevel" + postfix + ppf
            )
    
    def generate_packetrate(self, postfix="", ppf=""):
        print("Generating Packetrate ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.packetrate_dataset, config=self.config["dataloader"]["packetrate"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.packetrate_generator):
                self.packetrate_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
                self.processor.packetrate_post_process(
                    packetrate_result=(fivetuple, condition, packetrate),
                    result_filename="syn_packetrate" + _postfix + ppf
                )
        else:
            self.packetrate_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate = self.packetrate_evaluator.generate(generate_dataloader, self.packetrate_dataset.max_len)
            self.processor.packetrate_post_process(
                packetrate_result=(fivetuple, condition, packetrate),
                result_filename="syn_packetrate" + postfix + ppf
            )
    
    def generate_packetfield(self, postfix="", ppf=""):
        print("Generating Packetfield ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.packetfield_dataset, config=self.config["dataloader"]["packetfield"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.packetfield_generator):
                self.packetfield_generator.load(self.processor.output_folder, _postfix)
                packetinfo, packetfield = self.packetfield_evaluator.generate(generate_dataloader)
                self.processor.packetfield_post_process(
                    packetfield_result=(packetinfo, packetfield),
                    result_filename="syn_packetfield" + _postfix + ppf
                )
        else:
            self.packetfield_generator.load(self.processor.output_folder, postfix)
            packetinfo, packetfield = self.packetfield_evaluator.generate(generate_dataloader)
            self.processor.packetfield_post_process(
                packetfield_result=(packetinfo, packetfield),
                result_filename="syn_packetfield" + postfix + ppf
            )
    
    def generate_cascade_comp(self, generate=True, postfix="", ppf=""):
        print("Generating Flowlevel & Packetrate & Packetfield Together ...")

        generate_dataloader = data.build_generate_dataloader(dataset=self.condition_dataset, config=self.config["dataloader"]["cascade_comp"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.cascade_comp_generator.flowlevel_generator, tag="finetune"):
                if generate:
                    self.cascade_comp_generator.load(self.processor.output_folder, _postfix)
                    #condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate(generate_dataloader, self.condition_dataset.max_len)
                    condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate_faster(self.processor, generate_dataloader, self.condition_dataset.max_len)
                else:
                    condition = np.load(os.path.join(self.processor.postprocess_folder,"syn_comp" + _postfix + ".flowlevel.npz"))["condition"]
                    fivetuple = np.load(os.path.join(self.processor.postprocess_folder,"syn_comp" + _postfix + ".flowlevel.npz"))["fivetuple"]
                    packetrate = np.load(os.path.join(self.processor.postprocess_folder,"syn_comp" + _postfix + ".packetrate.npz"))["output"]
                    packetinfo = np.load(os.path.join(self.processor.postprocess_folder,"syn_comp" + _postfix + ".packetfield.npz"))["packetinfo"]
                    packetfield = np.load(os.path.join(self.processor.postprocess_folder,"syn_comp" + _postfix + ".packetfield.npz"))["packetfield"]
                self.processor.trace_post_process(
                    result=(condition, fivetuple, packetrate, packetinfo, packetfield),
                    result_filename="syn_comp" + _postfix + ppf
                )
        else:
            self.cascade_comp_generator.load(self.processor.output_folder, postfix)
            #condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate(generate_dataloader, self.condition_dataset.max_len)
            condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate_faster(self.processor, generate_dataloader, self.condition_dataset.max_len)
            self.processor.trace_post_process(
                result=(condition, fivetuple, packetrate, packetinfo, packetfield),
                result_filename="syn_comp" + postfix + ppf
            )
            
    def generate_copy(self):
        print("Generating Copy ...")
        
        self.cascade_comp_dataset.eval_mode()
        generate_dataloader = data.build_generate_dataloader_comp(dataset=self.cascade_comp_dataset, config=self.config["dataloader"]["cascade_comp"])
        condition, fivetuple, packetrate, packetinfo, packetfield = self.processor_evaluator.generate(generate_dataloader)
        self.processor.trace_post_process(
            result=(condition, fivetuple, packetrate, packetinfo, packetfield),
            result_filename="syn_copy"
        )
    
    def evaluate_all(self, generate=True, ppf=""):
        self.evaluate(generate=generate, postfix="ALL", ppf=ppf)

    def evaluate(self, generate=True, postfix="", ppf=""):
        #self.generate_flowlevel(postfix)
        #self.generate_packetrate(postfix)
        #self.generate_packetfield(postfix)
        self.generate_cascade_comp(generate=generate, postfix=postfix, ppf=ppf)

    def generate(self, postfix=""):

        self.generate_cascade_comp(postfix)
    
    def generate_from_another(self, postfix=""):

        self.another_processor = pre_post_processor.PrePostProcessor(config=self.config["another_pre_post_processor"])
        self.another_processor.pre_process()
        self.another_condition_dataset = data.ConditionDataset(path=self.another_processor.output_folder)

        generate_dataloader = data.build_generate_dataloader(dataset=self.another_condition_dataset, config=self.config["dataloader"]["cascade_comp"])
        if postfix == "ALL":
            for _postfix in self.find_all_checkpoints(self.cascade_comp_generator.flowlevel_generator, tag="finetune"):
                self.cascade_comp_generator.load(self.processor.output_folder, _postfix)
                condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate(generate_dataloader, self.another_condition_dataset.max_len)
                self.processor.trace_post_process(
                    result=(condition, fivetuple, packetrate, packetinfo, packetfield),
                    result_filename="another_syn_comp" + _postfix
                )
        else:
            self.cascade_comp_generator.load(self.processor.output_folder, postfix)
            condition, fivetuple, packetrate, packetinfo, packetfield = self.cascade_comp_evaluator.generate(generate_dataloader, self.another_condition_dataset.max_len)
            self.processor.trace_post_process(
                result=(condition, fivetuple, packetrate, packetinfo, packetfield),
                result_filename="another_syn_comp" + postfix
            )




    
    

    