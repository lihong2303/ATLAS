
import io
import os
import torch
import logging
import torch.distributed as dist
from collections import defaultdict
from omegaconf import OmegaConf
from Config.config import build_config,build_datamodule_kwargs
from definitions import MMCLArguments
from pytorch_lightning import seed_everything,Trainer
from Data import (SnliveDataModule,
                  VQADataModule,
                  VCRDataModule,
                  NLVR2DataModule,
                  PIQADataModule,
                  CommonsenseQADataModule,
                  iNaturalistDataModule,
                  Places365DataModule)
from pytorch_lightning.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint)
from Model import VILTLightningModule
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger


def main():
    config:MMCLArguments = build_config()
    
    if config.training.seed != -1:
        seed_everything(config.training.seed,workers=True)
    # Load logger
    tensorlogger = TensorBoardLogger(save_dir="Checkpoint",
                               name=config.training.cur_dataset,
                               version=config.training.cur_expt_name,
                               )
    csvlogger = CSVLogger(save_dir="Checkpoint",
                          name=config.training.cur_dataset,
                          version=config.training.cur_expt_name,)
    logger = logging.getLogger(__name__)
    
    # Load dataset
    if config.training.cur_dataset == "snlive":
        datamodule = SnliveDataModule(low_shot_config = config.datasets.vl.low_shot_config,
                                      model_key = config.model.key, ##
                                      VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                      VILT_tokenizer = config.model.VILT_tokenizer,
                                      **build_datamodule_kwargs(config.datasets.vl))
        batch_size = config.datasets.vl.batch_size
    elif config.training.cur_dataset == "vqa":
        datamodule = VQADataModule(low_shot_config = config.datasets.vl.low_shot_config,
                                   model_key = config.model.key,
                                    VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                    VILT_tokenizer = config.model.VILT_tokenizer,
                                   **build_datamodule_kwargs(config.datasets.vl))
        batch_size = config.datasets.vl.batch_size
    elif config.training.cur_dataset == "vcr":
        datamodule = VCRDataModule(low_shot_config = config.datasets.vl.low_shot_config,
                                   model_key = config.model.key,
                                    VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                    VILT_tokenizer = config.model.VILT_tokenizer,
                                   **build_datamodule_kwargs(config.datasets.vl))
        batch_size = config.datasets.vl.batch_size
    
    elif config.training.cur_dataset == "nlvr2":
        datamodule = NLVR2DataModule(low_shot_config = config.datasets.vl.low_shot_config,
                                   model_key = config.model.key,
                                    VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                    VILT_tokenizer = config.model.VILT_tokenizer,
                                   **build_datamodule_kwargs(config.datasets.vl))
        batch_size = config.datasets.vl.batch_size

    elif config.training.cur_dataset == "piqa":
        datamodule = PIQADataModule(low_shot_config = config.datasets.text.low_shot_config,
                                    model_key = config.model.key,
                                    VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                    VILT_tokenizer = config.model.VILT_tokenizer,
                                    **build_datamodule_kwargs(config.datasets.text))
        batch_size = config.datasets.text.batch_size 
    
    elif config.training.cur_dataset == "commonsenseqa":
        datamodule = CommonsenseQADataModule(low_shot_config = config.datasets.text.low_shot_config,
                                    model_key = config.model.key,
                                    VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                    VILT_tokenizer = config.model.VILT_tokenizer,
                                    **build_datamodule_kwargs(config.datasets.text))
        batch_size = config.datasets.text.batch_size

    elif config.training.cur_dataset == "iNaturalist":
        datamodule = iNaturalistDataModule(low_shot_config = config.datasets.image.low_shot_config,
                                           model_key = config.model.key,
                                               VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                               VILT_tokenizer = config.model.VILT_tokenizer,
                                               **build_datamodule_kwargs(config.datasets.image))
        batch_size = config.datasets.image.batch_size
    elif config.training.cur_dataset == "places365":
        datamodule = Places365DataModule(low_shot_config = config.datasets.image.low_shot_config,
                                           model_key = config.model.key,
                                               VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                               VILT_tokenizer = config.model.VILT_tokenizer,
                                               **build_datamodule_kwargs(config.datasets.image))
        batch_size = config.datasets.image.batch_size
    
    else:
        raise NotImplementedError
    
    datamodule.setup(stage="fit")
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    
    
    model = VILTLightningModule(batch_size=batch_size,
                                learning_rate=config.training.learning_rate,
                                adam_eps=config.training.adam_eps,
                                adam_weight_decay=config.training.adam_weight_decay,
                                adam_betas=config.training.adam_betas,
                                warmup_ratio=config.training.warmup_ratio,
                                max_steps=len(train_dataloader) * config.training.lightning.max_epochs,
                                VILT_ckpt_dir=config.model.VILT_ckpt_dir,
                                init_checkpoint_path=config.training.initialize_from_checkpoint,
                                classifier_in_dim=config.model.classifier_in_dim,
                                num_classes=config.model.num_classes,
                                update_method=config.model.update_method,
                                target_model = config.training.target_model,
                                adapter_weighted_method = config.training.adapter_weighted_method,
                                continual_sequence = config.training.continual_sequence,
                                cur_dataset=config.training.cur_dataset,
                                cl_setting = config.model.cl_setting,
                                adapter=config.model.adapter
                                )
    
    
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(dirpath=os.path.join(config.training.lightning.default_root_dir,config.training.cur_expt_name,"checkpoint"), # directory to save the model file.
                        filename="{epoch}-{val_acc:.4f}", # checkpoint filename
                        monitor="val_acc", # quantity to monitor
                        save_top_k=1, # the best k models according to the quantity monitored will be saved.
                        mode="max" 
                        )
    ]


    if config.training.resume_from_checkpoint != "None":
        trainer = Trainer(
            logger=[tensorlogger,csvlogger],
            **OmegaConf.to_container(config.training.lightning),
            callbacks=callbacks,
            resume_from_checkpoint=config.training.resume_from_checkpoint,
        )
    else:
        trainer = Trainer(
            logger=[tensorlogger,csvlogger],
            **OmegaConf.to_container(config.training.lightning),
            callbacks=callbacks
        )
    
    trainer.fit(model = model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
                )
    
    
if __name__ == "__main__":
    main()
