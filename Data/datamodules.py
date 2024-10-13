
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from typing import *
from definitions import HFDatasetInfo,TorchVisionDatasetInfo,LowShotConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


from Data.visionlanguage_datasets import (SnliVeDataset_ViLT,
                                          VQADataset_ViLT,
                                          VCRDataset_ViLT,
                                          NLVR2Dataset_ViLT)
from Data.vision_datasets import (iNaturalist2019_Dataset_VILT,
                                  Places365_Dataset_VILT)
from Data.language_datasets import (PIQA_Dataset_VILT,
                                    CommonsenseQA_VILT)


class iNaturalistDataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 allow_uneven_batches:bool=True,
                 **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self,stage=None):
        self.train_dataset = iNaturalist2019_Dataset_VILT(self.data_dir,
                                                            split="train",
                                                            low_shot_config=self.low_shot_config,
                                                            VILT_ckpt_dir = self.VILT_ckpt_dir,
                                                            VILT_tokenizer = self.VILT_tokenizer)
        self.valid_dataset = iNaturalist2019_Dataset_VILT(self.data_dir,
                                                            split="val",
                                                            low_shot_config=self.low_shot_config,
                                                            VILT_ckpt_dir = self.VILT_ckpt_dir,
                                                            VILT_tokenizer = self.VILT_tokenizer)
        self.collate_fn = self.train_dataset.iNatural2019_batch_collate
        
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)
    
class Places365DataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 allow_uneven_batches:bool=True,
                 **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.data_dir = data_dir
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers    
    def setup(self, stage=None):
        self.train_dataset = Places365_Dataset_VILT(self.data_dir,
                                                    split="train",
                                                    low_shot_config=self.low_shot_config,
                                                    VILT_ckpt_dir=self.VILT_ckpt_dir,
                                                    VILT_tokenizer=self.VILT_tokenizer)
        self.valid_dataset = Places365_Dataset_VILT(self.data_dir,
                                                    split="val",
                                                    low_shot_config=self.low_shot_config,
                                                    VILT_ckpt_dir=self.VILT_ckpt_dir,
                                                    VILT_tokenizer=self.VILT_tokenizer)
        
        self.collate_fn = self.train_dataset.Places365_batch_collate
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: self.collate_fn(x),
                          pin_memory=True)
    def val_dataloader(self):
        return DataLoader(dataset=self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: self.collate_fn(x),
                          pin_memory=True)
        

class PIQADataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 allow_uneven_batches:bool=True,
                 **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self,stage=None):
        self.train_dataset = PIQA_Dataset_VILT(VILT_ckpt_dir=self.VILT_ckpt_dir,
                                                VILT_tokenizer=self.VILT_tokenizer,
                                                data_dir=self.data_dir,
                                                split="train",
                                            low_shot_config = self.low_shot_config
                                                )
        self.val_dataset = PIQA_Dataset_VILT(VILT_ckpt_dir=self.VILT_ckpt_dir,
                                                VILT_tokenizer=self.VILT_tokenizer,
                                                data_dir=self.data_dir,
                                                split="val",
                                            low_shot_config = self.low_shot_config
                                                )
        self.collate_fn = self.train_dataset.piqa_batch_collate
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)
    
class CommonsenseQADataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 allow_uneven_batches:bool=True,
                 **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self,stage=None):
        self.train_dataset = CommonsenseQA_VILT(VILT_ckpt_dir=self.VILT_ckpt_dir,
                                                VILT_tokenizer=self.VILT_tokenizer,
                                                data_dir=self.data_dir,
                                                split="train",
                                                low_shot_config=self.low_shot_config)
        
        self.val_dataset = CommonsenseQA_VILT(VILT_ckpt_dir=self.VILT_ckpt_dir,
                                              VILT_tokenizer=self.VILT_tokenizer,
                                              data_dir=self.data_dir,
                                              split="val",
                                              low_shot_config=self.low_shot_config)
        
        self.collate_fn = self.train_dataset.CommonsenseQA_batch_collate

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=lambda x:self.collate_fn(x),
                          pin_memory=True)

class VQADataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int,
                 num_workers:int,
                 allow_uneven_batches:bool=True,
                 **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self,stage=None):
        if self.model_key == "VILT":
            self.train_dataset = VQADataset_ViLT(data_dir=self.data_dir,
                                            split="train",
                                            low_shot_config = self.low_shot_config,
                                            VILT_ckpt_dir = self.VILT_ckpt_dir,
                                            VILT_tokenizer = self.VILT_tokenizer)
            self.val_dataset = VQADataset_ViLT(data_dir=self.data_dir,
                                                split="val",
                                            low_shot_config = self.low_shot_config,
                                                VILT_ckpt_dir = self.VILT_ckpt_dir,
                                                VILT_tokenizer = self.VILT_tokenizer)
            
            self.collate_fn = self.train_dataset.vqa_batch_collate_ViLT
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.train_dataset,
                                           batch_size = self.batch_size,
                                           num_workers = self.num_workers,
                                           shuffle = True,
                                           pin_memory=True,
                                           collate_fn = lambda x:self.collate_fn(x))
        
    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset = self.val_dataset,
                                           batch_size = self.batch_size,
                                           num_workers = self.num_workers,
                                           shuffle = False,
                                           pin_memory=True,
                                           collate_fn = lambda x:self.collate_fn(x))
    
class NLVR2DataModule(LightningDataModule):
    def __init__(self,
                low_shot_config:LowShotConfig,
                model_key:str,
                VILT_ckpt_dir:str,
                VILT_tokenizer:str,
                data_dir:str,
                batch_size:int,
                num_workers:int,
                allow_uneven_batches:bool=True,
                **kwargs:Any):
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        self.train_dataset = NLVR2Dataset_ViLT(split="train",
                                             low_shot_config=self.low_shot_config,
                                             data_dir=self.data_dir,
                                             VILT_ckpt_dir=self.VILT_ckpt_dir,
                                             VILT_tokenizer=self.VILT_tokenizer)
        self.val_dataset = NLVR2Dataset_ViLT(split="dev",
                                             low_shot_config=self.low_shot_config,
                                             data_dir=self.data_dir,
                                             VILT_ckpt_dir=self.VILT_ckpt_dir,
                                             VILT_tokenizer=self.VILT_tokenizer)
        self.collate_fn = self.train_dataset.NLVR2_batch_collate

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset = self.train_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          pin_memory = True,
                          collate_fn = lambda x:self.collate_fn(x))
    def val_dataloader(self):
        return DataLoader(dataset = self.val_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = False,
                          pin_memory=True,
                          collate_fn = lambda x:self.collate_fn(x))
    

class VCRDataModule(LightningDataModule):
    def __init__(self,
                low_shot_config:LowShotConfig,
                model_key:str,
                VILT_ckpt_dir:str,
                VILT_tokenizer:str,
                data_dir:str,
                batch_size:int,
                num_workers:int,
                allow_uneven_batches:bool=True,
                **kwargs:Any):
        
        super().__init__()
        self.low_shot_config = low_shot_config
        self.model_key = model_key
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self,stage=None):
        self.train_dataset = VCRDataset_ViLT(split="train",
                                             low_shot_config=self.low_shot_config,
                                             data_dir=self.data_dir,
                                             VILT_ckpt_dir=self.VILT_ckpt_dir,
                                             VILT_tokenizer=self.VILT_tokenizer)
        self.val_dataset = VCRDataset_ViLT(split="val",
                                           low_shot_config=self.low_shot_config,
                                           data_dir=self.data_dir,
                                           VILT_ckpt_dir=self.VILT_ckpt_dir,
                                           VILT_tokenizer=self.VILT_tokenizer)
        self.collate_fn = self.train_dataset.VCR_batch_collate
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(dataset = self.train_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          pin_memory = True,
                          collate_fn = lambda x:self.collate_fn(x))
    def val_dataloader(self):
        return DataLoader(dataset = self.val_dataset,
                          batch_size = self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = False,
                          pin_memory=True,
                          collate_fn = lambda x:self.collate_fn(x))
    
class SnliveDataModule(LightningDataModule):
    def __init__(self,
                 low_shot_config:LowShotConfig,
                 model_key:str,
                 VILT_ckpt_dir:str,
                 VILT_tokenizer:str,
                 data_dir:str,
                 batch_size:int = 32,
                 test_batch_size:int = 32,
                 num_workers:int = 4,
                 allow_uneven_batches:bool=True):
        super().__init__()
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.VILT_tokenizer = VILT_tokenizer
        self.model_key = model_key
        self.data_root = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.low_shot_config = low_shot_config
        
    def setup(self,stage=None):
        if self.model_key == "VILT":
            self.train_dataset = SnliVeDataset_ViLT(split="train",
                                                    low_shot_config=self.low_shot_config,
                                                data_dir=self.data_root,
                                                VILT_ckpt_dir = self.VILT_ckpt_dir,
                                                VILT_tokenizer = self.VILT_tokenizer
                                                )
            self.val_dataset = SnliVeDataset_ViLT(split="dev",
                                                low_shot_config=self.low_shot_config,
                                                data_dir=self.data_root,
                                                VILT_ckpt_dir = self.VILT_ckpt_dir,
                                                VILT_tokenizer = self.VILT_tokenizer
                                                )

    def train_dataloader(self):
        collate_fn = self.train_dataset.snlive_batch_collate_ViLT
        return torch.utils.data.DataLoader(dataset = self.train_dataset,
                                        batch_size = self.batch_size,
                                        num_workers = self.num_workers,
                                        shuffle = True,
                                        pin_memory = True,
                                        collate_fn = lambda x:collate_fn(x),
                                        # sampler = train_sampler
                                        )
    def val_dataloader(self):
        collate_fn = self.train_dataset.snlive_batch_collate_ViLT
        return torch.utils.data.DataLoader(dataset = self.val_dataset,
                                           batch_size = self.batch_size,
                                           num_workers = self.num_workers,
                                           shuffle = False,
                                           pin_memory=True,
                                           collate_fn = lambda x:collate_fn(x)
                                           )
