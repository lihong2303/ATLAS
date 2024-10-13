
import os
import torch
from torch import nn,Tensor
from typing import Callable, Tuple,Any,Union,Optional
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from transformers import get_polynomial_decay_schedule_with_warmup
from Model.vilt import VILT_model
from definitions import Adapter,Vision_adapter_infos,Text_adapter_infos,CL_setting
from Utils.metric import vqa_compute_score_with_logits


class VILTLightningModule(LightningModule):
    def __init__(self,
                # training parameters
                batch_size:int = 128,
                learning_rate: float = 0.0002,
                adam_eps: float = 1.0e-08,
                adam_weight_decay: float = 0.01,
                adam_betas: Tuple[float, float] = (0.9, 0.999),
                warmup_ratio: int = 0.1,
                max_steps: int = 450000,
                classifier_in_dim:int = 1024,
                num_classes:int = 3,
                # Model
                VILT_ckpt_dir:str = '',
                init_checkpoint_path:str = '',
                update_method:str = "cls-head",
                target_model:str = None,
                adapter_weighted_method:str = None,
                continual_sequence: str = None,
                cur_dataset:str = None,
                cl_setting:Optional[CL_setting] = None,
                adapter:Optional[Adapter] = Adapter(Vision_adapter_infos(False,256),
                                                    Text_adapter_infos(False,256))
                ):
        super().__init__()
        self.batch_size = batch_size
        self.target_model = target_model
        self.cur_dataset = cur_dataset
        self.update_method = update_method
        self.cl_setting = cl_setting
        self.init_checkpoint_path = init_checkpoint_path
        self.VILT = VILT_model(VILT_ckpt_dir=VILT_ckpt_dir,
                               classifier_in_dim=classifier_in_dim,
                               num_classes=num_classes,
                               target_model=target_model,
                               adapter_weighted_method=adapter_weighted_method,
                               continual_sequence =continual_sequence,
                               cur_dataset=cur_dataset,
                               update_method=update_method,
                               adapter=adapter)
                
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.load_vilt_state_dict()
        
        if self.cur_dataset in ["snlive","piqa","iNaturalist","commonsenseqa","places365","vcr","nlvr2"]:
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.cur_dataset == "vqa":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = max_steps * warmup_ratio
        self.max_steps = max_steps
        if self.cur_dataset in ["snlive","piqa","iNaturalist","commonsenseqa","places365","vcr","nlvr2"]:
            self.metric = Accuracy(task="multiclass",
                                num_classes=num_classes)
        elif self.cur_dataset == "vqa":
            self.metric = vqa_compute_score_with_logits
    
    def load_vilt_state_dict(self):
        if self.init_checkpoint_path != "None":
            state_dict = torch.load(self.init_checkpoint_path)["state_dict"]
            self.load_state_dict(state_dict,strict=False)
            
        else:
            VILT_state_dict = torch.load(os.path.join(self.VILT_ckpt_dir,"vilt","pytorch_model.bin"))
            self.VILT.load_state_dict(VILT_state_dict,strict=False)
    
    def training_step(self,batch,batch_idx):
        loss,accuracy = self._step(batch,batch_idx)
        self.log("train_losses",
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                sync_dist=True,
                batch_size=self.batch_size)
        self.log("train_acc",
                accuracy,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size
                )
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss,accuracy = self._step(batch,batch_idx)
        self.log("val_losses",
                loss,
                prog_bar=True,
                logger=True,
                on_step=True,
                sync_dist=True,
                batch_size=self.batch_size
                )
        self.log("val_acc",
                accuracy,
                prog_bar=True,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                batch_size=self.batch_size)
        return loss
    def _step(self,batch,batch_idx):
        if self.training:
            self.VILT.vilt.encoder.ortho_loss = 0.
        if self.cur_dataset == "snlive":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.training:
                loss = self.loss_fn(output_logits,labels) + self.VILT.vilt.encoder.ortho_loss
            else:
                loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "piqa":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.cl_setting.key == "ewc":
                orig_loss = self.loss_fn(output_logits,labels)
                task,ewc_loss = self.ewc.compute_ewc_loss(self.VILT)
                loss = orig_loss + ewc_loss
            else:
                if self.training:
                    loss = self.loss_fn(output_logits,labels) + self.VILT.vilt.encoder.ortho_loss
                else:
                    loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "commonsenseqa":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.training:
                loss = self.loss_fn(output_logits,labels) + self.VILT.vilt.encoder.ortho_loss
            else:
                loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "iNaturalist":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.cl_setting.key == "ewc":
                orig_loss = self.loss_fn(output_logits,labels)
                task,ewc_loss = self.ewc.compute_ewc_loss(self.VILT)
                loss = orig_loss + ewc_loss
            else:
                if self.training:
                    loss = self.loss_fn(output_logits,labels)  + self.VILT.vilt.encoder.ortho_loss
                else:
                    loss = self.loss_fn(output_logits,labels)
            
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "places365":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.training:
                loss = self.loss_fn(output_logits,labels)  + self.VILT.vilt.encoder.ortho_loss
            else:
                loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)

        elif self.cur_dataset == "vqa":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.cl_setting.key == "ewc":
                orig_loss = self.loss_fn(output_logits,labels)* labels.shape[1]
                task,ewc_loss = self.ewc.compute_ewc_loss(self.VILT)
                loss = orig_loss + ewc_loss
            else:
                if self.training:
                    loss = self.loss_fn(output_logits,labels)* labels.shape[1]  + self.VILT.vilt.encoder.ortho_loss
                else:
                    loss = self.loss_fn(output_logits,labels)* labels.shape[1]
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "vcr":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.training:
                loss = self.loss_fn(output_logits,labels)  + self.VILT.vilt.encoder.ortho_loss
            else:
                loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)
        elif self.cur_dataset == "nlvr2":
            encodings = batch["encodings"]
            labels = batch["labels"]
            encoder_output,output_logits = self.VILT(encodings)
            if self.training:
                loss = self.loss_fn(output_logits,labels)  + self.VILT.vilt.encoder.ortho_loss
            else:
                loss = self.loss_fn(output_logits,labels)
            accuracy = self.metric(output_logits,labels)
        return loss,accuracy
    
    def configure_optimizers(self):
        return self.get_optimizers_for_lightning(
                                            update_method=self.update_method,
                                            model=self.VILT,
                                            learning_rate=self.learning_rate,
                                            adam_eps=self.adam_eps,
                                            adam_weight_decay=self.adam_weight_decay,
                                            adam_betas=self.adam_betas,
                                            warmup_steps=self.warmup_steps,
                                            max_steps=self.max_steps,
                                            )
    def get_optimizers_for_lightning(self,
                                     update_method:str,
                                     model:torch.nn.Module,
                                     learning_rate:float,
                                     adam_eps:float,
                                     adam_weight_decay:float,
                                     adam_betas:Tuple[float,float],
                                     warmup_steps: int,
                                     max_steps:int,
                                     ):
        trainable_params = []
        no_decay = ['bias', 'LayerNorm.weight']
        for name,params in model.named_parameters():
            if not any(nd in name for nd in no_decay):
                weight_decay = adam_weight_decay
            else:
                weight_decay = 0.0
            if update_method == "cls-head" and name.__contains__('classifier'):
                # only update classify head
                params.requires_grad = True
                trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate}]
            elif update_method == "adapter" and (name.__contains__('classifier') or name.__contains__(f'{self.cur_dataset}_adapter') or name.__contains__('add_adapter') or name.__contains__('add')):
                
                params.requires_grad = True
                trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate}]
            elif update_method == "fine-tuning":
                params.requires_grad = True
                trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate}]
            elif update_method == "task-incremental-generalize" and (name.__contains__('query_aftattn') or name.__contains__('query_aftmlp') or name.__contains__('classifier')):
                params.requires_grad = True
                if name.__contains__("query_aftattn") or name.__contains__("query_aftmlp") or name.__contains__("key_aftattn") or name.__contains__("key_aftmlp"):
                    trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate * 100}]
                else:
                    trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate * 1}]
            elif update_method == "task-incremental-update" and (name.__contains__(f'{self.cur_dataset}') or name.__contains__('add') or name.__contains__('classifier')):
                params.requires_grad = True
                if name.__contains__("query_aftattn") or name.__contains__("query_aftmlp") or name.__contains__("key_aftattn") or name.__contains__("key_aftmlp"):
                    trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate * 1}]
                elif name.__contains__("add_adapter"):
                    trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate * 1}]
                else:
                    trainable_params += [{'params':params,'weight_decay':weight_decay,'lr':learning_rate}]
            else:
                params.requires_grad = False
                
        if len(trainable_params) == 0:
            raise NotImplementedError(f"{update_method} is not suitable for current methods, please specify it!")
        
        optimizer = torch.optim.AdamW(params=trainable_params,
                                    betas=adam_betas,
                                    eps=adam_eps)
        
        
        scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                              num_warmup_steps=warmup_steps,
                                                              num_training_steps=max_steps,
                                                              lr_end=0,
                                                              power=1,)
        return [optimizer],[{"scheduler":scheduler,"interval":"step"}]