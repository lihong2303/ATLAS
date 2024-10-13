
import os
import torch
from typing import Any, Mapping, Optional,Union,List,Dict
from torch import Tensor,nn 
from definitions import Adapter
from transformers import ViltConfig
from Vilt.model import ViltModel
from Model.layers.mlp import MLP



class VILT_model(nn.Module):
    def __init__(self,
                 VILT_ckpt_dir:str,
                 classifier_in_dim:int,
                 num_classes:Optional[int],
                 target_model:str,
                 adapter_weighted_method:str,
                 continual_sequence:str,
                 cur_dataset:str,
                 update_method:str,
                 adapter:Optional[Adapter]=None
                 ):
        super().__init__()
        self.VILT_ckpt_dir = VILT_ckpt_dir
        self.target_model = target_model
        self.cur_dataset = cur_dataset
        vilt_config = ViltConfig.from_pretrained(os.path.join(VILT_ckpt_dir,"vilt"))
        self.vilt = ViltModel(vilt_config,
                              target_model=target_model,
                              adapter_weighted_method=adapter_weighted_method,
                              continual_sequence = continual_sequence,
                              cur_dataset=self.cur_dataset,
                              update_method=update_method,
                              adapter=adapter)
        if self.cur_dataset in ["piqa"]: # multi-label
            self.piqa_classifier = MLP(in_dim=classifier_in_dim,
                                out_dim=1,
                                hidden_dims=classifier_in_dim * 2,
                                activation=nn.GELU,
                                normalization=LayerNorm,
                                )
        elif self.cur_dataset == "snlive":
            self.snlive_classifier = MLP(in_dim=classifier_in_dim,
                                    out_dim=3,
                                    hidden_dims=classifier_in_dim * 2,
                                    activation=nn.GELU,
                                    normalization=LayerNorm,
                                    )
        elif self.cur_dataset == "vqa":
            self.vqa_classifier = MLP(in_dim=classifier_in_dim,
                                    out_dim=3129,
                                    hidden_dims=classifier_in_dim * 2,
                                    activation=nn.GELU,
                                    normalization=LayerNorm,
                                    )
        elif self.cur_dataset == "nlvr2":
            self.nlvr2_classifier = MLP(in_dim=classifier_in_dim*2,
                                    out_dim=2,
                                    hidden_dims=classifier_in_dim * 4,
                                    activation=nn.GELU,
                                    normalization=LayerNorm,
                                    )
        elif self.cur_dataset == "iNaturalist":
            self.iNaturalist_classifier = MLP(in_dim=classifier_in_dim,
                                    out_dim=1010,
                                    hidden_dims=classifier_in_dim * 2,
                                    activation=nn.GELU,
                                    normalization=LayerNorm,
                                    )
        elif self.cur_dataset == "places365":
            self.places365_classifier = MLP(in_dim=classifier_in_dim,
                                            out_dim=365,
                                            hidden_dims=classifier_in_dim * 2,
                                            activation=nn.GELU,
                                            normalization=LayerNorm,
                                            )
        elif self.cur_dataset in ["vcr"]: # multi-label
            self.vcr_classifier = MLP(in_dim=classifier_in_dim,
                                out_dim=1,
                                hidden_dims=classifier_in_dim * 2,
                                activation=nn.GELU,
                                normalization=LayerNorm,
                                )
        elif self.cur_dataset in ["commonsenseqa"]: # multi-label
            self.commonsenseqa_classifier = MLP(in_dim=classifier_in_dim,
                                                out_dim=1,
                                                hidden_dims=classifier_in_dim * 2,
                                                activation=nn.GELU,
                                                normalization=LayerNorm,
                                                )
        
    def forward(self,
                encodings):
        if self.cur_dataset == "piqa":
            bs = encodings["pixel_values"].size(0)
            unflat_input_ids = encodings['input_ids'].view(bs, 2, -1)
            unflat_attention_mask = encodings['attention_mask'].view(bs, 2, -1)
            unflat_token_type_ids = encodings['token_type_ids'].view(bs, 2, -1)
            pixel_values, pixel_mask = encodings['pixel_values'], encodings['pixel_mask']

            pooler_outputs = []
            for i in range(2):
                encodings = {
                    'input_ids': unflat_input_ids[:, i, :],
                    'attention_mask': unflat_attention_mask[:, i, :],
                    'token_type_ids': unflat_token_type_ids[:, i, :],
                    'pixel_values': pixel_values,
                    'pixel_mask': pixel_mask
                }
                pooled_out = self.vilt(**encodings).pooler_output
                pooler_outputs.append(pooled_out)
            pooled_output = torch.stack(pooler_outputs, dim=0).transpose(0, 1)

            output_logits = self.piqa_classifier(pooled_output).squeeze()
            
            return pooler_outputs,output_logits
        elif self.cur_dataset == "snlive":
            encoder_output = self.vilt(**encodings).pooler_output
            output_logits = self.snlive_classifier(encoder_output)
        
            return encoder_output,output_logits
        elif self.cur_dataset == "vqa":
            encoder_output = self.vilt(**encodings).pooler_output
            output_logits = self.vqa_classifier(encoder_output)
        
            return encoder_output,output_logits
        elif self.cur_dataset == "iNaturalist":
            encoder_output = self.vilt(**encodings).pooler_output
            output_logits = self.iNaturalist_classifier(encoder_output)
        
            return encoder_output,output_logits
        elif self.cur_dataset == "places365":
            encoder_output = self.vilt(**encodings).pooler_output
            output_logits = self.places365_classifier(encoder_output)
        
            return encoder_output,output_logits
        
        elif self.cur_dataset == "vcr":
            bs = encodings["pixel_values"].size(0)
            unflat_input_ids = encodings['input_ids']
            unflat_attention_mask = encodings['attention_mask']
            unflat_token_type_ids = encodings['token_type_ids']
            pixel_values, pixel_mask = encodings['pixel_values'], encodings['pixel_mask']

            pooler_outputs = []
            for i in range(4):
                encodings = {
                    'input_ids': unflat_input_ids[:, i, :],
                    'attention_mask': unflat_attention_mask[:, i, :],
                    'token_type_ids': unflat_token_type_ids[:, i, :],
                    'pixel_values': pixel_values,
                    'pixel_mask': pixel_mask
                }
                pooled_out = self.vilt(**encodings).pooler_output
                pooler_outputs.append(pooled_out)
            pooled_output = torch.stack(pooler_outputs, dim=0).transpose(0, 1)

            output_logits = self.vcr_classifier(pooled_output).squeeze()
            
            return pooler_outputs,output_logits
        elif self.cur_dataset == "nlvr2":
            bs = encodings["input_ids"].size(0)
            unflat_input_ids = encodings['input_ids']
            unflat_attention_mask = encodings['attention_mask']
            unflat_token_type_ids = encodings['token_type_ids']
            unflat_pixel_values = encodings['pixel_values']
            unflat_pixel_mask = encodings['pixel_mask']
            pooler_outputs = []
            for i in range(2):
                encodings = {
                'input_ids': unflat_input_ids,
                'attention_mask': unflat_attention_mask,
                'token_type_ids': unflat_token_type_ids,
                'pixel_values': unflat_pixel_values[:, i, :, :, :],
                'pixel_mask': unflat_pixel_mask[:, i, :, :],
                }
                pooled_out = self.vilt(**encodings).pooler_output
                pooler_outputs.append(pooled_out)
            pooled_output = torch.cat(pooler_outputs,dim=-1)

            output_logits = self.nlvr2_classifier(pooled_output)
            return pooler_outputs,output_logits

        elif self.cur_dataset == "commonsenseqa":
            bs = encodings["pixel_values"].size(0)
            unflat_input_ids = encodings['input_ids'].view(bs, 5, -1)
            unflat_attention_mask = encodings['attention_mask'].view(bs, 5, -1)
            unflat_token_type_ids = encodings['token_type_ids'].view(bs, 5, -1)
            pixel_values, pixel_mask = encodings['pixel_values'], encodings['pixel_mask']
            pooler_outputs = []
            for i in range(5):
                encodings = {
                    'input_ids': unflat_input_ids[:, i, :],
                    'attention_mask': unflat_attention_mask[:, i, :],
                    'token_type_ids': unflat_token_type_ids[:, i, :],
                    'pixel_values': pixel_values,
                    'pixel_mask': pixel_mask
                }
                pooled_out = self.vilt(**encodings).pooler_output
                pooler_outputs.append(pooled_out)
            pooled_output = torch.stack(pooler_outputs, dim=0).transpose(0, 1)

            output_logits = self.commonsenseqa_classifier(pooled_output).squeeze()
            
            return pooler_outputs,output_logits
        
        else:
            raise NotImplementedError
        
        
        
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)