
import torch
from torch import nn

from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vilt.modeling_vilt import (ViltAttention,
                                                    ViltIntermediate,
                                                    TextEmbeddings,
                                                    ViltPatchEmbeddings
                                                    )
from transformers.pytorch_utils import meshgrid
from definitions import Adapter_infos
from Model.layers.adapter import Adapter
from transformers.models.vilt.configuration_vilt import ViltConfig
from Model.layers.cross_attn import cross_attn_noproj
from typing import Any

class ViltEncoder(nn.Module):
    def __init__(self,
                 config,
                 target_model,
                 adapter_weighted_method,
                 continual_sequence,
                 cur_dataset,
                 update_method,
                 adapter):
        super().__init__()
        self.config = config

        self.ortho_loss = 0.

        self.layer = nn.ModuleList([ViltLayer(config,target_model, adapter_weighted_method,continual_sequence,cur_dataset,update_method,adapter.adapter_infos) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if self.training:
                layer_module.ortho_loss = 0.
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if self.training:
                self.ortho_loss += layer_module.ortho_loss

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        
        

class ViltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self,
                 config,
                 target_model,
                 adapter_weighted_method,
                 continual_sequence,
                 cur_dataset,
                 update_method,
                 adapter_infos:Adapter_infos):
        super().__init__()
        self.config = config
        self.target_model = target_model
        self.adapter_weighted_method = adapter_weighted_method
        self.continual_sequence = continual_sequence
        self.cur_dataset = cur_dataset
        self.update_method = update_method
        self.adapter_infos = adapter_infos
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViltAttention(config)
        self.intermediate = ViltIntermediate(config)
        self.output = ViltOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ortho_loss = 0.


        if adapter_infos.key:
            if self.target_model == "snlive-update":
                if self.update_method == "task-incremental-update" or self.update_method == "task-incremental-generalize" or self.update_method == "upstream-generalize":
                    
                    if self.continual_sequence == "multi-first":
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                    elif self.continual_sequence == "mono-first":
                        # Train the coefficients of current task on the adapter of the pervious task before use.

                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))   
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))  

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                
                else:
                    self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
            elif self.target_model == "piqa-update":
                if self.perceiver.key:
                    self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    
                elif self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # Train the coefficients of current task on the adapter of the pervious task before use.

                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))   
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))  

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                    else:
                        # Train the coefficients of current task on the adapter of the pervious task before use.

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            
                        
                        if self.adapter_weighted_method == "adapter-dimension-weighted":

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))   
                        elif self.adapter_weighted_method == "adapter-whole-weighted":

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))  

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                else:
                    self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
            elif self.target_model == "iNaturalist-update":
                if self.perceiver.key:
                    self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    
                elif self.update_method == "task-incremental-update" or self.update_method == "task-incremental-generalize" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # Train the adapter for current task based on the adapter from the pervious tasks and its corresponding coefficients.
                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        
                        # iNaturalist specific parameters
                        self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        
                        if self.adapter_weighted_method == "adapter-dimension-weighted":

                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                    else:
                        # Train the adapter for current task based on the adapter from the pervious tasks and its corresponding coefficients.
                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # iNaturalist specific parameters
                        self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # VQA specific parameters
                        self.vqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.vqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                else:
                    self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
            elif self.target_model == "vqa-update":
                if self.perceiver.key:
                    self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.vqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.vqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                elif self.update_method == "task-incremental-update" or "task=incremental-generalize" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # Train the adapter for current task based on the adapter from the pervious tasks and its corresponding coefficients.
                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # iNaturalist specific parameters
                        self.iNaturalist_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.iNaturalist_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)


                        # VQA specific parameters
                        self.vqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.vqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.iNaturalist_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.iNaturalist_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.iNaturalist_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.iNaturalist_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                    else:
                        # Train the adapter for current task based on the adapter from the pervious tasks and its corresponding coefficients.
                        # snlive specific parameters
                        self.snlive_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.snlive_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # piqa specific parameters
                        self.piqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.piqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        # VQA specific parameters
                        self.vqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                        self.vqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                                adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                            self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                                    adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(24,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(24,config.hidden_size))
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            self.snlive_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.snlive_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.snlive_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.snlive_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                            
                            self.piqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.piqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.piqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.piqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            self.vqa_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                            self.vqa_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))

                            self.vqa_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                            self.vqa_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                                self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                                
                                self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                                self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                        else:
                            raise NotImplementedError
                else:
                    self.vqa_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.vqa_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
            elif self.target_model == "self-update":
                if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "adapter":
                    self.add_query_aftattn = nn.Parameter(torch.randn(1,config.hidden_size))
                    self.add_query_aftmlp = nn.Parameter(torch.randn(1,config.hidden_size))
                    
                    self.add_key_aftattn = nn.Parameter(torch.randn(config.hidden_size))
                    self.add_key_aftmlp = nn.Parameter(torch.randn(config.hidden_size))
                    self.add_adapter_aftattn = Adapter(embed_dim=config.hidden_size,
                                        adapter_embed_dim=self.adapter_infos.adapter_embed_dim)
                    self.add_adapter_aftmlp = Adapter(embed_dim=config.hidden_size,
                                            adapter_embed_dim=self.adapter_infos.adapter_embed_dim)

    def forward(self, hidden_states, 
                attention_mask=None, head_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViLT, layernorm is applied before self-attention
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # First adapter
        if self.adapter_infos.key:
            if self.target_model == "snlive-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output

                        # Cross-attention query, key, value
                        attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        snlive_attn_kv = torch.cat([attn_query,snlive_attn_proj_output],dim=1)
                        cross_attn_out = cross_attn_noproj(q=attn_query,k = snlive_attn_kv,v = snlive_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Compute Cosine similiarity
                            n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            norm_attn = nn.functional.normalize(cross_attn_out,dim=1)
                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_attn,n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2
                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w)

                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            # Compute Cosine similiarity
                            n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            norm_attn = nn.functional.normalize(cross_attn_out,dim=-1)
                            
                            snlive_adapter_w = torch.einsum('bd,d->b',norm_attn.squeeze(1),n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                # Weighted
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                # Weighted
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w)
                        attention_output = adapter_out
                    else:
                         # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))

                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w)    
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                # Weighted
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w)    

                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            # query ortho loss
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.add_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.add_key_aftattn.view(1,-1)],dim=0)) * 0.1
                            else:
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1)],dim=0)) * 0.1
                    
                else:
                    # individual adapter
                    attention_output = self.snlive_adapter_aftattn(attention_output)
            elif self.target_model == "piqa-update":

                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))

                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w)    
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w)    

                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.add_query_aftattn.view(1,-1)],dim=0)) * 1e-7 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.add_key_aftattn.view(1,-1)],dim=0)) * 1e-7
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1)],dim=0)) * 1e-7 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1)],dim=0)) * 1e-7
                    else:
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        # Cross-attention query, key, value
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))

                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w)    
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            # compute cosine similarity

                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w)    

                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.piqa_query_aftattn.view(1,-1), self.add_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.piqa_key_aftattn.view(1,-1),self.add_key_aftattn.view(1,-1)],dim=0)) * 0.1

                else:
                    # adapter
                    attention_output = self.piqa_adapter_aftattn(attention_output)
            elif self.target_model == "iNaturalist-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        #iNaturalist attn adapter and projection
                        iNaturalist_attn_adapter_output = self.iNaturalist_adapter_aftattn(attention_output)
                        iNaturalist_attn_proj_output = iNaturalist_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))
                        iNaturalist_attn_query = self.iNaturalist_query_aftattn.expand(iNaturalist_attn_proj_output.size(0),self.iNaturalist_query_aftattn.size(0),self.iNaturalist_query_aftattn.size(1))
                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        iNaturalist_attn_kv = torch.cat([iNaturalist_attn_query,iNaturalist_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)
                        iNaturalist_cross_attn_out = cross_attn_noproj(q=iNaturalist_attn_query,k=iNaturalist_attn_kv,v=iNaturalist_attn_kv,num_heads=self.config.num_attention_heads)


                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=0)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bld,ld->bd',norm_iNaturalist_attn,iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,bd->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w)    
                        elif self.adapter_weighted_method == "adapter-whole-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=-1)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bd,d->b',norm_iNaturalist_attn.squeeze(1),iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w)  
                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1),self.add_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1),self.add_key_aftattn.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1)],dim=0)) * 0.1
                    else:
                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        #iNaturalist attn adapter and projection
                        iNaturalist_attn_adapter_output = self.iNaturalist_adapter_aftattn(attention_output)
                        iNaturalist_attn_proj_output = iNaturalist_attn_adapter_output

                        # vqa attn adapter and projection
                        vqa_attn_adapter_output = self.vqa_adapter_aftattn(attention_output)
                        vqa_attn_proj_output = vqa_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))
                        iNaturalist_attn_query = self.iNaturalist_query_aftattn.expand(iNaturalist_attn_proj_output.size(0),self.iNaturalist_query_aftattn.size(0),self.iNaturalist_query_aftattn.size(1))
                        vqa_attn_query = self.vqa_query_aftattn.expand(vqa_attn_proj_output.size(0),self.vqa_query_aftattn.size(0),self.vqa_query_aftattn.size(1))
                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        iNaturalist_attn_kv = torch.cat([iNaturalist_attn_query,iNaturalist_attn_proj_output],dim=1)
                        vqa_attn_kv = torch.cat([vqa_attn_query,vqa_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)
                        iNaturalist_cross_attn_out = cross_attn_noproj(q=iNaturalist_attn_query,k=iNaturalist_attn_kv,v=iNaturalist_attn_kv,num_heads=self.config.num_attention_heads)
                        vqa_cross_attn_out = cross_attn_noproj(q=vqa_attn_query,k=vqa_attn_kv,v=vqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=0)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=0)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bld,ld->bd',norm_iNaturalist_attn,iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bld,ld->bd',norm_vqa_attn,vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                        
                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,bd->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,bd->bld',vqa_attn_proj_output,vqa_adapter_w)  
                        elif self.adapter_weighted_method == "adapter-whole-weighted":   
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=-1)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=-1)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=-1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bd,d->b',norm_iNaturalist_attn.squeeze(1),iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bd,d->b',norm_vqa_attn.squeeze(1),vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w)        
                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1), self.add_query_aftattn.view(1,-1)],dim=0)) * 1e-7 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1), self.add_key_aftattn.view(1,-1)],dim=0)) * 1e-7
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1)],dim=0)) * 1e-7 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1)],dim=0)) * 1e-7


                else:
                    # adapter
                    attention_output = self.iNaturalist_adapter_aftattn(attention_output)
            elif self.target_model == "vqa-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":

                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        #iNaturalist attn adapter and projection
                        iNaturalist_attn_adapter_output = self.iNaturalist_adapter_aftattn(attention_output)
                        iNaturalist_attn_proj_output = iNaturalist_attn_adapter_output

                        # vqa attn adapter and projection
                        vqa_attn_adapter_output = self.vqa_adapter_aftattn(attention_output)
                        vqa_attn_proj_output = vqa_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))
                        iNaturalist_attn_query = self.iNaturalist_query_aftattn.expand(iNaturalist_attn_proj_output.size(0),self.iNaturalist_query_aftattn.size(0),self.iNaturalist_query_aftattn.size(1))
                        vqa_attn_query = self.vqa_query_aftattn.expand(vqa_attn_proj_output.size(0),self.vqa_query_aftattn.size(0),self.vqa_query_aftattn.size(1))
                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        iNaturalist_attn_kv = torch.cat([iNaturalist_attn_query,iNaturalist_attn_proj_output],dim=1)
                        vqa_attn_kv = torch.cat([vqa_attn_query,vqa_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)
                        iNaturalist_cross_attn_out = cross_attn_noproj(q=iNaturalist_attn_query,k=iNaturalist_attn_kv,v=iNaturalist_attn_kv,num_heads=self.config.num_attention_heads)
                        vqa_cross_attn_out = cross_attn_noproj(q=vqa_attn_query,k=vqa_attn_kv,v=vqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=0)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=0)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bld,ld->bd',norm_iNaturalist_attn,iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bld,ld->bd',norm_vqa_attn,vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                        
                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,bd->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,bd->bld',vqa_attn_proj_output,vqa_adapter_w)  
                        elif self.adapter_weighted_method == "adapter-whole-weighted":   
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            iNaturalist_n_K = nn.functional.normalize(self.iNaturalist_key_aftattn,dim=-1)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=-1)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)
                            norm_iNaturalist_attn = nn.functional.normalize(iNaturalist_cross_attn_out,dim=-1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            iNaturalist_adapter_w = torch.einsum('bd,d->b',norm_iNaturalist_attn.squeeze(1),iNaturalist_n_K)
                            iNaturalist_adapter_w = (iNaturalist_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bd,d->b',norm_vqa_attn.squeeze(1),vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                # Weighted
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',iNaturalist_attn_proj_output,iNaturalist_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w)        
                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":

                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1), self.add_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1), self.add_key_aftattn.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1),self.iNaturalist_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1),self.iNaturalist_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1)],dim=0)) * 0.1
                    else:
                        # snlive attn adapter and projection
                        snlive_attn_adapter_output = self.snlive_adapter_aftattn(attention_output)
                        snlive_attn_proj_output = snlive_attn_adapter_output
                        # piqa attn adapter and projection
                        piqa_attn_adapter_output = self.piqa_adapter_aftattn(attention_output)
                        piqa_attn_proj_output = piqa_attn_adapter_output

                        # vqa attn adapter and projection
                        vqa_attn_adapter_output = self.vqa_adapter_aftattn(attention_output)
                        vqa_attn_proj_output = vqa_attn_adapter_output

                        # Cross-attention query, key, value
                        snlive_attn_query = self.snlive_query_aftattn.expand(snlive_attn_proj_output.size(0),self.snlive_query_aftattn.size(0),self.snlive_query_aftattn.size(1))
                        piqa_attn_query = self.piqa_query_aftattn.expand(piqa_attn_proj_output.size(0),self.piqa_query_aftattn.size(0),self.piqa_query_aftattn.size(1))
                        vqa_attn_query = self.vqa_query_aftattn.expand(vqa_attn_proj_output.size(0),self.vqa_query_aftattn.size(0),self.vqa_query_aftattn.size(1))
                        snlive_attn_kv = torch.cat([snlive_attn_query,snlive_attn_proj_output],dim=1)
                        piqa_attn_kv = torch.cat([piqa_attn_query,piqa_attn_proj_output],dim=1)
                        vqa_attn_kv = torch.cat([vqa_attn_query,vqa_attn_proj_output],dim=1)
                        snlive_cross_attn_out = cross_attn_noproj(q=snlive_attn_query,k=snlive_attn_kv,v=snlive_attn_kv,num_heads=self.config.num_attention_heads)
                        piqa_cross_attn_out = cross_attn_noproj(q=piqa_attn_query,k=piqa_attn_kv,v=piqa_attn_kv,num_heads=self.config.num_attention_heads)
                        vqa_cross_attn_out = cross_attn_noproj(q=vqa_attn_query,k=vqa_attn_kv,v=vqa_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                            add_attn_proj_output = add_attn_adapter_output
                            add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                            add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                            add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=0)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=0)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=0)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=1)

                            snlive_adapter_w = torch.einsum('bld,ld->bd',norm_snlive_attn,snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bld,ld->bd',norm_piqa_attn,piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bld,ld->bd',norm_vqa_attn,vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                        
                            # Weighted
                            adapter_out = torch.einsum('bld,bd->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,bd->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,bd->bld',vqa_attn_proj_output,vqa_adapter_w)  
                        elif self.adapter_weighted_method == "adapter-whole-weighted":   
                            # compute cosine similarity
                            snlive_n_K = nn.functional.normalize(self.snlive_key_aftattn,dim=-1)
                            piqa_n_K = nn.functional.normalize(self.piqa_key_aftattn,dim=-1)
                            vqa_n_K = nn.functional.normalize(self.vqa_key_aftattn,dim=-1)

                            norm_snlive_attn = nn.functional.normalize(snlive_cross_attn_out,dim=-1)
                            norm_piqa_attn = nn.functional.normalize(piqa_cross_attn_out,dim=-1)
                            norm_vqa_attn = nn.functional.normalize(vqa_cross_attn_out,dim=-1)

                            snlive_adapter_w = torch.einsum('bd,d->b',norm_snlive_attn.squeeze(1),snlive_n_K)
                            snlive_adapter_w = (snlive_adapter_w + 1) / 2

                            piqa_adapter_w = torch.einsum('bd,d->b',norm_piqa_attn.squeeze(1),piqa_n_K)
                            piqa_adapter_w = (piqa_adapter_w + 1) / 2

                            vqa_adapter_w = torch.einsum('bd,d->b',norm_vqa_attn.squeeze(1),vqa_n_K)
                            vqa_adapter_w = (vqa_adapter_w + 1) / 2                    
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                                add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                                
                                add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                                add_adapter_w = (add_adapter_w + 1) / 2
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w) + torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                            else:
                                adapter_out = torch.einsum('bld,b->bld',snlive_attn_proj_output,snlive_adapter_w) + torch.einsum('bld,b->bld',piqa_attn_proj_output,piqa_adapter_w) + torch.einsum('bld,b->bld',vqa_attn_proj_output,vqa_adapter_w)        
                        attention_output = adapter_out

                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1), self.add_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1), self.add_key_aftattn.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftattn.view(1,-1),self.piqa_query_aftattn.view(1,-1), self.vqa_query_aftattn.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftattn.view(1,-1),self.piqa_key_aftattn.view(1,-1), self.vqa_key_aftattn.view(1,-1)],dim=0)) * 0.1
                else:
                    attention_output = self.vqa_adapter_aftattn(attention_output)
            elif self.target_model == "self-update":
                # adapter
                if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "adapter":
                    add_attn_adapter_output = self.add_adapter_aftattn(attention_output)
                    add_attn_proj_output = add_attn_adapter_output
                    add_attn_query = self.add_query_aftattn.expand(add_attn_proj_output.size(0),self.add_query_aftattn.size(0),self.add_query_aftattn.size(1))
                    add_attn_kv = torch.cat([add_attn_query,add_attn_proj_output],dim=1)
                    add_cross_attn_out = cross_attn_noproj(q=add_attn_query,k = add_attn_kv,v = add_attn_kv,num_heads=self.config.num_attention_heads)
                    add_n_K = nn.functional.normalize(self.add_key_aftattn,dim=-1)
                    add_norm_attn = nn.functional.normalize(add_cross_attn_out,dim=-1)
                    
                    add_adapter_w = torch.einsum('bd,d->b',add_norm_attn.squeeze(1),add_n_K)
                    add_adapter_w = (add_adapter_w + 1) / 2
                    # Weighted
                    adapter_out = torch.einsum('bld,b->bld',add_attn_proj_output,add_adapter_w)
                    attention_output = adapter_out
                    


        # first residual connection
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # in ViLT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)
        # Second_adapter
        if self.adapter_infos.key:
            if self.target_model == "snlive-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2
                            # Weighted
                            snlive_mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w)
                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                # Weighted
                                snlive_mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                # Weighted
                                snlive_mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w)
                        layer_output = snlive_mlp_adapter_out
                    else:
                        # mono-first
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output

                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        layer_output = mlp_adapter_out
                        
                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                    
                else:
                    layer_output = self.snlive_adapter_aftmlp(layer_output)
            elif self.target_model == "piqa-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output

                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                # Weighted
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        layer_output = mlp_adapter_out
                        
                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                    else:
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output

                        # Cross-attention, query, key,value

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w)
                        layer_output = mlp_adapter_out
                        
                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.piqa_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.piqa_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1

                else:
                    layer_output = self.piqa_adapter_aftmlp(layer_output)
            
            elif self.target_model == "iNaturalist-update":
                if self.update_method == "task-incremental-generalize" or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence == "multi-first":
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output
                        
                        # iNaturalist adapter,projection
                        iNaturalist_mlp_adapter_output = self.iNaturalist_adapter_aftmlp(layer_output)
                        iNaturalist_mlp_proj_output = iNaturalist_mlp_adapter_output

                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        iNaturalist_mlp_query = self.iNaturalist_query_aftmlp.expand(iNaturalist_mlp_proj_output.size(0),self.iNaturalist_query_aftmlp.size(0),self.iNaturalist_query_aftmlp.size(1))
                        iNaturalist_mlp_kv = torch.cat([iNaturalist_mlp_query,iNaturalist_mlp_proj_output],dim=1)
                        iNaturalist_cross_mlp_out = cross_attn_noproj(q=iNaturalist_mlp_query,k=iNaturalist_mlp_kv,v=iNaturalist_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=0)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bld,ld->bd',iNaturalist_norm_cross_mlp,iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2
                            
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,bd->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w)
                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=-1)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=-1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bd,d->b',iNaturalist_norm_cross_mlp.squeeze(1),iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2
                            
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w)
                        layer_output = mlp_adapter_out
                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                    else:
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output
                        
                        # iNaturalist adapter,projection
                        iNaturalist_mlp_adapter_output = self.iNaturalist_adapter_aftmlp(layer_output)
                        iNaturalist_mlp_proj_output = iNaturalist_mlp_adapter_output

                        # Vqa adapter, projection
                        vqa_mlp_adapter_output = self.vqa_adapter_aftmlp(layer_output)
                        vqa_mlp_proj_output = vqa_mlp_adapter_output


                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        iNaturalist_mlp_query = self.iNaturalist_query_aftmlp.expand(iNaturalist_mlp_proj_output.size(0),self.iNaturalist_query_aftmlp.size(0),self.iNaturalist_query_aftmlp.size(1))
                        iNaturalist_mlp_kv = torch.cat([iNaturalist_mlp_query,iNaturalist_mlp_proj_output],dim=1)
                        iNaturalist_cross_mlp_out = cross_attn_noproj(q=iNaturalist_mlp_query,k=iNaturalist_mlp_kv,v=iNaturalist_mlp_kv,num_heads=self.config.num_attention_heads)

                        vqa_mlp_query = self.vqa_query_aftmlp.expand(vqa_mlp_proj_output.size(0),self.vqa_query_aftmlp.size(0),self.vqa_query_aftmlp.size(1))
                        vqa_mlp_kv = torch.cat([vqa_mlp_query,vqa_mlp_proj_output],dim=1)
                        vqa_cross_mlp_out = cross_attn_noproj(q=vqa_mlp_query,k=vqa_mlp_kv,v=vqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=0)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bld,ld->bd',iNaturalist_norm_cross_mlp,iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=0)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=1)
                            vqa_adapter_mlp_w = torch.einsum('bld,ld->bd',vqa_norm_cross_mlp,vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,bd->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,bd->bld',iNaturalist_mlp_proj_output,vqa_adapter_mlp_w)

                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=-1)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=-1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bd,d->b',iNaturalist_norm_cross_mlp.squeeze(1),iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=-1)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=-1)
                            vqa_adapter_mlp_w = torch.einsum('bd,d->b',vqa_norm_cross_mlp.squeeze(1),vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w)

                        layer_output = mlp_adapter_out
                        # Orthogonal loss
                        if self.training:
                            # query ortho loss
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                
                else:
                    layer_output = self.iNaturalist_adapter_aftmlp(layer_output)

            elif self.target_model == "vqa-update":
                if self.update_method == "task-incremental-generalize"  or self.update_method == "task-incremental-update" or self.update_method == "upstream-generalize":
                    if self.continual_sequence=="multi-first":
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output
                        
                        # iNaturalist adapter,projection
                        iNaturalist_mlp_adapter_output = self.iNaturalist_adapter_aftmlp(layer_output)
                        iNaturalist_mlp_proj_output = iNaturalist_mlp_adapter_output

                        # Vqa adapter, projection
                        vqa_mlp_adapter_output = self.vqa_adapter_aftmlp(layer_output)
                        vqa_mlp_proj_output = vqa_mlp_adapter_output


                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        iNaturalist_mlp_query = self.iNaturalist_query_aftmlp.expand(iNaturalist_mlp_proj_output.size(0),self.iNaturalist_query_aftmlp.size(0),self.iNaturalist_query_aftmlp.size(1))
                        iNaturalist_mlp_kv = torch.cat([iNaturalist_mlp_query,iNaturalist_mlp_proj_output],dim=1)
                        iNaturalist_cross_mlp_out = cross_attn_noproj(q=iNaturalist_mlp_query,k=iNaturalist_mlp_kv,v=iNaturalist_mlp_kv,num_heads=self.config.num_attention_heads)

                        vqa_mlp_query = self.vqa_query_aftmlp.expand(vqa_mlp_proj_output.size(0),self.vqa_query_aftmlp.size(0),self.vqa_query_aftmlp.size(1))
                        vqa_mlp_kv = torch.cat([vqa_mlp_query,vqa_mlp_proj_output],dim=1)
                        vqa_cross_mlp_out = cross_attn_noproj(q=vqa_mlp_query,k=vqa_mlp_kv,v=vqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=0)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bld,ld->bd',iNaturalist_norm_cross_mlp,iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=0)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=1)
                            vqa_adapter_mlp_w = torch.einsum('bld,ld->bd',vqa_norm_cross_mlp,vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,bd->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,bd->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w)

                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2
                            
                            iNaturalist_mlp_n_K = nn.functional.normalize(self.iNaturalist_key_aftmlp,dim=-1)
                            iNaturalist_norm_cross_mlp = nn.functional.normalize(iNaturalist_cross_mlp_out,dim=-1)
                            iNaturalist_adapter_mlp_w = torch.einsum('bd,d->b',iNaturalist_norm_cross_mlp.squeeze(1),iNaturalist_mlp_n_K)
                            iNaturalist_adapter_mlp_w = (iNaturalist_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=-1)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=-1)
                            vqa_adapter_mlp_w = torch.einsum('bd,d->b',vqa_norm_cross_mlp.squeeze(1),vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',iNaturalist_mlp_proj_output,iNaturalist_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w)

                        layer_output = mlp_adapter_out
                        # Orthogonal loss
                        if self.training:
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1),self.add_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1),self.add_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                    # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.iNaturalist_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.iNaturalist_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                    else:
                        # Snlive adapter, projection
                        snlive_mlp_adapter_output = self.snlive_adapter_aftmlp(layer_output)
                        snlive_mlp_proj_output = snlive_mlp_adapter_output
                        # Piqa adapter,projection
                        piqa_mlp_adapter_output = self.piqa_adapter_aftmlp(layer_output)
                        piqa_mlp_proj_output = piqa_mlp_adapter_output

                        # Vqa adapter, projection
                        vqa_mlp_adapter_output = self.vqa_adapter_aftmlp(layer_output)
                        vqa_mlp_proj_output = vqa_mlp_adapter_output


                        # Cross-attention, query, key,value
                        snlive_mlp_query = self.snlive_query_aftmlp.expand(snlive_mlp_proj_output.size(0),self.snlive_query_aftmlp.size(0),self.snlive_query_aftmlp.size(1))
                        snlive_mlp_kv = torch.cat([snlive_mlp_query,snlive_mlp_proj_output],dim=1)
                        snlive_cross_mlp_out = cross_attn_noproj(q=snlive_mlp_query,k=snlive_mlp_kv,v=snlive_mlp_kv,num_heads=self.config.num_attention_heads)

                        piqa_mlp_query = self.piqa_query_aftmlp.expand(piqa_mlp_proj_output.size(0),self.piqa_query_aftmlp.size(0),self.piqa_query_aftmlp.size(1))
                        piqa_mlp_kv = torch.cat([piqa_mlp_query,piqa_mlp_proj_output],dim=1)
                        piqa_cross_mlp_out = cross_attn_noproj(q=piqa_mlp_query,k=piqa_mlp_kv,v=piqa_mlp_kv,num_heads=self.config.num_attention_heads)

                        vqa_mlp_query = self.vqa_query_aftmlp.expand(vqa_mlp_proj_output.size(0),self.vqa_query_aftmlp.size(0),self.vqa_query_aftmlp.size(1))
                        vqa_mlp_kv = torch.cat([vqa_mlp_query,vqa_mlp_proj_output],dim=1)
                        vqa_cross_mlp_out = cross_attn_noproj(q=vqa_mlp_query,k=vqa_mlp_kv,v=vqa_mlp_kv,num_heads=self.config.num_attention_heads)
                        
                        if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                            add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                            add_mlp_proj_output = add_mlp_adapter_output
                            add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                            add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                            add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)

                        if self.adapter_weighted_method == "adapter-dimension-weighted":
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=0)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=1)
                            snlive_adapter_mlp_w = torch.einsum('bld,ld->bd',snlive_norm_cross_mlp,snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=0)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=1)
                            piqa_adapter_mlp_w = torch.einsum('bld,ld->bd',piqa_norm_cross_mlp,piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=0)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=1)
                            vqa_adapter_mlp_w = torch.einsum('bld,ld->bd',vqa_norm_cross_mlp,vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            # Weighted
                            mlp_adapter_out = torch.einsum('bld,bd->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,bd->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,bd->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w)

                        elif self.adapter_weighted_method == "adapter-whole-weighted": 
                            # Cosine Similiarity
                            snlive_mlp_n_K = nn.functional.normalize(self.snlive_key_aftmlp,dim=-1)
                            snlive_norm_cross_mlp = nn.functional.normalize(snlive_cross_mlp_out,dim=-1)
                            snlive_adapter_mlp_w = torch.einsum('bd,d->b',snlive_norm_cross_mlp.squeeze(1),snlive_mlp_n_K)
                            snlive_adapter_mlp_w = (snlive_adapter_mlp_w + 1) / 2

                            piqa_mlp_n_K = nn.functional.normalize(self.piqa_key_aftmlp,dim=-1)
                            piqa_norm_cross_mlp = nn.functional.normalize(piqa_cross_mlp_out,dim=-1)
                            piqa_adapter_mlp_w = torch.einsum('bd,d->b',piqa_norm_cross_mlp.squeeze(1),piqa_mlp_n_K)
                            piqa_adapter_mlp_w = (piqa_adapter_mlp_w + 1) / 2

                            vqa_mlp_n_K = nn.functional.normalize(self.vqa_key_aftmlp,dim=-1)
                            vqa_norm_cross_mlp = nn.functional.normalize(vqa_cross_mlp_out,dim=-1)
                            vqa_adapter_mlp_w = torch.einsum('bd,d->b',vqa_norm_cross_mlp.squeeze(1),vqa_mlp_n_K)
                            vqa_adapter_mlp_w = (vqa_adapter_mlp_w + 1) / 2
                            
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                                add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                                add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                                add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w) + torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                            else:
                                mlp_adapter_out = torch.einsum('bld,b->bld',snlive_mlp_proj_output,snlive_adapter_mlp_w) + torch.einsum('bld,b->bld',piqa_mlp_proj_output,piqa_adapter_mlp_w) + torch.einsum('bld,b->bld',vqa_mlp_proj_output,vqa_adapter_mlp_w)

                        layer_output = mlp_adapter_out
                        # Orthogonal loss
                        if self.training:
                            # query ortho loss
                            if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "task-incremental-update":
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                # key ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                            else:
                                # query ortho loss
                                self.ortho_loss += ortho_penalty(torch.cat([self.snlive_query_aftmlp.view(1,-1),self.piqa_query_aftmlp.view(1,-1),self.vqa_query_aftmlp.view(1,-1)],dim=0)) * 0.1 # orthogonal coefficient
                                if self.update_method == "task-incremental-update":
                                    # key ortho loss
                                    self.ortho_loss += ortho_penalty(torch.cat([self.snlive_key_aftmlp.view(1,-1),self.piqa_key_aftmlp.view(1,-1),self.vqa_key_aftmlp.view(1,-1)],dim=0)) * 0.1
                else:
                    layer_output = self.vqa_adapter_aftmlp(layer_output)
            elif self.target_model == "self-update":
                if self.cur_dataset in ["vcr","commonsenseqa","places365","nlvr2"] and self.update_method == "adapter":
                    add_mlp_adapter_output = self.add_adapter_aftmlp(layer_output)
                    add_mlp_proj_output = add_mlp_adapter_output
                    add_mlp_query = self.add_query_aftmlp.expand(add_mlp_proj_output.size(0),self.add_query_aftmlp.size(0),self.add_query_aftmlp.size(1))
                    add_mlp_kv = torch.cat([add_mlp_query,add_mlp_proj_output],dim=1)
                    add_cross_mlp_out = cross_attn_noproj(q=add_mlp_query,k=add_mlp_kv,v=add_mlp_kv,num_heads=self.config.num_attention_heads)
                    add_mlp_n_K = nn.functional.normalize(self.add_key_aftmlp,dim=-1)
                    add_norm_cross_mlp = nn.functional.normalize(add_cross_mlp_out,dim=-1)
                    add_adapter_mlp_w = torch.einsum('bd,d->b',add_norm_cross_mlp.squeeze(1),add_mlp_n_K)
                    add_adapter_mlp_w = (add_adapter_mlp_w + 1) / 2
                    # Weighted
                    snlive_mlp_adapter_out = torch.einsum('bld,b->bld',add_mlp_proj_output,add_adapter_mlp_w)
                    layer_output = self.add_adapter_aftmlp(layer_output)
            
            
        layer_output = layer_output + hidden_states
        outputs = (layer_output,) + outputs 

        return outputs


def ortho_penalty(t):
    multi = t @ t.T
    output = ((multi - torch.eye(t.shape[0]).to(t.device))**2).mean() / (t.size(1) * t.size(0))
    return output

class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # text embeddings
        self.text_embeddings = TextEmbeddings(config)
        # patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViltPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type (text/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def visual_embed(self, pixel_values, pixel_mask, max_image_length=200):
        _, _, ph, pw = self.patch_embeddings.projection.weight.shape
        x = self.patch_embeddings(pixel_values)
        x_mask = pixel_mask[:, None, :, :].float()
        x_mask = nn.functional.interpolate(x_mask, size=(x.shape[2], x.shape[3])).long()
        x_h = x_mask[:, 0].sum(dim=1)[:, 0]
        x_w = x_mask[:, 0].sum(dim=2)[:, 0]

        batch_size, num_channels, height, width = x.shape
        patch_dim = self.config.image_size // self.config.patch_size
        spatial_pos = self.position_embeddings[:, 1:, :].transpose(1, 2).view(1, num_channels, patch_dim, patch_dim)
        pos_embed = torch.cat(
            [
                nn.functional.pad(
                    nn.functional.interpolate(
                        spatial_pos,
                        size=(h, w),
                        mode="bilinear",
                        align_corners=True,
                    ),
                    (0, width - w, 0, height - h),
                )
                for h, w in zip(x_h, x_w)
            ],
            dim=0,
        )

        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        x = x.flatten(2).transpose(1, 2)
        # Set `device` here, otherwise `patch_index` will always be on `CPU` and will fail near the end for torch>=1.13
        patch_index = torch.stack(
            meshgrid(torch.arange(x_mask.shape[-2]), torch.arange(x_mask.shape[-1]), indexing="ij"), dim=-1
        ).to(device=x_mask.device)
        patch_index = patch_index[None, None, :, :, :]
        patch_index = patch_index.expand(x_mask.shape[0], x_mask.shape[1], -1, -1, -1)
        patch_index = patch_index.flatten(1, 3)
        x_mask = x_mask.flatten(1)

        if max_image_length < 0 or max_image_length is None or not isinstance(max_image_length, int):
            # suppose aug is 800 x 1333, then, maximum effective res is 800 x 1333 (if one side gets bigger, the other will be constrained and be shrinked)
            # (800 // self.patch_size) * (1333 // self.patch_size) is the maximum number of patches that single image can get.
            # if self.patch_size = 32, 25 * 41 = 1025
            # if res is 384 x 640, 12 * 20 = 240
            effective_resolution = x_h * x_w
            max_image_length = effective_resolution.max()
        else:
            effective_resolution = x_h * x_w
            max_image_length = min(effective_resolution.max(), max_image_length)

        valid_idx = x_mask.nonzero(as_tuple=False)
        non_valid_idx = (1 - x_mask).nonzero(as_tuple=False)
        unique_rows = valid_idx[:, 0].unique()
        valid_row_idx = [valid_idx[valid_idx[:, 0] == u] for u in unique_rows]
        non_valid_row_idx = [non_valid_idx[non_valid_idx[:, 0] == u] for u in unique_rows]

        valid_nums = [v.size(0) for v in valid_row_idx]
        non_valid_nums = [v.size(0) for v in non_valid_row_idx]
        pad_nums = [max_image_length - v for v in valid_nums]

        select = []
        for i, (v, nv, p) in enumerate(zip(valid_nums, non_valid_nums, pad_nums)):
            if p <= 0:
                valid_choice = torch.multinomial(torch.ones(v).float(), max_image_length)
                select.append(valid_row_idx[i][valid_choice])
            else:
                pad_choice = torch.multinomial(torch.ones(nv).float(), p, replacement=True)
                select.append(torch.cat([valid_row_idx[i], non_valid_row_idx[i][pad_choice]], dim=0))

        select = torch.cat(select, dim=0)
        x = x[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)
        x_mask = x_mask[select[:, 0], select[:, 1]].view(batch_size, -1)
        # `patch_index` should be on the same device as `select` (for torch>=1.13), which is ensured at definition time.
        patch_index = patch_index[select[:, 0], select[:, 1]].view(batch_size, -1, 2)
        pos_embed = pos_embed[select[:, 0], select[:, 1]].view(batch_size, -1, num_channels)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = torch.cat(
            (self.position_embeddings[:, 0, :][:, None, :].expand(batch_size, -1, -1), pos_embed), dim=1
        )
        x = x + pos_embed
        x = self.dropout(x)

        x_mask = torch.cat([torch.ones(x_mask.shape[0], 1).to(x_mask), x_mask], dim=1)

        return x, x_mask, (patch_index, (height, width))

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
    ):
        
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # PART 2: patch embeddings (with interpolated position encodings)
        if image_embeds is None:
            image_embeds, image_masks, patch_index = self.visual_embed(
                pixel_values, pixel_mask, max_image_length=self.config.max_image_length
            )
        else:
            image_masks = pixel_mask.flatten(1)

        # PART 3: add modality type embeddings
        # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device)
        )

        # PART 4: concatenate
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        masks = torch.cat([attention_mask, image_masks], dim=1)

        return embeddings, masks
    

class ViltPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class ViltOutput(nn.Module):
    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states