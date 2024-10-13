
import torch
import torch.nn.functional as F
from typing import Callable
from torch import nn,Tensor

class Adapter(nn.Module):
    """A common Adapter module.

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 embed_dim:int,
                 adapter_embed_dim:int,
                 activation:Callable[...,nn.Module] = nn.Sigmoid):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_embed_dim = adapter_embed_dim
        self.activation = activation
        
        self.downsample_layer = nn.Linear(embed_dim,
                                          adapter_embed_dim)
        self.act = activation()
        
        self.upsample_layer = nn.Linear(adapter_embed_dim,
                                        embed_dim)
        
    def forward(self,input:Tensor) -> Tensor:
        x = self.downsample_layer(input)
        x = self.act(x) * x
        x = self.upsample_layer(x)
        output = x + input
        return output

