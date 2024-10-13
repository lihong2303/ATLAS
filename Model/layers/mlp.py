
import torch
from typing import Optional, Union,List,Callable
from torch import nn,Tensor

class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.

    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if normalization:
                layers.append(normalization(hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)