import torch
from torch import nn
from typing import Any

from nets.modules.net_utils import input_embed
from nets.modules.encoder import MHAEncoder


class Critic(nn.Module):
    """Critic network"""

    def __init__(
            self,
            input_dim1: int,
            input_dim2: int,
            embed_dim: int,
            num_heads: int,
            num_blocks: int,
            normalization: str,
            combined_mha=True,
            *args, **kwargs) -> None:
        """
        Initializes the Critic network.

        Args:
            input_dim1 (torch.Tensor): The input dimension for input 1.
            input_dim2 (torch.Tensor): The input dimension for input 2.
            embed_dim (int): The embedding dimension.
            num_heads (int): Number of heads for MHA layers.
            num_blocks (int): The number of blocks in the encoder.
            normalization (str): The normalization type.
            combined_mha (bool): Whether to use combined MHA or not.
        """
        super(Critic, self).__init__()

        # Combined MHA encoder
        self.combined_mha = combined_mha

        # Initial embedding
        self.init_embed = nn.Linear(input_dim1, embed_dim)
        self.embed_depot = nn.Linear(2, embed_dim)

        # MHA Encoder
        self.encoder = MHAEncoder(
            embed_dim=embed_dim,
            node_dim2=input_dim2,
            num_heads=num_heads,
            num_blocks=num_blocks,
            normalization=normalization,
            combined=combined_mha
        )

        # Critic value prediction
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, batch: dict | torch.Tensor, env: Any) -> torch.Tensor:
        """
        Forward pass of the Critic module.

        Args:
            batch (torch.Tensor or dict): The batch data.
            env (Any): The environment.

        Returns:
            torch.Tensor: The critic value predicted by the Critic network.
        """
        # Initialize state
        state = env.get_state(batch)
        del batch

        # Initial embedding
        x = input_embed(state, self.init_embed, self.embed_depot)
        x = (x, state.obs) if self.combined_mha else (x, )

        # Graph embedding (encoder)
        graph_embeddings = self.encoder(*x)

        # Critic value prediction
        return self.value_head(graph_embeddings[1])[:, 0]
