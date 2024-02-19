import math
import torch
import numpy as np
from torch import nn
from typing import Tuple


class SkipConnection(nn.Module):
    """Skip connection module"""

    def __init__(self, module: nn.Module) -> None:
        """
        Initializes the SkipConnection module.

        Args:
            module (nn.Module): The module to apply the skip connection to.
        """
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input1: torch.Tensor, input2: torch.Tensor = None) -> \
            torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SkipConnection module.

        Args:
            input1 (torch.Tensor): The input tensor.
            input2 (torch.Tensor): Optional second input tensor.

        Returns:
            torch.Tensor | tuple: The output tensor(s).
        """
        if input2 is None:
            return input1 + self.module(input1)
        else:
            x1, x2 = self.module(input1, input2)
            return input1 + x1, input2 + x2


class Normalization(nn.Module):
    """Normalization module"""

    def __init__(self, embed_dim: int, normalization: str = 'batch') -> None:
        """
        Initializes the Normalization module.

        Args:
            embed_dim (int): The dimension of the embedding.
            normalization (str): The normalization type ('batch' or 'instance').
        """
        super(Normalization, self).__init__()

        # Get normalization type
        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        # Get normalization layer
        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the parameters of the normalization layer."""
        for name, param in self.named_parameters():
            std = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-std, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Normalization module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class MHA(nn.Module):
    """
    Multi-Head Attention module.
    """

    def __init__(self, num_heads: int, input_dim: int, embed_dim: int, val_dim: int = None, key_dim: int = None) -> \
            None:
        """
        Initializes the MHA module.

        Args:
            num_heads (int): The number of attention heads.
            input_dim (int): The input dimension.
            embed_dim (int): The embedding dimension.
            val_dim (int): The value dimension.
            key_dim (int): The key dimension.
        """
        super(MHA, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // num_heads
        if key_dim is None:
            key_dim = val_dim

        # Dimensions
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # Normalization factor
        self.norm_factor = 1 / math.sqrt(key_dim)

        # Query, Key, Value linear projections
        self.W_query = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(num_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(num_heads, input_dim, val_dim))

        # Output linear projection
        self.W_out = nn.Parameter(torch.Tensor(num_heads, val_dim, embed_dim))

        # Normalize params
        self.init_parameters()

    def init_parameters(self) -> None:
        """Initializes the parameters of the MHA module."""
        for param in self.parameters():
            std = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-std, std)

    def forward(self, q: torch.Tensor, h: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass of the MHA module.

        Args:
            q (torch.Tensor): The query tensor. Shape: (batch_size, num_query, input_dim)
            h (torch.Tensor): The input tensor (optional). Shape: (batch_size, num_nodes, input_dim)
            mask (torch.Tensor): The mask tensor (optional). Mask should contain 1 if attention is not possible (i.e.
                                 mask is negative adjacency). Shape: (batch_size, num_query, graph_size)

        Returns:
            torch.Tensor: The output tensor.
        """

        # If h != q, compute self-attention. Else compute (crossed-)attention
        if h is None:
            h = q

        # h: (batch_size, num_nodes, input_dim)
        batch_size, num_nodes, input_dim = h.size()
        num_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        # Flatten
        h_flat = h.contiguous().view(-1, input_dim)
        q_flat = q.contiguous().view(-1, input_dim)

        # Last dimension can be different for keys and values
        shp = (self.num_heads, batch_size, num_nodes, -1)
        shp_q = (self.num_heads, batch_size, num_query, -1)

        # Calculate queries: (num_heads, num_query, num_nodes, key/val_size)
        query = torch.matmul(q_flat, self.W_query).view(shp_q)

        # Calculate keys and values: (n_heads, batch_size, graph_size, key/val_size)
        key = torch.matmul(h_flat, self.W_key).view(shp)
        value = torch.matmul(h_flat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(query, key.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, num_query, num_nodes).expand_as(compatibility)
            compatibility[mask] = -np.inf

        # Normalize logits with softmax
        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan, so we fix them to 0
        if mask is not None:
            _attn = attn.clone()
            _attn[mask] = 0
            attn = _attn

        # Add the value
        heads = torch.matmul(attn, value)

        # Apply out projection
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.num_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, num_query, self.embed_dim)
        return out


class CombinedMHA(nn.Module):
    """Combined Multi-Head Attention module"""

    def __init__(self, num_heads: int, input_dim: int, embed_dim: int) -> None:
        """
        Initializes the CombinedMHA module.

        Args:
            num_heads (int): The number of attention heads.
            input_dim (int): The input dimension.
            embed_dim (int): The embedding dimension.
        """
        super(CombinedMHA, self).__init__()
        self.self_attention1 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.self_attention2 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.attention1 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.attention2 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.combined_attention1 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.combined_attention2 = MHA(num_heads, input_dim=input_dim, embed_dim=embed_dim)

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the CombinedMHA module.

        Args:
            input1 (torch.Tensor): The first input tensor.
            input2 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        x1_x1 = self.self_attention1(input1)
        x2_x2 = self.self_attention2(input2)

        x1_x2 = self.attention1(input1, input2)
        x2_x1 = self.attention2(input2, input1)

        out1 = self.combined_attention1(x1_x1, x1_x2)
        out2 = self.combined_attention2(x2_x2, x2_x1)
        return out1, out2


class MHABlock(nn.Module):
    """Multi-Head Attention Block module."""
    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            node_dim1: int = None,
            node_dim2: int = None,
            normalization: str = 'batch',
            feed_forward_hidden: int = 512,
            combined: bool = False) -> None:
        """
        Initializes the MHABlock module.

        Args:
            num_heads (int): The number of attention heads.
            embed_dim (int): The embedding dimension.
            node_dim1 (int): The dimension of the nodes for input 1.
            node_dim2 (int): The dimension of the nodes for input 2.
            normalization (str): The normalization type ('batch' or 'instance').
            feed_forward_hidden (int): The hidden size of the feed-forward layer.
            combined (bool): Whether to use combined MHA or not.
        """
        super(MHABlock, self).__init__()

        # Use Combined MHA or standard MHA
        self.combined = combined

        # To map input to embedding space
        self.init_embed1 = nn.Linear(node_dim1, embed_dim) if node_dim1 is not None else None
        self.init_embed2 = nn.Linear(node_dim2, embed_dim) if node_dim2 is not None else None

        # Attention operations
        attention_type = CombinedMHA if combined else MHA
        self.attention_block = SkipConnection(
            attention_type(
                num_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
            )
        )

        # Rest of skip connections + normalization + feed forward layers
        self.projection1 = [nn.Sequential(
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        ) for _ in range(2 if combined else 1)]
        if combined:
            self.projection1, self.projection2 = self.projection1
        else:
            self.projection1 = self.projection1[0]

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MHABlock module.

        Args:
            x (tuple): A tuple containing the input tensors.

        Returns:
            tuple: A tuple containing the output tensors.
        """
        if not isinstance(x, tuple):
            x = (x, None)
        x1, x2 = x

        # Batch multiply to get initial embeddings of nodes
        h1 = self.init_embed1(x1.view(-1, x1.size(-1))).view(*x1.size()[:2], -1) if self.init_embed1 is not None else x1
        h2 = self.init_embed2(x2.view(-1, x2.size(-1))).view(*x2.size()[:2], -1) if self.init_embed2 is not None else x2

        # Apply MHA block
        if self.combined:
            h1, h2 = self.attention_block(h1, h2)
            h = (self.projection1(h1), self.projection2(h2))
        else:
            h = self.attention_block(h1)
            h = (self.projection1(h), None)
        return h


class MHAEncoder(nn.Module):
    """Multi-Head Attention Encoder module."""

    def __init__(
            self,
            num_heads: int,
            embed_dim: int,
            num_blocks: int,
            node_dim1: int = None,
            node_dim2: int = None,
            normalization: str = 'batch',
            feed_forward_hidden: int = 512,
            combined: bool = True) -> None:
        """
        Initializes the MHAEncoder module.

        Args:
            num_heads (int): The number of attention heads.
            embed_dim (int): The embedding dimension.
            num_blocks (int): The number of blocks in the encoder.
            node_dim1 (int): The dimension of the nodes for input 1.
            node_dim2 (int): The dimension of the nodes for input 2.
            normalization (str): The normalization type ('batch' or 'instance').
            feed_forward_hidden (int): The hidden size of the feed-forward layer.
            combined (bool): Whether to use combined MHA or not.
        """
        super(MHAEncoder, self).__init__()

        # Dimension of initial embedding of each input (embedding is only applied at the beginning of the 1st block)
        node_dim1 = [None if i > 0 else node_dim1 for i in range(num_blocks)]
        node_dim2 = [None if i > 0 else node_dim2 for i in range(num_blocks)]

        # To map input to embedding space
        # self.init_embed = nn.Linear(node_dim1, embed_dim) if node_dim1 is not None else None  # Only for the 1st block

        # Initialize Multi-Head Attention (MHA) blocks
        self.layers = nn.Sequential(*(
            MHABlock(
                    num_heads,
                    embed_dim,
                    node_dim1=node_dim1[i],
                    node_dim2=node_dim2[i],
                    normalization=normalization,
                    feed_forward_hidden=feed_forward_hidden,
                    combined=combined
            )
            for i in range(num_blocks)
        ))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MHAEncoder module.

        Args:
            x1 (torch.Tensor): The first input tensor.
            x2 (torch.Tensor): The second input tensor (optional).

        Returns:
            tuple: A tuple containing the output tensors.
        """

        # Apply MHA blocks
        h1, h2 = self.layers((x1, x2))

        # Output the embeddings and the average of the embedding along dimension 1
        output = (
            h1,             # (batch_size, num_nodes, embed_dim)
            h1.mean(dim=1)  # average to get embedding of graph, (batch_size, embed_dim)
        )
        if x2 is not None:
            output += (h2, h2.mean(dim=1))
        return output
