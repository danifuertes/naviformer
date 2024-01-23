import torch
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input1, input2=None):
        if input2 is None:
            return input1 + self.module(input1)
        else:
            x1, x2 = self.module(input1, input2)
            return input1 + x1, input2 + x2


class MultiHeadBiAttention(nn.Module):

    def __init__(self, n_heads, input_dim, embed_dim):
        super(MultiHeadBiAttention, self).__init__()
        self.self_attention1 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.self_attention2 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.attention1 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.attention2 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.combined_attention1 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )
        self.combined_attention2 = MultiHeadAttention(
            n_heads,
            input_dim=input_dim,
            embed_dim=embed_dim
        )

    def forward(self, input1, input2):
        x1_x1 = self.self_attention1(input1)
        x2_x2 = self.self_attention2(input2)

        x1_x2 = self.attention1(input1, input2)
        x2_x1 = self.attention2(input2, input1)

        out1 = self.combined_attention1(x1_x1, x1_x2)
        out2 = self.combined_attention2(x2_x2, x2_x1)
        return out1, out2


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)
        # attn = torch.sigmoid(compatibility)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        return out


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class BiEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            node_dim1=None,
            node_dim2=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(BiEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed1 = nn.Linear(node_dim1, embed_dim) if node_dim1 is not None else None
        self.init_embed2 = nn.Linear(node_dim2, embed_dim) if node_dim2 is not None else None

        self.attention_block = SkipConnection(
            MultiHeadBiAttention(
                n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
            )
        )

        self.projection1, self.projection2 = [nn.Sequential(
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        ) for _ in range(2)]

    def forward(self, x, mask=None):
        x1, x2 = x
        assert mask is None, "TODO mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h1 = self.init_embed1(x1.view(-1, x1.size(-1))).view(*x1.size()[:2], -1) if self.init_embed1 is not None else x1
        h2 = self.init_embed2(x2.view(-1, x2.size(-1))).view(*x2.size()[:2], -1) if self.init_embed2 is not None else x2

        h1, h2 = self.attention_block(h1, h2)
        h = (self.projection1(h1), self.projection2(h2))
        return h


class GraphEncoderCMHA(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim1=None,
            node_dim2=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphEncoderCMHA, self).__init__()
        node_dim1 = [None if i > 0 else node_dim1 for i in range(n_layers)]  # Init embed is only for the first layer
        node_dim2 = [None if i > 0 else node_dim2 for i in range(n_layers)]  # Init embed is only for the first layer
        self.layers = nn.Sequential(*(
            BiEncoder(
                    n_heads,
                    embed_dim,
                    node_dim1=node_dim1[i],
                    node_dim2=node_dim2[i],
                    normalization=normalization,
                    feed_forward_hidden=feed_forward_hidden
            )
            for i in range(n_layers)
        ))

    def forward(self, x1, x2):
        h1, h2 = self.layers((x1, x2))
        return (
            h1,  # (batch_size, graph_size, embed_dim)
            h1.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
            h2,  # (batch_size, graph_size, embed_dim)
            h2.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )
