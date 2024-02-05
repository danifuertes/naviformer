from torch import nn

from nets.modules.net_utils import input_embed
from nets.modules.encoder import MHAEncoder


class Critic(nn.Module):

    def __init__(
        self,
        input_dim1,
        input_dim2,
        embed_dim,
        num_blocks,
        normalization,
        combined_mha=True
    ):
        super(Critic, self).__init__()

        # Combined MHA encoder
        self.combined_mha = combined_mha

        # Initial embedding
        self.init_embed = nn.Linear(input_dim1, embed_dim)
        self.embed_depot = nn.Linear(2, embed_dim)

        # MHA Encoder
        self.encoder = MHAEncoder(
            num_heads=8,
            embed_dim=embed_dim,
            node_dim2=input_dim2,  # Obstacles (circles): x_center, y_center, radius
            num_blocks=num_blocks,
            normalization=normalization
        )

        # Critic value prediction
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, batch, env):

        # Initialize state
        state = env.get_state(batch)
        del batch

        # Initial embedding
        x = input_embed(state, self.init_embed, self.embed_depot)
        x = (x, state.obs) if self.combined_mha else (x, None)

        # Graph embedding (encoder)
        graph_embeddings = self.encoder(*x)

        # Critic value prediction
        return self.value_head(graph_embeddings[1])[:, 0]
