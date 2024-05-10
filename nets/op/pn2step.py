import math
import torch
from torch import nn

from ..modules import *


class Fixed(NamedTuple):
    """
    Context for decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    obs_map: torch.Tensor
    obs_grid: torch.Tensor

    def __getitem__(self, key: slice | torch.Tensor) -> 'Fixed':
        """
        Get item based on the key.

        Args:
            key (slice or torch.Tensor): The index or slice.

        Returns:
            Fixed: The sliced named tuple.
        """
        assert torch.is_tensor(key) or isinstance(key, slice)
        return Fixed(
            obs_map=self.obs_map[key],
            obs_grid=self.obs_grid[key],
        )


class Encoder(nn.Module):
    """Maps a graph represented as an input sequence to a hidden vector"""

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.init_hx, self.init_cx = self.init_hidden(hidden_dim)

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        output, hidden = self.lstm(x, hidden)
        return output, hidden

    def init_hidden(self, hidden_dim):
        """Trainable initial hidden state"""
        std = 1. / math.sqrt(hidden_dim)
        enc_init_hx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_hx.data.uniform_(-std, std)

        enc_init_cx = nn.Parameter(torch.FloatTensor(hidden_dim))
        enc_init_cx.data.uniform_(-std, std)
        return enc_init_hx, enc_init_cx


class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10.):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits


class PN2Step(nn.Module):
    """Pointer network in a 2-step way"""

    def __init__(self,
                 embed_dim: int = 128,
                 num_obs: tuple = (0, 0),
                 num_heads: int = 8,
                 tanh_clipping: float = 10.,
                 mask_inner=True,
                 mask_logits=True,
                 **kwargs) -> None:
        """
        Initialize Pointer network 2-step model.

        Args:
            embed_dim (int): Dimension of embeddings.
            num_obs (tuple): (Minimum, Maximum) number of obstacles.
            num_heads (int): Number of heads for MHA layers.
            tanh_clipping (float): Clip tanh values.
            normalization (str): Type of normalization.
        """
        super(PN2Step, self).__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension should be dividable by number of heads, " \
                                           f"found embed_dim={embed_dim} and num_heads={num_heads}"

        # Dimensions
        self.embed_dim = embed_dim                       # Dimension of embeddings
        self.num_obs = num_obs                           # (Minimum, Maximum) number of obstacles
        self.patch_size = 16                             # Size of local maps
        self.map_size = 64                               # Size of global map

        # Decoder parameters
        self.temp = 1.0                                  # SoftMax temperature parameter
        self.decode_type = None                          # Greedy or sampling
        self.tanh_clipping = tanh_clipping               # Clip tanh values

        # Node dimension: x, y, prize, max_length
        self.node_dim = 4
        
        # Initial embedding projection
        std = 1. / math.sqrt(embed_dim)
        self.node_embed = nn.Parameter(torch.FloatTensor(self.node_dim, embed_dim))
        self.node_embed.data.uniform_(-std, std)

        # Placeholder for the starting node
        self.init_node_placeholder = nn.Parameter(torch.FloatTensor(embed_dim))
        self.init_node_placeholder.data.uniform_(-std, std)

        # Encoder LSTM
        self.lstm_enc = Encoder(embed_dim, embed_dim)

        # Decoder LSTM
        self.lstm_dec = nn.LSTMCell(embed_dim, embed_dim)

        # Attention
        self.pointer = Attention(embed_dim, use_tanh=tanh_clipping > 0, C=tanh_clipping)
        self.glimpse = Attention(embed_dim, use_tanh=False)
        self.softmax = nn.Softmax(dim=1)
        self.num_glimpses = 1
        self.tanh_exploration = tanh_clipping

        # Mask
        self.mask_glimpses = mask_inner
        self.mask_logits = mask_logits
        self.decode_type = None  # Needs to be set explicitly before use

        # Initialize fixed data (computed only during the first iteration)
        self.fixed_data = None

    def set_decode_type(self, decode_type: str, temp: float | None = None) -> None:
        """
        Set decode type: either greedy (exploitation) or sampling (exploration).

        Args:
            decode_type (str): Type of decoding.
            temp (float): SoftMax temperature parameter (optional).
        """
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, batch: dict | torch.Tensor, env: Any, eval: bool = True) -> \
            Tuple[Any, Any, torch.Tensor, torch.Tensor] | Tuple[Any, Any]:
        """
        Forward pass of the model.

        Args:
            batch (dict or torch.Tensor): Batch data.
            env (Any): Environment data.
            eval (bool): Indicates if model is in eval mode, hence returning the actions and success

        Returns:
            tuple: Total reward, total log probability, actions (if eval=True), and success (if eval=True).
        """

        # Initialize state and other info
        state = env.get_state(batch)
        del batch
        done, total_reward, total_log_prob, actions = False, 0, 0, tuple()

        # Calculate graph embeddings during the first iteration
        self.fixed_data = self.precompute(state.obs)

        # Iterate until each environment from the batch reaches a terminal state
        hidden_enc, hidden_dec = None, None
        while not done:
            
            embeddings, hidden_enc = self.encoder(state, hidden_enc)
            hidden_dec = (hidden_enc[0][-1], hidden_enc[1][-1]) if hidden_dec is None else hidden_dec

            # Predict actions and (log) probabilities for current state
            action, log_prob, hidden_dec = self.step(state, embeddings, hidden_dec)

            # Get reward and update state based on the action predicted
            state = state.step(action)
            reward, done = state.reward, state.done

            # Update info
            actions = actions + (action,)
            total_reward += reward
            total_log_prob += log_prob

        # Check success
        success = state.check_success()

        # Return reward and log probabilities
        if eval:
            return total_reward, total_log_prob, torch.stack(actions, dim=1), success
        return total_reward, total_log_prob

    def step(self, state: Any, embeddings: torch.Tensor, hidden_dec: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Execute a step.

        Args:
            state (Any): State information.

        Returns:
            tuple: Actions and log probabilities.
        """

        # Transformer decoder
        log_probs, actions, hidden_dec = self.decoder(state, embeddings, hidden_dec)

        # Combine log probabilities in one tensor
        log_probs = self.select_log_probs(log_probs, actions)

        # Return actions and log probabilities
        return actions, log_probs, hidden_dec

    def encoder(self, state: Any, hidden: torch.Tensor = None) -> torch.Tensor:
        """
        Encoder for the model.

        Args:
            state (Any): State information.

        Returns:
            torch.Tensor: Embeddings.
        """
        batch_size = state.get_batch_size()
        
        # Calculate node embedding
        node_embedding = self.get_node_embedding(state)

        # If first iteration
        if hidden is None:
            
            # Initialize hidden state of encoder and decoder LSTMs
            h0 = c0 = torch.autograd.Variable(
                torch.zeros(1, batch_size, self.embed_dim, out=node_embedding.data.new()),
                requires_grad=False
            )
            hidden = (h0, c0)
            
            # Placeholder for first current node
            current_node = self.init_node_placeholder.unsqueeze(0).repeat(batch_size, 1)
        
        # Rest of iterations
        else:
            
            # Current node embedding is calculated from embedding of last visited node
            current_node = torch.gather(
                input=node_embedding,
                dim=0,
                index=state.prev_node.contiguous().view(1, batch_size, 1).expand(1, batch_size, *node_embedding.size()[2:])
            ).squeeze(0)
        
        # Calculate context embedding
        context, hidden = self.get_context_embedding(node_embedding, *hidden)
        
        # Return encoded data
        return (context, current_node), hidden

    def precompute(self, obs: torch.Tensor | None = None) -> Fixed:
        """
        Precompute Encoder embeddings.

        Args:
            obs (torch.Tensor or None): Obstacles information.

        Returns:
            Precomputed Fixed data.
        """

        # Obstacle embeddings
        if self.num_obs[1]:

            # Create obstacle map for direction prediction
            obs_data = create_obs_map(obs, self.patch_size, self.map_size)

        # No obstacles
        else:
            obs_data = (None, None, None)
        return Fixed(*obs_data)

    def decoder(self, state: Any, embeddings: torch.Tensor, hidden_dec: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Decoder for the model.

        Args:
            state (Any): State information.

        Returns:
            tuple: Log probabilities and selected indices.
        """

        # Predict log probabilities for each node
        log_probs_node, hidden_dec = self.predict_node(state, embeddings, hidden_dec)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected_node = self.select_node(log_probs_node.exp(), state.get_mask_nodes())

        # Return actions and log probabilities
        return log_probs_node, selected_node, hidden_dec
    
    def get_node_embedding(self, state):
        
        # Dimensions
        batch_size = state.get_batch_size()
        num_regions = state.get_num_regions()
        
        # Get regions with depots
        regions = state.get_regions()
        
        # Get regions' prizes
        prizes = state.get_prizes()[..., None]
        
        # Get distance from each node to the depot
        dist2regions = state.get_dist2regions(state.position)
        
        # Get max length nd substract the distance from each node to the depot. Then, normalize it by diving the result by max length
        max_length = (state.get_remaining_length()[..., None] - dist2regions)[..., None]
        max_length = max_length / state.max_length[:, None, None].expand(*max_length.shape)

        # Concatenate spatial info (loc), prize info (prize) and temporal info (max_length)
        data = torch.cat((regions, prizes, max_length), dim=-1)

        # Apply node embedding
        node_embedding = torch.mm(
            data.transpose(0, 1).contiguous().view(-1, self.node_dim),
            self.node_embed
        ).view(num_regions, batch_size, -1)
        return node_embedding
    
    def get_context_embedding(self, node_embedding, h, c):
        context, hidden_enc = self.lstm_enc(node_embedding, (h, c))
        return context, hidden_enc

    def predict_node(self, state: Any, embeddings: torch.Tensor, hidden_dec: Tuple):
        """
        Predict log probabilities for each node.

        Args:
            state (Any): State information.

        Returns:
            torch.Tensor: Log probabilities.
        """        
        mask = state.get_mask_nodes()
        context, current_node = embeddings
        
        # LSTM
        h, c = self.lstm_dec(current_node, hidden_dec)
        
        # Attention model
        for _ in range(self.num_glimpses):
            ref, logits = self.glimpse(h, context)
            # For the glimpses, only mask before softmax, so we have always an L1 norm 1 readout vector
            if self.mask_glimpses:
                logits[mask] = -math.inf
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] = [batch_size x h_dim x 1]
            h = torch.bmm(ref, self.softmax(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(h, context)  # query = h, ref = context
        
        # Masking before softmax makes probs sum to one
        if self.mask_logits:
            logits[mask] = -math.inf
        
        # Calculate log_softmax for better numerical stability
        log_probs = torch.log_softmax(logits, dim=1)
        if not self.mask_logits:
            log_probs[mask] = -math.inf

        # Return log probabilities
        return log_probs, (h, c)
    
    def select_node(self, probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        ArgMax or sample from probabilities to select next node.

        Args:
            probs (torch.Tensor): Probabilities.
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: Selected indices.
        """

        # ArgMax (Exploitation)
        if self.decode_type == "greedy":
            _, selected = probs.max(dim=1)
            if mask is not None:
                assert not mask.gather(dim=1, index=selected.unsqueeze(-1)).data.any(), \
                    "Decode greedy: infeasible action has maximum probability"

        # Sample (Exploration)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(num_samples=1).squeeze(dim=1)

            # Sampling can fail due to GPU bug: https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            if mask is not None:
                while mask.gather(dim=1, index=selected.unsqueeze(dim=-1)).data.any():
                    print('Sampled bad values, resampling!')
                    selected = probs.multinomial(num_samples=1).squeeze(dim=1)
        else:
            assert False, "Unknown decode type"
        return selected

    @staticmethod
    def select_log_probs(log_probs: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        """
        Calculate log likelihood for loss function.

        Args:
            log_probs (torch.Tensor): Log probabilities.
            selected (torch.Tensor): Selected indices.

        Returns:
            torch.Tensor: Log probability.
        """
        log_prob = log_probs.gather(1, selected[..., None])[..., 0]
        assert (log_prob > -1000).data.all(), "Log probabilities should not be -inf, check sampling procedure!"
        return log_prob
