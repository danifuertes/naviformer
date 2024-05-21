import math
import torch
from torch import nn
from neural_astar.planner import NeuralAstar as NAStar
from neural_astar.utils.training import load_from_ptl_checkpoint

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


class GPNNAStar(nn.Module):
    """Graph Pointer network"""

    def __init__(self,
                 embed_dim: int = 128,
                 num_dirs: int = 4,
                 num_obs: tuple = (0, 0),
                 num_heads: int = 8,
                 tanh_clipping: float = 10.,
                 mask_inner=True,
                 mask_logits=True,
                 **kwargs) -> None:
        """
        Initialize Pointer network model.

        Args:
            embed_dim (int): Dimension of embeddings.
            num_dirs (int): Number of the directions the agent can choose to move.
            num_obs (tuple): (Minimum, Maximum) number of obstacles.
            num_heads (int): Number of heads for MHA layers.
            tanh_clipping (float): Clip tanh values.
            normalization (str): Type of normalization.
        """
        super(GPNNAStar, self).__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension should be dividable by number of heads, " \
                                           f"found embed_dim={embed_dim} and num_heads={num_heads}"

        # Dimensions
        self.embed_dim = embed_dim                       # Dimension of embeddings
        self.num_obs = num_obs                           # (Minimum, Maximum) number of obstacles

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

        # Decoder LSTM
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)

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
        
        # Weights for the GNN
        self.W1 = nn.Linear(embed_dim, embed_dim)
        self.W2 = nn.Linear(embed_dim, embed_dim)
        self.W3 = nn.Linear(embed_dim, embed_dim)

        # Aggregation function for the GNN
        self.agg_1 = nn.Linear(embed_dim, embed_dim)
        self.agg_2 = nn.Linear(embed_dim, embed_dim)
        self.agg_3 = nn.Linear(embed_dim, embed_dim)

        # Parameters to regularize the GNN
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)

        # Neural A* model
        self.patch_size = 16      # Size of local maps
        self.map_size = 64        # Size of global map
        self.na_star = NAStar(encoder_arch='CNN')
        self.na_star.load_state_dict(load_from_ptl_checkpoint(
            "./benchmarks/nop/methods/neural-astar/model/mazes_032_moore_c8/lightning_logs/"
        ))

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

    def forward(self, batch: dict | torch.Tensor, env: Any, test: bool = True) -> \
            Tuple[Any, Any, torch.Tensor, torch.Tensor] | Tuple[Any, Any]:
        """
        Forward pass of the model.

        Args:
            batch (dict or torch.Tensor): Batch data.
            env (Any): Environment data.
            test (bool): Indicates if model is in test mode, hence returning the actions and success

        Returns:
            tuple: Total reward, total log probability, actions (if test=True), and success (if test=True).
        """

        # Initialize state and other info
        state = env.get_state(batch)
        del batch
        done, total_reward, total_log_prob, actions = False, 0, 0, tuple()

        # Calculate graph embeddings during the first iteration
        self.fixed_data = self.precompute(state.obs)

        # Iterate until each environment from the batch reaches a terminal state
        hidden = None
        while not done:
            
            embeddings, hidden = self.encoder(state, hidden)

            # Predict actions and (log) probabilities for current state
            action, path, log_prob, hidden = self.step(state, embeddings, hidden)

            # Get reward and update state based on the action predicted
            state = state.step(action, path=path[..., 1:])
            reward, done = state.reward, state.done
            
            # Combine action (next node) with path
            num_steps = path.shape[1]
            action = action.reshape(-1, 1, 1).expand(action.shape[0], num_steps, 1)
            action = torch.cat((action, path[..., 1:]), dim=-1)

            # Update info
            actions = actions + (action,)
            total_reward += reward
            total_log_prob += log_prob

        # Check success
        success = state.check_success()

        # Return reward and log probabilities
        if test:
            return total_reward, total_log_prob, torch.cat(actions, dim=1), success
        return total_reward, total_log_prob

    def step(self, state: Any, embeddings: torch.Tensor, hidden: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Execute a step.

        Args:
            state (Any): State information.

        Returns:
            tuple: Actions and log probabilities.
        """

        # Transformer decoder
        log_probs_node, selected_node, _, selected_direction, hidden = self.decoder(state, embeddings, hidden)

        # Combine log probabilities in one tensor
        log_prob = self.select_log_probs(log_probs_node, selected_node)

        # Return actions and log probabilities
        return selected_node, selected_direction, log_prob, hidden

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
        context = self.get_context_embedding(node_embedding)
        
        # Return encoded data
        return (context, current_node), None

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

    def decoder(self, state: Any, embeddings: torch.Tensor, hidden: Tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
        """
        Decoder for the model.

        Args:
            state (Any): State information.

        Returns:
            tuple: Log probabilities and selected indices.
        """

        # Predict log probabilities for each node
        log_probs_node, hidden = self.predict_node(state, embeddings, hidden)

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected_node = self.select_node(log_probs_node.exp(), state.get_mask_nodes())

        # Predict next action (direction from current position to next node to visit)
        selected_direction = self.predict_direction(state, selected_node)
        log_probs_direction = 0

        # Return actions and log probabilities
        return log_probs_node, selected_node, log_probs_direction, selected_direction, hidden
    
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
    
    def get_context_embedding(self, node_embedding):
        
        # Dimensions
        num_regions, batch_size, embed_dim = node_embedding.shape
        node_embedding = node_embedding.view(-1, embed_dim)
        
        # GNN
        context = self.r1 * self.W1(node_embedding) + (1 - self.r1) * torch.nn.functional.relu(self.agg_1(node_embedding))
        context = self.r2 * self.W2(context) + (1 - self.r2) * torch.nn.functional.relu(self.agg_2(context))
        context = self.r3 * self.W3(context) + (1 - self.r3) * torch.nn.functional.relu(self.agg_3(context))
        return context.view(num_regions, batch_size, -1)  # output: (sourceL x batch_size x embedding_dim)

    def predict_node(self, state: Any, embeddings: torch.Tensor, hidden: Tuple):
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
        h, c = self.lstm(current_node, hidden)
        
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

    def predict_direction(self, state: Any, next_node: torch.Tensor) -> torch.Tensor:
        """
        Predict direction/angle.

        Args:
            state (torch.Tensor): State information.
            next_node (torch.Tensor): Next node.

        Returns:
            torch.Tensor: Predicted directions.
        """
        batch_size = next_node.shape[0]
        batch_ids = torch.arange(batch_size, dtype=torch.int64, device=next_node.device)
        obs_map = self.obs2maps(state.obs)
        
        # Get current position
        start = state.position
        start_map = self.node2map(start, obs_map, batch_ids)

        # Get next selected goal
        goal = state.get_regions()[batch_ids, next_node]
        goal_map = self.node2map(goal, obs_map, batch_ids)
        
        # Ensure nodes do not fall within obstacle location after rescale
        obs_map[start_map == 1] = 1
        obs_map[goal_map == 1] = 1

        # Predict path with NA*
        try:
            predicted_map = self.na_star(
                obs_map[:, None].contiguous(),
                start_map[:, None].contiguous(),
                goal_map[:, None].contiguous()
            ).paths[:, 0]
            selected_direction = self.map2dirs(predicted_map, start, goal)
        except:
            path = torch.stack((start, goal), dim=1)
            selected_direction = torch.cat((
                torch.zeros_like(path[..., 0, None]), path
            ), dim=-1)
        return selected_direction
    
    def obs2maps(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Create obstacles' map.

        Args:
            obs (torch.Tensor): obstacles.

        Returns:
            torch.Tensor: obstacles' map.
        """
        batch_size, num_obs, _ = obs.shape
        x, y = torch.meshgrid(
            torch.linspace(0, 1, self.map_size, device=obs.device),
            torch.linspace(0, 1, self.map_size, device=obs.device),
            indexing="ij"
        )
        xy = torch.stack((x, y), dim=-1).expand([batch_size, num_obs, self.map_size, self.map_size, 2])

        # Calculate distance from each point of the meshgrid to each obstacle center
        distances = (xy - obs[..., None, None, :2]).norm(2, dim=-1)

        # Create the masks by comparing distances with radius
        return (distances > obs[..., None, None, 2] + 0.02).all(dim=1).permute(0, 2, 1).float()
    
    def node2map(self, node: torch.Tensor, obs_map: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a map for either the start or goal nodes as required by Neural A*.

        Args:
            node (torch.Tensor): coordinates of the start or goal node.
            obs_map (torch.Tensor): obstacles' map.
            batch_ids (torch.Tensor): arange of batch ids.

        Returns:
            torch.Tensor: map for start/goal node.
        """
        node_x = (node[..., 0] * self.map_size).type(torch.int64)
        node_x = torch.clamp(node_x, 0, self.map_size - 1).long()
        node_y = (node[..., 1] * self.map_size).type(torch.int64)
        node_y = torch.clamp(node_y, 0, self.map_size - 1).long()
        node_map = torch.zeros_like(obs_map)
        node_map[batch_ids, node_y, node_x] = 1
        return node_map
    
    def map2dirs(self, predicted_map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, max_iters: int = 300) -> torch.Tensor:
        """
        Convert map with path predicted by Neural A* into a sequence of coordinates.

        Args:
            predicted_map (torch.Tensor): map with the predicted path predicted by Neural A*.
            start (torch.Tensor): coordinates of starting node.
            goal (torch.Tensor): coordinates of goal node.
            max_iters (int, optional): max number of iterations. Defaults to 300.

        Returns:
            torch.Tensor: _description_
        """
        pos = start.clone()
        
        # Initialize mask that allows different path lengths across elements from the same batch
        not_done = torch.ones_like(pos[:, 0]).bool()
        
        # Add starting node at the beginning of the path
        path = [
            torch.cat((torch.zeros_like(pos[:, 0, None]).long(), start), dim=-1)
        ]
        
        # Sequentially decode path
        i = 0
        while not_done.any().item() and i < max_iters:
            dirs = -torch.ones_like(pos[:, 0]).long()
            
            # Get next position from current node
            p, d, predicted_map = self.max_adjacent(predicted_map, pos)
            
            # Save next position if not done yet
            pos[not_done], dirs[not_done] = p[not_done], d[not_done]
            path.append(torch.cat((dirs[:, None], pos), dim=-1))
            
            # Check if done
            not_done = ~self.check_coords(pos, goal)
            i += 1
        
        # Add goal node at the end of the path
        path.append(
            torch.cat((-torch.ones_like(pos[:, 0, None]).long(), goal), dim=-1)
        )
        
        # Return path as tensor
        return torch.stack(path, dim=1)
        
    @ staticmethod
    def max_adjacent(maps: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From given position (coords) and map with path predicted by Neural A*, get the next adjacent position of the path.

        Args:
            maps (torch.Tensor): map with the predicted path predicted by Neural A*.
            coords (torch.Tensor): current position.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: next position, next direction, updated map (with current position removed).
        """
        
        # Dimensions
        batch_size, height, width = maps.size()
        
        # Get coordinates in map reference system
        x = (coords[..., 0] * width).clamp(0, width - 1)
        y = (coords[..., 1] * height).clamp(0, height - 1)
        
        # Set current coordinates to zero in the map
        maps[torch.arange(batch_size).to(maps.device), y.long(), x.long()] = 0
        
        # Generate indices for adjacent positions including diagonals
        row_offsets = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], device=maps.device)
        col_offsets = torch.tensor([1, 1, 0, -1, -1, -1, 0, 1], device=maps.device)
        row_idx = (
            y[:, None].expand(batch_size, len(row_offsets)) + \
            row_offsets[None].expand(batch_size, len(row_offsets))
        ).clamp(0, height - 1).long()
        col_idx = (
            x[:, None].expand(batch_size, len(col_offsets)) + \
            col_offsets[None].expand(batch_size, len(col_offsets))
        ).clamp(0, width - 1).long()
        
        # Get values of adjacent positions
        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, 8).to(maps.device)
        adjacent_values = maps[batch_idx, row_idx, col_idx]
        
        # Find maximum value (directions)
        _, dirs = torch.max(adjacent_values.view(batch_size, -1), dim=-1)
        
        # Gather coordinates of maximum values
        next_coords = torch.stack((
            col_idx.gather(1, dirs[:, None])[:, 0] / height,
            row_idx.gather(1, dirs[:, None])[:, 0] / width,
        ), dim=-1)
        return next_coords, dirs, maps
    
    @staticmethod
    def check_coords(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Check if 2 positions are the same (or very close to each other).

        Args:
            coords1 (torch.Tensor): first position.
            coords2 (torch.Tensor): second position.

        Returns:
            torch.Tensor: boolean indicating if the positions are the same or not.
        """
        return torch.linalg.norm(coords1 - coords2, dim=-1) < 0.02

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
