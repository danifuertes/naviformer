import os
import math
import torch
from torch import nn

from .modules import *


class NaviFormer(nn.Module):
    """NaviFormer neural network"""

    def __init__(self,
                 embed_dim: int = 128,
                 num_dirs: int = 4,
                 combined_mha: bool = True,
                 two_step: str = '',
                 num_obs: tuple = (0, 0),
                 num_heads: int = 8,
                 num_blocks: int = 2,
                 tanh_clipping: float = 10.,
                 normalization: str = 'batch',
                 checkpoint_enc: bool = False,
                 **kwargs) -> None:
        """
        Initialize NaviFormer model.

        Args:
            embed_dim (int): Dimension of embeddings.
            num_dirs (int): Number of the directions the agent can choose to move.
            combined_mha (bool): Whether to use combined/standard MHA encoder.
            two_step (str): Pre-trained route planner for 2-step navigation planner.
            num_obs (tuple): (Minimum, Maximum) number of obstacles.
            num_heads (int): Number of heads for MHA layers.
            num_blocks (int): Number of encoding blocks.
            tanh_clipping (float): Clip tanh values.
            normalization (str): Type of normalization.
            checkpoint_enc (bool): Whether to checkpoint to decrease memory usage.
        """
        super(NaviFormer, self).__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension should be dividable by number of heads, " \
                                           f"found embed_dim={embed_dim} and num_heads={num_heads}"

        # Problem parameters
        self.num_obs = num_obs                           # (Minimum, Maximum) number of obstacles
        self.agent_id = 0                                # Agent ID (for decentralized multiagent problem)

        # Dimensions
        self.embed_dim = embed_dim                       # Dimension of embeddings

        # Encoder parameters
        self.combined_mha = combined_mha                 # Use combined/standard MHA encoder
        self.num_heads = num_heads                       # Number of heads for MHA layers
        self.num_blocks = num_blocks                     # Number of encoding blocks
        self.checkpoint_enc = checkpoint_enc             # Checkpoint to decrease memory usage

        # Decoder parameters
        self.temp = 1.0                                  # SoftMax temperature parameter
        self.decode_type = None                          # Greedy or sampling
        self.tanh_clipping = tanh_clipping               # Clip tanh values

        # Last node embedding (embed_dim) + remaining length and current position (3) + number of obstacles (max_obs)
        step_context_dim = embed_dim + 3 + num_obs[1]

        # Node dimension: x, y, prize (Nav_OP)
        node_dim = 3

        # Pre-trained (2-step) Transformer route planner
        if os.path.isdir(two_step) or os.path.isfile(two_step):
            self.base_route_model.set_decode_type("greedy", temp=self.temp)
            print(f"Loaded base route planner model {two_step} for 2-step NaviFormer")
            print('Freezing base route planner model layers for 2-step NaviFormer')
            for name, p in self.base_route_model.named_parameters():
                p.requires_grad = False
            self.two_step = True
        else:
            self.two_step = False

            # Special embedding projection for depot node
            self.init_embed_depot = nn.Linear(2, embed_dim)

            # Initial embedding
            self.init_embed = nn.Linear(node_dim, embed_dim)

            # Encoder embedding
            self.embedder = MHAEncoder(  # Node to node to obstacle attention
                num_heads=num_heads,
                embed_dim=embed_dim,
                node_dim2=3 if num_obs[1] else None,  # Obstacles (circles): x_center, y_center, radius
                num_blocks=num_blocks,
                normalization=normalization,
                combined=combined_mha
            )

        # Project graph embedding to get decoder embeddings for MHA: key, value, key_logit (so 3 * embed_dim)
        self.project_graph = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Project averaged graph embedding (across nodes) for state embedding
        self.project_graph_mean = nn.Linear(embed_dim, embed_dim, bias=False)

        # Project averaged obstacle embedding (across obstacles) for state embedding
        self.project_obs_mean = nn.Linear(embed_dim, embed_dim, bias=False)

        # Project state embedding
        self.project_state = nn.Linear(step_context_dim, embed_dim, bias=False)

        # Projection for the result of inner MHA (num_heads * val_dim == embed_dim, so input is embed_dim)
        self.project_mha = nn.Linear(embed_dim, embed_dim, bias=False)

        # Direction dimensions
        conv_dim = 4              # Convolution dimension
        num_maps = 4 * 2          # Number of local maps: 4 to represent obstacles and 4 to represent goals
        self.num_dirs = num_dirs  # Number of actions
        self.patch_size = 16      # Size of local maps
        self.map_size = 64        # Size of global map

        # Direction embeddings
        self.position_embed = nn.Linear(2, embed_dim)   # Embedding for the agent position
        self.path_prediction = nn.Sequential(               # Prediction layers
            nn.Conv2d(num_maps, conv_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dim, affine=True),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dim, affine=True),
            nn.Flatten(),
            nn.Linear(conv_dim * self.patch_size * self.patch_size // 2, embed_dim),
            nn.ReLU(),
            (nn.BatchNorm1d if normalization == 'batch' else nn.InstanceNorm1d)(embed_dim, affine=True),
            nn.Linear(embed_dim, self.num_dirs),
        )

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
        embeddings = self.encoder(state)  # Transformer encoder
        self.fixed_data = self.precompute(embeddings, state.obs)

        # Iterate until each environment from the batch reaches a terminal state
        while not done:

            # Predict actions and (log) probabilities for current state
            action, log_prob = self.step(state)

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

    def step(self, state: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Execute a step.

        Args:
            state (Any): State information.

        Returns:
            tuple: Actions and log probabilities.
        """

        # Transformer decoder
        log_probs_node, selected_node, log_probs_direction, selected_direction = self.decoder(state)

        # Combine actions in one tensor
        actions = torch.stack((selected_node, selected_direction), dim=1)

        # Combine log probabilities in one tensor
        log_prob_node = self.select_log_probs(log_probs_node, selected_node)
        log_prob_direction = self.select_log_probs(log_probs_direction, selected_direction)
        log_prob = self.combine_log_probs(log_prob_node, log_prob_direction)

        # Return actions and log probabilities
        return actions, log_prob

    def encoder(self, state: Any) -> torch.Tensor:
        """
        Encoder for the model.

        Args:
            state (Any): State information.

        Returns:
            torch.Tensor: Embeddings.
        """

        # Pre-trained (2-step) Transformer decoder
        if self.two_step:
            embeddings = self.base_route_model.encoder(state)
            return embeddings

        # Joint Transformer encoder
        init_embed = input_embed(state, self.init_embed, self.init_embed_depot)
        init_embed = (init_embed, state.obs) if self.combined_mha else (init_embed, )
        h = self.embedder(*init_embed)
        embeddings = (h[0], h[2]) if self.combined_mha else h[0]
        return embeddings

    def precompute(self, embeddings: torch.Tensor, obs: torch.Tensor | None = None) -> AttentionModelFixed:
        """
        Precompute Encoder embeddings.

        Args:
            embeddings (torch.Tensor): Embeddings.
            obs (torch.Tensor or None): Obstacles information.

        Returns:
            Precomputed AttentionModelFixed data.
        """

        # Pre-trained (2-step) Transformer encoder precompute
        if self.two_step:
            return self.base_route_model.precompute(embeddings, obs, map_info=(self.patch_size, self.map_size))

        # Obstacle embeddings
        if self.num_obs[1]:
            graph_embedding, obs_embedding = embeddings

            # Project averaged obstacle embedding (across obstacles) for state embedding
            obs_embedding_mean = self.project_obs_mean(obs_embedding.mean(1))

            # Create obstacle map for direction prediction
            obs_map, obs_grid = create_obs_map(obs, self.patch_size, self.map_size)
            obs_data = (obs_embedding_mean, obs_map, obs_grid)

        # No obstacles
        else:
            graph_embedding = embeddings
            obs_data = (None, None, None)

        # Project averaged graph embedding (across nodes) for state embedding
        graph_embedding_mean = self.project_graph_mean(graph_embedding.mean(1))

        # Project graph embedding for decoder
        key, value, key_logit = self.project_graph(graph_embedding).chunk(3, dim=-1)

        # Multiple heads required for key and value during MHA operation. Not needed for key_logit during SHA operation
        key_val_data = (
            make_heads(self.num_heads, key),
            make_heads(self.num_heads, value),
            key_logit.contiguous()
        )
        return AttentionModelFixed(graph_embedding, graph_embedding_mean, *key_val_data, *obs_data)

    def decoder(self, state: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decoder for the model.

        Args:
            state (Any): State information.

        Returns:
            tuple: Log probabilities and selected indices.
        """

        # Pre-trained (2-step) Transformer decoder
        if self.two_step:
            log_probs_node, selected_node = self.base_route_model.decoder(state)

        # Standard (1-step) decoding
        else:

            # Predict log probabilities for each node
            log_probs_node = self.predict_node(state)

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected_node = self.select_node(log_probs_node.exp(), state.get_mask_nodes())

        # Predict next action (direction from current position to next node to visit)
        log_probs_direction = self.predict_direction(state, selected_node)
        selected_direction = self.select_direction(log_probs_direction.exp())

        # Return actions and log probabilities
        return log_probs_node, selected_node, log_probs_direction, selected_direction

    def predict_node(self, state: Any, normalize: bool = True):
        """
        Predict log probabilities for each node.

        Args:
            state (Any): State information.
            normalize (bool): Whether to normalize.

        Returns:
            torch.Tensor: Log probabilities.
        """

        # Compute state embedding
        state_embedding = self.project_state(self.get_state_embedding(self.fixed_data.graph_embedding, state))

        # Compute the decoder's query from the state embedding
        query = self.fixed_data.graph_embedding_mean + self.fixed_data.obs_embedding_mean + state_embedding

        # Apply MHA
        mask = state.get_mask_nodes()
        query_logit = self.mha_decoder(
            query=query,
            key=self.fixed_data.key,
            value=self.fixed_data.value,
            mask=mask,
        )

        # Apply SHA
        log_probs = self.sha_decoder(
            query_logit=query_logit,
            key_logit=self.fixed_data.key_logit,
            mask=mask,
            normalize=normalize
        )

        # Return log probabilities
        return log_probs

    @staticmethod
    def get_state_embedding(embeddings: torch.Tensor, state: Any) -> torch.Tensor:
        """
        Get state embedding.

        Args:
            embeddings (torch.Tensor): Embeddings.
            state (Any): State information.

        Returns:
            torch.Tensor: State embedding.
        """

        # Get current node index and expand it to (B x 1 x H) to allow gathering from embedding (B x N x H)
        current_node = state.prev_node.contiguous()[:, None, None].expand(-1, 1, embeddings.size(-1))

        # Get embedding of current node
        last_node_embed = torch.gather(input=embeddings, dim=1, index=current_node)[:, 0]

        # Return context: (embedding of last node, remaining time/length, current position, distance to obstacles)
        return torch.cat((last_node_embed, state.length[..., None], state.position, state.get_dist2obs()), dim=-1)

    def mha_decoder(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> \
            torch.Tensor:
        """
        Multi-Head Attention (MHA) mechanism.

        Args:
            query (torch.Tensor): Query.
            key (torch.Tensor): Key.
            value (torch.Tensor): Value.
            mask (torch.Tensor): Mask.

        Returns:
            torch.Tensor: MHA output.
        """

        # Dimensions
        batch_size, embed_dim = query.size()
        key_size = value_size = embed_dim // self.num_heads

        # Rearrange query dimensions: (num_heads, batch_size, 1, key_size)
        query = query.view(batch_size, self.num_heads, 1, key_size).permute(1, 0, 2, 3)

        # Transpose key: (num_heads, batch_size, key_size, num_nodes)
        key = key.transpose(-2, -1)

        # Batch matrix multiplication to compute compatibilities: (num_heads, batch_size, 1, num_nodes)
        compatibility = torch.matmul(query, key) / torch.sqrt(torch.tensor(embed_dim))

        # Ban nodes prohibited by the mask
        compatibility[mask[None, :, None].expand_as(compatibility)] = -math.inf

        # Apply softmax
        compatibility = torch.softmax(compatibility, dim=-1)

        # Batch matrix multiplication with value to compute output: (num_heads, batch_size, 1, value_size)
        output = torch.matmul(compatibility, value)

        # Project to get glimpse/updated context node embedding: (batch_size, 1, embed_dim)
        return self.project_mha(
            output.permute(1, 2, 0, 3).contiguous().view(-1, 1, self.num_heads * value_size)
        )

    def sha_decoder(self,
                    query_logit: torch.Tensor,
                    key_logit: torch.Tensor,
                    mask: torch.Tensor,
                    normalize: bool = True) -> torch.Tensor:
        """
        Single-Head Attention (SHA) mechanism.

        Args:
            query_logit (torch.Tensor): Logit for query.
            key_logit (torch.Tensor): Logit for key.
            mask (torch.Tensor): Mask.
            normalize (bool): Whether to normalize.

        Returns:
            torch.Tensor: Log probabilities.
        """

        # Embedding dimension
        embed_dim = query_logit.size(-1)

        # Transpose key
        key_logit = key_logit.transpose(-2, -1)

        # Batch matrix multiplication to compute logits: (batch_size, 1, num_nodes) -> logits = 'compatibility'
        logits = torch.matmul(query_logit, key_logit) / torch.sqrt(torch.tensor(embed_dim))

        # Remove extra dimension of size 1
        logits = logits.squeeze(dim=1)

        # Clip the logits (logits are non-normalized probabilities)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping

        # Apply the mask to the logits
        logits[mask] = -math.inf

        # Normalize the logits (with log_softmax) to get log probabilities
        if normalize:
            log_probs = torch.log_softmax(logits / self.temp, dim=-1)

        # Check log_probs are not NaN and return them
        assert not torch.isnan(log_probs).any(), "NaNs found in logits"
        return log_probs

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
            torch.Tensor: Log probabilities.
        """
        batch_ids = torch.arange(next_node.shape[0], dtype=torch.int64, device=next_node.device)

        # Get next selected goal
        goal = state.get_regions()[batch_ids, next_node]

        # Get local maps
        maps = create_local_maps(
            state.position, self.fixed_data.obs_map, self.fixed_data.obs_grid, goal, self.patch_size, self.map_size
        )

        # Apply prediction layers
        policy = self.path_prediction(maps)

        # Ban prohibited dirs
        policy[state.get_mask_dirs()] = -math.inf

        # Return normalized (softmax) log probabilities
        log_probs = torch.log_softmax(policy, dim=-1)
        return log_probs

    def select_direction(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Select direction.

        Args:
            probs (torch.Tensor): Probabilities.

        Returns:
            torch.Tensor: Selected direction.
        """

        # ArgMax (Exploitation)
        if self.decode_type == "greedy":
            _, action = probs.max(dim=1)

        # Sample (Exploration)
        elif self.decode_type == "sampling":
            action = probs.multinomial(num_samples=1).squeeze(dim=1)
        else:
            assert False, "Unknown decode type"
        return action

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

    @staticmethod
    def combine_log_probs(log_probs_node: torch.Tensor, log_probs_direction: torch.Tensor) -> torch.Tensor:
        """
        Combine log probabilities.

        Args:
            log_probs_node (torch.Tensor): Log probabilities for nodes.
            log_probs_direction (torch.Tensor): Log probabilities for directions.

        Returns:
            torch.Tensor: Combined log probabilities.
        """
        return (log_probs_node + log_probs_direction) / 2
