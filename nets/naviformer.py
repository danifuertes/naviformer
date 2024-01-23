import math
import os.path
from torch import nn
from torch.utils.checkpoint import checkpoint

# from utils.functions import sample_many, adapt_multi_nav, load_model

from .modules import *


def get_context(embeddings, state):
    """Returns the context per step, optionally for multiple steps at once (for efficient eval of the model)."""

    # Get current node index and expand it to (B x 1 x H) to allow gathering from embedding (B x N x H)
    current_node = state['prev_node'].contiguous()[:, None, None].expand(-1, 1, embeddings.size(-1))

    # Get embedding of current node
    last_node_embed = torch.gather(input=embeddings, dim=1, index=current_node)[:, 0]

    # Return context: (embedding of last node, remaining time/length, current position, distance to obstacles)
    return torch.cat((last_node_embed, state['length'][..., None], state['position'], state['dist2obs']), dim=-1)


class NaviFormer(nn.Module):

    def __init__(self,
                 embed_dim,
                 hidden_dim,
                 problem,
                 combined_mha=True,
                 two_step='',
                 num_heads=8,
                 num_blocks=2,
                 tanh_clipping=10.,
                 normalization='batch',
                 checkpoint_enc=False,
                 shrink_size=None,
                 num_depots=1,
                 num_agents=1,
                 info_th=0.2,
                 max_obs=0,
                 **kwargs):
        super(NaviFormer, self).__init__()
        assert embed_dim % num_heads == 0

        # Problem parameters
        self.num_agents = num_agents                     # Number of agents
        self.num_depots = num_depots                     # Number of depots
        self.max_obs = max_obs                           # Maximum number of obstacles
        self.problem = problem                           # Type of problem to solve
        self.agent_id = 0                                # Agent ID (for decentralized multiagent problem)
        self.info_th = info_th                           # Communication distance (for decentralized multiagent problem)

        # Dimensions
        self.embed_dim = embed_dim                       # Dimension of embeddings
        self.hidden_dim = hidden_dim                     # Dimension of hidden layers

        # Encoder parameters
        self.combined_mha = combined_mha                 # Use combined/standard MHA encoder
        self.num_heads = num_heads                       # Number of heads for MHA layers
        self.num_blocks = num_blocks                     # Number of encoding blocks
        self.checkpoint_enc = checkpoint_enc             # Checkpoint to decrease memory usage

        # Decoder parameters
        self.temp = 1.0                                  # SoftMax temperature parameter
        self.decode_type = None                          # Greedy or sampling
        self.shrink_size = shrink_size                   # Shrink batch size to decrease memory usage
        self.tanh_clipping = tanh_clipping               # Clip tanh values

        # Last node embedding (embed_dim) + remaining length and current position (3) + number of obstacles (max_obs)
        step_context_dim = embed_dim + 3 + max_obs

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
                node_dim2=3 if max_obs > 0 else None,  # Obstacles (circles): x_center, y_center, radius
                num_blocks=self.num_blocks,
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
        conv_dim = 4            # Convolution dimension
        self.num_actions = 4    # Number of actions
        self.patch_size = 16    # Size of local maps
        self.map_size = 64      # Size of global map

        # Direction embeddings
        self.position_embed = nn.Linear(2, embed_dim)   # Embedding for the agent position
        self.path_prediction = nn.Sequential(               # Prediction layers
            nn.Conv2d(self.num_actions * 2, conv_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dim, affine=True),
            nn.Conv2d(conv_dim, conv_dim, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(conv_dim, affine=True),
            nn.Flatten(),
            nn.Linear(conv_dim * self.patch_size * self.patch_size // 2, hidden_dim),
            nn.ReLU(),
            (nn.BatchNorm1d if normalization == 'batch' else nn.InstanceNorm1d)(hidden_dim, affine=True),
            nn.Linear(hidden_dim, self.num_actions),
        )

        # Initialize fixed data (computed only during the first iteration)
        self.fixed_data = None

    def set_decode_type(self, decode_type, temp=None):
        """Either greedy (exploitation) or sampling (exploration)."""
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, state):

        # Calculate graph embeddings during the first iteration
        if 'inputs' in state:
            embeddings = self.encoder(state['inputs'])  # Transformer encoder
            obs = state['inputs']['obs'] if 'obs' in state['inputs'] else None
            self.fixed_data = self.precompute(embeddings, obs)
        assert self.fixed_data is not None, "Graph embeddings are missing, make sure you use the encoder"

        # Transformer decoder
        logits_node, selected_node, logits_direction, selected_direction = self.decoder(state)

        # Return actions and logits
        actions = (selected_node, selected_direction)  # TODO: stack them
        logits = self.combine_logits(logits_node, logits_direction)
        return actions, logits

    def combine_logits(self, logits_nodes, logits_directions):  # TODO: use calc_log_likelihood to combine both
        return logits_nodes + logits_directions


        # # Calculate costs based on the predictions made
        # cost, mask = self.problem.get_costs(states)
        #
        # # Log likelihood computed here since it can be of different lengths, which may cause problems with DataParallel
        # ll_nodes = self.calc_log_likelihood(logits_nodes, pi[..., 0].type(torch.int64), mask)
        # ll_actions = self.calc_log_likelihood(logits_actions, pi[..., 1].type(torch.int64), mask)
        #
        # # Adapt multi-agent case
        # ll, cost = adapt_multi_nav(ll_nodes, cost, ll_actions)
        #
        # # Return costs, log likelihood and predictions
        # if return_pi:
        #     return cost, ll, pi
        # return cost, ll

    def encoder(self, inputs):

        # Pre-trained (2-step) Transformer decoder
        if self.two_step:
            embeddings = self.base_route_model.encoder(inputs)
            return embeddings

        # Joint Transformer encoder
        init_embed = input_embed(inputs, self.init_embed, self.init_embed_depot)
        init_embed = (init_embed, inputs['obs']) if self.combined_mha else (init_embed, )

        # Only checkpoint if we need gradients
        if self.checkpoint_enc and self.training:
            h = checkpoint(self.embedder, *init_embed)
        else:
            h = self.embedder(*init_embed)
        embeddings = (h[0], h[2]) if self.combined_mha else h[0]
        return embeddings

    def precompute(self, embeddings, obs=None):
        """Precompute Encoder embeddings."""

        # Pre-trained (2-step) Transformer encoder precompute
        if self.two_step:
            return self.base_route_model.precompute(embeddings, obs, map_info=(self.patch_size, self.map_size))

        # Obstacle embeddings
        if self.max_obs:
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

    def decoder(self, state):

        # Pre-trained (2-step) Transformer decoder
        if self.two_step:
            log_probs_node, selected_node = self.base_route_model.decoder(state)

        # Standard (1-step) decoding
        else:

            # Predict log probabilities for each node
            log_probs_node = self.get_log_probs(state)
            probs_node = log_probs_node.exp()

            # Select the indices of the next nodes in the sequences, result (batch_size) long
            selected_node = self.select_node(probs_node, state['mask_nodes'])

        # Predict next action (direction from current position to next node to visit)
        selected_direction, log_probs_direction = self.predict_direction(state, selected_node)
        return log_probs_node, selected_node, log_probs_direction, selected_direction

    def get_log_probs(self, state, normalize=True):
        """Predict log probabilities."""

        # Compute state_embedding
        state_embedding = self.project_state(get_context(self.fixed_data.graph_embedding, state))

        # Compute the decoder's query from the state embedding
        query = self.fixed_data.graph_embedding_mean + self.fixed_data.obs_embedding_mean + state_embedding

        # Apply MHA
        query_logit = self.mha_decoder(
            query=query,
            key=self.fixed_data.key,
            value=self.fixed_data.value,
            mask=state['mask_nodes'],
        )

        # Apply SHA
        log_probs = self.sha_decoder(
            query_logit=query_logit,
            key_logit=self.fixed_data.key_logit,
            mask=state['mask_nodes'],
            normalize=normalize
        )

        # Return log probabilities
        return log_probs

    def mha_decoder(self, query, key, value, mask):
        """Multi-Head Attention (MHA) mechanism"""

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

    def sha_decoder(self, query_logit, key_logit, mask, normalize=True):
        """Single-Head Attention (SHA) mechanism"""

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

    def predict_direction(self, state, next_node):
        """Direction/Angle prediction."""

        # Get next selected goal
        goal = state['regions'].get_regions_by_index(next_node)

        # Get local maps
        maps = create_local_maps(
            state['position'], self.fixed_data.obs_map, self.fixed_data.obs_grid, goal, self.patch_size, self.map_size
        )

        # Apply prediction layers
        policy = self.path_prediction(maps)

        # Ban prohibited actions
        policy[state['mask_actions']] = -math.inf

        # Normalize policy with softmax
        prob = torch.softmax(policy, dim=-1)

        # Get action
        if self.decode_type == "greedy":
            _, action = prob.max(dim=1)
        elif self.decode_type == "sampling":
            action = prob.multinomial(num_samples=1).squeeze(dim=1)
        else:
            assert False, "Unknown decode type"

        # Get log probabilities
        log_prob = torch.log_softmax(policy, dim=-1)

        # Return action and log_probabilities
        return action, log_prob

    def sample_many(self, inputs, batch_rep=1, iter_rep=1):
        """
        A bit ugly, but we need to pass the embeddings as well. Making a tuple will not work with the problem.get_cost
        function.
        """
        h = self.encoder(inputs)
        return sample_many(

            # Need to unpack tuple into arguments
            lambda inp: self.inner(*inp),

            # Don't need embeddings as input to get_costs
            lambda cost: self.problem.get_costs(cost),

            # Pack input with embeddings (additional input)
            (inputs, h),
            batch_rep, iter_rep, self.max_obs
        )

    def select_node(self, probs, mask):
        """ArgMax or sample from probabilities to select next node."""

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

    def calc_log_likelihood(self, _log_p_nodes, nodes, mask):
        """Calculate log likelihood for loss function."""

        # Get log probabilities corresponding to selected nodes
        log_p_nodes = []
        for k in range(self.num_agents):
            log_p_nodes.append(_log_p_nodes[..., k, :].gather(2, nodes[..., k, None]).squeeze(-1))

            # Optional: mask out actions irrelevant to objective, so they do not get reinforced
            if mask is not None:
                log_p_nodes[-1][mask] = 0
            assert (log_p_nodes[-1] > -1000).data.all(), "Log probs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        ll_nodes = torch.stack(log_p_nodes, dim=-1)
        num_nonzero = torch.count_nonzero(ll_nodes, 1)
        num_nonzero[num_nonzero == 0] = 1
        return ll_nodes.sum(1) / num_nonzero
