import math
import os.path

from torch import nn
from torch.utils.checkpoint import checkpoint

# from utils.functions import sample_many, adapt_multi_nav, load_model
# from utils.beam_search import CachedLookup

from .modules import *
# from utils import load_model


def get_context(embeddings, state):
    """Returns the context per step, optionally for multiple steps at once (for efficient eval of the model)."""

    # Get current node index
    current_node = state.get_current_idx()

    # Get dimensions
    batch_size, num_steps = current_node.size()
    emb_dim = embeddings.size(-1)

    # Get embedding of current node
    last_node_embed = torch.gather(
        embeddings,
        1,
        current_node.contiguous().view(batch_size, num_steps, 1).expand(batch_size, num_steps, emb_dim)
    ).view(batch_size, num_steps, emb_dim)

    # Get remaining time
    remaining_time = state.get_remaining_length()[..., None]

    # Get current coordinates
    current_coords = state.get_current_coords()

    # Get distance to obstacles
    dist2obs = state.get_dist2obs()

    # Return context (concatenate previously mentioned data)
    return torch.cat((last_node_embed, remaining_time, current_coords, dist2obs), -1)


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
                 mask_inner=True,
                 mask_logits=True,
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

        # Mask parameters
        self.mask_inner = mask_inner                     # Mask inner MHA values while decoding
        self.mask_logits = mask_logits                   # Mask logit values while decoding

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

        # Decoder embeddings for MHA: glimpse key, glimpse value, logit key (so 3 * embed_dim)
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Decoder embedding for context
        self.project_fixed_obs = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embed_dim, bias=False)

        # Projection for the result of inner MHA (num_heads * val_dim == embed_dim, so input is embed_dim)
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)

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

        # Initialize fixed embeddings (computed only during the first iteration)
        self.fixed_embeddings = None

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
            self.fixed_embeddings = self.precompute(embeddings, obs)
        assert self.graph_embeddings is not None, "Graph embeddings are missing, make sure you use the encoder"

        # Transformer decoder
        logits_node, selected_node, logits_direction, selected_direction = self.decoder(self.fixed_embeddings, state)

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

    def inner(self, inputs, embeddings):
        """Transformer Decoder. Contrary to the Encoder, the Decoder is iteratively executed once per time step."""

        # Initialize output probabilities and chosen indexes
        out_nodes, out_directions, sequences = [[[] for _ in range(self.num_agents)] for _ in range(3)]

        # Initialize problem state
        states = [self.problem.make_state(inputs) for _ in range(self.num_agents)]

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        obs = inputs['obs'] if 'obs' in inputs else None
        fixed = self.precompute(embeddings, obs)

        # Batch dimension
        batch_size = states[0].ids.size(0)

        # Perform decoding steps
        i = 0
        while not (self.shrink_size is None and all([state.all_finished() for state in states]) and self.agent_id == 0):

            if self.shrink_size is not None:
                for s, state in enumerate(states):
                    unfinished = torch.nonzero(state.get_finished() == 0)
                    if len(unfinished) == 0:
                        break
                    unfinished = unfinished[:, 0]
                    # Check if we can shrink by at least shrink_size and if this leaves at least 16
                    # (otherwise batch norm will not work well, and it is inefficient anyway)
                    if 16 <= len(unfinished) <= state.ids.size(0) - self.shrink_size:
                        # Filter states
                        states[s] = state[unfinished]
                        fixed = fixed[unfinished]

            # Get next state
            state = states[self.agent_id]

            # Share info between neighbors
            other_states = [s for j, s in enumerate(states) if j != self.agent_id]  # All states except current
            state = state.update_visited(other_states, limit=self.info_th)

            # Transformer decoder
            logits_node, selected_node, logits_direction, selected_direction = self.decoder(fixed, state)

            # Update state
            states[self.agent_id] = state.update(selected_node, selected_direction)
            self.agent_id = self.agent_id + 1 if self.agent_id < self.num_agents - 1 else 0

            # Now make log_p and selected desired output size by 'unshrinking'
            if self.shrink_size is not None and state.ids.size(0) < batch_size:
                logits_node_, selected_node_ = logits_node, selected_node
                logits_node = logits_node_.new_zeros(batch_size, *logits_node_.size()[1:])
                logits_node[state.ids[:, 0]] = logits_node_
                selected_node = selected_node_.new_zeros(batch_size)
                selected_node[state.ids[:, 0]] = selected_node_

            # Collect output of step
            out_nodes[self.agent_id].append(logits_node[:, 0, :])
            out_directions[self.agent_id].append(logits_direction[:, 0, :])
            sequences[self.agent_id].append(torch.stack((selected_node, selected_direction), dim=-1))
            i += 1

        # Collected lists, return Tensors (batch_size x length_tour x num_agents x num_actions)
        out_nodes = torch.stack([torch.stack(output, dim=-1) for output in out_nodes], 1).permute(0, 3, 1, 2)
        out_directions = torch.stack([torch.stack(output, dim=-1) for output in out_directions], 1).permute(0, 3, 1, 2)
        sequences = torch.stack([torch.stack(sequence, dim=-1) for sequence in sequences], 1).permute(0, 3, 1, 2)
        return out_nodes, out_directions, sequences, states

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

    def precompute(self, embeddings, obs=None, num_steps=1):
        """Precompute Encoder embeddings."""

        # Pre-trained (2-step) Transformer encoder precompute
        if self.two_step:
            fixed = self.base_route_model.precompute(embeddings, obs, map_info=(self.patch_size, self.map_size))
            return fixed

        # Embeddings for obstacles
        if self.max_obs:
            embeddings, obs_embeddings = embeddings
            obs_embed = obs_embeddings.mean(1)  # TODO: ??? should be obs_embeddings.mean(1)
            obs_embed = self.project_fixed_obs(obs_embed)[:, None, :]
            obs_map, obs_grid = create_obs_map(obs, self.patch_size, self.map_size)
        else:
            obs_embed, obs_map, obs_grid = None, None, None
        obs_data = (obs_embed, obs_map, obs_grid)

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        # The projection of the node embeddings for the attention is calculated once up front
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            make_heads(self.num_heads, glimpse_key_fixed, num_steps),
            make_heads(self.num_heads, glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data, *obs_data)

    def precompute_fixed(self, inputs):
        """
        Use a CachedLookup such that if we repeatedly index this object with the same index we only need to do the
        lookup once... this is the case if all elements in the batch have maximum batch size.
        """
        if self.max_obs:
            h1, _, h2, _ = self.embedder(self._init_embed(inputs), inputs['obs'])
            embeddings = (h1, h2)
        else:
            embeddings = self.embedder(self._init_embed(inputs))[0]
        return CachedLookup(self._precompute(embeddings, inputs['obs'] if 'obs' in inputs else None))

    def decoder(self, fixed, state):

        # Pre-trained (2-step) Transformer decoder
        # if self.two_step:
        #     logits_node, selected_node = self.base_route_model.decoder(fixed, state)
        #     selected_direction, logits_direction = self.predict_direction(state, selected_node, fixed)
        #     return logits_node, selected_node, logits_direction, selected_direction

        # Predict probabilities for each node
        logits_node, mask = self.get_logits(fixed, state)
        probs_node = logits_node.exp()
        logits_node_no_inf = logits_node.clone()
        logits_node_no_inf[logits_node_no_inf == -math.inf] = 0

        # Select the indices of the next nodes in the sequences, result (batch_size) long
        selected_node = self.select_node(probs_node[:, 0], mask[:, 0])

        # Predict next action (direction from current position to next node to visit)
        selected_direction, logits_direction = self.predict_direction(state, selected_node, fixed)
        return logits_node, selected_node, logits_direction, selected_direction

    def get_logits(self, fixed, state, normalize=True):
        """Predict node probabilities."""

        # Compute query = context node embedding
        step_context = get_context(fixed.node_embeddings, state)
        query = fixed.context_node_projected + self.project_step_context(step_context) + fixed.obs_embed

        # Get keys and values for the decoder
        glimpse_k, glimpse_v, logit_k = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

        # Compute the mask
        mask = state.get_mask()

        # Compute logits (non-normalized log_p)
        logits = self.mha_decoder(query, glimpse_k, glimpse_v, logit_k, mask, normalize=normalize)
        return logits, mask

    def mha_decoder(self, query, glimpse_k, glimpse_v, logit_k, mask, normalize=True):
        """Multi-Head Attention (MHA) mechanism."""

        # Dimensions
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.num_heads

        # Compute the glimpse, rearrange dimensions: (num_heads, batch_size, num_steps, 1, key_size)
        glimpse_q = query.view(batch_size, num_steps, self.num_heads, 1, key_size).permute(2, 0, 1, 3, 4)
        glimpse_k = glimpse_k.transpose(-2, -1)

        # Batch matrix multiplication to compute compatibilities (num_heads, batch_size, num_steps, num_nodes)
        compatibility = torch.matmul(glimpse_q, glimpse_k) / torch.sqrt(torch.tensor(glimpse_q.size(-1)))

        # Ban nodes prohibited by the mask
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf
        compatibility = torch.softmax(compatibility, dim=-1)

        # Batch matrix multiplication to compute heads (num_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(compatibility, glimpse_v)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embed_dim)
        final_q = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.num_heads * val_size)
        )

        # Batch matrix multiplication to compute logits (batch_size, num_steps, num_nodes) -> logits = 'compatibility'
        logit_k = logit_k.transpose(-2, -1)
        logits = torch.matmul(final_q, logit_k).squeeze(-2) / torch.sqrt(torch.tensor(final_q.size(-1)))

        # From logits compute the log probabilities by clipping, masking and normalizing (softmax)
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf
        if normalize:
            logits = torch.log_softmax(logits / self.temp, dim=-1)
        assert not torch.isnan(logits).any(), "NaNs found in logits"
        return logits

    def predict_direction(self, state, next_node, fixed):
        """Direction/Angle prediction."""

        # Get current position
        position = state.get_current_coords()

        # Get next selected goal
        goal = state.get_coords(next_node[:, None])[:, 0]

        # Get local maps
        maps = create_local_maps(position, fixed.obs_map, fixed.obs_grid, goal, self.patch_size, self.map_size)

        # Apply prediction layers
        policy = self.path_prediction(maps)

        # Ban prohibited actions and normalize with softmax
        policy[state.get_mask_actions(policy)[:, 0]] = -math.inf
        prob = torch.softmax(policy, -1)

        # Get action
        if self.decode_type == "greedy":
            _, action = prob.max(1)
        elif self.decode_type == "sampling":
            action = prob.multinomial(1)[:, 0]
        else:
            assert False, "Unknown decode type"

        # Get log probabilities
        log_prob = torch.log_softmax(policy, -1)

        # Return action and log_probabilities
        return action, log_prob[:, None]

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
        assert torch.eq(probs, probs).all(), "Probs should not contain any nans"

        # ArgMax (Exploitation)
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            if mask is not None:
                assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                    "Decode greedy: infeasible action has maximum probability"

        # Sample (Exploration)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Sampling can fail due to GPU bug: https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            if mask is not None:
                while mask.gather(1, selected.unsqueeze(-1)).data.any():
                    print('Sampled bad values, resampling!')
                    selected = probs.multinomial(1).squeeze(1)
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
