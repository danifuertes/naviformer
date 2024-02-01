import gym
import torch
from torch.utils.data import DataLoader

from envs.nop.nop_utils import NopDataset
from utils import move_to


class NopRegions:

    def __init__(self, regions, prizes=None, depot_ini=None, depot_end=None, min_value=0, max_value=1):

        # Coordinates
        self.regions = regions

        # Dimensions
        assert len(regions.shape) == 3, \
            "Regions' coordinates should have 3 dimensions: batch_size, num_regions, num_dims"
        self.batch_size, self.num_regions, self.num_dims = regions.shape

        # Device
        self.device = regions.device

        # Prizes
        self.prizes = prizes if prizes is not None else torch.ones(
            size=(self.batch_size, self.num_regions), dtype=torch.long, device=self.device
        )

        # Depots
        self.depot_ini = depot_ini
        self.depot_end = depot_end

        # Normalization
        self.min_value = min_value
        self.max_value = max_value

    def __len__(self):
        return self.num_regions

    def normalize(self):
        self.regions = (self.regions - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, min_value=0, max_value=1):
        self.regions = self.regions * (max_value - min_value) + min_value

    def is_depot_ini(self):
        return self.depot_ini is not None

    def is_depot_end(self):
        return self.depot_end is not None

    def get_depot_ini(self):
        return self.depot_ini

    def get_depot_end(self):
        return self.depot_end if self.is_depot_end() else self.depot_ini

    def get_regions(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((self.depot_ini[:, None], self.regions), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.regions, self.depot_end[:, None]), axis=-2)
        else:
            return torch.cat((self.depot_ini[:, None], self.regions, self.depot_end[:, None]), axis=-2)

    def get_prizes(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.prizes
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((torch.zeros_like(self.depot_ini[:, None]), self.prizes), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.prizes, torch.zeros_like(self.depot_end[:, None])), axis=-2)
        else:
            return torch.cat(
                (
                    torch.zeros_like(self.depot_ini[:, 0, None]),
                    self.prizes,
                    torch.zeros_like(self.depot_end[:, 0, None])
                ), axis=-1
            )

    def get_batch_size(self):
        return self.regions.shape[0]

    def get_num_regions(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1]
        elif self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1] + 1
        elif not self.is_depot_ini() and self.is_depot_end():
            return self.regions.shape[1] + 1
        else:
            return self.regions.shape[1] + 2

    def get_num_dims(self):
        return self.regions.shape[2]

    def get_regions_by_index(self, idx):
        batch_ids = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
        return self.get_regions()[batch_ids, idx]

    def get_prize_by_index(self, idx):
        batch_idx = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
        return self.get_prizes()[batch_idx, idx]

    def get_end_idx(self):
        return self.get_num_regions() - 1 if self.is_depot_end() else 0


class NopEnv(gym.Env):

    def __init__(self, batch_size=1024, num_workers=16, device=None, time_step=2e-2, baseline=None, num_actions=4,
                 *args, **kwargs):

        # Number of actions / directions
        self.num_actions = num_actions

        # Device
        self.device = device

        # Problem name
        self.name = 'nop'

        # Baseline
        self.baseline = baseline

        # Data
        dataset = NopDataset(*args, **kwargs)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # Steps (number of batches per epoch)
        self.num_steps = len(dataset) // batch_size

        # Initialize scenario
        self.regions = None
        self.max_length = None
        self.obs = None

        # Initialize state
        self.position = None
        self.visited = None
        self.length = None
        self.is_traveling = None
        self.obs_bumped = None
        self.i = torch.zeros(1, dtype=torch.int64, device=device)

        # Initialize actions
        self.prev_node = None
        self.prev_action = None

        # Time step
        self.time_step = time_step

    def step(self, action, fixed_data=None):

        # Not finished flag (terminal states are: time out, reach end depot, bump obstacle)
        nf = ~self.finished()

        # Action = (Next node to visit, Next action/direction to follow)
        next_node_idx = action[..., 0]
        next_action = action[..., 1]

        # Update next position
        polar = torch.polar(
            torch.zeros_like(self.position[:, 0]) + torch.tensor(self.time_step),
            next_action * 2 * torch.pi / 4
        )
        new_position = self.position.clone()
        new_position[nf] = self.position[nf] + torch.stack((polar.real, polar.imag), -1)[nf]

        # Update length of route
        self.length = self.length + (new_position - self.position).norm(p=2, dim=-1)
        assert not torch.isinf(self.length).any(), "Length is inf"

        # Distance to next node
        next_node_coords = self.regions.get_regions_by_index(next_node_idx)
        dist2next = (new_position - next_node_coords).norm(p=2, dim=-1)

        # Check whether the agent has just arrived to next node or it is still traveling
        self.is_traveling[dist2next > self.time_step] = True    # Traveling
        self.is_traveling[dist2next <= self.time_step] = False  # Not traveling
        self.is_traveling[~nf] = False                          # Once on terminal state, do not travel

        # Mask visited regions
        self.visited = self.visited.scatter(-1, next_node_idx[..., None], 1)

        # Reward
        reward = torch.zeros(nf.shape[0], device=self.device)

        # Reward: visiting next node
        condition = torch.logical_and(
            torch.logical_and(~self.is_traveling, nf),          # Agent is not traveling and has not finished yet
            dist2next <= self.time_step                         # Next node is visited
        )
        reward[condition] += 60 * self.regions.get_prize_by_index(next_node_idx)[condition] / (
                (self.regions.get_regions()[..., 0] >= 0).sum(1) - 2
        )[condition]

        # Penalty: distance to next node
        reward[nf] -= dist2next[nf] * 0.3

        # Reward: reaching end depot within time limit
        end_idx = self.regions.get_end_idx()
        dist2end = (new_position - self.regions.get_regions_by_index(end_idx)).norm(p=2, dim=-1)
        condition = torch.logical_and(
            torch.logical_and(
                torch.eq(next_node_idx, end_idx),               # Next node == end depot
                (dist2end <= self.time_step)                    # End depot visited
            ),
            nf                                                  # Not finished
        )
        reward[condition] = reward[condition] + 20

        # Penalty: not reaching end depot within time limit
        condition = torch.logical_and(
            torch.logical_and(
                (self.max_length - self.length < 0),            # Time limit is surpassed
                (dist2end > self.time_step)                     # End depot not visited
            ),
            nf                                                  # Not finished
        )
        reward[condition] = reward[condition] - 5

        # Penalty: bumping into obstacles
        if self.obs is not None:
            self.obs_bumped[
                torch.ge(
                    self.obs[..., 2],
                    (new_position[:, None] - self.obs[..., :2]).norm(p=2, dim=-1)
                ).any(dim=-1)
            ] = True                                            # obs_bumped = 1 if agent is inside obstacle
            condition = torch.logical_and(
                self.obs_bumped,                                # Bumped into obstacle
                nf                                              # Not finished
            )
            reward[condition] = reward[condition] - 5

        # Update state
        self.position = new_position
        self.prev_node = next_node_idx
        self.prev_action = next_action

        # Return state, reward, done, info
        self.i = self.i + 1
        state = {
            'regions': self.regions.get_regions(),
            'position': self.position,
            'length': self.length,
            'dist2obs': self.get_dist2obs(),
            'mask_nodes': self.get_mask_nodes(),
            'mask_actions': self.get_mask_actions(),
            'prev_node': self.prev_node,
            'inputs': fixed_data
        }
        return state, -reward, self.finished().all().item(), None  # Negative reward since torch always minimizes

    def reset(self, **kwargs):

        # Load data on device
        batch = next(iter(self.dataloader))
        batch = move_to(batch, self.device)

        # Return initial state
        state = self.get_state_from_batch(batch)
        return state

    def get_state_from_batch(self, batch):

        # Regions
        self.regions = NopRegions(
            regions=batch['loc'],
            prizes=batch['prize'],
            depot_ini=batch['depot'],
            depot_end=batch['depot2'] if 'depot2' in batch else batch['depot']
        )

        # Dimensions
        batch_size = self.regions.get_batch_size()
        num_regions = self.regions.get_num_regions()

        # Maximum allowed length
        self.max_length = batch['max_length']

        # Obstacles
        self.obs = batch['obs'] if 'obs' in batch else None
        self.obs_bumped = torch.zeros(batch_size, dtype=torch.int64, device=self.device) if self.obs is not None else None

        # Mask of visited regions
        self.visited = torch.zeros(size=(batch_size, num_regions), dtype=torch.uint8, device=self.device)
        self.visited[..., 0] = 1
        self.visited[self.regions.get_regions()[..., 0] < 0] = 1  # Block dummy (negative) regions if any

        # Count iterations
        self.i = 0

        # Previously visited node
        self.prev_node = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Previous action
        self.prev_action = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Position
        self.position = self.regions.depot_ini

        # Traveled length
        self.length = torch.zeros(batch_size, device=self.device)

        # Misc
        self.is_traveling = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        # Return state
        state = {
            'regions': self.regions.get_regions(),
            'position': self.position,
            'length': self.length,
            'dist2obs': self.get_dist2obs(),
            'mask_nodes': self.get_mask_nodes(),
            'mask_actions': self.get_mask_actions(),
            'prev_node': self.prev_node,
            'inputs': batch
        }
        return state

    def finished(self):

        # Check conditions for finishing episode
        return torch.logical_and(
            torch.tensor(self.i > 0, device=self.device),                                               # Not first step
            torch.logical_or(
                torch.logical_and(
                    torch.eq(self.prev_node, self.regions.get_end_idx()),                               # Going to end
                    (self.position - self.regions.get_depot_end()).norm(p=2, dim=-1) <= self.time_step  # On end depot
                ),
                torch.logical_or(
                    self.obs_bumped,                                                                    # No bumping
                    self.get_remaining_length() < 0                                                     # Still on time
                )
            )
        )

    def get_remaining_length(self):
        return self.max_length - self.length

    def get_mask_nodes(self):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions, depends on already visited and remaining
        capacity. 0 = feasible, 1 = infeasible. Forbids to visit depot twice in a row, unless all nodes have been
        visited.
        """
        batch_ids = torch.arange(self.regions.get_batch_size(), dtype=torch.int64, device=self.device)
        end_idx = self.regions.get_end_idx()

        # Define mask (with visited nodes)
        visited_ = self.visited.to(torch.bool)
        mask = visited_ | visited_[..., end_idx, None]

        # Block initial depot
        mask[..., 0] = 1

        # While traveling, do not change agent's mind
        mask[self.is_traveling] = 1  # Comment this line to allow changing agent's mind (careful with lengths then)
        mask[batch_ids[self.is_traveling], self.prev_node[self.is_traveling]] = 0

        # End depot can always be visited, but once visited, cannot leave the place
        mask[~self.is_traveling, end_idx] = 0
        finished = self.finished()
        mask[finished] = 1
        mask[finished, end_idx] = 0  # Always allow visiting end depot to prevent running out of nodes to choose
        return mask

    def get_mask_actions(self):

        # Initialize mask
        mask = torch.zeros((self.regions.get_batch_size(), self.num_actions), dtype=torch.bool, device=self.device)

        # Ban actions (directions) that lead out of the map | TODO: adapt for more than 4 actions
        mask[self.position[..., 0] + self.time_step > 1, 0] = 1
        mask[self.position[..., 1] + self.time_step > 1, 1] = 1
        mask[self.position[..., 0] - self.time_step < 0, 2] = 1
        mask[self.position[..., 1] - self.time_step < 0, 3] = 1

        # No more restrictions are required during the first step, so return the mask
        if self.i < 1:
            return mask.bool()

        # Avoid performing the action opposite to that performed before
        banned_actions = self.prev_action[..., None] + self.num_actions / 2
        banned_actions[banned_actions > self.num_actions - 1] -= self.num_actions
        mask = mask.scatter(-1, banned_actions.long(), 1)
        return mask.bool()

    def get_dist2obs(self):
        return (self.position[:, None] - self.obs[..., :2]).norm(dim=-1) - self.obs[..., 2]

    def render(self):
        return
