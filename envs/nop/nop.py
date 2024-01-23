import gym
import torch


class Regions:

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

        # Initial depot
        self.depot_ini = depot_ini
        assert len(depot_ini.shape) == 3, \
            "Depot's coordinates should have 3 dimensions: batch_size, num_regions, num_dims"

        # End depot
        self.depot_end = depot_end
        assert len(depot_end.shape) == 3, \
            "Depot's coordinates should have 3 dimensions: batch_size, num_regions, num_dims"

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
        return self.depot_ini is None

    def is_depot_end(self):
        return self.depot_end is None

    def get_depot_ini(self):
        return self.depot_ini

    def get_depot_end(self):
        return self.depot_end

    def get_regions(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((self.depot_ini, self.regions), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.regions, self.depot_end), axis=-2)
        else:
            return torch.cat((self.depot_ini, self.regions, self.depot_end), axis=-2)

    def get_prizes(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.prizes
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((torch.zeros_like(self.depot_ini), self.prizes), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.prizes, torch.zeros_like(self.depot_end)), axis=-2)
        else:
            return torch.cat((torch.zeros_like(self.depot_ini), self.prizes, torch.zeros_like(self.depot_end)), axis=-2)

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
        batch_ids = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)[:, None]
        return self.get_regions()[batch_ids, idx]

    def get_prize_by_index(self, idx):
        batch_idx = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)[:, None]
        return self.get_prizes()[batch_idx, idx]

    def get_end_idx(self):
        return self.get_num_regions() - 1 if self.is_depot_end() else 0


class NopEnv(gym.Env):

    def __init__(self, inputs, time_step=2e-2):

        # Regions
        self.regions = Regions(
            regions=inputs['loc'],
            prizes=inputs['prize'],
            depot_ini=inputs['depot'],
            depot_end=inputs['depot2'] if 'depot2' in inputs else inputs['depot']
        )

        # Dimensions
        batch_size = self.regions.get_batch_size()
        num_regions_depots = self.regions.get_num_regions()

        # Device
        device = self.regions.device

        # Maximum allowed length
        self.max_length = inputs['max_length'][:, None]

        # Obstacles
        self.obs = inputs['obs'] if 'obs' in inputs else None
        self.obs_bumped = torch.zeros(batch_size, dtype=torch.int64, device=device) if self.obs is not None else None

        # Mask of visited regions
        self.visited = torch.zeros(
            size=(batch_size, 1, num_regions_depots), dtype=torch.uint8, device=device
        )
        self.visited[..., 0] = 1
        self.visited[self.regions.get_regions()[:, None, :, 0] < 0] = 1  # Block dummy (negative) regions if any

        # Count iterations
        self.i = torch.zeros(1, dtype=torch.int64, device=device)

        # Previously visited node
        self.prev_node = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Previous action
        self.prev_action = torch.zeros(batch_size, 1, dtype=torch.long, device=device)

        # Position
        self.position = self.regions.depot_ini[:, None, :]

        # Time step
        self.time_step = time_step

        # Traveled length
        self.length = torch.zeros(batch_size, 1, device=device)

        # Reward
        self.reward = torch.zeros(batch_size, 1, device=device)

        # Misc
        self.is_traveling = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def step(self, action):
        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Not finished flag (terminal states are: time out, reach end depot, bump obstacle)
        nf = ~self.finished()[:, 0]

        # Action = (Next node to visit, Next action/direction to follow)
        next_node_idx = action[0][..., None]
        next_action = action[1][..., None]

        # Update next position
        polar = torch.polar(
            torch.zeros_like(self.position[:, 0, 0]) + torch.tensor(self.time_step),
            next_action[:, 0] * 2 * torch.pi / 4
        )
        new_position = self.position.clone()
        new_position[nf] = self.position[nf] + torch.stack((polar.real, polar.imag), -1)[:, None][nf]

        # Update length of route
        self.length = self.length + (new_position - self.position).norm(p=2, dim=-1)
        assert not torch.isinf(self.length).any(), "Length is inf"

        # Distance to next node
        next_node_coords = self.regions.get_regions_by_index(next_node_idx)
        dist2next = (new_position - next_node_coords).norm(p=2, dim=-1)[:, 0]

        # Check whether the agent has just arrived to next node or it is still traveling
        self.is_traveling[dist2next > self.time_step] = True    # Traveling
        self.is_traveling[dist2next <= self.time_step] = False  # Not traveling
        self.is_traveling[~nf] = False                          # Once on terminal state, do not travel

        # Mask visited regions
        self.visited = self.visited.scatter(-1, next_node_idx[:, :, None], 1)

        # Reward: visiting next node
        condition = torch.logical_and(
            torch.logical_and(~self.is_traveling, nf),          # Agent is not traveling and has not finished yet
            dist2next <= self.time_step                         # Next node is visited
        )
        self.reward[condition] += 60 * self.regions.get_prize_by_index(next_node_idx)[condition] / (
                (self.regions.get_regions()[..., 0] >= 0).sum(1) - 2
        )[condition, None]

        # Penalty: distance to next node
        self.reward[nf] -= dist2next[:, None][nf] * 0.3

        # Reward: reaching end depot within time limit
        end_idx = self.regions.get_end_idx()
        dist2end = (new_position - self.regions.get_regions_by_index(end_idx)).norm(p=2, dim=-1)
        condition = torch.logical_and(
            torch.logical_and(
                torch.eq(next_node_idx[:, 0], end_idx),         # Next node == end depot
                (dist2end <= self.time_step)[:, 0]              # End depot visited
            ),
            nf                                                  # Not finished
        )
        self.reward[condition] = self.reward[condition] + 20

        # Penalty: not reaching end depot within time limit
        condition = torch.logical_and(
            torch.logical_and(
                (self.max_length - self.length < 0)[:, 0],      # Time limit is surpassed
                (dist2end > self.time_step)[:, 0]               # End depot not visited
            ),
            nf                                                  # Not finished
        )
        self.reward[condition] = self.reward[condition] - 5

        # Penalty: bumping into obstacles
        if self.obs is not None:
            self.obs_bumped[
                torch.ge(
                    self.obs[..., 2],
                    (new_position - self.obs[..., :2]).norm(p=2, dim=-1)
                ).any(dim=-1)
            ] = True                                            # obs_bumped = 1 if agent is inside obstacle
            condition = torch.logical_and(
                self.obs_bumped,                                # Bumped into obstacle
                nf                                              # Not finished
            )
            self.reward[condition] = self.reward[condition] - 5

        # Update state
        self.position = new_position
        self.prev_node = next_node_idx
        self.prev_action = next_action

        state = {
            'position': self.position,
        }

        return state, self.reward, self.finished().all(), None

    def finished(self):
        # All must be returned to depot (and at least 1 step since at start). More efficient than checking the mask
        end_idx = self.regions.get_end_idx()
        return torch.logical_and(
            torch.tensor(self.i.item() > 0, device=self.prev_node.device),
            torch.logical_or(
                torch.logical_and(
                    torch.eq(self.prev_node, end_idx),
                    (self.position - self.regions.get_regions_by_index(end_idx)).norm(p=2, dim=-1) <= self.time_step
                ),
                torch.logical_or(self.obs_bumped[:, None], self.get_remaining_length() < 0)
            )
        )

    def reset(self):
        return

    def render(self):
        return

    def get_remaining_length(self):
        return self.max_length - self.length
