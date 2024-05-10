import torch
from torch.utils.data import DataLoader
from typing import NamedTuple, Any, Tuple

from envs.nop.nop_utils import NopDataset


class NopEnv:
    """
    Navigation Orienteering Problem (NOP) Environment class.

    Args:
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        device (torch.device): Torch device.
        num_dirs (int): Number of possible actions.
        time_step (float): Time step.
    """

    def __init__(self,
                 batch_size: int = 1024,
                 num_workers: int = 0,
                 device: torch.types.Device = None,
                 num_dirs: int = 4,
                 time_step: float = 2e-2,
                 *args, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
            device (torch.device): Torch device.
            num_dirs (int): Number of possible actions.
            time_step (float): Time step.
        """

        # Name of the env
        self.name = "nop"

        # Data loader
        dataset = NopDataset(*args, **kwargs)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # Number of steps
        self.num_steps = len(dataset) // batch_size

        # Device
        self.device = device

        # Number of actions
        self.num_dirs = num_dirs

        # Time step
        self.time_step = time_step

    def get_state(self, batch: dict | torch.Tensor) -> Any:
        """
        Generate a state from a given batch (scenario).

        Args:
            batch (dict or torch.Tensor): batch of data.
        """
        return NopState.initialize(batch, num_dirs=self.num_dirs, time_step=self.time_step)


class NopState(NamedTuple):
    """Navigation Orienteering Problem (NOP) State class. Contains non-static info about the episode"""

    # Scenario
    regions: torch.Tensor           # Visitable regions
    prizes: torch.Tensor            # Prize for visiting each region
    depot_ini: torch.Tensor         # Initial depot
    depot_end: torch.Tensor         # End depot
    max_length: torch.Tensor        # Time/distance limit
    obs: torch.Tensor               # Obstacles
    obs_bumped: torch.Tensor        # Indicates if agent has bumped an obstacle

    # State
    visited: torch.Tensor           # Keeps track of nodes that have been visited
    prev_node: torch.Tensor         # Previous node selected to visit
    prev_action: torch.Tensor       # Previous action taken
    position: torch.Tensor          # Current agent position
    length: torch.Tensor            # Time/distance of the travel
    is_traveling: torch.Tensor      # Indicates if agent has reached a region or is traveling

    # Misc
    i: torch.Tensor                 # Keeps track of step
    device: torch.device            # Torch device (gpu or cpu)
    num_dirs: int                   # Number of directions to follow
    time_step: float                # Duration/length of a step
    reward: torch.Tensor            # Last collected reward
    done: bool                      # Terminal state for every element of the batch
    min_value: float                # Minimum normalization value
    max_value: float                # Maximum normalization value

    def __getitem__(self, key):
        """
        Get item(s) from the state using indexing.

        Args:
            key: Index or slice to retrieve.

        Returns:
            NopState: New state containing selected elements.
        """
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            i=self.i[key],
            regions=self.regions[key],
            prizes=self.prizes[key],
            depot_ini=self.depot_ini[key],
            depot_end=self.depot_end[key],
            max_length=self.max_length[key],
            obs=self.obs[key],
            obs_bumped=self.obs_bumped[key],
            visited=self.visited[key],
            prev_node=self.prev_node[key],
            prev_action=self.prev_action[key],
            position=self.position[key],
            length=self.length[key],
            is_traveling=self.is_traveling[key],
            reward=self.reward[key],
        )

    @staticmethod
    def initialize(batch: dict | torch.Tensor,
                   num_dirs: int = 4,
                   time_step: float = 2e-2,
                   min_value: int = 0,
                   max_value: int = 1) -> Any:
        """
        Initialize the state.

        Args:
            batch (dict or torch.Tensor): Input batch.
            num_dirs (int): Number of possible directions.
            time_step (float): Time step.
            min_value (float): Minimum value for normalization.
            max_value (float): Maximum value for normalization.

        Returns:
            NopState: Initialized state.
        """

        # Device
        device = batch[list(batch.keys())[0]].device

        # Count iterations
        i = torch.zeros(1, device=device)

        # Regions
        regions = batch['loc']

        # Prizes
        prizes = batch['prize']

        # Depots
        depot_ini = batch['depot_ini']
        depot_end = batch['depot_end']

        # Maximum allowed length
        max_length = batch['max_length']

        # Dimensions
        all_regions = torch.cat((depot_ini[:, None], regions, depot_end[:, None]), axis=-2)
        batch_size, num_regions, _ = all_regions.shape

        # Obstacles
        obs = batch['obs'] if 'obs' in batch else None
        obs_bumped = torch.zeros(batch_size, dtype=torch.int64, device=device) if obs is not None else None

        # Mask of visited regions
        visited = torch.zeros(size=(batch_size, num_regions), dtype=torch.uint8, device=device)
        visited[..., 0] = 1
        visited[all_regions[..., 0] < 0] = 1  # Block dummy (negative) regions if any

        # Previously visited node
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Previous action
        prev_action = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Traveled length
        length = torch.zeros(batch_size, device=device)

        # Traveling info
        is_traveling = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Reward and done
        reward = torch.zeros(1, device=device)
        done = False

        # Create State
        return NopState(
            i=i,
            device=device,
            regions=regions,
            prizes=prizes,
            depot_ini=depot_ini,
            depot_end=depot_end,
            max_length=max_length,
            obs=obs,
            obs_bumped=obs_bumped,
            visited=visited,
            prev_node=prev_node,
            prev_action=prev_action,
            position=depot_ini,
            length=length,
            is_traveling=is_traveling,
            num_dirs=num_dirs,
            time_step=time_step,
            min_value=min_value,
            max_value=max_value,
            reward=reward,
            done=done
        )

    def step(self, action: torch.Tensor, *args, **kwargs) -> Any:
        """
        Take a step in the environment.

        Args:
            action (torch.Tensor): Batch of actions to take.

        Returns:
            NopState: Updated state.
        """

        # Not finished flag (terminal states are: time out, reach end depot, bump obstacle)
        nf = ~self.finished()

        # Action = (Next node to visit, Next action/direction to follow)
        next_node_idx = action[..., 0]
        next_action = action[..., 1]

        # Update next position
        polar = torch.polar(
            torch.zeros_like(self.position[:, 0]) + torch.tensor(self.time_step),
            next_action * 2 * torch.pi / self.num_dirs
        )
        new_position = self.position.clone()
        new_position[nf] = self.position[nf] + torch.stack((polar.real, polar.imag), -1)[nf]

        # Update length of route
        length = self.length + (new_position - self.position).norm(p=2, dim=-1)  # self.length + self.time_step

        # Distance to next node
        next_node_coords = self.get_regions_by_index(next_node_idx)
        dist2next = (new_position - next_node_coords).norm(p=2, dim=-1)

        # Check whether the agent has just arrived to next node or it is still traveling
        is_traveling = self.is_traveling
        is_traveling[dist2next > self.time_step] = True  # Traveling
        is_traveling[dist2next <= self.time_step] = False  # Not traveling
        is_traveling[~nf] = False  # Once on terminal state, do not travel

        # Mask visited regions
        visited = self.visited.scatter(-1, next_node_idx[..., None], 1)

        # Reward
        reward = torch.zeros(nf.shape[0], device=self.device)

        # Reward: visiting next node
        condition = torch.logical_and(
            torch.logical_and(~is_traveling, nf),  # Agent is not traveling and has not finished yet
            dist2next <= self.time_step  # Next node is visited
        )
        reward[condition] += 60 * self.get_prize_by_index(next_node_idx)[condition] / (
                (self.get_regions()[..., 0] >= 0).sum(1) - 2
        )[condition]

        # Penalty: distance to next node
        reward[nf] -= dist2next[nf] * 0.3

        # Reward: reaching end depot within time limit
        end_idx = self.get_end_idx()
        dist2end = (new_position - self.get_regions_by_index(end_idx)).norm(p=2, dim=-1)
        condition = torch.logical_and(
            torch.logical_and(
                torch.eq(next_node_idx, end_idx),  # Next node == end depot
                (dist2end <= self.time_step)  # End depot visited
            ),
            nf  # Not finished
        )
        reward[condition] = reward[condition] + 20

        # Penalty: not reaching end depot within time limit
        condition = torch.logical_and(
            torch.logical_and(
                (self.max_length - length < 0),  # Time limit is surpassed
                (dist2end > self.time_step)  # End depot not visited
            ),
            nf  # Not finished
        )
        reward[condition] = reward[condition] - 5

        # Penalty: bumping into obstacles
        obs_bumped = self.obs_bumped
        if self.obs is not None:
            obs_bumped[
                torch.ge(
                    self.obs[..., 2],
                    self.get_dist2obs(position=new_position)
                ).any(dim=-1)
            ] = True  # obs_bumped = 1 if agent is inside obstacle
            condition = torch.logical_and(
                obs_bumped,  # Bumped into obstacle
                nf  # Not finished
            )
            reward[condition] = reward[condition] - 5

        # Update state and return reward and done
        return self._replace(
            i=self.i+1,
            obs_bumped=obs_bumped,
            visited=visited,
            prev_node=next_node_idx,
            prev_action=next_action,
            position=new_position,
            length=length,
            is_traveling=is_traveling,
            reward=-reward,   # Negative reward since torch always minimizes
            done=self.finished().all().item()
        )

    def finished(self) -> torch.Tensor:
        """
        Check if the episode is finished.

        Returns:
            torch.Tensor: Tensor indicating whether the episode is finished for each element of the batch.
        """

        # Check conditions for finishing episode
        return torch.logical_and(
            self.i > 0,  # Not first step
            torch.logical_or(
                torch.logical_and(
                    torch.eq(self.prev_node, self.get_end_idx()),  # Going to end depot
                    (self.position - self.get_depot_end()).norm(p=2, dim=-1) <= self.time_step  # On end depot
                ),
                torch.logical_or(
                    self.obs_bumped,  # No bumping
                    self.get_remaining_length() < 0  # Still on time
                )
            )
        )

    def check_success(self) -> torch.Tensor:
        """
        Check if the episode is successful.

        Returns:
            torch.Tensor: Tensor indicating whether the episode is successful for each element of the batch.
        """
        return torch.logical_and(
            self.finished(),  # Finished
            ~torch.logical_or(
                self.obs_bumped,  # No bumping
                self.get_remaining_length() < 0  # Still on time
            )
        )

    def get_remaining_length(self) -> torch.Tensor:
        """
        Get the remaining length at each moment.

        Returns:
            torch.Tensor: Remaining length for each element of the batch.
        """
        return self.max_length - self.length

    def get_dist2obs(self, position: torch.Tensor = None) -> torch.Tensor:
        """
        Get the distance to obstacles.

        Args:
            position (torch.Tensor): Current position.

        Returns:
            torch.Tensor: Distance to obstacles for each element of the batch.
        """
        if self.obs is None:
            return None
        if position is None:
            position = self.position
        return (position[:, None] - self.obs[..., :2]).norm(p=2, dim=-1)

    def get_features(self) -> Tuple:
        """
        Get the features of the environment.

        Returns:
            tuple: Tuple containing the features.
        """
        return self.prizes,

    def get_mask_nodes(self) -> torch.Tensor:
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions, depends on already visited and remaining
        capacity. 0 = feasible, 1 = infeasible. Forbids to visit depot twice in a row, unless all nodes have been
        visited.

        Returns:
            torch.Tensor: Mask for nodes.
        """
        batch_ids = torch.arange(self.get_batch_size(), dtype=torch.int64, device=self.device)
        end_idx = self.get_end_idx()

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

    def get_mask_dirs(self) -> torch.Tensor:
        """
        Get the mask for directions.

        Returns:
            torch.Tensor: Mask for directions.
        """

        # Initialize mask
        mask = torch.zeros((self.get_batch_size(), self.num_dirs), dtype=torch.bool, device=self.device)

        # Ban actions (directions) that lead out of the map | TODO: adapt for more than 4 dirs
        mask[self.position[..., 0] + self.time_step > 1, 0] = 1
        mask[self.position[..., 1] + self.time_step > 1, 1] = 1
        mask[self.position[..., 0] - self.time_step < 0, 2] = 1
        mask[self.position[..., 1] - self.time_step < 0, 3] = 1

        # No more restrictions are required during the first step, so return the mask
        if self.i < 1:
            return mask.bool()

        # Avoid performing the action opposite to that performed before
        banned_actions = self.prev_action[..., None] + self.num_dirs / 2
        banned_actions[banned_actions > self.num_dirs - 1] -= self.num_dirs
        mask = mask.scatter(-1, banned_actions.long(), 1)
        return mask.bool()

    def is_depot_ini(self) -> bool:
        """
        Check if the initial depot exists.

        Returns:
            bool: True if the initial depot exists, False otherwise.
        """
        return self.depot_ini is not None

    def is_depot_end(self) -> bool:
        """
        Check if the end depot exists.

        Returns:
            bool: True if the end depot exists, False otherwise.
        """
        return self.depot_end is not None

    def get_depot_ini(self) -> torch.Tensor | None:
        """
        Get the initial depot.

        Returns:
            torch.Tensor: Initial depot.
        """
        return self.depot_ini

    def get_depot_end(self):
        """
        Get the end depot.

        Returns:
            torch.Tensor: End depot.
        """
        return self.depot_end if self.is_depot_end() else self.depot_ini

    def get_regions(self) -> torch.Tensor:
        """
        Get the coordinates of the regions.

        Returns:
            torch.Tensor: Regions.
        """
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((self.depot_ini[:, None], self.regions), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.regions, self.depot_end[:, None]), axis=-2)
        else:
            return torch.cat((self.depot_ini[:, None], self.regions, self.depot_end[:, None]), axis=-2)
        
    def get_dist2regions(self, position) -> torch.Tensor:
        """
        Calculates distance from given position (batch_size, 2) to each of the regions (batch_size, num_regions, 2).

        Args:
            position (_type_): coordinates, with shape (batch_size, 2).

        Returns:
            torch.Tensor: calculated distance, with shape (batch_size).
        """
        num_regions = self.get_num_regions()
        return (position.tile(num_regions).reshape(-1, num_regions, 2) - self.get_regions()).norm(p=2, dim=-1)

    def get_prizes(self) -> torch.Tensor:
        """
        Get the prizes.

        Returns:
            torch.Tensor: Prizes.
        """
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

    def get_batch_size(self) -> int:
        """
        Get the batch size.

        Returns:
            int: Batch size.
        """
        return self.regions.shape[0]

    def get_num_regions(self) -> int:
        """
        Get the number of regions.

        Returns:
            int: Number of regions.
        """
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1]
        elif self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1] + 1
        elif not self.is_depot_ini() and self.is_depot_end():
            return self.regions.shape[1] + 1
        else:
            return self.regions.shape[1] + 2

    def get_num_dims(self) -> int:
        """
        Get the number of dimensions.

        Returns:
            int: Number of dimensions.
        """
        return self.regions.shape[2]

    def get_regions_by_index(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get regions by index.

        Args:
            idx (torch.Tensor): Indices.

        Returns:
            torch.Tensor: Regions.
        """
        batch_ids = torch.arange(self.get_batch_size(), dtype=torch.int64, device=self.device)
        return self.get_regions()[batch_ids, idx]

    def get_prize_by_index(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Get prize by index.

        Args:
            idx (torch.Tensor): Indices.

        Returns:
            torch.Tensor: Prize.
        """
        batch_idx = torch.arange(self.get_batch_size(), dtype=torch.int64, device=self.device)
        return self.get_prizes()[batch_idx, idx]

    def get_end_idx(self) -> int:
        """
        Get the index of the end depot.

        Returns:
            int: Index of the end depot.
        """
        return self.get_num_regions() - 1 if self.is_depot_end() else 0

    def normalize(self, regions: torch.Tensor) -> torch.Tensor:
        """
        Normalize regions.

        Args:
            regions (torch.Tensor): Regions to normalize.

        Returns:
            torch.Tensor: Normalized regions.
        """
        return (regions - self.min_value) / (self.max_value - self.min_value)

    @staticmethod
    def denormalize(regions: torch.Tensor, min_value: float = 0, max_value: float = 1) -> torch.Tensor:
        """
        Denormalize regions.

        Args:
            regions (torch.Tensor): Regions to denormalize.
            min_value (float): Minimum value.
            max_value (float): Maximum value.

        Returns:
            torch.Tensor: Denormalized regions.
        """
        return regions * (max_value - min_value) + min_value
