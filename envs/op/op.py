import torch
from torch.utils.data import DataLoader
from typing import NamedTuple, Any, Tuple

from envs.op.op_utils import OpDataset


class OpEnv:
    """
    Orienteering Problem (OP) Environment class.

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
                 eps: float = 0.,
                 *args, **kwargs) -> None:
        """
        Initialize the environment.

        Args:
            batch_size (int): Batch size.
            num_workers (int): Number of workers for data loading.
            device (torch.device): Torch device.
            num_dirs (int): Number of possible actions.
            time_step (float): Time step.
            eps (float): tolerance, useful for 2-step methods. It gives some margin to reach the end depot on time.

        """

        # Name of the env
        self.name = "nop"

        # Data loader
        dataset = OpDataset(*args, **kwargs)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        # Number of steps
        self.num_steps = len(dataset) // batch_size

        # Device
        self.device = device

        # Number of actions
        self.num_dirs = num_dirs

        # Time step
        self.time_step = time_step
        
        # Tolerance
        self.eps = eps

    def get_state(self, batch: dict | torch.Tensor) -> Any:
        """
        Generate a state from a given batch (scenario).

        Args:
            batch (dict or torch.Tensor): batch of data.
        """
        return OpState.initialize(batch, num_dirs=self.num_dirs, time_step=self.time_step, eps=self.eps)


class OpState(NamedTuple):
    """Orienteering Problem (NOP) State class. Contains non-static info about the episode"""

    # Scenario
    regions: torch.Tensor           # Visitable regions
    prizes: torch.Tensor            # Prize for visiting each region
    depot_ini: torch.Tensor         # Initial depot
    depot_end: torch.Tensor         # End depot
    max_length: torch.Tensor        # Time/distance limit
    obs: torch.Tensor               # Obstacles

    # State
    visited: torch.Tensor           # Keeps track of nodes that have been visited
    prev_node: torch.Tensor         # Previous node selected to visit
    position: torch.Tensor          # Current agent position
    length: torch.Tensor            # Time/distance of the travel

    # Misc
    i: torch.Tensor                 # Keeps track of step
    device: torch.device            # Torch device (gpu or cpu)
    num_dirs: int                   # Number of directions to follow
    time_step: float                # Duration/length of a step
    reward: torch.Tensor            # Last collected reward
    done: bool                      # Terminal state for every element of the batch
    min_value: float                # Minimum normalization value
    max_value: float                # Maximum normalization value
    eps: float                      # Tolerance, useful for 2-step methods. It gives some margin to reach the end depot on time.

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
            position=self.position[key],
            length=self.length[key],
            is_traveling=self.is_traveling[key],
            reward=self.reward[key],
        )

    @staticmethod
    def initialize(batch: dict | torch.Tensor,
                   num_dirs: int = 4,
                   time_step: float = 2e-2,
                   min_value: float = 0.,
                   max_value: float = 1.,
                   eps: float = 0.) -> Any:
        """
        Initialize the state.

        Args:
            batch (dict or torch.Tensor): Input batch.
            num_dirs (int): Number of possible directions.
            time_step (float): Time step.
            min_value (float): Minimum value for normalization.
            max_value (float): Maximum value for normalization.
            eps (float): tolerance, useful for 2-step methods. It gives some margin to reach the end depot on time.

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

        # Mask of visited regions
        visited = torch.zeros(size=(batch_size, num_regions), dtype=torch.uint8, device=device)
        visited[..., 0] = 1
        visited[all_regions[..., 0] < 0] = 1  # Block dummy (negative) regions if any

        # Previously visited node
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)

        # Traveled length
        length = torch.zeros(batch_size, device=device)

        # Reward and done
        reward = torch.zeros(1, device=device)
        done = False

        # Create State
        return OpState(
            i=i,
            device=device,
            regions=regions,
            prizes=prizes,
            depot_ini=depot_ini,
            depot_end=depot_end,
            max_length=max_length,
            obs=obs,
            visited=visited,
            prev_node=prev_node,
            position=depot_ini,
            length=length,
            num_dirs=num_dirs,
            time_step=time_step,
            min_value=min_value,
            max_value=max_value,
            eps=eps,
            reward=reward,
            done=done
        )

    def step(self, action: torch.Tensor, path: torch.Tensor = None, *args, **kwargs) -> Any:
        """
        Take a step in the environment.

        Args:
            action (torch.Tensor): Batch of actions to take.
            path (torch.Tensor): Batch of paths between previous and new chosen node.

        Returns:
            NopState: Updated state.
        """
        next_node_idx = action

        # Update next position
        new_position = self.get_regions_by_index(next_node_idx)

        # Update length of route
        if path is None:
            length = self.length + (new_position - self.position).norm(p=2, dim=-1)  # self.length + self.time_step
        else:
            length = self.length + self.get_path_length(path)

        # Mask visited regions
        visited = self.visited.scatter(-1, next_node_idx[..., None], 1)

        # Reward
        reward = self.get_prize_by_index(next_node_idx)
        
        # Update state and return reward and done
        return self._replace(
            i=self.i+1,
            visited=visited,
            prev_node=next_node_idx,
            position=new_position,
            length=length,
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
                self.get_remaining_length() < 0  # No more time
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
            self.get_remaining_length() >= 0  # Still on time
        )

    def get_remaining_length(self) -> torch.Tensor:
        """
        Get the remaining length at each moment.

        Returns:
            torch.Tensor: Remaining length for each element of the batch.
        """
        return self.max_length - self.length
        
    @staticmethod
    def get_path_length(path):
        # Compute mask for dummy values
        mask = (path != -1).all(dim=-1)

        # Compute differences between consecutive coordinates
        diffs = path[:, 1:] - path[:, :-1]

        # Compute distances between consecutive coordinates
        distances = torch.linalg.norm(diffs, dim=-1)

        # Apply mask to distances
        distances_masked = distances * mask[:, :-1]

        # Compute path lengths by summing distances along the sequence
        path_lengths = distances_masked.sum(dim=-1)
        return path_lengths

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
        end_idx = self.get_end_idx()
        batch_ids = torch.arange(self.get_batch_size(), dtype=torch.int64, device=self.device)
        
        # Check which nodes can be visited without exceeding the max_length constraint
        exceeds_length = self.length[:, None] + (
            self.get_regions()[batch_ids] - self.position[:, None]
        ).norm(p=2, dim=-1) + self.eps > self.max_length[:, None] 

        # Define mask (with visited nodes)
        visited_ = self.visited.to(torch.bool)
        mask = visited_ | visited_[..., end_idx, None] | exceeds_length

        # Block initial depot
        mask[..., 0] = 1

        # End depot can always be visited, but once visited, cannot leave the place
        finished = self.finished()
        mask[finished] = 1
        mask[..., end_idx] = 0  # Always allow visiting end depot to prevent running out of nodes to choose
        return mask

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
