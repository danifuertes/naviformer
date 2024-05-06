import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
from typing import Tuple
from datetime import timedelta
from torch.utils.data import Dataset

from utils import load_dataset


class NopDataset(Dataset):
    """
    Dataset class for the Navigation Orienteering Problem (NOP).

    Args:
        num_nodes (int): Number of nodes.
        num_depots (int): Number of depots.
        max_length (float): Maximum length.
        max_nodes (int): Maximum number of nodes.
        num_obs (tuple): (Minimum, Maximum) number of obstacles.
        rad_obs (tuple): (Minimum, Maximum) radious of obstacles.
        data_dist (str): Data distribution ('const', 'unif', or 'dist').
        num_samples (int): Number of samples.
        offset (int): Offset.
        filename (str): File name.
        desc (str): Description.
    """

    def __init__(self,
                 num_nodes: int = 20,
                 num_depots: int = 1,
                 max_length: float = 2.,
                 max_nodes: int = 0,
                 num_obs: tuple = (0, 0),
                 rad_obs: tuple = (.05, .2),
                 data_dist: str = 'const',
                 num_samples: int = 1e6,
                 offset: int = 0,
                 filename: str = '',
                 desc: str = '',
                 **kwargs) -> None:
        """Initialize NopDataset with the given parameters."""
        super(NopDataset, self).__init__()

        # Load dataset from file
        if os.path.exists(filename):
            assert os.path.splitext(filename)[1] == '.pkl' or os.path.isdir(filename), f"{filename} is not a valid path"

            # Load file
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            # Load elements from file
            elements = ['depot_ini', 'depot_end', 'loc', 'prize', 'max_length']
            num_data, num_elements = len(data[0]), len(elements)
            if num_data == num_elements + 1:
                elements.append('obs')
            else:
                assert num_data == num_elements, \
                    f"Wrong data input from {filename}, found {num_data} elements, expected {num_elements}"
            elements = sorted(elements)
            self.data = [
                {
                    element: torch.tensor(element_data[i]) if isinstance(element_data[i], float)
                    else torch.FloatTensor(element_data[i])
                    for i, element in enumerate(elements)
                }
                for element_data in tqdm(data[offset:offset + num_samples], desc=desc.ljust(15))
            ]

        # No file, so create new dataset
        else:

            # Parameters to generate instances
            params = {
                'num_nodes': num_nodes,
                'data_dist': data_dist,
                'num_depots': num_depots,
                'max_length': max_length,
                'max_nodes': max_nodes,
                'num_obs': num_obs,
                'rad_obs': rad_obs,
            }

            # This manner of generating data is necessary when obstacles are generated
            if num_obs[1]:

                # Create DataLoader to generate batches of data
                from torch.utils.data import DataLoader
                torch.multiprocessing.set_sharing_strategy('file_system')
                batch_size, num_workers, self.data, count = 1024, 16, [], num_samples  # TODO: num_workers=16 may fail
                dataloader = DataLoader(NopInstance(num_samples, **params), batch_size, num_workers=num_workers)

                # Save batches into data list
                for batch in tqdm(dataloader, desc=desc.ljust(15)):
                    splits = {k: v.split(1, dim=0) for k, v in batch.items()}
                    min_count = min(batch_size, count)
                    for i in range(min_count):
                        instance = {k: v[i].squeeze(0) for k, v in splits.items()}
                        self.data.append(instance)
                    count -= min_count

            # Standard manner of generating data, simpler and faster (if no obstacles)
            else:
                self.data = [generate_instance(**params) for _ in tqdm(range(num_samples), desc=desc.ljust(15))]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict | torch.Tensor:
        """Get a sample from the dataset at the given index."""
        return self.data[idx]


class NopDatasetLarge(Dataset):
    """
    Dataset class for large instances of the Navigation Orienteering Problem (NOP).

    Args:
        filename (str): File name.
        distribution (str): Data distribution.
        num_depots (int): Number of depots.
        num_obs (tuple): (Minimum, Maximum) number of obstacles.
    """

    def __init__(self,
                 filename: str | None = None,
                 distribution: str = 'coop',
                 num_depots: int = 1,
                 num_obs: tuple = (0, 0),
                 **kwargs) -> None:
        """Initialize NopDatasetLarge with the given parameters."""
        super(NopDatasetLarge, self).__init__()
        assert distribution is not None, "Data distribution must be specified for OP"
        assert os.path.splitext(filename)[1] == '.pkl' or os.path.isdir(filename)
        assert os.path.isdir(filename)
        self.filename = filename
        self.num_depots = num_depots
        self.num_obs = num_obs

        print('Loading dataset...')
        self.size = len(os.listdir(filename))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.size

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset at the given index."""
        data = load_dataset(os.path.join(self.filename, str(idx).zfill(9)))
        elements = ['depot_ini', 'depot_end', 'loc', 'prize', 'max_length']
        if self.num_obs[1]:
            elements.append('obs')
        elements = sorted(elements)
        return {element: torch.FloatTensor(data[i]) for i, element in enumerate(elements)}


class NopInstance(Dataset):
    """
    Instance class for the Navigation Orienteering Problem (NOP).

    Args:
        num_samples (int): Number of samples.
        num_nodes (int): Number of nodes.
        data_dist (str): Data distribution.
        num_depots (int): Number of depots.
        max_length (float): Maximum length.
        num_obs (tuple): (Minimum, Maximum) number of obstacles.
        rad_obs (tuple): (Minimum, Maximum) radious of obstacles.
        max_nodes (int): Maximum number of nodes.
    """

    def __init__(self,
                 num_samples: int,
                 num_nodes: int,
                 data_dist: str,
                 num_depots: int,
                 max_length: float,
                 num_obs: tuple,
                 rad_obs: tuple,
                 max_nodes: int) -> None:
        """Initialize NopInstance with the given parameters."""
        super(NopInstance, self).__init__()
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.data_dist = data_dist
        self.num_depots = num_depots
        self.max_length = max_length
        self.max_nodes = max_nodes
        self.num_obs = num_obs
        self.rad_obs = rad_obs

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, item: int) -> dict | torch.Tensor:
        """Get a sample from the dataset at the given index."""
        return generate_instance(
            num_nodes=self.num_nodes,
            data_dist=self.data_dist,
            num_depots=self.num_depots,
            max_length=self.max_length,
            max_nodes=self.max_nodes,
            num_obs=self.num_obs,
            rad_obs=self.rad_obs,
        )


def generate_instance(num_nodes: int,
                      data_dist: str,
                      num_depots: int = 1,
                      max_length: float = 2.,
                      max_nodes: int = 0,
                      num_obs: tuple = (0, 0),
                      rad_obs: tuple = (.05, .2)) -> dict:
    """
    Generate an instance (scenario) of the Navigation Orienteering Problem (NOP).

    Args:
        num_nodes (int): Number of nodes.
        data_dist (str): Data distribution ('const', 'unif', or 'dist').
        num_depots (int): Number of depots.
        max_length (float): Maximum length.
        max_nodes (int): Maximum number of nodes.
        num_obs (tuple): (Minimum, Maximum) number of obstacles.
        rad_obs (tuple): (Minimum, Maximum) radous of obstacles.

    Returns:
        dict: Instance dictionary.
    """

    # Obstacles
    obs = generate_obstacles(*num_obs, *rad_obs) if num_obs[1] else None

    # Regions (that do not collide with obstacles)
    num_nodes = max_nodes if max_nodes else num_nodes
    loc, depot_ini, depot_end = generate_regions(num_nodes, num_depots=num_depots, obs=obs)

    # Constant prizes (Fischetti et al. 1998)
    if data_dist == 'const':
        prize = torch.ones(num_nodes)

    # Prizes sampled from uniform distribution (Fischetti et al. 1998)
    elif data_dist == 'unif':
        prize = (1 + torch.randint(0, 100, size=(num_nodes,))) / 100.

    # Prizes grow with the distance to the end depot (Fischetti et al. 1998)
    else:
        assert data_dist == 'dist', f"'{data_dist}' not in data_dist list: ['const'. 'unif', 'dist']"
        d = depot_ini if depot_end is None else depot_end
        prize_ = (d[None, :] - loc).norm(p=2, dim=-1)
        prize = (1 + (prize_ / prize_.max(dim=-1, keepdim=True)[0] * 99).int()).float() / 100.

    # Generate problems with max_nodes, get only random subsets, and fill the rest with dummys
    if max_nodes:
        num_nodes = torch.randint(low=10, high=max_nodes + 1, size=(1,))[0]
        loc[num_nodes:] = -1
        prize[num_nodes:] = 0
        max_length = (num_nodes + 60) / 40  # Simple linear regression: num_regions = 40 * max_length - 60
    else:
        max_length = torch.tensor(max_length)

    # Output dataset
    dictionary = {'loc': loc, 'prize': prize, 'depot_ini': depot_ini, 'depot_end': depot_end, 'max_length': max_length}

    # Obstacles
    if num_obs[1]:
        dictionary['obs'] = obs
    return dictionary


def generate_obstacles(min_obs: int = 0, max_obs: int = 5, r_min: float = .05, r_max: float = .2) -> torch.Tensor:
    """
    Generate obstacles (circles).

    Args:
        min_obs (int): Minimum number of obstacles.
        max_obs (int): Maximum number of obstacles.
        r_min (float): Minimum radius of obstacles.
        r_max (float): Maximum radius of obstacles.

    Returns:
        torch.Tensor: Obstacles.
    """

    # Number of obstacles
    num_obs = torch.randint(low=min_obs, high=max_obs + 1, size=[1])[0]

    # Generate random obstacles (circles)
    radius = torch.rand(num_obs) * (r_max - r_min) + r_min
    center = torch.rand((num_obs, 2))
    obstacles = torch.cat((center, radius[..., None]), dim=-1)

    # Pad with (-1, -1, 0) where num_obstacles < max_obs
    obstacles = torch.nn.functional.pad(
        input=obstacles,
        pad=(0, 0, 0, max_obs - obstacles.shape[0]),
        mode='constant',
        value=-1
    )
    obstacles[..., 2][obstacles[..., 2] == -1] = 0
    return obstacles


def generate_regions(num_nodes: int, num_depots: int = 1, obs: torch.Tensor | None = None) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate regions.

    Args:
        num_nodes (int): Number of nodes.
        num_depots (int): Number of depots.
        obs (torch.Tensor): Obstacles.

    Returns:
        tuple: Regions, initial depot, and end depot.
    """
    num_nodes = num_nodes + 1 if num_depots == 1 else num_nodes + 2

    # No obstacles
    if obs is None:
        points = torch.FloatTensor(num_nodes, 2).uniform_(0, 1)

    # Obstacles
    else:
        num_obs = obs.shape[0]
        obs_center = obs[..., :2]
        obs_radius = obs[..., 2]

        # Meshgrid
        grid_size = 64
        x, y = torch.meshgrid(torch.linspace(0, 1, grid_size), torch.linspace(0, 1, grid_size), indexing="ij")
        xy = torch.stack((x, y), dim=-1).expand([num_obs, grid_size, grid_size, 2])

        # Calculate distance squared from each point of the meshgrid to each obstacle center
        distances = (xy - obs_center[..., None, None, :]).norm(2, dim=-1)

        # Create the masks by comparing distances with squared radius
        mask = (distances > obs_radius[..., None, None] + 0.02).all(dim=0).T

        # Generate non-colliding points
        non_colliding_points = torch.nonzero(mask)
        non_colliding_points = torch.stack((non_colliding_points[..., 1], non_colliding_points[..., 0]), dim=-1)
        points = non_colliding_points[torch.randperm(non_colliding_points.shape[0])[:num_nodes]] / float(
            grid_size - 1)

    # Separate regions and depots
    depot_ini = points[0]
    depot_end = points[1] if num_depots == 2 else points[0]
    loc = points[num_depots:]
    return loc, depot_ini, depot_end


def print_nop_results(results: tuple) -> None:
    """
    Print NOP results.

    Args:
        results (tuple): Results tuple with reward, actions, success, duration, num_nodes, and parallelism.
    """

    # Get results info
    reward, actions, success, duration, num_nodes, parallelism = results

    # Get number of nodes visited
    visits = []
    for i, action in enumerate(actions):
        nodes = np.array(action)[:, 0]
        unique_nodes = len(np.unique(nodes)) - 1  # Remove one since end depot should not count
        if success[i]:
            visits.append(unique_nodes)

    # Print reward
    print("\nREWARD")
    print(f"\tAverage reward: {-np.mean(reward):.4f} +- {2 * np.std(reward) / np.sqrt(len(reward)):.4f}")
    print(f"\tMax reward: {-np.min(reward):.4f} | Min reward: {-np.max(reward):.4f}")

    # Print success rate
    print("\nSUCCESS")
    print(f"\tFound success in {np.sum(success)}/{len(success)} scenarios")
    print(f"\tRate of success: {np.mean(success):.4f} +- {2 * np.std(success) / np.sqrt(len(success)):.4f}")

    # Print number of nodes visited
    node_rate = np.array(visits) / (np.mean(num_nodes) / 2)  # Max length is fixed to allow visiting half of the nodes
    print("\nNUMBER OF NODES VISITED")
    print(f"\tAverage number of nodes visited: {np.mean(visits):.4f} +- {2 * np.std(visits) / np.sqrt(len(visits)):.4f}")
    print(f"\tMax number of nodes visited: {np.max(visits):.0f} | Min number of nodes visited: {np.min(visits):.0f}")
    print(f"\tRate of nodes visited: {np.mean(node_rate):.4f} +- {2 * np.std(node_rate) / np.sqrt(len(node_rate)):.4f}")

    # Print serial duration
    mean_time, ci_time = np.mean(duration), 2 * np.std(duration) / np.sqrt(len(duration))
    print("\nTIME")
    print(f"\tAverage serial duration: {mean_time} +- {ci_time} seconds")

    # Print parallel duration
    mean_time, ci_time = np.mean(duration) / parallelism, 2 * np.std(duration) / np.sqrt(len(duration)) / parallelism
    print(f"\tAverage parallel duration: {mean_time} +- {ci_time} seconds")

    # Print total time
    total_time = np.sum(duration) / parallelism
    print(f"\tCalculated total duration: {timedelta(seconds=int(total_time))} ({total_time} seconds)\n")
