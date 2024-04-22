import torch
from typing import Any, Tuple, NamedTuple


def set_decode_type(model: torch.nn.Module | torch.nn.DataParallel, decode_type: str) -> None:
    """
    Set the decoding type for the model.

    Args:
        model (torch.nn.Module): The model to set decode type for.
        decode_type (str): The decode type to set ('greedy' or 'sampling').
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def input_embed(state: Any, embed: torch.nn.Module, embed_depot: torch.nn.Module) -> torch.Tensor:
    """
    Embedding for the inputs.

    Args:
        state (Any): The state of the environment.
        embed (torch.nn.Module): The embedding module.
        embed_depot (torch.nn.Module): The embedding module for the depot.

    Returns:
        torch.Tensor: The concatenated embeddings.
    """

    # Input embedding of start depot (coordinates) and nodes (coordinates and prizes)
    embeddings = (
        embed_depot(state.get_depot_ini())[:, None, :],
        embed(
            torch.cat((
                state.regions,
                *(feature[..., None] for feature in state.get_features())
            ), dim=-1)
        )
    )

    # Input embedding of end depot (coordinates)
    if state.is_depot_end():
        embeddings = embeddings + (embed_depot(state.get_depot_end())[:, None, :], )

    # Return concatenated embeddings
    return torch.cat(embeddings, dim=1)


def make_heads(num_heads: int, x: torch.Tensor) -> torch.Tensor:
    """
    Create heads for Multi-Head Attention.

    Args:
        num_heads (int): The number of attention heads.
        x (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The reshaped tensor.
    """
    return x.contiguous().view(x.size(0), x.size(1), num_heads, -1).permute(2, 0, 1, 3)  # (N_heads, B, N, H)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    graph_embedding: torch.Tensor
    graph_embedding_mean: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    key_logit: torch.Tensor
    obs_embedding_mean: torch.Tensor
    obs_map: torch.Tensor
    obs_grid: torch.Tensor

    def __getitem__(self, key: slice | torch.Tensor) -> 'AttentionModelFixed':
        """
        Get item based on the key.

        Args:
            key (slice or torch.Tensor): The index or slice.

        Returns:
            AttentionModelFixed: The sliced named tuple.
        """
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            graph_embedding=self.graph_embedding[key],
            graph_embedding_mean=self.graph_embedding_mean[key],
            key=self.key[:, key],  # dim 0 are the heads
            value=self.value[:, key],  # dim 0 are the heads
            key_logit=self.key_logit[key],
            obs_embedding_mean=self.obs_embedding_mean[key],
            obs_map=self.obs_map[key],
            obs_grid=self.obs_grid[key],
        )


def create_obs_map(obs: torch.Tensor, patch_size: int, map_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a map of the scenario representing the obstacles as bidimensional gaussian distributions.

    Args:
        obs (torch.Tensor): The tensor representing the obstacles.
        patch_size (int): The size of the patches.
        map_size (int): The size of the map.

    Returns:
        tuple: The obstacle map and grid.
    """

    batch_size, num_obs, _ = obs.shape
    device = obs.device
    padding = patch_size // 2  # To ensure that patches do not exceed image boundaries

    # Define meshgrid
    x = torch.linspace(0, map_size, map_size).to(device)
    y = torch.linspace(0, map_size, map_size).to(device)
    x, y = torch.meshgrid(x, y, indexing="ij")
    x = x.expand(batch_size, map_size, map_size)
    y = y.expand(batch_size, map_size, map_size)

    # Calculate global map
    z = torch.zeros(batch_size, map_size, map_size).to(device)
    for i in range(num_obs):
        x0 = obs[:, i, 0].view(-1, 1, 1) * map_size
        y0 = obs[:, i, 1].view(-1, 1, 1) * map_size
        s = obs[:, i, 2].view(-1, 1, 1) * map_size + 0.01
        g = 1 / (2 * torch.pi * s * s) * torch.exp(
            -(torch.div((x - x0) ** 2, (2 * s ** 2)) + torch.div((y - y0) ** 2, (2 * s ** 2)))
        )  # https://stackoverflow.com/questions/69024270/how-to-create-a-normal-2d-distribution-in-pytorch
        max_g = g.view(batch_size, -1).max(dim=-1).values
        w = torch.where((max_g > 0).view(-1, 1, 1), g / max_g.view(-1, 1, 1), g)
        z += w
    z = z.permute(0, 2, 1)
    z = torch.nn.functional.pad(z, (padding, padding, padding, padding), mode='constant', value=0)

    # Create meshgrid of coordinates for each batch element
    grid_range = torch.arange(0, patch_size).to(device)
    grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch_size, patch_size, patch_size, 2)
    return z, grid


def create_local_maps(
        pos: torch.Tensor,
        obs_map: torch.Tensor,
        obs_grid: torch.Tensor,
        goal: torch.Tensor,
        patch_size: int,
        map_size: int) -> torch.Tensor:
    """
    Create local maps.

    Args:
        pos (torch.Tensor): The position tensor.
        obs_map (torch.Tensor): The obstacle map.
        obs_grid (torch.Tensor): The obstacle grid.
        goal (torch.Tensor): The goal tensor.
        patch_size (int): The patch size.
        map_size (int): The size of the map.

    Returns:
        torch.Tensor: The local maps.
    """

    # Get agent position in obstacle map
    map_x = torch.floor(pos[..., 0] * map_size).type(torch.long)
    map_x[map_x < 0] = 0
    map_x[map_x >= map_size] = map_size - 1
    map_y = torch.floor(pos[..., 1] * map_size).type(torch.long)
    map_y[map_y < 0] = 0
    map_y[map_y >= map_size] = map_size - 1

    # Get obstacles patches
    offset_x = map_x[:, None, None] + obs_grid[..., 0]
    offset_y = map_y[:, None, None] + obs_grid[..., 1]
    patches = obs_map[torch.arange(obs_map.size(0))[:, None, None], offset_y, offset_x].permute(0, 2, 1)
    north = patches[:, patch_size // 2:, :].permute(0, 2, 1)
    south = patches[:, :patch_size // 2, :].permute(0, 2, 1)
    west = patches[:, :, :patch_size // 2]
    east = patches[:, :, patch_size // 2:]
    obs_patches = torch.stack((east, north, west, south), dim=1)

    # Get goal patches (project the position of the goal into the local map)
    goal_x = torch.floor(goal[..., 0] * map_size).type(torch.long)
    condition = goal_x > map_x + patch_size // 2
    goal_x[condition] = map_x[condition] + patch_size // 2
    condition = goal_x < map_x - patch_size // 2
    goal_x[condition] = map_x[condition] - patch_size // 2
    goal_x = goal_x - map_x + patch_size // 2
    goal_y = torch.floor(goal[..., 1] * map_size).type(torch.long)
    condition = goal_y > map_y + patch_size // 2
    goal_y[condition] = map_y[condition] + patch_size // 2
    condition = goal_y < map_y - patch_size // 2
    goal_y[condition] = map_y[condition] - patch_size // 2
    goal_y = goal_y - map_y + patch_size // 2
    gx = -(obs_grid[..., 0] - goal_x.view(-1, 1, 1)) ** 2 / (2 * (patch_size // 8) ** 2)
    gy = -(obs_grid[..., 1] - goal_y.view(-1, 1, 1)) ** 2 / (2 * (patch_size // 8) ** 2)
    patches = 1 / (2 * torch.pi * (patch_size // 8) ** 2) * torch.exp(gx + gy)

    # Combine goal patches
    north = patches[:, patch_size // 2:, :].permute(0, 2, 1)
    south = patches[:, :patch_size // 2, :].permute(0, 2, 1)
    west = patches[:, :, :patch_size // 2]
    east = patches[:, :, patch_size // 2:]
    goal_patches = torch.stack((east, north, west, south), dim=1)

    # Return obstacle patches and goal patches
    return torch.concat((obs_patches, goal_patches), dim=1)
