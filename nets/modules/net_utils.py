import torch
from typing import NamedTuple


def set_decode_type(model, decode_type):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def input_embed(inputs, embed, embed_depot):
    """Embedding for the inputs"""

    # Navigation Orienteering Problem features
    features = ('prize',)

    # Input embedding of start depot (coordinates) and nodes (coordinates and prizes)
    embeddings = (
        embed_depot(inputs['depot'])[:, None, :],
        embed(
            torch.cat((
                inputs['loc'],
                *(inputs[feat][:, :, None] for feat in features)
            ), dim=-1))
    )

    # Input embedding of end depot (coordinates)
    if 'depot2' in inputs:
        embeddings = embeddings + (embed_depot(inputs['depot2'])[:, None, :], )

    # Return concatenated embeddings
    return torch.cat(embeddings, dim=1)


def make_heads(num_heads, v, num_steps=None):
    """Create heads for Multi-Head Attention."""
    assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps
    return (
        v.contiguous().view(v.size(0), v.size(1), v.size(2), num_heads, -1)
        .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), num_heads, -1)
        .permute(3, 0, 1, 2, 4)  # (num_heads, batch_size, num_steps, graph_size, head_dim)
    )


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor
    obs_embed: torch.Tensor
    obs_map: torch.Tensor
    obs_grid: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
            glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
            logit_key=self.logit_key[key],
            obs_embed=self.obs_embed[key],
            obs_map=self.obs_map[key],
            obs_grid=self.obs_grid[key]
        )


def create_obs_map(obs, patch_size, map_size):
    """Create a map of the scenario representing the obstacles as bidimensional gaussian distributions."""

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


def create_local_maps(pos, obs_map, obs_grid, goal, patch_size, map_size):

    # Get agent position in obstacle map
    map_x = torch.floor(pos[..., 0] * map_size).type(torch.long)
    map_x[map_x < 0] = 0
    map_x[map_x >= map_size] = map_size - 1
    map_y = torch.floor(pos[..., 1] * map_size).type(torch.long)
    map_y[map_y < 0] = 0
    map_y[map_y >= map_size] = map_size - 1

    # Get obstacles patches
    offset_x = map_x[:, None] + obs_grid[..., 0]
    offset_y = map_y[:, None] + obs_grid[..., 1]
    patches = obs_map[torch.arange(obs_map.shape[0])[:, None, None], offset_y, offset_x].permute(0, 2, 1)
    north = patches[:, patch_size // 2:, :].permute(0, 2, 1)
    south = patches[:, :patch_size // 2, :].permute(0, 2, 1)
    west = patches[:, :, :patch_size // 2]
    east = patches[:, :, patch_size // 2:]
    obs_patches = torch.stack((east, north, west, south), dim=1)

    # Get goal patches (project the position of the goal into the local map)
    map_x, map_y = map_x[:, 0], map_y[:, 0]
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


def create_global_obs_map(obs, patch_size, map_size):
    """Create a map of the scenario representing the obstacles as bidimensional gaussian distributions."""

    batch_size, num_obs, _ = obs.shape
    device = obs.device
    padding = patch_size // 2  # To ensure patches don't exceed image boundaries

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

    # Create meshgrid of coordinates for each batch element
    grid_range = torch.arange(0, map_size).to(device)
    grid_x, grid_y = torch.meshgrid(grid_range, grid_range, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch_size, map_size, map_size, 2)
    return z, grid


def create_global_maps(pos, obs_map, obs_grid, goal, patch_size, map_size):
    cond = (obs_grid - pos[:, None] * map_size).norm(2, dim=-1) <= 2
    obs_map = torch.where(cond, 5 * torch.ones_like(obs_map), obs_map)
    cond = (obs_grid - goal[:, None, None] * map_size).norm(2, dim=-1) <= 2
    obs_map = torch.where(cond, -5 * torch.ones_like(obs_map), obs_map)
    return obs_map[:, None]
