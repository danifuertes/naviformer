import torch
from typing import Tuple
from neural_astar.planner import VanillaAstar as VAStar


class AStar:
    
    def __init__(self, obs, grid_size=(32, 32), device='cpu', scale=100, model=None) -> None:
        self.scale = scale
        self.map_size = grid_size
        self.step_size = 1 / max(grid_size)
        self.batch_size = 1
        self.device = device
        if model is None:
            self.model = VAStar().to(self.device) if model is None else model
        else:
            self.model = model
        self.model.eval()
        self.obs_map = self.create_obs_map(obs, scale=scale)
    
    def create_obs_map(self, obs, scale, *args, **kwargs):
        obs_map = torch.ones((self.batch_size, ) + self.map_size, device=self.device)
        grid = torch.stack(
            torch.meshgrid(torch.arange(self.map_size[0]), torch.arange(self.map_size[1]), indexing="ij"), axis=-1
        )
        obs = torch.tensor(obs).to(self.device) / scale
        obs = torch.index_select(obs, 1, torch.LongTensor([1, 0, 2]).to(self.device))
        for ob in obs:
            if ob[0] < 0 or ob[1] < 0 or ob[2] <= 0:
                continue
            ob = ob * max(self.map_size)
            distance = torch.linalg.norm(grid - ob[:2], axis=-1)
            condition = distance <= ob[2]
            condition = condition.reshape(1, *self.map_size).expand(self.batch_size, *self.map_size)
            obs_map[condition] = 0
        return obs_map
            
    @staticmethod
    def set_obs_map():
        pass

    
    def planning(self, sx, sy, gx, gy, **kwargs):
        batch_ids = torch.arange(self.batch_size, dtype=torch.int64, device=self.device)
        
        start = torch.tensor([[sx, sy]], device=self.device) / 200
        goal = torch.tensor([[gx, gy]], device=self.device) / 200
        
        start_map = self.node2map(start, self.obs_map, batch_ids)
        goal_map = self.node2map(goal, self.obs_map, batch_ids)
        
        try:
            # self.check_error(self.obs_map, start)  # For some reason, NAStar fails if there are obstacles adjacent to start/goal positions
            # self.check_error(self.obs_map, goal)
            predicted_map = self.model(
                self.obs_map[:, None].contiguous(),
                start_map[:, None].contiguous(),
                goal_map[:, None].contiguous()
            ).paths[:, 0]
            path = self.map2dirs(predicted_map, start, goal)
            if predicted_map.sum() < 1:
                raise IndexError
            success = True
        except IndexError:
            path = torch.stack((start, goal), dim=1)
            path = torch.cat((
                torch.zeros_like(path[..., 0, None]), path
            ), dim=-1)
            success = False
        return path[0, :, 1] * self.scale, path[0, :, 2] * self.scale, success
    
    def node2map(self, node: torch.Tensor, obs_map: torch.Tensor, batch_ids: torch.Tensor) -> torch.Tensor:
        """
        Create a map for either the start or goal nodes as required by Neural A*.

        Args:
            node (torch.Tensor): coordinates of the start or goal node.
            obs_map (torch.Tensor): obstacles' map.
            batch_ids (torch.Tensor): arange of batch ids.

        Returns:
            torch.Tensor: map for start/goal node.
        """
        node_x = (node[..., 0] * self.map_size[1]).type(torch.int64)
        node_x = torch.clamp(node_x, 0, self.map_size[1] - 1).long()
        node_y = (node[..., 1] * self.map_size[0]).type(torch.int64)
        node_y = torch.clamp(node_y, 0, self.map_size[0] - 1).long()
        node_map = torch.zeros_like(obs_map)
        node_map[batch_ids, node_y, node_x] = 1
        return node_map
    
    def map2dirs(self, predicted_map: torch.Tensor, start: torch.Tensor, goal: torch.Tensor, max_iters: int = 300) -> torch.Tensor:
        """
        Convert map with path predicted by Neural A* into a sequence of coordinates.

        Args:
            predicted_map (torch.Tensor): map with the predicted path predicted by Neural A*.
            start (torch.Tensor): coordinates of starting node.
            goal (torch.Tensor): coordinates of goal node.
            max_iters (int, optional): max number of iterations. Defaults to 300.

        Returns:
            torch.Tensor: _description_
        """
        pos = start.clone().float()
        
        # Initialize mask that allows different path lengths across elements from the same batch
        not_done = torch.ones_like(pos[:, 0]).bool()
        
        # Add starting node at the beginning of the path
        path = [
            torch.cat((torch.zeros_like(pos[:, 0, None]).long(), start), dim=-1)
        ]
        
        # Sequentially decode path
        i = 0
        while not_done.any().item() and i < max_iters:
            dirs = -torch.ones_like(pos[:, 0]).long()
            
            # Get next position from current node
            p, d, predicted_map = self.max_adjacent(predicted_map, pos)
            
            # Save next position if not done yet
            pos[not_done], dirs[not_done] = p[not_done], d[not_done]
            path.append(torch.cat((dirs[:, None], pos), dim=-1))
            
            # Check if done
            not_done = ~self.check_coords(pos, goal)
            i += 1
        
        # Add goal node at the end of the path
        path.append(
            torch.cat((-torch.ones_like(pos[:, 0, None]).long(), goal), dim=-1)
        )
        
        # Return path as tensor
        return torch.stack(path, dim=1)
        
    @ staticmethod
    def max_adjacent(maps: torch.Tensor, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        From given position (coords) and map with path predicted by Neural A*, get the next adjacent position of the path.

        Args:
            maps (torch.Tensor): map with the predicted path predicted by Neural A*.
            coords (torch.Tensor): current position.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: next position, next direction, updated map (with current position removed).
        """
        
        # Dimensions
        batch_size, height, width = maps.size()
        
        # Get coordinates in map reference system
        x = (coords[..., 0] * width).clamp(0, width - 1)
        y = (coords[..., 1] * height).clamp(0, height - 1)
        
        # Set current coordinates to zero in the map
        maps[torch.arange(batch_size).to(maps.device), y.long(), x.long()] = 0
        
        # Generate indices for adjacent positions including diagonals
        row_offsets = torch.tensor([0, 1, 1, 1, 0, -1, -1, -1], device=maps.device)
        col_offsets = torch.tensor([1, 1, 0, -1, -1, -1, 0, 1], device=maps.device)
        row_idx = (
            y[:, None].expand(batch_size, len(row_offsets)) + \
            row_offsets[None].expand(batch_size, len(row_offsets))
        ).clamp(0, height - 1).long()
        col_idx = (
            x[:, None].expand(batch_size, len(col_offsets)) + \
            col_offsets[None].expand(batch_size, len(col_offsets))
        ).clamp(0, width - 1).long()
        
        # Get values of adjacent positions
        batch_idx = torch.arange(batch_size)[:, None].expand(batch_size, 8).to(maps.device)
        adjacent_values = maps[batch_idx, row_idx, col_idx]
        
        # Find maximum value (directions)
        _, dirs = torch.max(adjacent_values.view(batch_size, -1), dim=-1)
        
        # Gather coordinates of maximum values
        next_coords = torch.stack((
            col_idx.gather(1, dirs[:, None])[:, 0] / height,
            row_idx.gather(1, dirs[:, None])[:, 0] / width,
        ), dim=-1)
        return next_coords, dirs, maps
    
    @staticmethod
    def check_coords(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Check if 2 positions are the same (or very close to each other).

        Args:
            coords1 (torch.Tensor): first position.
            coords2 (torch.Tensor): second position.

        Returns:
            torch.Tensor: boolean indicating if the positions are the same or not.
        """
        return torch.linalg.norm(coords1 - coords2, dim=-1) < 0.02

