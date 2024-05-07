import torch
from neural_astar.planner import VanillaAstar as VAStar
from neural_astar.utils.training import load_from_ptl_checkpoint
# from neural_astar.utils.data import visualize_results, create_dataloader


class AStar:
    
    def __init__(self, obs, grid_size=(32, 32), batch_size=1, device='cpu', scale=100) -> None:
        self.scale = scale
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.device = device
        self.model = VAStar().to(self.device)
        self.model.eval()
        self.create_obs_map(obs, scale=scale)
    
    def create_obs_map(self, obs, scale, *args, **kwargs):
        self.obs_map = torch.ones((self.batch_size, 1) + self.grid_size, device=self.device)
        grid = torch.stack(
            torch.meshgrid(torch.arange(self.grid_size[0]), torch.arange(self.grid_size[1])), axis=-1
        )
        obs = torch.tensor(obs).to(self.device) / scale
        obs = torch.index_select(obs, 1, torch.LongTensor([1, 0, 2]).to(self.device))
        for ob in obs:
            if ob[0] < 0 or ob[1] < 0 or ob[2] <= 0:
                continue
            ob = ob * max(self.grid_size)
            distance = torch.linalg.norm(grid - ob[:2], axis=-1)
            condition = distance <= ob[2]
            condition = condition.reshape(1, 1, *self.grid_size).expand(self.batch_size, 1, *self.grid_size)
            self.obs_map[condition] = 0
        return
            
    @staticmethod
    def set_obs_map():
        pass

    
    def planning(self, sx, sy, gx, gy, **kwargs):
        sx = int(sx * self.grid_size[0] / self.scale)
        sy = int(sy * self.grid_size[1] / self.scale)
        gx = int(gx * self.grid_size[0] / self.scale)
        gy = int(gy * self.grid_size[1] / self.scale)
        
        # Start and goal maps
        start_maps = torch.zeros((self.batch_size, 1) + self.grid_size, device=self.device)
        start_maps[..., sy, sx] = 1
        goal_maps = torch.zeros((self.batch_size, 1) + self.grid_size, device=self.device)
        goal_maps[..., gy, gx] = 1
        
        # dataloader = create_dataloader("./benchmarks/nop/methods/neural-astar/planning-datasets/data/mpd/mazes_032_moore_c8.npz", "test", 8)
        # map_designs, start_maps, goal_maps, _ = next(iter(dataloader))

        # Make predictions
        outputs = self.model(self.obs_map, start_maps, goal_maps)

        # Calculate path from 2D map
        for i in range(self.batch_size):
            pos = (sy, sx)
            path_map = outputs.paths[i, 0]
            path = []
            while not (pos[0] == gy and pos[1] == gx):
                next_pos, path_map = self.find_next_max_adjacent(path_map, pos)
                path.append(next_pos)
                pos = next_pos
        path = torch.tensor(path, device=self.device)
        return path[:, 1], path[:, 0], True
    
    @staticmethod
    def argmax2d(x):
        return (x == torch.max(x)).nonzero()[0]
    
    def find_next_max_adjacent(self, map2d, coord):
        
        # Extract coordinates
        y, x = coord
        
        # Set current position to zero in the map
        map2d[y, x] = 0
        
        # Get adjacent positions including diagonals
        adjacent_positions = [
            (y-1, x-1), (y-1, x), (y-1, x+1),
            (y, x-1),             (y, x+1),
            (y+1, x-1), (y+1, x), (y+1, x+1)
        ]
        
        # Filter out positions outside tensor boundaries
        adjacent_positions = [(i, j) for i, j in adjacent_positions if 0 <= i < map2d.size(0) and 0 <= j < map2d.size(1)]
        
        # Get values of adjacent positions
        adjacent_values = [map2d[i, j] for i, j in adjacent_positions]
        assert torch.tensor(adjacent_values, device=self.device).sum() == 1
        
        # Find maximum value
        max_value = torch.max(torch.tensor(adjacent_values))
        
        # Find positions with maximum value
        max_positions = [pos for pos, val in zip(adjacent_positions, adjacent_values) if val == max_value][0]
        return max_positions, map2d
