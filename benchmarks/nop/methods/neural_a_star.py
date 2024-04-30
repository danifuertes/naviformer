import torch
from neural_astar.planner import NeuralAstar as NAStar
from neural_astar.utils.training import load_from_ptl_checkpoint
# from neural_astar.utils.data import visualize_results, create_dataloader


class NeuralAStar:
    
    def __init__(self, grid_size=(32, 32), batch_size=1, device='cpu') -> None:
        self.grid_size = grid_size
        self.batch_size = batch_size
        self.device = device
        self.model = NAStar(encoder_arch='CNN').to(self.device)
        self.model.load_state_dict(load_from_ptl_checkpoint(
            "./benchmarks/nop/methods/neural-astar/model/mazes_032_moore_c8/lightning_logs/"
        ))
        self.model.eval()

    
    def planning(self, sx, sy, gx, gy, map2d, **kwargs):
        map2d = torch.tensor(map2d).to(self.device)
        
        # Start and goal coordinates
        start_coord = torch.cat((sx, sy), dim=-1).to(self.device)
        goal_coord = torch.cat((gx, gy), dim=-1).to(self.device)
        
        # Start and goal maps
        start_maps = torch.zeros((self.batch_size, 1) + self.grid_size, device=self.device)
        start_maps[start_coord]
        goal_maps = torch.zeros((self.batch_size, 1) + self.grid_size, device=self.device)
        goal_maps[goal_coord]
        
        # dataloader = create_dataloader("./benchmarks/nop/methods/neural-astar/planning-datasets/data/mpd/mazes_032_moore_c8.npz", "test", 8)
        # map_designs, start_maps, goal_maps, _ = next(iter(dataloader))

        # Make predictions
        outputs = self.model(map2d, start_maps, goal_maps)

        # Calculate path from 2D map
        for i in range(self.batch_size):
            pos = start_coord[i, 0]
            path_map = outputs.paths[i, 0]
            path = []
            while pos[0] != goal_coord[i, 0, 0] or pos[1] != goal_coord[i, 0, 1]:
                next_pos, path_map = self.find_next_max_adjacent(path_map, pos)
                path.append(next_pos)
                pos = next_pos
        return path
    
    @staticmethod
    def argmax2d(x):
        return (x == torch.max(x)).nonzero()[0]
    
    @staticmethod
    def find_next_max_adjacent(map2d, coord):
        
        # Extract coordinates
        x, y = coord
        
        # Get adjacent positions including diagonals
        adjacent_positions = [
            (x-1, y-1), (x-1, y), (x-1, y+1),
            (x, y-1),             (x, y+1),
            (x+1, y-1), (x+1, y), (x+1, y+1)
        ]
        
        # Filter out positions outside tensor boundaries
        adjacent_positions = [(i, j) for i, j in adjacent_positions if 0 <= i < map2d.size(0) and 0 <= j < map2d.size(1)]
        
        # Get values of adjacent positions
        adjacent_values = [map2d[i, j] for i, j in adjacent_positions]
        
        # Find maximum value
        max_value = torch.max(torch.tensor(adjacent_values))
        
        # Find positions with maximum value
        max_positions = [pos for pos, val in zip(adjacent_positions, adjacent_values) if val == max_value][0]
        map2d[max_positions[0], max_positions[1]] = 0
        return max_positions, map2d
