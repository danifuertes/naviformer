import torch
import numpy as np
import matplotlib.pyplot as plt

from nets.modules.transpath.models import Autoencoder
from nets.modules.transpath.modules.planners import DifferentiableDiagAstar

def load_transpath_models(weights_dir='tmp/transpath/weights', device='cpu'):
    # Load trained autoencoders
    model_cf = Autoencoder(mode='k').to(device)
    model_focal = Autoencoder(mode='f').to(device)

    model_cf.load_state_dict(torch.load(f'{weights_dir}/cf.pth', map_location=device))
    model_focal.load_state_dict(torch.load(f'{weights_dir}/focal.pth', map_location=device))

    model_cf.eval()
    model_focal.eval()

    # Initialize planners
    planners = {
        'cf': DifferentiableDiagAstar(mode='k').to(device),
        'focal': DifferentiableDiagAstar(mode='f', f_w=100).to(device),
        'wastar': DifferentiableDiagAstar(mode='default', h_w=2).to(device),
        'astar': DifferentiableDiagAstar(mode='default', h_w=1).to(device)
    }

    return model_cf, model_focal, planners

# def create_start_goal_matrix(start, goal, height, width, device='cpu'):
#     sg_matrix = torch.zeros((1, 1, height, width), device=device)
#     for point in [start, goal]:
#         y, x = point[0]
#         sg_matrix[0, 0, y, x] = 1.0
#     return sg_matrix

# def coords_to_mask(coords, height, width, device='cpu'):
#     """Converts (B, 2) coordinates to binary masks (B, 1, H, W)"""
#     B = coords.shape[0]
#     mask = torch.zeros((B, 1, height, width), device=device)
#     for i in range(B):
#         y, x = coords[i]
#         mask[i, 0, y, x] = 1.0
#     return mask

def create_start_goal_matrix(starts, goals, height, width, device='cpu'):
    """
    Create a tensor of shape (B, 1, H, W) with 1s at start and goal locations.
    """
    B = starts.shape[0]
    sg_matrix = torch.zeros((B, 1, height, width), device=device)
    batch_indices = torch.arange(B, device=device)

    sg_matrix[batch_indices, 0, starts[:, 0], starts[:, 1]] = 1.0
    sg_matrix[batch_indices, 0, goals[:, 0], goals[:, 1]] = 1.0
    return sg_matrix

def coords_to_mask(coords, height, width, device='cpu'):
    """Converts (B, 2) coordinates to binary masks (B, 1, H, W)"""
    B = coords.shape[0]
    mask = torch.zeros((B, 1, height, width), device=device)
    batch_indices = torch.arange(B, device=device)
    mask[batch_indices, 0, coords[:, 0], coords[:, 1]] = 1.0
    return mask

def plan_path(map_design, start, goal, model_cf, model_focal, planners, method='cf'):
    """
    map_design: (1, 1, H, W) torch.Tensor, 1 = free, 0 = obstacle
    start, goal: (1, 2) torch.LongTensor
    method: 'cf', 'focal', 'wastar', 'astar'
    """
    device = map_design.device
    B, C, H, W = map_design.shape
    assert C == 1, "Map must have shape (B, 1, H, W)"

    start_mask = coords_to_mask(start, H, W, device)
    goal_mask = coords_to_mask(goal, H, W, device)

    with torch.no_grad():
        if method == 'cf':
            sg_matrix = create_start_goal_matrix(start, goal, H, W, device)  # only goal used for CF
            inputs = torch.cat([map_design, sg_matrix], dim=1)
            pred_map = (model_cf(inputs) + 1) / 2
            planner = planners['cf']

        elif method == 'focal':
            sg_matrix = create_start_goal_matrix(start, goal, H, W, device)
            inputs = torch.cat([map_design, sg_matrix], dim=1)
            pred_map = (model_focal(inputs) + 1) / 2
            planner = planners['focal']

        elif method == 'wastar':
            pred_map = (map_design == 1).float()  # free space
            planner = planners['wastar']

        elif method == 'astar':
            pred_map = (map_design == 1).float()
            planner = planners['astar']

        else:
            raise ValueError(f"Unknown method: {method}")
        
        path_output = planner(
            pred_map,
            start_mask,
            goal_mask,
            (map_design == 1).float()
        )

    return path_output.paths.squeeze(dim=1)  # (B, H, W) binary mask of path

def plot_path_with_annotations(path_mask, map_design, start, goal, title="Planned Path"):
    """
    path_mask: (1, H, W) tensor with 1s along the planned path
    map_design: (1, 1, H, W) binary tensor (1 = free, 0 = obstacle)
    start, goal: (1, 2) long tensors with (y, x) coordinates
    """
    path_np = path_mask[0].cpu().numpy()
    map_np = map_design[0, 0].cpu().numpy()
    H, W = map_np.shape

    rgb = np.ones((H, W, 3), dtype=np.float32)  # white background

    # Obstacles → black
    rgb[map_np == 0] = [0.0, 0.0, 0.0]

    # Path → green
    rgb[path_np == 1] = [0.0, 1.0, 0.0]

    # Start → blue dot
    y, x = start[0].cpu().numpy()
    rgb[y, x] = [0.0, 0.0, 1.0]

    # Goal → red dot
    y, x = goal[0].cpu().numpy()
    rgb[y, x] = [1.0, 0.0, 0.0]

    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":

    # Load models
    model_cf, model_focal, planners = load_transpath_models()

    # Dummy binary map: 64x64 grid
    H, W = 64, 64
    binary_map_np = np.ones((1, 1, H, W), dtype=np.float32)
    binary_map_np[0, 0, 20:30, 20:45] = 0  # obstacle region

    map_design = torch.tensor(binary_map_np)

    # Define start and goal
    start = torch.tensor([[5, 5]])   # shape: (1, 2)
    goal = torch.tensor([[58, 58]])  # shape: (1, 2)

    # Get path using WA* + CF model
    path_mask = plan_path(map_design, start, goal,
                        model_cf, model_focal, planners,
                        method='cf')

    # Visualize (optional)
    plot_path_with_annotations(path_mask, map_design, start, goal, title="Planned Path (WA* + CF)")
