import argparse

import numpy as np
import torch

from utils import *
from envs import load_problem
from nets import load_model_eval
from benchmarks import solve_nop, parse_runs

from neural_astar.planner import NeuralAstar as NAStar
from neural_astar.planner import VanillaAstar as VAStar
from neural_astar.utils.training import load_from_ptl_checkpoint


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
ROUTE_PLANNERS = global_vars()['ROUTE_PLANNERS']
PATH_PLANNERS = global_vars()['PATH_PLANNERS']


def get_options(args: list = None) -> argparse.Namespace:
    """
    Parse command-line arguments and return options.

    Args:
        args (list): List of command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')

    # Model
    parser.add_argument('--model', type=str, help="Path to trained model. For benchmarks, combine route and path"
                                                  "planners with '-'. Example: ga-a_star")

    # Problem
    parser.add_argument('--problem', type=str, default='nop', help=f"Problem to solve: {', '.join(PROBLEMS)}")
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of visitable nodes")
    parser.add_argument('--max_nodes', type=int, default=0, help='Max number of nodes (random num_nodes padded with'
                        'dummys until max_nodes). Max length depends on the random num_nodes. Disable with max_nodes=0')
    parser.add_argument('--num_depots', type=int, default=2, help="Number of depots. Options are 1 or 2. 1"
                        "means start and end depot are the same. 2 means they are different")
    parser.add_argument('--max_length', type=float, default=2., help="Normalized time limit to solve the problem")
    parser.add_argument('--data_dist', type=str, default='const',
                        help=f"Data distribution (reward values of regions) of OP. Options: {', '.join(DATA_DIST)}")
    parser.add_argument('--num_obs', type=int, nargs='+', default=(0, 0), help="Tuple of 2 values indicating min and max number of obstacles")
    parser.add_argument('--rad_obs', type=float, nargs='+', default=(.02, .12), help="Tuple of 2 values indicating min and max radious of obstacles")
    parser.add_argument('--eps', type=float, default=0., help="Tolerance, useful for 2-step methods (problem must be 'op'). "
                        "It gives some margin to reach the end depot on time")

    # Agents
    parser.add_argument('--num_agents', type=int, default=1, help="Number of agents")
    parser.add_argument('--num_dirs', type=int, default=4, help="Number of directions the agent can choose to move")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--dataset', type=str, default='', help="Path to load dataset. If more than one scenario in,"
                        "the dataset, only the first one is used. If None, a random one is generated")
    opts = parser.parse_args(args)

    # Use CUDA or CPU
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Check options are correct
    check_visualize_options(opts)
    return opts


def compute_benchmark(
        opts: argparse.Namespace,
        batch: torch.Tensor | dict,
        device: torch.types.Device,
        **kwargs) -> Tuple[np.ndarray, np.ndarray | dict, str, bool]:
    """
    Apply an algorithm from the benchmark to calculate the visualized path.

    Args:
        opts (argparse.Namespace): Options.
        batch (torch.Tensor or dict): Input batch.
        device (torch.device): The device (only for Neural A*)

    Returns:
        tuple: A tuple containing the actions, batch, model name, and success flag.
    """

    # Get baseline algorithms
    route_planner, path_planner = opts.model.split('-')
    runs, route_planner = parse_runs(route_planner)
    assert route_planner in ROUTE_PLANNERS, f"'{route_planner}' not in route planners list: {ROUTE_PLANNERS}"
    assert path_planner in PATH_PLANNERS,   f"'{path_planner}' not in route planners list: {PATH_PLANNERS}"
    route_name = {
        'ga': 'GA',
        'ortools': 'OR-Tools',
    }.get(route_planner)
    path_name = {
        'a_star': 'A*',
        'd_star': 'D*',
        'na_star': 'NA*',
    }.get(path_planner)
    model_name = route_name + '-' + path_name
    if path_planner == 'na_star':
        model = NAStar(encoder_arch='CNN').to(device)
        model.load_state_dict(load_from_ptl_checkpoint(
            "./benchmarks/nop/methods/neural-astar/model/mazes_032_moore_c8/lightning_logs/"
        ))
    elif path_planner == 'a_star':
        model = VAStar().to(device)
    else:
        model = None

    # Prepare inputs
    batch, dict_keys = batch2numpy(batch, to_list=True)

    # Calculate tours
    _, actions, _, success, _ = solve_nop(
        directory=None,
        instance_name=None,
        scenario=batch,
        route_planner=route_planner,
        path_planner=path_planner,
        disable_cache=False,
        sec_local_search=runs,
        model=model,
        eps = opts.eps,
    )

    # Lists to numpy arrays
    actions = np.array(actions)
    if dict_keys is None:
        batch = np.array(batch)
    else:
        new_batch = {}
        for i, k in enumerate(dict_keys):
            new_batch[k] = np.array(batch[i])
        batch = new_batch
    return actions, batch, model_name, success


def compute_network(
        opts: argparse.Namespace,
        batch: torch.Tensor | dict,
        env: Any,
        device: torch.types.Device,
        **kwargs) -> Tuple[np.ndarray, np.ndarray | dict, str, bool]:
    """
    Apply a neural network to calculate the visualized path.

    Args:
        opts (argparse.Namespace): Options.
        batch (torch.Tensor or dict): Input batch.
        env (Any): The environment.
        device (torch.device): The device.

    Returns:
        tuple: A tuple containing the actions, batch, model name, and success flag.
    """

    # Load model (Transformer, PN, GPN) for evaluation on the chosen device
    model, args = load_model_eval(opts.model)
    model.to(device)
    model_name = args['fancy_name']

    # Inputs to device
    batch = move_to(batch, device)

    # Calculate tours
    with torch.no_grad():
        _, _, actions, _ = model(batch, env)

    # Inputs to numpy
    batch, _ = batch2numpy(batch)

    # Tours to numpy
    end_ids = 0 if opts.num_depots == 1 else opts.num_nodes + 1
    actions = actions2numpy(actions, end_ids)
    success = True
    return actions, batch, model_name, success


def main(opts: argparse.Namespace) -> None:
    """
    Main function.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.
    """

    # Set the device
    device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)

    # Load env
    env = problem(
        batch_size=1,
        num_workers=0,
        device=device,
        num_nodes=opts.num_nodes,
        num_samples=1,
        filename=opts.dataset,
        data_dist=opts.data_dist,
        num_depots=opts.num_depots,
        max_length=opts.max_length,
        max_nodes=opts.max_nodes,
        num_obs=opts.num_obs,
        rad_obs=opts.rad_obs,
        num_dirs=opts.num_dirs,
        eps=opts.eps,
        desc='Load data'
    )

    # Load batch
    batch = next(iter(env.dataloader))

    # Use benchmark algorithm or trained model
    method = compute_network if os.path.exists(opts.model) else compute_benchmark

    # Apply algorithm
    actions, batch, model_name, success = method(opts=opts, batch=batch, env=env, device=device)

    # Print results
    if isinstance(actions, list):
        for i, action in enumerate(actions):
            print(f"Agent {i + 1} - Nodes: {action[0]}\n Directions: {action[1]}")
    else:
        if actions.shape[1] == 2:
            print(f"Nodes: {actions.transpose(1, 0)[0]}\n Directions: {actions.transpose(1, 0)[1]}")
        else:
            print(f"Nodes: {actions.transpose(1, 0)[0]}\n Directions: {actions[:, 1:]}")

    # Plot results
    if opts.num_agents == 1:
        actions = [actions]
    plot(actions, batch, env.name, model_name, data_dist=opts.data_dist, success=success, num_dirs=opts.num_dirs)
    

if __name__ == "__main__":
    main(get_options())
