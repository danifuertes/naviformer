import os
import torch
import argparse
import numpy as np

from envs import load_problem
from utils import load_model_eval, global_vars, str2bool, set_seed, plot, batch2numpy, actions2numpy, move_to


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
MODELS = global_vars()['MODELS']
ROUTE_PLANNERS = global_vars()['ROUTE_PLANNERS']
PATH_PLANNERS = global_vars()['PATH_PLANNERS']


def arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed to use')

    # Model
    parser.add_argument('model', type=str, help="Path to trained model. For benchmarks, combine route and path"
                                                "planners with '-'. Example: ga-a_star")

    # Problem
    parser.add_argument('--problem', type=str, default='nop', help=f"Problem to solve: {', '.join(PROBLEMS)}")
    parser.add_argument('--num_nodes', type=int, default=20, help="Number of visitable nodes")
    parser.add_argument('--max_nodes', type=int, default=0, help='Max number of nodes (random num_nodes padded with'
                        'dummys until max_nodes). Max length depends on the random num_nodes. Disable with max_nodes=0')
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. 1"
                        "means start and end depot are the same. 2 means they are different")
    parser.add_argument('--max_length', type=float, default=2., help="Normalized time limit to solve the problem")
    parser.add_argument('--data_dist', type=str, default='const',
                        help=f"Data distribution (reward values of regions) of OP. Options: {', '.join(DATA_DIST)}")
    parser.add_argument('--max_obs', type=int, default=0,
                        help='Max number of obstacles (random number of obstacles padded with dummys)')

    # Agents
    parser.add_argument('--num_agents', type=int, default=1, help="Number of agents")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--dataset', type=str, default='', help='Path to load dataset. If more than one scenario in,'
                        'the dataset, only the first one is used. If None, a random one is generated')
    opts = parser.parse_args(args)
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Check options are correct
    assert opts.seed >= 0,              f"seed must be non-negative, found {opts.seed}"
    assert opts.problem in PROBLEMS,    f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert opts.num_agents > 0,         f"num_agents must be positive, found: {opts.num_agents}"
    assert opts.num_nodes > 0,          f"num_nodes must be positive, found: {opts.num_nodes}"
    assert opts.max_nodes >= 0,         f"max_nodes must be non-negative, found: {opts.max_nodes}"
    assert opts.num_depots in [1, 2],   f"num_depots must be 1 or 2, found: {opts.num_depots}"
    assert opts.max_length > 0,         f"max_length must be positive, found: {opts.max_length}"
    assert opts.data_dist in DATA_DIST, f"'{opts.data_dist}' not in data_dist list: {DATA_DIST}"
    assert opts.max_obs >= 0,           f"max_obs must be non-negative, found: {opts.max_obs}"

    # Set seed for reproducibility
    set_seed(opts.seed)
    return opts


def baselines(baseline, inputs):

    # Get baseline algorithms
    route_planner, path_planner = baseline.split('-')
    runs, route_planner = get_runs(route_planner)
    assert route_planner in ROUTE_PLANNERS, f"'{route_planner}' not in route planners list: {ROUTE_PLANNERS}"
    assert path_planner in PATH_PLANNERS,   f"'{path_planner}' not in route planners list: {PATH_PLANNERS}"
    route_name = {
        'ga': 'GA',
        'ortools': 'OR-Tools'
    }.get(route_planner)
    path_name = {
        'a_star': 'A*',
        'd_star': 'D*'
    }.get(path_planner)
    model_name = route_name + '-' + path_name

    # Prepare inputs
    inputs = inputs2numpy(inputs, True)

    # Calculate tours
    _, tour, _, success = solve_nav_op(
        directory=None,
        name=None,
        depot=inputs['depot'],
        loc=inputs['loc'],
        prize=inputs['prize'],
        max_length=inputs['max_length'],
        depot2=inputs['depot2'],
        obs=inputs['obs'],
        route_planner=route_planner,
        path_planner=path_planner,
        disable_cache=False,
        sec_local_search=runs
    )

    # Lists to numpy arrays
    if isinstance(inputs, dict):
        for k, v in inputs.items():
            inputs[k] = np.array(v)
    else:
        inputs = np.array(inputs)
    return np.array(tour).squeeze()[None], inputs, model_name, success


def main(opts):

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
        max_obs=opts.max_obs,
        desc='Load data'
    )

    # Load batch
    batch = next(iter(env.dataloader))

    # Use baseline instead of a trained model
    if not os.path.exists(opts.model):
        actions, batch, model_name, success = baselines(opts.model, batch)

    # Use trained model
    else:

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
        batch = batch2numpy(batch)

        # Tours to numpy
        end_ids = 0 if opts.num_depots == 1 else opts.num_nodes + 1
        actions = actions2numpy(actions, end_ids)
        success = True

    # Print results
    if isinstance(actions, list):
        for i, action in enumerate(actions):
            print(f"Agent {i + 1} - Nodes: {action[0]}\n Directions: {action[1]}")
    else:
        print(f"Nodes: {actions.transpose(1, 0)[0]}\n Directions: {actions.transpose(1, 0)[1]}")

    # Plot results
    if opts.num_agents == 1:
        actions = [actions]
    plot(actions, batch, env.name, model_name, data_dist=opts.data_dist, success=success)


if __name__ == "__main__":
    main(arguments())
