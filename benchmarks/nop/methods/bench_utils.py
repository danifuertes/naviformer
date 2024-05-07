import os
import re
import time
import argparse

import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Any
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

from .a_star import AStar
from .d_star import DStar
from .d_star_lite import DStarLite
from .na_star import NeuralAStar
from .ortools import solve_op_ortools
from .genetic import solve_op_genetic
from utils import save_dataset, load_dataset


def calc_cost_dist(tour: np.ndarray, coords: np.ndarray, prize: np.ndarray, end_id: int, success: bool) -> \
        Tuple[float, float]:
    """
    Calculate cost (negative reward) and distance for a tour.

    Args:
        tour (numpy.ndarray): Tour.
        coords (numpy.ndarray): Coordinates.
        prize (numpy.ndarray): Prize values.
        end_id (int): ID of the end.
        success (bool): Whether the tour is successful.

    Returns:
        tuple: Cost and distance.
    """
    cost = 0
    distance = 0
    for t in np.unique(tour):
        if t == 0 or t == end_id:
            continue  # Skip depots
        cost += 10 * prize[int(t) - 1] / (len(prize) / 2)
    prev = None
    for coord in coords:
        if prev is None:
            prev = coord
            continue
        d = np.linalg.norm(coord - prev)
        cost -= 0.3 * d
        distance += d
        prev = coord
    cost += 20 if success else -10
    return -cost, distance


def parse_runs(method: str) -> Tuple[int, str]:
    """
    Parse the method name to get the number of runs and the method itself.

    Args:
        method (str): Method name.

    Returns:
        tuple: Number of runs and method.
    """
    match = re.match(r'^([a-z]+)(\d*)$', method)
    assert match
    method = match[1]
    runs = 0 if match[2] == '' else int(match[2])
    return runs, method


def multiprocessing(
        opts: argparse.Namespace,
        func: Any,
        directory: str,
        dataset: Any,
        use_multiprocessing: bool = True,
        **kwargs) -> Tuple[list, int]:
    """
    Perform multiprocessing for function execution.

    Args:
        opts (argparse.Namespace): Parsed command-line arguments.
        func (Any): Function to execute.
        directory (str): Directory path.
        dataset (Any): Dataset.
        use_multiprocessing (bool): Whether to use multiprocessing.

    Returns:
        tuple: Results and number of CPUs employed.
    """
    directory = os.path.join(Path(__file__).parent.parent.parent, directory)

    # Kwargs to args
    args = [v for v in kwargs.values()]

    # Get number of CPUs
    num_cpus = os.cpu_count() if opts.cpus is None else opts.cpus

    # Get dataset
    w = len(str(len(dataset) - 1))
    offset = getattr(opts, 'offset', None)
    if offset is None:
        offset = 0
    ds = dataset[offset:(offset + opts.n if opts.n is not None else len(dataset))]

    # Use ThreadPool to calculate results
    pool_cls = (Pool if use_multiprocessing and num_cpus > 1 else ThreadPool)
    with pool_cls(num_cpus) as pool:
        results = list(tqdm(pool.imap(
            func,
            [(directory, str(i + offset).zfill(w), problem, *args) for i, problem in enumerate(ds)]
        ), total=len(ds), mininterval=opts.progress_bar_mininterval))

    # Return results
    failed = [str(i + offset) for i, res in enumerate(results) if res is None]
    assert len(failed) == 0, "Some instances failed: {}".format(" ".join(failed))
    return results, num_cpus


def path_planning(obs: np.ndarray, method: str = 'a_star', scale: int = 100, margin: int = 5) -> Any:
    """
    Perform (low-level) path planning.

    Args:
        obs (np.ndarray): List of obstacles.
        method (str): Method name for path planning.
        scale (int): Scale factor for obstacle map.
        margin (int): Margin value for obstacle map.

    Returns:
        Any: Path Planner.
    """

    # A*
    if method == 'a_star':
        # grid_size, robot_radius = 2, 2
        # planner = AStar(obs, margin=margin, scale=scale, resolution=grid_size, rr=robot_radius)
        planner = AStar(obs, scale=scale, grid_size=(scale, scale))

    # D*
    elif method == 'd_star':
        planner = DStar(obs, margin=margin, scale=scale)
        
    # Neural A*
    elif method == 'na_star':
        planner = NeuralAStar(obs, scale=scale, grid_size=(scale, scale))

    # D* Lite
    else:
        assert method == 'd_star_lite', 'Path planner not in list: [a_star, d_lite, na_star, d_star_lite]'
        planner = DStarLite(obs, [0, 0], [0, 0])
    return planner


def route_planning(
        depot_ini: list,
        loc: list,
        prize: list,
        max_length: float,
        depot_end: list,
        id_map: list,
        end_id: int,
        method: str = 'ortools',
        sec_local_search: int = 0,
        eps: float = 0.) -> Tuple[list, int, int]:
    """
    Perform (high-level) route planning.

    Args:
        depot_ini (list): Initial depot coordinates.
        loc (list): Location coordinates.
        prize (list): Prize values.
        max_length (float): Maximum length.
        depot_end (list): End depot coordinates.
        id_map (list): ID map.
        end_id (int): index of the end depot.
        method (str): Method name.
        sec_local_search (int): local search seconds.
        eps (float): Epsilon value to avoid surpassing `max_length`.

    Returns:
        tuple: Goal coordinates, goal index, and on depot status.
    """

    # Genetic Algorithm
    if method == 'genetic':
        tour = solve_op_genetic(  # Index 0 = start depot | Index 1 = end depot
            [(*pos, p) for p, pos in zip([0, 0] + prize, [depot_ini, depot_end] + loc)],
            max_length - eps, return_sol=True, verbose=False
        )[1] if len(loc) > 1 else [(*depot_ini, 0, 0, 0), (*depot_end, 0, 1, 0)]
        next_node = tour[1][3]
        goal_id = (end_id if next_node == 1 else next_node) if next_node <= 1 else id_map[next_node - 2] + 1
        goal_coords = tour[1][:2]
        on_depot = next_node == 1
        delete_id = next_node - 2

    # OR-Tools
    else:
        assert method == 'ortools', 'Route planner not in list: [ga, ortools]'
        _, tour = solve_op_ortools(
            depot_ini, loc, prize, max_length - eps, sec_local_search=sec_local_search, depot2=depot_end
        )
        next_node = tour[1]
        goal_id = end_id if next_node == len(loc) + 1 else (0 if next_node == 0 else id_map[next_node - 1] + 1)
        goal_coords = ([depot_ini] + loc + [depot_end])[next_node]
        on_depot = goal_id == end_id
        delete_id = next_node - 1
    if delete_id != len(loc):
        loc.pop(delete_id)
        prize.pop(delete_id)
        id_map.pop(delete_id)
    return goal_coords, goal_id, on_depot


def solve_nop(directory: str | None,
              instance_name: str | None,
              scenario: list,
              route_planner: str = 'ortools',
              path_planner: str = 'a_star',
              disable_cache: bool = False,
              sec_local_search: int = 0) -> Tuple[float, list, float, bool, int]:
    """
    Solve the Navigation Orienteering Problem (NOP).

    Args:
        directory (str or None): Directory path.
        instance_name (str or None): Name of the instance (typically a number, such as '001', '002', etc.).
        scenario (list): Scenario information.
        route_planner (str): Route planner.
        path_planner (str): Path planner.
        disable_cache (bool): Whether to disable caching.
        sec_local_search (int): Secondary local search.

    Returns:
        tuple: Cost, tour (list of coordinates), duration, and success.
    """
    depot_end, depot_ini, loc, max_length, obs, prize = scenario

    # Filename
    problem_filename = os.path.join(
        directory, "{}.{}-{}{}.pkl".format(
            instance_name, route_planner, path_planner, sec_local_search if sec_local_search > 0 else ''
        )
    ) if directory is not None or instance_name is not None else ''

    # Get results from cache
    if os.path.isfile(problem_filename) and not disable_cache:
        (cost, nav, success, duration, num_nodes) = load_dataset(problem_filename)

    # Calculate results
    else:
        num_nodes = len(loc)

        # Upscale scenario for path planner
        scale = 200
        eps = 0.1 * scale
        depot_ini, depot_end = scale * np.array(depot_ini), scale * np.array(depot_end)
        loc = (scale * np.array(loc)).astype(int).tolist()
        obs = (scale * np.array(obs))
        obs[obs == -scale] = -1
        obs = obs.tolist()
        max_length_copy = max_length
        max_length *= scale
        prize_copy = prize.copy()

        # Initialize parameters
        planner = path_planning(obs, method=path_planner, scale=scale)
        finished, success, start, nav = False, False, depot_ini, np.array([[0, *(depot_ini / scale)]])
        id_map, end_id = [m for m in range(len(loc))], len(loc) + 1

        # Iterate until episode finishes
        time_start = time.time()
        while not finished:

            # Route planning
            goal_coords, goal_id, on_depot = route_planning(
                start.tolist(), loc, prize, max_length, depot_end.tolist(), id_map, end_id, method=route_planner,
                sec_local_search=sec_local_search, eps=eps
            ) if max_length - eps >= np.linalg.norm(depot_end - start) else (depot_end, end_id, True)

            # Path planning
            rx, ry, success_path = planner.planning(*start, *goal_coords, limit=max_length)
            if path_planner == 'd_star':
                planner.set_obs_map()
            steps = np.concatenate(([start], np.stack((rx, ry), axis=1), [goal_coords]), axis=0)
            error = len(rx) == 1 and np.linalg.norm(start - goal_coords) > planner.resolution / scale
            start = np.array(goal_coords)

            # Update data
            max_length -= np.sum(np.linalg.norm(steps[1:] - steps[:-1], axis=1))
            nav = np.concatenate((
                nav, np.concatenate((np.repeat(goal_id, len(steps))[:, None], steps / scale), axis=1)
            ), axis=0)

            # Check if finished
            bumped = np.any([np.any(np.linalg.norm(np.array(ob[:2]) - steps, axis=1) < ob[2]) for ob in obs]) or error
            success = not bumped and success_path  # max_length >= np.linalg.norm(depot_end - start) and
            finished = on_depot or not success

        # Measure clock time
        duration = time.time() - time_start

        # Calculate cost/reward
        cost, distance = calc_cost_dist(nav[..., 0], nav[..., 1:], prize_copy, end_id, success)
        success = success if distance <= max_length_copy else False

        # Save results
        nav = nav.tolist()
        if directory is not None:
            save_dataset((cost, nav, success, duration, num_nodes), problem_filename)
    return cost, nav, success, duration, num_nodes
