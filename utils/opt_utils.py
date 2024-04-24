import os
import torch
import random
import argparse
import numpy as np


def global_vars() -> dict:
    """
    Return a dictionary containing global variables used in the utilities.

    Returns:
        dict: A dictionary containing global variables.
    """
    return {
        'PROBLEMS': ['nop', 'op'],
        'DATA_DIST': ['const', 'unif', 'dist'],
        'MODELS': ['naviformer', 'pn', 'gpn'],
        'NORM': ['batch', 'instance'],
        'DECODE_STRATEGY': ['greedy', 'sample'],
        'BASELINES': ['rollout', 'critic', 'exponential', None],
        'ROUTE_PLANNERS': ['ga', 'ortools'],
        'PATH_PLANNERS': ['a_star', 'd_star']
    }


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
MODELS = global_vars()['MODELS']
NORM = global_vars()['NORM']
DECODE_STRATEGY = global_vars()['DECODE_STRATEGY']
BASELINES = global_vars()['BASELINES']
ROUTE_PLANNERS = global_vars()['ROUTE_PLANNERS']
PATH_PLANNERS = global_vars()['PATH_PLANNERS']


def str2bool(v: str) -> bool:
    """
    Transform string inputs into boolean.

    Args:
        v (str): The string input.

    Returns:
        bool: The string input transformed to boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed: int) -> None:
    """
    Set the seed for various random number generators for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def check_make_data_options(opts: argparse.Namespace) -> None:
    """
    Check if the options for creating data are valid.

    Args:
        opts (argparse.Namespace): The parsed command-line arguments.
    """
    assert opts.seed >= 0,                          f"seed must be non-negative, found {opts.seed}"
    assert opts.problem in PROBLEMS,                f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert np.all(np.array(opts.num_nodes) > 0),    f"num_nodes must be positive, found: {opts.num_nodes}"
    assert opts.num_depots in [1, 2],               f"num_depots must be 1 or 2, found: {opts.num_depots}"
    assert np.all(np.array(opts.max_length) > 0),   f"max_length must be positive, found: {opts.max_length}"
    assert opts.max_obs >= 0,                       f"max_obs must be non-negative, found: {opts.max_obs}"
    assert opts.max_nodes == 0 or opts.max_nodes >= 10, \
        f"max_nodes must be non-negative and considerably large (>= 10), found: {opts.max_nodes}"
    assert len(opts.num_nodes) == len(opts.max_length), \
        f"num_nodes and max_length must have same length, found {opts.num_nodes} and {opts.max_length}"
    for dist in opts.data_dist:
        assert dist in DATA_DIST,                   f"'{dist}' not in data_dist list: {DATA_DIST}"


def check_visualize_options(opts: argparse.Namespace) -> None:
    """
    Check if the options for visualization are valid.

    Args:
        opts (argparse.Namespace): The parsed command-line arguments.
    """
    assert opts.seed >= 0,              f"seed must be non-negative, found {opts.seed}"
    assert opts.problem in PROBLEMS,    f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert opts.num_agents > 0,         f"num_agents must be positive, found: {opts.num_agents}"
    assert opts.num_nodes > 0,          f"num_nodes must be positive, found: {opts.num_nodes}"
    assert opts.max_nodes >= 0,         f"max_nodes must be non-negative, found: {opts.max_nodes}"
    assert opts.num_depots in [1, 2],   f"num_depots must be 1 or 2, found: {opts.num_depots}"
    assert opts.max_length > 0,         f"max_length must be positive, found: {opts.max_length}"
    assert opts.data_dist in DATA_DIST, f"'{opts.data_dist}' not in data_dist list: {DATA_DIST}"
    assert opts.max_obs >= 0,           f"max_obs must be non-negative, found: {opts.max_obs}"


def check_nop_benchmark_options(opts: argparse.Namespace) -> None:
    """
    Check if the options for NOP benchmarking are valid.

    Args:
        opts (argparse.Namespace): The parsed command-line arguments.
    """
    assert opts.route_planner in ROUTE_PLANNERS, f"'{opts.route_planner}' not in route planners list: {ROUTE_PLANNERS}"
    assert opts.path_planner in PATH_PLANNERS,   f"'{opts.path_planner}' not in route planners list: {PATH_PLANNERS}"
    assert opts.o is None or len(opts.datasets) == 1, f"Cannot specify result filename with more than one dataset"
    assert opts.problem, f"'{opts.problem}' not in problem list: {PROBLEMS}"
    if opts.n is not None:
        assert opts.n > 0,                           f"n must be positive, found: {opts.n}"
    if opts.offset is not None:
        assert opts.offset > 0,                      f"offset must be positive, found: {opts.offset}"
    if opts.n is not None and opts.offset is not None:
        assert opts.n > opts.offset,                 f"n must be > offset, found n={opts.n} and offset={opts.offset}"


def check_test_options(opts: argparse.Namespace) -> None:
    """
    Check if the options for testing are valid.

    Args:
        opts (argparse.Namespace): The parsed command-line arguments.
    """
    assert opts.batch_size > 0,                      f"batch_size must be positive, found: {opts.batch_size}"
    assert opts.num_workers >= 0,                    f"num_workers must be non negative, found: {opts.num_workers}"
    assert opts.decode_strategy in DECODE_STRATEGY,  f"'{opts.decode_strategy}' not in strategy list: {DECODE_STRATEGY}"
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"


def check_train_options(opts: argparse.Namespace) -> None:
    """
    Check if the options for training are valid.

    Args:
        opts (argparse.Namespace): The parsed command-line arguments.
    """
    assert opts.seed >= 0,              f"seed must be non-negative, found {opts.seed}"
    assert opts.problem,                f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert opts.num_agents > 0,         f"num_agents must be positive, found: {opts.num_agents}"
    assert opts.num_nodes > 0,          f"num_nodes must be positive, found: {opts.num_nodes}"
    assert opts.max_nodes >= 0,         f"max_nodes must be non-negative, found: {opts.max_nodes}"
    assert opts.num_depots in [1, 2],   f"num_depots must be 1 or 2, found: {opts.num_depots}"
    assert opts.max_length > 0,         f"max_length must be positive, found: {opts.max_length}"
    assert opts.data_dist in DATA_DIST, f"'{opts.data_dist}' not in data_dist list: {DATA_DIST}"
    assert opts.max_obs >= 0,           f"max_obs must be non-negative, found: {opts.max_obs}"
    assert opts.epoch_size > 0,         f"epoch_size must be positive, found: {opts.epoch_size}"
    assert opts.val_size > 0,           f"val_size must be positive, found: {opts.val_size}"
    assert opts.model in MODELS,        f"{opts.model} not in model list: {MODELS}"
    assert opts.num_blocks > 0,         f"num_blocks must be positive, found: {opts.num_blocks}"
    assert opts.embed_dim > 0,          f"embed_dim must be positive, found: {opts.embed_dim}"
    assert opts.normalization in NORM,  f"{opts.normalization} not in normalization list: {NORM}"
    assert opts.tanh_clipping >= 0,     f"tanh_clipping must be non-negative, found: {opts.tanh_clipping}"
    assert opts.batch_size > 0,         f"batch_size must be positive, found: {opts.batch_size}"
    assert opts.eval_batch_size > 0,    f"eval_batch_size must be positive, found: {opts.eval_batch_size}"
    assert opts.epochs > 0,             f"epochs must be positive, found: {opts.epochs}"
    assert opts.first_epoch >= 0,       f"first_epoch must be non_negative, found: {opts.first_epoch}"
    assert opts.lr_model > 0,           f"lr_model must be positive, found: {opts.lr_model}"
    assert opts.lr_critic > 0,          f"lr_critic must be positive, found: {opts.lr_critic}"
    assert opts.lr_decay > 0,           f"lr_decay must be positive, found: {opts.lr_decay}"
    assert opts.exp_beta > 0,           f"exp_beta must be positive, found: {opts.exp_beta}"
    assert opts.max_grad_norm >= 0,     f"max_grad_norm must be non negative, found: {opts.max_grad_norm}"
    assert opts.baseline in BASELINES,  f"{opts.baseline} not in normalization list: {BASELINES}"
    assert opts.bl_alpha > 0,           f"bl_alpha must be positive, found: {opts.bl_alpha}"
    assert opts.num_workers >= 0,       f"num_workers must be non negative, found: {opts.num_workers}"
    assert opts.checkpoint_epochs >= 0, f"checkpoint_epochs must be non negative, found: {opts.checkpoint_epochs}"
    assert opts.log_step >= 0,          f"log_step must be non negative, found: {opts.log_step}"
    assert opts.bl_warmup >= 0,         f"bl_warmup must be non negative, found: {opts.bl_warmup}"
    assert (opts.bl_warmup == 0) or (opts.baseline == 'rollout'), f"bl_warmup only for baseline=rollout"
    # assert opts.epoch_size % opts.batch_size == 0, f"epoch_size must be multiple of batch_size, found {opts.epoch_size}"
    assert opts.combined_mha and opts.max_obs or not opts.combined_mha, \
        f"If combined_mha=True, then it is required that max_obs > 0"
    if os.path.isdir(opts.two_step) or opts.two_step.endswith('.pkl') or opts.two_step_train:
        assert opts.two_step_train and opts.problem == 'op', \
            "To train a route planner for future use of 2-step model, try with option --problem op"
    else:
        assert opts.unfreeze_epoch < 0, \
            "Positive unfreeze_epoch only allowed for 2-step approach. Try with --to_step or set unfreeze_epoch to -1"
