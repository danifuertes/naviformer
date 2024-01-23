import torch
import random
import argparse
import numpy as np


def global_vars():
    return {
        'PROBLEMS': ['nop', 'op'],
        'DATA_DIST': ['const', 'unif', 'dist'],
        'MODELS': ['naviformer'],
        'NORM': ['batch', 'instance'],
        'BASELINES': ['rollout', 'critic', 'exponential', None],
        'ROUTE_PLANNERS': ['ga', 'ortools'],
        'PATH_PLANNERS': ['a_star', 'd_star']
    }


def str2bool(v):
    """
    Transform string inputs into boolean.
    :param v: string input.
    :return: string input transformed to boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
