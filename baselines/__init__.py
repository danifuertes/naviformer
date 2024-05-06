import torch
import argparse
from typing import Any

from .basic import NoBaseline
from .warmup import WarmupBaseline
from .critic import CriticBaseline
from .rollout import RolloutBaseline
from .exponential import ExponentialBaseline
from nets.modules.critic import Critic


def load_baseline(opts: argparse.Namespace, model: torch.nn.Module, problem: Any, load_data: dict):
    """
    Load the specified baseline.

    Args:
        opts (argparse.Namespace): Options for loading the baseline.
        model (torch.nn.Module): The model.
        problem (str): The problem.
        load_data (dict): Data to load.

    Returns:
        Baseline: The loaded baseline.
    """

    # Exponential baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)

    # Critic baseline (Actor-Critic)
    elif opts.baseline == 'critic':
        baseline = CriticBaseline(
            Critic(
                input_dim1=3,
                input_dim2=3 if opts.num_obs[1] else None,
                embed_dim=opts.embed_dim,
                num_blocks=opts.num_blocks,
                normalization=opts.normalization,
                combined_mha=opts.combined_mha,
            ).to(opts.device)
        )

    # Rollout baseline
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)

    # No baseline
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    # Warmup baseline (for a few of the initial epochs)
    if opts.bl_warmup > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])
    return baseline
