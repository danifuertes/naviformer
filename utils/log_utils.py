import argparse
import os
import json

import torch
from tensorboard_logger import Logger as TbLogger


def config_logger(opts: argparse.Namespace) -> TbLogger | None:
    """
    Prints and saves arguments into a json file. Optionally sets up TensorBoard logging.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        TbLogger or None: TensorBoard logger instance if enabled, otherwise None.
    """
    tb_logger = None
    if not opts.eval_only:

        # Save arguments so exact configuration can always be found
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
        with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
            json.dump(vars(opts), f, indent=True)

        # Optionally configure tensorboard
        tb_dir = os.path.join(opts.save_dir, 'log_dir')
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        if opts.use_tensorboard:
            tb_logger = TbLogger(tb_dir)

    # Print opts
    for k, v in vars(opts).items():
        print("'{}': {}".format(k, v))
    print()
    return tb_logger


def log_values(
        cost: torch.Tensor,
        grad_norms: tuple,
        epoch: int,
        batch_id: int,
        step: int,
        log_likelihood: torch.Tensor,
        reinforce_loss: torch.Tensor,
        loss_bl: torch.Tensor,
        tb_logger: TbLogger | None,
        opts: argparse.Namespace) -> None:
    """
    Log values during training to the console and optionally to TensorBoard.

    Args:
        cost (Tensor): Cost values.
        grad_norms (tuple): Tuple containing gradient norms before and after clipping.
        epoch (int): Current epoch number.
        batch_id (int): Batch ID.
        step (int): Current step number.
        log_likelihood (Tensor): Log likelihood values.
        reinforce_loss (Tensor): Reinforce loss values.
        loss_bl (Tensor): Baseline loss values.
        tb_logger (TbLogger or None): TensorBoard logger instance.
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """

    # Get average cost
    avg_cost = cost.mean().item()

    # Get gradients
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print(f" Epoch: {epoch}, batch_id: {batch_id}, avg_cost: {avg_cost:.4f}, loss: {reinforce_loss:.4f}")
    print(f"grad_norm: {grad_norms[0]:.4f}, clipped: {grad_norms_clipped[0]:.4f}")

    # Log values to tensorboard
    if opts.use_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)
        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)
        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)

        # Log critic-related values to tensorboard
        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', loss_bl.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
