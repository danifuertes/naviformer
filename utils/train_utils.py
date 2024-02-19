import torch
import argparse
import numpy as np
import torch.types
from tqdm import tqdm
from typing import Any, Tuple


def move_to(var: dict | torch.Tensor, device: torch.types.Device) -> torch.Tensor | dict:
    """
    Move tensor or dictionary of tensors to specified device.

    Args:
        var (torch.Tensor or dict): Tensor or dictionary of tensors to move to the device.
        device (torch.device): Device to move the tensor(s) to.

    Returns:
        torch.Tensor or dict: Moved tensor or dictionary of moved tensors.
    """
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def set_decode_type(model: torch.nn.Module, decode_type: str):
    """
    Set the decoding type for the model.

    Args:
        model (torch.nn.Module): The model.
        decode_type (str): The decoding strategy.
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def load_lr_scheduler(opts: argparse.Namespace, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Load the learning rate scheduler for the optimizer.

    Args:
        opts (argparse.Namespace): Options for the training.
        optimizer (torch.optim.Optimizer): The optimizer.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The learning rate scheduler.
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)


def load_optimizer(opts: argparse.Namespace, model: torch.nn.Module, baseline: Any, load_data: dict) -> \
        torch.optim.Optimizer:
    """
    Load the optimizer.

    Args:
        opts (argparse.Namespace): Options for the training.
        model (torch.nn.Module): The model.
        baseline: Baseline.
        load_data: Data to load.

    Returns:
        torch.optim.Optimizer: The optimizer.
    """

    # Load optimizer
    optimizer = torch.optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)
    return optimizer


def resume_training(opts: argparse.Namespace, model: torch.nn.Module, baseline: Any, load_data: dict, epoch: int = 0) \
        -> Tuple[argparse.Namespace, torch.nn.Module, Any]:
    """
    Resume training from a specific epoch.

    Args:
        opts (argparse.Namespace): Options for the training.
        model (torch.nn.Module): The model.
        baseline (Any): Baseline.
        load_data (dict): Data to load.
        epoch (int): The epoch to resume from.

    Returns:
        tuple: Options, model, and baseline after resuming training.
    """

    # Resume training
    torch.set_rng_state(load_data['rng_state'])
    if opts.use_cuda:
        torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

    # Set the random states. Dumping of state was done before epoch callback, so do that now (model is loaded)
    baseline.epoch_callback(model, epoch)
    print("Resuming after {}".format(epoch))
    opts.epoch_start = epoch + 1
    return opts, model, baseline


def clip_grad_norms(param_groups: list, max_norm: float = np.inf):
    """
    Clip gradients to a maximum norm.

    Args:
        param_groups (list): Parameter groups.
        max_norm (float): Maximum norm for clipping.

    Returns:
        tuple: Grad norms before and after clipping.
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else np.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def rollout(model: torch.nn.Module, env: Any, desc: str = '') -> torch.Tensor:
    """
    Perform rollout of the model.

    Args:
        model (torch.nn.Module): The model.
        env (Any): Environment.
        desc (str): Description.

    Returns:
        torch.Tensor: Rollout rewards.
    """

    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    # Calculate rewards for each batch
    rewards = []
    for batch in tqdm(env.dataloader, desc=desc):
        batch = move_to(batch, device=env.device)
        with torch.no_grad():
            reward, _, _, _ = model(batch, env)
        rewards.append(reward.data.cpu())
    return torch.cat(rewards, dim=0)


def validate(model: torch.nn.Module, env: Any) -> float:
    """
    Validate the model through rollouts.

    Args:
        model (torch.nn.Module): The model.
        env (Any): Environment.

    Returns:
        torch.Tensor: Average cost.
    """
    cost = rollout(model, env, desc='Validating')
    avg_cost = cost.mean()
    print(f"Validation overall avg_cost: {avg_cost} +- {torch.std(cost) / np.sqrt(len(cost))}")
    return avg_cost
