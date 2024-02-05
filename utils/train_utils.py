import os
import torch
import numpy as np
from tqdm import tqdm


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def set_decode_type(model, decode_type):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


def load_lr_scheduler(opts, optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)


def load_optimizer(opts, model, baseline, load_data):

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


def resume_training(opts, model, baseline, load_data):

    # Resume training
    epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])
    torch.set_rng_state(load_data['rng_state'])
    if opts.use_cuda:
        torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])

    # Set the random states. Dumping of state was done before epoch callback, so do that now (model is loaded)
    baseline.epoch_callback(model, epoch_resume)
    print("Resuming after {}".format(epoch_resume))
    opts.epoch_start = epoch_resume + 1
    return opts, model, baseline


def clip_grad_norms(param_groups, max_norm=np.inf):
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


def rollout(model, env, desc=''):

    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    # Calculate rewards for each batch
    rewards = []
    for batch in tqdm(env.dataloader, desc=desc):
        batch = move_to(batch, device=env.device)
        with torch.no_grad():
            reward, _, _ = model(batch, env)
        rewards.append(reward.data.cpu())
    return torch.cat(rewards, dim=0)


def validate(model, env):
    cost = rollout(model, env, desc='Validating')
    avg_cost = cost.mean()
    print(f"Validation overall avg_cost: {avg_cost} +- {torch.std(cost) / np.sqrt(len(cost))}")
    return avg_cost
