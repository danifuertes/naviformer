import os
import time
import torch
import argparse

from utils.opt_utils import set_seed, str2bool, global_vars


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
MODELS = global_vars()['MODELS']
NORM = global_vars()['NORM']
BASELINES = global_vars()['BASELINES']


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="NaviFormer: a Transformer-like model to solve Combinatorial Optimization Problems, specially the"
                    "NOP (Navigation Orienteering Problem), with DRL (Deep Reinforcement Learning)."
    )
    parser.add_argument('--seed', type=int, default=1234, help="Random seed to use")

    # Problem
    parser.add_argument('--problem', type=str, default='nav_op', help=f"Problem to solve: {', '.join(PROBLEMS)}")
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
    parser.add_argument('--info_th', type=float, default=0.2, help="Min dist among agents to share info (DecPOMDP)")

    # Data
    parser.add_argument('--epoch_size', type=int, default=1280000, help="Number of instances per epoch during training")
    parser.add_argument('--train_dataset', type=str, default='', help="Dataset file to use for training")
    parser.add_argument('--val_dataset', type=str, default='', help="Dataset file to use for validation")
    parser.add_argument('--val_size', type=int, default=10000, help="Number of instances to report val performance")

    # Model
    parser.add_argument('--model', type=str, default='attention', help=f"Model: {', '.join(MODELS)}")
    parser.add_argument('--combined_mha', type=str2bool, default=True, help="Use combined/standard MHA encoder")
    parser.add_argument('--num_heads', type=int, default=8, help="Number of multi-heads for attention operations")
    parser.add_argument('--num_blocks', type=int, default=3, help="Number of blocks in the encoder/critic network")
    parser.add_argument('--hidden_dim', type=int, default=128, help="Dimension of hidden layers in Enc/Dec")
    parser.add_argument('--embed_dim', type=int, default=128, help="Dimension of input embedding")
    parser.add_argument('--normalization', default='batch', help="Normalization type, 'batch' or 'instance'")
    parser.add_argument('--tanh_clipping', type=float, default=10., help="Clip the parameters to within +- this value"
                        "using tanh. Set to 0 to not perform any clipping.")
    parser.add_argument('--two_step', type=str, default='', help=f"Path to base model for baseline 2-step approach")
    parser.add_argument('--two_step_train', type=str2bool, default=False, help=f"Train 2-step route planning model")
    parser.add_argument('--unfreeze_epoch', type=int, default=-1,
                        help=f"Epoch to unfreeze 2-step route planning model. Use -1 to keep frozen")

    # Training
    parser.add_argument('--batch_size', type=int, default=512, help="Number of instances per batch during training")
    parser.add_argument('--eval_batch_size', type=int, default=1024, help="Batch size during (baseline) evaluation")
    parser.add_argument('--epochs', type=int, default=100, help="The number of epochs to train")
    parser.add_argument('--first_epoch', type=int, default=0, help="Initial epoch (relevant for learning rate decay)")
    parser.add_argument('--lr_model', type=float, default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--lr_critic', type=float, default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--lr_decay', type=float, default=1.0, help="Learning rate decay per epoch")
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help="Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)")
    parser.add_argument('--exp_beta', type=float, default=0.8, help="Exponential moving average baseline decay")
    parser.add_argument('--checkpoint_enc', type=str2bool, default=False,
                        help="Set to decrease memory usage by checkpointing encoder")
    parser.add_argument('--shrink_size', type=int, default=None, help="Shrink the batch size if at least this many"
                        "instances in the batch are finished to save memory (default None means no shrinking)")

    # Baseline
    parser.add_argument('--baseline', type=str, default=None, help=f"Baseline to train with: ','.join({BASELINES})")
    parser.add_argument('--bl_alpha', type=float, default=0.05,
                        help="Significance in the t-test for updating rollout baseline")
    parser.add_argument('--bl_warmup', type=int, default=None,
                        help="Number of epochs to warmup the baseline, default None means 1 for rollout (exponential"
                             "used for warmup phase), 0 otherwise. Can only be used with rollout baseline.")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--eval_only', type=str2bool, default=False, help="Set this value to only evaluate model")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers loading data batches")
    parser.add_argument('--output_dir', default='outputs', help="Directory to write output models to")
    parser.add_argument('--checkpoint_epochs', type=int, default=1,
                        help="Save checkpoint every n epochs (default 1), 0 to save no checkpoints")
    parser.add_argument('--load_path', type=str, help="Path to load model parameters and optimizer state from")
    parser.add_argument('--resume', type=str, help="Resume from previous checkpoint file")
    parser.add_argument('--log_step', type=int, default=50, help="Log info every log_step steps")
    parser.add_argument('--use_tensorboard', type=str2bool, default=True, help="Log TensorBoard files")
    parser.add_argument('--use_progress_bar', type=str2bool, default=True, help="Use progress bar")
    opts = parser.parse_args(args)

    # Number of nodes
    opts.num_nodes = opts.max_nodes if opts.max_nodes > 0 else opts.num_nodes

    # Warmup epochs for baselines
    if opts.bl_warmup is None:
        opts.bl_warmup = 1 if opts.baseline == 'rollout' else 0

    # Check options are ok
    opts = check_options(opts)

    # Use CUDA or CPU
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Filenames
    opts.time_txt = time.strftime("%Y%m%dT%H%M%S")
    num_agents_str = f"_{opts.num_agents}agents" if opts.num_agents > 1 else ""
    num_depots_str = f"_{opts.num_depots}depots" if opts.num_depots > 1 else ""
    size_str = f"{opts.num_nodes}max" if opts.max_nodes > 0 else f"{opts.num_nodes}"
    data_str = f"{opts.data_dist}{size_str}"
    opts.save_dir = os.path.join(
        opts.output_dir,
        f"{opts.problem}_{data_str}",
        f"{opts.model}_{opts.baseline}{num_agents_str}{num_depots_str}_{opts.time_txt}"
    )

    # Set seed for reproducibility
    set_seed(opts.seed)
    return opts


def check_options(opts):
    assert opts.seed >= 0,              f"seed must be non-negative, found {opts.seed}"
    assert opts.problem,                f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert opts.num_agents > 0,         f"num_agents must be positive, found: {opts.num_agents}"
    assert 0 <= opts.info_th <= 1,      f"info_th must be in range [0, 1], found: {opts.info_th}"
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
    assert opts.hidden_dim > 0,         f"hidden_dim must be positive, found: {opts.hidden_dim}"
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
    assert opts.epoch_size % opts.batch_size == 0, f"epoch_size must be multiple of batch_size, found {opts.epoch_size}"
    assert opts.combined_mha and opts.max_obs or not opts.combined_mha, \
        f"If combined_mha=True, then it is required that max_obs > 0"
    if opts.shrink_size is not None:
        assert opts.shrink_size > 0,    f"shrink_size must be positive, found: {opts.shrink_size}"
    if os.path.isdir(opts.two_step) or opts.two_step.endswith('.pkl') or opts.two_step_train:
        if opts.two_step_train:
            assert opts.problem == 'op', "To train a 2-step route planner, try with option --problem op"
            opts.model = opts.model + '_2step'
    else:
        if opts.two_step != '':
            print(f"Baseline 2-step approach model {opts.two_step} not found. Using standard 1-step approach")
            opts.two_step = ''
        assert opts.unfreeze_epoch < 0, \
            "Positive unfreeze_epoch only allowed for 2-step approach. Try with --to_step or set unfreeze_epoch to -1"
    return opts
