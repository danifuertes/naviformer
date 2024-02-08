import time

from utils import *
from envs import load_problem
from baselines import load_baseline


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
MODELS = global_vars()['MODELS']
NORM = global_vars()['NORM']
BASELINES = global_vars()['BASELINES']


def get_options(args=None):
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--embed_dim', type=int, default=128, help="Dimension of embeddings")
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
    opts = parser.parse_args(args)

    # Number of nodes
    opts.num_nodes = opts.max_nodes if opts.max_nodes > 0 else opts.num_nodes

    # Warmup epochs for baselines
    if opts.bl_warmup is None:
        opts.bl_warmup = 1 if opts.baseline == 'rollout' else 0

    # Check options are ok
    check_train_options(opts)

    # Two-step option
    if os.path.isdir(opts.two_step) or opts.two_step.endswith('.pkl') or opts.two_step_train:
        opts.model = opts.model + '_2step'
    else:
        if opts.two_step != '':
            print(f"Baseline 2-step approach model {opts.two_step} not found. Using standard 1-step approach")
            opts.two_step = ''

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


def main(opts):

    # Tensorboard logger
    tb_logger = config_logger(opts)

    # Set device after Tensorboard logger, since torch device is not serializable by JSON
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)

    # Load model
    model, load_data, first_epoch = load_model_train(opts)
    first_epoch = first_epoch if first_epoch > 0 else opts.first_epoch

    # Load baseline
    baseline = load_baseline(opts, model, problem, load_data)

    # Load optimizer
    optimizer = load_optimizer(opts, model, baseline, load_data)

    # Load learning rate scheduler, decay by lr_decay once per epoch
    lr_scheduler = load_lr_scheduler(opts, optimizer)

    # Resume training
    if opts.resume:
        opts, model, baseline = resume_training(opts, model, baseline, load_data, epoch=first_epoch)

    # Load validation env
    val_env = problem(
        batch_size=opts.eval_batch_size,
        num_workers=opts.num_workers,
        device=opts.device,
        num_nodes=opts.num_nodes,
        num_samples=opts.val_size,
        filename=opts.val_dataset,
        data_dist=opts.data_dist,
        num_depots=opts.num_depots,
        max_length=opts.max_length,
        max_nodes=opts.max_nodes,
        max_obs=opts.max_obs,
        desc='Validation data'
    )

    # Evaluate validation env
    if opts.eval_only:
        validate(model, val_env, opts)

    # Train
    else:
        for epoch in range(first_epoch, opts.first_epoch + opts.epochs):

            # Measure training time
            start_time = time.time()

            # Load training data
            train_env = problem(
                batch_size=opts.batch_size,
                num_workers=opts.num_workers,
                device=opts.device,
                baseline=baseline,
                num_nodes=opts.num_nodes,
                num_samples=opts.epoch_size,
                data_dist=opts.data_dist,
                num_depots=opts.num_depots,
                max_length=opts.max_length,
                filename=opts.train_dataset,
                max_nodes=opts.max_nodes,
                max_obs=opts.max_obs,
                desc='Train data'
            )

            # Current step
            step = epoch * train_env.num_steps

            # Tensorboard info
            if opts.use_tensorboard:
                tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

            # Put model in train mode
            model.train()
            set_decode_type(model, "sampling")

            # Unfreeze 2-step base route planner model if requested
            if epoch == opts.unfreeze_epoch:
                print('Unfreezing base route planner model layers for 2-step NaviFormer')
                for name, p in get_inner_model(model).base_route_model.named_parameters():
                    p.requires_grad = True

            # Train one epoch
            print(f"Start train epoch {epoch}, lr={optimizer.param_groups[0]['lr']}")
            for batch_id, batch in enumerate(tqdm(train_env.dataloader, desc='Training'.ljust(15))):
                batch = move_to(batch, device=opts.device)

                # Run episode
                rewards, log_prob, _, _ = model(batch, train_env)

                # Run baseline episode
                rewards_bl, loss_bl = baseline.eval(batch, rewards, train_env)

                # Calculate loss function
                reinforce_loss = ((rewards - rewards_bl) * log_prob).mean()
                loss = reinforce_loss + loss_bl

                # Perform backward pass
                optimizer.zero_grad()
                loss.backward()

                # Clip gradient norms and get (clipped) gradient norms for logging
                grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)

                # Perform optimization step
                optimizer.step()

                # Logging
                if step % int(opts.log_step) == 0:
                    log_values(
                        rewards, grad_norms, epoch, batch_id, step, log_prob, reinforce_loss, loss_bl, tb_logger, opts
                    )
                step += 1

            # Measure training time and report results
            epoch_duration = time.time() - start_time
            print(f"Finished epoch {epoch}, took {time.strftime('%H:%M:%S', time.gmtime(epoch_duration))}")

            # Save trained model
            if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.epochs - 1:
                print('Saving model and state...')
                save_model(model, optimizer, baseline, opts.save_dir, epoch)

            # Validate
            avg_reward = validate(model, val_env)

            # Tensorboard info
            if not opts.use_tensorboard:
                tb_logger.log_value('val_avg_reward', avg_reward, step)

            # Update callback
            baseline.epoch_callback(model, epoch)

            # Update lr_scheduler
            lr_scheduler.step()


if __name__ == "__main__":
    main(get_options())
    print('Finished')
