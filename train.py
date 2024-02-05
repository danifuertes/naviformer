from utils import *
from envs import load_problem
from baselines import load_baseline


def main(opts):

    # Tensorboard logger
    tb_logger = config_logger(opts)

    # Set device after Tensorboard logger, since torch device is not serializable by JSON
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")

    # Load problem
    problem = load_problem(opts.problem)

    # Load model
    model, load_data = load_model_train(opts)

    # Load baseline
    baseline = load_baseline(opts, model, problem, load_data)

    # Load optimizer
    optimizer = load_optimizer(opts, model, baseline, load_data)

    # Load learning rate scheduler, decay by lr_decay once per epoch
    lr_scheduler = load_lr_scheduler(opts, optimizer)

    # Resume training
    if opts.resume:
        opts, model, baseline = resume_training(opts, model, baseline, load_data)

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
        for epoch in range(opts.first_epoch, opts.first_epoch + opts.epochs):

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
            print("Start train epoch {}, lr={}".format(epoch, optimizer.param_groups[0]['lr']))
            for batch_id, batch in enumerate(tqdm(train_env.dataloader, desc='Training'.ljust(15))):
                batch = move_to(batch, device=opts.device)

                # Run episode
                reward, log_prob, _ = model(batch, train_env)

                # Run baseline episode
                reward_bl, loss_bl = baseline.eval(batch, reward, train_env)

                # Calculate loss function
                reinforce_loss = ((reward - reward_bl) * log_prob).mean()
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
                        reward, grad_norms, epoch, batch_id, step, log_prob, reinforce_loss, loss_bl, tb_logger, opts
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
