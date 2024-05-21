import time

from utils import *
from nets import load_model_eval
from envs import load_problem, print_results


def get_options() -> argparse.Namespace:
    """
    Parse command line arguments to configure testing options for evaluating the neural network model.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('datasets', nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--eps', type=float, default=0., help="Tolerance, useful for 2-step methods (problem must be 'op'). "
                        "It gives some margin to reach the end depot on time")

    # Model
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--decode_strategy', type=str, default='greedy', help="Sampling (sample) or Greedy (greedy)")
    parser.add_argument('--softmax_temp', type=parse_softmax_temp, default=1, help="Softmax temperature (sampling)")

    # Results
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument('-f', action='store_true', help="Set true to overwrite")
    parser.add_argument('-o', default=None, help="Name of the results file to write")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size")
    parser.add_argument('--num_workers', type=int, default=16, help="Number of parallel workers loading data batches")
    opts = parser.parse_args()

    # Use CUDA or CPU
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

    # Check options are correct
    check_test_options(opts)
    return opts


def main(opts):
    """
    Main function to evaluate the neural network model on given datasets.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """

    # Device
    device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Load model
    model, args = load_model_eval(opts.model, decode=opts.decode_strategy, temp=opts.softmax_temp)
    model.to(device)
    model_name = args['fancy_name']
    problem_name = args['problem']
    num_dirs = args['num_dirs']

    # Load problem
    problem = load_problem(problem_name)

    # Test on each dataset
    for dataset_path in opts.datasets:

        # Output filename to save results
        out_file, out_dir = get_results_file(opts, dataset_path, problem_name, model_name)
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(check_extension(out_file)) and not opts.f:
            results = load_dataset(out_file)
        else:
            results = [[] for _ in range(5)]

            # Load test env
            print(f"Test dataset {dataset_path}:")
            test_env = problem(
                batch_size=opts.batch_size,
                num_workers=opts.num_workers,
                device=device,
                filename=dataset_path,
                num_dirs=num_dirs,
                eps=opts.eps,
                desc='Load data'
            )

            # Get batches from dataset
            for batch in tqdm(test_env.dataloader, desc='Testing'.ljust(15)):
                batch = move_to(batch, device=device)

                # Run episode
                with torch.no_grad():
                    start = time.time()
                    rewards, _, actions, success = model(batch, test_env)
                    duration = time.time() - start

                # Collect results
                results[0] = [*results[0], *rewards.tolist()]
                results[1] = [*results[1], *actions.tolist()]
                results[2] = [*results[2], *success.tolist()]
                results[3] = [*results[3], duration]
                results[4] = [*results[4], *(batch['loc'][..., 0] > 0).sum(dim=1).detach().cpu().numpy().tolist()]

            # Add parallelism info to results
            parallelism = opts.batch_size
            results.append(parallelism)

        # Print results
        print_results(problem_name)(results)

        # Save results
        save_dataset(results, out_file)
    print('Finished')


if __name__ == "__main__":
    main(get_options())
