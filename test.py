import os.path

from utils import *
from envs import load_problem, print_results


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']
DECODE_STRATEGY = global_vars()['DECODE_STRATEGY']


def get_options():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('datasets', nargs='+', help="Filename of the dataset(s) to evaluate")

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

    # Check options are correct
    assert opts.batch_size > 0,                      f"eval_batch_size must be positive, found: {opts.batch_size}"
    assert opts.num_workers >= 0,                    f"num_workers must be non negative, found: {opts.num_workers}"
    assert opts.decode_strategy in DECODE_STRATEGY,  f"'{opts.decode_strategy}' not in strategy list: {DECODE_STRATEGY}"
    assert opts.o is None or len(opts.datasets) == 1, "Cannot specify result filename with more than one dataset"

    # Use CUDA or CPU
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda
    return opts


def main(opts):

    # Device
    device = torch.device("cuda" if opts.use_cuda else "cpu")

    # Load model
    model, args = load_model_eval(opts.model, decode=opts.decode_strategy, temp=opts.softmax_temp)
    model.to(device)
    model_name = args['fancy_name']
    problem_name = args['problem']

    # Load problem
    problem = load_problem(problem_name)

    # Test on each dataset
    for dataset in opts.datasets:
        results = [[] for _ in range(5)]

        # Prepare data dir to save results
        dataset_name, ext = os.path.splitext(dataset.replace('/', '_'))
        if opts.o is None:
            results_dir = os.path.join(opts.results_dir, problem_name, dataset_name)
            os.makedirs(results_dir, exist_ok=True)
            out_file = os.path.join(results_dir, model_name)
        else:
            out_file = opts.o
        assert opts.f or not os.path.isfile(out_file), "File already exists! Try running with -f option to overwrite."

        # Load test env
        print(f"Test dataset {dataset}:")
        test_env = problem(
            batch_size=opts.batch_size,
            num_workers=opts.num_workers,
            device=device,
            filename=dataset,
            desc='Load data'
        )

        # Get batches from dataset
        for batch_id, batch in enumerate(tqdm(test_env.dataloader, desc='Testing'.ljust(15))):
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
            results[4] = [*results[4], batch['loc'].shape[1]]

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
