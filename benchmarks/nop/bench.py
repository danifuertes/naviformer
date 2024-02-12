from utils import *
from envs import print_results
from benchmarks import parse_runs, multiprocessing, solve_nop


PROBLEMS = global_vars()['PROBLEMS']


def get_options() -> argparse.Namespace:
    """
    Parse command line arguments and return options.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()

    # Problem
    parser.add_argument('--problem', type=str, default='nop', help=f"Problem to solve: {', '.join(PROBLEMS)}")

    # Input data
    parser.add_argument("--datasets", nargs='+', help="Filename of the dataset(s) to evaluate")
    parser.add_argument('--offset', type=int, help="Offset where to start processing")
    parser.add_argument('-n', type=int, help="Number of instances to process")

    # Output data
    parser.add_argument('--results_dir', default='results', help="Name of results directory")
    parser.add_argument("-o", default=None, help="Name of the results file to write")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")

    # Algorithms
    parser.add_argument("--route_planner", help="Name of the route planner to evaluate: [ortools, opga]")
    parser.add_argument("--path_planner", help="Name of the path planner to evaluate: [a_star, d_star_lite]")

    # Misc
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--multiprocessing', type=str2bool, default=False, help='Use multiprocessing')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    opts = parser.parse_args()

    # Get chosen local search runs
    opts.runs, opts.route_planner = parse_runs(opts.route_planner)

    # Check options are correct
    check_nop_benchmark_options(opts)
    return opts


def run_func(args, **kwargs) -> Tuple:
    """
    Initialize the run function for multiprocessing.

    Returns:
        Any: Result of the function execution.
    """
    # Initialize run function for multiprocessing
    return solve_nop(*args, **kwargs)


def main(opts) -> None:
    """
    Main function for running Navigation Orienteering Problem (NOP) benchmarks.

    Args:
        opts (argparse.Namespace): Parsed command-line arguments.
    """

    # For each dataset
    for dataset_path in opts.datasets:
        absolute_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(
            Path(__file__).parent.parent.parent, dataset_path
        )

        # Algorithm
        model_name = '-'.join([opts.route_planner, opts.path_planner])

        # Output filename to save results
        out_file, _ = get_results_file(opts, dataset_path, opts.problem, model_name)
        out_file = os.path.join(Path(__file__).parent.parent.parent, out_file)

        # Target directory to save executions
        target_dir = os.path.join(out_file, 'samples')
        assert opts.f or not os.path.isdir(target_dir), "Target dir already exists! Try with -f option to overwrite"
        os.makedirs(target_dir, exist_ok=True)

        # Load dataset
        dataset = load_dataset(absolute_path)

        # Apply algorithm and get results
        results, parallelism = multiprocessing(
            opts=opts,
            func=run_func,
            directory=target_dir,
            dataset=dataset,
            route_planner=opts.route_planner,
            path_planner=opts.path_planner,
            disable_cache=opts.disable_cache,
            sec_local_search=opts.runs,
            use_multiprocessing=opts.multiprocessing,
        )

        # Print results
        print_results(opts.problem)(results)


if __name__ == '__main__':
    main(get_options())
