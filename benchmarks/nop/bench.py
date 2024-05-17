from utils import *
from envs import print_results
from benchmarks import parse_runs, multiprocessing, solve_nop

from neural_astar.planner import NeuralAstar as NAStar
from neural_astar.planner import VanillaAstar as VAStar
from neural_astar.utils.training import load_from_ptl_checkpoint


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
    parser.add_argument("--path_planner", help="Name of the path planner to evaluate: [a_star, d_star, na_star]")

    # Misc
    parser.add_argument('--use_cuda', type=str2bool, default=True, help="True to use CUDA (only for na_star)")
    parser.add_argument("--cpus", type=int, help="Number of CPUs to use, defaults to all cores")
    parser.add_argument('--multiprocessing', type=str2bool, default=False, help='Use multiprocessing')
    parser.add_argument('--disable_cache', action='store_true', help='Disable caching')
    parser.add_argument('--progress_bar_mininterval', type=float, default=0.1, help='Minimum interval')
    opts = parser.parse_args()

    # Use CUDA or CPU (only for Neural A*)
    opts.use_cuda = torch.cuda.is_available() and opts.use_cuda

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

    # Device (only for Neural A*)
    device = torch.device("cuda" if opts.use_cuda else "cpu")

    # For each dataset
    for dataset_path in opts.datasets:
        absolute_path = dataset_path if os.path.isabs(dataset_path) else os.path.join(
            Path(__file__).parent.parent.parent, dataset_path
        )

        # Algorithm
        model_name = '-'.join([opts.route_planner, opts.path_planner])
        if opts.path_planner == 'na_star':
            model = NAStar(encoder_arch='CNN').to(device)
            model.load_state_dict(load_from_ptl_checkpoint(
                "./benchmarks/nop/methods/neural-astar/model/mazes_032_moore_c8/lightning_logs/"
            ))
        elif opts.path_planner == 'a_star':
            model = VAStar().to(device)
        else:
            model = None

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
            model=model,
            use_multiprocessing=opts.multiprocessing,
        )

        # Collect results
        new_results = [[] for _ in range(5)]
        for result in results:
            new_results[0] = [*new_results[0], result[0]]
            new_results[1] = [*new_results[1], result[1]]
            new_results[2] = [*new_results[2], result[2]]
            new_results[3] = [*new_results[3], result[3]]
            new_results[4] = [*new_results[4], result[4]]
        results = new_results

        # Add parallelism info to results
        results.append(parallelism)

        # Print results
        print_results(opts.problem)(results)


if __name__ == '__main__':
    main(get_options())
