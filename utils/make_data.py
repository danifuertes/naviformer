import os
import argparse
from pathlib import Path

from envs import load_dataset
from utils import save_dataset, set_seed, global_vars, check_make_data_options


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']


def data2list(data):
    datalist = []
    for d in data:
        datalist.append([])
        for k in sorted(d.keys()):
            datalist[-1].append(d[k].tolist())
    return datalist


def get_options() -> argparse.Namespace:
    """
    Parse command line arguments to configure dataset generation options.

    Returns:
        argparse.Namespace: Parsed arguments as a namespace object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    # Data
    parser.add_argument("--num_samples", type=int, default=10000, help="Size of the dataset")

    # Problem
    parser.add_argument('--problem', type=str, default='nop', help=f"Problem to solve: {', '.join(PROBLEMS)}")
    parser.add_argument('--num_nodes', type=int, nargs='+', default=[20, 50, 100],
                        help="Number of visitable nodes (default 20, 50, 100)")
    parser.add_argument('--max_nodes', type=int, default=0, help='Max number of nodes (random num_nodes padded with'
                        'dummys until max_nodes). Max length depends on the random num_nodes. Disable with max_nodes=0')
    parser.add_argument('--num_depots', type=int, default=1, help="Number of depots. Options are 1 or 2. num_depots=1"
                        "means that the start and end depot are the same. num_depots=2 means that they are different")
    parser.add_argument('--max_length', type=float, nargs='+', default=[2, 3, 4],
                        help="Normalized time limit to solve the problem")
    parser.add_argument('--data_dist', type=str, nargs='+', default=['const'],
                        help=f"Data distribution (reward values of regions) of OP. Options: {', '.join(DATA_DIST)}")
    parser.add_argument('--num_obs', type=int, nargs='+', default=(0, 0), help='Tuple of 2 values indicating min and max number of obstacles')
    parser.add_argument('--rad_obs', type=float, nargs='+', default=(.02, .12), help='Tuple of 2 values indicating min and max radious of obstacles')

    # Misc
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset (test, validation...)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    opts = parser.parse_args()

    # Data dir should be relative to project working dir, not to current dir (utils)
    opts.data_dir = os.path.join(Path(__file__).parent.parent, opts.data_dir)

    # Set seed for reproducibility
    set_seed(opts.seed)

    # Check options are correct
    check_make_data_options(opts)
    return opts


def main(opts: argparse.Namespace) -> None:
    """
    Main function to generate and save datasets based on specified options.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        None
    """

    # For each data distribution
    for dist in opts.data_dist:
        print('Distribution = {}'.format(dist))

        # For each number of nodes
        for i, num_node in enumerate(opts.num_nodes):
            print(f"Number of nodes = {opts.max_nodes if opts.max_nodes else num_node}")

            # Directory and filename
            data_dir = os.path.join(
                opts.data_dir,
                opts.problem,
                f"{opts.num_depots}depots",
                dist,
                f'max{opts.max_nodes}' if opts.max_nodes else str(num_node)
            )
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            length_str = '' if opts.max_nodes else '_T{}'.format(
                int(opts.max_length[i]) if opts.max_length[i].is_integer() else opts.max_length[i]
            )
            obs_str = f"_{opts.num_obs[0]}-{opts.num_obs[1]}obs"
            # obs_str = '_j{}obs'.format(OBS) if opts.max_obs else ''
            filename = os.path.join(
                data_dir,
                f"{opts.name}_seed{opts.seed}{length_str}{obs_str}.pkl"
            )

            # Generate dataset
            dataset = data2list(
                load_dataset(opts.problem)(
                    num_samples=opts.num_samples,
                    num_nodes=num_node,
                    data_dist=dist,
                    max_length=opts.max_length[i],
                    num_depots=opts.num_depots,
                    max_nodes=opts.max_nodes,
                    num_obs=opts.num_obs,
                    rad_obs=opts.rad_obs,
                    desc='\tGenerating data').data
            )

            # Save dataset
            if num_node > 100 and opts.dataset_size > 1e6:
                step, count, c = 1, 0, 0
                while count < opts.dataset_size:
                    save_dataset(
                        dataset[int(count):int(min(count + step, len(dataset)))],
                        os.path.join(filename.replace('.pkl', ''), str(c).zfill(9))
                    )
                    count += step
                    c += 1
            else:
                save_dataset(dataset, filename)

            if i == 0 and opts.max_nodes > 0:
                break
    print('Finished')


if __name__ == "__main__":
    main(get_options())
