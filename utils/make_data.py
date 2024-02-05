import os
import argparse
import numpy as np
from pathlib import Path

from envs import load_dataset
from utils import save_dataset, set_seed, global_vars


PROBLEMS = global_vars()['PROBLEMS']
DATA_DIST = global_vars()['DATA_DIST']


def data2list(data):
    datalist = []
    for d in data:
        datalist.append([])
        for v in d.values():
            datalist[-1].append(v.tolist())
    return datalist


def get_options():
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
    parser.add_argument('--max_obs', type=int, default=0, help='Maximum number of obstacles')

    # Misc
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset (test, validation...)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    opts = parser.parse_args()

    # Check options are correct
    assert opts.seed >= 0,                          f"seed must be non-negative, found {opts.seed}"
    assert opts.problem in PROBLEMS,                f"'{opts.problem}' not in problem list: {PROBLEMS}"
    assert np.all(np.array(opts.num_nodes) > 0),    f"num_nodes must be positive, found: {opts.num_nodes}"
    assert opts.num_depots in [1, 2],               f"num_depots must be 1 or 2, found: {opts.num_depots}"
    assert np.all(np.array(opts.max_length) > 0),   f"max_length must be positive, found: {opts.max_length}"
    assert opts.max_obs >= 0,                       f"max_obs must be non-negative, found: {opts.max_obs}"
    assert opts.max_nodes == 0 or opts.max_nodes >= 10, \
        f"max_nodes must be non-negative and considerably large (>= 10), found: {opts.max_nodes}"
    assert len(opts.num_nodes) == len(opts.max_length), \
        f"num_nodes and max_length must have same length, found {opts.num_nodes} and {opts.max_length}"
    for dist in opts.data_dist:
        assert dist in DATA_DIST,                   f"'{dist}' not in data_dist list: {DATA_DIST}"

    # Data dir should be relative to project working dir, not to current dir (utils)
    opts.data_dir = os.path.join(Path(__file__).parent.parent, opts.data_dir)

    # Set seed
    set_seed(opts.seed)
    return opts


def main(opts):

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
            length_str = '' if opts.max_nodes else '_L{}'.format(
                int(opts.max_length[i]) if opts.max_length[i].is_integer() else opts.max_length[i]
            )
            obs_str = '_{}obs'.format(opts.max_obs) if opts.max_obs else ''
            # obs_str = '_j{}obs'.format(OBS) if opts.max_obs else ''
            filename = os.path.join(
                data_dir,
                "{}_seed{}{}{}.pkl".format(opts.name, opts.seed, length_str, obs_str)
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
                    max_obs=opts.max_obs,
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
