import os
import argparse
from typing import Tuple

from .op import *
from .nop import *
from .modules.net_utils import *
from utils import load_cpu, load_args, load_file, load_file_or_dir, get_inner_model


MODELS = {
    'nop': {
        'naviformer': NaviFormer,
        'pn': PN,
        'gpn': GPN,
    },
    'op': {
        'naviformer2step': NaviFormer2Step,
        'pn2step': PN2Step,
        'gpn2step': GPN2Step,
        'naviformer_na_star': NaviFormerNAStar,
    }
}
FANCY_NAME = {
    'naviformer': 'NaviFormer',
    'naviformer2step': 'NaviFormer2Step',
    'pn2step': 'PN2Step',
    'gpn2step': 'GPN2Step',
    'naviformer_na_star': 'NaviFormerNAStar',
    'pn': 'PN',
    'gpn': 'GPN',
}


def load_model_train(opts: argparse.Namespace) -> Tuple[torch.nn.Module, dict, int]:
    """
    Loads a model for training based on specified options.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.

    Returns:
        tuple: A tuple containing the loaded model, loaded data, and the first epoch.
    """

    # Choose model
    model_class = MODELS.get(opts.problem, '').get(opts.model, None)
    assert model_class is not None, f"Unknown model '{opts.model}' for given problem '{opts.problem}'"

    # Pre-trained (2-step) Transformer route planner
    if os.path.isdir(opts.two_step) or os.path.isfile(opts.two_step):
        two_step, _ = load_model_eval(opts.two_step)
    else:
        two_step = None

    # Load model
    model = model_class(
        embed_dim=opts.embed_dim,
        num_blocks=opts.num_blocks,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        num_obs=opts.num_obs,
        combined_mha=opts.combined_mha,
        two_step=two_step,
        num_dirs=opts.num_dirs,
    ).to(opts.device)

    # Multi-GPU
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    load_data, first_epoch = {}, 0
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        model_filename, _, first_epoch = load_file_or_dir(path=load_path)
        print(f"  [*] Loading data from {model_filename}")
        load_data = load_cpu(model_filename)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    return model, load_data, first_epoch


def load_model_eval(path: str,
                    epoch: int = None,
                    decode: str = 'greedy',
                    temp: float | None = None,
                    kwargs=None) -> Tuple[torch.nn.Module, dict]:
    """
    Loads a model for evaluation based on specified options.

    Args:
        path (str): Path to the model file or directory.
        epoch (int, optional): The epoch to load if `path` is a directory. Defaults to None.
        decode (str, optional): The decoding strategy. Defaults to 'greedy'.
        temp (float, optional): Softmax temperature for sampling. Defaults to None.
        kwargs (dict, optional): Additional keyword arguments for model initialization. Defaults to None.

    Returns:
        tuple: A tuple containing the loaded model and its arguments.
    """

    # Get model filename
    model_filename, path, _ = load_file_or_dir(path=path, epoch=epoch)

    # Load arguments
    args = load_args(os.path.join(path, 'args.json'))

    # Load model
    args_model, args_problem = args.get('model', ''), args.get('problem', '')
    model_class = MODELS.get(args_problem, {}).get(args_model, None)
    assert model_class is not None, f"Unknown model '{args_model}' for given problem '{args_problem}'"
    kwargs = {} if kwargs is None else kwargs
    model = model_class(
        embed_dim=args.get('embed_dim', 128),
        num_heads=args.get('num_heads', 8),
        num_blocks=args.get('num_blocks', 2),
        normalization=args.get('normalization', 'batch'),
        tanh_clipping=args.get('tanh_clipping', 10.),
        combined_mha=args.get('combined_mha', False),
        # two_step=args.get('two_step', None),
        num_obs=args.get('num_obs', (0, 0)),
        num_dirs=args.get('num_dirs', 4),
        **kwargs
    )

    # Get fancy name
    args['fancy_name'] = FANCY_NAME.get(args.get('model', ''), 'NoName')

    # Overwrite model parameters by parameters to load
    load_data = load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, *_ = load_file(model_filename, model)

    # Put in eval mode
    model.eval()
    model.set_decode_type(decode, temp=temp)
    return model, args
