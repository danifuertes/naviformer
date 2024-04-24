import os
import argparse
from typing import Tuple

from .modules.net_utils import *
from .naviformer import NaviFormer
from .naviformer2step import NaviFormer2Step
from .pn import PN
from .gpn import GPN
from utils import load_cpu, load_args, load_file, load_file_or_dir, get_inner_model


def load_model_train(opts: argparse.Namespace, ensure_instance: str = '', two_step: str = '') -> \
        Tuple[torch.nn.Module, dict, int]:
    """
    Loads a model for training based on specified options.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.
        ensure_instance (str, optional): The expected class of the loaded model. Defaults to ''.
        two_step (str, optional): Path to a pretrained 2-step route planner. Defaults to ''.

    Returns:
        tuple: A tuple containing the loaded model, loaded data, and the first epoch.
    """

    # Choose model
    model_class = {
        'naviformer': NaviFormer,
        'naviformer_2step': NaviFormer2Step,
        'pn': PN,
        'gpn': GPN,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    # Load model
    model = model_class(
        embed_dim=opts.embed_dim,
        num_blocks=opts.num_blocks,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        max_obs=opts.max_obs,
        combined_mha=opts.combined_mha,
        two_step=opts.two_step,
        num_dirs=opts.num_dirs,
    ).to(opts.device)

    # Ensure model is an instance of the correct class
    if ensure_instance != '':
        assert isinstance(model, model_class[ensure_instance]), \
            f"Loaded model should be instance of {ensure_instance} ({model_class['ensure_instance']}), got: {model})"

    # Pre-trained (2-step) Transformer route planner
    if os.path.isdir(two_step) or os.path.isfile(two_step):
        model.base_route_model, _ = load_model_eval(two_step, kwargs={
            'num_depots': opts.num_depots,
            'num_agents': opts.num_agents,
            'info_th': opts.info_th,
            'max_obs': opts.max_obs
        }, ensure_instance='naviformer_2step')

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
                    ensure_instance: str = '',
                    decode: str = 'greedy',
                    temp: float | None = None,
                    kwargs=None) -> Tuple[torch.nn.Module, dict]:
    """
    Loads a model for evaluation based on specified options.

    Args:
        path (str): Path to the model file or directory.
        epoch (int, optional): The epoch to load if `path` is a directory. Defaults to None.
        ensure_instance (str, optional): The expected class of the loaded model. Defaults to ''.
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
    model_class = {
        'naviformer': NaviFormer,
        'naviformer_2step': NaviFormer2Step,
        'pointer': PointerNetwork,
        'gpointer': GraphPointerNetwork,
    }.get(args.get('model', ''), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    kwargs = {} if kwargs is None else kwargs
    model = model_class(
        embed_dim=args.get('embed_dim', 128),
        num_heads=args.get('num_heads', 8),
        num_blocks=args.get('num_blocks', 2),
        normalization=args.get('normalization', 'batch'),
        tanh_clipping=args.get('tanh_clipping', 10.),
        combined_mha=args.get('combined_mha', False),
        two_step=args.get('two_step', ''),
        max_obs=args.get('max_obs', 5),
        num_dirs=args.get('num_dirs', 4),
        **kwargs
    )

    # Get fancy name
    args['fancy_name'] = {
        'naviformer': 'NaviFormer',
        'naviformer_2step': 'NaviFormer2Step'
    }.get(args.get('model', ''), 'NoName')

    # Ensure model is an instance of the correct class
    if ensure_instance != '':
        assert isinstance(model, model_class[ensure_instance]), \
            f"Loaded model should be instance of {ensure_instance} ({model_class['ensure_instance']}), got: {model})"

    # Overwrite model parameters by parameters to load
    load_data = load_cpu(model_filename)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, *_ = load_file(model_filename, model)

    # Put in eval mode
    model.eval()
    model.set_decode_type(decode, temp=temp)
    return model, args
