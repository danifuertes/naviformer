import os
import json
import torch
from nets import *


def get_inner_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def load_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""
    print('  [*] Loading model from {}'.format(load_path))

    # Load the model parameters from a saved state
    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    # Get state_dict
    load_optimizer_state_dict = None
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    # Update model state_dict
    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def save_model(model, optimizer, baseline, save_dir, epoch):
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict()
        },
        os.path.join(save_dir, 'epoch-{}.pt'.format(epoch))
    )


def load_model_train(opts, problem, ensure_instance='', two_step=''):

    # Choose model
    model_class = {
        'naviformer': NaviFormer,
        'naviformer_2step': NaviFormer2Step,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)

    # Load model
    model = model_class(
        embed_dim=opts.embed_dim,
        num_blocks=opts.num_blocks,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_enc,
        max_obs=opts.max_obs,
        combined_mha=opts.combined_mha,
        two_step=opts.two_step
    ).to(opts.device)

    # Ensure model is an instance of the correct class
    if ensure_instance != '':
        assert isinstance(model, model_class[ensure_instance]), \
            f"Loaded model should be instance of {ensure_instance} ({model_class['ensure_instance']}), got: {model})"

    # Pre-trained (2-step) Transformer route planner
    if os.path.isdir(two_step) or os.path.isfile(two_step):
        model.base_route_model, _ = load_model_eval(two_step, problem, kwargs={
            'num_depots': opts.num_depots,
            'num_agents': opts.num_agents,
            'info_th': opts.info_th,
            'max_obs': opts.max_obs
        }, ensure_instance='naviformer_2step')

    # Multi-GPU
    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load data
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = load_cpu(load_path)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    return model, load_data


def load_model_eval(path, problem, epoch=None, kwargs=None, ensure_instance=''):

    # Path indicates the saved epoch
    if os.path.isfile(path):
        model_filename = path
        path = os.path.dirname(model_filename)

    # Path indicates the directory where epochs are saved
    elif os.path.isdir(path):
        if epoch is None:
            epoch = max(
                int(os.path.splitext(filename)[0].split("-")[1])
                for filename in os.listdir(path)
                if os.path.splitext(filename)[1] == '.pt'
            )
        model_filename = os.path.join(path, 'epoch-{}.pt'.format(epoch))
    else:
        assert False, f"{path} is not a valid directory or file"

    # Load arguments
    args = load_args(os.path.join(path, 'args.json'))

    # Load problem
    # problem = load_problem(args['problem'])

    # Load model
    model_class = {
        'naviformer': NaviFormer,
        'naviformer_2step': NaviFormer2Step,
    }.get(args.get('model', 'naviformer'), None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    kwargs = {} if kwargs is None else kwargs
    model = model_class(
        embed_dim=args['embed_dim'],
        num_heads=args.get('num_heads', 8),
        num_blocks=args['num_blocks'],
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_enc=args.get('checkpoint_enc', False),
        combined_mha=args.get('combined_mha'),
        two_step=args.get('two_step', ''),
        **kwargs
    )

    # Ensure model is an instance of the correct class
    if ensure_instance != '':
        assert isinstance(model, model_class[ensure_instance]), \
            f"Loaded model should be instance of {ensure_instance} ({model_class['ensure_instance']}), got: {model})"

    # Overwrite model parameters by parameters to load
    load_data = load_cpu(model_filename)
    # for i in range(len(model.state_dict().keys())):
    #     print(f"{list(model.state_dict().keys())[i]} | {list(load_data.get('model', {}).keys())[i]}")
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, *_ = load_file(model_filename, model)

    # Put in eval mode
    model.eval()
    return model, args


def load_args(filename):
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_dist' not in args:
        args['data_dist'] = None
        probl, *dist = args['problem'].split("_")
        if probl == "op":
            args['problem'] = probl
            args['data_dist'] = dist[0]
    return args
