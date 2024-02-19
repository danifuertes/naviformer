import os
import json
import torch
from typing import Any, Tuple


def get_inner_model(model: torch.nn.Module | torch.nn.DataParallel) -> torch.nn.Module:
    """
    Get the inner model from a DataParallel wrapper.

    Args:
        model (torch.nn.Module or torch.nn.DataParallel): The model to extract the inner model from.

    Returns:
        torch.nn.Module: The inner model.
    """
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_cpu(load_path: str) -> Any:
    """
    Loads data from a file while ensuring it is loaded onto the CPU.

    Args:
        load_path (str): The path to the file containing the data.

    Returns:
        Any: The loaded data.
    """
    return torch.load(load_path, map_location=lambda storage, loc: storage)


def load_file(load_path: str, model: torch.nn.Module) -> Tuple[torch.nn.Module, dict]:
    """
    Loads the model parameters from a file and returns the model and the optimizer state dict (if it is in the file).

    Args:
        load_path (str): The path to the file containing the saved model.
        model (torch.nn.Module): The model to load parameters into.

    Returns:
        tuple: A tuple containing the loaded model and the optimizer state dict (or None if not found).
    """
    print('  [*] Loading model from {}'.format(load_path))

    # Load the model parameters from a saved state
    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    # Get state_dict
    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_optimizer_state_dict = None
        load_model_state_dict = load_data.state_dict()

    # Update model state_dict
    state_dict = model.state_dict()
    state_dict.update(load_model_state_dict)
    model.load_state_dict(state_dict)
    return model, load_optimizer_state_dict


def load_file_or_dir(path: str, epoch: int | None = None) -> Tuple[str, str, int]:
    """
    Determines whether the provided path is a file or directory and returns the appropriate model filename. If it is a
    directory, it returns the latest saved model filename when there are more than one in the directory.

    Args:
        path (str): The path to the file or directory containing the saved model(s).
        epoch (int, optional): The epoch to load from if `path` is a directory. Defaults to None.

    Returns:
        tuple: A tuple containing the model filename, its directory, and the next epoch.
    """

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
        model_filename = os.path.join(path, f"epoch-{str(epoch).zfill(3)}.pt")
    else:
        assert False, f"{path} is not a valid directory or file"
    return model_filename, path, epoch + 1


def save_model(
        model: torch.nn.Module | torch.nn.DataParallel,
        optimizer: torch.optim.Optimizer,
        baseline: Any,
        save_dir: str,
        epoch: int) -> None:
    """
    Saves the model, optimizer state, and other information to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        baseline: The baseline.
        save_dir (str): The directory to save the model to.
        epoch (int): The current epoch.

    Returns:
        None
    """
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            'baseline': baseline.state_dict()
        },
        os.path.join(save_dir, f"epoch-{str(epoch).zfill(3)}.pt")
    )


def load_args(filename: str) -> dict:
    """
    Loads arguments from a JSON file.

    Args:
        filename (str): The path to the JSON file containing arguments.

    Returns:
        dict: The loaded arguments.
    """
    with open(filename, 'r') as f:
        args = json.load(f)

    # Backwards compatibility
    if 'data_dist' not in args:
        args['data_dist'] = None
        problem, *dist = args['problem'].split("_")
        if problem == "op":
            args['problem'] = problem
            args['data_dist'] = dist[0]
    return args
