import os
import torch
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple


def parse_softmax_temp(raw_temp: float | str) -> float:
    """
    Parse the softmax temperature for action sampling.

    Args:
        raw_temp (str or float): If str, it can be a filename containing temperature values.

    Returns:
        float: The parsed softmax temperature.
    """
    if os.path.isfile(raw_temp):
        return np.loadtxt(raw_temp)[-1, 0]
    return float(raw_temp)


def get_results_file(opts: argparse.Namespace, data_path: str, problem_name: str, model_name: str) -> Tuple[str, str]:
    """
    Create the filename and directory to save the results of the evaluation from the dataset, problem, and model used.

    Args:
        opts (argparse.Namespace): Parsed command line arguments.
        data_path (str): Path to the dataset file.
        problem_name (str): Name of the problem being evaluated.
        model_name (str): Name of the model being evaluated.

    Returns:
        tuple: A tuple containing the filename and directory to save the results.
    """

    # Prepare data dir to save results
    dataset_name, ext = os.path.splitext(data_path.replace('/', '_'))
    if opts.o is None:
        results_dir = os.path.join(opts.results_dir, problem_name, dataset_name)  # TODO: should use absolute path
        results_file = os.path.join(results_dir, model_name)
    else:
        results_file = opts.o
        results_dir = Path(results_file).parent
    assert opts.f or not os.path.isfile(results_file), "File already exists! Try running with -f option to overwrite."
    return results_file, results_dir


def check_extension(filename: str) -> str:
    """
    Check that the extension of a dataset filename corresponds to a pickle file extension.

    Args:
        filename (str): Path to the pickle file containing the dataset.

    Returns:
        str: Path to the pickle file containing the dataset with correct extension.
    """
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def load_dataset(filename: str) -> object:
    """
    Load a dataset from a pickle file.

    Args:
        filename (str): Path to the pickle file containing the dataset.

    Returns:
        object: Loaded dataset object.
    """
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)


def save_dataset(dataset: object, filename: str) -> None:
    """
    Save a dataset to a pickle file.

    Args:
        dataset (object): Dataset object to be saved.
        filename (str): Path to save the pickle file.

    Returns:
        None
    """
    filedir = os.path.split(filename)[0]
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def batch2numpy(batch: torch.Tensor | dict, to_list: bool = False) -> np.ndarray | dict:
    """
    Convert a batch of tensors to numpy arrays.

    Args:
        batch (torch.Tensor or dict): Batch of tensors to convert.
        to_list (bool): Whether to convert to a list of numpy arrays.

    Returns:
        numpy.ndarray or dict: Converted batch of numpy arrays.
    """
    if isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = v.cpu().detach().numpy()
            batch[k] = batch[k].tolist() if to_list else batch[k].squeeze()
    else:
        batch = batch.cpu().detach().numpy()
        batch = batch.tolist() if to_list else batch.squeeze()
    return batch


def actions2numpy(actions: torch.Tensor | list, end_ids: int) -> np.ndarray | list:
    """
    Convert action tensors to numpy arrays.

    Args:
        actions (Tensor or list): Action tensors to convert.
        end_ids (int): Identifier for the end action (end depot).

    Returns:
        numpy.ndarray or list: Converted action numpy arrays.
    """
    if isinstance(actions, list):
        out = []
        for i, action in enumerate(actions):
            out.append(action.cpu().detach().numpy().squeeze(0))
            if out[-1][0] != 0:
                out[-1] = np.concatenate(([0], out[-1]), axis=-1)
            elif out[-1][-1] != end_ids:
                out[-1] = np.concatenate((out[-1], [end_ids]), axis=-1)
    else:
        out = actions.cpu().detach().numpy().squeeze(0)
    return out
