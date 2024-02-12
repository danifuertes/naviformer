from typing import Any

from .nop.nop import NopEnv, NopDataset
from .nop.nop_utils import print_nop_results


def load_problem(name: str) -> Any:
    """
    Load the problem based on its name.

    Args:
        name (str): Name of the problem.

    Returns:
        class: Problem class.
    """
    problem = {
        'nop': NopEnv,
    }.get(name, None)
    assert problem is not None, f"Currently unsupported problem: {name}!"
    return problem


def load_dataset(name: str) -> Any:
    """
    Load the dataset based on its name.

    Args:
        name (str): Name of the dataset.

    Returns:
        class: Dataset class.
    """
    dataset = {
        'nop': NopDataset,
    }.get(name, None)
    assert dataset is not None, f"Currently unsupported dataset: {name}!"
    return dataset


def print_results(name: str) -> Any:
    """
    Print results based on their name.

    Args:
        name (str): Name of the results.

    Returns:
        function: Function to print results.
    """
    func = {
        'nop': print_nop_results,
    }.get(name, None)
    assert func is not None, f"Currently unsupported function: {name}!"
    return func
