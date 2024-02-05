from .nop.nop import NopEnv, NopDataset
from .nop.nop_utils import print_nop_results


def load_problem(name):
    problem = {
        'nop': NopEnv,
    }.get(name, None)
    assert problem is not None, f"Currently unsupported problem: {name}!"
    return problem


def load_dataset(name):
    dataset = {
        'nop': NopDataset,
    }.get(name, None)
    assert dataset is not None, f"Currently unsupported dataset: {name}!"
    return dataset


def print_results(name):
    func = {
        'nop': print_nop_results,
    }.get(name, None)
    assert func is not None, f"Currently unsupported function: {name}!"
    return func
