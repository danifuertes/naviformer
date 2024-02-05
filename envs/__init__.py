from .nop.nop import NopEnv, NopDataset


def load_problem(name):
    problem = {
        'nop': NopEnv,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem


def load_dataset(name):
    dataset = {
        'nop': NopDataset,
    }.get(name, None)
    assert dataset is not None, "Currently unsupported dataset: {}!".format(name)
    return dataset
