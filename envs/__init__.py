from .nop.nop import NopEnv


def load_problem(name):
    problem = {
        'nop': NopEnv,
    }.get(name, None)
    assert problem is not None, "Currently unsupported problem: {}!".format(name)
    return problem
