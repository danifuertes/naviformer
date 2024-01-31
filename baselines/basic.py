

class BasicBaseline(object):

    def eval(self, x, c, e):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


class NoBaseline(BasicBaseline):

    def eval(self, x, c, e):
        return 0, 0  # No baseline, no loss
