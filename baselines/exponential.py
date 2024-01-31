from .basic import BasicBaseline


class ExponentialBaseline(BasicBaseline):

    def __init__(self, beta):
        super(BasicBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x, c, e):

        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self):
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']
