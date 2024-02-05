from .basic import BasicBaseline
from .exponential import ExponentialBaseline


class WarmupBaseline(BasicBaseline):

    def __init__(self, baseline, num_epochs=1, warmup_exp_beta=0.8, ):
        super(BasicBaseline, self).__init__()
        assert num_epochs > 0, "n_epochs to warmup must be positive"
        self.alpha = 0
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.num_epochs = num_epochs

    def eval(self, x, c, s):

        if self.alpha == 1:
            return self.baseline.eval(x, c, s)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c, s)
        v, l = self.baseline.eval(x, c, s)
        vw, lw = self.warmup_baseline.eval(x, c, s)

        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model, epoch):
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.num_epochs)
        if epoch < self.num_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)
