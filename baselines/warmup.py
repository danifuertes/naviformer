import torch
from typing import Any, Tuple

from .basic import BasicBaseline
from .exponential import ExponentialBaseline


class WarmupBaseline(BasicBaseline):

    def __init__(self, baseline: Any, num_epochs: int = 1, warmup_exp_beta: float = 0.8) -> None:
        """
        Initialize the RolloutBaseline.

        Args:
            baseline (Any): Model.
            num_epochs (int): Number of epochs.
            warmup_exp_beta (float): Exponential parameter during warmup.
        """
        super(BasicBaseline, self).__init__()
        assert num_epochs > 0, "n_epochs to warmup must be positive"
        self.alpha = 0
        self.baseline = baseline
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.num_epochs = num_epochs

    def eval(self, x: dict | torch.Tensor, c: torch.Tensor, e: Any) -> \
            Tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Evaluate the baseline.

        Args:
            x (dict or torch.Tensor): Input batch.
            c (torch.Tensor): Cost (negative reward) found by the model.
            e (Any): Environment.

        Returns:
            tuple: Tuple containing the baseline value and the loss.
        """
        if self.alpha == 1:
            return self.baseline.eval(x, c, e)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c, e)
        v, l = self.baseline.eval(x, c, e)
        vw, lw = self.warmup_baseline.eval(x, c, e)

        # Return convex combination of baseline and of loss
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha * lw)

    def epoch_callback(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Epoch callback.

        Args:
            model (torch.Tensor): Model.
            epoch (int): Epoch number.
        """
        # Need to call epoch callback of inner model (also after first epoch if we have not used it)
        self.baseline.epoch_callback(model, epoch)
        self.alpha = (epoch + 1) / float(self.num_epochs)
        if epoch < self.num_epochs:
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self) -> dict:
        """
        Get the state dictionary.

        Returns:
            dict: State dictionary.
        """
        # Checkpointing within warmup stage makes no sense, only save inner baseline
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary of the baseline.

        Args:
            state_dict: State dictionary to load.
        """
        # Checkpointing within warmup stage makes no sense, only load inner baseline
        self.baseline.load_state_dict(state_dict)
