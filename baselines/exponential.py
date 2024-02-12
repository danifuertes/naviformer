import torch
from typing import Any, Tuple
from .basic import BasicBaseline


class ExponentialBaseline(BasicBaseline):

    def __init__(self, beta):
        """
        Initialize the ExponentialBaseline.

        Args:
            beta (float): Exponential parameter.
        """
        super(BasicBaseline, self).__init__()

        self.beta = beta
        self.v = None

    def eval(self, x: dict, c: torch.Tensor, e: Any) -> Tuple[torch.Tensor | float, torch.Tensor | float]:
        """
        Evaluate the baseline.

        Args:
            x (dict): Input batch.
            c (torch.Tensor): Cost (negative reward) found by the model.
            e (Any): Environment.

        Returns:
            tuple: Tuple containing the baseline value and the loss.
        """
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()

        self.v = v.detach()  # Detach since we never want to backprop
        return self.v, 0  # No loss

    def state_dict(self) -> dict:
        """
        Get the state dictionary.

        Returns:
            dict: State dictionary.
        """
        return {
            'v': self.v
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary of the baseline.

        Args:
            state_dict: State dictionary to load.
        """
        self.v = state_dict['v']
