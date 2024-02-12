import torch
from typing import Any, Tuple


class BasicBaseline(object):

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
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self) -> list:
        """
        Get learnable parameters of the baseline.

        Returns:
            list: List of learnable parameters.
        """
        return []

    def epoch_callback(self, model: torch.nn.Module, epoch: int) -> None:
        """
        Epoch callback.

        Args:
            model (torch.Tensor): Model.
            epoch (int): Epoch number.
        """
        pass

    def state_dict(self) -> dict:
        """
        Get the state dictionary.

        Returns:
            dict: State dictionary.
        """
        return {}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary of the baseline.

        Args:
            state_dict: State dictionary to load.
        """
        pass


class NoBaseline(BasicBaseline):

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
        return 0, 0  # No baseline, no loss
