import torch
from typing import Any, Tuple
from .basic import BasicBaseline


class CriticBaseline(BasicBaseline):

    def __init__(self, critic):
        super(BasicBaseline, self).__init__()
        self.critic = critic

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

        # Predict critic value
        v = self.critic(x, e)

        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), torch.nn.functional.mse_loss(v, c.detach())

    def get_learnable_parameters(self) -> list:
        """
        Get learnable parameters of the baseline.

        Returns:
            list: List of learnable parameters.
        """
        return list(self.critic.parameters())

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
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Load the state dictionary of the baseline.

        Args:
            state_dict: State dictionary to load.
        """
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # Backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})
