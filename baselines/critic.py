import torch
from .basic import BasicBaseline


class CriticBaseline(BasicBaseline):

    def __init__(self, critic):
        super(BasicBaseline, self).__init__()
        self.critic = critic

    def eval(self, x, c, s):

        # Predict critic value
        v = self.critic(x, s)

        # Detach v since actor should not backprop through baseline, only for loss
        return v.detach(), torch.nn.functional.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):  # Backwards compatibility
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})
