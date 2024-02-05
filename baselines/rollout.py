import copy
import torch
from scipy.stats import ttest_rel

from .basic import BasicBaseline
from utils import rollout, get_inner_model


class RolloutBaseline(BasicBaseline):

    def __init__(self, model, problem, opts, epoch=0):
        super(BasicBaseline, self).__init__()
        self.opts = opts
        self.problem = problem
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, env=None):

        # Get current epoch
        self.epoch = epoch

        # Copy model for baseline
        self.model = copy.deepcopy(model)

        # Generate new baseline data when updating model to prevent overfitting to the baseline dataset
        self.env = self.problem(
            batch_size=self.opts.eval_batch_size,
            num_workers=self.opts.num_workers,
            device=self.opts.device,
            num_nodes=self.opts.num_nodes,
            num_samples=self.opts.val_size,
            distribution=self.opts.data_dist,
            num_depots=self.opts.num_depots,
            max_length=self.opts.max_length,
            max_nodes=self.opts.max_nodes,
            max_obs=self.opts.max_obs,
            desc='Baseline data'
        ) if env is None else env

        # Rollout and get reward
        self.reward_bl = rollout(self.model, self.env, desc='Baseline').cpu().numpy()
        self.mean = self.reward_bl.mean()

    def eval(self, x, c, s):

        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        with torch.no_grad():
            v, _ = self.model(x, s)

        # There is no loss
        return v, 0

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """

        # Evaluate candidate model on evaluation dataset
        candidate_vals = rollout(model, self.env, self.opts, desc='Baseline').cpu().numpy()
        candidate_mean = candidate_vals.mean()
        print(f"Epoch {epoch} candidate mean {candidate_mean}, baseline epoch {self.epoch}, "
              f"mean {self.mean}, difference {candidate_mean - self.mean}")

        # Calc p-value
        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.reward_bl)
            assert t < 0, "T-statistic should be negative"
            p_val = p / 2  # one-sided
            print("p-value: {}".format(p_val))

            # Update baseline
            if p_val < self.opts.bl_alpha:
                print("Update baseline")
                self._update_model(model, epoch)

    def state_dict(self):
        return {'model': self.model, 'env': self.env, 'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['env'])
