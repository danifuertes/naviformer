import gym
import torch


class Regions:

    def __init__(self, regions, prizes=None, depot_ini=None, depot_end=None, min_value=0, max_value=1):

        # Coordinates
        self.regions = regions

        # Dimensions
        assert len(regions.shape) == 3, \
            "Regions' coordinates should have 3 dimensions: batch_size, num_regions, num_dims"
        self.batch_size, self.num_regions, self.num_dims = regions.shape

        # Prizes
        self.prizes = prizes if prizes is not None else torch.zeros_like(regions[..., 0])

        # Initial depot
        self.depot_ini = depot_ini
        assert len(depot_ini.shape) == 3, \
            "Depot's coordinates should have 3 dimensions: batch_size, num_regions, num_dims"

        # End depot
        self.depot_end = depot_end
        assert len(depot_end.shape) == 3, \
            "Depot's coordinates should have 3 dimensions: batch_size, num_regions, num_dims"

        # Normalization
        self.min_value = min_value
        self.max_value = max_value

        # Device
        self.device = regions.device

    def __len__(self):
        return self.num_regions

    def normalize(self):
        self.regions = (self.regions - self.min_value) / (self.max_value - self.min_value)

    def denormalize(self, min_value=0, max_value=1):
        self.regions = self.regions * (max_value - min_value) + min_value

    def is_depot_ini(self):
        return self.depot_ini is None

    def is_depot_end(self):
        return self.depot_end is None

    def get_regions(self):
        return self.regions

    def get_depot_ini(self):
        return self.depot_ini

    def get_depot_end(self):
        return self.depot_end

    def get_regions_depots(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions
        elif self.is_depot_ini() and not self.is_depot_end():
            return torch.cat((self.depot_ini, self.regions), axis=-2)
        elif not self.is_depot_ini() and self.is_depot_end():
            return torch.cat((self.regions, self.depot_end), axis=-2)
        else:
            return torch.cat((self.depot_ini, self.regions, self.depot_end), axis=-2)

    def get_num_dims(self):
        return self.regions.shape[2]

    def get_batch_size(self):
        return self.regions.shape[0]

    def get_num_regions(self):
        return self.regions.shape[1]

    def get_num_regions_depots(self):
        if not self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1]
        elif self.is_depot_ini() and not self.is_depot_end():
            return self.regions.shape[1] + 1
        elif not self.is_depot_ini() and self.is_depot_end():
            return self.regions.shape[1] + 1
        else:
            return self.regions.shape[1] + 2


class NopEnv(gym.Env):

    def __init__(self, inputs):

        # Regions
        self.regions = Regions(
            regions=inputs['loc'],
            prizes=inputs['prize'],
            depot_ini=inputs['depot'],
            depot_end=inputs['depot2'] if 'depot2' in inputs else inputs['depot']
        )

        # Maximum allowed length
        self.max_length = inputs['max_length'][:, None]

        # Obstacles
        self.obs = inputs['obs'] if 'obs' in inputs else None

        # Mask of visited regions
        self.visited = torch.zeros(
            size=(self.regions.get_batch_size(), 1, self.regions.get_num_regions_depots()),
            dtype=torch.uint8,
            device=self.regions.device
        )
        self.visited[..., 0] = 1
        self.visited[self.regions.get_regions_depots()[:, None, :, 0] < 0] = 1  # Block dummy (negative) regions if any

    def step(self, action):
        return

    def reset(self):
        return

    def render(self):
        return
