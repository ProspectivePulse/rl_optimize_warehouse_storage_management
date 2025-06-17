import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WarehouseEnv(gym.Env):
    def __init__(self, grid_size=5, num_items=5, max_steps=50):
        super(WarehouseEnv, self).__init__()

        self.grid_size = grid_size
        self.num_items = num_items
        self.max_steps = max_steps
        self.steps_taken = 0

        self.warehouse = np.zeros((grid_size, grid_size), dtype=int)
        self.item_locations = {}
        self.item_demand = np.random.randint(1, 10, size=num_items + 1)

        self.action_space = spaces.Discrete((num_items + 1) * grid_size * grid_size)
        self.observation_space = spaces.Box(
            low=0, high=num_items, shape=(grid_size * grid_size,), dtype=int
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.warehouse.fill(0)
        self.item_locations.clear()
        self.steps_taken = 0

        for item in range(1, self.num_items + 1):
            row, col = np.random.randint(0, self.grid_size, size=2)
            while self.warehouse[row, col] != 0:
                row, col = np.random.randint(0, self.grid_size, size=2)
            self.warehouse[row, col] = item
            self.item_locations[item] = (row, col)

        return self._get_obs(), {}

    def _get_obs(self):
        return self.warehouse.flatten()

    def step(self, action):
        self.steps_taken += 1

        item_id = action // (self.grid_size * self.grid_size)
        target_row = (action // self.grid_size) % self.grid_size
        target_col = action % self.grid_size

        if item_id == 0 or item_id not in self.item_locations:
            return self._get_obs(), -1, False, False, {}

        current_row, current_col = self.item_locations[item_id]

        if self.warehouse[target_row, target_col] == 0:
            self.warehouse[current_row, current_col] = 0
            self.warehouse[target_row, target_col] = item_id
            self.item_locations[item_id] = (target_row, target_col)

        distance = abs(target_row) + abs(target_col)
        reward = self.item_demand[item_id] / (distance + 1)

        done = self.steps_taken >= self.max_steps or self._goal_reached()
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def _goal_reached(self):
        for item_id, (row, col) in self.item_locations.items():
            if self.item_demand[item_id] >= 7 and (row + col) > 2:
                return False
        return True

    def render(self):
        print(self.warehouse)
