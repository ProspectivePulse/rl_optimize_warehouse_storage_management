# src/warehouse/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class WarehouseEnv(gym.Env):
    """
    A warehouse environment conforming to the modern Gymnasium API.
    An agent must physically move to pick up and drop items to organize them.
    """
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 4}

    def __init__(self, grid_size=5, num_items=5, max_steps=100, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.num_items = num_items
        self.max_steps = max_steps
        self.agent_pos = [0, 0]
        self.agent_carries_item = 0
        self.steps_taken = 0
        self.warehouse = np.zeros((grid_size, grid_size), dtype=int)
        self.item_locations = {}
        self.item_demand = np.random.randint(1, 10, size=num_items + 1)
        self.item_demand[0] = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5) # 0:Up, 1:Down, 2:Left, 3:Right, 4:Interact
        self.observation_space = spaces.Dict({
            "agent_pos": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=int),
            "agent_carries": spaces.Discrete(num_items + 1),
            "warehouse_layout": spaces.Box(low=0, high=num_items, shape=(grid_size, grid_size), dtype=int)
        })

    def _get_obs(self):
        return {
            "agent_pos": np.array(self.agent_pos, dtype=int),
            "agent_carries": self.agent_carries_item,
            "warehouse_layout": self.warehouse
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [0, 0]
        self.agent_carries_item = 0
        self.steps_taken = 0
        self.warehouse.fill(0)
        self.item_locations.clear()

        for item in range(1, self.num_items + 1):
            while True:
                row, col = self.np_random.integers(0, self.grid_size, size=2)
                if self.warehouse[row, col] == 0:
                    self.warehouse[row, col] = item
                    self.item_locations[item] = (row, col)
                    break
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps_taken += 1
        reward = -0.01

        if action < 4:
            new_pos = self.agent_pos.copy()
            if action == 0: new_pos[0] -= 1
            elif action == 1: new_pos[0] += 1
            elif action == 2: new_pos[1] -= 1
            elif action == 3: new_pos[1] += 1
            if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                self.agent_pos = new_pos
            else:
                reward -= 0.1
        elif action == 4:
            agent_r, agent_c = self.agent_pos
            if self.agent_carries_item > 0:
                if self.warehouse[agent_r, agent_c] == 0:
                    item_id = self.agent_carries_item
                    self.warehouse[agent_r, agent_c] = item_id
                    self.item_locations[item_id] = (agent_r, agent_c)
                    self.agent_carries_item = 0
                    distance = agent_r + agent_c
                    reward += self.item_demand[item_id] / (distance + 1) * 5
                else:
                    reward -= 0.2
            else:
                item_on_floor = self.warehouse[agent_r, agent_c]
                if item_on_floor > 0:
                    self.agent_carries_item = item_on_floor
                    self.warehouse[agent_r, agent_c] = 0
                    del self.item_locations[item_on_floor]
                    reward += 0.1
                else:
                    reward -= 0.2

        terminated = self.steps_taken >= self.max_steps
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(7, 7))
            render_grid = np.copy(self.warehouse)
            agent_r, agent_c = self.agent_pos
            
            agent_render_value = self.num_items + 2
            if self.agent_carries_item > 0:
                agent_render_value += 1
            render_grid[agent_r, agent_c] = agent_render_value

            cmap_colors = [(1, 1, 1, 1)] # 0 = white
            viridis = plt.colormaps.get('viridis')
            item_colors = viridis(np.linspace(0, 1, self.num_items))
            cmap_colors.extend(item_colors)
            cmap_colors.append((1, 0, 0, 1)) # agent = red
            cmap_colors.append((1, 1, 0, 1)) # agent carrying = yellow
            
            custom_cmap = colors.ListedColormap(cmap_colors)
            bounds = np.arange(self.num_items + 4) - 0.5
            norm = colors.BoundaryNorm(bounds, custom_cmap.N)

            ax.imshow(render_grid, cmap=custom_cmap, norm=norm)

            ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
            ax.grid(which="minor", color="black", linewidth=0.5)
            ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    item_id = self.warehouse[r,c]
                    if item_id > 0:
                        ax.text(c, r, str(item_id), ha='center', va='center', color='white', weight='bold')

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', edgecolor='black', label='Agent (Empty)'),
                Patch(facecolor='yellow', edgecolor='black', label='Agent (Carrying)'),
                Patch(facecolor=viridis(0.5), edgecolor='black', label='Item'),
                Patch(facecolor='white', edgecolor='black', label='Empty Space')
            ]
            ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=2)
            plt.tight_layout(rect=[0, 0.05, 1, 1])

            canvas = FigureCanvas(fig)
            canvas.draw()
            width, height = fig.canvas.get_width_height()
            image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(height, width, 4)[:,:,:3]
            plt.close(fig)
            return image