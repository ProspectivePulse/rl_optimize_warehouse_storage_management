# src/warehouse/utils.py

import os
import yaml
import imageio
import re
from stable_baselines3.common.callbacks import BaseCallback

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class SaveAnimationCallback(BaseCallback):
    """
    Callback for saving a training animation (as frames).
    """
    def __init__(self, save_freq: int, animation_dir: str, verbose: int = 0):
        super(SaveAnimationCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.animation_dir = animation_dir
        os.makedirs(self.animation_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            img = self.training_env.render()
            filename = os.path.join(self.animation_dir, f"frame_{self.num_timesteps:06d}.png")
            imageio.imwrite(filename, img)
        return True

def create_training_gif(frames_dir, output_path, fps=2):
    """Creates a GIF from saved frames in a directory."""
    try:
        image_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
        
        def sort_key(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0
        image_files.sort(key=sort_key)
        
        if not image_files:
            print(f"Warning: No image files found in '{frames_dir}'. Skipping GIF creation.")
            return

        frames = [imageio.imread(os.path.join(frames_dir, f)) for f in image_files]
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Training animation saved successfully to {output_path}")

    except FileNotFoundError:
        print(f"Warning: The directory '{frames_dir}' was not found. Skipping GIF creation.")
    except Exception as e:
        print(f"An error occurred during GIF creation: {e}")