# src/warehouse/evaluate.py

import os
import imageio
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN

from .utils import load_config
from .env import WarehouseEnv

def evaluate():
    """Main evaluation function."""
    # --- 1. Load Configuration and Model ---
    config = load_config()
    paths = config['paths']
    env_params = config['env_params']

    model_path = os.path.join(paths['models_dir'], paths['best_model_name']) # Evaluate the best model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run training first.")
        return

    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path)

    # --- 2. Create Environment ---
    print("Creating evaluation environment...")
    eval_env = WarehouseEnv(
        grid_size=env_params['grid_size'],
        num_items=env_params['num_items'],
        max_steps=env_params['max_steps'],
        render_mode="rgb_array"
    )
    eval_env_wrapped = FlattenObservation(eval_env)

    # --- 3. Run a Single Episode ---
    frames = []
    obs, _ = eval_env_wrapped.reset()
    print("Starting evaluation episode...")

    for step in range(env_params['max_steps']):
        frames.append(eval_env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env_wrapped.step(action)
        
        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps.")
            break
    
    eval_env_wrapped.close()
    
    # --- 4. Save the GIF ---
    if frames:
        print(f"Saving evaluation GIF to {paths['evaluation_gif']}...")
        imageio.mimsave(paths['evaluation_gif'], frames, fps=10, loop=0)
        print("Evaluation GIF saved successfully.")
    else:
        print("Warning: No frames were captured during evaluation.")

if __name__ == '__main__':
    evaluate()