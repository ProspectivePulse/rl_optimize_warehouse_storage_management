# src/warehouse/train.py

import os
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Use relative imports for files within the same package
from .utils import load_config, SaveAnimationCallback, create_training_gif
from .env import WarehouseEnv
from .agent import create_dqn_model

def train():
    """Main training function."""
    # --- 1. Load Configuration ---
    config = load_config()
    paths = config['paths']
    training_params = config['training']
    env_params = config['env_params']
    animation_params = config['animation']
    
    # --- 2. Create and Wrap the Environment ---
    print("Creating training environment...")
    env = WarehouseEnv(
        grid_size=env_params['grid_size'],
        num_items=env_params['num_items'],
        max_steps=env_params['max_steps'],
        render_mode="rgb_array"
    )
    env = FlattenObservation(env)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # --- 3. Set up Callbacks ---
    print("Setting up callbacks...")
    eval_freq = training_params['eval_freq']
    
    # Evaluation environment setup
    eval_env = WarehouseEnv(
        grid_size=env_params['grid_size'],
        num_items=env_params['num_items'],
        max_steps=env_params['max_steps']
    )
    eval_env = FlattenObservation(eval_env)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])

    # Create paths for models if they don't exist
    os.makedirs(paths['models_dir'], exist_ok=True)
    best_model_path = os.path.join(paths['models_dir'], paths['best_model_name'])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=paths['models_dir'],
        log_path=paths['logs_dir'],
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    animation_callback = SaveAnimationCallback(
        save_freq=animation_params['save_freq'],
        animation_dir=paths['animation_frames_dir']
    )
    
    callbacks = [eval_callback, animation_callback]

    # --- 4. Create and Train the Agent ---
    print("Creating DQN agent...")
    model = create_dqn_model(env, config)

    print(f"Starting training for {training_params['total_timesteps']} timesteps...")
    model.learn(
        total_timesteps=training_params['total_timesteps'],
        callback=callbacks,
        log_interval=training_params['log_interval']
    )
    
    # --- 5. Save Final Model and Create Training GIF ---
    final_model_path = os.path.join(paths['models_dir'], paths['final_model_name'])
    model.save(final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    create_training_gif(
        frames_dir=paths['animation_frames_dir'],
        output_path=paths['training_gif']
    )

if __name__ == '__main__':
    train()