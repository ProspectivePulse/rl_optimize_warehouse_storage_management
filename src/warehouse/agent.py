# src/warehouse/agent.py

import torch
from stable_baselines3 import DQN

def create_dqn_model(env, config):
    """
    Creates a DQN model with parameters from the config file.
    """
    model_params = config['model_params']
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Combine train_freq from config into the expected tuple format
    train_freq = (model_params['train_freq'], "step")

    model = DQN(
        env=env,
        policy=model_params['policy'],
        learning_rate=model_params['learning_rate'],
        buffer_size=model_params['buffer_size'],
        learning_starts=model_params['learning_starts'],
        batch_size=model_params['batch_size'],
        gamma=model_params['gamma'],
        train_freq=train_freq,
        target_update_interval=model_params['target_update_interval'],
        exploration_fraction=model_params['exploration_fraction'],
        exploration_final_eps=model_params['exploration_final_eps'],
        verbose=model_params['verbose'],
        tensorboard_log=config['paths']['logs_dir'],
        device=device
    )
    return model