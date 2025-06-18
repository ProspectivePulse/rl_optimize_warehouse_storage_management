import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from warehouse.env import WarehouseEnv
from warehouse.agent import create_agent
from warehouse.utils import load_config


def train_agent():
    # Load config.yaml from root directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    config = load_config(config_path)

    # Extract configuration sections
    env_config = config["env"]
    agent_config = config["agent"]
    train_config = config["train"]

    # Create monitored training and evaluation environments
    env = DummyVecEnv([lambda: Monitor(WarehouseEnv(**env_config))])
    eval_env = DummyVecEnv([lambda: WarehouseEnv(**env_config)])

    # Create the agent with config
    model = create_agent(env, agent_config)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=train_config["save_path"],
        name_prefix="warehouse_dqn"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(train_config["save_path"], "best"),
        log_path="logs/dqn_eval/",
        eval_freq=5000,
        deterministic=True,
    )

    # Train model
    total_timesteps = train_config["total_episodes"] * train_config["max_timesteps"]
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    # Save final model
    model.save(os.path.join(train_config["save_path"], "dqn_warehouse_final"))
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train_agent()
