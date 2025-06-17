import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from warehouse.env import WarehouseEnv
from warehouse.agent import create_agent


def train_agent(total_timesteps=20000):
    env = DummyVecEnv([lambda: Monitor(WarehouseEnv())])
    eval_env = DummyVecEnv([lambda: WarehouseEnv()])

    model = create_agent(env)

    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path="models/dqn/", name_prefix="warehouse_dqn"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn_best/",
        log_path="logs/dqn_eval/",
        eval_freq=5000,
        deterministic=True,
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, eval_callback])

    model.save("models/dqn_warehouse_final")
    print("Training complete. Model saved.")


if __name__ == "__main__":
    train_agent()
