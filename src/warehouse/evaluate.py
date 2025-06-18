from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from warehouse.env import WarehouseEnv
from warehouse.utils import load_config


def evaluate_agent():
    config = load_config()

    model_path = config["train"]["save_path"] + ".zip"  # e.g. "models/trained_agent.zip"
    episodes = config.get("evaluate", {}).get("num_episodes", 5)
    render = config.get("evaluate", {}).get("render", False)

    env = DummyVecEnv([
        lambda: WarehouseEnv(
            grid_size=config["env"]["grid_size"],
            num_items=config["env"]["num_items"],
            max_steps=config["env"]["max_steps"]
        )
    ])
    
    model = DQN.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward

        print(f"Episode {ep + 1}: Total Reward = {total_reward:.2f}")


if __name__ == "__main__":
    evaluate_agent()
