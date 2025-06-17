from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from warehouse.env import WarehouseEnv


def evaluate_agent(model_path="models/dqn_warehouse_final.zip", episodes=5):
    env = DummyVecEnv([lambda: WarehouseEnv()])
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
