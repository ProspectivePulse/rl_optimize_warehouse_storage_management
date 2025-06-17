from stable_baselines3 import DQN

def create_agent(env, tensorboard_log="./logs/dqn_warehouse/"):
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=10,
        batch_size=64,
        gamma=0.99,
        train_freq=(4, "step"),
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device="auto",
    )
    return model
