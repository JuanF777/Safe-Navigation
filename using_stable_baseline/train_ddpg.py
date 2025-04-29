# train_ddpg.py

from stable_baselines3 import DDPG
from stable_baselines3.ddpg.policies import MlpPolicy
from carlatest_ddpg import CarlaEnv

env = CarlaEnv()

model = DDPG(
    MlpPolicy,
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=5000,
    batch_size=128,
    tau=0.005,
    gamma=0.99,
    train_freq=(1, "step"),
    gradient_steps=1,
    tensorboard_log="./ddpg_logs/",
    policy_kwargs=dict(net_arch=[256, 256])
)

successes, collisions, timeouts = 0, 0, 0
episodes = 0
max_steps = 200000

obs = env.reset()

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=False) 
    obs, reward, done, info = env.step(action)

    if done:
        if info.get("event") == "collision":
            collisions += 1
        elif info.get("event") == "timeout":
            timeouts += 1
        elif reward > 0:
            successes += 1
        else:
            collisions += 1 
        episodes += 1
        obs = env.reset()

model.save("ddpg_forward")
print("\nTraining complete.")
print(f"Total training episodes: {episodes}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts: {timeouts}")

env.close()
