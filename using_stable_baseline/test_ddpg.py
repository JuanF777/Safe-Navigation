# test_ddpg.py

from stable_baselines3 import DDPG
from carlatest_ddpg import CarlaEnv

env = CarlaEnv()
model = DDPG.load("ddpg_forward", env=env)

obs = env.reset()

successes, collisions, timeouts = 0, 0, 0
episodes = 0
max_steps = 40000

for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)  # fully deterministic now
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

print("\nTesting complete.")
print(f"Total testing episodes: {episodes}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts: {timeouts}")

env.close()
