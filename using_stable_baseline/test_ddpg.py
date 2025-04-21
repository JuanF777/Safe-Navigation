from stable_baselines3 import DDPG
from carlatest_ddpg import CarlaEnv

env = CarlaEnv()
model = DDPG.load("ddpg", env=env)

obs = env.reset()

max_steps = 2000
episode_steps = 0
episode_num = 1

successes = 0
collisions = 0
timeouts = 0

print(f"Starting DDPG evaluation for {max_steps} total steps")

for i in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    print(f"[Step {i}] Rel_X: {obs[0]:.2f}, Rel_Y: {obs[1]:.2f}, Angle: {obs[4]:.2f}, Steer: {action[0]:.2f}")

    episode_steps += 1

    if done:
        print(f"\n Episode {episode_num} ended after {episode_steps} steps")

        if reward >= 100:
            print("Success: Reached goal!")
            successes += 1
        elif reward <= -100:
            print("Crash detected!")
            collisions += 1
        else:
            print("Timeout")
            timeouts += 1

        print(f" Distance traveled: {env.get_total_distance():.2f} m")

        episode_num += 1
        episode_steps = 0
        obs = env.reset()

print("\nEvaluation complete.")
print(f"Total episodes: {episode_num - 1}")
print(f"Successes: {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts: {timeouts}")
