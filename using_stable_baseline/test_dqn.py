from stable_baselines3 import DQN
from carlatest import CarlaEnv

env = CarlaEnv()
model = DQN.load("dqn", env=env)

obs = env.reset()

max_steps = 1000
episode_steps = 0
episode_num = 1

successes = 0
collisions = 0
timeouts = 0

print(f"Starting evaluation for {max_steps} total steps")

for i in range(max_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    episode_steps += 1

    if done:
        print(f"\n📘 Episode {episode_num} ended after {episode_steps} steps")

        if reward >= 100:
            print("Success: Reached goal!")
            successes += 1
        elif reward <= -100:
            print("Crash detected!")
            collisions += 1
        else:
            print("Timed out or unknown reason")
            timeouts += 1

        print(f"  Distance traveled this episode: {env.get_total_distance():.2f} meters")

        episode_num += 1
        episode_steps = 0
        obs = env.reset()

        

# Final summary
print("\n Evaluation complete.")
print(f"Total episodes: {episode_num - 1}")
print(f"Successes:   {successes}")
print(f"Collisions: {collisions}")
print(f"Timeouts:    {timeouts}")
