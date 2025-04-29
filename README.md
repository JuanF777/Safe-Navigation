# Safe-Navigation
This is a repository for implementing safe navigation using DQN and DDPG models to train and test in CARLA simulator

Folder using_stable_baseline: Consists of multiple files of environment set up, training, and testing of DQN and DDPG algorithms in CARLA environment.

Each algorithm includes:
- Carla set up file - setting up the envionrment and functions needed for the agent to perform actions
- Training: Define the algorith using stable baseline policy, train the model for set amount of steps, save into a .zip file
-Test: Test given .zip file for given number of steps.