# train_agent.py
import os
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from fg_env import FlightGearEnv

# Create logs directory if it doesn't exist
log_dir = "./logs_rl/"
os.makedirs(log_dir, exist_ok=True)

# Create the FlightGear environment
# render_mode='human' means FlightGear window will show the simulation
env = FlightGearEnv(render_mode='human')

# It's good practice to wrap environments in a VecEnv for Stable Baselines3
# Even for a single environment, this makes it compatible with more features.
# You could use multiple environments for parallel training later (SubprocVecEnv)
vec_env = make_vec_env(lambda: env, n_envs=1) # n_envs=1 for single simulation

# Initialize the PPO agent
# MlpPolicy: Multi-layer Perceptron (feedforward neural network) policy
# gamma: discount factor (how much to value future rewards)
# n_steps: number of steps to run for each environment per update
# ent_coef: entropy coefficient (encourages exploration)
# vf_coef: value function coefficient
# learning_rate: controls how much the model parameters are adjusted
# verbose: 1 for progress bar
# model = PPO("MlpPolicy", vec_env, verbose=1,
#             gamma=0.99, n_steps=2048, ent_coef=0.01, vf_coef=0.5,
#             learning_rate=0.0003,
#             tensorboard_log="./ppo_flightgear_tensorboard/")
model = SAC("MlpPolicy", vec_env, verbose=1,
            gamma=0.99, ent_coef=0.01,
            learning_rate=0.0003,
            tensorboard_log="./sac_flightgear_tensorboard/")
# Create a callback to save the model every 100,000 steps
checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path="./models/",
                                         name_prefix="fg_SAC_model")

print("Starting training...")
try:
    for i in range(1,10):
    # Train the agent for a large number of timesteps
    # The more complex the task, the more timesteps are needed.
    # This might take hours or days depending on your setup.
        model.learn(total_timesteps=10, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("Training interrupted.")
finally:
    # Save the final model
    model.save("fg_SAC_takeoff_climb_final")
    print("Model saved as fg_SAC_takeoff_climb_final.zip")
    # Close the environment
    vec_env.close()
    print("Environment closed.")
