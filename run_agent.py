# run_agent.py
from stable_baselines3 import PPO
from stable_baselines3 import SAC
import time

from fg_env import FlightGearEnv

# Path to your trained model
# MODEL_PATH = "fg_ppo_takeoff_climb_final.zip" # Or a checkpoint like fg_ppo_model_100000_steps.zip
MODEL_PATH = "fg_SAC_takeoff_climb.zip" # Or a checkpoint like fg_ppo_model_100000_steps.zip
print(f"Loading model from {MODEL_PATH}...")
try:
    # Create the environment instance (same as training)
    env = FlightGearEnv(render_mode='human') # Render mode is human for visualization

    # Load the trained model
    # The 'env' argument is important for the model to know the observation/action spaces
    # model = PPO.load(MODEL_PATH, env=env)
    model = SAC.load(MODEL_PATH, env=env)
    print("Model loaded successfully.")

    # Run the agent for a few episodes
    num_episodes = 5
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        print(f"\n--- Starting Episode {episode + 1} ---")
        while not done and not truncated:
            # Predict the action (deterministic=True for deployment)
            action, _states = model.predict(obs, deterministic=True)

            # Take a step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated # If terminated (e.g., crash or success), stop
            episode_reward += reward
            step_count += 1

            # Print current state (optional, for monitoring)
            # This is already handled by env.render() if you call it, but can be verbose.
            # For deployment, you might only print key metrics.
            # if step_count % 10 == 0: # Print every 10 steps
            #     env.render()

            # Add a small delay for human observation
            # time.sleep(0.1) # This is in addition to the env's sim_time_step

        print(f"--- Episode {episode + 1} Finished ---")
        print(f"Total Steps: {step_count}")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Final Altitude: {info['altitude_ft']:.2f} ft")
        print(f"Final Airspeed: {info['airspeed_kt']:.2f} kt")
        if terminated:
            print("Episode terminated.")
        if truncated:
            print("Episode truncated (max steps reached).")


except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'env' in locals() and env:
        env.close()
        print("Environment closed.")
