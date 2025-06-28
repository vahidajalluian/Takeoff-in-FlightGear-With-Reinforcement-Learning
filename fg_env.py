# fg_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from warnings import warn

from fg_client import FgClient # Import your FgClient

class FlightGearEnv(gym.Env):
    """
    Custom Environment for FlightGear v1.0.
    Action Space: Continuous (Elevator, Aileron, Throttle)
    Observation Space: Continuous (Airspeed, Altitude, Vertical Speed, Pitch, Roll)
    Goal: Takeoff and climb to a target altitude (e.g., 2000 ft) while maintaining stability.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 10}

    def __init__(self, render_mode=None):
        super().__init__()

        self.fg_client = FgClient(host='127.0.0.1', port=5051, savelog=False) # No need for internal logging for RL

        # Define Action Space:
        # We'll use a continuous action space for finer control.
        # Actions: [elevator, aileron, throttle_change]
        # Elevator: -1 (pitch down) to 1 (pitch up)
        # Aileron: -1 (roll left) to 1 (roll right)
        # Throttle_change: -1 (decrease throttle) to 1 (increase throttle)
        # Max throttle will be 1.0, min 0.0
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0, -0.1]), # Small throttle change per step
                                       high=np.array([1.0, 1.0, 0.1]),
                                       dtype=np.float32)

        # Define Observation Space:
        # airspeed_kt (0-300)
        # altitude_ft (0-5000)
        # vertical_speed_fps (-50 to 50)
        # pitch_deg (-90 to 90)
        # roll_deg (-180 to 180)
        # current_elevator (-1 to 1)
        # current_aileron (-1 to 1)
        # current_throttle (0 to 1)

        # Normalize these to a 0-1 or -1 to 1 range for neural networks
        self.observation_space = spaces.Box(low=np.array([0., 0., -1., -1., -1., -1., -1., 0.]),
                                            high=np.array([1., 1., 1., 1., 1., 1., 1., 1.]),
                                            shape=(8,), dtype=np.float32)

        self.sim_time_step = 0.01  # Simulation step time in seconds
        self.current_throttle = 0.0 # Agent will adjust this

        # Takeoff/Climb targets
        self.target_takeoff_airspeed = 100 # knots
        self.target_altitude = 2000 # ft
        self.max_episode_steps = 5000 # Max steps per episode (e.g., 500 * 0.5s = 250s)

        self.steps_since_reset = 0
        self.render_mode = render_mode

        self.initial_alt_ft = 0.0 # To be set in reset
        self.initial_airspeed_kt = 0.0 # To be set in reset

    def _get_obs(self):
        """Fetches current observations from FlightGear and normalizes them."""
        airspeed = self.fg_client.get_prop_float('/velocities/airspeed-kt')
        altitude = self.fg_client.altitude_ft()
        vert_speed = self.fg_client.vertical_speed_fps()
        pitch = self.fg_client.get_prop_float('/orientation/pitch-deg')
        roll = self.fg_client.get_prop_float('/orientation/roll-deg')
        elevator = self.fg_client.get_elevator()
        aileron = self.fg_client.get_aileron() # Correct variable name

        # Normalize observations
        # Airspeed: 0-300 kt -> 0-1
        norm_airspeed = np.clip(airspeed / 300.0, 0.0, 1.0)
        # Altitude: 0-5000 ft (for initial climb) -> 0-1
        norm_altitude = np.clip(altitude / 5000.0, 0.0, 1.0)
        # Vertical speed: -50 to 50 fps -> -1 to 1
        norm_vert_speed = np.clip(vert_speed / 50.0, -1.0, 1.0)
        # Pitch: -90 to 90 deg -> -1 to 1
        norm_pitch = np.clip(pitch / 90.0, -1.0, 1.0)
        # Roll: -180 to 180 deg -> -1 to 1
        norm_roll = np.clip(roll / 180.0, -1.0, 1.0)
        # Control surfaces are already -1 to 1
        norm_elevator = elevator
        norm_aileron = aileron # Corrected to 'aileron'
        # Throttle is 0 to 1
        norm_throttle = self.current_throttle

        return np.array([norm_airspeed, norm_altitude, norm_vert_speed,
                            norm_pitch, norm_roll, norm_elevator, norm_aileron,
                            norm_throttle], dtype=np.float32)

    def _get_info(self):
        """Returns additional info for debugging/logging."""
        return {
            "airspeed_kt": self.fg_client.get_prop_float('/velocities/airspeed-kt'),
            "altitude_ft": self.fg_client.altitude_ft(),
            "vertical_speed_fps": self.fg_client.vertical_speed_fps(),
            "pitch_deg": self.fg_client.get_prop_float('/orientation/pitch-deg'),
            "roll_deg": self.fg_client.get_prop_float('/orientation/roll-deg'),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("\n--- Resetting FlightGear Environment ---")
        # Reset the simulator to a known good state.
        # This requires FlightGear to be running in a specific location (e.g., EHAM runway)
        # and needs to be *fast* for RL training.
        # The best way is often to use FlightGear's built-in reset commands or
        # set specific lat/lon/alt/heading.
        # For simplicity, we'll just set key properties here.
        # A more robust solution might involve loading a saved scenario.
        
        # Ensure pause is off
        self.fg_client.set_prop('/sim/pause', 0)
        time.sleep(3) # Short pause
        # Reset controls
        self.fg_client.set_elevator(0.0)
        self.fg_client.set_aileron(0.0)
        self.current_throttle = 0 # Set initial throttle for takeoff attempt
        self.fg_client.set_throttle(self.current_throttle)

        self.fg_client.set_throttle(0.0) # Start with idle throttle
        self.fg_client.set_prop('/controls/gear/brake-parking', 1) # Brakes on
        self.fg_client.set_prop('/controls/flight/flaps', 0.0) # Flaps up

        # Reset location (example: Schiphol EHAM runway 22, C172P default spawn)
        # These properties need to be set precisely for the aircraft to reset on the runway
        # This is often the trickiest part of FlightGear RL environments.
        # If the aircraft spawns in the air or crashes immediately, adjust these.
        
        self.fg_client.set_prop('/position/latitude-deg', 37.606867)
        self.fg_client.set_prop('/position/longitude-deg', -122.380681)
        self.fg_client.set_prop('/position/altitude-ft', 20.838517 ) 
        self.fg_client.set_prop('/orientation/heading-deg', 26.988086) 
        self.fg_client.set_prop('/orientation/pitch-deg', 0.216976)
        self.fg_client.set_prop('/orientation/roll-deg', 0.366835)
        self.fg_client.set_prop('/velocities/airspeed-kt', 0.0)
        self.fg_client.set_prop('/velocities/vertical-speed-fps', 0.0)
        self.fg_client.set_prop('/velocities/pitch-rate', 0.0)
        self.fg_client.set_prop('/velocities/roll-rate', 0.0)
        self.fg_client.set_prop('/velocities/heading-rate', 0.0)

        # Allow time for FlightGear to process reset commands
        time.sleep(0.5) # Short pause
        self.fg_client.set_prop('/sim/pause', 1)

        self.fg_client.set_prop('/controls/gear/brake-parking', 0) # Release brakes for takeoff
        self.fg_client.resetSim()
        time.sleep(0.5) # Short pause
        
        self.current_throttle = 0.75 # Set initial throttle for takeoff attempt
        self.fg_client.set_throttle(self.current_throttle)

        self.steps_since_reset = 0
        # self.initial_alt_ft = self.fg_client.altitude_ft()
        # self.initial_airspeed_kt = self.fg_client.get_prop_float('/velocities/airspeed-kt')

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self.steps_since_reset += 1

        # Apply actions
        elevator_cmd, aileron_cmd, throttle_change_cmd = action
        print(elevator_cmd, aileron_cmd, throttle_change_cmd)
        # Apply elevator and aileron directly
        self.fg_client.set_elevator(elevator_cmd)
        self.fg_client.set_aileron(aileron_cmd)

        # Update and apply throttle
        self.current_throttle = np.clip(self.current_throttle + throttle_change_cmd, 0.0, 1.0)
        self.fg_client.set_throttle(self.current_throttle)

        # Advance simulator time
        self.fg_client.tic()
        self.fg_client.toc(self.sim_time_step)

        # Get new observations
        observation = self._get_obs()
        info = self._get_info()

        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False

        airspeed = info["airspeed_kt"]
        altitude = info["altitude_ft"]
        vertical_speed = info["vertical_speed_fps"]
        pitch = info["pitch_deg"]
        roll = info["roll_deg"]

        # Reward for gaining airspeed during takeoff roll
        if altitude < self.initial_alt_ft + 50: # Still on or just above ground
            reward += np.clip(airspeed / self.target_takeoff_airspeed, 0.0, 1.0) * 0.1 # Max 0.1 reward
            # Reward for increasing altitude
            if altitude > self.initial_alt_ft + 10: # If actually lifted off
                 reward += 1.0 # Significant reward for liftoff
        else: # Once airborne
            # Reward for climbing towards target altitude
            if altitude < self.target_altitude:
                alt_diff = self.target_altitude - altitude
                reward += (1.0 - np.clip(alt_diff / (self.target_altitude - self.initial_alt_ft), 0.0, 1.0)) * 1.0
                # Reward for positive vertical speed
                reward += np.clip(vertical_speed / 20.0, 0.0, 1.0) * 0.5 # Max 0.5 for good climb rate
            else:
                # Reached target altitude - give large reward and terminate
                reward += 10.0
                terminated = True
                print(f"SUCCESS: Reached target altitude {self.target_altitude} ft!")

            # Penalize excessive pitch/roll for stability
            reward -= abs(pitch) * 0.01 # Mild penalty for nose up/down
            reward -= abs(roll) * 0.005 # Milder penalty for roll

            # Penalize excessive vertical speed or negative vertical speed when airborne
            if vertical_speed < -5.0: # Rapid descent
                reward -= 2.0
            if vertical_speed > 30.0: # Too steep climb
                reward -= 0.5

            # Penalize if airspeed drops too low after takeoff (stall risk)
            if altitude > self.initial_alt_ft + 50 and airspeed < 50:
                reward -= 1.0

        # Penalize for crashing (altitude too low after liftoff)
        if altitude < self.initial_alt_ft and airspeed > 10: # Below runway level and moving fast
            reward -= 10.0
            terminated = True
            print("CRASHED: Altitude too low!")

        # Penalize if aircraft goes wildly out of control (e.g., extreme pitch/roll)
        if abs(pitch) > 45 or abs(roll) > 90:
            reward -= 5.0
            terminated = True
            print("CRASHED: Out of control (extreme pitch/roll)!")

        # Episode termination if max steps reached
        if self.steps_since_reset >= self.max_episode_steps:
            truncated = True
            print("Episode truncated: Max steps reached.")

        return observation, reward, terminated, truncated, info

    def render(self):
        # In 'human' mode, FlightGear itself is the rendering.
        # We can print current state for console output.
        if self.render_mode == 'human':
            info = self._get_info()
            print(f"Step: {self.steps_since_reset:4d} | "
                  f"Airspeed: {info['airspeed_kt']:6.2f} kt | "
                  f"Altitude: {info['altitude_ft']:7.2f} ft | "
                  f"VS: {info['vertical_speed_fps']:6.2f} fps | "
                  f"Pitch: {info['pitch_deg']:6.2f} deg | "
                  f"Roll: {info['roll_deg']:6.2f} deg | "
                  f"Elev: {self.fg_client.get_elevator():5.2f} | "
                  f"Ail: {self.fg_client.get_aileron():5.2f} | "
                  f"Thr: {self.current_throttle:5.2f} | "
                  f"Reward: {self.total_reward:7.2f}") # self.total_reward will be updated in train_agent

    def close(self):
        if self.fg_client:
            self.fg_client.close()
            print("FlightGear client closed.")
