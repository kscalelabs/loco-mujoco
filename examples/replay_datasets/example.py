import numpy as np
import os
from pathlib import Path
# Import the specific environment directly
from loco_mujoco.environments import UnitreeG1

# --- Configuration ---
# Define the EXACT path to YOUR local trajectory file
# Use the absolute path you provided
local_squat_traj_path = Path("/home/ali/repos/ksim_stuff/loco-mujoco/loco_mujoco/datasets/local/UnitreeG1/squat.npz")

# Check if the source trajectory file exists
if not local_squat_traj_path.exists():
    print(f"Error: Local trajectory file not found at {local_squat_traj_path}")
    print("Please ensure the path is correct.")
    exit()

# --- Create UnitreeG1 Environment ---
print("Creating UnitreeG1 environment...")
# Instantiate the environment directly.
# Remove th_params, load_trajectory will handle control_dt.
env = UnitreeG1()
print("Environment created.")

# --- Load the Local Trajectory ---
print(f"Loading trajectory from {local_squat_traj_path}...")
# Use the correct load_trajectory method with traj_path
env.load_trajectory(traj_path=str(local_squat_traj_path))
print("Trajectory loaded.")

# --- Play the Trajectory ---
print("Playing trajectory...")
# Play one short episode
env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)
print("Playback finished.")


