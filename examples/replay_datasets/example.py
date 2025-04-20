import numpy as np
import os
from pathlib import Path
# Import the KBotV2 environment directly
from loco_mujoco.environments import KBotV2

# --- Configuration ---
# Define the EXACT path to YOUR local trajectory file (still using the G1 squat)
local_g1_squat_traj_path = Path("/home/ali/repos/ksim_stuff/loco-mujoco/loco_mujoco/datasets/local/UnitreeG1/squat.npz")

# Check if the source trajectory file exists
if not local_g1_squat_traj_path.exists():
    print(f"Error: Local G1 trajectory file not found at {local_g1_squat_traj_path}")
    print("Please ensure the path is correct.")
    exit()

# --- Create KBotV2 Environment ---
print("Creating KBotV2 environment...")
env = KBotV2()
print("Environment created.")

# --- Load the Local (Unitree G1) Trajectory ---
print(f"Loading G1 trajectory from {local_g1_squat_traj_path} into KBotV2...")
# Use the load_trajectory method, suppressing the model mismatch warning
env.load_trajectory(traj_path=str(local_g1_squat_traj_path))
print("Trajectory loaded.")

# --- Play the Trajectory ---
print("Playing G1 trajectory on KBotV2 (expect incorrect motion)...")
# Play one short episode
env.play_trajectory(n_episodes=1, n_steps_per_episode=500, render=True)
print("Playback finished.")


