import numpy as np
import os
from pathlib import Path
# Import the KBotV2 environment directly
from loco_mujoco.environments import KBotV2

# --- Configuration ---
# Define the EXACT path to YOUR local trajectory file
local_squat_traj_path = Path("loco_mujoco/datasets/local/UnitreeG1/squat.npz")

# Check if the file exists
if not local_squat_traj_path.exists():
    print(f"Error: Local trajectory file not found at {local_squat_traj_path}")
    exit()

print(f"Loading data from: {local_squat_traj_path}")

# Load the .npz file
try:
    data = np.load(local_squat_traj_path, allow_pickle=True)
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Print the keys (names of arrays stored in the file)
print("\nKeys found in the file:")
keys = list(data.keys())
print(keys)

# Print shape and head for each array
print("\nData Inspection (Head):")
for key in keys:
    try:
        array = data[key]
        print(f"\n--- Key: '{key}' ---")
        print(f"Shape: {array.shape}")
        print(f"Data Type: {array.dtype}")
        # Print the first 5 rows (or fewer if less than 5 rows exist)
        head_rows = min(5, array.shape[0]) if len(array.shape) > 0 else 0
        if head_rows > 0:
            print("Head:")
            print(array[:head_rows])
        elif len(array.shape) == 0: # Handle scalar arrays
             print(f"Value: {array}")
        else: # Handle empty arrays
            print("Array is empty.")
    except Exception as e:
        print(f"Could not process key '{key}': {e}")

# Close the file handle
data.close()

print("\nInspection finished.")

