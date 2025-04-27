import mujoco
import time
import logging
import os
import jax
from loco_mujoco.environments import LocoEnv
from loco_mujoco.smpl.retargeting import load_robot_conf_file, to_t_pose

# --- Configuration ---
env_name = "KBotV2"
# env_name = "UnitreeG1"
# env_name = "UnitreeH1"
# env_name = "ToddlerBot"
# -------------------

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.info(f"Visualizing T-pose for {env_name}")

env = None

# Load config and initialize environment
robot_conf = load_robot_conf_file(env_name)
env_cls = LocoEnv.registered_envs[env_name]
env_params = getattr(robot_conf, 'env_params', {})
th_params = dict(random_start=False, fixed_start_conf=(0, 0))
env = env_cls(**env_params, th_params=th_params)
env.reset(jax.random.key(0))

# Set T-pose and update simulation state
to_t_pose(env, robot_conf)
mujoco.mj_forward(env._model, env._data)

# Render loop
logging.info("Starting render loop. Close window or Ctrl+C to exit.")
env.render() # Initialize viewer
if env.viewer is None:
    logging.error("Failed to initialize viewer.")
    exit()

try:
    while True:
        env.render()
        time.sleep(0.01)
except KeyboardInterrupt:
    logging.info("Render loop interrupted. Exiting...")


