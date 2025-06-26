import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):

    np.random.seed(seed)

    cmu_walking_dataset = ["CMU/CMU/91/91_02_poses",
                           "CMU/CMU/91/91_03_poses",
                           "CMU/CMU/91/91_04_poses",
                           "CMU/CMU/91/91_10_poses",
                           "CMU/CMU/91/91_11_poses",
                           "CMU/CMU/91/91_12_poses",
                           "CMU/CMU/91/91_13_poses",
                           "CMU/CMU/91/91_14_poses",
                           "CMU/CMU/91/91_15_poses",
                           "CMU/CMU/91/91_17_poses",
                           "CMU/CMU/91/91_18_poses",
                           "CMU/CMU/91/91_19_poses",
                           "CMU/CMU/91/91_20_poses",
                           "CMU/CMU/91/91_21_poses",
                           "CMU/CMU/91/91_22_poses",
                           "CMU/CMU/91/91_23_poses",
                           "CMU/CMU/91/91_27_poses",
                           "CMU/CMU/91/91_28_poses",
                           "CMU/CMU/91/91_29_poses",
                           "CMU/CMU/91/91_30_poses",
                           "CMU/CMU/91/91_31_poses",
                           "CMU/CMU/91/91_32_poses",
                           "CMU/CMU/91/91_33_poses",
                           "CMU/CMU/91/91_34_poses",
                           "CMU/CMU/91/91_35_poses",
                           "CMU/CMU/91/91_36_poses",
                           "CMU/CMU/91/91_37_poses",
                           "CMU/CMU/91/91_38_poses",
                           "CMU/CMU/91/91_57_poses",
                           ]

    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("KBotV2",
                                # if SMPL and AMASS are installed, you can use the following:
                                # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"]),
                                amass_dataset_conf=AMASSDatasetConf(cmu_walking_dataset),
                                n_substeps=20)

    traj = env.th.traj

    traj.save("cmu_walking_91.npz")

    # env.play_trajectory(n_episodes=3, n_steps_per_episode=10000, render=True)

    # breakpoint()
if __name__ == '__main__':
    experiment()
