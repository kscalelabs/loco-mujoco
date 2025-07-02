import numpy as np
from loco_mujoco.task_factories import ImitationFactory, LAFAN1DatasetConf, DefaultDatasetConf, AMASSDatasetConf


def experiment(seed=0):
#                           "CMU/CMU/91/91_03_poses",

    np.random.seed(seed)

    cmu_walking_dataset = [#"CMU/CMU/91/91_02_poses",
                           "CMU/CMU/91/91_03_poses",
                        #    "CMU/CMU/91/91_04_poses",
                        #    "CMU/CMU/91/91_10_poses",
                        #    "CMU/CMU/91/91_11_poses",
                        #    "CMU/CMU/91/91_12_poses",
                        #    "CMU/CMU/91/91_13_poses",
                        #    "CMU/CMU/91/91_14_poses",
                        #    "CMU/CMU/91/91_15_poses",
                        #    "CMU/CMU/91/91_17_poses",
                        #    "CMU/CMU/91/91_18_poses",
                        #    "CMU/CMU/91/91_19_poses",
                        #    "CMU/CMU/91/91_20_poses",
                        #    "CMU/CMU/91/91_21_poses",
                           "CMU/CMU/91/91_22_poses",
                        #    "CMU/CMU/91/91_23_poses",
                        #    "CMU/CMU/91/91_27_poses",
                        #    "CMU/CMU/91/91_28_poses",
                        #    "CMU/CMU/91/91_29_poses",
                        #    "CMU/CMU/91/91_30_poses",
                        #    "CMU/CMU/91/91_31_poses",
                        #    "CMU/CMU/91/91_32_poses",
                        #    "CMU/CMU/91/91_33_poses",
                        #    "CMU/CMU/91/91_34_poses",
                        #    "CMU/CMU/91/91_35_poses",
                        #    "CMU/CMU/91/91_36_poses",
                        #    "CMU/CMU/91/91_37_poses",
                        #    "CMU/CMU/91/91_38_poses",
                        #    "CMU/CMU/91/91_57_poses",
                           ]

    dances = [
        "DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses",
        "DanceDB/DanceDB/20120731_StefanosTheodorou/Stefanos_1os_antrikos_karsilamas_C3D_poses",
        "DanceDB/DanceDB/20120805_VasoAristeidou/Vaso_Aristeidou_Zeimpekiko_v1_poses",
        "DanceDB/DanceDB/20120807_CliodelaVara/Clio_Flamenco_C3D_poses",
        "DanceDB/DanceDB/20120807_VasoAristeidou/Vasso_Bachata_01_poses",
        "DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v3_C3D_poses",
        "DanceDB/DanceDB/20130216_AnnaCharalambous/Anna_Curiosity_C3D_poses",
        "DanceDB/DanceDB/20130216_AnnaCharalambous/Anna_Happy_C3D_poses",
        "DanceDB/DanceDB/20130216_AnnaCharalambous/Anna_Sad_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Annoyed_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Bored_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Excited_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Happy_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Relaxed_C3D_poses",
        "DanceDB/DanceDB/20131001_OliviaKyriakides/Olivia_Mix_C3D_poses",
        "DanceDB/DanceDB/20131001_SophieKamenou/Sophie_Afraid_C3D_poses",
        "DanceDB/DanceDB/20131001_SophieKamenou/Sophie_Tired_C3D_poses",
        "DanceDB/DanceDB/20131001_SophieKamenou/Sophie_Happy_C3D_poses",
        "DanceDB/DanceDB/20140506_AnnaCortesi/AnnaCortesi_BellyDance2_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Bachata_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Bachata_v2_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Reggaeton_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Salsa_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Zeibekiko_Fast_C3D_poses",
        "DanceDB/DanceDB/20140526_StephanosKoullapis/StefanosKoullapis_Zeibekiko_Slow_C3D_poses",

    ]

    # Good dances:
    # 18 - belly dance?
    # 19 - Bachata
    # 20 - Bachata 2
    # 21 - Reggaeton
    # 22 - Salsa

    # # example --> you can add as many datasets as you want in the lists!
    env = ImitationFactory.make("KBotV2",
                                # if SMPL and AMASS are installed, you can use the following:
                                # amass_dataset_conf=AMASSDatasetConf(["DanceDB/DanceDB/20120911_TheodorosSourmelis/Capoeira_Theodoros_v2_C3D_poses"]),
                                # amass_dataset_conf=AMASSDatasetConf(dances[22]),
                                amass_dataset_conf=AMASSDatasetConf(cmu_walking_dataset),
                                n_substeps=20)

    traj = env.th.traj

    traj.save("walk_3_22.npz")

    # env.play_trajectory(n_episodes=3, n_steps_per_episode=10000, render=True)

    # breakpoint()
if __name__ == '__main__':
    experiment()
