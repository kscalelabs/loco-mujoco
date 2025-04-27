from typing import List, Union, Tuple
import numpy as np
import mujoco
from mujoco import MjSpec

import loco_mujoco
from loco_mujoco.core import ObservationType, Observation
from loco_mujoco.environments.humanoids.base_robot_humanoid import BaseRobotHumanoid
from loco_mujoco.core.utils import info_property


class KBotV2(BaseRobotHumanoid):
    """
    Mujoco environment of the KBotV2 robot.
    """

    mjx_enabled = False  # Set to True if you create a Mjx-compatible version

    def __init__(
        self,
        spec: Union[str, MjSpec] = None,
        observation_spec: List[Observation] = None,
        actuation_spec: List[str] = None,
        **kwargs,
    ) -> None:
        """
        Constructor.

        Args:
            spec (Union[str, MjSpec]): Specification of the environment. Can be a path to the XML file or an MjSpec object.
                If none is provided, the default XML file is used.
            observation_spec (List[Observation], optional): List defining the observation space. Defaults to None.
            actuation_spec (List[str], optional): List defining the action space. Defaults to None.
            **kwargs: Additional parameters for the environment.
        """

        if spec is None:
            spec = self.get_default_xml_file_path()

        # load the model specification
        spec = mujoco.MjSpec.from_file(spec) if not isinstance(spec, MjSpec) else spec

        # get the observation and action specification
        if observation_spec is None:
            # get default
            observation_spec = self._get_observation_specification(spec)
        else:
            # parse
            observation_spec = self.parse_observation_spec(observation_spec)
        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        # modify the specification if needed (add logic here if needed for variations)
        # e.g., if self.mjx_enabled: spec = self._modify_spec_for_mjx(spec)

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            **kwargs,
        )

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[Observation]:
        """
        Returns the observation specification of the environment.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[Observation]: List of observations.
        """
        # Adapt joint names based on your kbot_v2.xml
        observation_spec = [  # ------------- JOINT POS -------------
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),
            ObservationType.JointPos("q_right_shoulder_pitch", xml_name="dof_right_shoulder_pitch_03"),
            ObservationType.JointPos("q_right_shoulder_roll", xml_name="dof_right_shoulder_roll_03"),
            ObservationType.JointPos("q_right_shoulder_yaw", xml_name="dof_right_shoulder_yaw_02"),
            ObservationType.JointPos("q_right_elbow", xml_name="dof_right_elbow_02"),
            ObservationType.JointPos("q_right_wrist", xml_name="dof_right_wrist_00"),
            ObservationType.JointPos("q_left_shoulder_pitch", xml_name="dof_left_shoulder_pitch_03"),
            ObservationType.JointPos("q_left_shoulder_roll", xml_name="dof_left_shoulder_roll_03"),
            ObservationType.JointPos("q_left_shoulder_yaw", xml_name="dof_left_shoulder_yaw_02"),
            ObservationType.JointPos("q_left_elbow", xml_name="dof_left_elbow_02"),
            ObservationType.JointPos("q_left_wrist", xml_name="dof_left_wrist_00"),
            ObservationType.JointPos("q_right_hip_pitch", xml_name="dof_right_hip_pitch_04"),
            ObservationType.JointPos("q_right_hip_roll", xml_name="dof_right_hip_roll_03"),
            ObservationType.JointPos("q_right_hip_yaw", xml_name="dof_right_hip_yaw_03"),
            ObservationType.JointPos("q_right_knee", xml_name="dof_right_knee_04"),
            ObservationType.JointPos("q_right_ankle", xml_name="dof_right_ankle_02"),
            ObservationType.JointPos("q_left_hip_pitch", xml_name="dof_left_hip_pitch_04"),
            ObservationType.JointPos("q_left_hip_roll", xml_name="dof_left_hip_roll_03"),
            ObservationType.JointPos("q_left_hip_yaw", xml_name="dof_left_hip_yaw_03"),
            ObservationType.JointPos("q_left_knee", xml_name="dof_left_knee_04"),
            ObservationType.JointPos("q_left_ankle", xml_name="dof_left_ankle_02"),
            # ------------- JOINT VEL -------------
            ObservationType.FreeJointVel("dq_root", xml_name="root"),
            ObservationType.JointVel("dq_right_shoulder_pitch", xml_name="dof_right_shoulder_pitch_03"),
            ObservationType.JointVel("dq_right_shoulder_roll", xml_name="dof_right_shoulder_roll_03"),
            ObservationType.JointVel("dq_right_shoulder_yaw", xml_name="dof_right_shoulder_yaw_02"),
            ObservationType.JointVel("dq_right_elbow", xml_name="dof_right_elbow_02"),
            ObservationType.JointVel("dq_right_wrist", xml_name="dof_right_wrist_00"),
            ObservationType.JointVel("dq_left_shoulder_pitch", xml_name="dof_left_shoulder_pitch_03"),
            ObservationType.JointVel("dq_left_shoulder_roll", xml_name="dof_left_shoulder_roll_03"),
            ObservationType.JointVel("dq_left_shoulder_yaw", xml_name="dof_left_shoulder_yaw_02"),
            ObservationType.JointVel("dq_left_elbow", xml_name="dof_left_elbow_02"),
            ObservationType.JointVel("dq_left_wrist", xml_name="dof_left_wrist_00"),
            ObservationType.JointVel("dq_right_hip_pitch", xml_name="dof_right_hip_pitch_04"),
            ObservationType.JointVel("dq_right_hip_roll", xml_name="dof_right_hip_roll_03"),
            ObservationType.JointVel("dq_right_hip_yaw", xml_name="dof_right_hip_yaw_03"),
            ObservationType.JointVel("dq_right_knee", xml_name="dof_right_knee_04"),
            ObservationType.JointVel("dq_right_ankle", xml_name="dof_right_ankle_02"),
            ObservationType.JointVel("dq_left_hip_pitch", xml_name="dof_left_hip_pitch_04"),
            ObservationType.JointVel("dq_left_hip_roll", xml_name="dof_left_hip_roll_03"),
            ObservationType.JointVel("dq_left_hip_yaw", xml_name="dof_left_hip_yaw_03"),
            ObservationType.JointVel("dq_left_knee", xml_name="dof_left_knee_04"),
            ObservationType.JointVel("dq_left_ankle", xml_name="dof_left_ankle_02"),
            # ------------- SENSORS -------------
            # ObservationType.SensorAccelerometer("imu_acc", xml_name="imu_acc"),
            # ObservationType.SensorGyro("imu_gyro", xml_name="imu_gyro"),
            # ObservationType.SensorMagnetometer("imu_mag", xml_name="imu_mag"),
            # ObservationType.SensorFramePos("base_link_pos", xml_name="base_link_pos"),
            # ObservationType.SensorFrameQuat("base_link_quat", xml_name="base_link_quat"),
            # ObservationType.SensorFrameLinVel("base_link_vel", xml_name="base_link_vel"),
            # ObservationType.SensorFrameAngVel("base_link_ang_vel", xml_name="base_link_ang_vel"),
            # ObservationType.SensorForce("left_foot_force", xml_name="left_foot_force"),
            # ObservationType.SensorForce("right_foot_force", xml_name="right_foot_force")
        ]

        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action names (motor names).
        """
        # Adapt motor names based on your kbot_v2.xml
        action_spec = [
            "dof_right_shoulder_pitch_03_ctrl",
            "dof_right_shoulder_roll_03_ctrl",
            "dof_right_shoulder_yaw_02_ctrl",
            "dof_right_elbow_02_ctrl",
            "dof_right_wrist_00_ctrl",
            "dof_left_shoulder_pitch_03_ctrl",
            "dof_left_shoulder_roll_03_ctrl",
            "dof_left_shoulder_yaw_02_ctrl",
            "dof_left_elbow_02_ctrl",
            "dof_left_wrist_00_ctrl",
            "dof_right_hip_pitch_04_ctrl",
            "dof_right_hip_roll_03_ctrl",
            "dof_right_hip_yaw_03_ctrl",
            "dof_right_knee_04_ctrl",
            "dof_right_ankle_02_ctrl",
            "dof_left_hip_pitch_04_ctrl",
            "dof_left_hip_roll_03_ctrl",
            "dof_left_hip_yaw_03_ctrl",
            "dof_left_knee_04_ctrl",
            "dof_left_ankle_02_ctrl",
        ]
        return action_spec

    @classmethod
    def get_default_xml_file_path(cls) -> str:
        """
        Returns the default XML file path for the KBotV2 environment.
        """
        return (loco_mujoco.PATH_TO_MODELS / "kbot_v2" / "kbot_v2.xml").as_posix()

    @info_property
    def sites_for_mimic(self) -> List[str]:
        """
        Returns the default sites that are used for mimic. Needs to match the sites added in kbot_v2.xml.
        """
        return [
            "pelvis_mimic",
            "upper_body_mimic",
            "head_mimic",
            "left_shoulder_mimic",
            "left_elbow_mimic",
            "left_wrist_mimic",
            "left_hip_mimic",
            "left_knee_mimic",
            # "left_ankle_mimic",
            "left_foot_mimic",
            "right_shoulder_mimic",
            "right_elbow_mimic",
            "right_wrist_mimic",
            "right_hip_mimic",
            "right_knee_mimic",
            # "right_ankle_mimic",
            "right_foot_mimic",
        ]

    @info_property
    def root_body_name(self) -> str:
        """
        Returns the name of the root body of the robot in the MuJoCo xml.
        """
        # Assuming 'Torso_Side_Right' is the main body connected to the floating base link's frame
        # If 'floating_base_link' itself has the main mass/inertia, use that.
        return "Torso_Side_Right"

    @info_property
    def upper_body_xml_name(self) -> str:
        """
        Returns the name of the upper body specified in the XML file.
        """
        # Corresponds to root_body_name for this robot structure it seems
        return "Torso_Side_Right"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        """
        Returns the name of the free joint of the root specified in the XML file.
        """
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        """
        Returns the healthy range of the root height. This is only used when HeightBasedTerminalStateHandler is used.
        Adjust these values based on your robot's expected operational height.
        """
        return (0.7, 1.5)  # Example range, adjust as needed
