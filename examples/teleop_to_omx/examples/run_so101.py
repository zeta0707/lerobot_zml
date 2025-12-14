import time

import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from teleop_android import (
    AndroidPhone,
    GripperToJoint,
    MapPhoneActionToRobotAction,
    WristJoints,
)

#:


# WARN: Set this to the path of your URDF file for the SO101, you must edit it to
# add "lower_arm_frame_link", see README.
PATH_URDF = "../phone_to_so100/urdf"

FPS = 30

#:

# Initialize the robot
robot_config = SO101FollowerConfig(
    port="/dev/tty.usbmodem5A460829821", id="arm_follower_0", use_degrees=True
)
robot = SO101Follower(robot_config)

# Initialize the telooperator
teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
teleop_config.camera_offset = np.array([0.0, -0.01, 0.05])
teleop_device = AndroidPhone(config=teleop_config)

kinematics_solver = RobotKinematics(
    urdf_path=PATH_URDF,
    target_frame_name="lower_arm_frame_link",
    joint_names=list(robot.bus.motors.keys()),
)

# Build pipeline to convert phone action to ee pose action to joint action
phone_to_robot_joints_processor = RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
](
    steps=[
        MapPhoneActionToRobotAction(),
        EEReferenceAndDelta(
            kinematics=kinematics_solver,
            end_effector_step_sizes={"x": 0.5, "y": 0.5, "z": 0.5},
            motor_names=list(robot.bus.motors.keys()),
            use_latched_reference=True,
        ),
        EEBoundsAndSafety(
            end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
            max_ee_step_m=0.10,
        ),
        GripperToJoint(clip_min=-5.0, clip_max=95.0),
        InverseKinematicsEEToJoints(
            kinematics=kinematics_solver,
            motor_names=list(robot.bus.motors.keys()),
            initial_guess_current_joints=True,
        ),
        WristJoints(
            motor_names=list(robot.bus.motors.keys()),
            kinematics=kinematics_solver,
        ),
    ],
    to_transition=robot_action_observation_to_transition,
    to_output=transition_to_robot_action,
)

# Connect to the robot and teleoperator
robot.connect()
teleop_device.connect()

# Init rerun viewer
init_rerun(session_name="phone_so101_teleop")

if not robot.is_connected or not teleop_device.is_connected:
    raise ValueError("Robot or teleop is not connected!")

print("Starting teleop loop. Move your phone to teleoperate the robot...")
while True:
    t0 = time.perf_counter()

    # Get robot observation
    robot_obs = robot.get_observation()

    # Get teleop action
    phone_obs = teleop_device.get_action()

    # Phone -> EE pose -> Joints transition
    joint_action = phone_to_robot_joints_processor((phone_obs, robot_obs))

    # Send action to robot
    _ = robot.send_action(joint_action)

    # Visualize
    log_rerun_data(observation=phone_obs, action=joint_action)

    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))
