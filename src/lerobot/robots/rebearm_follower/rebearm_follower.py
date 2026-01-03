#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.lewansol import (
    LewansoulMotorsBus,
    OperatingMode,
)

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_rebearm_follower import REBEARMFollowerConfig

logger = logging.getLogger(__name__)


class REBEARMFollower(Robot):
    """
    Rebearm Follower Arm designed by TheRobotStudio and Hugging Face.
    """

    config_class = REBEARMFollowerConfig
    name = "rebearm_follower"

    def __init__(self, config: REBEARMFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = LewansoulMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "lx16a", norm_mode_body),
                "shoulder_lift": Motor(2, "lx16a", norm_mode_body),
                "elbow_flex": Motor(3, "lx16a", norm_mode_body),
                "wrist_flex": Motor(4, "lx16a", norm_mode_body),
                "wrist_roll": Motor(5, "lx16a", norm_mode_body),
                "gripper": Motor(6, "lx16a", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"\nRunning calibration of {self}")
        print(
            "Just torque off, don't run calibration "
        )
        self.bus.disable_torque()

    def configure(self) -> None:
        print(
            "follower->configure"
        )
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        #print(
        #    "follower->get_observation"
        #)
        obs_dict = {}
        for motor in self.bus.motors:
            obs_each = self.bus.send("Present_Position", motor=motor, value=None)
            #print(f'Motor {motor} - action_each: {action_each}')
            if obs_each is not None:
                obs_dict.update(obs_each) 
            else:
                print(f"Warning: Got None response for motor {motor}")
        
        # Add .pos suffix to all keys
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, int]) -> dict[str, int]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        #print(
        #    "follower->send_action"
        #)
        
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        #print('GoalPos:', goal_pos)
        for  motor, val in goal_pos.items():
            self.bus.send("Goal_Position", motor=motor, value = val)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
    
    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
