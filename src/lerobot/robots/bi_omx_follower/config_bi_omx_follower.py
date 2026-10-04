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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


# Draccus 무한루프 방지용 순수 데이터클래스 (양쪽 모두 OMX)
@dataclass
class OmxArmConfig:
    port: str = "/dev/ttyACM0"
    disable_torque_on_disconnect: bool = True
    max_relative_target: float | dict[str, float] | None = None
    use_degrees: bool = False
    # cameras 필드 없음 — 카메라는 BiOmxFollowerConfig.cameras에서 최상위로 관리


@RobotConfig.register_subclass("bi_omx_follower")
@dataclass
class BiOmxFollowerConfig(RobotConfig):
    left_arm_config: OmxArmConfig = field(default_factory=lambda: OmxArmConfig(port="/dev/ttyACM0"))
    right_arm_config: OmxArmConfig = field(default_factory=lambda: OmxArmConfig(port="/dev/ttyACM1"))
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
