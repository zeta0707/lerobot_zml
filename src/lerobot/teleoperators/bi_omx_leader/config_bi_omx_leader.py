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

from ..config import TeleoperatorConfig


# Draccus 무한루프 방지용 순수 데이터클래스 (양쪽 모두 OMX)
@dataclass
class OmxLeaderArmConfig:
    """Plain (non-registered) per-arm configuration for OMX leader, used inside
    BiOmxLeaderConfig. Intentionally NOT a TeleoperatorConfig subclass / NOT registered,
    to avoid draccus recursively expanding every registered TeleoperatorConfig (including
    BiOmxLeaderConfig itself) when building the CLI schema.
    """

    port: str
    gripper_open_pos: float = 60.0


@TeleoperatorConfig.register_subclass("bi_omx_leader")
@dataclass
class BiOmxLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi OMX Leader teleoperators."""

    left_arm_config: OmxLeaderArmConfig = field(
        default_factory=lambda: OmxLeaderArmConfig(port="/dev/ttyACM2")
    )
    right_arm_config: OmxLeaderArmConfig = field(
        default_factory=lambda: OmxLeaderArmConfig(port="/dev/ttyACM3")
    )
