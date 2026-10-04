# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
# See this link for LX-16A Command
# https://github.com/madhephaestus/lx16a-servo/blob/master/lx-16a%20LewanSoul%20Bus%20Servo%20Communication%20Protocol.pdf

import logging
from copy import deepcopy
from enum import Enum
from pprint import pformat

from lerobot.motors.encoding_utils import decode_sign_magnitude, encode_sign_magnitude
from lerobot.utils.decorators import check_if_not_connected

from ..motors_bus import Motor, MotorCalibration, MotorsBus, NameOrID, Value, get_address

from .tables import (
    AVAILABLE_BAUDRATES,
    MODEL_BAUDRATE_TABLE,
    MODEL_CONTROL_TABLE,
    MODEL_ENCODING_TABLE,
    MODEL_NUMBER_TABLE,
    MODEL_RESOLUTION,
)

DEFAULT_PROTOCOL_VERSION = 0
DEFAULT_BAUDRATE = 115_200
DEFAULT_TIMEOUT_MS = 1000

NORMALIZED_DATA = ["Goal_Position", "Present_Position"]

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    # position servo mode
    POSITION = 0
    # The motor is in constant speed mode, which is controlled by parameter 0x2e, and the highest bit 15 is
    # the direction bit
    VELOCITY = 1
    # PWM open-loop speed regulation mode, with parameter 0x2c running time parameter control, bit11 as
    # direction bit
    PWM = 2
    # In step servo mode, the number of step progress is represented by parameter 0x2a, and the highest bit 15
    # is the direction bit
    STEP = 3


class DriveMode(Enum):
    NON_INVERTED = 0
    INVERTED = 1


class TorqueMode(Enum):
    ENABLED = 1
    DISABLED = 0


def _split_into_byte_chunks(value: int, length: int) -> list[int]:
    import lx16aservo_sdk as lx16a

    if length == 1:
        data = [value]
    elif length == 2:
        data = [lx16a.LX16A_LOBYTE(value), lx16a.LX16A_HIBYTE(value)]
    elif length == 4:
        data = [
            lx16a.LX16A_LOBYTE(lx16a.LX16A_LOWORD(value)),
            lx16a.LX16A_HIBYTE(lx16a.LX16A_LOWORD(value)),
            lx16a.LX16A_LOBYTE(lx16a.LX16A_HIWORD(value)),
            lx16a.LX16A_HIBYTE(lx16a.LX16A_HIWORD(value)),
        ]
    return data



class LewansoulMotorsBus(MotorsBus):
    """
    The Lewansoul implementation for a MotorsBus. It relies on the python Lewansoul sdk to communicate with
    the motors. For more info, see the Lewansoul SDK Documentation:
     """
    
    apply_drive_mode = True
    available_baudrates = deepcopy(AVAILABLE_BAUDRATES)
    default_baudrate = DEFAULT_BAUDRATE
    default_timeout = DEFAULT_TIMEOUT_MS
    model_baudrate_table = deepcopy(MODEL_BAUDRATE_TABLE)
    model_ctrl_table = deepcopy(MODEL_CONTROL_TABLE)
    model_encoding_table = deepcopy(MODEL_ENCODING_TABLE)
    model_number_table = deepcopy(MODEL_NUMBER_TABLE)
    model_resolution_table = deepcopy(MODEL_RESOLUTION)
    normalized_data = deepcopy(NORMALIZED_DATA)

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = DEFAULT_PROTOCOL_VERSION,
    ):
        super().__init__(port, motors, calibration)
        self.protocol_version = protocol_version
        import lx16aservo_sdk as lx16a

        self.port_handler = lx16a.PortHandler(self.port)
        self.packet_handler = lx16a.PacketHandler(protocol_version)
        self.sync_reader = lx16a.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
        self.sync_writer = lx16a.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)
        self._comm_success = lx16a.COMM_SUCCESS
        self._no_error = 0x00

    def _assert_protocol_is_compatible(self, instruction_name: str) -> None:
        if instruction_name == "sync_read" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Sync Read' is not available with Feetech motors using Protocol 1. Use 'Read' sequentially instead."
            )
        if instruction_name == "broadcast_ping" and self.protocol_version == 1:
            raise NotImplementedError(
                "'Broadcast Ping' is not available with Feetech motors using Protocol 1. Use 'Ping' sequentially instead."
            )

    def _assert_same_firmware(self) -> None:
        pass

    def _handshake(self) -> None:
        pass

    def _find_single_motor(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        self._find_single_motor_p0(motor, initial_baudrate)

    def _find_single_motor_p0(self, motor: str, initial_baudrate: int | None = None) -> tuple[int, int]:
        model = self.motors[motor].model
        search_baudrates = (
            [initial_baudrate] if initial_baudrate is not None else self.model_baudrate_table[model]
        )
        expected_model_nb = self.model_number_table[model]

        for baudrate in search_baudrates:
            #self.set_baudrate(baudrate)
            id_model = self.broadcast_ping()
            if id_model:
                found_id, found_model = next(iter(id_model.items()))
                print(found_id, found_model)
                if found_model != expected_model_nb:
                    raise RuntimeError(
                        f"Found one motor on {baudrate=} with id={found_id} but it has a "
                        f"model number '{found_model}' different than the one expected: '{expected_model_nb}'. "
                        f"Make sure you are connected only connected to the '{motor}' motor (model '{model}')."
                    )
                print(baudrate, found_id)
                return baudrate, found_id

        raise RuntimeError(f"Motor '{motor}' (model '{model}') was not found. Make sure it is connected.")


    def configure_motors(self) -> None:
        #nothing to configure, use default
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    def read_calibration(self) -> dict[str, MotorCalibration]:
        offsets, mins, maxes = {}, {}, {}
        for motor in self.motors:
            mins[motor] = self.read("Min_Position_Limit", motor, normalize=False)
            maxes[motor] = self.read("Max_Position_Limit", motor, normalize=False)
            offsets[motor] = (
                self.read("Homing_Offset", motor, normalize=False) if self.protocol_version == 0 else 0
            )

        calibration = {}
        for motor, m in self.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=offsets[motor],
                range_min=mins[motor],
                range_max=maxes[motor],
            )

        return calibration

    def write_calibration(self, calibration_dict: dict[str, MotorCalibration]) -> None:
        pass

    def _get_half_turn_homings(self, positions: dict[NameOrID, Value]) -> dict[NameOrID, Value]:
        """
        On Feetech Motors:
        Present_Position = Actual_Position - Homing_Offset
        """
        half_turn_homings = {}
        for motor, pos in positions.items():
            model = self._get_motor_model(motor)
            max_res = self.model_resolution_table[model] - 1
            half_turn_homings[motor] = pos - int(max_res / 2)

        return half_turn_homings

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        ids = [1,2,3,4,5,6]
        for _id in ids:
            txpacket= [_id, 4, 31, 0]
            self.packet_handler.txPacket(self.port_handler, txpacket)
        
    def _disable_torque(self, motor_id: int, model: str, num_retry: int = 0) -> None:
        txpacket= [motor_id, 4, 31, 0]
        self.packet_handler.txPacket(self.port_handler, txpacket)

    def enable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        ids = [1,2,3,4,5,6]
        for _id in ids:
            txpacket= [_id, 4, 31,1]
            self.packet_handler.txPacket(self.port_handler, txpacket)

    def _encode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = encode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _decode_sign(self, data_name: str, ids_values: dict[int, int]) -> dict[int, int]:
        for id_ in ids_values:
            model = self._id_to_model(id_)
            encoding_table = self.model_encoding_table.get(model)
            if encoding_table and data_name in encoding_table:
                sign_bit = encoding_table[data_name]
                ids_values[id_] = decode_sign_magnitude(ids_values[id_], sign_bit)

        return ids_values

    def _split_into_byte_chunks(self, value: int, length: int) -> list[int]:
        return _split_into_byte_chunks(value, length)

    def _broadcast_ping(self) -> tuple[dict[int, int], int]:
        import lx16aservo_sdk as lx16a

        data_list = {}
        #ping result expected length, [85, 85, 1, 4, 14, 1, 235]
        status_length = 7

        rx_length = 0
        wait_length = status_length * lx16a.MAX_ID

        tx_time_per_byte = (1000.0 / self.port_handler.getBaudRate()) * 10.0

        txpacket= [254, 3, 14]

        result = self.packet_handler.txPacket(self.port_handler, txpacket)
        if result != lx16a.COMM_SUCCESS:
            self.port_handler.is_using = False
            return data_list, result

        # set rx timeout
        self.port_handler.setPacketTimeoutMillis((wait_length * tx_time_per_byte) + (3.0 * lx16a.MAX_ID) + 16.0)

        rxpacket = []
        while not self.port_handler.isPacketTimeout() and rx_length < wait_length:
            rxpacket += self.port_handler.readPort(wait_length - rx_length)
            rx_length = len(rxpacket)

        print("BROAD_PING", rxpacket)
        self.port_handler.is_using = False

        if rx_length == 0:
            return data_list, lx16a.COMM_RX_TIMEOUT

        while True:
            if rx_length < status_length:
                return data_list, lx16a.COMM_RX_CORRUPT

            # find packet header
            for idx in range(0, (rx_length - 1)):
                if (rxpacket[idx] == 0x55) and (rxpacket[idx + 1] == 0x55):
                    break

            if idx == 0:  # found at the beginning of the packet
                # calculate checksum
                checksum = 0
                for idx in range(2, status_length - 1):  # except header & checksum
                    checksum += rxpacket[idx]

                checksum = ~checksum & 0xFF
                if rxpacket[status_length - 1] == checksum:
                    result = lx16a.COMM_SUCCESS
                    data_list[rxpacket[lx16a.PKT_ID]] = lx16a.COMM_SUCCESS

                    del rxpacket[0:status_length]
                    rx_length = rx_length - status_length

                    if rx_length == 0:
                        return data_list, result
                else:
                    result = lx16a.COMM_RX_CORRUPT
                    # remove header (0xFF 0xFF)
                    del rxpacket[0:2]
                    rx_length = rx_length - 2
            else:
                # remove unnecessary packets
                del rxpacket[0:idx]
                rx_length = rx_length - idx

    def broadcast_ping(self, num_retry: int = 0, raise_on_error: bool = False) -> dict[int, int] | None:
        self._assert_protocol_is_compatible("broadcast_ping")
        for n_try in range(1 + num_retry):
            ids_status, comm = self._broadcast_ping()
            if self._is_comm_success(comm):
                break
            logger.debug(f"Broadcast ping failed on port '{self.port}' ({n_try=})")
            logger.debug(self.packet_handler.getTxRxResult(comm))

        if not self._is_comm_success(comm):
            if raise_on_error:
                raise ConnectionError(self.packet_handler.getTxRxResult(comm))
            return

        ids_errors = {id_: status for id_, status in ids_status.items() if self._is_error(status)}
        if ids_errors:
            display_dict = {id_: self.packet_handler.getRxPacketError(err) for id_, err in ids_errors.items()}
            logger.error(f"Some motors found returned an error status:\n{pformat(display_dict, indent=4)}")

        return self._read_model_number(list(ids_status), raise_on_error)
    
    def _read_model_number(self, motor_ids: list[int], raise_on_error: bool = False) -> dict[int, int]:
        import lx16aservo_sdk as lx16a

        model_numbers = {}
        for id_ in motor_ids:
            model_numbers[id_] = lx16a.LX16A_MODEL_NUM

        return model_numbers

    @check_if_not_connected
    def send(self, data_name: str, motor: str, value: int | None = None, *, num_retry: int = 0) -> dict[str, float | int]:
        """Read or write a single LX-16A motor using the servo's native position commands.

        Supports "Present_Position" (read, returns the angle as float) and "Goal_Position" (write).
        """
        id_ = self.motors[motor].id
        if data_name == "Present_Position":
            angle, comm, error = self.packet_handler.get_action(self.port_handler, id_)
            return {self._id_to_name(id_): float(angle)}
        elif data_name == "Goal_Position":
            value = int(value)
            self.packet_handler.set_action(self.port_handler, id_, value)
            return {self._id_to_name(id_): value}
        raise ValueError(f"Unsupported data_name for LX-16A send(): {data_name!r}")
