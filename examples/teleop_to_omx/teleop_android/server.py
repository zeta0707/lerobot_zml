# REFS: https://github.com/SpesRobotics/teleop/blob/main/teleop/__init__.py

import json
import logging
import socket
from pathlib import Path
from typing import Callable, List, Optional, TypedDict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

#:
# Types for data sent from the Android app


class Position(TypedDict):
    x: float
    y: float
    z: float


class Orientation(TypedDict):
    x: float
    y: float
    z: float
    w: float


class Pose(TypedDict):
    position: Position
    orientation: Orientation


class Control(TypedDict):
    x: float
    y: float
    isFineControl: bool
    isActive: bool


#:


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        return f"Error: {e}"


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


class TeleopServer:
    """
    Simple WebSocket server for receiving position and orientation data.

    Args:
        host: The host IP address. Defaults to "0.0.0.0".
        port: The port number. Defaults to 4443.
        certs_dir: Path to directory containing SSL certificate files (server.crt and server.key).
            If None, defaults to looking for certs relative to the package location.
    """

    def __init__(
        self, host: str = "0.0.0.0", port: int = 4443, certs_dir: Optional[str] = None
    ):
        self.__logger = logging.getLogger("teleop")
        self.__logger.setLevel(logging.INFO)
        self.__logger.addHandler(logging.StreamHandler())

        self.__host = host
        self.__port = port
        self.__certs_dir = certs_dir
        self.__pose_callbacks: List[Callable[[Pose], None]] = []
        self.__control_callbacks: List[Callable[[Control], None]] = []

        self.__app = FastAPI()
        self.__manager = ConnectionManager()

        # Configure logging
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

        self.__setup_routes()

    def subscribe_pose(self, callback: Callable[[Pose], None]) -> None:
        """
        Subscribe to receive updates when pose data is received.

        Args:
            callback: A function that will be called with pose data.
                The callback receives a PoseData dict with 'position' and 'orientation' keys.
        """
        self.__pose_callbacks.append(callback)

    def subscribe_control(self, callback: Callable[[Control], None]) -> None:
        """
        Subscribe to receive updates when control data is received.

        Args:
            callback: A function that will be called with control data.
                The callback receives a ControlData dict with 'x', 'y', 'isFineControl', and 'isActive' keys.
        """
        self.__control_callbacks.append(callback)

    def __notify_pose_subscribers(self, data: Pose):
        """Notify all pose subscribers with the received data."""
        for callback in self.__pose_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.__logger.error(f"Error in pose callback: {e}")

    def __notify_control_subscribers(self, data: Control):
        """Notify all control subscribers with the received data."""
        for callback in self.__control_callbacks:
            try:
                callback(data)
            except Exception as e:
                self.__logger.error(f"Error in control callback: {e}")

    def __setup_routes(self):
        """Set up FastAPI routes."""

        @self.__app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.__manager.connect(websocket)
            self.__logger.info("Client connected")

            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)

                    if message.get("type") == "pose":
                        pose_data: Pose = message.get("data", {})
                        self.__logger.debug(f"Received pose data: {pose_data}")
                        self.__notify_pose_subscribers(pose_data)
                    elif message.get("type") == "control":
                        control_data: Control = message.get("data", {})
                        self.__logger.debug(f"Received control data: {control_data}")
                        self.__notify_control_subscribers(control_data)

            except WebSocketDisconnect:
                self.__manager.disconnect(websocket)
                self.__logger.info("Client disconnected")

    def run(
        self, ssl_certfile: Optional[str] = None, ssl_keyfile: Optional[str] = None
    ) -> None:
        """
        Run the WebSocket server. This method is blocking.

        Args:
            ssl_certfile: Path to SSL certificate file. If None, uses certs_dir/server.crt.
            ssl_keyfile: Path to SSL key file. If None, uses certs_dir/server.key.
        """
        # Use default certificate paths if not provided
        if ssl_certfile is None or ssl_keyfile is None:
            if self.__certs_dir is not None:
                cert_dir = Path(self.__certs_dir)
            else:
                cert_dir = Path(__file__).parent.parent / "certs"
            ssl_certfile = str(cert_dir / "server.crt")
            ssl_keyfile = str(cert_dir / "server.key")

            if not Path(ssl_certfile).exists() or not Path(ssl_keyfile).exists():
                raise FileNotFoundError(
                    f"SSL certificate files not found at {cert_dir}. "
                    "Please generate them using the instructions in the README."
                )

        self.__logger.info(f"Server started at {self.__host}:{self.__port}")
        self.__logger.info(
            f"WebSocket endpoint available at wss://{get_local_ip()}:{self.__port}/ws"
        )

        uvicorn.run(
            self.__app,
            host=self.__host,
            port=self.__port,
            log_level="warning",
            ssl_certfile=ssl_certfile,
            ssl_keyfile=ssl_keyfile,
        )
