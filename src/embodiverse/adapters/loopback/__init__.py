"""The loopback adapter: a stand-in robot that runs any description.

`LoopbackRobot` runs in lockstep, `RealtimeLoopbackRobot` in real time with
the shape of a hardware adapter. Neither needs a model file, a simulator or
a robot. It is the template for hardware adapters (docs/adding-a-robot.md).
"""

from embodiverse.adapters.loopback.model import MasslessJoints
from embodiverse.adapters.loopback.robot import LoopbackRobot, RealtimeLoopbackRobot
from embodiverse.adapters.loopback.transport import LoopbackTransport

__all__ = [
    "LoopbackRobot",
    "LoopbackTransport",
    "MasslessJoints",
    "RealtimeLoopbackRobot",
]
