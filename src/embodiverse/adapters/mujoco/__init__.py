"""The MuJoCo adapter: any robot with an MJCF model, simulated behind the robot port.

`MujocoRobot` runs in lockstep, `RealtimeMujocoRobot` in real time.

Install it with the `sim` extra (`embodiverse[sim]`). docs/adding-a-robot.md
says what a model needs.
"""

from embodiverse.adapters.mujoco.robot import MujocoRobot, RealtimeMujocoRobot

__all__ = ["MujocoRobot", "RealtimeMujocoRobot"]
