"""Meta-tests of the robot conformance kit.

The loopback adapter, an ideal robot in memory, passes every check. Broken
variants of it, each with one bug an adapter could have, fail the check
meant for that bug, with a message that names the likely cause.
"""

import math
import time
from collections.abc import Callable
from dataclasses import replace
from importlib.resources import as_file, files
from typing import TypeVar

import numpy as np
import pytest

from embodiverse.adapters.loopback import LoopbackRobot
from embodiverse.contracts.conventions import xyzw_to_wxyz
from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.robot import Capabilities, Robot
from embodiverse.contracts.types import JointCommand, Mode, RobotState
from embodiverse.tools.conformance.robot import (
    RobotConformance,
    SimulationProbe,
    check_description,
    check_imu,
    check_joint_order,
    check_lifecycle,
    check_modes,
    check_rejects_malformed_commands,
    check_state,
    check_timestamps,
    check_timing,
    check_units,
)


def load_testbot() -> RobotDescription:
    with as_file(files("embodiverse.robots.testbot") / "description.yaml") as path:
        return load_description(path)


def with_imu_mount(
    description: RobotDescription, rpy: tuple[float, float, float]
) -> RobotDescription:
    """Return `description` with its IMU mounted at `rpy`."""
    data = description.model_dump()
    data["imu"]["mount_rpy"] = rpy
    return RobotDescription.model_validate(data)


TESTBOT = load_testbot()

# An IMU turned about two axes, so that a mistake with the mount changes
# the readings at every pose.
MOUNTED = with_imu_mount(TESTBOT, (math.pi / 2, 0.0, math.pi / 2))


# Broken variants, each with one bug an adapter could have.


class SwappedJoints(LoopbackRobot):
    """Reports the first two joints swapped, as a wrong motor map would."""

    def read(self) -> RobotState:
        state = super().read()
        order = [1, 0, *range(2, self.description.num_joints)]
        return replace(
            state,
            position=state.position[order],
            velocity=state.velocity[order],
            effort=state.effort[order],
        )


class PositionsInDegrees(LoopbackRobot):
    """Reports joint positions in degrees."""

    def read(self) -> RobotState:
        state = super().read()
        return replace(state, position=np.degrees(state.position))


class ImuInWxyzOrder(LoopbackRobot):
    """Reports the IMU orientation in MuJoCo's (w, x, y, z) order."""

    def read(self) -> RobotState:
        state = super().read()
        imu = replace(state.imu, orientation=xyzw_to_wxyz(state.imu.orientation))
        return replace(state, imu=imu)


class ImuMountIgnored(LoopbackRobot):
    """Reports the base's orientation as the IMU's, ignoring the mount."""

    def read(self) -> RobotState:
        state = super().read()
        imu = replace(state.imu, orientation=self.model.base_orientation)
        return replace(state, imu=imu)


class DampThroughSetMode(LoopbackRobot):
    """Lets a program damp the robot with set_mode(DAMP)."""

    def set_mode(self, mode: Mode) -> None:
        if mode is Mode.DAMP:
            self._check_connected()
            with self._mode_lock:
                self._control = (Mode.DAMP, self._damp)
            return
        super().set_mode(mode)


class StopWithoutDamp(LoopbackRobot):
    """Drops to zero torque (PASSIVE) on operator_stop instead of damping."""

    def operator_stop(self) -> None:
        with self._mode_lock:
            if self._connected:
                self._control = (Mode.PASSIVE, self._passive)


class AcceptsNaN(LoopbackRobot):
    """Checks commands, but lets non-finite values through."""

    def write(self, command: JointCommand) -> None:
        finite = {
            name: np.nan_to_num(getattr(command, name), nan=0.0, posinf=0.0, neginf=0.0)
            for name in ("position", "velocity", "effort", "kp", "kd")
        }
        super().write(replace(command, **finite))


class FrozenClock(LoopbackRobot):
    """Stamps every state with the time it was connected."""

    def read(self) -> RobotState:
        return replace(super().read(), stamp_ns=0)


class ReadsBeforeConnect(LoopbackRobot):
    """Returns a state from read before connect."""

    def read(self) -> RobotState:
        return self._latest()


class ChangingCapabilities(LoopbackRobot):
    """Returns a different max_rate_hz on every call."""

    calls = 0

    def capabilities(self) -> Capabilities:
        self.calls += 1
        return replace(super().capabilities(), max_rate_hz=100.0 * self.calls)


class TemperatureMissing(LoopbackRobot):
    """Declares motor temperatures but never reports them."""

    def capabilities(self) -> Capabilities:
        return replace(super().capabilities(), temperature=True)


class SlowRead(LoopbackRobot):
    """Waits for new data in read."""

    def read(self) -> RobotState:
        time.sleep(0.0005)
        return super().read()


RobotT = TypeVar("RobotT", bound=LoopbackRobot)


def connected(
    robot_type: type[RobotT], description: RobotDescription = TESTBOT
) -> RobotT:
    robot = robot_type(description)
    robot.connect()
    return robot


# The kit on the ideal robot.


# The kit's default timing budget of 1 ms is meant for a robot's own
# computer. These tests also run on shared development machines, where the
# scheduler can pause a thread for milliseconds; 5 ms still shows that read
# and write never wait for the model.
SHARED_MACHINE_BUDGET_S = 0.005


class TestLoopbackRobot(RobotConformance):
    """The ideal robot passes every check."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: LoopbackRobot(TESTBOT)

    @pytest.fixture
    def probe(self, robot: Robot) -> SimulationProbe | None:
        assert isinstance(robot, LoopbackRobot)
        return robot


def test_checks_follow_the_declared_capabilities() -> None:
    capabilities = Capabilities(effort_command=False, temperature=True, max_rate_hz=50)
    robot = LoopbackRobot(TESTBOT, capabilities)
    robot.connect()
    check_description(robot)
    check_state(robot)
    check_rejects_malformed_commands(robot)


def test_the_imu_check_applies_the_mount() -> None:
    robot = connected(LoopbackRobot, MOUNTED)
    check_imu(robot, robot)


def test_the_checks_leave_the_robot_in_passive() -> None:
    robot = connected(LoopbackRobot)
    for check in (check_joint_order, check_units, check_imu):
        check(robot, robot)
        assert robot.read().mode is Mode.PASSIVE
    for check in (check_timing, check_modes):
        check(robot)
        assert robot.read().mode is Mode.PASSIVE


# Each broken variant fails the check meant for its bug.


def test_swapped_joints_fail_the_joint_order_check() -> None:
    robot = connected(SwappedJoints)
    match = (
        r"^joint 'left_hip_pitch' did not move when commanded: .*; joint"
        r" 'left_knee' moved -0\.\d+ rad instead: joint order may be wrong"
    )
    with pytest.raises(AssertionError, match=match):
        check_joint_order(robot, robot)


def test_positions_in_degrees_fail_the_units_check() -> None:
    robot = connected(PositionsInDegrees)
    check_joint_order(robot, robot)
    match = r"^joint 'left_hip_pitch' settled at .*: positions look like degrees"
    with pytest.raises(AssertionError, match=match):
        check_units(robot, robot)


def test_an_imu_in_wxyz_order_fails_the_imu_check() -> None:
    robot = connected(ImuInWxyzOrder)
    match = (
        r"^IMU orientation with the base level: expected \(0\.000, 0\.000,"
        r" 0\.000, 1\.000\), .* observed \(1\.000, 0\.000, 0\.000, 0\.000\),"
        r" 180\.0° apart .*: it is in \(w, x, y, z\) order"
    )
    with pytest.raises(AssertionError, match=match):
        check_imu(robot, robot)


def test_an_ignored_imu_mount_fails_the_imu_check() -> None:
    robot = connected(ImuMountIgnored, MOUNTED)
    match = (
        r"^IMU orientation with the base level: .*: it is the orientation of the"
        r" base, not of the IMU frame: the IMU's mount_rpy \(1\.571, 0\.000,"
        r" 1\.571\) is not applied$"
    )
    with pytest.raises(AssertionError, match=match):
        check_imu(robot, robot)


def test_damp_through_set_mode_fails_the_modes_check() -> None:
    robot = connected(DampThroughSetMode)
    match = (
        r"^set_mode\(DAMP\) in PASSIVE returned, expected it to raise ValueError:"
        r" DAMP is entered only through operator_stop"
    )
    with pytest.raises(AssertionError, match=match):
        check_modes(robot)


def test_a_stop_without_damp_fails_the_modes_check() -> None:
    robot = connected(StopWithoutDamp)
    match = (
        r"^after operator_stop\(\) in PASSIVE, read\(\)\.mode is PASSIVE, expected"
        r" DAMP: operator_stop must enter DAMP from every mode$"
    )
    with pytest.raises(AssertionError, match=match):
        check_modes(robot)


def test_accepting_nan_fails_the_command_check() -> None:
    robot = connected(AcceptsNaN)
    match = (
        r"^write\(\) accepted a command with NaN in position for joint"
        r" 'left_hip_pitch', expected ValueError: write must call check_command"
    )
    with pytest.raises(AssertionError, match=match):
        check_rejects_malformed_commands(robot)


def test_a_frozen_clock_fails_the_timestamp_check() -> None:
    robot = connected(FrozenClock)
    match = (
        r"^stamp_ns increased by 0 ns after advance\(0\.002\), expected 2000000 ns"
        r" within 1 µs"
    )
    with pytest.raises(AssertionError, match=match):
        check_timestamps(robot)


def test_reading_before_connect_fails_the_lifecycle_check() -> None:
    match = (
        r"^read\(\) before connect\(\) returned, expected it to raise RuntimeError:"
        r" a robot must refuse to be used before it is connected$"
    )
    with pytest.raises(AssertionError, match=match):
        check_lifecycle(lambda: ReadsBeforeConnect(TESTBOT))


def test_changing_capabilities_fail_the_description_check() -> None:
    robot = connected(ChangingCapabilities)
    match = r"^capabilities\(\) returned .*, expected the same value on every call"
    with pytest.raises(AssertionError, match=match):
        check_description(robot)


def test_missing_temperatures_fail_the_state_check() -> None:
    robot = connected(TemperatureMissing)
    match = r"^temperature is None, but capabilities\(\)\.temperature is True"
    with pytest.raises(AssertionError, match=match):
        check_state(robot)


def test_a_slow_read_fails_the_timing_check() -> None:
    robot = connected(SlowRead)
    match = r"^read\(\) took \d+\.\d+ ms at the 95th percentile of 50 calls"
    with pytest.raises(AssertionError, match=match):
        check_timing(robot, budget_s=0.0002, calls=50)
