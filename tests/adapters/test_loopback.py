"""Tests of the loopback adapter.

In lockstep, every built-in description passes the whole conformance kit
on the loopback, with the adapter as its own probe; in real time, testbot
passes every check that needs no probe. The other tests check how the
massless joints behave in each mode, that the real-time transport follows
the clock, and that `embodiverse run` works on a draft description with no
model file.
"""

import re
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np
import pytest

from embodiverse.adapters.loopback import LoopbackRobot, RealtimeLoopbackRobot
from embodiverse.app.cli import main
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.registry import robot_description
from embodiverse.contracts.robot import Robot, Steppable
from embodiverse.contracts.types import JointCommand, Mode
from embodiverse.tools.conformance.robot import RobotConformance, SimulationProbe

TESTBOT = robot_description("testbot")
DEFAULT_POSE = np.array([joint.default_position for joint in TESTBOT.joints])
ZEROS = np.zeros(TESTBOT.num_joints)
PERIOD_S = 1.0 / TESTBOT.rates.physics_hz

# The kit's default timing budget of 1 ms is meant for a robot's own
# computer. These tests also run on shared development machines, where the
# scheduler can pause a thread for milliseconds; 5 ms still shows that read
# and write never wait for the model.
SHARED_MACHINE_BUDGET_S = 0.005


def command(description: RobotDescription, position: np.ndarray) -> JointCommand:
    zeros = np.zeros(description.num_joints)
    return JointCommand(
        position=position,
        velocity=zeros,
        effort=zeros,
        kp=np.array([joint.gains.kp for joint in description.joints]),
        kd=np.array([joint.gains.kd for joint in description.joints]),
    )


# The kit.


class TestLoopback(RobotConformance):
    """In lockstep, every built-in description passes the whole kit."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S
    # A massless joint approaches its target with the time constant
    # (friction + kd) / kp, which is 0.28 s for the K1's head (kp 4, kd 1):
    # too slow for the kit's default of 0.3 s.
    joint_settle_s = 0.5

    @pytest.fixture(params=["testbot", "g1_29dof", "pm01", "k1_22dof"])
    def make_robot(self, request: pytest.FixtureRequest) -> Callable[[], Robot]:
        description = robot_description(request.param)
        return lambda: LoopbackRobot(description)

    @pytest.fixture
    def probe(self, robot: Robot) -> SimulationProbe | None:
        assert isinstance(robot, LoopbackRobot)
        return robot


class TestLoopbackRealtime(RobotConformance):
    """In real time, testbot passes every check that needs no probe."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: RealtimeLoopbackRobot(TESTBOT)


def test_only_lockstep_is_steppable() -> None:
    lockstep = LoopbackRobot(TESTBOT)
    assert isinstance(lockstep, Robot)
    assert isinstance(lockstep, Steppable)
    assert isinstance(lockstep, SimulationProbe)
    real_time = RealtimeLoopbackRobot(TESTBOT)
    assert isinstance(real_time, Robot)
    assert not isinstance(real_time, Steppable)
    assert not hasattr(real_time, "advance")


# Behaviour in lockstep.


@pytest.fixture
def robot() -> Iterator[LoopbackRobot]:
    robot = LoopbackRobot(TESTBOT)
    robot.connect()
    yield robot
    robot.close()


def test_starts_at_the_default_pose_at_time_zero(robot: LoopbackRobot) -> None:
    state = robot.read()
    assert state.stamp_ns == 0
    assert state.mode is Mode.PASSIVE
    np.testing.assert_array_equal(state.position, DEFAULT_POSE)
    np.testing.assert_array_equal(state.velocity, ZEROS)
    assert state.temperature is None
    np.testing.assert_allclose(state.imu.linear_acceleration, [0.0, 0.0, 9.80665])


def test_joints_follow_a_command_and_settle_on_it(robot: LoopbackRobot) -> None:
    robot.set_mode(Mode.HOLD)
    robot.set_mode(Mode.POLICY)
    target = DEFAULT_POSE + 0.3
    robot.write(command(TESTBOT, target))
    robot.advance(PERIOD_S)
    early = robot.read()
    assert np.all(early.velocity > 0.0)
    assert np.all(early.position > DEFAULT_POSE)
    assert np.all(early.position < target)
    robot.advance(0.5)
    settled = robot.read()
    np.testing.assert_allclose(settled.position, target, atol=1e-3)
    assert np.all(np.abs(settled.velocity) < 1e-3)
    assert settled.stamp_ns == round(0.502 * 1e9)


def test_passive_and_damp_leave_the_joints_where_they_are(robot: LoopbackRobot) -> None:
    moved = DEFAULT_POSE + 0.2
    robot.set_joint_positions(moved)
    robot.advance(0.5)
    # Without stiffness or feed-forward torque, nothing moves the joints;
    # the implicit step only rounds the position by a few ulps.
    np.testing.assert_allclose(robot.read().position, moved, rtol=0, atol=1e-9)
    robot.operator_stop()
    robot.advance(0.5)
    state = robot.read()
    assert state.mode is Mode.DAMP
    np.testing.assert_allclose(state.position, moved, rtol=0, atol=1e-9)
    assert np.all(np.abs(state.velocity) < 1e-9)


def test_hold_pulls_the_joints_back_to_the_latched_positions(
    robot: LoopbackRobot,
) -> None:
    robot.set_mode(Mode.HOLD)
    robot.set_joint_positions(DEFAULT_POSE + 0.2)
    robot.advance(0.5)
    np.testing.assert_allclose(robot.read().position, DEFAULT_POSE, atol=1e-3)


def test_advance_takes_whole_physics_steps_only(robot: LoopbackRobot) -> None:
    for seconds in (0.0, -0.002, 0.001, 0.003, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive whole number of physics steps"):
            robot.advance(seconds)
    assert robot.read().stamp_ns == 0


def test_the_probe_refuses_a_robot_that_is_not_connected() -> None:
    robot = LoopbackRobot(TESTBOT)
    calls: tuple[Callable[[], object], ...] = (
        lambda: robot.advance(PERIOD_S),
        lambda: robot.set_base_pose((0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        robot.release_base,
        lambda: robot.set_joint_positions(DEFAULT_POSE),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="is not connected"):
            call()


# Behaviour in real time.


def test_real_time_follows_the_clock_and_applies_the_law() -> None:
    robot = RealtimeLoopbackRobot(TESTBOT)
    robot.connect()
    try:
        robot.set_mode(Mode.HOLD)
        robot.set_mode(Mode.POLICY)
        target = DEFAULT_POSE + 0.3
        start_s, start = time.monotonic(), robot.read().stamp_ns
        robot.write(command(TESTBOT, target))
        time.sleep(0.3)
        state, end_s = robot.read(), time.monotonic()
        simulated, elapsed = (state.stamp_ns - start) / 1e9, end_s - start_s
        # Never ahead of the clock, and not far behind it.
        assert 0.5 * elapsed < simulated <= elapsed + 2.0 * PERIOD_S
        np.testing.assert_allclose(state.position, target, atol=0.02)
    finally:
        robot.close()
    assert not any(t.name == "sender-testbot" for t in threading.enumerate())


# The command line.


def summary(output: str) -> dict[str, str]:
    lines = [re.split(r"\s{2,}", line, maxsplit=1) for line in output.splitlines()]
    return {label: value for label, value in lines}


def test_run_drives_a_draft_description_without_a_model_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["run", "--robot", "pm01", "--adapter", "loopback", "--policy", "zero"]
    assert main([*argv, "--seconds", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = summary(captured.out)
    assert lines["robot"] == "pm01 through loopback, in lockstep"
    assert lines["ticks"] == "50"
    assert lines["faults"] == "none"


def test_run_drives_a_draft_description_in_real_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["run", "--robot", "g1_29dof", "--adapter", "loopback_realtime"]
    assert main([*argv, "--policy", "zero", "--seconds", "0.2"]) == 0
    lines = summary(capsys.readouterr().out)
    assert lines["robot"] == "g1_29dof through loopback_realtime, in real time"
    assert lines["ticks"] == "10"
