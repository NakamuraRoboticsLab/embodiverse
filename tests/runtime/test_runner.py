"""Tests of the runner, end to end with testbot in MuJoCo.

Most tests run in lockstep, so they are deterministic and take a fraction
of real time; one checks the rate in real time. testbot's joints, in
description order, with their default positions and gains, are in
docs/robots/testbot.md. Its MJCF starts it standing in its default pose.
"""

import ast
import inspect
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from embodiverse.adapters.mujoco import MujocoRobot, RealtimeMujocoRobot
from embodiverse.contracts.conventions import IDENTITY_QUATERNION, rotate_vector
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    DescriptorError,
    PolicyDescriptor,
    load_descriptor,
    validate_against,
)
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.registry import robot_description
from embodiverse.contracts.robot import Capabilities, Robot, Steppable
from embodiverse.contracts.types import Fault, JointCommand, Mode, RobotState
from embodiverse.runtime import runner as runner_module
from embodiverse.runtime.policy import OnnxPolicy, ZeroPolicy
from embodiverse.runtime.runner import (
    ConstantCommand,
    Runner,
    RunnerConfig,
    StartRefused,
    VelocityCommandSample,
    zero_policy_descriptor,
)
from embodiverse.runtime.safety import SafetyLayer

AffinePolicy = Callable[..., Path]
DescriptorWriter = Callable[[dict[str, Any], Path], Path]

TESTBOT = robot_description("testbot")
JOINTS = TESTBOT.joint_names
DEFAULTS = np.array([joint.default_position for joint in TESTBOT.joints])
KP = np.array([joint.gains.kp for joint in TESTBOT.joints])
KD = np.array([joint.gains.kd for joint in TESTBOT.joints])
KNEES = [JOINTS.index("left_knee"), JOINTS.index("right_knee")]
RATE_HZ = 50.0
NO_RAMP = RunnerConfig(start_ramp_s=0.0)


@pytest.fixture
def testbot() -> Iterator[MujocoRobot]:
    """testbot in lockstep, connected, standing in its default pose."""
    robot = MujocoRobot(TESTBOT)
    robot.connect()
    yield robot
    robot.close()


class Recorder:
    """A robot in lockstep that records what is done to it.

    It passes every call on to `robot`, and adds `faults` to every state it
    reads, as an adapter that detects a problem reports it.
    """

    def __init__(self, robot: MujocoRobot, faults: tuple[Fault, ...] = ()) -> None:
        self.robot = robot
        self.faults = faults
        self.calls: list[str] = []
        self.commands: list[JointCommand] = []

    @property
    def description(self) -> RobotDescription:
        return self.robot.description

    def capabilities(self) -> Capabilities:
        return self.robot.capabilities()

    def connect(self) -> None:
        self.calls.append("connect")
        self.robot.connect()

    def read(self) -> RobotState:
        self.calls.append("read")
        state = self.robot.read()
        return replace(state, faults=state.faults + self.faults)

    def write(self, command: JointCommand) -> None:
        self.calls.append("write")
        self.commands.append(command)
        self.robot.write(command)

    def set_mode(self, mode: Mode) -> None:
        self.calls.append(f"set_mode {mode.name}")
        self.robot.set_mode(mode)

    def operator_stop(self) -> None:
        self.calls.append("operator_stop")
        self.robot.operator_stop()

    def close(self) -> None:
        self.calls.append("close")
        self.robot.close()

    def advance(self, seconds: float) -> None:
        self.calls.append("advance")
        self.robot.advance(seconds)


class ScriptedPolicy:
    """A policy that returns `action(tick)` and keeps every observation it sees."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        action: Callable[[int], NDArray[np.float32]] | None = None,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.observations: list[NDArray[np.float32]] = []
        self._action = action or (lambda tick: np.zeros(output_size, np.float32))

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        self.observations.append(observation.copy())
        return self._action(len(self.observations) - 1)

    def reset(self) -> None:
        self.observations.clear()


def descriptor_data(**changes: Any) -> dict[str, Any]:
    """Return the zero policy's descriptor for testbot as data, with `changes`."""
    data = zero_policy_descriptor(TESTBOT, rate_hz=RATE_HZ).model_dump(mode="json")
    data.update(changes)
    return data


def descriptor(**changes: Any) -> PolicyDescriptor:
    return PolicyDescriptor.model_validate(descriptor_data(**changes))


def zero_runner(robot: Robot, **options: Any) -> Runner:
    return Runner(robot, ZeroPolicy(3, 6), zero_policy_descriptor(TESTBOT), **options)


def pelvis_height(robot: MujocoRobot | RealtimeMujocoRobot) -> float:
    return float(robot.data.joint("floating_base").qpos[2])


def tilt(state: RobotState) -> float:
    """Return the angle between the IMU's z axis and the world's, in rad."""
    up = rotate_vector(state.imu.orientation, (0.0, 0.0, 1.0))
    return math.acos(min(1.0, float(up[2])))


def on_gantry(robot: MujocoRobot) -> None:
    """Hold testbot's pelvis 1 m above the floor, so its legs hang free."""
    robot.set_base_pose((0.0, 0.0, 1.0), IDENTITY_QUATERNION)


def onnx_descriptor(
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
    bias: list[float],
    **changes: Any,
) -> PolicyDescriptor:
    """Return a descriptor for an ONNX policy that always outputs `bias`."""
    path = affine_policy(np.zeros((3, 6)), bias)
    return load_descriptor(policy_descriptor(descriptor_data(**changes), path))


# The whole pipeline.


def test_the_zero_policy_keeps_testbot_standing(testbot: MujocoRobot) -> None:
    height = pelvis_height(testbot)
    runner = zero_runner(testbot)
    stats = runner.run(5.0)
    assert stats.ticks == 5 * RATE_HZ
    assert (stats.duration_s, stats.rate_hz, stats.jitter_s) == (5.0, RATE_HZ, 0.0)
    assert stats.lockstep
    assert dict(stats.faults) == {}
    assert stats.overruns == 0
    assert not stats.latched and stats.latched_by == ()
    assert stats.left_policy is None
    state = testbot.read()
    assert state.stamp_ns == 7_000_000_000  # the 2 s start ramp, then 5 s
    assert state.mode is Mode.POLICY
    assert abs(pelvis_height(testbot) - height) < 0.02
    assert tilt(state) < math.radians(3.0)
    # The PD loop gives way a little under testbot's weight: about 0.03 rad
    # at the knees.
    np.testing.assert_allclose(state.position, DEFAULTS, atol=0.05)
    runner.stop()
    assert testbot.read().mode is Mode.HOLD


def test_a_constant_onnx_policy_moves_the_knees_to_their_targets(
    testbot: MujocoRobot,
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
) -> None:
    # Each knee's target is its default, 0.6 rad, plus 0.25 × 0.8 = 0.2 rad.
    output = [0.0, 0.8, 0.0, 0.0, 0.8, 0.0]
    action = {"type": "position_offset", "joints": list(JOINTS), "scale": 0.25}
    constant = onnx_descriptor(affine_policy, policy_descriptor, output, action=action)
    policy = OnnxPolicy.from_descriptor(constant, TESTBOT)
    on_gantry(testbot)
    stats = Runner(testbot, policy, constant, config=NO_RAMP).run(1.0)
    assert dict(stats.faults) == {}
    position = testbot.read().position
    np.testing.assert_allclose(position[KNEES], 0.8, atol=0.01)
    others = [index for index in range(6) if index not in KNEES]
    np.testing.assert_allclose(position[others], DEFAULTS[others], atol=0.01)


def test_a_policy_that_outputs_nan_holds_on_its_first_tick(
    testbot: MujocoRobot,
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
) -> None:
    nan = onnx_descriptor(affine_policy, policy_descriptor, [math.nan] * 6)
    policy = OnnxPolicy.from_descriptor(nan, TESTBOT)
    height = pelvis_height(testbot)
    runner = Runner(testbot, policy, nan)
    first = runner.run(1.0 / RATE_HZ)
    assert first.ticks == 1
    assert first.latched
    assert [fault.code for fault in first.latched_by] == ["policy.non_finite"]
    assert dict(first.faults) == {"policy.non_finite": 1}
    # Held from then on: the policy does not run again, and testbot stands.
    rest = runner.run(3.0)
    assert rest.latched
    assert dict(rest.faults) == {}
    assert testbot.read().mode is Mode.POLICY
    assert abs(pelvis_height(testbot) - height) < 0.02
    assert tilt(testbot.read()) < math.radians(3.0)


# Checks before anything moves.


def test_refuses_another_description_before_touching_the_robot() -> None:
    recorder = Recorder(MujocoRobot(TESTBOT))
    other = descriptor(
        robot={"name": "testbot", "description_hash": "sha256:" + "0" * 64}
    )
    with pytest.raises(DescriptorError, match="trained for a different description"):
        Runner(recorder, ZeroPolicy(3, 6), other)
    assert recorder.calls == []


def test_refuses_a_policy_whose_sizes_differ_from_the_descriptor_s() -> None:
    message = (
        "the policy does not fit its descriptor: it takes 4 values, but the"
        " descriptor's observation has 3: observation_size 3 times history"
        " length 1; it returns 5 values, but the descriptor's action drives 6"
        " joints"
    )
    with pytest.raises(ValueError, match=f"^{message}$"):
        Runner(MujocoRobot(TESTBOT), ZeroPolicy(4, 5), zero_policy_descriptor(TESTBOT))


def test_refuses_a_safety_layer_for_another_description() -> None:
    data = TESTBOT.model_dump()
    data["joints"][1]["gains"]["kp"] = 81.0
    other = SafetyLayer(RobotDescription.model_validate(data))
    with pytest.raises(ValueError, match="the safety layer was made for a descrip"):
        zero_runner(MujocoRobot(TESTBOT), safety=other)


def test_refuses_to_start_when_preflight_finds_a_fault(testbot: MujocoRobot) -> None:
    fault = Fault(code="imu.error", message="the IMU reports an error")
    recorder = Recorder(testbot, faults=(fault,))
    runner = zero_runner(recorder)
    message = (
        "cannot start the policy on robot 'testbot':\n- imu.error: the IMU reports"
        " an error"
    )
    with pytest.raises(StartRefused, match=f"^{message}$"):
        runner.run(1.0)
    assert set(recorder.calls) == {"read"}
    assert testbot.read().mode is Mode.PASSIVE
    runner.stop()  # nothing to stop
    assert set(recorder.calls) == {"read"}


def test_refuses_to_start_after_a_person_s_stop(testbot: MujocoRobot) -> None:
    testbot.operator_stop()
    message = (
        "the robot is in DAMP, and a policy starts only from PASSIVE or HOLD; after"
        " a person's stop, only a person resets the robot"
    )
    with pytest.raises(StartRefused, match=message):
        zero_runner(testbot).start()
    assert testbot.read().mode is Mode.DAMP


# Starting.


def test_the_start_ramp_moves_every_joint_to_its_default_position(
    testbot: MujocoRobot,
) -> None:
    on_gantry(testbot)
    crouch = DEFAULTS + np.array([-0.4, 0.8, -0.4, -0.2, 0.4, -0.2])
    testbot.set_joint_positions(crouch)
    recorder = Recorder(testbot)
    runner = zero_runner(recorder, config=RunnerConfig(start_ramp_s=1.0))
    runner.start()
    assert recorder.calls[:3] == ["read", "set_mode HOLD", "set_mode POLICY"]
    ramp = recorder.commands
    assert len(ramp) == RATE_HZ
    for tick, command in enumerate(ramp, start=1):
        expected = crouch + tick / len(ramp) * (DEFAULTS - crouch)
        np.testing.assert_allclose(command.position, expected, atol=1e-12)
        np.testing.assert_array_equal(command.kp, KP)
        np.testing.assert_array_equal(command.kd, KD)
    # The hanging legs' weight pulls the hips about 0.01 rad off their targets.
    np.testing.assert_allclose(testbot.read().position, DEFAULTS, atol=0.02)


def test_driven_joints_get_the_descriptor_s_gains_and_held_joints_the_description_s(
    testbot: MujocoRobot,
) -> None:
    gains = {
        "left_hip_pitch": {"kp": 55.0, "kd": 1.1},
        "left_knee": {"kp": 75.0, "kd": 1.2},
        "left_ankle_pitch": {"kp": 35.0, "kd": 1.3},
    }
    left_leg = descriptor(
        action={"type": "position_offset", "joints": "left_leg", "scale": 0.25},
        held_joints="right_leg",
        gains={"source": "explicit", "values": gains},
    )
    recorder = Recorder(testbot)
    stats = Runner(recorder, ZeroPolicy(3, 3), left_leg).run(0.5)
    assert stats.ticks == 25
    assert len(recorder.commands) == 100 + 25  # the start ramp, then the policy
    for command in recorder.commands:
        assert command.kp.tolist() == [55.0, 75.0, 35.0, 60.0, 80.0, 40.0]
        assert command.kd.tolist() == [1.1, 1.2, 1.3, 2.0, 2.0, 2.0]
    np.testing.assert_allclose(recorder.commands[-1].position, DEFAULTS)


# Each tick.


def test_the_observation_has_the_raw_last_action_and_the_time_since_the_start(
    testbot: MujocoRobot,
) -> None:
    # A frame is the gait phase, with a period of four ticks, then the last
    # action, which is the policy's output before the action's scale.
    observations = [
        {"term": "gait_phase", "params": {"period": 4 / RATE_HZ}},
        {"term": "last_action"},
    ]
    timed = descriptor(observations=observations, observation_size=8)
    policy = ScriptedPolicy(
        8, 6, lambda tick: np.full(6, 0.01 * (tick + 1), np.float32)
    )
    Runner(testbot, policy, timed, config=NO_RAMP).run(5 / RATE_HZ)
    seen = np.array(policy.observations)
    np.testing.assert_allclose(
        seen[:, :2],
        [[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        seen[:, 2:], [[0.01 * tick] * 6 for tick in range(5)], rtol=1e-6
    )


class FrozenCommand:
    """A command source that stops updating its command at `until_ns`."""

    def __init__(self, until_ns: int) -> None:
        self.until_ns = until_ns

    def read(self, now_ns: int) -> VelocityCommandSample:
        return VelocityCommandSample(
            command=np.array([0.8, 0.3, 0.2]), stamp_ns=min(now_ns, self.until_ns)
        )


def test_a_stale_command_stops_the_robot_walking(testbot: MujocoRobot) -> None:
    ranges = {"lin_vel_x": [-0.5, 0.5], "lin_vel_y": [0, 0], "ang_vel_z": [0, 0]}
    walking = descriptor(
        command={"type": "velocity", "ranges": ranges},
        observations=[{"term": "velocity_command"}],
    )
    policy = ScriptedPolicy(3, 6)
    runner = Runner(
        testbot, policy, walking, commands=FrozenCommand(500_000_000), config=NO_RAMP
    )
    stats = runner.run(1.5)
    # The command stops at 0.5 s and is stale once more than 0.5 s old: from
    # the tick at 1.02 s, the 52nd of 75, on.
    assert dict(stats.faults) == {"command.stale": 24}
    assert not stats.latched
    commands = np.array(policy.observations)
    assert commands.shape == (75, 3)
    # Clipped to the descriptor's ranges while fresh; zero once stale.
    assert (commands[:51] == [0.5, 0.0, 0.0]).all()
    assert (commands[51:] == 0.0).all()


def test_a_constant_command_is_always_fresh() -> None:
    sample = ConstantCommand(0.3, 0.0, -0.1).read(123_456)
    assert sample.command.tolist() == [0.3, 0.0, -0.1]
    assert sample.stamp_ns == 123_456
    assert not sample.command.flags.writeable
    with pytest.raises(ValueError, match="a velocity command has 3 values"):
        VelocityCommandSample(command=np.zeros(2), stamp_ns=0)


# Stopping.


class StoppingCommand:
    """A command source through which a person stops the robot at `at_ns`."""

    def __init__(self, robot: Robot, at_ns: int) -> None:
        self.robot = robot
        self.at_ns = at_ns

    def read(self, now_ns: int) -> VelocityCommandSample:
        if now_ns >= self.at_ns:
            self.robot.operator_stop()  # a person, not the runner
        return VelocityCommandSample(command=np.zeros(3), stamp_ns=now_ns)


def test_a_person_s_stop_ends_the_run_and_the_robot_stays_damped(
    testbot: MujocoRobot,
) -> None:
    recorder = Recorder(testbot)
    commands = StoppingCommand(testbot, at_ns=1_000_000_000)
    runner = zero_runner(recorder, commands=commands, config=NO_RAMP)
    stats = runner.run(3.0)
    # The stop arrives during the tick at 1 s, the 51st; the next one sees it.
    assert stats.ticks == 51
    assert stats.left_policy is Mode.DAMP
    assert not runner.started
    runner.stop()
    assert testbot.read().mode is Mode.DAMP
    assert [call for call in recorder.calls if call.startswith("set_mode")] == [
        "set_mode HOLD",
        "set_mode POLICY",
    ]
    with pytest.raises(StartRefused, match="the robot is in DAMP"):
        runner.start()


def test_an_error_during_the_run_leaves_the_robot_held(testbot: MujocoRobot) -> None:
    def fail(tick: int) -> NDArray[np.float32]:
        if tick == 3:
            raise RuntimeError("the policy failed")
        return np.zeros(6, np.float32)

    runner = Runner(
        testbot, ScriptedPolicy(3, 6, fail), zero_policy_descriptor(TESTBOT)
    )
    with pytest.raises(RuntimeError, match="the policy failed"):
        runner.run(1.0)
    assert testbot.read().mode is Mode.HOLD
    assert not runner.started


def test_the_runner_never_calls_operator_stop_nor_closes_the_robot(
    testbot: MujocoRobot,
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
) -> None:
    # Not in its code at all...
    tree = ast.parse(inspect.getsource(runner_module))
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not used & {"operator_stop", "close", "connect"}
    # ...and not when a policy fails and the robot is held, nor when it stops.
    nan = onnx_descriptor(affine_policy, policy_descriptor, [math.nan] * 6)
    recorder = Recorder(testbot)
    runner = Runner(recorder, OnnxPolicy.from_descriptor(nan, TESTBOT), nan)
    assert runner.run(1.0).latched
    runner.stop()
    assert testbot.read().mode is Mode.HOLD
    assert not {"operator_stop", "close", "connect"} & set(recorder.calls)


def test_run_takes_a_positive_number_of_seconds(testbot: MujocoRobot) -> None:
    runner = zero_runner(testbot)
    for seconds in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="seconds must be a positive number"):
            runner.run(seconds)
    assert testbot.read().mode is Mode.PASSIVE


# The zero policy's descriptor.


def test_the_zero_policy_descriptor_drives_every_joint_with_no_policy_file() -> None:
    zero = zero_policy_descriptor(TESTBOT, rate_hz=100.0)
    validate_against(zero, TESTBOT)
    assert zero.policy is None
    assert (zero.robot.name, zero.robot.description_hash) == (
        "testbot",
        content_hash(TESTBOT),
    )
    assert zero.rate_hz == 100.0
    assert zero.action.joints == JOINTS
    assert zero.action.scale == 1.0
    assert zero.gains.source == "description"
    assert zero.held_joints == ()
    assert [term.term for term in zero.observations] == ["projected_gravity"]
    assert zero.observation_size == 3
    ranges = zero.command.ranges
    assert ranges.lin_vel_x == ranges.lin_vel_y == ranges.ang_vel_z == (0.0, 0.0)
    assert zero.provenance == "external"


# Real time.


def test_real_time_keeps_the_rate() -> None:
    robot = RealtimeMujocoRobot(TESTBOT)
    assert not isinstance(robot, Steppable)
    robot.connect()
    try:
        runner = zero_runner(robot, config=RunnerConfig(start_ramp_s=0.1))
        assert not runner.lockstep
        started = time.monotonic()
        stats = runner.run(0.5)
        elapsed = time.monotonic() - started
        runner.stop()
    finally:
        robot.close()
    assert not stats.lockstep
    assert stats.ticks == 25
    assert stats.rate_hz == pytest.approx(RATE_HZ, rel=0.1)
    assert stats.duration_s == pytest.approx(0.5, rel=0.1)
    assert elapsed >= 0.5
