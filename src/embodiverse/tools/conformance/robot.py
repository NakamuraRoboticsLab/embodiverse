"""The robot conformance kit: the robot port's semantics as executable checks.

The docstrings of `embodiverse.contracts.robot` say what an adapter must
do; each check here tests part of it, and raises AssertionError with what
it expected, what it observed and the likely cause. An adapter that passes
the kit behaves as the port says (docs/adding-a-robot.md).

The most dangerous adapter bugs are silent: joints in the wrong order,
degrees instead of radians, a quaternion in the wrong order, an IMU mounted
wrongly. A simulated robot catches them by moving to known poses through a
`SimulationProbe`. Hardware adapters run the checks that need no probe.

To run the kit, subclass `RobotConformance` in a test module and provide a
factory for new, unconnected robots:

    from embodiverse.tools.conformance.robot import RobotConformance

    class TestMyRobot(RobotConformance):
        @pytest.fixture
        def make_robot(self):
            return lambda: MyRobot(load_description("my_robot.yaml"))

A simulated robot also overrides the `probe` fixture. The checks are plain
functions too, for use outside pytest. They are grouped below in the order
the test class runs them: the description, the lifecycle, the state, time,
modes and commands, then the checks that need a probe.
"""

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import replace
from itertools import pairwise
from typing import Protocol, runtime_checkable

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts.conventions import (
    IDENTITY_QUATERNION,
    STANDARD_GRAVITY,
    multiply_quaternions,
    normalize_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_rpy,
    rotate_vector_inverse,
)
from embodiverse.contracts.description import Joint, RobotDescription
from embodiverse.contracts.robot import MODE_TRANSITIONS, Capabilities, Robot, Steppable
from embodiverse.contracts.types import ImuState, JointCommand, Mode, RobotState
from embodiverse.contracts.types import check_state as check_state_values

__all__ = [
    "RobotConformance",
    "SimulationProbe",
    "check_description",
    "check_imu",
    "check_joint_order",
    "check_lifecycle",
    "check_modes",
    "check_rejects_malformed_commands",
    "check_state",
    "check_timestamps",
    "check_timing",
    "check_units",
]

Vector = NDArray[np.float64]


@runtime_checkable
class SimulationProbe(Protocol):
    """Test-only control of a simulated robot, to put it in a known state.

    It is not part of the robot port: the runtime never uses it, and
    hardware has none. A simulator adapter implements it, on the robot
    itself or on a separate object, so the kit can compare what `read`
    reports with what it set up. Positions and orientations are in the
    simulator's world frame (REP-103: z up), joint values in description
    order, all in SI units.
    """

    def set_base_pose(self, position: ArrayLike, orientation_xyzw: ArrayLike) -> None:
        """Put the base at `position` (m) and `orientation_xyzw`, and hold it there.

        The base has zero velocity and stays at that pose, supported against
        gravity as on a gantry, until `release_base`: its IMU reads specific
        force +9.81 m/s² up, not free fall. While it is held, the joints
        still move under the current mode's control. Calling it again moves
        the held base to the new pose.
        """
        ...

    def release_base(self) -> None:
        """Stop holding the base, so it moves freely again."""
        ...

    def set_joint_positions(self, positions: ArrayLike) -> None:
        """Set every joint's position at once, in rad, with zero velocity.

        The simulation continues from the new positions. The mode and its
        targets do not change: HOLD keeps the targets it captured on entry.
        """
        ...

    def advance(self, seconds: float) -> None:
        """Simulate the next `seconds` of time, as `Steppable.advance` does."""
        ...


# --- The description --------------------------------------------------------


def check_description(robot: Robot) -> None:
    """Check that `description` and `capabilities()` are typed and never change."""
    description = robot.description
    if not isinstance(description, RobotDescription):
        raise AssertionError(
            f"description is a {type(description).__name__}, expected a"
            " RobotDescription: load it with load_description"
        )
    if robot.description != description:
        raise AssertionError(
            "description changed from one access to the next, expected the same"
            " description for the robot's lifetime: build it once, when the"
            " adapter is created"
        )
    capabilities = robot.capabilities()
    if not isinstance(capabilities, Capabilities):
        raise AssertionError(
            f"capabilities() returned a {type(capabilities).__name__}, expected"
            " Capabilities"
        )
    again = robot.capabilities()
    if again != capabilities:
        raise AssertionError(
            f"capabilities() returned {capabilities}, then {again}, expected the"
            " same value on every call: what a robot supports must not change"
        )


# --- The lifecycle ----------------------------------------------------------


def check_lifecycle(make_robot: Callable[[], Robot]) -> None:
    """Check `connect` and `close`, and that a robot is refused outside them.

    `make_robot` returns a new robot that is not connected. Before `connect`
    and after `close`, `read`, `write` and `set_mode` must raise
    RuntimeError and `operator_stop` must do nothing. `connect` starts in
    PASSIVE and works once; `close` is idempotent.
    """
    robot = make_robot()
    description = robot.description
    command = _command(description, _default_pose(description))
    uses = (
        ("read()", robot.read),
        ("write()", lambda: robot.write(command)),
        ("set_mode(HOLD)", lambda: robot.set_mode(Mode.HOLD)),
    )
    try:
        for name, call in uses:
            _expect_error(
                RuntimeError,
                call,
                f"{name} before connect()",
                "a robot must refuse to be used before it is connected",
            )
        _expect_success(
            robot.operator_stop,
            "operator_stop() before connect()",
            "operator_stop never raises, and does nothing before connect",
        )
        robot.connect()
        mode = robot.read().mode
        if mode is not Mode.PASSIVE:
            raise AssertionError(
                f"after connect(), read().mode is {mode.name}, expected PASSIVE:"
                " connect must leave the robot in PASSIVE, and operator_stop"
                " before connect must do nothing"
            )
        _expect_error(
            RuntimeError,
            robot.connect,
            "a second connect()",
            "a robot is connected at most once",
        )
    except BaseException:
        _close_quietly(robot)
        raise
    robot.close()
    _expect_success(robot.close, "a second close()", "close must be idempotent")
    for name, call in uses:
        _expect_error(
            RuntimeError,
            call,
            f"{name} after close()",
            "a closed robot must refuse to be used",
        )
    _expect_success(
        robot.operator_stop,
        "operator_stop() after close()",
        "operator_stop never raises, and does nothing after close",
    )
    _expect_error(
        RuntimeError,
        robot.connect,
        "connect() after close()",
        "a robot is connected at most once",
    )


# --- The state --------------------------------------------------------------


def check_state(robot: Robot) -> None:
    """Check that `read()` returns a complete state that fits the description.

    Every array has the right shape and is float64, every value is finite,
    the IMU quaternion has unit length, and temperatures are present exactly
    when `capabilities().temperature` says so.
    """
    description = robot.description
    state = robot.read()
    if not isinstance(state, RobotState):
        raise AssertionError(
            f"read() returned a {type(state).__name__}, expected a RobotState"
        )
    if isinstance(state.stamp_ns, bool) or not isinstance(state.stamp_ns, int):
        raise AssertionError(
            f"stamp_ns is {state.stamp_ns!r}, expected an int number of"
            " nanoseconds: round the time to whole nanoseconds"
        )
    joints = description.num_joints
    per_joint = f", one per joint of robot {description.name!r} in description order"
    arrays = (
        ("position", state.position, joints, per_joint),
        ("velocity", state.velocity, joints, per_joint),
        ("effort", state.effort, joints, per_joint),
        ("imu.orientation", state.imu.orientation, 4, ""),
        ("imu.angular_velocity", state.imu.angular_velocity, 3, ""),
        ("imu.linear_acceleration", state.imu.linear_acceleration, 3, ""),
    )
    for name, array, size, meaning in arrays:
        if array.shape != (size,) or array.dtype != np.float64:
            raise AssertionError(
                f"{name} is a {array.dtype} array of shape {array.shape}, expected"
                f" float64 of shape ({size},){meaning}"
            )
    try:
        check_state_values(state, description)
    except ValueError as error:
        raise AssertionError(
            f"read() returned a state that fails check_state: {error}; expected"
            " finite values, one per joint: convert every measurement to SI"
            " units in description order"
        ) from error
    norm = float(np.linalg.norm(state.imu.orientation))
    if abs(norm - 1.0) > 1e-6:
        raise AssertionError(
            f"imu.orientation has length {norm}, expected a unit quaternion"
        )
    reports = robot.capabilities().temperature
    if reports and state.temperature is None:
        raise AssertionError(
            "temperature is None, but capabilities().temperature is True:"
            " report motor temperatures, or declare temperature=False"
        )
    if not reports and state.temperature is not None:
        raise AssertionError(
            "temperature is set, but capabilities().temperature is False:"
            " leave it None, or declare temperature=True"
        )


# --- Time -------------------------------------------------------------------


def check_timestamps(robot: Robot, *, reads: int = 10, wait_periods: int = 10) -> None:
    """Check that `stamp_ns` never decreases and increases as time passes.

    A Steppable robot's time passes only in `advance`: the stamp must not
    change between reads, and must increase by the advanced time, within
    1 µs. Any other robot is read again after `wait_periods` physics
    periods of sleep, and must report a newer state.
    """
    period = 1.0 / robot.description.rates.physics_hz
    stamps = [robot.read().stamp_ns for _ in range(reads)]
    if isinstance(robot, Steppable):
        time.sleep(period)
        stamps.append(robot.read().stamp_ns)
        if stamps[-1] != stamps[0]:
            raise AssertionError(
                f"stamp_ns changed from {stamps[0]} to {stamps[-1]} ns between"
                " reads without advance(), expected no change: a Steppable"
                " robot stamps its state with simulation time, which only"
                " advance moves"
            )
        for steps in (1, 10):
            seconds = steps * period
            before = robot.read().stamp_ns
            robot.advance(seconds)
            after = robot.read().stamp_ns
            stamps += [before, after]
            expected = round(seconds * 1e9)
            if abs(after - before - expected) > 1_000:
                raise AssertionError(
                    f"stamp_ns increased by {after - before} ns after"
                    f" advance({seconds:g}), expected {expected} ns within 1 µs:"
                    " stamp the state with the simulation time, in ns"
                )
    else:
        before = stamps[-1]
        time.sleep(wait_periods * period)
        after = robot.read().stamp_ns
        stamps.append(after)
        if after <= before:
            raise AssertionError(
                f"stamp_ns stayed at {before} ns over {wait_periods * period:g} s,"
                " expected a newer state: read must return the latest state,"
                " stamped when it was measured on the monotonic clock"
            )
    for before, after in pairwise(stamps):
        if after < before:
            raise AssertionError(
                f"stamp_ns decreased from {before} to {after} ns between two"
                " reads, expected it never to decrease: stamp states on a"
                " monotonic clock, time.monotonic_ns() on hardware"
            )


def check_timing(robot: Robot, *, budget_s: float = 0.001, calls: int = 200) -> None:
    """Check that `read` and `write` return within budget, at the 95th percentile.

    Each is called `calls` times and must take at most `budget_s` at the
    95th percentile. Both are timed in POLICY, writing a command that holds the current
    position, as a control loop would. The robot is left in PASSIVE.
    """
    description = robot.description
    _reach(robot, Mode.POLICY)
    command = _command(description, robot.read().position)
    for _ in range(10):
        robot.read()
        robot.write(command)
    read_times = _durations(robot.read, calls)
    write_times = _durations(lambda: robot.write(command), calls)
    _reach(robot, Mode.PASSIVE)
    for name, durations, cause in (
        (
            "read()",
            read_times,
            "read must return the latest state without waiting for new data:"
            " receive state on another thread and keep the latest",
        ),
        (
            "write()",
            write_times,
            "write must not wait for the robot: hand the command to the"
            " transport and return",
        ),
    ):
        slow = float(np.percentile(durations, 95))
        if slow > budget_s:
            raise AssertionError(
                f"{name} took {slow * 1e3:.3f} ms at the 95th percentile of"
                f" {calls} calls, expected at most {budget_s * 1e3:.3f} ms: {cause}"
            )


# --- Modes ------------------------------------------------------------------


def check_modes(robot: Robot) -> None:
    """Check every mode transition, `operator_stop` in every mode, and the errors.

    Every transition in `MODE_TRANSITIONS`, and setting the current mode
    again, must work and be reported by the next `read`. Every other
    transition, and `set_mode(Mode.DAMP)` from any mode, must raise
    ValueError and leave the mode unchanged. `operator_stop`, called from
    another thread, must enter DAMP from every mode. The robot is left in
    PASSIVE.
    """
    for current in (Mode.PASSIVE, Mode.HOLD, Mode.POLICY):
        for target in Mode:
            _reach(robot, current)
            _check_transition(robot, current, target)
    for current in Mode:
        _reach(robot, current)
        _stop(robot, current)
    for target in Mode:
        _reach(robot, Mode.DAMP)
        _check_transition(robot, Mode.DAMP, target)
    _reach(robot, Mode.PASSIVE)


# --- Commands ---------------------------------------------------------------


def check_rejects_malformed_commands(robot: Robot) -> None:
    """Check that `write` raises ValueError for every malformed command.

    The commands have arrays of the wrong length, NaN or infinity in each
    field, a negative kp or kd, or a feed-forward effort that the robot's
    capabilities say it cannot apply. They are written in the robot's
    current mode; the `robot` fixture's is PASSIVE, which discards commands,
    so one that is wrongly accepted never reaches the motors.
    """
    description = robot.description
    joints = description.num_joints
    names = description.joint_names
    valid = _command(description, _default_pose(description))
    _expect_success(
        lambda: robot.write(valid),
        "write() of a valid command (the default pose, with the description's gains)",
        "write must accept every command that passes check_command",
    )
    rejected = (
        "write must call check_command(command, description) in every mode,"
        " before anything else"
    )
    for length in sorted({max(joints - 1, 1), joints + 1} - {joints}):
        zeros = np.zeros(length)
        command = JointCommand(
            position=zeros, velocity=zeros, effort=zeros, kp=zeros, kd=zeros
        )
        _expect_rejected(
            robot,
            command,
            f"arrays of length {length} for a robot with {joints} joints",
            rejected,
        )
    for field in ("position", "velocity", "effort", "kp", "kd"):
        for value, label in (
            (math.nan, "NaN"),
            (math.inf, "+inf"),
            (-math.inf, "-inf"),
        ):
            _expect_rejected(
                robot,
                _with_value(valid, field, 0, value),
                f"{label} in {field} for joint {names[0]!r}",
                rejected,
            )
    for field in ("kp", "kd"):
        _expect_rejected(
            robot,
            _with_value(valid, field, joints - 1, -1.0),
            f"a negative {field} for joint {names[-1]!r}",
            rejected,
        )
    effort = _with_value(valid, "effort", 0, 1.0)
    if robot.capabilities().effort_command:
        _expect_success(
            lambda: robot.write(effort),
            f"write() of a command with 1 N·m of effort for joint {names[0]!r}",
            "capabilities().effort_command is True, so feed-forward effort must"
            " be accepted",
        )
    else:
        _expect_rejected(
            robot,
            effort,
            f"1 N·m of effort for joint {names[0]!r}",
            "capabilities().effort_command is False, so write must reject any"
            " non-zero effort rather than drop it",
        )


# --- Checks with a simulation probe -----------------------------------------


def check_joint_order(
    robot: Robot,
    probe: SimulationProbe,
    *,
    offset_rad: float = 0.2,
    settle_s: float = 0.3,
    still_tolerance_rad: float = 0.05,
    base_height_m: float = 2.0,
) -> None:
    """Check that commanding one joint moves that joint, and only that one.

    The base is held in the air, `base_height_m` above the ground, so the
    feet touch nothing. For each joint in turn, starting from the default
    pose, POLICY commands the default pose with only that joint offset by
    up to `offset_rad` toward its farther limit (at most a quarter of its
    range), with the description's gains. After `settle_s`, the joint must
    have moved at least half the offset toward its target, and every other
    joint less than `still_tolerance_rad`.
    """
    description = robot.description
    names = description.joint_names
    _hold_in_air(robot, probe, base_height_m)
    _set_mode(robot, Mode.POLICY)
    for index, joint in enumerate(description.joints):
        offset = _offset(joint, offset_rad)
        baseline = _settle_at_default(robot, probe, settle_s)
        target = _default_pose(description)
        target[index] += offset
        robot.write(_command(description, target))
        _advance(probe, settle_s, description.rates.physics_hz)
        moved = robot.read().position - baseline
        toward = moved[index] * math.copysign(1.0, offset)
        others = [
            other
            for other in range(len(names))
            if other != index and abs(moved[other]) >= still_tolerance_rad
        ]
        most = max(others, key=lambda other: abs(moved[other]), default=None)
        if toward < 0.5 * abs(offset):
            observed = (
                f"joint {names[index]!r} did not move when commanded: expected at"
                f" least {0.5 * abs(offset):.3f} rad toward a target"
                f" {offset:+.3f} rad away, observed {moved[index]:+.3f} rad"
            )
            if most is not None:
                raise AssertionError(
                    f"{observed}; joint {names[most]!r} moved {moved[most]:+.3f}"
                    " rad instead: joint order may be wrong; map description"
                    " order to the motors by motor_index, in read and in write"
                )
            if toward <= -0.5 * abs(offset):
                raise AssertionError(
                    f"{observed}, away from the target: the joint's sign may be"
                    " inverted between read and write"
                )
            raise AssertionError(
                f"{observed}, and no other joint moved: commands written in"
                " POLICY may not be applied, or the joint cannot move"
            )
        if most is not None:
            raise AssertionError(
                f"joint {names[most]!r} moved {moved[most]:+.3f} rad when only"
                f" {names[index]!r} was commanded to move, expected less than"
                f" {still_tolerance_rad} rad: joint order may be wrong, or the"
                " joints are coupled"
            )
    _release(robot, probe)


def check_units(
    robot: Robot,
    probe: SimulationProbe,
    *,
    offset_rad: float = 0.2,
    settle_s: float = 0.3,
    tolerance_rad: float = 0.05,
    base_height_m: float = 2.0,
) -> None:
    """Check that each joint settles where commanded, and its velocity agrees.

    With the base held in the air, POLICY commands each joint in turn to the
    default pose plus a known offset, as `check_joint_order` does. After
    `settle_s` the measured position must equal the target within
    `tolerance_rad`, which catches positions in degrees, wrong gear ratios
    and inverted signs. Over the first 20 ms of the motion, the reported
    velocity must have the motion's sign and agree with the change in
    position within a factor of 2, which catches the same errors in the
    velocity.
    """
    description = robot.description
    names = description.joint_names
    rate = description.rates.physics_hz
    window_steps = max(1, round(0.02 * rate))
    window_s = window_steps / rate
    _hold_in_air(robot, probe, base_height_m)
    _set_mode(robot, Mode.POLICY)
    for index, joint in enumerate(description.joints):
        name = names[index]
        offset = _offset(joint, offset_rad)
        baseline = _settle_at_default(robot, probe, settle_s)
        target = _default_pose(description)
        target[index] += offset
        robot.write(_command(description, target))
        start = robot.read().position[index]
        velocities: list[float] = []
        for _ in range(window_steps):
            probe.advance(1.0 / rate)
            velocities.append(float(robot.read().velocity[index]))
        early = robot.read().position[index] - start
        _advance(probe, max(settle_s - window_s, 1.0 / rate), rate)
        measured = float(robot.read().position[index])
        if abs(measured - target[index]) > tolerance_rad:
            moved = measured - baseline[index]
            raise AssertionError(
                f"joint {name!r} settled at {measured:+.4f} rad when commanded to"
                f" {target[index]:+.4f} rad, expected within {tolerance_rad} rad;"
                f" it moved {moved:+.4f} rad for a commanded {offset:+.4f} rad:"
                f" {_position_cause(moved / offset)}"
            )
        if abs(early) < 0.01 * abs(offset):
            raise AssertionError(
                f"joint {name!r} moved only {early:+.5f} rad in the first"
                f" {window_s * 1e3:.0f} ms after a command {offset:+.3f} rad away,"
                " expected it to start at once: a command written in POLICY"
                " takes effect at the next physics step"
            )
        mean_velocity = float(np.mean(velocities))
        speed = early / window_s
        if not 0.5 <= mean_velocity / speed <= 2.0:
            raise AssertionError(
                f"joint {name!r} moved at {speed:+.4f} rad/s on average over"
                f" {window_s * 1e3:.0f} ms, but its velocity averaged"
                f" {mean_velocity:+.4f} rad/s, expected the same within a factor"
                f" of 2: {_velocity_cause(mean_velocity / speed)}"
            )
    _release(robot, probe)


def check_imu(
    robot: Robot,
    probe: SimulationProbe,
    *,
    settle_s: float = 0.1,
    angle_tolerance_rad: float = math.radians(1.0),
    acceleration_tolerance: float = 0.3,
    angular_velocity_tolerance: float = 0.05,
    base_height_m: float = 2.0,
) -> None:
    """Check the IMU against a base held still at known orientations.

    The base is held in the air, in HOLD, level, then rolled +10°, pitched
    +10° and turned +30° in yaw. At each pose the IMU orientation must equal
    the base orientation composed with the description's IMU `mount_rpy`,
    within `angle_tolerance_rad`. The heading of `ImuState.orientation` is
    arbitrary, so orientations are compared up to a constant heading: by the
    direction of gravity in the IMU frame, and by the rotation from the level
    pose. The accelerometer must read the specific force of a supported
    body, `rotate_vector_inverse(imu_orientation, (0, 0, 9.81))`, within
    `acceleration_tolerance` (m/s²), and the gyro must read less than
    `angular_velocity_tolerance` (rad/s).
    """
    description = robot.description
    mount_rpy = description.imu.mount_rpy
    mount = quaternion_from_rpy(*mount_rpy)
    _hold_in_air(robot, probe, base_height_m)
    readings: list[ImuState] = []
    for _, base in _IMU_POSES:
        probe.set_base_pose((0.0, 0.0, base_height_m), base)
        _advance(probe, settle_s, description.rates.physics_hz)
        readings.append(robot.read().imu)
    bases = [base for _, base in _IMU_POSES]
    measured = [imu.orientation for imu in readings]
    errors = _orientation_errors(_reported_correctly, bases, mount, measured)
    if max(errors) > angle_tolerance_rad:
        index = next(i for i, e in enumerate(errors) if e > angle_tolerance_rad)
        pose, base = _IMU_POSES[index]
        cause = next(
            (
                text.format(rpy=_format(mount_rpy))
                for model, text in _IMU_MISTAKES
                if max(_orientation_errors(model, bases, mount, measured))
                <= angle_tolerance_rad
            ),
            "check the IMU frame, the description's mount_rpy and the quaternion order",
        )
        raise AssertionError(
            f"IMU orientation with the base {pose}: expected"
            f" {_format(multiply_quaternions(base, mount))}, the base orientation"
            f" composed with the mount, observed {_format(measured[index])},"
            f" {math.degrees(errors[index]):.1f}° apart (up to heading), more"
            f" than {math.degrees(angle_tolerance_rad):.1f}°: {cause}"
        )
    accelerations = [imu.linear_acceleration for imu in readings]
    errors = _acceleration_errors(_specific_force, bases, mount, accelerations)
    if max(errors) > acceleration_tolerance:
        index = next(i for i, e in enumerate(errors) if e > acceleration_tolerance)
        pose, base = _IMU_POSES[index]
        cause = next(
            (
                text
                for model, text in _ACCELERATION_MISTAKES
                if max(_acceleration_errors(model, bases, mount, accelerations))
                <= acceleration_tolerance
            ),
            "check the accelerometer's frame and units (m/s²)",
        )
        raise AssertionError(
            f"IMU linear_acceleration with the base {pose}: expected"
            f" {_format(_specific_force(base, mount))} m/s², the specific force"
            f" of a supported body in the IMU frame, observed"
            f" {_format(accelerations[index])}, {errors[index]:.2f} m/s² apart,"
            f" more than {acceleration_tolerance}: {cause}"
        )
    for (pose, _), imu in zip(_IMU_POSES, readings, strict=True):
        rate = float(np.linalg.norm(imu.angular_velocity))
        if rate > angular_velocity_tolerance:
            raise AssertionError(
                f"IMU angular_velocity with the base {pose}: expected less than"
                f" {angular_velocity_tolerance} rad/s, as the base is held still,"
                f" observed {_format(imu.angular_velocity)} rad/s: check that"
                " the probe holds the base still, and the gyro's units and bias"
            )
    _release(robot, probe)


# --- The pytest test class --------------------------------------------------


class RobotConformance:
    """The robot conformance kit as a pytest test class; subclass it to run it.

    A subclass, named `Test...` so that pytest collects it, defines a
    `make_robot` fixture returning a factory for new, unconnected robots.
    For a simulated robot it also overrides `probe` to return the
    `SimulationProbe` of the robot made by the `robot` fixture, for example
    `return robot` if the adapter is its own probe; without a probe, the
    checks that need one are skipped. The tolerances below are class
    attributes, so a subclass can tune them for its robot.
    """

    timing_budget_s: float = 0.001
    """Budget for one `read` or `write`, at the 95th percentile, in s."""

    joint_offset_rad: float = 0.2
    """Largest offset commanded to a joint in the joint order and units checks."""

    joint_settle_s: float = 0.3
    """Time allowed for joints to reach a commanded position, in s."""

    joint_still_tolerance_rad: float = 0.05
    """How far a joint that is not commanded to move may move, in rad."""

    joint_position_tolerance_rad: float = 0.05
    """How far a settled joint may be from its target, in rad."""

    imu_settle_s: float = 0.1
    """Time allowed for the IMU to settle at each held pose, in s."""

    imu_angle_tolerance_rad: float = math.radians(1.0)
    """How far the IMU orientation may be from the expected one, in rad."""

    imu_acceleration_tolerance: float = 0.3
    """How far the accelerometer may be from the expected specific force, in m/s²."""

    imu_angular_velocity_tolerance: float = 0.05
    """Largest gyro reading allowed while the base is held still, in rad/s."""

    base_height_m: float = 2.0
    """Height at which the probe holds the base, high enough for the feet to
    clear the ground, in m."""

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        """A factory for new, unconnected robots. Subclasses must define it."""
        pytest.fail(
            f"{type(self).__name__} must define a make_robot fixture that returns"
            " a factory for new, unconnected robots",
            pytrace=False,
        )

    @pytest.fixture
    def robot(self, make_robot: Callable[[], Robot]) -> Iterator[Robot]:
        """A new robot, connected for the test and closed after it."""
        robot = make_robot()
        robot.connect()
        try:
            yield robot
        finally:
            robot.close()

    @pytest.fixture
    def probe(self, robot: Robot) -> SimulationProbe | None:
        """The probe of `robot`, or None for a robot that has none."""
        return None

    def test_description(self, robot: Robot) -> None:
        check_description(robot)

    def test_lifecycle(self, make_robot: Callable[[], Robot]) -> None:
        check_lifecycle(make_robot)

    def test_state(self, robot: Robot) -> None:
        check_state(robot)

    def test_timestamps(self, robot: Robot) -> None:
        check_timestamps(robot)

    def test_timing(self, robot: Robot) -> None:
        check_timing(robot, budget_s=self.timing_budget_s)

    def test_modes(self, robot: Robot) -> None:
        check_modes(robot)

    def test_rejects_malformed_commands(self, robot: Robot) -> None:
        check_rejects_malformed_commands(robot)

    def test_joint_order(self, robot: Robot, probe: SimulationProbe | None) -> None:
        check_joint_order(
            robot,
            _require(probe),
            offset_rad=self.joint_offset_rad,
            settle_s=self.joint_settle_s,
            still_tolerance_rad=self.joint_still_tolerance_rad,
            base_height_m=self.base_height_m,
        )

    def test_units(self, robot: Robot, probe: SimulationProbe | None) -> None:
        check_units(
            robot,
            _require(probe),
            offset_rad=self.joint_offset_rad,
            settle_s=self.joint_settle_s,
            tolerance_rad=self.joint_position_tolerance_rad,
            base_height_m=self.base_height_m,
        )

    def test_imu(self, robot: Robot, probe: SimulationProbe | None) -> None:
        check_imu(
            robot,
            _require(probe),
            settle_s=self.imu_settle_s,
            angle_tolerance_rad=self.imu_angle_tolerance_rad,
            acceleration_tolerance=self.imu_acceleration_tolerance,
            angular_velocity_tolerance=self.imu_angular_velocity_tolerance,
            base_height_m=self.base_height_m,
        )


# --- Helpers ----------------------------------------------------------------

_GRAVITY = np.array([0.0, 0.0, STANDARD_GRAVITY])
_UP = np.array([0.0, 0.0, 1.0])

_IMU_POSES: tuple[tuple[str, Vector], ...] = (
    ("level", np.array(IDENTITY_QUATERNION)),
    ("rolled +10°", quaternion_from_axis_angle((1.0, 0.0, 0.0), math.radians(10.0))),
    ("pitched +10°", quaternion_from_axis_angle((0.0, 1.0, 0.0), math.radians(10.0))),
    (
        "turned +30° in yaw",
        quaternion_from_axis_angle((0.0, 0.0, 1.0), math.radians(30.0)),
    ),
)
"""The poses at which `check_imu` holds the base, the level one first."""

_ReadingModel = Callable[[Vector, Vector], Vector]
"""What an adapter reports for an IMU reading, given the base orientation and
the mount."""


def _reported_correctly(base: Vector, mount: Vector) -> Vector:
    return multiply_quaternions(base, mount)


_IMU_MISTAKES: tuple[tuple[_ReadingModel, str], ...] = (
    (
        lambda base, mount: np.roll(multiply_quaternions(base, mount), 1),
        "it is in (w, x, y, z) order, as MuJoCo reports it, but the port uses"
        " (x, y, z, w): convert it with wxyz_to_xyzw",
    ),
    (
        lambda base, mount: np.roll(multiply_quaternions(base, mount), 2),
        "its components are out of order: a (w, x, y, z) quaternion was"
        " converted with xyzw_to_wxyz instead of wxyz_to_xyzw",
    ),
    (
        lambda base, mount: np.roll(multiply_quaternions(base, mount), -1),
        "its components are out of order: a quaternion already in (x, y, z, w)"
        " order was converted again with wxyz_to_xyzw",
    ),
    (
        lambda base, mount: _conjugate(multiply_quaternions(base, mount)),
        "it is the inverse rotation: report the IMU frame's orientation in the"
        " world, not the world's in the IMU frame",
    ),
    (
        lambda base, mount: base,
        "it is the orientation of the base, not of the IMU frame: the IMU's"
        " mount_rpy {rpy} is not applied",
    ),
    (
        lambda base, mount: multiply_quaternions(base, _conjugate(mount)),
        "the IMU's mount_rpy {rpy} is applied inverted",
    ),
    (
        lambda base, mount: multiply_quaternions(mount, base),
        "the IMU's mount_rpy {rpy} is applied on the wrong side: the IMU"
        " orientation is the base orientation times the mount",
    ),
)
"""Mistakes an adapter can make with the IMU orientation, and their causes."""

_ACCELERATION_MISTAKES: tuple[tuple[_ReadingModel, str], ...] = (
    (
        lambda base, mount: -_specific_force(base, mount),
        "it reads gravity, not specific force: invert its sign; a level IMU at"
        " rest reads +9.81 m/s² on z",
    ),
    (
        lambda base, mount: np.zeros(3),
        "it reads free fall: the probe must support the held base, as a gantry"
        " does, or the reading leaves out gravity",
    ),
    (
        lambda base, mount: _GRAVITY,
        "it is expressed in the world frame: express it in the IMU frame",
    ),
    (
        lambda base, mount: rotate_vector_inverse(base, _GRAVITY),
        "it is expressed in the base frame: apply the IMU's mount",
    ),
)
"""Mistakes an adapter can make with the accelerometer, and their causes."""


def _require(probe: SimulationProbe | None) -> SimulationProbe:
    if probe is None:
        pytest.skip("no SimulationProbe: this check runs only on simulated robots")
    return probe


def _default_pose(description: RobotDescription) -> Vector:
    return np.array([joint.default_position for joint in description.joints])


def _command(description: RobotDescription, position: ArrayLike) -> JointCommand:
    """Return a command for `position`, with the description's gains."""
    zeros = np.zeros(description.num_joints)
    return JointCommand(
        position=np.asarray(position, dtype=np.float64),
        velocity=zeros,
        effort=zeros,
        kp=np.array([joint.gains.kp for joint in description.joints]),
        kd=np.array([joint.gains.kd for joint in description.joints]),
    )


def _with_value(
    command: JointCommand, field: str, index: int, value: float
) -> JointCommand:
    """Return `command` with element `index` of `field` set to `value`."""
    array = np.array(getattr(command, field))
    array[index] = value
    return replace(command, **{field: array})


def _offset(joint: Joint, offset_rad: float) -> float:
    """Return a safe offset from the joint's default, toward its farther limit."""
    lower, upper = joint.limits.position
    size = min(offset_rad, 0.25 * (upper - lower))
    up = upper - joint.default_position >= joint.default_position - lower
    return size if up else -size


def _format(values: ArrayLike) -> str:
    return "(" + ", ".join(f"{value:.3f}" for value in np.asarray(values)) + ")"


def _durations(call: Callable[[], object], calls: int) -> Vector:
    durations = np.empty(calls)
    for index in range(calls):
        start = time.perf_counter()
        call()
        durations[index] = time.perf_counter() - start
    return durations


def _expect_error(
    error: type[Exception], call: Callable[[], object], what: str, cause: str
) -> None:
    try:
        call()
    except error:
        return
    except Exception as other:
        raise AssertionError(
            f"{what} raised {type(other).__name__} ({other}), expected"
            f" {error.__name__}: {cause}"
        ) from other
    raise AssertionError(
        f"{what} returned, expected it to raise {error.__name__}: {cause}"
    )


def _expect_success(call: Callable[[], object], what: str, cause: str) -> None:
    try:
        call()
    except Exception as error:
        raise AssertionError(
            f"{what} raised {type(error).__name__} ({error}), expected it to"
            f" succeed: {cause}"
        ) from error


def _expect_rejected(
    robot: Robot, command: JointCommand, what: str, cause: str
) -> None:
    try:
        robot.write(command)
    except ValueError:
        return
    except Exception as other:
        raise AssertionError(
            f"write() of a command with {what} raised {type(other).__name__}"
            f" ({other}), expected ValueError: {cause}"
        ) from other
    raise AssertionError(
        f"write() accepted a command with {what}, expected ValueError: {cause}"
    )


def _close_quietly(robot: Robot) -> None:
    """Close `robot` while another error is being raised, keeping that error."""
    try:
        robot.close()
    except Exception:
        pass


# Modes


def _path(current: Mode, target: Mode) -> list[Mode]:
    """Return the modes `set_mode` passes through from `current` to `target`."""
    paths = {current: list[Mode]()}
    queue = deque([current])
    while queue:
        mode = queue.popleft()
        for following in MODE_TRANSITIONS[mode]:
            if following not in paths:
                paths[following] = [*paths[mode], following]
                queue.append(following)
    return paths[target]


def _reach(robot: Robot, target: Mode) -> None:
    """Bring `robot` to `target` by allowed transitions, or by a stop for DAMP."""
    current = robot.read().mode
    if current is target:
        return
    if target is Mode.DAMP:
        _stop(robot, current)
        return
    for mode in _path(current, target):
        _set_mode(robot, mode)


def _set_mode(robot: Robot, target: Mode) -> None:
    """Change to `target`, which the table allows, and check that it took."""
    current = robot.read().mode
    try:
        robot.set_mode(target)
    except ValueError as error:
        raise AssertionError(
            f"set_mode({target.name}) in {current.name} raised ValueError"
            f" ({error}), expected it to succeed: MODE_TRANSITIONS allows it;"
            " call check_transition with the adapter's current mode"
        ) from error
    mode = robot.read().mode
    if mode is not target:
        raise AssertionError(
            f"after set_mode({target.name}) in {current.name}, read().mode is"
            f" {mode.name}, expected {target.name}: the next read must report"
            " the new mode"
        )


def _check_transition(robot: Robot, current: Mode, target: Mode) -> None:
    """Check `set_mode(target)` in `current`: allowed, or refused unchanged."""
    if target is not Mode.DAMP and (
        target is current or target in MODE_TRANSITIONS[current]
    ):
        _set_mode(robot, target)
        return
    if target is Mode.DAMP:
        reason = "DAMP is entered only through operator_stop"
    else:
        reachable = " or ".join(sorted(m.name for m in MODE_TRANSITIONS[current]))
        reason = f"{current.name} can change only to {reachable}"
    what = f"set_mode({target.name}) in {current.name}"
    cause = f"{reason}; call check_transition(current, target) in set_mode"
    _expect_error(ValueError, lambda: robot.set_mode(target), what, cause)
    mode = robot.read().mode
    if mode is not current:
        raise AssertionError(
            f"{what} raised ValueError, but read().mode is now {mode.name},"
            f" expected {current.name}: a refused transition must leave the"
            " mode unchanged"
        )


def _stop(robot: Robot, current: Mode) -> None:
    """Call `operator_stop` from another thread, and check that DAMP follows."""
    errors: list[Exception] = []

    def stop() -> None:
        try:
            robot.operator_stop()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=stop, name="operator-stop", daemon=True)
    thread.start()
    thread.join(timeout=1.0)
    if thread.is_alive():
        raise AssertionError(
            f"operator_stop() in {current.name}, called from another thread, did"
            " not return within 1 s, expected it to return at once: it must"
            " never wait for the control thread"
        )
    if errors:
        raise AssertionError(
            f"operator_stop() in {current.name} raised"
            f" {type(errors[0]).__name__} ({errors[0]}), expected it never to"
            " raise"
        ) from errors[0]
    mode = robot.read().mode
    if mode is not Mode.DAMP:
        raise AssertionError(
            f"after operator_stop() in {current.name}, read().mode is"
            f" {mode.name}, expected DAMP: operator_stop must enter DAMP from"
            " every mode"
        )


# Moving the robot with a probe


def _advance(probe: SimulationProbe, seconds: float, physics_hz: float) -> None:
    """Advance by `seconds`, rounded to a positive whole number of steps."""
    probe.advance(max(1, round(seconds * physics_hz)) / physics_hz)


def _hold_in_air(robot: Robot, probe: SimulationProbe, height_m: float) -> None:
    """Hold the base level in the air, with the joints in HOLD at the default pose."""
    _reach(robot, Mode.PASSIVE)
    probe.set_base_pose((0.0, 0.0, height_m), IDENTITY_QUATERNION)
    probe.set_joint_positions(_default_pose(robot.description))
    _set_mode(robot, Mode.HOLD)


def _settle_at_default(robot: Robot, probe: SimulationProbe, settle_s: float) -> Vector:
    """Command the default pose in POLICY, let it settle, and return the positions."""
    description = robot.description
    default = _default_pose(description)
    probe.set_joint_positions(default)
    robot.write(_command(description, default))
    _advance(probe, settle_s, description.rates.physics_hz)
    return robot.read().position


def _release(robot: Robot, probe: SimulationProbe) -> None:
    _reach(robot, Mode.PASSIVE)
    probe.release_base()


_DEGREES_PER_RADIAN = math.degrees(1.0)


def _position_cause(ratio: float) -> str:
    """Name the likely cause of a reported move `ratio` times the commanded one."""
    if abs(ratio - _DEGREES_PER_RADIAN) < 0.1 * _DEGREES_PER_RADIAN:
        return "positions look like degrees (×57.3): convert them to radians"
    if abs(ratio + 1.0) < 0.2:
        return (
            "the move reads with the wrong sign: the joint's direction differs"
            " between read and write"
        )
    if abs(ratio) < 0.2:
        return (
            "the joint barely moved: the command may reach another joint (see"
            " check_joint_order), or the gains are too low to reach the target"
        )
    if abs(ratio - 1.0) < 0.2:
        return "the move is right but the position is offset: check the joint's zero"
    return (
        f"the reported move is {ratio:.2f} times the commanded one: a gear ratio"
        " or unit conversion may be wrong"
    )


def _velocity_cause(ratio: float) -> str:
    """Name the likely cause of a velocity `ratio` times the observed speed."""
    if abs(ratio - _DEGREES_PER_RADIAN) < 0.1 * _DEGREES_PER_RADIAN:
        return "velocities look like degrees per second: convert them to rad/s"
    if ratio < 0.0:
        return "the velocity has the wrong sign: check the joint's direction"
    if ratio < 0.1:
        return "the velocity reads about zero: report the measured joint velocity"
    return (
        f"the velocity is {ratio:.2f} times the speed: a gear ratio or unit"
        " conversion may be wrong"
    )


# The IMU


def _conjugate(q: ArrayLike) -> Vector:
    x, y, z, w = normalize_quaternion(q)
    return np.array([-x, -y, -z, w])


def _rotation_angle(a: ArrayLike, b: ArrayLike) -> float:
    """Return the angle of the rotation between two quaternions, in rad."""
    dot = abs(float(np.dot(normalize_quaternion(a), normalize_quaternion(b))))
    return 2.0 * math.acos(min(dot, 1.0))


def _vector_angle(u: Vector, v: Vector) -> float:
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.acos(max(-1.0, min(cosine, 1.0)))


def _orientation_errors(
    model: _ReadingModel, bases: list[Vector], mount: Vector, measured: list[Vector]
) -> list[float]:
    """Return, for each pose, how far `measured` is from what `model` reports.

    Orientations are compared up to a constant heading, which the port
    leaves arbitrary: by the direction of gravity in the IMU frame, and by
    the rotation from the first (level) pose.
    """
    first = model(bases[0], mount)
    errors: list[float] = []
    for base, reading in zip(bases, measured, strict=True):
        expected = model(base, mount)
        tilt = _vector_angle(
            rotate_vector_inverse(expected, _UP), rotate_vector_inverse(reading, _UP)
        )
        turn = _rotation_angle(
            multiply_quaternions(_conjugate(first), expected),
            multiply_quaternions(_conjugate(measured[0]), reading),
        )
        errors.append(max(tilt, turn))
    return errors


def _specific_force(base: Vector, mount: Vector) -> Vector:
    """Return the accelerometer reading of a supported body at rest."""
    return rotate_vector_inverse(multiply_quaternions(base, mount), _GRAVITY)


def _acceleration_errors(
    model: _ReadingModel, bases: list[Vector], mount: Vector, measured: list[Vector]
) -> list[float]:
    """Return, for each pose, how far `measured` is from what `model` reports."""
    return [
        float(np.linalg.norm(reading - model(base, mount)))
        for base, reading in zip(bases, measured, strict=True)
    ]
