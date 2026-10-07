"""Runtime data: the robot state and joint commands that cross the robot port.

Adapters produce a `RobotState` from hardware or a simulator and consume a
`JointCommand`. The safety layer, the observation builder and the runner
read them on every control tick. Arrays of joint values follow the robot
description's joint order, and every quantity is in SI units (see
docs/architecture.md).

The objects are immutable, so no component can change data that another
component is reading: fields cannot be reassigned, and every array is a
read-only float64 copy made on construction. Construction checks structure
only, so a non-finite value can be represented and rejected as data;
`check_state` and `check_command` check the values against a description.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts.conventions import normalize_quaternion
from embodiverse.contracts.description import RobotDescription

__all__ = [
    "Fault",
    "ImuState",
    "JointCommand",
    "Mode",
    "RobotState",
    "check_command",
    "check_state",
]


class Mode(str, Enum):
    """What the adapter does with the motors."""

    PASSIVE = "passive"
    """Zero torque: the joints move freely."""

    HOLD = "hold"
    """Hold position with the description's default gains; commands are not applied."""

    POLICY = "policy"
    """Joint commands are applied."""

    DAMP = "damp"
    """Damping only (kp = 0): the joints resist motion but hold no position.

    Entered only by a person's stop control, never automatically: a humanoid
    without stiffness falls unless something is holding it.
    """


@dataclass(frozen=True, slots=True)
class Fault:
    """A problem detected by an adapter or by the safety layer."""

    code: str
    """The kind of fault, for code that reacts to it.

    Lowercase snake_case parts joined by dots, from the general to the
    specific, e.g. `state.stale` or `motor.over_temperature`.
    """

    message: str
    """An explanation for people, e.g. in logs."""

    joint: str | None = None
    """Canonical name of the joint concerned, or None if not about one joint."""


# The classes holding arrays set eq=False. A generated __eq__ would compare
# numpy arrays with ==, which works element by element, and the truth value
# of the resulting array is ambiguous, so comparing would raise. Instances
# compare by identity instead.


@dataclass(frozen=True, slots=True, eq=False)
class ImuState:
    """A reading of the robot's IMU.

    The orientation is normalized on construction, unless it holds a
    non-finite value: that is kept as given, for `check_state` to reject.
    Raises ValueError if a field is not one-dimensional, has the wrong number
    of elements, or the orientation has zero length.
    """

    orientation: NDArray[np.float64]
    """Orientation of the IMU frame in a world-fixed frame with z up, as an
    (x, y, z, w) quaternion. Its heading (yaw) is arbitrary and may drift.
    `rotate_vector(orientation, v)` maps `v` from the IMU frame to that frame.
    """

    angular_velocity: NDArray[np.float64]
    """Angular velocity of the IMU frame, in rad/s, expressed in the IMU frame."""

    linear_acceleration: NDArray[np.float64]
    """Specific force, as an accelerometer measures it, in m/s², in the IMU frame.

    This is acceleration minus gravity, so a level IMU at rest reads about
    +9.81 on z.
    """

    def __post_init__(self) -> None:
        orientation = _vector(self.orientation, "orientation", 4)
        if np.isfinite(orientation).all():
            try:
                orientation = normalize_quaternion(orientation)
            except ValueError as error:
                raise ValueError(f"orientation: {error}") from None
        _store(self, "orientation", orientation)
        _store(
            self,
            "angular_velocity",
            _vector(self.angular_velocity, "angular_velocity", 3),
        )
        _store(
            self,
            "linear_acceleration",
            _vector(self.linear_acceleration, "linear_acceleration", 3),
        )


@dataclass(frozen=True, slots=True, eq=False)
class RobotState:
    """The robot's state at one instant, as its adapter reports it.

    Raises ValueError if an array is not one-dimensional, or if `position`,
    `velocity` and `effort` differ in length. `check_state` checks every
    length against a description, `temperature`'s included.
    """

    stamp_ns: int
    """When the state was measured, in ns on the monotonic clock (simulation
    time in simulators)."""

    position: NDArray[np.float64]
    """Joint positions, in rad."""

    velocity: NDArray[np.float64]
    """Joint velocities, in rad/s."""

    effort: NDArray[np.float64]
    """Joint torques, measured or estimated, in N·m."""

    imu: ImuState
    """The reading of the IMU named in the description."""

    mode: Mode
    """The mode the adapter is in."""

    temperature: NDArray[np.float64] | None = None
    """Motor temperatures, one per joint, in °C; None if the robot does not
    report them."""

    faults: tuple[Fault, ...] = ()
    """Faults the adapter has detected; empty if there are none."""

    def __post_init__(self) -> None:
        _store_joint_arrays(self, ("position", "velocity", "effort"))
        if self.temperature is not None:
            _store(self, "temperature", _vector(self.temperature, "temperature"))
        object.__setattr__(self, "faults", tuple(self.faults))


@dataclass(frozen=True, slots=True, eq=False)
class JointCommand:
    """Targets, feed-forward torques and gains for every joint.

    For each joint, the adapter applies

        tau = kp * (position - q) + kd * (velocity - dq) + effort

    where `q` and `dq` are the joint's measured position and velocity. Where
    the description's gains apply at the motor (`side: motor`), the adapter
    maps targets and gains to the motors.

    Raises ValueError if an array is not one-dimensional or the arrays differ
    in length.
    """

    position: NDArray[np.float64]
    """Target joint positions, in rad."""

    velocity: NDArray[np.float64]
    """Target joint velocities, in rad/s."""

    effort: NDArray[np.float64]
    """Feed-forward joint torques, in N·m."""

    kp: NDArray[np.float64]
    """Position stiffness, in N·m/rad."""

    kd: NDArray[np.float64]
    """Velocity damping, in N·m·s/rad."""

    def __post_init__(self) -> None:
        _store_joint_arrays(self, ("position", "velocity", "effort", "kp", "kd"))


def check_state(state: RobotState, description: RobotDescription) -> None:
    """Check that `state` fits `description` and holds only finite values.

    Raises ValueError, naming the field and any joints concerned, if an array
    of joint values does not have one element per joint, or if any value,
    the IMU's and the temperatures included, is not finite.
    """
    _check_joint_values("position", state.position, description)
    _check_joint_values("velocity", state.velocity, description)
    _check_joint_values("effort", state.effort, description)
    if state.temperature is not None:
        _check_joint_values("temperature", state.temperature, description)
    _check_finite("imu.orientation", state.imu.orientation)
    _check_finite("imu.angular_velocity", state.imu.angular_velocity)
    _check_finite("imu.linear_acceleration", state.imu.linear_acceleration)


def check_command(command: JointCommand, description: RobotDescription) -> None:
    """Check that `command` fits `description`, is finite and has no negative gain.

    Raises ValueError, naming the field and any joints concerned, if an array
    does not have one element per joint, a value is not finite, or a gain is
    negative.
    """
    _check_joint_values("position", command.position, description)
    _check_joint_values("velocity", command.velocity, description)
    _check_joint_values("effort", command.effort, description)
    _check_joint_values("kp", command.kp, description)
    _check_joint_values("kd", command.kd, description)
    _check_not_negative("kp", command.kp, description)
    _check_not_negative("kd", command.kd, description)


def _vector(
    value: ArrayLike, name: str, size: int | None = None
) -> NDArray[np.float64]:
    """Return a new one-dimensional float64 array copied from `value`."""
    array = np.array(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if size is not None and array.size != size:
        raise ValueError(f"{name} must have {size} elements, got {array.size}")
    return array


def _store(obj: object, name: str, array: NDArray[np.float64]) -> None:
    """Make `array` read-only and set it as field `name` of the frozen `obj`."""
    array.flags.writeable = False
    object.__setattr__(obj, name, array)


def _store_joint_arrays(obj: object, names: tuple[str, ...]) -> None:
    """Replace the named fields of `obj` with read-only arrays of equal length."""
    first = _vector(getattr(obj, names[0]), names[0])
    _store(obj, names[0], first)
    for name in names[1:]:
        array = _vector(getattr(obj, name), name)
        if array.size != first.size:
            raise ValueError(
                f"{name} has length {array.size} but {names[0]} has length {first.size}"
            )
        _store(obj, name, array)


def _check_joint_values(
    name: str, values: NDArray[np.float64], description: RobotDescription
) -> None:
    if values.size != description.num_joints:
        raise ValueError(
            f"{name} has length {values.size} but robot {description.name!r}"
            f" has {description.num_joints} joints"
        )
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError(f"{name} is not finite for {_joints(~finite, description)}")


def _check_not_negative(
    name: str, values: NDArray[np.float64], description: RobotDescription
) -> None:
    negative = values < 0
    if negative.any():
        raise ValueError(f"{name} is negative for {_joints(negative, description)}")


def _check_finite(name: str, values: NDArray[np.float64]) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} is not finite")


def _joints(mask: NDArray[np.bool_], description: RobotDescription) -> str:
    """Name the joints where `mask` is true, in description order."""
    indices: list[int] = np.flatnonzero(mask).tolist()
    names = [repr(description.joint_names[i]) for i in indices]
    return ("joint " if len(names) == 1 else "joints ") + ", ".join(names)
