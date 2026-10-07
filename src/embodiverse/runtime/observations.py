"""Policy observations, built from the named terms of a policy descriptor.

`ObservationBuilder` turns a descriptor's `observations` and `history` into
the exact input the policy expects, on every tick. It builds each term with
the factory registered under the term's name, multiplies the values each
term returns by the entry's `scale`, concatenates the terms into one frame,
and lays out the last `history.length` frames as `history.layout` says.

The built-in terms are the factories in this module. They are registered as
entry points in the `embodiverse.observations` group, in the package's
pyproject.toml, as a plugin package registers its own, so they are found
the same way and `embodiverse list` shows them. Each factory's docstring
states the term's exact definition: its frame, sign, order and units. A
policy's inputs must match the training framework's, value for value, so a
term's definition never changes; a different quantity gets a new name.
docs/policies.md describes the terms and the builder.
"""

import math
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from embodiverse.contracts.conventions import quaternion_from_rpy, rotate_vector
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    ObservationTerm,
    PolicyDescriptor,
    resolve_joints,
)
from embodiverse.contracts.observation import Term, TermFactory, TermInputs
from embodiverse.contracts.registry import OBSERVATIONS, load

__all__ = [
    "ObservationBuilder",
    "base_angular_velocity",
    "gait_phase",
    "joint_position_relative",
    "joint_velocity",
    "last_action",
    "projected_gravity",
    "term_factory",
    "velocity_command",
]


def term_factory(name: str) -> TermFactory:
    """Return the factory registered as `name` in the `embodiverse.observations` group.

    Raises RegistryError, from `load`, if no term has that name, listing the
    names that do, or if importing the factory fails. Raises TypeError if
    the entry point names an object that cannot be called.
    """
    factory = load(OBSERVATIONS, name)
    if not callable(factory):
        raise TypeError(
            f"observation term {name!r} is registered as {factory!r}, which is not"
            f" callable: an entry point in group {OBSERVATIONS!r} must name a"
            " TermFactory"
        )
    return cast(TermFactory, factory)


class ObservationBuilder:
    """Builds a policy's input from its descriptor, one tick at a time.

    On creation, it builds every term of `descriptor.observations` with the
    factory registered under its name, in order, and checks that the sizes
    of the terms add up to `descriptor.observation_size`. It does not check
    the descriptor against `description` (`validate_against` does), but the
    factories raise ValueError for a joint the description does not have.

    Each `build` computes one frame: each term's values, multiplied by its
    entry's `scale`, concatenated in order. The builder keeps the last
    `history.length` frames, oldest first, and returns them laid out as
    `history.layout` says: `per_term`, each term's frames together, oldest
    to newest, then the next term's; or `per_frame`, whole frames, oldest to
    newest.

    The first `build` after creation or `reset` fills the whole history
    with copies of its frame, as Isaac Lab's history buffers do on their
    first push after a reset, so the policy never sees frames that were not
    observed.

    Raises RegistryError if a term's name is not registered, ValueError if
    a factory rejects its entry, a scale's length differs from its term's
    size, or the frame size differs from `observation_size`, and TypeError
    if a factory returns something that is not a `Term`.
    """

    def __init__(
        self, descriptor: PolicyDescriptor, description: RobotDescription
    ) -> None:
        terms: list[_Entry] = []
        start = 0
        for index, entry in enumerate(descriptor.observations):
            where = f"observations[{index}] ({entry.term})"
            term = term_factory(entry.term)(entry, description, descriptor)
            if not isinstance(term, Term):
                raise TypeError(
                    f"{where}: the factory returned {term!r}, which is not a Term:"
                    " a term has a size and is called with the tick's inputs"
                )
            size = term.size
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(
                    f"{where}: the term's size must be a non-negative integer,"
                    f" got {size!r}"
                )
            scale = _scale(entry.scale, size, where)
            terms.append(_Entry(entry.term, where, term, size, scale, start))
            start += size
        if start != descriptor.observation_size:
            listing = "".join(f"\n- {item.where}: {item.size}" for item in terms)
            raise ValueError(
                f"observation_size is {descriptor.observation_size}, but the terms"
                f" give {start} values per frame:{listing}"
            )
        self._terms = tuple(terms)
        self._frame_size = start
        length = descriptor.history.length
        self._frames = np.zeros((length, start))
        self._order = _layout(
            [item.size for item in terms], length, descriptor.history.layout
        )
        self._started = False

    @property
    def frame_size(self) -> int:
        """Number of values in one frame: the descriptor's `observation_size`."""
        return self._frame_size

    @property
    def size(self) -> int:
        """Number of values `build` returns: `frame_size` times the history length."""
        return self._order.size

    @property
    def term_sizes(self) -> tuple[tuple[str, int], ...]:
        """The name and size of each term, in input order."""
        return tuple((item.name, item.size) for item in self._terms)

    def build(self, inputs: TermInputs) -> NDArray[np.float32]:
        """Compute this tick's frame, add it to the history, and return the input.

        Returns a new float32 array of `size` values, as ONNX policies take
        them. Values are not checked: a non-finite state gives non-finite
        inputs, which reach the policy, and its output, as data. Raises
        ValueError, naming the term, if a term returns the wrong number of
        values; the history is then left unchanged.
        """
        frame = np.empty(self._frame_size)
        for item in self._terms:
            values = np.asarray(item.term(inputs), dtype=np.float64)
            if values.shape != (item.size,):
                raise ValueError(
                    f"{item.where} returned an array of shape {values.shape}, but"
                    f" the term's size is {item.size}"
                )
            frame[item.start : item.start + item.size] = values * item.scale
        if self._started:
            self._frames[:-1] = self._frames[1:]
            self._frames[-1] = frame
        else:
            self._frames[:] = frame
            self._started = True
        return self._frames.reshape(-1)[self._order].astype(np.float32)

    def reset(self) -> None:
        """Forget the history: the next `build` fills it with its frame."""
        self._started = False


# The built-in term factories. Each one is registered in pyproject.toml
# under its own name.


def base_angular_velocity(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `base_angular_velocity`: the base's angular velocity, in the base frame.

    3 values, (x, y, z), in rad/s: the IMU's `angular_velocity`, rotated from
    the IMU frame into the base frame through the IMU's mount, that is
    `rotate_vector(quaternion_from_rpy(*imu.mount_rpy), angular_velocity)`.
    The IMU is fixed to the base, so this is the base's own angular
    velocity, as Isaac Lab's `base_ang_vel` gives it.

    Takes no joints and no parameters. Raises ValueError if the entry has
    either, or if the IMU is not mounted on the base link.
    """
    _check_no_joints(entry)
    _check_params(entry)
    return _BaseAngularVelocity(_imu_to_base(entry, description))


def projected_gravity(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `projected_gravity`: the direction of gravity, in the base frame.

    3 values, (x, y, z), a unit vector: the world's downward direction
    (0, 0, -1) expressed in the base frame,
    `rotate_vector(mount, rotate_vector_inverse(imu.orientation, (0, 0, -1)))`
    with `mount = quaternion_from_rpy(*imu.mount_rpy)`. It is (0, 0, -1)
    with the base level; (sin a, 0, -cos a) with the base pitched nose down
    by a; and (0, -sin a, -cos a) with it rolled left side up by a. It does
    not depend on the heading, which the IMU leaves arbitrary. Isaac Lab
    calls it `projected_gravity`, the articulation's `projected_gravity_b`.

    Takes no joints and no parameters. Raises ValueError if the entry has
    either, or if the IMU is not mounted on the base link.
    """
    _check_no_joints(entry)
    _check_params(entry)
    return _ProjectedGravity(_imu_to_base(entry, description))


def velocity_command(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `velocity_command`: the velocity command the policy follows.

    3 values: `TermInputs.command`, (lin_vel_x, lin_vel_y, ang_vel_z), in
    m/s and rad/s, in the base frame. Isaac Lab gives it as
    `generated_commands` of a base velocity command.

    Takes no joints and no parameters. Raises ValueError if the entry has
    either.
    """
    _check_no_joints(entry)
    _check_params(entry)
    return _VelocityCommand()


def joint_position_relative(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `joint_position_relative`: joint positions relative to the default pose.

    One value per joint of the entry's `joints`, in that order, in rad: the
    joint's position minus its `default_position` in the description. Isaac
    Lab calls it `joint_pos_rel`.

    Takes joints, required, and no parameters. Raises ValueError if the
    entry has no joints, names a joint or group the description does not
    have, or has parameters.
    """
    names = _joint_names(entry, description)
    _check_params(entry)
    defaults = [description.joint(name).default_position for name in names]
    return _JointPositionRelative(
        _indices(names, description), np.array(defaults, dtype=np.float64)
    )


def joint_velocity(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `joint_velocity`: joint velocities.

    One value per joint of the entry's `joints`, in that order, in rad/s:
    the joint's velocity. Isaac Lab calls it `joint_vel`, and its
    `joint_vel_rel` is the same while the default joint velocities are
    zero, as they are unless set otherwise.

    Takes joints, required, and no parameters. Raises ValueError if the
    entry has no joints, names a joint or group the description does not
    have, or has parameters.
    """
    names = _joint_names(entry, description)
    _check_params(entry)
    return _JointVelocity(_indices(names, description))


def last_action(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `last_action`: the policy's previous output.

    One value per joint of the descriptor's `action.joints`, in output
    order: `TermInputs.last_action`, the output as the policy returned it,
    before the action's scale; zeros before the first output. Isaac Lab
    calls it `last_action`.

    Takes no joints and no parameters. Raises ValueError if the entry has
    either, or if the action names a joint or group the description does
    not have.
    """
    _check_no_joints(entry)
    _check_params(entry)
    try:
        driven = resolve_joints(descriptor.action.joints, description)
    except ValueError as error:
        raise ValueError(
            f"observation term {entry.term!r}: action.joints: {error}"
        ) from error
    return _LastAction(len(driven))


def gait_phase(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `gait_phase`: a clock for the gait, as the sine and cosine of its phase.

    2 values, (sin 2πp, cos 2πp), where the phase
    p = (elapsed_s mod period) / period runs from 0 to 1 over each period.
    They are (0, 1) when the policy starts and (1, 0) a quarter period
    later. unitree_rl_gym's G1 walking task computes them this way.

    Takes no joints, and one parameter, required: `period`, the gait
    period, a positive number of seconds. Raises ValueError if the entry
    has joints, has no `period`, has a `period` that is not a positive
    number, or has another parameter.
    """
    _check_no_joints(entry)
    _check_params(entry, "period")
    return _GaitPhase(_positive_number(entry, "period", "the gait period, in s"))


# The terms the built-in factories return. Each is called on every tick,
# so everything that does not change between ticks is computed by its
# factory.


@dataclass(frozen=True, slots=True, eq=False)
class _BaseAngularVelocity:
    imu_to_base: NDArray[np.float64]
    size: int = field(default=3, init=False)

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return self.imu_to_base @ inputs.state.imu.angular_velocity


@dataclass(frozen=True, slots=True, eq=False)
class _ProjectedGravity:
    imu_to_base: NDArray[np.float64]
    size: int = field(default=3, init=False)

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return self.imu_to_base @ _gravity_in_imu(inputs.state.imu.orientation)


@dataclass(frozen=True, slots=True, eq=False)
class _VelocityCommand:
    size: int = field(default=3, init=False)

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.command


@dataclass(frozen=True, slots=True, eq=False)
class _JointPositionRelative:
    indices: NDArray[np.intp]
    defaults: NDArray[np.float64]

    @property
    def size(self) -> int:
        return self.indices.size

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.state.position[self.indices] - self.defaults


@dataclass(frozen=True, slots=True, eq=False)
class _JointVelocity:
    indices: NDArray[np.intp]

    @property
    def size(self) -> int:
        return self.indices.size

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.state.velocity[self.indices]


@dataclass(frozen=True, slots=True, eq=False)
class _LastAction:
    size: int

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.last_action


@dataclass(frozen=True, slots=True, eq=False)
class _GaitPhase:
    period: float
    size: int = field(default=2, init=False)

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        angle = 2.0 * math.pi * (inputs.elapsed_s % self.period) / self.period
        return np.array([math.sin(angle), math.cos(angle)])


@dataclass(frozen=True, slots=True, eq=False)
class _Entry:
    """A built term, with what the builder needs to place its values."""

    name: str
    where: str
    term: Term
    size: int
    scale: NDArray[np.float64]
    start: int


def _gravity_in_imu(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the world's (0, 0, -1) in the frame of orientation `q`, (x, y, z, w).

    It equals `rotate_vector_inverse(q, (0, 0, -1))`: minus the last row of
    the rotation matrix of `q`, which is unit length (ImuState normalizes
    it). Written out, it costs a few multiplications rather than a general
    rotation, and a non-finite `q` gives non-finite values rather than an
    exception, so bad data reaches the safety layer as data.
    """
    x, y, z, w = q
    return np.array(
        [2.0 * (w * y - x * z), -2.0 * (y * z + w * x), 2.0 * (x * x + y * y) - 1.0]
    )


def _imu_to_base(
    entry: ObservationTerm, description: RobotDescription
) -> NDArray[np.float64]:
    """Return the matrix that maps vectors from the IMU frame to the base frame.

    Its columns are the IMU's axes in the base frame, computed with
    `rotate_vector`, so `matrix @ v` equals `rotate_vector(mount, v)`.
    """
    imu = description.imu
    if imu.parent != description.base_link:
        raise ValueError(
            f"observation term {entry.term!r} reads the IMU, which is mounted on"
            f" {imu.parent!r}, not on the base link {description.base_link!r}: its"
            " readings can be expressed in the base frame through its mount only"
            " if it is fixed to the base"
        )
    mount = quaternion_from_rpy(*imu.mount_rpy)
    return np.column_stack([rotate_vector(mount, axis) for axis in np.eye(3)])


def _check_no_joints(entry: ObservationTerm) -> None:
    if entry.joints is not None:
        raise ValueError(
            f"observation term {entry.term!r} takes no joints, but joints is"
            f" {_selection(entry.joints)}"
        )


def _joint_names(
    entry: ObservationTerm, description: RobotDescription
) -> tuple[str, ...]:
    """Return the names of the joints the entry selects, in its order."""
    if entry.joints is None:
        raise ValueError(
            f"observation term {entry.term!r} needs joints: a joint group of the"
            " description, or a list of joint names"
        )
    try:
        return resolve_joints(entry.joints, description)
    except ValueError as error:
        raise ValueError(f"observation term {entry.term!r}: {error}") from error


def _indices(names: tuple[str, ...], description: RobotDescription) -> NDArray[np.intp]:
    """Return the positions of the named joints in the description's joint arrays."""
    return np.array([description.index_of(name) for name in names], dtype=np.intp)


def _check_params(entry: ObservationTerm, *allowed: str) -> None:
    """Check that the entry's parameters are all among `allowed`."""
    unknown = [name for name in entry.params if name not in allowed]
    if not unknown:
        return
    names = ", ".join(repr(name) for name in unknown)
    noun = "parameter" if len(unknown) == 1 else "parameters"
    takes = ", ".join(repr(name) for name in allowed) or "no parameters"
    raise ValueError(
        f"observation term {entry.term!r} has unknown {noun} {names}; it takes {takes}"
    )


def _positive_number(entry: ObservationTerm, name: str, meaning: str) -> float:
    """Return the entry's parameter `name`, which must be a positive number."""
    if name not in entry.params:
        raise ValueError(
            f"observation term {entry.term!r} needs parameter {name!r}: {meaning}"
        )
    value = entry.params[name]
    # bool is a subclass of int, but `period: true` is a mistake, not a number.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"observation term {entry.term!r}: parameter {name!r} must be a number,"
            f" got {value!r}"
        )
    if value <= 0:
        raise ValueError(
            f"observation term {entry.term!r}: parameter {name!r} must be positive,"
            f" got {value!r}"
        )
    return float(value)


def _scale(
    scale: float | tuple[float, ...], size: int, where: str
) -> NDArray[np.float64]:
    """Return the entry's scale as one factor per value of its term."""
    if isinstance(scale, tuple):
        if len(scale) != size:
            raise ValueError(
                f"{where}: scale has {len(scale)} elements, but the term has"
                f" {size} values"
            )
        return np.array(scale, dtype=np.float64)
    return np.full(size, scale, dtype=np.float64)


def _layout(sizes: list[int], length: int, layout: str) -> NDArray[np.intp]:
    """Return where each value of the input comes from in the stored frames.

    The frames are stored as one array of `length` rows, oldest first, each
    a frame whose terms have `sizes`. Element i of the result is the index,
    in that array flattened, of the input's value i.
    """
    frame_size = sum(sizes)
    frames = np.arange(length * frame_size, dtype=np.intp).reshape(length, frame_size)
    if layout == "per_frame":
        return frames.reshape(-1)
    pieces: list[NDArray[np.intp]] = []
    start = 0
    for size in sizes:
        # This term's columns in every frame, row by row: oldest to newest.
        pieces.append(frames[:, start : start + size].reshape(-1))
        start += size
    return np.concatenate(pieces)


def _selection(selection: str | tuple[str, ...]) -> str:
    """Return a joint selection as written in a descriptor."""
    if isinstance(selection, str):
        return repr(selection)
    return "[" + ", ".join(selection) + "]"
