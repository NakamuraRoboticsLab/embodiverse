"""Observation terms: the interface for computing a policy's inputs.

A policy descriptor lists the terms of the policy's input by name, in input
order: `base_angular_velocity`, `joint_position_relative` on the legs, and
so on. Each name is an entry point in the `embodiverse.observations` group
(see `embodiverse.contracts.registry`) that names a `TermFactory`. The
observation builder calls the factory once per entry of the descriptor's
`observations`, to build a `Term`, and calls the term on every control tick
with that tick's `TermInputs`. The builder, not the term, applies the
entry's `scale`, concatenates the terms into a frame and keeps the history.

The interface lives in the contracts so that a package providing terms
depends on nothing else. The built-in terms are in
`embodiverse.runtime.observations` and are registered like any plugin.
docs/policies.md defines each built-in term and shows how to write one.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import ObservationTerm, PolicyDescriptor
from embodiverse.contracts.types import RobotState

__all__ = ["Term", "TermFactory", "TermInputs"]


@dataclass(frozen=True, slots=True, eq=False)
class TermInputs:
    """Everything a term may read on one control tick.

    Like the runtime data in `embodiverse.contracts.types`, the
    arrays are read-only float64 copies made on construction, and
    construction checks their structure only, so non-finite values pass
    through as data. Raises ValueError if `command` does not have three
    elements or `last_action` is not one-dimensional.
    """

    state: RobotState
    """The robot's state on this tick."""

    command: NDArray[np.float64]
    """The velocity command, (lin_vel_x, lin_vel_y, ang_vel_z), in m/s and
    rad/s, in the base frame: x forward, y left, z up."""

    last_action: NDArray[np.float64]
    """The policy's previous output, as the policy returned it, before any
    scaling: one value per joint the action drives, in output order. Zeros
    before the policy's first output."""

    elapsed_s: float
    """Time since the policy started, in s."""

    def __post_init__(self) -> None:
        _store(self, "command", _vector(self.command, "command", 3))
        _store(self, "last_action", _vector(self.last_action, "last_action"))
        object.__setattr__(self, "elapsed_s", float(self.elapsed_s))


@runtime_checkable
class Term(Protocol):
    """One term of a policy's input, built by a `TermFactory` for one entry.

    The observation builder calls the term once per control tick, from the
    control thread. The term returns its values for that tick, unscaled,
    with exactly `size` elements; the builder checks the count and applies
    the entry's `scale`. A term's values should depend on its inputs alone:
    `ObservationBuilder.reset` restarts the history, not the terms, so
    anything that changes over time, such as a gait phase, is computed from
    `TermInputs.elapsed_s`.
    """

    @property
    def size(self) -> int:
        """Number of values the term returns; fixed when the term is built."""
        ...

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        """Return the term's `size` values for this tick, unscaled.

        The builder copies the values, so the result may be one of the
        read-only arrays of `inputs`.
        """
        ...


TermFactory = Callable[[ObservationTerm, RobotDescription, PolicyDescriptor], Term]
"""Builds the `Term` for one entry of a policy descriptor's `observations`.

An entry point in the `embodiverse.observations` group names one, and its
name is the term's name in descriptors. The observation builder calls it
once, when it is created, with the entry, the description of the robot the
policy runs on, and the whole descriptor, for terms that depend on other
fields, such as the action's joints. The factory resolves the entry's
`joints` with `resolve_joints`, checks its `params`, and does once whatever
does not change between ticks, such as finding joint indices. It raises
ValueError, naming the term, for an unknown or missing parameter, a joint
selection the term does not take or needs, or a joint the robot does not
have.
"""


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
