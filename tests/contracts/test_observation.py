"""Tests of the observation term interface."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.typing import NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import ObservationTerm, PolicyDescriptor
from embodiverse.contracts.observation import Term, TermFactory, TermInputs
from embodiverse.contracts.types import ImuState, Mode, RobotState


def state(joints: int = 2) -> RobotState:
    imu = ImuState(
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.array([0.0, 0.0, 9.81]),
    )
    zeros = np.zeros(joints)
    return RobotState(
        stamp_ns=0,
        position=zeros,
        velocity=zeros,
        effort=zeros,
        imu=imu,
        mode=Mode.POLICY,
    )


def inputs(**changes: object) -> TermInputs:
    fields: dict[str, object] = {
        "state": state(),
        "command": np.array([0.5, 0.0, 0.25]),
        "last_action": np.zeros(2),
        "elapsed_s": 0.0,
    }
    fields.update(changes)
    return TermInputs(**fields)  # pyright: ignore[reportArgumentType]


class Constant:
    """A term that returns the same values on every tick."""

    def __init__(self, values: list[float]) -> None:
        self.values = np.array(values)

    @property
    def size(self) -> int:
        return self.values.size

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return self.values


def constant(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    return Constant([1.0, 2.0])


def test_arrays_are_read_only_float64_copies() -> None:
    command = np.array([1, 0, 0])
    last_action = [0.5, -0.5]
    built = inputs(command=command, last_action=last_action, elapsed_s=2)
    command[0] = 7
    assert built.command.dtype == np.float64
    assert built.command.tolist() == [1.0, 0.0, 0.0]
    assert built.last_action.tolist() == [0.5, -0.5]
    assert built.elapsed_s == 2.0
    assert isinstance(built.elapsed_s, float)
    for array in (built.command, built.last_action):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1.0


def test_fields_cannot_be_reassigned() -> None:
    built = inputs()
    with pytest.raises(FrozenInstanceError):
        built.elapsed_s = 1.0  # pyright: ignore[reportAttributeAccessIssue]


def test_non_finite_values_are_kept_as_data() -> None:
    built = inputs(command=[np.nan, 0.0, 0.0], elapsed_s=np.inf)
    assert np.isnan(built.command[0])
    assert built.elapsed_s == np.inf


def test_the_command_has_three_elements() -> None:
    with pytest.raises(ValueError, match="^command must have 3 elements, got 2$"):
        inputs(command=[0.5, 0.0])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(
            "command",
            np.zeros((1, 3)),
            r"^command must be one-dimensional, got shape \(1, 3\)$",
            id="command",
        ),
        pytest.param(
            "last_action",
            np.zeros((2, 2)),
            r"^last_action must be one-dimensional, got shape \(2, 2\)$",
            id="last action",
        ),
    ],
)
def test_arrays_are_one_dimensional(field: str, value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inputs(**{field: value})


def test_a_term_is_anything_with_a_size_that_can_be_called() -> None:
    term = Constant([1.0, 2.0])
    assert isinstance(term, Term)
    assert term.size == 2
    assert term(inputs()).tolist() == [1.0, 2.0]
    assert not isinstance(lambda inputs: np.zeros(3), Term)  # no size
    assert not isinstance(np.zeros(3), Term)  # a size, but not callable


def test_a_factory_is_a_callable_that_builds_a_term() -> None:
    factory: TermFactory = constant
    entry = ObservationTerm(term="constant")
    descriptor = PolicyDescriptor.model_construct()
    description = RobotDescription.model_construct()
    assert factory(entry, description, descriptor).size == 2
