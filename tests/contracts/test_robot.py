import itertools
import math
from pathlib import Path

import pytest

from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.robot import (
    MODE_TRANSITIONS,
    Capabilities,
    Robot,
    Steppable,
    check_transition,
)
from embodiverse.contracts.types import JointCommand, Mode, RobotState

FIXTURE = Path(__file__).parents[1] / "fixtures/robots/minimal/description.yaml"

ALLOWED = {
    (Mode.PASSIVE, Mode.PASSIVE),
    (Mode.PASSIVE, Mode.HOLD),
    (Mode.HOLD, Mode.HOLD),
    (Mode.HOLD, Mode.PASSIVE),
    (Mode.HOLD, Mode.POLICY),
    (Mode.POLICY, Mode.POLICY),
    (Mode.POLICY, Mode.HOLD),
    (Mode.DAMP, Mode.PASSIVE),
}
ALL_PAIRS = list(itertools.product(Mode, repeat=2))


class FullRobot:
    """Has every member of `Robot` and `Steppable`; the bodies are never run."""

    @property
    def description(self) -> RobotDescription:
        return load_description(FIXTURE)

    def capabilities(self) -> Capabilities:
        return Capabilities(effort_command=True, temperature=False, max_rate_hz=500.0)

    def connect(self) -> None: ...

    def read(self) -> RobotState:
        raise NotImplementedError

    def write(self, command: JointCommand) -> None: ...

    def set_mode(self, mode: Mode) -> None: ...

    def operator_stop(self) -> None: ...

    def close(self) -> None: ...

    def advance(self, seconds: float) -> None: ...


class RobotWithoutStop:
    """Has every member of `Robot` except `operator_stop`."""

    @property
    def description(self) -> RobotDescription:
        return load_description(FIXTURE)

    def capabilities(self) -> Capabilities:
        return Capabilities(effort_command=True, temperature=False, max_rate_hz=500.0)

    def connect(self) -> None: ...

    def read(self) -> RobotState:
        raise NotImplementedError

    def write(self, command: JointCommand) -> None: ...

    def set_mode(self, mode: Mode) -> None: ...

    def close(self) -> None: ...


def test_capabilities_keep_their_values() -> None:
    capabilities = Capabilities(effort_command=False, temperature=True, max_rate_hz=50)
    assert not capabilities.effort_command
    assert capabilities.temperature
    assert capabilities.max_rate_hz == 50


@pytest.mark.parametrize("rate", [0.0, -1.0, math.inf, math.nan])
def test_capabilities_reject_a_rate_that_is_not_positive_and_finite(
    rate: float,
) -> None:
    match = f"^max_rate_hz must be a positive finite number, got {rate}$"
    with pytest.raises(ValueError, match=match):
        Capabilities(effort_command=True, temperature=True, max_rate_hz=rate)


def test_mode_transitions_match_the_table() -> None:
    assert dict(MODE_TRANSITIONS) == {
        Mode.PASSIVE: {Mode.HOLD},
        Mode.HOLD: {Mode.PASSIVE, Mode.POLICY},
        Mode.POLICY: {Mode.HOLD},
        Mode.DAMP: {Mode.PASSIVE},
    }
    assert all(isinstance(targets, frozenset) for targets in MODE_TRANSITIONS.values())


def test_mode_transitions_are_read_only() -> None:
    with pytest.raises(TypeError, match="does not support item assignment"):
        MODE_TRANSITIONS[Mode.DAMP] = frozenset()  # pyright: ignore[reportIndexIssue]
    assert MODE_TRANSITIONS[Mode.DAMP] == {Mode.PASSIVE}


@pytest.mark.parametrize(
    ("current", "target"),
    ALL_PAIRS,
    ids=[f"{current.name}-{target.name}" for current, target in ALL_PAIRS],
)
def test_check_transition_accepts_exactly_the_allowed_transitions(
    current: Mode, target: Mode
) -> None:
    if (current, target) in ALLOWED:
        check_transition(current, target)
        return
    if target is Mode.DAMP:
        reason = "DAMP is entered only through operator_stop$"
    else:
        reason = f"{current.name} can change only to "
    match = f"^cannot change mode from {current.name} to {target.name}: {reason}"
    with pytest.raises(ValueError, match=match):
        check_transition(current, target)


def test_check_transition_names_the_reachable_modes() -> None:
    with pytest.raises(ValueError, match="POLICY can change only to HOLD$"):
        check_transition(Mode.POLICY, Mode.PASSIVE)


def test_a_class_with_every_member_is_a_robot() -> None:
    assert isinstance(FullRobot(), Robot)


def test_a_class_without_operator_stop_is_not_a_robot() -> None:
    assert not isinstance(RobotWithoutStop(), Robot)


def test_a_class_with_advance_is_steppable() -> None:
    assert isinstance(FullRobot(), Steppable)


def test_a_class_without_advance_is_not_steppable() -> None:
    assert not isinstance(RobotWithoutStop(), Steppable)
