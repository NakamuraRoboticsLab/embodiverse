import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.types import (
    Fault,
    ImuState,
    JointCommand,
    Mode,
    RobotState,
    check_command,
    check_state,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/robots/minimal/description.yaml"


def valid(joints: int = 2) -> tuple[RobotState, JointCommand]:
    """Return a valid state and command for a robot with `joints` joints."""
    zeros = np.zeros(joints)
    imu = ImuState(
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.array([0.0, 0.0, 9.81]),
    )
    state = RobotState(
        stamp_ns=1_000_000,
        position=zeros,
        velocity=zeros,
        effort=zeros,
        imu=imu,
        mode=Mode.POLICY,
        temperature=np.full(joints, 30.0),
    )
    command = JointCommand(
        position=zeros,
        velocity=zeros,
        effort=zeros,
        kp=np.full(joints, 150.0),
        kd=np.full(joints, 4.0),
    )
    return state, command


@pytest.fixture
def description() -> RobotDescription:
    return load_description(FIXTURE)


def test_mode_values_are_lowercase_names() -> None:
    assert [mode.value for mode in Mode] == ["passive", "hold", "policy", "damp"]
    assert all(mode == mode.name.lower() for mode in Mode)
    assert Mode("hold") is Mode.HOLD


def test_fault_joint_defaults_to_none() -> None:
    fault = Fault(code="state.stale", message="no state for 20 ms")
    assert fault.joint is None


def test_arrays_are_converted_to_float64() -> None:
    state, command = valid()
    state = replace(state, position=[1, 2], velocity=np.array([3, 4]))
    imu = replace(state.imu, angular_velocity=[0, 0, 1])
    command = replace(command, kp=np.array([150, 150], dtype=np.int32))
    for array in (state.position, state.velocity, imu.angular_velocity, command.kp):
        assert array.dtype == np.float64
    assert state.position.tolist() == [1.0, 2.0]
    assert command.kp.tolist() == [150.0, 150.0]


def test_arrays_are_copies() -> None:
    source = np.zeros(2)
    vector = np.zeros(3)
    state, command = valid()
    state = replace(state, position=source, temperature=source)
    imu = replace(state.imu, angular_velocity=vector)
    command = replace(command, kp=source)
    source[0] = 1.0
    vector[0] = 1.0
    assert state.position[0] == 0.0
    assert state.temperature is not None
    assert state.temperature[0] == 0.0
    assert imu.angular_velocity[0] == 0.0
    assert command.kp[0] == 0.0


def test_arrays_are_read_only() -> None:
    state, command = valid()
    assert state.temperature is not None
    arrays = [
        state.position,
        state.velocity,
        state.effort,
        state.temperature,
        state.imu.orientation,
        state.imu.angular_velocity,
        state.imu.linear_acceleration,
        command.position,
        command.velocity,
        command.effort,
        command.kp,
        command.kd,
    ]
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 1.0


def test_fields_cannot_be_reassigned() -> None:
    state, command = valid()
    fault = Fault(code="state.stale", message="no state for 20 ms")
    fields: list[tuple[object, str]] = [
        (state, "stamp_ns"),
        (state, "position"),
        (state, "mode"),
        (state, "faults"),
        (state.imu, "orientation"),
        (command, "kp"),
        (fault, "code"),
    ]
    for obj, field in fields:
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)


def test_instances_have_no_dict() -> None:
    state, command = valid()
    fault = Fault(code="state.stale", message="no state for 20 ms")
    for obj in (state, state.imu, command, fault):
        assert not hasattr(obj, "__dict__")


def test_instances_compare_by_identity() -> None:
    state, _ = valid()
    assert state == state
    assert state != replace(state)


def test_faults_compare_by_value() -> None:
    first = Fault(code="state.stale", message="no state for 20 ms")
    second = Fault(code="state.stale", message="no state for 20 ms")
    assert first == second
    assert len({first, second}) == 1


def test_faults_are_stored_as_a_tuple() -> None:
    fault = Fault(code="motor.over_temperature", message="hot", joint="left_knee")
    state, _ = valid()
    state = replace(state, faults=[fault])
    assert state.faults == (fault,)


def test_orientation_is_normalized() -> None:
    state, _ = valid()
    imu = replace(state.imu, orientation=[0.0, 0.0, 2.0, 2.0])
    assert np.allclose(imu.orientation, [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)])


def test_orientation_of_zero_length_is_rejected() -> None:
    state, _ = valid()
    with pytest.raises(ValueError, match="^orientation: quaternion has zero length$"):
        replace(state.imu, orientation=[0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "position",
            np.zeros((2, 1)),
            r"position must be one-dimensional, got shape \(2, 1\)",
        ),
        ("effort", 0.0, r"effort must be one-dimensional, got shape \(\)"),
        ("velocity", np.zeros(3), "velocity has length 3 but position has length 2"),
        ("effort", np.zeros(1), "effort has length 1 but position has length 2"),
        ("temperature", np.zeros((2, 2)), "temperature must be one-dimensional"),
    ],
)
def test_state_rejects_malformed_arrays(field: str, value: Any, match: str) -> None:
    state, _ = valid()
    with pytest.raises(ValueError, match=f"^{match}"):
        replace(state, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("kp", [[150.0, 150.0]], r"kp must be one-dimensional, got shape \(1, 2\)"),
        ("kd", np.zeros(3), "kd has length 3 but position has length 2"),
        ("effort", np.zeros(1), "effort has length 1 but position has length 2"),
    ],
)
def test_command_rejects_malformed_arrays(field: str, value: Any, match: str) -> None:
    _, command = valid()
    with pytest.raises(ValueError, match=f"^{match}"):
        replace(command, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("orientation", [0.0, 0.0, 1.0], "orientation must have 4 elements, got 3"),
        (
            "angular_velocity",
            np.zeros(4),
            "angular_velocity must have 3 elements, got 4",
        ),
        (
            "linear_acceleration",
            np.zeros(2),
            "linear_acceleration must have 3 elements",
        ),
        (
            "linear_acceleration",
            [[0.0, 0.0, 9.81]],
            r"linear_acceleration must be one-dimensional, got shape \(1, 3\)",
        ),
    ],
)
def test_imu_rejects_malformed_vectors(field: str, value: Any, match: str) -> None:
    state, _ = valid()
    with pytest.raises(ValueError, match=f"^{match}"):
        replace(state.imu, **{field: value})


def test_construction_accepts_non_finite_values_and_negative_gains() -> None:
    state, command = valid()
    imu = replace(state.imu, orientation=[math.nan, 0.0, 0.0, 1.0])
    state = replace(state, position=[math.nan, 0.0], imu=imu)
    command = replace(command, effort=[math.inf, 0.0], kp=[-1.0, 150.0])
    assert math.isnan(state.position[0])
    assert math.isnan(state.imu.orientation[0])
    assert math.isinf(command.effort[0])
    assert command.kp[0] == -1.0


def test_valid_objects_pass_their_checks(description: RobotDescription) -> None:
    state, command = valid()
    check_state(state, description)
    check_state(replace(state, temperature=None), description)
    check_command(command, description)
    check_command(replace(command, kp=np.zeros(2), kd=np.zeros(2)), description)


def test_checks_reject_arrays_of_the_wrong_length(
    description: RobotDescription,
) -> None:
    state, command = valid(joints=3)
    match = "^position has length 3 but robot 'minimal' has 2 joints$"
    with pytest.raises(ValueError, match=match):
        check_state(state, description)
    with pytest.raises(ValueError, match=match):
        check_command(command, description)


def test_check_state_rejects_temperature_of_the_wrong_length(
    description: RobotDescription,
) -> None:
    state, _ = valid()
    state = replace(state, temperature=[30.0, 30.0, 30.0])
    match = "^temperature has length 3 but robot 'minimal' has 2 joints$"
    with pytest.raises(ValueError, match=match):
        check_state(state, description)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["position", "velocity", "effort", "temperature"])
def test_check_state_rejects_non_finite_joint_values(
    description: RobotDescription, field: str, bad: float
) -> None:
    state, _ = valid()
    state = replace(state, **{field: [0.0, bad]})
    with pytest.raises(
        ValueError, match=f"^{field} is not finite for joint 'right_knee'$"
    ):
        check_state(state, description)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("orientation", [0.0, 0.0, 0.0, 1.0]),
        ("angular_velocity", [0.0, 0.0, 0.0]),
        ("linear_acceleration", [0.0, 0.0, 9.81]),
    ],
)
def test_check_state_rejects_non_finite_imu_values(
    description: RobotDescription, field: str, value: list[float], bad: float
) -> None:
    state, _ = valid()
    imu = replace(state.imu, **{field: [bad, *value[1:]]})
    with pytest.raises(ValueError, match=rf"^imu\.{field} is not finite$"):
        check_state(replace(state, imu=imu), description)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("field", ["position", "velocity", "effort", "kp", "kd"])
def test_check_command_rejects_non_finite_values(
    description: RobotDescription, field: str, bad: float
) -> None:
    _, command = valid()
    command = replace(command, **{field: [bad, 0.0]})
    with pytest.raises(
        ValueError, match=f"^{field} is not finite for joint 'left_knee'$"
    ):
        check_command(command, description)


@pytest.mark.parametrize("field", ["kp", "kd"])
def test_check_command_rejects_negative_gains(
    description: RobotDescription, field: str
) -> None:
    _, command = valid()
    command = replace(command, **{field: [1.0, -1.0]})
    with pytest.raises(
        ValueError, match=f"^{field} is negative for joint 'right_knee'$"
    ):
        check_command(command, description)


def test_checks_name_every_joint_concerned(description: RobotDescription) -> None:
    state, command = valid()
    state = replace(state, velocity=[math.nan, math.inf])
    match = "^velocity is not finite for joints 'left_knee', 'right_knee'$"
    with pytest.raises(ValueError, match=match):
        check_state(state, description)
    command = replace(command, kd=[-4.0, -4.0])
    match = "^kd is negative for joints 'left_knee', 'right_knee'$"
    with pytest.raises(ValueError, match=match):
        check_command(command, description)
