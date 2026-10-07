"""Tests of the safety layer, on testbot.

testbot's joints, in description order, with their position limits, effort
limits and gains, are in docs/robots/testbot.md. The default thresholds give
a position margin of 0.05 rad, gains capped at twice the description's,
and motors hot from 70 °C.
"""

import ast
import inspect
import math
from collections.abc import Callable
from functools import cache
from importlib.resources import as_file, files
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import ArrayLike
from pydantic import ValidationError

from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.robot import Robot
from embodiverse.contracts.types import (
    Fault,
    ImuState,
    JointCommand,
    Mode,
    RobotState,
    check_command,
)
from embodiverse.runtime import safety
from embodiverse.runtime.safety import (
    COMMAND_STALE,
    FAULT_ACTIONS,
    JOINT_OUT_OF_LIMITS,
    LOOP_OVERRUN,
    MOTOR_HOT,
    POLICY_NON_FINITE,
    STATE_NON_FINITE,
    STATE_STALE,
    Assessment,
    SafetyAction,
    SafetyConfig,
    SafetyLayer,
    SafetyResult,
    most_severe,
)

TESTBOT = "embodiverse.robots.testbot"
JOINTS = (
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
)
DEFAULTS = (-0.3, 0.6, -0.3, -0.3, 0.6, -0.3)
LOWER = np.array([-1.5, 0.0, -0.8, -1.5, 0.0, -0.8])
UPPER = np.array([0.5, 2.2, 0.6, 0.5, 2.2, 0.6])
KP = (60.0, 80.0, 40.0, 60.0, 80.0, 40.0)
KD = (2.0,) * 6
ZEROS = (0.0,) * 6
MARGIN = 0.05

NOW_NS = 10_000_000_000
NAN = math.nan
INF = math.inf

NONE = SafetyAction.NONE
ZERO_VELOCITY = SafetyAction.ZERO_VELOCITY
HOLD = SafetyAction.HOLD


@cache
def robot() -> RobotDescription:
    with as_file(files(TESTBOT).joinpath("description.yaml")) as path:
        return load_description(path)


def layer(**config: Any) -> SafetyLayer:
    return SafetyLayer(robot(), SafetyConfig(**config))


def state(
    position: ArrayLike = DEFAULTS,
    velocity: ArrayLike = ZEROS,
    effort: ArrayLike = ZEROS,
    temperature: ArrayLike | None = (30.0,) * 6,
    orientation: ArrayLike = (0.0, 0.0, 0.0, 1.0),
    angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
    linear_acceleration: ArrayLike = (0.0, 0.0, 9.81),
    faults: tuple[Fault, ...] = (),
    stamp_ns: int = NOW_NS,
) -> RobotState:
    imu = ImuState(
        orientation=np.array(orientation, dtype=np.float64),
        angular_velocity=np.array(angular_velocity, dtype=np.float64),
        linear_acceleration=np.array(linear_acceleration, dtype=np.float64),
    )
    return RobotState(
        stamp_ns=stamp_ns,
        position=np.array(position, dtype=np.float64),
        velocity=np.array(velocity, dtype=np.float64),
        effort=np.array(effort, dtype=np.float64),
        imu=imu,
        mode=Mode.POLICY,
        temperature=None if temperature is None else np.array(temperature),
        faults=faults,
    )


def command(
    position: ArrayLike = DEFAULTS,
    velocity: ArrayLike = ZEROS,
    effort: ArrayLike = ZEROS,
    kp: ArrayLike = KP,
    kd: ArrayLike = KD,
) -> JointCommand:
    return JointCommand(
        position=np.array(position, dtype=np.float64),
        velocity=np.array(velocity, dtype=np.float64),
        effort=np.array(effort, dtype=np.float64),
        kp=np.array(kp, dtype=np.float64),
        kd=np.array(kd, dtype=np.float64),
    )


def check(
    safety_layer: SafetyLayer,
    robot_state: RobotState | None = None,
    *,
    now_ns: int = NOW_NS,
    command_age_s: float = 0.0,
    overrun: bool = False,
) -> Assessment:
    """Check `robot_state`, a healthy state by default, with fresh inputs."""
    return safety_layer.check(
        state() if robot_state is None else robot_state,
        now_ns=now_ns,
        command_age_s=command_age_s,
        overrun=overrun,
    )


def codes(faults: tuple[Fault, ...]) -> list[str]:
    return [fault.code for fault in faults]


def one_joint(index: int, value: float, others: ArrayLike = ZEROS) -> list[float]:
    """Return `others` with element `index` replaced by `value`."""
    values = np.array(others, dtype=np.float64)
    values[index] = value
    return values.tolist()


def assert_command(
    actual: JointCommand,
    position: ArrayLike,
    velocity: ArrayLike = ZEROS,
    effort: ArrayLike = ZEROS,
    kp: ArrayLike = KP,
    kd: ArrayLike = KD,
) -> None:
    assert actual.position.tolist() == np.asarray(position, dtype=float).tolist()
    assert actual.velocity.tolist() == np.asarray(velocity, dtype=float).tolist()
    assert actual.effort.tolist() == np.asarray(effort, dtype=float).tolist()
    assert actual.kp.tolist() == np.asarray(kp, dtype=float).tolist()
    assert actual.kd.tolist() == np.asarray(kd, dtype=float).tolist()


# Actions.


def test_actions_are_ordered_by_severity() -> None:
    # As strings, "hold" < "none" < "zero_velocity"; the actions compare by
    # severity instead.
    assert NONE < ZERO_VELOCITY < HOLD
    assert HOLD > ZERO_VELOCITY > NONE
    assert HOLD >= HOLD and NONE <= NONE
    assert sorted([HOLD, NONE, ZERO_VELOCITY]) == [NONE, ZERO_VELOCITY, HOLD]
    assert max([ZERO_VELOCITY, HOLD, NONE]) is HOLD
    assert [action.severity for action in SafetyAction] == [0, 1, 2]


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        ([], NONE),
        ([NONE], NONE),
        ([NONE, ZERO_VELOCITY, NONE], ZERO_VELOCITY),
        ([ZERO_VELOCITY, HOLD, NONE], HOLD),
        ([HOLD, ZERO_VELOCITY], HOLD),
    ],
)
def test_most_severe(actions: list[SafetyAction], expected: SafetyAction) -> None:
    assert most_severe(actions) is expected
    assert most_severe(iter(actions)) is expected


# The fault table.


def test_the_fault_table() -> None:
    assert dict(FAULT_ACTIONS) == {
        "command.stale": ZERO_VELOCITY,
        "state.stale": HOLD,
        "state.non_finite": HOLD,
        "policy.non_finite": HOLD,
        "loop.overrun": HOLD,
        "motor.hot": ZERO_VELOCITY,
    }


def test_the_fault_table_is_read_only() -> None:
    table: Any = FAULT_ACTIONS
    with pytest.raises(TypeError):
        table[COMMAND_STALE] = HOLD
    assert FAULT_ACTIONS[COMMAND_STALE] is ZERO_VELOCITY


def astuple(
    result: Assessment | SafetyResult,
) -> tuple[tuple[Fault, ...], SafetyAction]:
    return result.faults, result.action


# How each fault is produced, returning what the layer found and the action
# it took. A test checks that every fault of the table is here.
DETECTIONS: dict[
    str, Callable[[SafetyLayer], tuple[tuple[Fault, ...], SafetyAction]]
] = {
    STATE_STALE: lambda safety_layer: astuple(
        check(safety_layer, now_ns=NOW_NS + 60_000_000)
    ),
    STATE_NON_FINITE: lambda safety_layer: astuple(
        check(safety_layer, state(velocity=one_joint(1, NAN)))
    ),
    COMMAND_STALE: lambda safety_layer: astuple(check(safety_layer, command_age_s=0.6)),
    MOTOR_HOT: lambda safety_layer: astuple(
        check(safety_layer, state(temperature=one_joint(4, 75.0, (30.0,) * 6)))
    ),
    LOOP_OVERRUN: lambda safety_layer: astuple(check(safety_layer, overrun=True)),
    POLICY_NON_FINITE: lambda safety_layer: astuple(
        safety_layer.filter(command(position=one_joint(0, NAN, DEFAULTS)), state())
    ),
}


def test_every_fault_in_the_table_has_a_detection_test() -> None:
    assert set(DETECTIONS) == set(FAULT_ACTIONS)


@pytest.mark.parametrize("code", sorted(FAULT_ACTIONS))
def test_every_fault_in_the_table_is_detected_and_takes_its_action(
    code: str,
) -> None:
    faults, action = DETECTIONS[code](layer())
    assert codes(faults) == [code]
    assert action is FAULT_ACTIONS[code]


@pytest.mark.parametrize("code", sorted(FAULT_ACTIONS))
def test_adapter_faults_in_the_table_take_its_action(code: str) -> None:
    fault = Fault(code=code, message="reported by the adapter", joint="left_knee")
    assessment = check(layer(), state(faults=(fault,)))
    assert assessment == Assessment((fault,), FAULT_ACTIONS[code])


# Checking a tick's inputs.


def test_healthy_inputs_raise_no_fault() -> None:
    assert check(layer()) == Assessment((), NONE)


def test_a_stale_state_holds() -> None:
    assessment = check(layer(), now_ns=NOW_NS + 50_000_001)
    assert assessment == Assessment(
        (
            Fault(
                code=STATE_STALE,
                message="the state is 50.0 ms old, more than the 50 ms allowed",
            ),
        ),
        HOLD,
    )


def test_a_state_exactly_as_old_as_the_timeout_is_not_stale() -> None:
    assert check(layer(), now_ns=NOW_NS + 50_000_000) == Assessment((), NONE)


@pytest.mark.parametrize("value", [NAN, INF, -INF])
@pytest.mark.parametrize(
    ("field", "where"),
    [
        ("position", "position for joint 'left_knee'"),
        ("velocity", "velocity for joint 'left_knee'"),
        ("effort", "effort for joint 'left_knee'"),
        ("temperature", "temperature for joint 'left_knee'"),
        ("orientation", "imu.orientation"),
        ("angular_velocity", "imu.angular_velocity"),
        ("linear_acceleration", "imu.linear_acceleration"),
    ],
)
def test_a_non_finite_state_holds(field: str, where: str, value: float) -> None:
    healthy: dict[str, Any] = {
        "position": DEFAULTS,
        "velocity": ZEROS,
        "effort": ZEROS,
        "temperature": (30.0,) * 6,
        "orientation": (0.0, 0.0, 0.0, 1.0),
        "angular_velocity": (0.0, 0.0, 0.0),
        "linear_acceleration": (0.0, 0.0, 9.81),
    }
    changes: dict[str, Any] = {field: one_joint(1, value, healthy[field])}
    broken = state(**changes)
    assessment = check(layer(), broken)
    message = f"the state is not finite: {where}"
    assert assessment == Assessment((Fault(STATE_NON_FINITE, message),), HOLD)


def test_a_non_finite_state_is_one_fault_naming_every_field_and_joint() -> None:
    broken = state(
        position=[NAN, 0.6, -0.3, -0.3, INF, -0.3], orientation=(NAN, 0.0, 0.0, 1.0)
    )
    (fault,) = check(layer(), broken).faults
    assert fault.message == (
        "the state is not finite: position for joints 'left_hip_pitch',"
        " 'right_knee'; imu.orientation"
    )
    assert fault.joint is None


def test_a_stale_velocity_command_stops_the_robot_moving() -> None:
    assessment = check(layer(), command_age_s=0.75)
    message = "the velocity command was last updated 0.75 s ago; the limit is 0.5 s"
    assert assessment == Assessment((Fault(COMMAND_STALE, message),), ZERO_VELOCITY)


def test_a_velocity_command_exactly_as_old_as_the_timeout_is_not_stale() -> None:
    assert check(layer(), command_age_s=0.5) == Assessment((), NONE)


@pytest.mark.parametrize(
    ("age", "message"),
    [
        (INF, "the velocity command has never been updated; the limit is 0.5 s"),
        (NAN, "the velocity command's age is not a number; the limit is 0.5 s"),
    ],
)
def test_a_velocity_command_of_unknown_age_is_stale(age: float, message: str) -> None:
    assessment = check(layer(), command_age_s=age)
    assert assessment == Assessment((Fault(COMMAND_STALE, message),), ZERO_VELOCITY)


def test_a_hot_motor_stops_the_robot_moving_and_names_its_joint() -> None:
    # Hot from 80 - 10 = 70 °C.
    assessment = check(layer(), state(temperature=one_joint(1, 70.0, (30.0,) * 6)))
    message = (
        "the motor of joint 'left_knee' is at 70.0 °C, at or above 70 °C"
        " (the 80 °C limit less a 10 °C margin)"
    )
    assert assessment == Assessment(
        (Fault(MOTOR_HOT, message, joint="left_knee"),), ZERO_VELOCITY
    )


def test_a_motor_just_below_the_threshold_is_not_hot() -> None:
    warm = state(temperature=one_joint(1, 69.99, (30.0,) * 6))
    assert check(layer(), warm) == Assessment((), NONE)


def test_each_hot_motor_is_a_fault_naming_its_joint() -> None:
    temperatures = [30.0, 30.0, 71.0, 30.0, 30.0, 90.0]
    assessment = check(layer(), state(temperature=temperatures))
    assert [(fault.code, fault.joint) for fault in assessment.faults] == [
        (MOTOR_HOT, "left_ankle_pitch"),
        (MOTOR_HOT, "right_ankle_pitch"),
    ]
    assert assessment.action is ZERO_VELOCITY


def test_the_temperature_thresholds_are_configurable() -> None:
    warm = state(temperature=one_joint(0, 55.0, (30.0,) * 6))
    assert check(layer(), warm).faults == ()
    hot = layer(temperature_limit_c=60.0, temperature_margin_c=5.0)
    assert codes(check(hot, warm).faults) == [MOTOR_HOT]


def test_a_robot_without_temperatures_has_no_hot_motors() -> None:
    assert check(layer(), state(temperature=None)) == Assessment((), NONE)


def test_an_overrun_holds() -> None:
    assessment = check(layer(), overrun=True)
    message = "the previous control tick took longer than its period"
    assert assessment == Assessment((Fault(LOOP_OVERRUN, message),), HOLD)


@pytest.mark.parametrize("action", [ZERO_VELOCITY, HOLD])
def test_unknown_adapter_faults_take_the_configured_action(
    action: SafetyAction,
) -> None:
    fault = Fault(code="simulation.stopped", message="a physics step failed")
    assessment = check(layer(unknown_fault_action=action), state(faults=(fault,)))
    assert assessment == Assessment((fault,), action)


def test_unknown_adapter_faults_hold_by_default() -> None:
    fault = Fault(code="motor.encoder_lost", message="no encoder", joint="left_knee")
    assert check(layer(), state(faults=(fault,))) == Assessment((fault,), HOLD)


# The most severe action wins.


def test_a_hot_motor_and_a_stale_command_stop_the_robot_moving() -> None:
    hot = state(temperature=one_joint(0, 75.0, (30.0,) * 6))
    assessment = check(layer(), hot, command_age_s=1.0)
    assert codes(assessment.faults) == [COMMAND_STALE, MOTOR_HOT]
    assert assessment.action is ZERO_VELOCITY


def test_a_fault_that_holds_wins_over_one_that_stops() -> None:
    assessment = check(layer(), command_age_s=1.0, overrun=True)
    assert codes(assessment.faults) == [COMMAND_STALE, LOOP_OVERRUN]
    assert assessment.action is HOLD


def test_an_adapter_fault_that_holds_wins_over_one_that_stops() -> None:
    faults = (
        Fault(code=MOTOR_HOT, message="hot", joint="left_knee"),
        Fault(code="power.low", message="battery low"),
    )
    assessment = check(layer(), state(faults=faults))
    assert assessment == Assessment(faults, HOLD)


def test_every_fault_is_reported_in_order() -> None:
    adapter = Fault(code="power.low", message="battery low")
    broken = state(
        velocity=one_joint(0, NAN),
        temperature=one_joint(5, 99.0, (30.0,) * 6),
        faults=(adapter,),
    )
    assessment = check(
        layer(), broken, now_ns=NOW_NS + 10**9, command_age_s=INF, overrun=True
    )
    assert codes(assessment.faults) == [
        STATE_STALE,
        STATE_NON_FINITE,
        COMMAND_STALE,
        MOTOR_HOT,
        LOOP_OVERRUN,
        "power.low",
    ]
    assert assessment.faults[-1] is adapter
    assert assessment.action is HOLD


# Latching.


def test_hold_is_latched_until_reset() -> None:
    safety_layer = layer()
    stale = check(safety_layer, now_ns=NOW_NS + 10**9)
    assert stale.action is HOLD
    assert safety_layer.latched
    assert safety_layer.latched_by == stale.faults

    # The cause has cleared, but HOLD stays, and the policy's commands are
    # replaced by the hold command.
    assert check(safety_layer) == Assessment((), HOLD)
    result = safety_layer.filter(command(), state())
    assert result.faults == ()
    assert result.action is HOLD
    assert result.command is safety_layer.hold_command(state())

    safety_layer.reset()
    assert not safety_layer.latched
    assert safety_layer.latched_by == ()
    assert check(safety_layer) == Assessment((), NONE)
    assert safety_layer.filter(command(), state()).action is NONE


def test_a_non_finite_command_latches_hold() -> None:
    safety_layer = layer()
    result = safety_layer.filter(command(kd=one_joint(2, INF, KD)), state())
    assert result.action is HOLD
    assert safety_layer.latched_by == result.faults
    assert check(safety_layer) == Assessment((), HOLD)
    safety_layer.reset()
    assert check(safety_layer) == Assessment((), NONE)


def test_an_adapter_fault_that_holds_latches() -> None:
    safety_layer = layer()
    fault = Fault(code="power.low", message="battery low")
    assert check(safety_layer, state(faults=(fault,))).action is HOLD
    assert check(safety_layer) == Assessment((), HOLD)


def test_zero_velocity_is_not_latched() -> None:
    safety_layer = layer()
    assert check(safety_layer, command_age_s=1.0).action is ZERO_VELOCITY
    assert check(safety_layer, command_age_s=0.0) == Assessment((), NONE)

    hot = state(temperature=one_joint(0, 75.0, (30.0,) * 6))
    assert check(safety_layer, hot).action is ZERO_VELOCITY
    assert not safety_layer.latched
    assert safety_layer.filter(command(), hot).action is NONE
    assert check(safety_layer) == Assessment((), NONE)


def test_the_latch_names_only_the_faults_that_hold() -> None:
    safety_layer = layer()
    assessment = check(safety_layer, command_age_s=1.0, overrun=True)
    assert safety_layer.latched_by == (assessment.faults[1],)
    assert codes(safety_layer.latched_by) == [LOOP_OVERRUN]


def test_a_later_hold_does_not_change_what_latched_it() -> None:
    safety_layer = layer()
    first = check(safety_layer, overrun=True)
    second = check(safety_layer, now_ns=NOW_NS + 10**9)
    assert codes(second.faults) == [STATE_STALE]
    assert second.action is HOLD
    assert safety_layer.latched_by == first.faults


# Non-finite commands and the hold command.


@pytest.mark.parametrize("value", [NAN, INF, -INF])
@pytest.mark.parametrize("field", ["position", "velocity", "effort", "kp", "kd"])
def test_a_non_finite_command_is_replaced_by_the_hold_command(
    field: str, value: float
) -> None:
    safety_layer = layer()
    targets = (-0.2, 0.7, -0.1, -0.4, 0.5, -0.2)
    gains = (50.0, 70.0, 30.0, 50.0, 70.0, 30.0)
    safe = command(position=targets, velocity=(0.1,) * 6, kp=gains)
    assert safety_layer.filter(safe, state()).action is NONE

    healthy: dict[str, Any] = {
        "position": targets,
        "velocity": ZEROS,
        "effort": ZEROS,
        "kp": gains,
        "kd": KD,
    }
    broken = command(**{**healthy, field: one_joint(3, value, healthy[field])})
    result = safety_layer.filter(broken, state())
    message = f"the policy's command is not finite: {field} for joint 'right_hip_pitch'"
    assert result.faults == (Fault(POLICY_NON_FINITE, message),)
    assert result.action is HOLD
    assert_command(result.command, targets, kp=gains)
    assert result.command is safety_layer.hold_command(state())


def test_a_non_finite_command_is_one_fault_naming_every_field_and_joint() -> None:
    broken = command(position=one_joint(1, NAN, DEFAULTS), kd=[INF, 2, 2, 2, 2, NAN])
    (fault,) = layer().filter(broken, state()).faults
    assert fault.message == (
        "the policy's command is not finite: position for joint 'left_knee'; kd for"
        " joints 'left_hip_pitch', 'right_ankle_pitch'"
    )
    assert fault.joint is None


def test_the_first_hold_command_holds_the_measured_positions_inside_the_limits() -> (
    None
):
    measured = [
        0.6,  # left_hip_pitch, above its upper limit, 0.5
        -0.1,  # left_knee, below its lower limit, 0.0
        -0.2,  # left_ankle_pitch, inside its limits
        -1.48,  # right_hip_pitch, inside its limits but within the margin
        2.2,  # right_knee, at its upper limit
        NAN,  # right_ankle_pitch, not measured: its default position
    ]
    hold = layer().hold_command(state(position=measured))
    assert_command(
        hold, [0.5 - MARGIN, 0.0 + MARGIN, -0.2, -1.5 + MARGIN, 2.2 - MARGIN, -0.3]
    )
    check_command(hold, robot())


def test_the_hold_command_holds_the_last_safe_targets_with_their_gains() -> None:
    safety_layer = layer()
    targets = (-0.2, 0.7, -0.1, -0.4, 0.5, -0.2)
    gains = (50.0, 70.0, 30.0, 50.0, 70.0, 30.0)
    damping = (1.0, 1.5, 1.0, 1.0, 1.5, 1.0)
    safe = command(
        targets, velocity=(0.3,) * 6, effort=(2.0,) * 6, kp=gains, kd=damping
    )
    safety_layer.filter(safe, state())
    hold = safety_layer.hold_command(state(position=ZEROS))
    assert_command(hold, targets, kp=gains, kd=damping)


def test_the_hold_targets_do_not_follow_the_robot_if_it_sags() -> None:
    safety_layer = layer()
    first = safety_layer.hold_command(state(position=DEFAULTS))
    sagged = [value - 0.1 for value in DEFAULTS]
    second = safety_layer.hold_command(state(position=sagged))
    assert second is first
    assert second.position.tolist() == list(DEFAULTS)


def test_reset_forgets_the_last_safe_command() -> None:
    safety_layer = layer()
    safety_layer.filter(command(position=(-0.2,) * 3 + (-0.4,) * 3), state())
    safety_layer.reset()
    hold = safety_layer.hold_command(state(position=DEFAULTS))
    assert_command(hold, DEFAULTS)


# Clamping.


def test_a_command_inside_the_limits_passes_unchanged_as_a_new_command() -> None:
    safety_layer = layer()
    original = command(
        position=(-0.2, 0.7, 0.0, -0.4, 1.5, 0.3),
        velocity=(1.0, -2.0, 0.5, 0.0, 19.0, -19.0),
        effort=(10.0, -20.0, 5.0, -59.0, 79.0, 0.0),
        kp=(120.0, 160.0, 80.0, 0.0, 100.0, 40.0),
        kd=(4.0, 3.0, 0.0, 2.0, 1.0, 4.0),
    )
    result = safety_layer.filter(original, state())
    assert result.faults == ()
    assert result.action is NONE
    assert result.command is not original
    assert_command(
        result.command,
        original.position,
        original.velocity,
        original.effort,
        original.kp,
        original.kd,
    )
    assert set(safety_layer.stats.clamps.values()) == {0}


def test_position_targets_are_kept_inside_the_limits_less_the_margin() -> None:
    targets = [0.6, -0.1, 0.0, -1.5, 2.2, -2.0]
    result = layer().filter(command(position=targets), state())
    assert result.faults == ()
    assert result.action is NONE
    assert result.command.position.tolist() == [
        0.5 - MARGIN,  # high
        0.0 + MARGIN,  # low
        0.0,  # inside
        -1.5 + MARGIN,  # at the lower limit
        2.2 - MARGIN,  # at the upper limit
        -0.8 + MARGIN,  # far below
    ]


def test_the_position_margin_is_configurable() -> None:
    result = layer(position_margin_rad=0.0).filter(
        command(position=[0.6, -0.1, 0.0, -1.5, 2.2, -2.0]), state()
    )
    assert result.command.position.tolist() == [0.5, 0.0, 0.0, -1.5, 2.2, -0.8]


def test_a_range_narrower_than_twice_the_margin_clamps_to_its_midpoint() -> None:
    file = files(TESTBOT).joinpath("description.yaml")
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    data["joints"][2].update(
        default_position=0.03,
        limits={"position": [0.0, 0.08], "velocity": 20.0, "effort": 30.0},
    )
    safety_layer = SafetyLayer(RobotDescription.model_validate(data))
    for target in (-0.5, 0.0, 0.03, 0.04, 0.08, 0.5):
        targets = one_joint(2, target, DEFAULTS)
        result = safety_layer.filter(command(position=targets), state())
        assert result.command.position[2] == (0.0 + 0.08) / 2
    hold = SafetyLayer(RobotDescription.model_validate(data)).hold_command(state())
    assert hold.position[2] == 0.04


def test_gains_are_capped_at_twice_the_description_s() -> None:
    # The caps are 2 × (60, 80, 40, 60, 80, 40) for kp and 2 × 2 for kd.
    result = layer().filter(
        command(
            kp=(200.0, 100.0, 80.0, 121.0, 160.0, 0.0),
            kd=(5.0, 4.0, 1.0, 4.5, 0.0, 100.0),
        ),
        state(),
    )
    assert result.faults == ()
    assert result.command.kp.tolist() == [120.0, 100.0, 80.0, 120.0, 160.0, 0.0]
    assert result.command.kd.tolist() == [4.0, 4.0, 1.0, 4.0, 0.0, 4.0]


def test_the_gain_ratio_is_configurable() -> None:
    result = layer(gain_ratio_max=1.0).filter(
        command(kp=(200.0,) * 6, kd=(3.0,) * 6), state()
    )
    assert result.command.kp.tolist() == list(KP)
    assert result.command.kd.tolist() == list(KD)


def test_effort_is_clipped_to_the_effort_limits() -> None:
    # The limits are 60, 80, 30, 60, 80 and 30 N·m.
    result = layer().filter(
        command(effort=(70.0, -90.0, 30.0, -61.0, 0.0, -29.5)), state()
    )
    assert result.faults == ()
    assert result.command.effort.tolist() == [60.0, -80.0, 30.0, -60.0, 0.0, -29.5]


def test_velocity_targets_are_clipped_to_the_velocity_limits() -> None:
    # Every joint's limit is 20 rad/s.
    result = layer().filter(
        command(velocity=(25.0, -25.0, 20.0, -20.0, 1.0, -1e9)), state()
    )
    assert result.command.velocity.tolist() == [20.0, -20.0, 20.0, -20.0, 1.0, -20.0]


def test_clamping_is_not_a_fault_but_is_counted_per_joint() -> None:
    safety_layer = layer()
    # left_hip_pitch: position and kp clamped; left_ankle_pitch: effort.
    clamped = command(
        position=one_joint(0, 0.6, DEFAULTS),
        effort=one_joint(2, 31.0),
        kp=one_joint(0, 500.0, KP),
    )
    for _ in range(2):
        result = safety_layer.filter(clamped, state())
        assert result.faults == ()
        assert result.action is NONE
    safety_layer.filter(command(kd=one_joint(5, 9.0, KD)), state())
    assert dict(safety_layer.stats.clamps) == {
        "left_hip_pitch": 2,
        "left_knee": 0,
        "left_ankle_pitch": 2,
        "right_hip_pitch": 0,
        "right_knee": 0,
        "right_ankle_pitch": 1,
    }
    assert safety_layer.stats.faults == {}


# The start check.


def test_preflight_accepts_a_healthy_state() -> None:
    assert layer().preflight(state()) == ()


def test_preflight_accepts_joints_at_their_limits() -> None:
    assert layer().preflight(state(position=LOWER)) == ()
    assert layer().preflight(state(position=UPPER)) == ()


def test_preflight_refuses_a_hot_motor() -> None:
    faults = layer().preflight(state(temperature=one_joint(4, 70.0, (30.0,) * 6)))
    assert [(fault.code, fault.joint) for fault in faults] == [
        (MOTOR_HOT, "right_knee")
    ]


def test_preflight_refuses_a_non_finite_state() -> None:
    faults = layer().preflight(state(linear_acceleration=(0.0, NAN, 9.81)))
    message = "the state is not finite: imu.linear_acceleration"
    assert faults == (Fault(STATE_NON_FINITE, message),)


@pytest.mark.parametrize(
    "fault",
    [
        Fault(code=MOTOR_HOT, message="hot", joint="left_knee"),
        Fault(code="simulation.stopped", message="a physics step failed"),
    ],
)
def test_preflight_refuses_any_adapter_fault(fault: Fault) -> None:
    assert layer().preflight(state(faults=(fault,))) == (fault,)


@pytest.mark.parametrize(
    ("index", "position", "message"),
    [
        (
            1,
            -0.1,
            "joint 'left_knee' is at -0.1 rad, outside its position limits"
            " [0, 2.2] rad",
        ),
        (
            3,
            0.51,
            "joint 'right_hip_pitch' is at 0.51 rad, outside its position limits"
            " [-1.5, 0.5] rad",
        ),
    ],
)
def test_preflight_refuses_a_joint_outside_its_limits(
    index: int, position: float, message: str
) -> None:
    faults = layer().preflight(state(position=one_joint(index, position, DEFAULTS)))
    assert faults == (Fault(JOINT_OUT_OF_LIMITS, message, joint=JOINTS[index]),)


def test_preflight_reports_every_reason_in_order() -> None:
    adapter = Fault(code="power.low", message="battery low")
    broken = state(
        position=[0.6, 0.6, -0.3, -0.3, 0.6, NAN],
        temperature=one_joint(2, 75.0, (30.0,) * 6),
        faults=(adapter,),
    )
    faults = layer().preflight(broken)
    assert [(fault.code, fault.joint) for fault in faults] == [
        (STATE_NON_FINITE, None),
        (JOINT_OUT_OF_LIMITS, "left_hip_pitch"),
        (MOTOR_HOT, "left_ankle_pitch"),
        ("power.low", None),
    ]


def test_preflight_neither_latches_nor_counts() -> None:
    safety_layer = layer()
    fault = Fault(code="power.low", message="battery low")
    assert safety_layer.preflight(state(faults=(fault,))) == (fault,)
    assert not safety_layer.latched
    assert safety_layer.stats.faults == {}
    assert check(safety_layer) == Assessment((), NONE)


def test_preflight_does_not_check_staleness() -> None:
    assert layer().preflight(state(stamp_ns=0)) == ()


# Statistics.


def test_stats_count_faults_by_code() -> None:
    safety_layer = layer()
    hot = state(temperature=[75.0, 75.0, 30.0, 30.0, 30.0, 30.0])
    check(safety_layer, hot, command_age_s=1.0)
    check(safety_layer, command_age_s=1.0)
    safety_layer.filter(command(position=one_joint(0, NAN, DEFAULTS)), state())
    expected = {COMMAND_STALE: 2, MOTOR_HOT: 2, POLICY_NON_FINITE: 1}
    assert dict(safety_layer.stats.faults) == expected
    safety_layer.reset()
    assert dict(safety_layer.stats.faults) == expected


def test_stats_are_a_read_only_snapshot() -> None:
    safety_layer = layer()
    stats = safety_layer.stats
    faults: Any = stats.faults
    clamps: Any = stats.clamps
    with pytest.raises(TypeError):
        faults[COMMAND_STALE] = 1
    with pytest.raises(TypeError):
        clamps["left_knee"] = 1
    check(safety_layer, command_age_s=1.0)
    safety_layer.filter(command(position=one_joint(1, 5.0, DEFAULTS)), state())
    assert stats.faults == {}
    assert stats.clamps["left_knee"] == 0
    assert safety_layer.stats.faults == {COMMAND_STALE: 1}
    assert safety_layer.stats.clamps["left_knee"] == 1


# Wiring mistakes.


def test_a_state_of_the_wrong_size_is_an_error() -> None:
    short = state(
        position=(0.0,) * 5, velocity=(0.0,) * 5, effort=(0.0,) * 5, temperature=None
    )
    message = "the state has 5 joint values, but robot 'testbot' has 6 joints"
    with pytest.raises(ValueError, match=message):
        check(layer(), short)
    with pytest.raises(ValueError, match=message):
        layer().preflight(short)
    with pytest.raises(ValueError, match=message):
        layer().hold_command(short)


def test_temperatures_of_the_wrong_size_are_an_error() -> None:
    with pytest.raises(ValueError, match="the state's temperature has 5 joint values"):
        check(layer(), state(temperature=(30.0,) * 5))


def test_a_command_of_the_wrong_size_is_an_error() -> None:
    short = JointCommand(
        position=np.zeros(5),
        velocity=np.zeros(5),
        effort=np.zeros(5),
        kp=np.ones(5),
        kd=np.ones(5),
    )
    with pytest.raises(ValueError, match="the command has 5 joint values"):
        layer().filter(short, state())


@pytest.mark.parametrize("field", ["kp", "kd"])
def test_a_negative_gain_is_an_error(field: str) -> None:
    gains = {"kp": KP, "kd": KD}
    changes: dict[str, Any] = {field: one_joint(4, -1.0, gains[field])}
    broken = command(**changes)
    with pytest.raises(ValueError, match=f"{field} is negative for joint 'right_knee'"):
        layer().filter(broken, state())


# Configuration.


def test_the_default_thresholds() -> None:
    config = SafetyConfig()
    assert config.model_dump() == {
        "state_timeout_s": 0.05,
        "command_timeout_s": 0.5,
        "position_margin_rad": 0.05,
        "temperature_limit_c": 80.0,
        "temperature_margin_c": 10.0,
        "gain_ratio_max": 2.0,
        "unknown_fault_action": HOLD,
    }
    assert SafetyLayer(robot()).config == config


def test_every_field_is_documented() -> None:
    for name, field in SafetyConfig.model_fields.items():
        assert field.description, f"{name} has no docstring"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_timeout_s", -0.01),
        ("state_timeout_s", 0.0),
        ("command_timeout_s", -1.0),
        ("command_timeout_s", 0.0),
        ("state_timeout_s", INF),
        ("position_margin_rad", -0.01),
        ("temperature_margin_c", -1.0),
        ("temperature_limit_c", NAN),
        ("gain_ratio_max", 0.99),
        ("gain_ratio_max", 0.0),
        ("gain_ratio_max", -2.0),
        ("unknown_fault_action", "none"),
        ("unknown_fault_action", "damp"),
        ("unknown_fault_action", "hold_everything"),
        ("state_timout_s", 0.05),
    ],
)
def test_config_rejects(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SafetyConfig.model_validate({field: value})


def test_config_accepts_the_least_values() -> None:
    config = SafetyConfig(
        position_margin_rad=0.0,
        temperature_margin_c=0.0,
        gain_ratio_max=1.0,
        unknown_fault_action=SafetyAction("zero_velocity"),
    )
    assert config.gain_ratio_max == 1.0
    assert config.unknown_fault_action is ZERO_VELOCITY


def test_config_is_frozen() -> None:
    config = SafetyConfig()
    with pytest.raises(ValidationError):
        config.state_timeout_s = 1.0


# Nothing the layer does damps the robot.


def test_there_is_no_damp_action() -> None:
    assert [action.name for action in SafetyAction] == ["NONE", "ZERO_VELOCITY", "HOLD"]
    assert "DAMP" not in SafetyAction.__members__
    assert Mode.DAMP.value not in {action.value for action in SafetyAction}
    with pytest.raises(ValueError):
        SafetyAction(Mode.DAMP.value)
    assert set(FAULT_ACTIONS.values()) <= set(SafetyAction)


def test_the_layer_holds_no_robot() -> None:
    assert list(inspect.signature(SafetyLayer).parameters) == ["description", "config"]
    safety_layer = layer()
    safety_layer.filter(command(), state())
    safety_layer.hold_command(state())
    for name, value in vars(safety_layer).items():
        assert not isinstance(value, Robot), name


def test_the_safety_module_never_calls_into_a_robot() -> None:
    # The module may not import the robot port, nor touch anything that
    # changes a robot's mode or reaches its motors.
    tree = ast.parse(inspect.getsource(safety))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "embodiverse.contracts.robot" not in imported
    forbidden = {
        "operator_stop",
        "set_mode",
        "write",
        "connect",
        "close",
        "DAMP",
        "PASSIVE",
    }
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not (used | names) & (forbidden | {"Mode", "Robot"})


def test_no_input_takes_the_stiffness_out_of_the_robot() -> None:
    # Random inputs, healthy and broken, through every method. Every command
    # the layer returns is valid, holds its targets inside the limits, and
    # keeps every joint stiff: the policy's gains here are never zero, and
    # the layer never lowers a gain below its cap.
    rng = np.random.default_rng(seed=0)
    description = robot()
    safety_layer = layer()
    adapter_codes = [*FAULT_ACTIONS, "power.low", "simulation.stopped"]
    caps = 2.0 * np.array(KP), 2.0 * np.array(KD)
    for _ in range(1000):
        position = rng.uniform(LOWER - 0.5, UPPER + 0.5)
        if rng.random() < 0.1:
            position[rng.integers(6)] = rng.choice([NAN, INF, -INF])
        faults: tuple[Fault, ...] = ()
        if rng.random() < 0.1:
            faults = (Fault(code=str(rng.choice(adapter_codes)), message="random"),)
        robot_state = state(
            position=position,
            temperature=rng.uniform(20.0, 90.0, size=6),
            faults=faults,
        )
        assessment = safety_layer.check(
            robot_state,
            now_ns=NOW_NS + int(rng.choice([0, 10**6, 10**8])),
            command_age_s=float(rng.choice([0.0, 0.2, 1.0, INF, NAN])),
            overrun=bool(rng.random() < 0.05),
        )
        assert assessment.action in (NONE, ZERO_VELOCITY, HOLD)
        safety_layer.preflight(robot_state)
        kp = rng.uniform(1.0, 300.0, size=6)
        kd = rng.uniform(0.1, 10.0, size=6)
        policy = command(
            position=rng.uniform(-3.0, 3.0, size=6),
            velocity=rng.uniform(-50.0, 50.0, size=6),
            effort=rng.uniform(-200.0, 200.0, size=6),
            kp=kp,
            kd=kd,
        )
        if rng.random() < 0.05:
            policy = command(position=one_joint(0, NAN, DEFAULTS))
        if assessment.action is HOLD:
            written = safety_layer.hold_command(robot_state)
        else:
            result = safety_layer.filter(policy, robot_state)
            written = result.command
            if result.action is NONE:
                assert written.kp.tolist() == np.minimum(kp, caps[0]).tolist()
                assert written.kd.tolist() == np.minimum(kd, caps[1]).tolist()
        check_command(written, description)
        assert (written.kp > 0).all()
        assert (written.kd > 0).all()
        assert (written.position >= LOWER + MARGIN - 1e-12).all()
        assert (written.position <= UPPER - MARGIN + 1e-12).all()
        if rng.random() < 0.05:
            safety_layer.reset()
