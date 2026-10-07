"""Tests of the adapter base, over a transport that never moves.

`StillTransport` reports the default pose, stamped on the monotonic clock,
and records every law it is sent, so the tests see exactly what the base
would send to a robot. `TransportRobot` over it passes every conformance
check that needs no probe, as does a minimal `RobotBase` subclass, and the
other tests check what the kit cannot: what is sent, when, and what `close`
and `operator_stop` do while the sender thread runs.
"""

import math
import threading
import time
from collections.abc import Callable
from importlib.resources import as_file, files

import numpy as np
import pytest

from embodiverse.adapters.base import (
    TRANSPORT_SEND_FAILED,
    MotorMap,
    RobotBase,
    Transport,
    TransportRobot,
)
from embodiverse.contracts.conventions import IDENTITY_QUATERNION, STANDARD_GRAVITY
from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.robot import Capabilities, Robot, Steppable
from embodiverse.contracts.types import Fault, ImuState, JointCommand, Mode, RobotState
from embodiverse.tools.conformance.robot import (
    RobotConformance,
    check_rejects_malformed_commands,
)


def load_testbot() -> RobotDescription:
    with as_file(files("embodiverse.robots.testbot") / "description.yaml") as path:
        return load_description(path)


TESTBOT = load_testbot()
DEFAULT_POSE = np.array([joint.default_position for joint in TESTBOT.joints])
KP = np.array([joint.gains.kp for joint in TESTBOT.joints])
KD = np.array([joint.gains.kd for joint in TESTBOT.joints])
ZEROS = np.zeros(TESTBOT.num_joints)
PERIOD_S = 1.0 / TESTBOT.rates.physics_hz
CAPABILITIES = Capabilities(effort_command=True, temperature=False, max_rate_hz=500.0)

# The kit's default timing budget of 1 ms is meant for a robot's own
# computer. These tests also run on shared development machines, where the
# scheduler can pause a thread for milliseconds; 5 ms still shows that read
# and write never wait for the sender thread.
SHARED_MACHINE_BUDGET_S = 0.005


def still_state(
    position: np.ndarray = DEFAULT_POSE, faults: tuple[Fault, ...] = ()
) -> RobotState:
    """A state at rest in `position`, level, stamped now."""
    imu = ImuState(
        orientation=np.array(IDENTITY_QUATERNION),
        angular_velocity=np.zeros(3),
        linear_acceleration=np.array([0.0, 0.0, STANDARD_GRAVITY]),
    )
    return RobotState(
        stamp_ns=time.monotonic_ns(),
        position=position,
        velocity=ZEROS,
        effort=ZEROS,
        imu=imu,
        mode=Mode.PASSIVE,
        faults=faults,
    )


class StillTransport:
    """A robot that never moves, and records every law it is sent."""

    def __init__(
        self,
        *,
        position: np.ndarray = DEFAULT_POSE,
        faults: tuple[Fault, ...] = (),
        open_error: Exception | None = None,
        first_state_after_s: float = 0.0,
        send_error: Exception | None = None,
    ) -> None:
        self.position = position
        self.faults = faults
        self.open_error = open_error
        self.first_state_after_s = first_state_after_s
        self.send_error = send_error
        self.sent: list[JointCommand] = []
        self.sent_at: list[float] = []
        self.opened_at: float | None = None
        self.closed = False
        self._lock = threading.Lock()

    def open(self) -> None:
        if self.open_error is not None:
            raise self.open_error
        self.opened_at = time.monotonic()

    def receive(self) -> RobotState | None:
        assert self.opened_at is not None
        if time.monotonic() - self.opened_at < self.first_state_after_s:
            return None
        return still_state(self.position, faults=self.faults)

    def send(self, law: JointCommand) -> None:
        with self._lock:
            self.sent.append(law)
            self.sent_at.append(time.monotonic())
        if self.send_error is not None:
            raise self.send_error

    def close(self) -> None:
        self.closed = True

    def sent_since(self, index: int) -> list[JointCommand]:
        with self._lock:
            return self.sent[index:]


def make_robot(transport: Transport | None = None, **options: float) -> TransportRobot:
    return TransportRobot(
        TESTBOT, CAPABILITIES, transport or StillTransport(), **options
    )


def wait_until(condition: Callable[[], bool], timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


def is_law(
    command: JointCommand, *, position: np.ndarray, kp: np.ndarray, kd: np.ndarray
) -> bool:
    return bool(
        np.array_equal(command.position, position)
        and np.array_equal(command.kp, kp)
        and np.array_equal(command.kd, kd)
        and not command.velocity.any()
        and not command.effort.any()
    )


# The kit.


class TestTransportRobot(RobotConformance):
    """A TransportRobot over the still transport passes every check without a probe."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: make_robot(damp_s=0.0)


class StillRobot(RobotBase):
    """The smallest adapter: RobotBase with two hooks filled in."""

    def __init__(self, description: RobotDescription) -> None:
        super().__init__(description, CAPABILITIES, damp_on_close=False)
        self.laws: list[JointCommand] = []

    def _open(self) -> None:
        pass

    def _latest(self) -> RobotState:
        return still_state()

    def _apply(self, law: JointCommand) -> None:
        self.laws.append(law)


class TestStillRobot(RobotConformance):
    """A minimal RobotBase subclass passes every check without a probe."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: StillRobot(TESTBOT)


def test_the_base_implements_the_port_and_is_not_steppable() -> None:
    for robot in (make_robot(), StillRobot(TESTBOT)):
        assert isinstance(robot, Robot)
        assert not isinstance(robot, Steppable)


# The laws.


def test_each_mode_applies_its_law() -> None:
    robot = StillRobot(TESTBOT)
    robot.connect()
    assert len(robot.laws) == 1
    assert is_law(robot.laws[-1], position=ZEROS, kp=ZEROS, kd=ZEROS)
    robot.set_mode(Mode.HOLD)
    assert is_law(robot.laws[-1], position=DEFAULT_POSE, kp=KP, kd=KD)
    robot.set_mode(Mode.POLICY)
    assert is_law(robot.laws[-1], position=DEFAULT_POSE, kp=KP, kd=KD)
    command = JointCommand(
        position=DEFAULT_POSE + 0.1, velocity=ZEROS, effort=ZEROS, kp=KP, kd=KD
    )
    robot.write(command)
    assert robot.laws[-1] is command
    robot.operator_stop()
    assert is_law(robot.laws[-1], position=ZEROS, kp=ZEROS, kd=KD)
    robot.set_mode(Mode.PASSIVE)
    assert is_law(robot.laws[-1], position=ZEROS, kp=ZEROS, kd=ZEROS)
    robot.close()
    assert is_law(robot.laws[-1], position=ZEROS, kp=ZEROS, kd=ZEROS)  # no DAMP


def test_hold_latches_the_positions_measured_on_entry() -> None:
    transport = StillTransport(position=DEFAULT_POSE + 0.2)
    robot = make_robot(transport, damp_s=0.0)
    robot.connect()
    try:
        robot.set_mode(Mode.HOLD)
        sent = transport.sent_since(0)
        assert wait_until(lambda: len(transport.sent_since(len(sent))) > 0)
        assert is_law(transport.sent[-1], position=DEFAULT_POSE + 0.2, kp=KP, kd=KD)
    finally:
        robot.close()


def test_commands_written_outside_policy_are_discarded() -> None:
    robot = StillRobot(TESTBOT)
    robot.connect()
    command = JointCommand(
        position=DEFAULT_POSE + 0.1, velocity=ZEROS, effort=ZEROS, kp=KP, kd=KD
    )
    for mode in (Mode.PASSIVE, Mode.HOLD):
        robot.set_mode(mode)
        laws = len(robot.laws)
        robot.write(command)
        assert len(robot.laws) == laws
    robot.operator_stop()
    laws = len(robot.laws)
    robot.write(command)
    assert len(robot.laws) == laws
    assert robot.read().mode is Mode.DAMP


def test_write_rejects_effort_the_robot_cannot_apply() -> None:
    capabilities = Capabilities(effort_command=False, temperature=False, max_rate_hz=50)
    robot = TransportRobot(TESTBOT, capabilities, StillTransport(), damp_s=0.0)
    robot.connect()
    try:
        check_rejects_malformed_commands(robot)
        effort = JointCommand(
            position=DEFAULT_POSE, velocity=ZEROS, effort=ZEROS + 1.0, kp=KP, kd=KD
        )
        with pytest.raises(ValueError, match="cannot apply feed-forward effort"):
            robot.write(effort)
    finally:
        robot.close()


def test_read_reports_the_mode_and_keeps_the_transport_faults() -> None:
    fault = Fault(
        code="motor.error", message="motor 3 reports error 7", joint="left_knee"
    )
    robot = make_robot(StillTransport(faults=(fault,)), damp_s=0.0)
    robot.connect()
    try:
        state = robot.read()
        assert state.mode is Mode.PASSIVE
        assert state.faults == (fault,)
        robot.set_mode(Mode.HOLD)
        state = robot.read()
        assert state.mode is Mode.HOLD
        assert state.faults == (fault,)
        np.testing.assert_array_equal(state.position, DEFAULT_POSE)
    finally:
        robot.close()


# The sender thread.


def test_the_sender_resends_the_current_law_every_period() -> None:
    transport = StillTransport()
    robot = make_robot(transport, damp_s=0.0)
    robot.connect()
    try:
        time.sleep(50 * PERIOD_S)
        sent = transport.sent_since(0)
        # 50 periods have passed; a loaded machine may miss some.
        assert 20 <= len(sent) <= 60
        assert all(is_law(law, position=ZEROS, kp=ZEROS, kd=ZEROS) for law in sent)
    finally:
        robot.close()


def test_a_new_law_is_sent_at_once() -> None:
    transport = StillTransport()
    robot = make_robot(transport, damp_s=0.0)
    robot.connect()
    try:
        before = len(transport.sent_since(0))
        start = time.monotonic()
        robot.set_mode(Mode.HOLD)
        assert wait_until(
            lambda: any(
                is_law(law, position=DEFAULT_POSE, kp=KP, kd=KD)
                for law in transport.sent_since(before)
            )
        )
        hold_sent_at = next(
            at
            for law, at in zip(transport.sent, transport.sent_at, strict=True)
            if is_law(law, position=DEFAULT_POSE, kp=KP, kd=KD)
        )
        # Well inside a period on a quiet machine; generous for a loaded one.
        assert hold_sent_at - start < 0.1
    finally:
        robot.close()


def test_close_damps_the_robot_and_keeps_sending_before_releasing_it() -> None:
    transport = StillTransport()
    robot = make_robot(transport, damp_s=20 * PERIOD_S)
    robot.connect()
    robot.set_mode(Mode.HOLD)
    before = len(transport.sent_since(0))
    start = time.monotonic()
    robot.close()
    elapsed = time.monotonic() - start
    assert transport.closed
    assert elapsed >= 20 * PERIOD_S
    damped = [
        law
        for law in transport.sent_since(before)
        if is_law(law, position=ZEROS, kp=ZEROS, kd=KD)
    ]
    assert len(damped) >= 2, "DAMP must be sent, and repeated, before the release"
    assert is_law(transport.sent[-1], position=ZEROS, kp=ZEROS, kd=KD)
    assert not any(t.name == "sender-testbot" for t in threading.enumerate())
    # Nothing is sent after the release.
    time.sleep(5 * PERIOD_S)
    assert len(transport.sent) == len(damped) + before


def test_operator_stop_from_another_thread_while_the_control_thread_writes() -> None:
    transport = StillTransport()
    robot = make_robot(transport, damp_s=0.0)
    robot.connect()
    try:
        robot.set_mode(Mode.HOLD)
        robot.set_mode(Mode.POLICY)
        command = JointCommand(
            position=DEFAULT_POSE + 0.1, velocity=ZEROS, effort=ZEROS, kp=KP, kd=KD
        )
        stopped = threading.Event()

        def stop() -> None:
            time.sleep(10 * PERIOD_S)
            robot.operator_stop()
            stopped.set()

        thread = threading.Thread(target=stop, name="operator-stop")
        thread.start()
        deadline = time.monotonic() + 1.0
        while not stopped.is_set() and time.monotonic() < deadline:
            robot.write(command)  # discarded once the stop has happened
            time.sleep(PERIOD_S)
        thread.join(timeout=1.0)
        assert stopped.is_set()
        assert robot.read().mode is Mode.DAMP
        stopped_at = len(transport.sent_since(0))
        robot.write(command)
        time.sleep(5 * PERIOD_S)
        later = transport.sent_since(stopped_at)
        assert later, "the sender keeps sending after a stop"
        assert all(is_law(law, position=ZEROS, kp=ZEROS, kd=KD) for law in later)
    finally:
        robot.close()


def test_a_failing_send_is_reported_as_a_fault_and_sending_continues() -> None:
    transport = StillTransport(send_error=OSError("the socket\nis closed"))
    robot = make_robot(transport, damp_s=0.0)
    robot.connect()
    try:
        assert wait_until(lambda: bool(robot.read().faults))
        assert robot.read().faults == (
            Fault(code=TRANSPORT_SEND_FAILED, message="OSError: the socket is closed"),
        )
        count = len(transport.sent_since(0))
        assert wait_until(lambda: len(transport.sent_since(0)) > count)
    finally:
        robot.close()


# Connecting.


def test_connect_waits_for_the_first_state() -> None:
    transport = StillTransport(first_state_after_s=5 * PERIOD_S)
    robot = make_robot(transport, damp_s=0.0)
    start = time.monotonic()
    robot.connect()
    try:
        assert time.monotonic() - start >= 5 * PERIOD_S
        assert robot.read().mode is Mode.PASSIVE
    finally:
        robot.close()


def test_connect_times_out_without_state_and_leaves_the_robot_unconnected() -> None:
    transport = StillTransport(first_state_after_s=math.inf)
    robot = make_robot(transport, connect_timeout_s=10 * PERIOD_S, damp_s=0.0)
    with pytest.raises(TimeoutError, match="no state arrived within 0.02 s"):
        robot.connect()
    assert transport.closed
    with pytest.raises(RuntimeError, match="not connected"):
        robot.read()
    robot.operator_stop()  # does nothing, and does not raise
    robot.close()


def test_connect_passes_on_a_connection_error() -> None:
    transport = StillTransport(open_error=ConnectionError("no robot at 192.0.2.1"))
    robot = make_robot(transport)
    with pytest.raises(ConnectionError, match="no robot at 192.0.2.1"):
        robot.connect()
    with pytest.raises(RuntimeError, match="not connected"):
        robot.read()


def test_the_hooks_must_be_implemented() -> None:
    robot = RobotBase(TESTBOT, CAPABILITIES, damp_on_close=False)
    with pytest.raises(NotImplementedError, match="must implement _open"):
        robot.connect()
    with pytest.raises(RuntimeError, match="not connected"):
        robot.read()


# The motor map.


def with_motor_indices(
    description: RobotDescription, indices: list[int]
) -> RobotDescription:
    data = description.model_dump()
    for joint, index in zip(data["joints"], indices, strict=True):
        joint["motor_index"] = index
    return RobotDescription.model_validate(data)


def test_motor_map_is_the_identity_for_descriptions_in_motor_order() -> None:
    motors = MotorMap(TESTBOT)
    assert motors.slots == TESTBOT.num_joints
    values = np.arange(TESTBOT.num_joints, dtype=float)
    np.testing.assert_array_equal(motors.to_motors(values), values)
    np.testing.assert_array_equal(motors.from_motors(values), values)


def test_motor_map_scatters_and_gathers_by_motor_index() -> None:
    description = with_motor_indices(TESTBOT, [4, 0, 7, 2, 9, 1])
    motors = MotorMap(description, slots=12)
    values = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    scattered = motors.to_motors(values, fill=math.nan)
    assert scattered.shape == (12,)
    np.testing.assert_array_equal(scattered[[4, 0, 7, 2, 9, 1]], values)
    assert np.isnan(scattered[[3, 5, 6, 8, 10, 11]]).all()
    np.testing.assert_array_equal(motors.from_motors(scattered), values)
    assert MotorMap(description).slots == 10


def test_motor_map_refuses_wrong_sizes() -> None:
    description = with_motor_indices(TESTBOT, [4, 0, 7, 2, 9, 1])
    with pytest.raises(ValueError, match="need at least 10 slots, not 8"):
        MotorMap(description, slots=8)
    motors = MotorMap(description)
    with pytest.raises(ValueError, match="one value per joint, 6"):
        motors.to_motors(np.zeros(5))
    with pytest.raises(ValueError, match="at least 10 values"):
        motors.from_motors(np.zeros(9))
