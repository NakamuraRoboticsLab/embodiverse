"""The adapter base: the robot port's semantics, implemented once.

Every adapter must do the same things in the same way: refuse to be used
before `connect` and after `close`, keep the mode table of
`embodiverse.contracts.robot`, hold the positions measured on entering
HOLD, apply a command only in POLICY, check every command against the
description, enter DAMP from any thread when a person stops the robot, and
damp a real robot before releasing it. Written separately for each robot,
these come out subtly different. `RobotBase` implements them once; an
adapter fills in how state arrives and how the current control law is
sent.

Three things are here:

- `RobotBase` implements `Robot` for any adapter. A subclass fills in four
  hooks: `_open`, `_latest`, `_apply` and `_close`. The base keeps the mode
  and the **law**: the `JointCommand` the motors should apply right now,
  which is zeros in PASSIVE, the latched positions with the description's
  gains in HOLD, the latest command in POLICY (HOLD's law until the first
  one), and damping only in DAMP. The MuJoCo adapter is built on it.
- `Transport` is the shape of one vendor's wire: open it, receive the
  newest state, send a law, close it. The vendor's messages, units, joint
  order and quaternion order are converted inside the transport, so what
  crosses it is already in the port's terms.
- `TransportRobot` is `RobotBase` plus a sender thread, for hardware. The
  thread re-sends the current law at `description.rates.physics_hz`
  whether or not the runner wrote anything, because a robot keeps applying
  the last command it received: stopping publication is not a stop. It
  sends a new law as soon as one is set, enters DAMP and keeps sending it
  for a moment before `close` releases the robot, and reports a send that
  failed as a fault. The loopback adapter is built on it, and a hardware
  adapter is a `TransportRobot` with the vendor's transport.

`MotorMap` converts between description joint order and a vendor's motor
arrays by each joint's `motor_index`, in both directions, for transports
and the pure functions they are built from.

Threading, as the port requires: one control thread calls every method
but `operator_stop`, which may come from any thread at any time. The base
guards the mode and the lifecycle with one lock, held only for a few
assignments, so `operator_stop` never waits for the control thread. The
hooks run on the control thread, except `_apply`, which `operator_stop`
also calls, under the lock: it must return at once and must not wait for
the control thread. A `Transport`'s `send` runs on the sender thread while
`receive` runs on the control thread, so a transport makes those two safe
to call together.

docs/adding-a-robot.md shows how to build a hardware adapter on it.
"""

import contextlib
import threading
import time
from dataclasses import replace
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.robot import Capabilities, check_transition
from embodiverse.contracts.types import (
    Fault,
    JointCommand,
    Mode,
    RobotState,
    check_command,
)

__all__ = [
    "TRANSPORT_SEND_FAILED",
    "MotorMap",
    "RobotBase",
    "Transport",
    "TransportRobot",
]

TRANSPORT_SEND_FAILED = "transport.send_failed"
"""Code of the fault a `TransportRobot` reports once a send has raised."""


class MotorMap:
    """Between description joint order and a vendor's motor arrays.

    A vendor's messages carry one slot per motor, in the vendor's order,
    often with more slots than the robot has joints; the description gives
    each joint's slot as its `motor_index`. `to_motors` scatters an array in
    description order into the slots, filling the unused ones, and
    `from_motors` gathers the joints' slots back into description order.

    `slots` is the length of the vendor's arrays; by default, one more than
    the largest motor index. Raises ValueError if it is smaller than that.
    """

    def __init__(self, description: RobotDescription, slots: int | None = None) -> None:
        self.index: NDArray[np.intp] = np.array(
            [joint.motor_index for joint in description.joints], dtype=np.intp
        )
        """Each joint's motor index, in description order."""
        needed = int(self.index.max()) + 1
        if slots is None:
            slots = needed
        elif slots < needed:
            raise ValueError(
                f"robot {description.name!r} uses motor index {needed - 1}, so its"
                f" motor arrays need at least {needed} slots, not {slots}"
            )
        self.slots: int = slots
        """The length of the vendor's motor arrays."""

    def to_motors(self, values: ArrayLike, fill: float = 0.0) -> NDArray[np.float64]:
        """Return `values`, in description order, scattered into the motor slots.

        Slots no joint uses hold `fill`. Raises ValueError if `values` does
        not have one element per joint.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.shape != self.index.shape:
            raise ValueError(
                f"expected one value per joint, {self.index.size}, got shape"
                f" {array.shape}"
            )
        motors = np.full(self.slots, fill, dtype=np.float64)
        motors[self.index] = array
        return motors

    def from_motors(self, values: ArrayLike) -> NDArray[np.float64]:
        """Return the joints' slots of `values`, a motor array, in description order.

        Raises ValueError if `values` has fewer than `slots` elements.
        """
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 1 or array.size < self.slots:
            raise ValueError(
                f"expected a motor array of at least {self.slots} values, got shape"
                f" {array.shape}"
            )
        return array[self.index]


class RobotBase:
    """`Robot`, except for how state arrives and how the law reaches the motors.

    A subclass calls `__init__` with the description, its capabilities and
    whether `close` must damp the robot first (true for hardware, false for
    a simulator), and fills in the hooks:

    - `_open()`: open the connection and return once the first state can
      be read. Raise ConnectionError if the robot cannot be reached, or
      TimeoutError if it sends no state in time; the robot then stays
      unconnected.
    - `_latest()`: return the newest state, without waiting for new data.
      Its `mode` is replaced by the current mode and the faults from
      `_faults` are appended, so the subclass fills `mode` with anything,
      conventionally PASSIVE.
    - `_apply(law)`: the law changed; make the motors apply it. Called under
      the mode lock, from `operator_stop` as well as from the control
      thread, so it must return at once. The default does nothing, for a
      subclass whose stepping or sending thread reads `_control` itself.
    - `_close()`: release the robot and its connection; block until it is
      released. The default does nothing.
    - `_faults()`: the faults to report in every state, none by default.

    The subclass reads `_control`, the current `(mode, law)` pair, wherever
    it applies the law; the pair is replaced as one tuple, so a reader sees
    a consistent pair without the lock. Everything else is final.
    """

    def __init__(
        self,
        description: RobotDescription,
        capabilities: Capabilities,
        *,
        damp_on_close: bool,
    ) -> None:
        self._description = description
        self._capabilities = capabilities
        self._damp_on_close = damp_on_close
        joints = description.joints
        zeros = np.zeros(description.num_joints)
        self._kp = np.array([joint.gains.kp for joint in joints])
        self._kd = np.array([joint.gains.kd for joint in joints])
        self._passive = JointCommand(
            position=zeros, velocity=zeros, effort=zeros, kp=zeros, kd=zeros
        )
        self._damp = JointCommand(
            position=zeros, velocity=zeros, effort=zeros, kp=zeros, kd=self._kd
        )
        self._control: tuple[Mode, JointCommand] = (Mode.PASSIVE, self._passive)
        # Guards the mode and the lifecycle flags. Held only for a few
        # assignments and `_apply`, so operator_stop never waits long.
        self._mode_lock = threading.Lock()
        self._connected = False
        self._closed = False

    # --- Robot ---------------------------------------------------------------

    @property
    def description(self) -> RobotDescription:
        return self._description

    def capabilities(self) -> Capabilities:
        return self._capabilities

    def connect(self) -> None:
        """Open the connection through `_open`, then start in PASSIVE.

        Raises RuntimeError if the robot was already connected or has been
        closed, and whatever `_open` raises, leaving the robot unconnected.
        """
        with self._mode_lock:
            if self._connected or self._closed:
                state = "closed" if self._closed else "already connected"
                raise RuntimeError(
                    f"{self._name()} is {state}: a robot is connected at most once"
                )
        self._open()
        with self._mode_lock:
            self._control = (Mode.PASSIVE, self._passive)
            self._connected = True
            self._apply(self._passive)

    def read(self) -> RobotState:
        """Return the newest state from `_latest`, in the current mode."""
        self._check_connected()
        state = self._latest()
        mode = self._control[0]
        faults = self._faults()
        if state.mode is mode and not faults:
            return state
        return replace(state, mode=mode, faults=(*state.faults, *faults))

    def write(self, command: JointCommand) -> None:
        """Check `command`; in POLICY, make it the law."""
        self._check_connected()
        check_command(command, self._description)
        if not self._capabilities.effort_command and np.any(command.effort != 0.0):
            raise ValueError(
                f"{self._name()} cannot apply feed-forward effort, but the command"
                " has non-zero effort: send zeros, or declare effort_command=True"
            )
        with self._mode_lock:
            if self._control[0] is Mode.POLICY:
                self._control = (Mode.POLICY, command)
                self._apply(command)

    def set_mode(self, mode: Mode) -> None:
        """Change to `mode` as `MODE_TRANSITIONS` allows, and apply its law."""
        self._check_connected()
        # Measured before the lock is taken: `_latest` may take the
        # subclass's own locks, and must never do so under the mode lock.
        measured: NDArray[np.float64] | None = None
        if mode is Mode.HOLD:
            measured = self._latest().position
        with self._mode_lock:
            current, law = self._control
            check_transition(current, mode)
            if mode is current:
                return
            if measured is not None:
                law = self._hold_law(measured)
            elif mode is Mode.PASSIVE:
                law = self._passive
            # POLICY keeps HOLD's law until the first command is written.
            self._control = (mode, law)
            self._apply(law)

    def operator_stop(self) -> None:
        """Enter DAMP from any mode; does nothing before connect or after close."""
        with self._mode_lock:
            if not self._connected:
                return
            self._control = (Mode.DAMP, self._damp)
            # operator_stop never raises, whatever a hook does.
            with contextlib.suppress(Exception):
                self._apply(self._damp)

    def close(self) -> None:
        """Enter DAMP if the robot damps on close, then release it through `_close`.

        Idempotent. A robot that was never connected is only marked closed.
        """
        with self._mode_lock:
            if self._closed:
                return
            connected = self._connected
            if connected and self._damp_on_close:
                self._control = (Mode.DAMP, self._damp)
                self._apply(self._damp)
            self._connected = False
            self._closed = True
        if connected:
            self._close()

    # --- Hooks ---------------------------------------------------------------

    def _open(self) -> None:
        """Open the connection; return once `_latest` can be called."""
        raise NotImplementedError(f"{type(self).__name__} must implement _open")

    def _latest(self) -> RobotState:
        """Return the newest state, without waiting for new data."""
        raise NotImplementedError(f"{type(self).__name__} must implement _latest")

    def _apply(self, law: JointCommand) -> None:
        """The law changed; make the motors apply it. Must return at once."""

    def _close(self) -> None:
        """Release the robot and its connection; block until it is released."""

    def _faults(self) -> tuple[Fault, ...]:
        """The faults to report in every state; none by default."""
        return ()

    # --- For subclasses ------------------------------------------------------

    def _name(self) -> str:
        """How the robot is called in error messages."""
        return f"{type(self).__name__} for robot {self._description.name!r}"

    def _check_connected(self) -> None:
        """Raise RuntimeError unless the robot is connected."""
        if not self._connected:
            state = "closed" if self._closed else "not connected: call connect() first"
            raise RuntimeError(f"{self._name()} is {state}")

    def _hold_law(self, position: NDArray[np.float64]) -> JointCommand:
        """The law of HOLD: hold `position` with the description's gains."""
        return JointCommand(
            position=position,
            velocity=self._passive.velocity,
            effort=self._passive.effort,
            kp=self._kp,
            kd=self._kd,
        )


class Transport(Protocol):
    """One robot's wire, in the port's terms.

    `open` and `close` are called on the control thread, by `connect` and
    `close`, when the sender thread is not running. `receive` is called on
    the control thread and `send` on the sender thread, at the same time,
    so the two must be safe to call together. None of them blocks for long:
    `receive` returns the newest state already received, and `send` hands
    the law to the wire and returns.

    What crosses the transport is in description joint order, SI units and
    (x, y, z, w) quaternions, stamped on the monotonic clock in ns; the
    vendor's formats are converted inside it.
    """

    def open(self) -> None:
        """Open the connection and start receiving state.

        Raises ConnectionError if the robot cannot be reached. Returns as
        soon as state can arrive; `TransportRobot.connect` waits for it.
        """
        ...

    def receive(self) -> RobotState | None:
        """Return the newest state received, or None if none has arrived yet.

        Once a state has been returned, later calls return the newest one
        and never None. The state's `mode` is ignored; its `faults` are
        reported.
        """
        ...

    def send(self, law: JointCommand) -> None:
        """Send `law` to the robot, without waiting for it.

        Raises if the send fails; `TransportRobot` reports the failure as
        a fault and keeps sending.
        """
        ...

    def close(self) -> None:
        """Close the connection. Called once, after the last `send`."""
        ...


class TransportRobot(RobotBase):
    """`RobotBase` over a `Transport`, with a sender thread: the shape of hardware.

    `connect` opens the transport, waits up to `connect_timeout_s` for its
    first state, raising TimeoutError if none arrives, and starts the
    sender thread. The thread sends the current law every physics period
    (`1 / description.rates.physics_hz`) and at once when the law changes.
    `read` returns the transport's newest state. `close` enters DAMP, keeps
    sending it for `damp_s`, stops the thread and closes the transport.

    If a send raises, the thread records the error as a fault with code
    `transport.send_failed`, reported in every later state, and keeps
    sending: a robot that stops hearing from its controller is in a worse
    place than one that hears the same law twice.
    """

    def __init__(
        self,
        description: RobotDescription,
        capabilities: Capabilities,
        transport: Transport,
        *,
        connect_timeout_s: float = 5.0,
        damp_s: float = 0.1,
    ) -> None:
        super().__init__(description, capabilities, damp_on_close=True)
        self._transport = transport
        self._connect_timeout_s = connect_timeout_s
        self._damp_s = damp_s
        self._period_s = 1.0 / description.rates.physics_hz
        self._last: RobotState | None = None
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._send_failure: Fault | None = None
        self._overruns = 0

    @property
    def overruns(self) -> int:
        """How many periodic sends started more than a period late."""
        return self._overruns

    def _open(self) -> None:
        self._transport.open()
        deadline = time.monotonic() + self._connect_timeout_s
        while (state := self._transport.receive()) is None:
            if time.monotonic() >= deadline:
                with contextlib.suppress(Exception):
                    self._transport.close()
                raise TimeoutError(
                    f"{self._name()}: no state arrived within"
                    f" {self._connect_timeout_s:g} s of opening the transport"
                )
            time.sleep(self._period_s)
        self._last = state
        self._stopping.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"sender-{self._description.name}", daemon=True
        )
        self._thread.start()

    def _latest(self) -> RobotState:
        state = self._transport.receive()
        if state is not None:
            self._last = state
            return state
        assert self._last is not None, "connect waits for the first state"
        return self._last

    def _apply(self, law: JointCommand) -> None:
        self._wake.set()

    def _close(self) -> None:
        # DAMP is the law by now (`close` set it); let the thread send it
        # for a moment, so a dropped message is repeated.
        time.sleep(self._damp_s)
        self._stopping.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self._transport.close()

    def _faults(self) -> tuple[Fault, ...]:
        failure = self._send_failure
        return () if failure is None else (failure,)

    def _run(self) -> None:
        """Send the current law every period, and at once when it changes."""
        period = self._period_s
        deadline = time.monotonic() + period
        while True:
            remaining = deadline - time.monotonic()
            if remaining > 0 and self._wake.wait(remaining):
                # A new law: send it now, and keep the periodic cadence.
                self._wake.clear()
            else:
                deadline += period
                now = time.monotonic()
                if deadline < now:
                    self._overruns += 1
                    deadline = now + period
            if self._stopping.is_set():
                return
            try:
                self._transport.send(self._control[1])
            except Exception as error:
                message = " ".join(f"{type(error).__name__}: {error}".split())
                self._send_failure = Fault(code=TRANSPORT_SEND_FAILED, message=message)
