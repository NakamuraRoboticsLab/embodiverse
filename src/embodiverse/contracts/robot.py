"""The robot port: the one interface through which the runtime drives a robot.

Every robot and every simulator, for example a Unitree G1 or a MuJoCo model,
is reached through an adapter that implements `Robot`. Code above the
adapters never imports a vendor SDK or a simulator, so moving from simulation
to hardware, or between simulators, means swapping the adapter. The port
exchanges `RobotState` and `JointCommand` (see `embodiverse.contracts.types`),
in description joint order and SI units; vendor formats are converted inside
the adapter. docs/architecture.md summarizes the semantics below.

Threading: one control thread calls every method except `operator_stop`, so
adapters need not make the others safe to call concurrently. `operator_stop`
may be called from any thread at any time, including while the control
thread is inside another method or stalled.

Modes change only through `set_mode`, `operator_stop` and `close`; an adapter
never changes mode on its own. It reports the problems it detects in
`RobotState.faults`, and the safety layer decides how to respond. Automatic
faults never cause DAMP. The safety layer handles them by holding the robot
or by stopping it walking, which zeroes the velocity command
(docs/policies.md): a humanoid without stiffness falls unless something is
holding it, and only a person can tell whether something is.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.types import JointCommand, Mode, RobotState

__all__ = [
    "MODE_TRANSITIONS",
    "Capabilities",
    "Robot",
    "Steppable",
    "check_transition",
]


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What a robot supports, where robots differ.

    Raises ValueError if `max_rate_hz` is not a positive finite number.
    """

    effort_command: bool
    """Whether the robot applies feed-forward torque (`JointCommand.effort`).

    If False, `write` raises ValueError for a command with any non-zero effort.
    """

    temperature: bool
    """Whether the robot reports motor temperatures.

    If True, `RobotState.temperature` is never None; if False, it is always None.
    """

    max_rate_hz: float
    """Highest rate, in Hz, at which calling `write` is useful.

    Writing faster is allowed but gains nothing.
    """

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_rate_hz) and self.max_rate_hz > 0):
            raise ValueError(
                f"max_rate_hz must be a positive finite number, got {self.max_rate_hz}"
            )


MODE_TRANSITIONS: Mapping[Mode, frozenset[Mode]] = MappingProxyType(
    {
        Mode.PASSIVE: frozenset({Mode.HOLD}),
        Mode.HOLD: frozenset({Mode.PASSIVE, Mode.POLICY}),
        Mode.POLICY: frozenset({Mode.HOLD}),
        Mode.DAMP: frozenset({Mode.PASSIVE}),
    }
)
"""The modes `Robot.set_mode` can reach from each mode.

Setting the current mode again is also allowed and does nothing, except for
DAMP, which is never a `set_mode` target: `Robot.operator_stop` enters it,
and so does `Robot.close` on hardware. The table is shaped so that:

- POLICY is entered only from HOLD, so the robot is held with known gains
  when the policy takes over;
- POLICY is left only to HOLD, so the robot never drops from a policy
  straight to zero torque;
- DAMP is left only by a deliberate reset to PASSIVE, so a program cannot
  resume a policy after a person stopped the robot.
"""


def check_transition(current: Mode, target: Mode) -> None:
    """Check that `set_mode(target)` is allowed in mode `current`.

    Returns if `target` is in `MODE_TRANSITIONS[current]`, or equals `current`
    and is not DAMP. Otherwise raises ValueError naming both modes.
    """
    if target is Mode.DAMP:
        raise ValueError(
            f"cannot change mode from {current.name} to DAMP:"
            " DAMP is entered only through operator_stop"
        )
    if target is current or target in MODE_TRANSITIONS[current]:
        return
    reachable = " or ".join(sorted(mode.name for mode in MODE_TRANSITIONS[current]))
    raise ValueError(
        f"cannot change mode from {current.name} to {target.name}:"
        f" {current.name} can change only to {reachable}"
    )


@runtime_checkable
class Robot(Protocol):
    """A robot or simulated robot, as the runtime sees it.

    An instance is used once: created, connected, used and closed.
    `description` and `capabilities` are fixed for its lifetime and can be
    called at any point in it. `read`, `write` and `set_mode` raise
    RuntimeError before `connect` and after `close`.
    """

    @property
    def description(self) -> RobotDescription:
        """The description of the robot model, which fixes the joint order."""
        ...

    def capabilities(self) -> Capabilities:
        """What this robot supports; the same value on every call."""
        ...

    def connect(self) -> None:
        """Open the connection and block until the first state is available.

        Afterwards the mode is PASSIVE. Raises ConnectionError if the robot
        cannot be reached, or TimeoutError if it sends no state in time.
        Raises RuntimeError if the robot was already connected or has been
        closed.
        """
        ...

    def read(self) -> RobotState:
        """Return the latest state, without waiting for new data.

        Two calls may return the same measurements; `stamp_ns` never decreases
        from one call to the next. The state's `mode` is the mode at the time
        of the call, so it reflects every `set_mode` and `operator_stop` that
        returned before the call. Arrays have one element per joint of the description,
        in SI units; values are as measured, so a non-finite reading is passed
        on for the safety layer to reject.

        Raises RuntimeError if not connected.
        """
        ...

    def write(self, command: JointCommand) -> None:
        """Send `command`, without waiting for the robot.

        The command takes effect only in POLICY mode, where the latest command
        is applied until the next one replaces it. In other modes it is
        accepted and discarded; on entering POLICY the robot keeps holding its
        HOLD targets until the first `write` after the change.

        Raises ValueError, in every mode, if the command fails `check_command`
        against the description, or has a non-zero `effort` and the robot's
        capabilities lack `effort_command`. Raises RuntimeError if not
        connected.
        """
        ...

    def set_mode(self, mode: Mode) -> None:
        """Change to `mode`, as `MODE_TRANSITIONS` allows.

        Entering HOLD holds the joint positions measured at that moment, with
        zero target velocity, no feed-forward torque and the description's
        default gains. Entering PASSIVE applies zero torque. Setting the
        current mode again does nothing; in particular, HOLD keeps its targets.
        Returns without waiting for the robot; the next `read` reports the new
        mode.

        Raises ValueError, as `check_transition` does, for a transition not in
        the table, and always for DAMP. Raises RuntimeError if not connected.
        """
        ...

    def operator_stop(self) -> None:
        """Enter DAMP from any mode: damping only, kp = 0 with the description's kd.

        Called only from a person's stop control. Never raises, may be called
        from any thread, and is idempotent; before `connect` and after `close`
        it does nothing. It must not wait for the control thread. When it
        returns, every later `read` reports DAMP and later commands are
        discarded. It is atomic with respect to `set_mode`: if the two
        overlap, the result is as if one ran entirely before the other. DAMP
        is left only by `set_mode(Mode.PASSIVE)`.
        """
        ...

    def close(self) -> None:
        """Release the robot and its connection.

        Hardware adapters enter DAMP first, so the robot is not left applying
        a command that nothing updates. Blocks until the robot is released.
        Idempotent; afterwards `read`, `write` and `set_mode` raise
        RuntimeError.
        """
        ...


@runtime_checkable
class Steppable(Protocol):
    """A simulated robot run in lockstep with its caller.

    A robot that implements `Steppable` as well as `Robot` advances its
    simulated time only when `advance` is called. Between calls the simulation
    is paused: `read` returns the same measurements and the robot does not
    move, however long the caller takes. Runs are therefore reproducible and
    independent of the speed of the machine.
    """

    def advance(self, seconds: float) -> None:
        """Simulate the next `seconds` of time, then return.

        The current mode's control is applied at every physics step, using
        the latest command in POLICY. Afterwards `read` returns the state at
        the new time, its `stamp_ns` greater by `seconds`, rounded to whole
        nanoseconds.

        Called from the control thread. Raises ValueError unless `seconds` is
        a positive whole number of physics steps (each
        `1 / description.rates.physics_hz` seconds), to within floating-point
        rounding. Raises RuntimeError if not connected.
        """
        ...
