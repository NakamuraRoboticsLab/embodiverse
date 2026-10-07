"""The safety layer: what stands between a policy and the robot's motors.

Every tick's inputs pass through `SafetyLayer.check` before the policy
runs, and every command the policy produces passes through
`SafetyLayer.filter` before it reaches the robot. `SafetyLayer.preflight`
decides whether a policy may start at all. The layer reads the robot's
state and the policy's commands and returns verdicts and safe commands; it
holds no reference to the robot and never calls it, so it cannot change
the robot's mode. The runner acts on what it returns.

Four rules shape it:

- **Automatic faults never cause DAMP.** A humanoid without stiffness falls
  unless something is holding it, and only a person can tell whether
  something is. Only a person, through `Robot.operator_stop`, damps the
  robot, so `SafetyAction` has no member for it.
- **An automatic response is to stop moving, or to hold.** ZERO_VELOCITY
  runs the policy with a zero velocity command, so a walking robot comes to
  a stand. HOLD stops using the policy's output and holds the last safe
  joint targets.
- **The fault table is data.** `FAULT_ACTIONS` maps each fault code to its
  action, and adapter-reported codes that are not in it map to
  `SafetyConfig.unknown_fault_action`.
- **HOLD is latched.** Once taken, it stays until `SafetyLayer.reset`, which
  only a person or the runner's start sequence calls. ZERO_VELOCITY is not
  latched: it clears when its cause clears.

The thresholds are in `SafetyConfig`. Its defaults are general starting
points; check them against each robot's own ratings, such as its motors'
temperature limit. docs/policies.md describes the layer and its fault
table.
"""

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.types import Fault, JointCommand, RobotState

__all__ = [
    "COMMAND_STALE",
    "FAULT_ACTIONS",
    "JOINT_OUT_OF_LIMITS",
    "LOOP_OVERRUN",
    "MOTOR_HOT",
    "POLICY_NON_FINITE",
    "STATE_NON_FINITE",
    "STATE_STALE",
    "Assessment",
    "SafetyAction",
    "SafetyConfig",
    "SafetyLayer",
    "SafetyResult",
    "SafetyStats",
    "most_severe",
]


class SafetyAction(str, Enum):
    """What the runner does in response to faults, from the least to the most severe.

    Actions compare by severity, HOLD > ZERO_VELOCITY > NONE, so `max` and
    `most_severe` give the most severe of several.

    There is deliberately no DAMP member. Damping only, with no stiffness,
    makes a humanoid fall unless something is holding it, so no automatic
    response may cause it: only a person's stop control does, through
    `Robot.operator_stop`.
    """

    NONE = "none"
    """Carry on: run the policy and apply its filtered command."""

    ZERO_VELOCITY = "zero_velocity"
    """Stop moving: run the policy with a zero velocity command, so that a
    walking robot comes to a stand. Not latched."""

    HOLD = "hold"
    """Stop using the policy: write `SafetyLayer.hold_command` instead of its
    output, holding the last safe joint targets. Latched until
    `SafetyLayer.reset`."""

    @property
    def severity(self) -> int:
        """The action's rank: 0 for NONE, 1 for ZERO_VELOCITY, 2 for HOLD."""
        return _SEVERITY[self]

    # str's comparisons would order the actions alphabetically, which puts
    # ZERO_VELOCITY above HOLD, so they are replaced by comparisons by severity.

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SafetyAction):
            return NotImplemented
        return self.severity < other.severity

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SafetyAction):
            return NotImplemented
        return self.severity <= other.severity

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SafetyAction):
            return NotImplemented
        return self.severity > other.severity

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SafetyAction):
            return NotImplemented
        return self.severity >= other.severity


# The members are declared from the least to the most severe.
_SEVERITY: Mapping[SafetyAction, int] = MappingProxyType(
    {action: rank for rank, action in enumerate(SafetyAction)}
)


def most_severe(actions: Iterable[SafetyAction]) -> SafetyAction:
    """Return the most severe of `actions`, or NONE if there are none."""
    return max(actions, default=SafetyAction.NONE)


# The fault codes the safety layer reports. Adapters report their own codes
# in `RobotState.faults`; a code of theirs that is in the table maps
# through it.

STATE_STALE = "state.stale"
"""The robot state is older than `SafetyConfig.state_timeout_s`."""

STATE_NON_FINITE = "state.non_finite"
"""A value of the robot state, the IMU's and the temperatures included, is
NaN or infinite."""

COMMAND_STALE = "command.stale"
"""The velocity command has not been updated for longer than
`SafetyConfig.command_timeout_s`."""

MOTOR_HOT = "motor.hot"
"""A motor is at or above `temperature_limit_c - temperature_margin_c`.

One fault per hot joint, with the joint named: the fault is also the alarm
that the runner reports to the operator.
"""

LOOP_OVERRUN = "loop.overrun"
"""The previous control tick took longer than its period."""

POLICY_NON_FINITE = "policy.non_finite"
"""A value of the command built from the policy's output is NaN or infinite."""

JOINT_OUT_OF_LIMITS = "joint.out_of_limits"
"""A joint is outside its position limits. Reported only by `preflight`,
which refuses to start a policy, so it has no action in the table."""

FAULT_ACTIONS: Mapping[str, SafetyAction] = MappingProxyType(
    {
        COMMAND_STALE: SafetyAction.ZERO_VELOCITY,
        STATE_STALE: SafetyAction.HOLD,
        STATE_NON_FINITE: SafetyAction.HOLD,
        POLICY_NON_FINITE: SafetyAction.HOLD,
        LOOP_OVERRUN: SafetyAction.HOLD,
        MOTOR_HOT: SafetyAction.ZERO_VELOCITY,
    }
)
"""The action for each fault the safety layer handles while a policy runs.

A stale velocity command or a hot motor stops the robot moving: both are
recoverable, and standing still is safe. Everything that means the policy's
inputs or outputs can no longer be trusted holds: a stale or non-finite
state, a non-finite command and a late tick. Codes that adapters report and
that are not in the table map to `SafetyConfig.unknown_fault_action`.
"""


class SafetyConfig(BaseModel):
    """The safety layer's thresholds.

    Each default is a general starting point, explained in its field's
    docstring; check them against each robot's own ratings.

    Raises pydantic's ValidationError, a ValueError, for an unknown field, a
    non-finite number, a timeout that is not positive, a negative margin, a
    gain ratio below 1, or an `unknown_fault_action` of NONE.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, use_attribute_docstrings=True
    )

    state_timeout_s: float = Field(default=0.05, gt=0)
    """Longest age of the robot state, in s: the state is stale if
    `now_ns - state.stamp_ns` is longer.

    0.05 s is two and a half periods of a 50 Hz policy and 25 periods of a
    500 Hz robot: long enough not to trip on scheduling jitter or a few lost
    messages, short enough that a robot walking at 1 m/s has moved 5 cm on
    old data.
    """

    command_timeout_s: float = Field(default=0.5, gt=0)
    """Longest time without an update to the velocity command, in s.

    Joysticks and planners publish at 10 Hz or more, so 0.5 s is several
    lost messages; at 1 m/s, the robot walks at most 0.5 m on a command
    that nothing is sending any more.
    """

    position_margin_rad: float = Field(default=0.05, ge=0)
    """How far inside its position limits every joint target is kept, in rad.

    Targets are clamped to `[lower + margin, upper - margin]`, or to the
    range's midpoint if it is narrower than twice the margin. 0.05 rad
    (about 3°) keeps a target that a PD loop overshoots off the joint's
    mechanical stop.
    """

    temperature_limit_c: float = 80.0
    """The motors' highest safe temperature, in °C.

    80 °C is a placeholder until each robot's motor rating is known; the
    robot description has no field for it yet.
    """

    temperature_margin_c: float = Field(default=10.0, ge=0)
    """How far below `temperature_limit_c` a motor counts as hot, in °C.

    A motor at `temperature_limit_c - temperature_margin_c` or above stops
    the robot walking. 10 °C leaves time to stand still and cool before
    the vendor's own protection cuts the motor's torque, which would drop
    the robot.
    """

    gain_ratio_max: float = Field(default=2.0, ge=1)
    """Largest multiple of the description's gains a command may use, per joint.

    `kp` is capped at `gain_ratio_max * kp` of the description, and `kd`
    likewise. Policies are trained with gains at or near the description's;
    a gain far above them is more likely a unit or copying mistake than a
    choice, and stiff gains destabilize the PD loop and amplify sensor
    noise. 2 still allows deliberately stiffer tuning.
    """

    unknown_fault_action: SafetyAction = SafetyAction.HOLD
    """The action for an adapter-reported fault whose code is not in
    `FAULT_ACTIONS`: ZERO_VELOCITY or HOLD, never NONE.

    HOLD: a fault nobody has classified gets the most severe automatic
    response. NONE is refused, so an unclassified fault always stops the
    robot moving.
    """

    @field_validator("unknown_fault_action")
    @classmethod
    def _check_unknown_fault_action(cls, action: SafetyAction) -> SafetyAction:
        if action is SafetyAction.NONE:
            raise ValueError(
                "unknown_fault_action must be zero_velocity or hold: a fault"
                " nobody has classified must not be ignored"
            )
        return action


_DEFAULT_CONFIG = SafetyConfig()


@dataclass(frozen=True, slots=True)
class Assessment:
    """What `SafetyLayer.check` found before the policy ran."""

    faults: tuple[Fault, ...]
    """The faults found on this tick, in the order `check` looks for them;
    empty if there are none."""

    action: SafetyAction
    """The most severe action of the faults, or HOLD while HOLD is latched."""


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """What `SafetyLayer.filter` made of the policy's command."""

    command: JointCommand
    """The command to write to the robot: the filtered command, or the hold
    command if the policy's command was not finite or HOLD is latched."""

    faults: tuple[Fault, ...]
    """`policy.non_finite` if the policy's command was not finite; otherwise
    empty. Clamping is not a fault."""

    action: SafetyAction
    """HOLD if the command was not finite or HOLD is latched; otherwise NONE."""


@dataclass(frozen=True, slots=True)
class SafetyStats:
    """A summary of what the safety layer has done since it was created."""

    faults: Mapping[str, int]
    """How many times `check` and `filter` have reported each fault code."""

    clamps: Mapping[str, int]
    """For each joint, by name, how many filtered commands had a value of
    that joint clamped."""


class SafetyLayer:
    """Checks every tick's inputs and filters every command, for one robot.

    Per tick, the runner calls `check` with the state before running the
    policy, and acts on its action: NONE runs the policy; ZERO_VELOCITY
    runs it with a zero velocity command; HOLD does not run it and writes
    `hold_command(state)`. When the policy runs, the runner passes the
    command built from its output through `filter` and writes the result's
    command. Before a policy starts, `preflight` must return no faults.

    HOLD is latched: once any HOLD action has been taken, by `check` or by
    `filter`, `check` reports HOLD and `filter` returns the hold command
    until `reset` is called. A policy that produced NaN or ran late, or
    that saw stale or broken inputs, is no longer trusted, and a person
    decides when to start it again. ZERO_VELOCITY is not latched: a stale
    joystick or a warm motor is recoverable, and the robot walks again when
    the command is fresh or the motor has cooled.

    The layer holds no reference to the robot and never calls it. Limits
    come from `description`, thresholds from `config`; both are fixed when
    the layer is created. Methods are called from the control thread.
    """

    def __init__(
        self, description: RobotDescription, config: SafetyConfig = _DEFAULT_CONFIG
    ) -> None:
        self._description = description
        self._config = config
        joints = description.joints
        self._names = description.joint_names
        lower = np.array([joint.limits.position[0] for joint in joints])
        upper = np.array([joint.limits.position[1] for joint in joints])
        margin = config.position_margin_rad
        # Where the range is narrower than twice the margin, both bounds are
        # its midpoint.
        narrow = upper - lower < 2.0 * margin
        middle = (lower + upper) / 2.0
        self._lower = lower
        self._upper = upper
        self._target_lower = np.where(narrow, middle, lower + margin)
        self._target_upper = np.where(narrow, middle, upper - margin)
        self._velocity_max = np.array([joint.limits.velocity for joint in joints])
        self._effort_max = np.array([joint.limits.effort for joint in joints])
        self._kp = np.array([joint.gains.kp for joint in joints])
        self._kd = np.array([joint.gains.kd for joint in joints])
        self._kp_max = config.gain_ratio_max * self._kp
        self._kd_max = config.gain_ratio_max * self._kd
        defaults = np.array([joint.default_position for joint in joints])
        self._default_targets = np.clip(
            defaults, self._target_lower, self._target_upper
        )
        self._zeros = np.zeros(len(joints))
        self._state_timeout_ns = round(config.state_timeout_s * 1e9)
        self._hot_c = config.temperature_limit_c - config.temperature_margin_c

        self._last_safe: JointCommand | None = None
        self._hold: JointCommand | None = None
        self._latched_by: tuple[Fault, ...] = ()
        self._fault_counts: dict[str, int] = {}
        self._clamp_counts = np.zeros(len(joints), dtype=np.int64)

    @property
    def description(self) -> RobotDescription:
        """The description the limits come from."""
        return self._description

    @property
    def config(self) -> SafetyConfig:
        """The thresholds in use."""
        return self._config

    @property
    def latched(self) -> bool:
        """Whether HOLD is latched, until `reset`."""
        return bool(self._latched_by)

    @property
    def latched_by(self) -> tuple[Fault, ...]:
        """The faults that latched HOLD, from the tick that latched it; empty
        if HOLD is not latched."""
        return self._latched_by

    @property
    def stats(self) -> SafetyStats:
        """Fault counts by code and clamp counts by joint since the layer was
        created, as a read-only snapshot. `reset` does not clear them."""
        clamps = dict(zip(self._names, self._clamp_counts.tolist(), strict=True))
        return SafetyStats(
            faults=MappingProxyType(dict(self._fault_counts)),
            clamps=MappingProxyType(clamps),
        )

    def preflight(self, state: RobotState) -> tuple[Fault, ...]:
        """Return the reasons a policy may not start from `state`; empty if none.

        The runner must not enter POLICY while this returns any fault. It
        reports, in this order: a non-finite value in the state
        (`state.non_finite`); each joint outside its position limits
        (`joint.out_of_limits`); each hot motor (`motor.hot`); and every
        fault the adapter reports in the state. It does not latch, count or
        change anything.

        Raises ValueError if the state does not have one value per joint.
        """
        self._check_state_size(state)
        faults: list[Fault] = []
        non_finite = self._state_non_finite(state)
        if non_finite is not None:
            faults.append(non_finite)
        position = state.position
        outside = (position < self._lower) | (position > self._upper)
        for index in np.flatnonzero(outside).tolist():
            name = self._names[index]
            lower, upper = self._description.joints[index].limits.position
            faults.append(
                Fault(
                    code=JOINT_OUT_OF_LIMITS,
                    message=f"joint {name!r} is at {position[index]:.4g} rad, outside"
                    f" its position limits [{lower:g}, {upper:g}] rad",
                    joint=name,
                )
            )
        faults += self._hot_motors(state)
        faults += state.faults
        return tuple(faults)

    def check(
        self,
        state: RobotState,
        *,
        now_ns: int,
        command_age_s: float,
        overrun: bool = False,
    ) -> Assessment:
        """Check this tick's inputs, before the policy runs.

        `now_ns` is the time now, on the clock that stamps the state:
        `time.monotonic_ns()` on hardware, simulation time in lockstep.
        `command_age_s` is the time since the velocity command was last
        updated, in s, and `math.inf` if it never has been. `overrun` says
        whether the previous tick took longer than its period.

        Looks for, in this order: a stale state (`state.stale`); a
        non-finite state (`state.non_finite`); a stale velocity command
        (`command.stale`), including an age that is not a number; each hot
        motor (`motor.hot`); an overrun (`loop.overrun`); and the faults
        the adapter reports in the state, as it reports them. The action is
        the most severe of the faults' actions in `FAULT_ACTIONS`, with
        `config.unknown_fault_action` for adapter codes not in it, and HOLD
        while HOLD is latched. A HOLD latches it.

        Raises ValueError if the state does not have one value per joint.
        """
        self._check_state_size(state)
        faults: list[Fault] = []
        age_ns = now_ns - state.stamp_ns
        if age_ns > self._state_timeout_ns:
            faults.append(
                Fault(
                    code=STATE_STALE,
                    message=f"the state is {age_ns / 1e6:.1f} ms old, more than the"
                    f" {self._config.state_timeout_s * 1e3:g} ms allowed",
                )
            )
        non_finite = self._state_non_finite(state)
        if non_finite is not None:
            faults.append(non_finite)
        timeout_s = self._config.command_timeout_s
        if not command_age_s <= timeout_s:  # true for NaN too
            faults.append(
                Fault(
                    code=COMMAND_STALE,
                    message=f"{_age(command_age_s)}; the limit is {timeout_s:g} s",
                )
            )
        faults += self._hot_motors(state)
        if overrun:
            faults.append(
                Fault(
                    code=LOOP_OVERRUN,
                    message="the previous control tick took longer than its period",
                )
            )
        faults += state.faults
        return Assessment(tuple(faults), self._act(faults))

    def filter(self, command: JointCommand, state: RobotState) -> SafetyResult:
        """Make the command built from the policy's output safe to write.

        If any value of `command` is NaN or infinite, reports
        `policy.non_finite`, takes the HOLD action, which latches it, and
        returns `hold_command(state)`. While HOLD is latched, returns the
        hold command too.

        Otherwise returns a new command, which becomes the last safe one:
        position targets clamped to the joint's limits less
        `config.position_margin_rad` (to the range's midpoint where it is
        narrower than twice the margin); velocity targets clipped to ± the
        joint's velocity limit; `effort` clipped to ± its effort limit; and
        `kp` and `kd` capped at `config.gain_ratio_max` times the
        description's. Clamping is not a fault: it is counted per joint in
        `stats`.

        Raises ValueError if the command does not have one value per joint,
        or has a negative gain: the runner builds the gains from the policy
        descriptor and the description, so these are wiring mistakes, not
        faults of the policy.
        """
        self._check_size(command.position.size, "the command")
        fields = (
            ("position", command.position),
            ("velocity", command.velocity),
            ("effort", command.effort),
            ("kp", command.kp),
            ("kd", command.kd),
        )
        if not _all_finite([array for _, array in fields]):
            fault = Fault(
                code=POLICY_NON_FINITE,
                message=f"the policy's command is not finite: {self._where(fields)}",
            )
            return SafetyResult(self.hold_command(state), (fault,), self._act([fault]))
        if self._latched_by:
            return SafetyResult(self.hold_command(state), (), SafetyAction.HOLD)
        for name, gains in (("kp", command.kp), ("kd", command.kd)):
            negative = gains < 0
            if negative.any():
                raise ValueError(
                    f"{name} is negative for {self._joints(negative)}: gains come"
                    " from the policy descriptor and the description, which"
                    " cannot hold negative ones"
                )
        position = np.clip(command.position, self._target_lower, self._target_upper)
        velocity = np.clip(command.velocity, -self._velocity_max, self._velocity_max)
        effort = np.clip(command.effort, -self._effort_max, self._effort_max)
        kp = np.minimum(command.kp, self._kp_max)
        kd = np.minimum(command.kd, self._kd_max)
        self._clamp_counts += (
            (position != command.position)
            | (velocity != command.velocity)
            | (effort != command.effort)
            | (kp != command.kp)
            | (kd != command.kd)
        )
        safe = JointCommand(
            position=position, velocity=velocity, effort=effort, kp=kp, kd=kd
        )
        self._last_safe = safe
        self._hold = None
        return SafetyResult(safe, (), SafetyAction.NONE)

    def hold_command(self, state: RobotState) -> JointCommand:
        """Return the command that holds the robot where it is safe.

        It has the last safe command's position targets and gains, with
        zero velocity and zero effort. Before any safe command, it holds the
        measured positions instead, clamped into the joints' limits less the
        margin, with the description's default gains; a joint whose
        measured position is not finite gets its default position. That
        first hold command is remembered as the last safe one, so the
        targets do not follow the robot if it sags.

        The same object is returned until `filter` produces a new safe
        command or `reset` is called. Raises ValueError if the state is
        needed and does not have one value per joint.
        """
        if self._hold is not None:
            return self._hold
        if self._last_safe is None:
            self._check_state_size(state)
            measured = state.position
            targets = np.where(
                np.isfinite(measured),
                np.clip(measured, self._target_lower, self._target_upper),
                self._default_targets,
            )
            hold = JointCommand(
                position=targets,
                velocity=self._zeros,
                effort=self._zeros,
                kp=self._kp,
                kd=self._kd,
            )
            self._last_safe = hold
        else:
            last = self._last_safe
            hold = JointCommand(
                position=last.position,
                velocity=self._zeros,
                effort=self._zeros,
                kp=last.kp,
                kd=last.kd,
            )
        self._hold = hold
        return hold

    def reset(self) -> None:
        """Clear the HOLD latch and forget the last safe command.

        Only a person, or the runner's start sequence after `preflight`
        passes, should call it. The next hold command holds the positions
        measured then. `stats` keeps counting.
        """
        self._latched_by = ()
        self._last_safe = None
        self._hold = None

    def _act(self, faults: list[Fault]) -> SafetyAction:
        """Count `faults`, latch HOLD if one calls for it, and return the action."""
        actions = [self._action_of(fault) for fault in faults]
        for fault in faults:
            self._fault_counts[fault.code] = self._fault_counts.get(fault.code, 0) + 1
        if not self._latched_by and SafetyAction.HOLD in actions:
            self._latched_by = tuple(
                fault
                for fault, action in zip(faults, actions, strict=True)
                if action is SafetyAction.HOLD
            )
        if self._latched_by:
            return SafetyAction.HOLD
        return most_severe(actions)

    def _action_of(self, fault: Fault) -> SafetyAction:
        return FAULT_ACTIONS.get(fault.code, self._config.unknown_fault_action)

    def _hot_motors(self, state: RobotState) -> list[Fault]:
        """Return a `motor.hot` fault for each hot motor, in joint order.

        A temperature that is not finite is a broken reading, reported as
        `state.non_finite`, not a hot motor.
        """
        temperature = state.temperature
        if temperature is None:
            return []
        hot = (temperature >= self._hot_c) & np.isfinite(temperature)
        if not hot.any():
            return []
        config = self._config
        return [
            Fault(
                code=MOTOR_HOT,
                message=f"the motor of joint {self._names[index]!r} is at"
                f" {temperature[index]:.1f} °C, at or above {self._hot_c:g} °C"
                f" (the {config.temperature_limit_c:g} °C limit less a"
                f" {config.temperature_margin_c:g} °C margin)",
                joint=self._names[index],
            )
            for index in np.flatnonzero(hot).tolist()
        ]

    def _state_non_finite(self, state: RobotState) -> Fault | None:
        """Return a `state.non_finite` fault if any value of `state` is not finite."""
        imu = state.imu
        fields = [
            ("position", state.position),
            ("velocity", state.velocity),
            ("effort", state.effort),
            ("imu.orientation", imu.orientation),
            ("imu.angular_velocity", imu.angular_velocity),
            ("imu.linear_acceleration", imu.linear_acceleration),
        ]
        if state.temperature is not None:
            fields.append(("temperature", state.temperature))
        if _all_finite([array for _, array in fields]):
            return None
        return Fault(
            code=STATE_NON_FINITE,
            message=f"the state is not finite: {self._where(fields)}",
        )

    def _where(self, fields: Iterable[tuple[str, NDArray[np.float64]]]) -> str:
        """Name the fields, and the joints of joint fields, that are not finite."""
        parts: list[str] = []
        for name, values in fields:
            bad = ~np.isfinite(values)
            if not bad.any():
                continue
            if name.startswith("imu."):
                parts.append(name)
            else:
                parts.append(f"{name} for {self._joints(bad)}")
        return "; ".join(parts)

    def _joints(self, mask: NDArray[np.bool_]) -> str:
        """Name the joints where `mask` is true, in description order."""
        names = [repr(self._names[i]) for i in np.flatnonzero(mask).tolist()]
        return ("joint " if len(names) == 1 else "joints ") + ", ".join(names)

    def _check_state_size(self, state: RobotState) -> None:
        self._check_size(state.position.size, "the state")
        if state.temperature is not None:
            self._check_size(state.temperature.size, "the state's temperature")

    def _check_size(self, size: int, what: str) -> None:
        if size != len(self._names):
            raise ValueError(
                f"{what} has {size} joint values, but robot"
                f" {self._description.name!r} has {len(self._names)} joints"
            )


def _all_finite(arrays: list[NDArray[np.float64]]) -> bool:
    """Return whether every value of every array is finite.

    One check of the arrays joined is about three times faster than one
    check per array, for arrays as short as a robot's.
    """
    return bool(np.isfinite(np.concatenate(arrays)).all())


def _age(seconds: float) -> str:
    """Describe the age of the velocity command for a message."""
    if math.isnan(seconds):
        return "the velocity command's age is not a number"
    if math.isinf(seconds):
        return "the velocity command has never been updated"
    return f"the velocity command was last updated {seconds:g} s ago"
