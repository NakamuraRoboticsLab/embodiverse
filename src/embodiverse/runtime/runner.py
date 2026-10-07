"""The runner: runs a policy on a robot, one tick at a time, at a fixed rate.

On every tick, at the descriptor's `rate_hz`, the runner reads the robot's
state, has the safety layer check it, builds the policy's observation, runs
the policy, turns its action into a command for every joint, has the safety
layer filter that command, and writes it to the robot. While a policy runs,
the runner is the robot's one commander: nothing else writes to it.

The runner receives the robot already built and never imports an adapter,
so the same code runs a policy in simulation and on hardware.
How time passes depends on the robot:

- A robot that is also `Steppable`, a simulator in lockstep, is advanced by
  one period after each tick's command is written. Time is simulated, so a
  run is reproducible and goes as fast as the machine allows.
- Any other robot runs in real time. Ticks start at deadlines one period
  apart on the monotonic clock, and the runner sleeps until each. A tick
  that ends after its deadline is an overrun, which the next tick reports
  to the safety layer.

`Runner.start` checks that the robot may start (`SafetyLayer.preflight`),
enters HOLD and then POLICY, and moves every joint's target from where the
joint is to its default position over `RunnerConfig.start_ramp_s`, before
the policy's first tick. `Runner.stop` leaves the robot in HOLD. The runner
never damps the robot and never calls `Robot.operator_stop`: only a person
stops a robot that way. When one does, the robot leaves POLICY, and the
runner stops running the policy.

A `TickRecorder`, if one is given, receives every tick on which the runner
writes a command, start ramp included: the state read, the command written
and the faults the safety layer reported. `McapRecorder`, in
`embodiverse.runtime.recording`, records them to a file.

docs/policies.md describes the loop and the command line.
"""

import contextlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    Gains,
    ObservationTerm,
    PolicyDescriptor,
    PositionAction,
    RobotRef,
    VelocityCommand,
    VelocityRanges,
    resolve_joints,
    validate_against,
)
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.observation import TermInputs
from embodiverse.contracts.robot import Robot, Steppable
from embodiverse.contracts.types import Fault, JointCommand, Mode, RobotState
from embodiverse.runtime.observations import ObservationBuilder
from embodiverse.runtime.policy import Policy
from embodiverse.runtime.safety import SafetyAction, SafetyLayer

__all__ = [
    "CommandSource",
    "ConstantCommand",
    "RunStats",
    "Runner",
    "RunnerConfig",
    "StartRefused",
    "TickRecorder",
    "VelocityCommandSample",
    "zero_policy_descriptor",
]


@dataclass(frozen=True, slots=True, eq=False)
class VelocityCommandSample:
    """A velocity command for the policy to follow, and when it was last updated.

    As for the other runtime data, `command` is a read-only
    float64 copy, and construction checks its structure only, so a
    non-finite value passes through as data. Raises ValueError if `command`
    does not have three elements.
    """

    command: NDArray[np.float64]
    """(lin_vel_x, lin_vel_y, ang_vel_z), in m/s and rad/s, in the base frame:
    x forward, y left, z up."""

    stamp_ns: int
    """When the command was last updated, in ns, on the clock of the `now_ns`
    that `CommandSource.read` receives."""

    def __post_init__(self) -> None:
        command = np.array(self.command, dtype=np.float64)
        if command.shape != (3,):
            raise ValueError(
                "a velocity command has 3 values (lin_vel_x, lin_vel_y, ang_vel_z),"
                f" got shape {command.shape}"
            )
        command.flags.writeable = False
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "stamp_ns", int(self.stamp_ns))


@runtime_checkable
class CommandSource(Protocol):
    """Where the velocity command the policy follows comes from.

    The runner calls `read` once per tick, from the control thread. It must
    not block: a source fed from elsewhere, such as a joystick or a planner,
    returns the latest command it has received and when it received it.
    """

    def read(self, now_ns: int) -> VelocityCommandSample:
        """Return the latest velocity command.

        `now_ns` is the time of this tick: simulation time in lockstep, and
        `time.monotonic_ns()` in real time. The sample's `stamp_ns` is on
        the same clock, and `now_ns - stamp_ns` is the command's age, which
        the safety layer compares with its command timeout.
        """
        ...


class ConstantCommand:
    """A velocity command that never changes, and so is never stale.

    `read` stamps the command with the time it is given, so its age is
    always zero.
    """

    def __init__(
        self, lin_vel_x: float = 0.0, lin_vel_y: float = 0.0, ang_vel_z: float = 0.0
    ) -> None:
        self._command = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float64)

    def read(self, now_ns: int) -> VelocityCommandSample:
        """Return the command, stamped `now_ns`."""
        return VelocityCommandSample(command=self._command, stamp_ns=now_ns)


class TickRecorder(Protocol):
    """Receives what happened on each tick, to record it.

    The runner calls `record_tick` from the control thread, once for every
    tick of the start ramp and of the policy on which it writes a command,
    right after writing it. An error it raises ends the run as any other
    error does, leaving the robot in HOLD. The caller creates the recorder
    and closes it after the run; the runner never closes it.
    """

    def record_tick(
        self, state: RobotState, command: JointCommand, faults: tuple[Fault, ...]
    ) -> None:
        """Record one tick.

        `state` is the state the tick read, `command` the command it wrote
        to the robot, after the safety layer, and `faults` what the safety
        layer reported on the tick: those of `SafetyLayer.check`, then
        those of `SafetyLayer.filter`, if the command was filtered.
        """
        ...


class RunnerConfig(BaseModel):
    """How the runner starts a policy.

    Raises pydantic's ValidationError, a ValueError, for an unknown field,
    a non-finite number or a negative ramp time.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, use_attribute_docstrings=True
    )

    start_ramp_s: float = Field(default=2.0, ge=0)
    """Time over which `Runner.start` moves every joint's target from its
    measured position to its default position, in s, before the policy's
    first tick. 0 skips the ramp.

    A policy is trained to act from around its default pose, and its first
    command assumes it. A robot started from another pose, such as one
    lowered from a gantry or held in HOLD where it happened to be, would
    otherwise be pulled to the policy's first targets in one tick, at the
    gains' full torque. 2 s moves a joint that is 1 rad away at 0.5 rad/s.
    """


_DEFAULT_CONFIG = RunnerConfig()


class StartRefused(RuntimeError):
    """A policy may not start on the robot; the message lists the reasons."""


@dataclass(frozen=True, slots=True)
class RunStats:
    """What happened during one `Runner.run`."""

    ticks: int
    """Policy ticks run, each one state read and one command written."""

    duration_s: float
    """How long the run took, in s: simulated time in lockstep; in real
    time, monotonic time from the first tick's start to the end of the last
    tick's period."""

    rate_hz: float
    """Ticks per second of `duration_s`: the rate achieved."""

    jitter_s: float
    """Standard deviation of the intervals between the starts of consecutive
    ticks, in s. 0 in lockstep, where each interval is one period of
    simulated time, and with fewer than two ticks."""

    max_compute_s: float
    """Longest time one tick took, from reading the state to writing the
    command, in s, on the monotonic clock. In real time it must stay well
    below the period, or ticks overrun."""

    overruns: int
    """Ticks that ended after their deadline; always 0 in lockstep."""

    faults: Mapping[str, int]
    """How many times each fault code was reported during the run, by
    `SafetyLayer.check` and `SafetyLayer.filter`."""

    latched: bool
    """Whether HOLD was latched at the end of the run."""

    latched_by: tuple[Fault, ...]
    """The faults that latched HOLD; empty if it is not latched."""

    lockstep: bool
    """Whether the robot ran in lockstep, so that `duration_s` is simulated
    time."""

    left_policy: Mode | None
    """The mode the robot was found in after leaving POLICY, which ended the
    run early, such as DAMP after a person's stop; None if it stayed in
    POLICY."""


class Runner:
    """Runs one policy on one robot, tick by tick, at the descriptor's rate.

    The runner owns the robot while the policy runs, but not its lifetime:
    the caller connects the robot before `start` or `run`, and closes it
    after `stop`. The runner never closes it, and never calls
    `operator_stop`.

    When it is created, it checks everything that can be checked before the
    robot is touched, so a mismatch is refused before anything moves: the
    descriptor against the robot's description (`validate_against`, which
    compares the description's content hash), the safety layer's
    description, and the policy's sizes. It builds the observation builder,
    and works out, once, each joint's default position and gains: the
    descriptor's for the joints the action drives (explicit, or the
    description's), and the description's for the held joints.

    `safety` defaults to a `SafetyLayer` with the default thresholds of
    `SafetyConfig` (docs/policies.md); `commands` defaults to
    a zero `ConstantCommand`. `recorder`, if given, receives every tick on
    which a command is written, start ramp included.

    Raises DescriptorError if the descriptor does not fit the robot's
    description; ValueError if the safety layer was made for another
    description or the policy's sizes differ from the descriptor's; and
    what `ObservationBuilder` raises.
    """

    def __init__(
        self,
        robot: Robot,
        policy: Policy,
        descriptor: PolicyDescriptor,
        *,
        safety: SafetyLayer | None = None,
        commands: CommandSource | None = None,
        config: RunnerConfig = _DEFAULT_CONFIG,
        recorder: TickRecorder | None = None,
    ) -> None:
        description = robot.description
        validate_against(descriptor, description)
        if safety is None:
            safety = SafetyLayer(description)
        elif content_hash(safety.description) != content_hash(description):
            raise ValueError(
                f"the safety layer was made for a description of robot"
                f" {safety.description.name!r} other than the robot's: its limits"
                " would not be the robot's"
            )
        builder = ObservationBuilder(descriptor, description)
        driven = resolve_joints(descriptor.action.joints, description)
        _check_sizes(policy, descriptor, builder.size, len(driven))

        self._robot = robot
        self._policy = policy
        self._descriptor = descriptor
        self._safety = safety
        self._commands: CommandSource = (
            ConstantCommand() if commands is None else commands
        )
        self._builder = builder
        self._recorder = recorder

        joints = description.joints
        self._defaults = np.array([joint.default_position for joint in joints])
        self._driven = np.array(
            [description.index_of(name) for name in driven], dtype=np.intp
        )
        self._scale = np.broadcast_to(
            np.array(descriptor.action.scale, dtype=np.float64), (len(driven),)
        ).copy()
        # Held joints, and driven ones without explicit gains, use the
        # description's gains.
        self._kp = np.array([joint.gains.kp for joint in joints])
        self._kd = np.array([joint.gains.kd for joint in joints])
        if descriptor.gains.source == "explicit":
            values = descriptor.gains.values
            self._kp[self._driven] = [values[name].kp for name in driven]
            self._kd[self._driven] = [values[name].kd for name in driven]
        self._zeros = np.zeros(description.num_joints)
        ranges = descriptor.command.ranges
        bounds = np.array([ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z])
        self._command_lower, self._command_upper = bounds[:, 0], bounds[:, 1]
        self._no_command = np.zeros(3)

        rate_hz = descriptor.rate_hz
        self._rate_hz = rate_hz
        self._period_s = 1.0 / rate_hz
        self._period_ns = round(1e9 / rate_hz)
        ramp = config.start_ramp_s
        self._ramp_ticks = max(1, round(ramp * rate_hz)) if ramp > 0 else 0
        self._clock: _Clock = (
            _Lockstep(robot, self._period_s)
            if isinstance(robot, Steppable)
            else _RealTime(self._period_ns)
        )

        self._started = False
        self._last_action = np.zeros(len(driven))
        self._ticks_since_start = 0
        self._overrun = False

    @property
    def safety(self) -> SafetyLayer:
        """The safety layer every tick passes through."""
        return self._safety

    @property
    def lockstep(self) -> bool:
        """Whether the robot runs in lockstep: it is `Steppable`."""
        return isinstance(self._clock, _Lockstep)

    @property
    def started(self) -> bool:
        """Whether the policy has started and not stopped since."""
        return self._started

    def start(self) -> None:
        """Take control of the robot and bring it to the policy's default pose.

        1. Reads the robot's state; the robot must be in PASSIVE or HOLD.
        2. `SafetyLayer.preflight` must find no fault in that state.
        3. Resets the safety layer, enters HOLD, then POLICY.
        4. Over `RunnerConfig.start_ramp_s`, at the policy's rate, writes
           commands whose targets move every joint linearly from the
           position measured when the ramp began to its default position,
           with the gains the policy will use: the descriptor's for driven
           joints, the description's for held ones. Each tick is checked
           and filtered by the safety layer, as a policy tick is, with a
           fresh velocity command.
        5. Resets the safety layer, the observation builder and the policy,
           so the policy starts from its first tick.

        Raises StartRefused, a RuntimeError, listing the reasons: if the
        robot is in another mode or preflight finds faults, before anything
        changes on the robot; and if a fault holds the robot during the
        ramp, which leaves it in HOLD, or the robot leaves POLICY during
        it, after a person's stop. Raises RuntimeError if the policy has
        already started, and what the robot raises, such as RuntimeError
        if it is not connected.
        """
        if self._started:
            raise RuntimeError("the policy has already started; call stop() first")
        robot, safety = self._robot, self._safety
        state = robot.read()
        if state.mode not in (Mode.PASSIVE, Mode.HOLD):
            reason = f"the robot is in {state.mode.name}, and a policy starts only"
            reason += " from PASSIVE or HOLD"
            if state.mode is Mode.DAMP:
                reason += "; after a person's stop, only a person resets the robot"
            raise StartRefused(self._refusal([reason]))
        faults = safety.preflight(state)
        if faults:
            raise StartRefused(self._refusal(_describe(faults)))
        safety.reset()
        robot.set_mode(Mode.HOLD)
        try:
            robot.set_mode(Mode.POLICY)
            self._clock.begin(state)
            self._overrun = False
            self._ramp()
        except BaseException:
            # Whatever stopped the ramp, leave the robot held, then report it.
            with contextlib.suppress(Exception):
                self._leave_policy()
            raise
        safety.reset()
        self._builder.reset()
        self._policy.reset()
        self._last_action = np.zeros(self._policy.output_size)
        self._ticks_since_start = 0
        self._started = True

    def run(self, seconds: float) -> RunStats:
        """Run the policy for `seconds`, starting it first if it has not started.

        Runs `round(seconds * rate_hz)` ticks, at least one. Ends early if
        the robot leaves POLICY, as it does when a person stops it; the
        policy is then stopped, and `RunStats.left_policy` says the mode the
        robot was found in. If anything raises during the run, the robot is
        left in HOLD, as `stop` leaves it, before the error is passed on.

        Raises ValueError if `seconds` is not a positive finite number, and
        what `start` raises.
        """
        if not (math.isfinite(seconds) and seconds > 0):
            raise ValueError(f"seconds must be a positive number, got {seconds!r}")
        ticks = max(1, round(seconds * self._rate_hz))
        if not self._started:
            self.start()
        try:
            return self._run(ticks)
        except BaseException:
            # Whatever went wrong, leave the robot held, then report it.
            with contextlib.suppress(Exception):
                self.stop()
            raise

    def stop(self) -> None:
        """Stop running the policy, and leave the robot in HOLD.

        If the policy has started and the robot is in POLICY, changes the
        robot to HOLD, which holds the positions measured then with the
        description's gains. Does nothing if the policy has not started, so
        a refused start leaves the robot as it was. A robot that a person
        stopped stays in DAMP, which only a person leaves. The policy can be
        started again afterwards. Idempotent.
        """
        if not self._started:
            return
        self._started = False
        self._leave_policy()

    # --- Internals -----------------------------------------------------------

    def _leave_policy(self) -> None:
        """Change the robot to HOLD if it is in POLICY."""
        if self._robot.read().mode is Mode.POLICY:
            self._robot.set_mode(Mode.HOLD)

    def _ramp(self) -> None:
        """Move every joint's target from its measured to its default position."""
        robot, safety, clock = self._robot, self._safety, self._clock
        start = self._defaults  # replaced by the positions on the first tick
        for tick in range(1, self._ramp_ticks + 1):
            clock.wait()
            state = robot.read()
            if state.mode is not Mode.POLICY:
                mode = state.mode.name
                reason = f"the robot left POLICY during the start ramp: it is in {mode}"
                raise StartRefused(self._refusal([reason]))
            now_ns = clock.now_ns(state)
            assessment = safety.check(
                state,
                now_ns=clock.on_state_clock(now_ns),
                command_age_s=0.0,  # the ramp follows no velocity command
                overrun=self._overrun,
            )
            if assessment.action is SafetyAction.HOLD:
                # start() then changes the robot to HOLD.
                self._write(state, safety.hold_command(state), assessment.faults)
                faults = assessment.faults or safety.latched_by
                reasons = ["a fault held the robot during the start ramp"]
                raise StartRefused(self._refusal(reasons + _describe(faults)))
            if tick == 1:
                start = state.position
            fraction = tick / self._ramp_ticks
            command = JointCommand(
                position=start + fraction * (self._defaults - start),
                velocity=self._zeros,
                effort=self._zeros,
                kp=self._kp,
                kd=self._kd,
            )
            result = safety.filter(command, state)
            self._write(state, result.command, assessment.faults + result.faults)
            self._overrun = clock.end_tick()

    def _run(self, ticks: int) -> RunStats:
        """Run up to `ticks` policy ticks and return what happened."""
        robot, clock = self._robot, self._clock
        lockstep = self.lockstep
        faults: dict[str, int] = {}
        starts: list[int] = []
        max_compute_ns = 0
        overruns = 0
        left_policy: Mode | None = None
        for _ in range(ticks):
            started_ns = clock.wait()
            state = robot.read()
            if state.mode is not Mode.POLICY:
                left_policy = state.mode
                self._started = False
                break
            starts.append(state.stamp_ns if lockstep else started_ns)
            self._tick(state, faults)
            max_compute_ns = max(max_compute_ns, time.monotonic_ns() - started_ns)
            self._overrun = clock.end_tick()
            overruns += self._overrun
        ended_ns = clock.finish()
        done = len(starts)
        if lockstep:
            duration_ns = done * self._period_ns
        else:
            duration_ns = ended_ns - starts[0] if starts else 0
        intervals = np.diff(np.array(starts, dtype=np.float64)) / 1e9
        safety = self._safety
        return RunStats(
            ticks=done,
            duration_s=duration_ns / 1e9,
            rate_hz=done / (duration_ns / 1e9) if duration_ns > 0 else 0.0,
            jitter_s=float(np.std(intervals)) if intervals.size else 0.0,
            max_compute_s=max_compute_ns / 1e9,
            overruns=overruns,
            faults=MappingProxyType(faults),
            latched=safety.latched,
            latched_by=safety.latched_by,
            lockstep=lockstep,
            left_policy=left_policy,
        )

    def _tick(self, state: RobotState, faults: dict[str, int]) -> None:
        """Run one tick on `state`, which the robot reported in POLICY."""
        safety, clock = self._safety, self._clock
        now_ns = clock.now_ns(state)
        sample = self._commands.read(now_ns)
        assessment = safety.check(
            state,
            now_ns=clock.on_state_clock(now_ns),
            command_age_s=(now_ns - sample.stamp_ns) / 1e9,
            overrun=self._overrun,
        )
        _count(faults, assessment.faults)
        elapsed_s = self._ticks_since_start * self._period_s
        self._ticks_since_start += 1
        if assessment.action is SafetyAction.HOLD:
            self._write(state, safety.hold_command(state), assessment.faults)
            return
        if assessment.action is SafetyAction.ZERO_VELOCITY:
            command = self._no_command
        else:
            command = np.clip(sample.command, self._command_lower, self._command_upper)
        inputs = TermInputs(
            state=state,
            command=command,
            last_action=self._last_action,
            elapsed_s=elapsed_s,
        )
        action = self._policy(self._builder.build(inputs))
        if action.shape != self._last_action.shape:
            raise ValueError(
                f"the policy returned an action of shape {action.shape}, but its"
                f" output size is {self._last_action.size}"
            )
        position = self._defaults.copy()
        position[self._driven] += self._scale * action
        joint_command = JointCommand(
            position=position,
            velocity=self._zeros,
            effort=self._zeros,
            kp=self._kp,
            kd=self._kd,
        )
        result = safety.filter(joint_command, state)
        _count(faults, result.faults)
        self._write(state, result.command, assessment.faults + result.faults)
        self._last_action = action.astype(np.float64)

    def _write(
        self, state: RobotState, command: JointCommand, faults: tuple[Fault, ...]
    ) -> None:
        """Write `command`, which the tick that read `state` built, and record it."""
        self._robot.write(command)
        if self._recorder is not None:
            self._recorder.record_tick(state, command, faults)

    def _refusal(self, reasons: list[str]) -> str:
        name = self._robot.description.name
        lines = "".join(f"\n- {reason}" for reason in reasons)
        return f"cannot start the policy on robot {name!r}:{lines}"


def zero_policy_descriptor(
    description: RobotDescription, *, rate_hz: float = 50.0
) -> PolicyDescriptor:
    """Return a descriptor for running `ZeroPolicy` on the described robot.

    It names no policy file, because the zero policy has none. Every joint
    is driven, in description order, with scale 1 and the description's
    gains, so the zero policy's action holds every joint at its default
    position. The observation is the projected gravity, 3 values, which the
    zero policy ignores but the observation builder still computes. Every
    command range is zero, so the robot is never asked to walk. It names the
    description's robot and content hash, and its provenance is external.

    Run it with `ZeroPolicy(3, description.num_joints)`. The projected
    gravity needs the IMU on the base link, as the observation builder
    checks, and the runner checks the rate against the robot's rates.
    Raises pydantic's ValidationError, a ValueError, if `rate_hz` is not a
    positive finite number.
    """
    no_velocity = (0.0, 0.0)
    return PolicyDescriptor(
        schema_version=1,
        robot=RobotRef(
            name=description.name, description_hash=content_hash(description)
        ),
        rate_hz=rate_hz,
        command=VelocityCommand(
            type="velocity",
            ranges=VelocityRanges(
                lin_vel_x=no_velocity, lin_vel_y=no_velocity, ang_vel_z=no_velocity
            ),
        ),
        observations=(ObservationTerm(term="projected_gravity"),),
        observation_size=3,
        action=PositionAction(
            type="position_offset", joints=description.joint_names, scale=1.0
        ),
        gains=Gains(source="description"),
        provenance="external",
    )


# --- How time passes --------------------------------------------------------------


class _Clock(Protocol):
    """How ticks are timed: in lockstep with a simulator, or in real time."""

    def begin(self, state: RobotState) -> None:
        """Start timing, from a state just read."""
        ...

    def wait(self) -> int:
        """Wait until the next tick is due; return its start, monotonic, in ns."""
        ...

    def now_ns(self, state: RobotState) -> int:
        """Return the time of the tick that read `state`, for the commands."""
        ...

    def on_state_clock(self, now_ns: int) -> int:
        """Return `now_ns` on the clock that stamps the robot's state."""
        ...

    def end_tick(self) -> bool:
        """End a tick whose command was written; return whether it overran."""
        ...

    def finish(self) -> int:
        """End a run after its last tick; return the time, monotonic, in ns."""
        ...


class _Lockstep:
    """Lockstep: the simulator advances one period after each tick's write.

    Time is the simulation's: the tick's time is the state's stamp, and a
    tick cannot overrun, however long it takes on the machine.
    """

    def __init__(self, robot: Steppable, period_s: float) -> None:
        self._robot = robot
        self._period_s = period_s

    def begin(self, state: RobotState) -> None:
        pass

    def wait(self) -> int:
        return time.monotonic_ns()

    def now_ns(self, state: RobotState) -> int:
        return state.stamp_ns

    def on_state_clock(self, now_ns: int) -> int:
        return now_ns

    def end_tick(self) -> bool:
        self._robot.advance(self._period_s)
        return False

    def finish(self) -> int:
        return time.monotonic_ns()


class _RealTime:
    """Real time: ticks start at deadlines one period apart, on the monotonic clock.

    A tick is due one period after the previous one was due, and the clock
    sleeps until then. A tick that ends after its deadline overran; the
    next one then starts at once, and the deadlines follow from it, so the
    ticks never bunch up to catch up. A run that starts after the previous
    one's deadlines have passed also starts at once.

    The state may be stamped on another clock than the monotonic one: a
    simulator running in real time stamps it with simulation time. `begin`
    therefore notes a state's stamp and the monotonic time together, and the
    time on the state's clock is that stamp plus the monotonic time elapsed
    since. For a robot that stamps its state on the monotonic clock, as
    hardware does, this is the monotonic time less the age of that first
    state.
    """

    def __init__(self, period_ns: int) -> None:
        self._period_ns = period_ns
        self._offset_ns = 0
        self._next_ns = 0
        self._deadline_ns = 0

    def begin(self, state: RobotState) -> None:
        self._offset_ns = time.monotonic_ns() - state.stamp_ns
        self._next_ns = 0

    def wait(self) -> int:
        now = time.monotonic_ns()
        if now < self._next_ns:
            time.sleep((self._next_ns - now) / 1e9)
            due = self._next_ns
        else:
            due = now
        self._deadline_ns = self._next_ns = due + self._period_ns
        return time.monotonic_ns()

    def now_ns(self, state: RobotState) -> int:
        return time.monotonic_ns()

    def on_state_clock(self, now_ns: int) -> int:
        return now_ns - self._offset_ns

    def end_tick(self) -> bool:
        return time.monotonic_ns() > self._deadline_ns

    def finish(self) -> int:
        now = time.monotonic_ns()
        if now < self._next_ns:
            time.sleep((self._next_ns - now) / 1e9)
        return time.monotonic_ns()


def _check_sizes(
    policy: Policy, descriptor: PolicyDescriptor, input_size: int, output_size: int
) -> None:
    """Check the policy's sizes against those the descriptor gives."""
    problems: list[str] = []
    if policy.input_size != input_size:
        problems.append(
            f"it takes {policy.input_size} values, but the descriptor's observation"
            f" has {input_size}: observation_size {descriptor.observation_size}"
            f" times history length {descriptor.history.length}"
        )
    if policy.output_size != output_size:
        problems.append(
            f"it returns {policy.output_size} values, but the descriptor's action"
            f" drives {output_size} joints"
        )
    if problems:
        raise ValueError(
            "the policy does not fit its descriptor: " + "; ".join(problems)
        )


def _count(counts: dict[str, int], faults: tuple[Fault, ...]) -> None:
    for fault in faults:
        counts[fault.code] = counts.get(fault.code, 0) + 1


def _describe(faults: tuple[Fault, ...]) -> list[str]:
    return [f"{fault.code}: {fault.message}" for fault in faults]
