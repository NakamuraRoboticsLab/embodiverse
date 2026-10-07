"""Policy descriptors: how to run a trained policy, in one validated YAML file.

A policy file holds weights and nothing else. The descriptor that ships with
it states what the weights cannot: the robot description the policy was
trained against, identified by its content hash; the rate it runs at; the
command it follows; how its inputs are built, term by term, with scales and
history; and how its outputs become joint commands, with their gains. The
observation builder and the runner are driven by it, so these facts are
written once, next to the weights, rather than in deployment code. A policy
built into the platform that has no file, such as the zero policy, is
described the same way, with no `policy` field.

Validation happens in three stages. Loading checks everything that needs no
other file. `validate_against` checks that the descriptor fits a robot
description, and `verify_policy_file` that the policy file is the one the
descriptor names. The fields are documented in docs/policies.md.
"""

import hashlib
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from embodiverse.contracts.description import Rates, RobotDescription
from embodiverse.contracts.hashing import content_hash

__all__ = [
    "DescriptorError",
    "Gains",
    "History",
    "JointGains",
    "ObservationTerm",
    "PolicyDescriptor",
    "PolicyFile",
    "PositionAction",
    "RobotRef",
    "VelocityCommand",
    "VelocityRanges",
    "load_descriptor",
    "resolve_joints",
    "validate_against",
    "verify_policy_file",
]

_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9_]*")
_CHUNK_SIZE = 1 << 20


def _check_hash(value: str) -> str:
    if _HASH.fullmatch(value) is None:
        raise ValueError(
            f"expected 'sha256:' followed by 64 lowercase hex digits, got {value!r}"
        )
    return value


_Hash = Annotated[str, AfterValidator(_check_hash)]


class _Model(BaseModel):
    # Descriptors are read-only, misspelled keys are errors rather than
    # silently ignored, and every number must be finite.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class PolicyFile(_Model):
    """The policy file and the hash of its bytes."""

    file: str = Field(
        min_length=1,
        description="Path of the policy file; a relative path is relative to "
        "the descriptor file.",
    )
    sha256: _Hash = Field(
        description="SHA-256 of the file's bytes, as `sha256:` and 64 lowercase "
        "hex digits."
    )
    format: Literal["onnx"] = Field(default="onnx", description="File format.")


class RobotRef(_Model):
    """The robot description the policy was trained against."""

    name: str = Field(description="The description's `name`, e.g. `g1_29dof`.")
    description_hash: _Hash = Field(
        description="`content_hash` of the description, as `sha256:` and 64 "
        "lowercase hex digits."
    )


class VelocityRanges(_Model):
    """The range of each component of the velocity command, in the base frame."""

    lin_vel_x: tuple[float, float] = Field(
        description="Lower and upper forward velocity, in m/s."
    )
    lin_vel_y: tuple[float, float] = Field(
        description="Lower and upper leftward velocity, in m/s."
    )
    ang_vel_z: tuple[float, float] = Field(
        description="Lower and upper yaw rate, counterclockwise seen from above, "
        "in rad/s."
    )

    @field_validator("lin_vel_x", "lin_vel_y", "ang_vel_z")
    @classmethod
    def _check_order(cls, value: tuple[float, float]) -> tuple[float, float]:
        lower, upper = value
        if lower > upper:
            raise ValueError(f"lower bound {lower} must not exceed upper bound {upper}")
        return value


class VelocityCommand(_Model):
    """The command the policy follows: a velocity of the base."""

    type: Literal["velocity"] = Field(description="Kind of command.")
    ranges: VelocityRanges = Field(
        description="The commands the policy was trained to follow."
    )


class ObservationTerm(_Model):
    """One term of the policy's input, such as the joint velocities."""

    term: str = Field(
        description="snake_case name of the observation term, e.g. "
        "`projected_gravity`, resolved by the observation builder."
    )
    scale: float | tuple[float, ...] = Field(
        default=1.0,
        description="Factor the term's output is multiplied by; a tuple "
        "multiplies element by element.",
    )
    joints: str | tuple[str, ...] | None = Field(
        default=None,
        description="For a per-joint term: a joint group of the description, or "
        "a list of joint names, in the order the policy expects them.",
    )
    params: dict[str, float | int | str | bool | tuple[float, ...]] = Field(
        default={},
        description="Parameters of the term, e.g. a gait period in s.",
    )

    @field_validator("term")
    @classmethod
    def _check_term(cls, value: str) -> str:
        if _SNAKE_CASE.fullmatch(value) is None:
            raise ValueError(f"observation term name {value!r} is not snake_case")
        return value

    @model_validator(mode="after")
    def _check_scale_length(self) -> "ObservationTerm":
        if isinstance(self.joints, tuple) and isinstance(self.scale, tuple):
            problem = _length_problem("scale", self.scale, self.joints)
            if problem:
                raise ValueError(problem)
        return self


class History(_Model):
    """How many observation frames the policy sees, and how they are laid out."""

    length: int = Field(
        default=1, ge=1, description="Number of frames, the current one included."
    )
    order: Literal["oldest_first"] = Field(
        default="oldest_first", description="Order of the frames: oldest first."
    )
    layout: Literal["per_term", "per_frame"] = Field(
        default="per_term",
        description="`per_term`: each term's frames are contiguous, as Isaac Lab "
        "concatenates them. `per_frame`: whole frames are stacked.",
    )


class PositionAction(_Model):
    """How the policy's outputs become joint position targets.

    Output i drives the i-th joint of `joints`: its target position is the
    joint's default position plus scale times the output.
    """

    type: Literal["position_offset"] = Field(description="Kind of action.")
    joints: str | tuple[str, ...] = Field(
        description="The joints the policy drives, in output order: a joint "
        "group of the description, or a list of joint names."
    )
    scale: float | tuple[float, ...] = Field(
        description="Factor each output is multiplied by to give a position "
        "offset in rad; a tuple has one factor per joint."
    )

    @model_validator(mode="after")
    def _check_scale_length(self) -> "PositionAction":
        if isinstance(self.joints, tuple) and isinstance(self.scale, tuple):
            problem = _length_problem("scale", self.scale, self.joints)
            if problem:
                raise ValueError(problem)
        return self


class JointGains(_Model):
    """PD gains for one joint."""

    kp: float = Field(ge=0, description="Position stiffness, in N·m/rad.")
    kd: float = Field(ge=0, description="Velocity damping, in N·m·s/rad.")


class Gains(_Model):
    """Where the gains of the joints the policy drives come from."""

    source: Literal["description", "explicit"] = Field(
        default="description",
        description="`description`: the description's default gains. "
        "`explicit`: the gains in `values`.",
    )
    values: dict[str, JointGains] = Field(
        default={},
        description="Gains by joint name, for every joint the policy drives; "
        "empty unless `source` is `explicit`.",
    )

    @model_validator(mode="after")
    def _check_values(self) -> "Gains":
        if self.source == "description" and self.values:
            raise ValueError("values must be empty when source is 'description'")
        if self.source == "explicit" and not self.values:
            raise ValueError("values must give gains when source is 'explicit'")
        return self


class PolicyDescriptor(_Model):
    """Everything needed to run one trained policy on its robot."""

    schema_version: Literal[1] = Field(description="Version of this schema.")
    policy: PolicyFile | None = Field(
        default=None,
        description="The policy file and the hash of its bytes; None for a policy "
        "built into the platform that has no file, such as the zero policy.",
    )
    robot: RobotRef
    rate_hz: float = Field(gt=0, description="Rate the policy runs at, in Hz.")
    command: VelocityCommand
    observations: tuple[ObservationTerm, ...] = Field(
        min_length=1, description="The terms of one observation frame, in input order."
    )
    observation_size: int = Field(
        gt=0, description="Number of values in one observation frame."
    )
    history: History = History()
    action: PositionAction
    gains: Gains
    held_joints: str | tuple[str, ...] = Field(
        default=(),
        description="Joints the policy does not drive, held at their default "
        "positions with the description's gains: a joint group or a list of "
        "joint names.",
    )
    provenance: Literal["external"] | dict[str, object] = Field(
        default="external",
        description="`external` for a policy trained outside the platform.",
    )

    # Directory of the file the descriptor was loaded from. Kept out of the
    # fields so that the same file contents give the same dump at any path.
    _directory: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_explicit_gains(self) -> "PolicyDescriptor":
        joints = self.action.joints
        if self.gains.source == "explicit" and isinstance(joints, tuple):
            problems = _gain_problems(self.gains.values, joints)
            if problems:
                raise ValueError("; ".join(problems))
        return self

    def policy_path(self) -> Path:
        """Return the path of the policy file.

        Raises DescriptorError if the descriptor names no policy file, as
        for a policy built into the platform. A relative path resolves
        against the descriptor file's directory, so it raises ValueError if
        the descriptor was not loaded from a file.
        """
        file = _policy_file(self).file
        path = Path(file)
        if path.is_absolute():
            return path
        if self._directory is None:
            raise ValueError(
                f"cannot resolve relative policy path {file!r}: the"
                " descriptor was not loaded from a file"
            )
        return self._directory / path


class DescriptorError(ValueError):
    """A policy descriptor is invalid, or does not fit its robot or policy file."""


def load_descriptor(path: str | os.PathLike[str]) -> PolicyDescriptor:
    """Read and validate the policy descriptor at `path`.

    Raises DescriptorError, with the file path at the start of its message.
    """
    file = Path(path)
    try:
        # Reading from the file object makes YAML error positions name it.
        with file.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        reason = error.strerror or error
        raise DescriptorError(f"{file}: cannot read: {reason}") from error
    except UnicodeDecodeError as error:
        raise DescriptorError(f"{file}: not valid UTF-8: {error}") from error
    except yaml.YAMLError as error:
        raise DescriptorError(f"{file}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        found = "an empty document" if data is None else type(data).__name__
        raise DescriptorError(
            f"{file}: expected a mapping at the top level, got {found}"
        )
    try:
        descriptor = PolicyDescriptor.model_validate(data)
    except ValidationError as error:
        raise DescriptorError(f"{file}: {error}") from error
    descriptor._directory = file.absolute().parent  # pyright: ignore[reportPrivateUsage]
    return descriptor


def verify_policy_file(descriptor: PolicyDescriptor) -> None:
    """Check that the policy file's bytes have the hash the descriptor names.

    Raises DescriptorError, with the policy file's path at the start of its
    message, if the file cannot be read or its hash differs, and
    DescriptorError if the descriptor names no policy file, as for a policy
    built into the platform. Raises ValueError if the path is relative and
    the descriptor was not loaded from a file.
    """
    expected = _policy_file(descriptor).sha256
    path = descriptor.policy_path()
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK_SIZE):
                digest.update(chunk)
    except OSError as error:
        reason = error.strerror or error
        raise DescriptorError(
            f"{path}: cannot read the policy file: {reason}"
        ) from error
    actual = f"sha256:{digest.hexdigest()}"
    if actual != expected:
        raise DescriptorError(
            f"{path}: the policy file's hash is {actual}, but the descriptor"
            f" expects {expected}: the file is not the policy the descriptor"
            " describes"
        )


def resolve_joints(
    selection: str | tuple[str, ...], description: RobotDescription
) -> tuple[str, ...]:
    """Return the joint names `selection` stands for in `description`.

    A string is a joint group of the description and stands for its
    members; a tuple stands for itself. Raises ValueError naming the group,
    or every joint name, that the description does not have.
    """
    robot = description.name
    if isinstance(selection, str):
        if selection in description.joint_groups:
            return description.group(selection)
        hint = ""
        if selection in description.joint_names:
            hint = f"; to select the joint, write it as a list: [{selection}]"
        raise ValueError(f"no joint group {selection!r} in robot {robot!r}{hint}")
    names = set(description.joint_names)
    unknown = [name for name in selection if name not in names]
    if unknown:
        raise ValueError(f"no {_joints(unknown)} in robot {robot!r}")
    return selection


def validate_against(
    descriptor: PolicyDescriptor, description: RobotDescription
) -> None:
    """Check that `descriptor` fits `description`, the robot it is to run on.

    Collects every problem and raises one DescriptorError that lists them:
    the robot's name or description hash differs; the policy's rate is
    above the robot's highest control rate, or does not divide its physics
    rate into whole steps; a joint group or joint name is unknown; the
    action's joints and the held joints overlap, or together leave a joint
    uncommanded; a scale tuple's length differs from the number of joints
    it applies to; or explicit gains are missing for a driven joint or name
    another joint.
    """
    problems = _identity_problems(descriptor.robot, description)
    problems += _rate_problems(descriptor.rate_hz, description.rates)
    problems += _joint_problems(descriptor, description)
    if problems:
        lines = "".join(f"\n- {problem}" for problem in problems)
        raise DescriptorError(
            f"policy descriptor for robot {descriptor.robot.name!r} does not fit"
            f" robot description {description.name!r}:{lines}"
        )


def _policy_file(descriptor: PolicyDescriptor) -> PolicyFile:
    """Return the descriptor's policy file; raise DescriptorError if it has none."""
    if descriptor.policy is None:
        raise DescriptorError(
            f"the policy descriptor for robot {descriptor.robot.name!r} names no"
            " policy file: it describes a policy built into the platform"
        )
    return descriptor.policy


def _identity_problems(robot: RobotRef, description: RobotDescription) -> list[str]:
    problems: list[str] = []
    if robot.name != description.name:
        problems.append(
            f"robot.name is {robot.name!r}, but the description is of robot"
            f" {description.name!r}"
        )
    actual = content_hash(description)
    if robot.description_hash != actual:
        problems.append(
            "the policy was trained for a different description of the robot:"
            f" robot.description_hash is {robot.description_hash}, but this"
            f" description's hash is {actual}"
        )
    return problems


def _rate_problems(rate_hz: float, rates: Rates) -> list[str]:
    problems: list[str] = []
    if rate_hz > rates.control_hz_max:
        problems.append(
            f"rate_hz {rate_hz} Hz is above the robot's control_hz_max"
            f" {rates.control_hz_max} Hz"
        )
    steps = rates.physics_hz / rate_hz
    if round(steps) < 1 or not math.isclose(steps, round(steps), rel_tol=1e-9):
        problems.append(
            f"physics_hz {rates.physics_hz} Hz is not a whole multiple of rate_hz"
            f" {rate_hz} Hz: adapters step whole physics steps, so each policy"
            " step must last a whole number of them"
        )
    return problems


def _joint_problems(
    descriptor: PolicyDescriptor, description: RobotDescription
) -> list[str]:
    problems: list[str] = []

    def resolve(field: str, selection: str | tuple[str, ...]) -> tuple[str, ...] | None:
        try:
            return resolve_joints(selection, description)
        except ValueError as error:
            problems.append(f"{field}: {error}")
            return None

    for index, term in enumerate(descriptor.observations):
        if term.joints is None:
            continue
        field = f"observations[{index}] ({term.term})"
        joints = resolve(f"{field}.joints", term.joints)
        if joints is not None and isinstance(term.scale, tuple):
            problem = _length_problem(f"{field}.scale", term.scale, joints)
            problems += [problem] if problem else []

    action = descriptor.action
    driven = resolve("action.joints", action.joints)
    if driven is not None:
        problems += _repeat_problems("action.joints", driven)
        if isinstance(action.scale, tuple):
            problem = _length_problem("action.scale", action.scale, driven)
            problems += [problem] if problem else []
        if descriptor.gains.source == "explicit":
            known = set(description.joint_names)
            problems += _gain_problems(descriptor.gains.values, driven, known)

    held = resolve("held_joints", descriptor.held_joints)
    if held is not None:
        problems += _repeat_problems("held_joints", held)

    if driven is not None and held is not None:
        both = [name for name in driven if name in held]
        if both:
            problems.append(f"action.joints and held_joints both name {_joints(both)}")
        covered = set(driven) | set(held)
        left_out = [name for name in description.joint_names if name not in covered]
        if left_out:
            problems.append(
                f"action.joints and held_joints together leave out"
                f" {_joints(left_out)}: every joint must be driven by the action or"
                " held, so that every joint is commanded on every tick"
            )
    return problems


def _gain_problems(
    values: dict[str, JointGains],
    driven: tuple[str, ...],
    known: set[str] | None = None,
) -> list[str]:
    """Return what is wrong with explicit gain `values` for the `driven` joints.

    Every driven joint needs gains, and every key must be a driven joint.
    `known` is the robot's joint names, if known: a key outside it is
    reported as a joint the robot does not have.
    """
    problems: list[str] = []
    missing = [name for name in driven if name not in values]
    if missing:
        problems.append(f"gains.values has no gains for action {_joints(missing)}")
    extra = [name for name in values if name not in driven]
    unknown = [name for name in extra if known is not None and name not in known]
    if unknown:
        problems.append(f"gains.values names unknown {_joints(unknown)}")
    undriven = [name for name in extra if name not in unknown]
    if undriven:
        problems.append(
            f"gains.values names {_joints(undriven)}, which the action does not"
            " drive: held joints use the description's gains"
        )
    return problems


def _length_problem(
    field: str, scale: tuple[float, ...], joints: tuple[str, ...]
) -> str | None:
    """Return the problem if `scale` does not have one element per joint."""
    if len(scale) == len(joints):
        return None
    elements = _count(len(scale), "element")
    return f"{field} has {elements} for {_count(len(joints), 'joint')}"


def _repeat_problems(field: str, joints: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    repeated: list[str] = []
    for name in joints:
        if name in seen and name not in repeated:
            repeated.append(name)
        seen.add(name)
    if not repeated:
        return []
    return [f"{field} names {_joints(repeated)} more than once"]


def _joints(names: Iterable[str]) -> str:
    """Return `names` for a message: `joint 'a'` or `joints 'a', 'b'`."""
    items = [repr(name) for name in names]
    return f"joint {items[0]}" if len(items) == 1 else f"joints {', '.join(items)}"


def _count(number: int, noun: str) -> str:
    """Return `number` with `noun`, in the plural unless it is one."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"
