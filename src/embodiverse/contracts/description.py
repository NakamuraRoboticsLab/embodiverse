"""Robot descriptions: one validated YAML file per robot model.

A description states everything the platform needs to know about a robot
model: its joints in canonical order, with limits, default pose and gains;
its IMU and other sensors with their mounts; its rates; and paths to its
model files. Every adapter, policy descriptor and tool reads it. The fields
are documented in docs/adding-a-robot.md.
"""

import os
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

from embodiverse.contracts.conventions import is_valid_joint_name

__all__ = [
    "DescriptionError",
    "Gains",
    "Imu",
    "Joint",
    "JointLimits",
    "Models",
    "Rates",
    "RobotDescription",
    "Sensor",
    "Vendor",
    "load_description",
]

_SnakeCase = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class _Model(BaseModel):
    # Descriptions are read-only, misspelled keys are errors rather than
    # silently ignored, and every number must be finite.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Vendor(_Model):
    """The manufacturer and their name for the robot model."""

    name: str = Field(description="Manufacturer, e.g. `Unitree`.")
    model: str = Field(description="The manufacturer's model name, e.g. `G1`.")


class JointLimits(_Model):
    """A joint's position, velocity and effort limits."""

    position: tuple[float, float] = Field(
        description="Lower and upper position limits, in rad."
    )
    velocity: float = Field(gt=0, description="Maximum speed, in rad/s.")
    effort: float = Field(gt=0, description="Maximum torque, in N·m.")

    @field_validator("position")
    @classmethod
    def _check_order(cls, value: tuple[float, float]) -> tuple[float, float]:
        lower, upper = value
        if not lower < upper:
            raise ValueError(f"lower limit {lower} rad must be below upper {upper} rad")
        return value


class Gains(_Model):
    """Default PD gains for a joint.

    `side="motor"` means the gains apply in actuator space rather than joint
    space, as on parallel-linkage ankles where two motors drive two joints.
    """

    kp: float = Field(ge=0, description="Position stiffness, in N·m/rad.")
    kd: float = Field(ge=0, description="Velocity damping, in N·m·s/rad.")
    side: Literal["joint", "motor"] = Field(
        default="joint",
        description="Whether the gains apply at the joint or at the motor.",
    )


class Joint(_Model):
    """One actuated joint."""

    name: str = Field(description="Canonical snake_case name, e.g. `left_knee`.")
    vendor_name: str | None = Field(
        default=None, description="The vendor's name for the joint."
    )
    motor_index: int = Field(
        ge=0, description="Index of the joint's motor in the vendor's motor arrays."
    )
    default_position: float = Field(description="Position in the default pose, in rad.")
    limits: JointLimits
    gains: Gains

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not is_valid_joint_name(value):
            raise ValueError(f"joint name {value!r} is not lowercase snake_case")
        return value

    @model_validator(mode="after")
    def _check_default_position(self) -> "Joint":
        lower, upper = self.limits.position
        if not lower <= self.default_position <= upper:
            raise ValueError(
                f"joint {self.name!r}: default position {self.default_position} rad"
                f" is outside its position limits [{lower}, {upper}] rad"
            )
        return self


class _Mounted(_Model):
    """A device's frame and where it is mounted."""

    frame: str = Field(description="Name of the frame the device reports in.")
    parent: str = Field(description="The link the device is mounted on.")
    mount_xyz: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Position of the frame in the parent link, in m.",
    )
    mount_rpy: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Fixed-axis roll, pitch and yaw of the frame in the parent "
        "link, in rad.",
    )


class Imu(_Mounted):
    """The IMU used for state estimation, and where it is mounted."""


class Sensor(_Mounted):
    """A sensor other than the main IMU, and where it is mounted."""

    type: _SnakeCase = Field(
        description="Kind of sensor, e.g. `depth_camera`, `lidar`."
    )


class Rates(_Model):
    """The robot's loop rates."""

    physics_hz: float = Field(
        gt=0,
        description="Rate of the low-level loop, in Hz: the physics step in "
        "simulation, the motor-control loop on hardware.",
    )
    control_hz_max: float = Field(
        gt=0,
        description="Highest rate at which the robot accepts joint commands, in Hz.",
    )

    @model_validator(mode="after")
    def _check_control_rate(self) -> "Rates":
        if self.control_hz_max > self.physics_hz:
            raise ValueError(
                f"control_hz_max ({self.control_hz_max} Hz) must not exceed"
                f" physics_hz ({self.physics_hz} Hz)"
            )
        return self


class Models(_Model):
    """Paths to the robot's model files, as written in the description.

    Relative paths are relative to the description file; resolve them with
    `RobotDescription.model_path`.
    """

    mjcf: str | None = Field(default=None, description="MuJoCo MJCF model.")
    urdf: str | None = Field(default=None, description="URDF model.")


class RobotDescription(_Model):
    """Everything the platform needs to know about one robot model."""

    schema_version: Literal[1] = Field(description="Version of this schema.")
    name: _SnakeCase = Field(description="Robot model, e.g. `g1`.")
    status: Literal["draft", "active"] = Field(
        description="`draft` until the description is checked against the robot."
    )
    vendor: Vendor | None = None
    base_link: str = Field(
        description="Root link of the kinematic tree, e.g. `pelvis`."
    )
    joints: tuple[Joint, ...] = Field(
        min_length=1, description="Actuated joints, in canonical order."
    )
    joint_groups: dict[_SnakeCase, tuple[str, ...]] = Field(
        default={}, description="Named subsets of the joints, e.g. `legs`."
    )
    imu: Imu
    sensors: dict[_SnakeCase, Sensor] = Field(
        default={}, description="Other sensors, by name."
    )
    rates: Rates
    models: Models = Models()

    # Directory of the file the description was loaded from. Kept out of the
    # fields so that the same file contents give the same dump at any path.
    _directory: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_joints(self) -> "RobotDescription":
        names: set[str] = set()
        motors: dict[int, str] = {}
        for joint in self.joints:
            if joint.name in names:
                raise ValueError(f"joint {joint.name!r} is listed more than once")
            names.add(joint.name)
            if joint.motor_index in motors:
                raise ValueError(
                    f"joints {motors[joint.motor_index]!r} and {joint.name!r}"
                    f" both use motor index {joint.motor_index}"
                )
            motors[joint.motor_index] = joint.name
        return self

    @model_validator(mode="after")
    def _check_groups(self) -> "RobotDescription":
        names = set(self.joint_names)
        for group, members in self.joint_groups.items():
            for member in members:
                if member not in names:
                    raise ValueError(
                        f"joint group {group!r} names unknown joint {member!r}"
                    )
        return self

    @model_validator(mode="after")
    def _check_frames(self) -> "RobotDescription":
        owners = {self.base_link: "the base link"}
        devices = [("the IMU", self.imu.frame)]
        devices += [(f"sensor {key!r}", s.frame) for key, s in self.sensors.items()]
        for owner, frame in devices:
            if frame in owners:
                raise ValueError(
                    f"{owners[frame]} and {owner} both use frame {frame!r}"
                )
            owners[frame] = owner
        return self

    @property
    def joint_names(self) -> tuple[str, ...]:
        """Joint names in description order."""
        return tuple(joint.name for joint in self.joints)

    @property
    def num_joints(self) -> int:
        return len(self.joints)

    def index_of(self, name: str) -> int:
        """Return the position of joint `name` in description order."""
        for index, joint in enumerate(self.joints):
            if joint.name == name:
                return index
        raise KeyError(f"no joint {name!r} in robot {self.name!r}")

    def joint(self, name: str) -> Joint:
        """Return the joint called `name`."""
        return self.joints[self.index_of(name)]

    def group(self, name: str) -> tuple[str, ...]:
        """Return the joint names in group `name`."""
        try:
            return self.joint_groups[name]
        except KeyError:
            raise KeyError(f"no joint group {name!r} in robot {self.name!r}") from None

    def model_path(self, kind: Literal["mjcf", "urdf"]) -> Path | None:
        """Return the path of the `kind` model file, or None if there is none.

        Relative paths resolve against the description file's directory, so
        they raise ValueError if the description was not loaded from a file.
        """
        value = self.models.mjcf if kind == "mjcf" else self.models.urdf
        if value is None:
            return None
        path = Path(value)
        if path.is_absolute():
            return path
        if self._directory is None:
            raise ValueError(
                f"cannot resolve relative {kind} path {value!r}: robot {self.name!r}"
                " was not loaded from a file"
            )
        return self._directory / path


class DescriptionError(ValueError):
    """A robot description file could not be read or is invalid."""


def load_description(path: str | os.PathLike[str]) -> RobotDescription:
    """Read and validate the robot description at `path`.

    Raises DescriptionError, with the file path at the start of its message.
    """
    file = Path(path)
    try:
        # Reading from the file object makes YAML error positions name it.
        with file.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        reason = error.strerror or error
        raise DescriptionError(f"{file}: cannot read: {reason}") from error
    except UnicodeDecodeError as error:
        raise DescriptionError(f"{file}: not valid UTF-8: {error}") from error
    except yaml.YAMLError as error:
        raise DescriptionError(f"{file}: invalid YAML: {error}") from error
    if not isinstance(data, dict):
        found = "an empty document" if data is None else type(data).__name__
        raise DescriptionError(
            f"{file}: expected a mapping at the top level, got {found}"
        )
    try:
        description = RobotDescription.model_validate(data)
    except ValidationError as error:
        raise DescriptionError(f"{file}: {error}") from error
    description._directory = file.absolute().parent  # pyright: ignore[reportPrivateUsage]
    return description
