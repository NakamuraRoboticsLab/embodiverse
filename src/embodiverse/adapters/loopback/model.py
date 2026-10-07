"""The loopback's robot: joints without inertia, and a base that goes where it is put.

`MasslessJoints` is the simplest robot that behaves. Each joint has viscous
friction and no inertia, so under a PD law it moves toward its target
exponentially and stays still when the gains are zero. The base has no
dynamics at all: it rests where `set_base_pose` puts it, supported, so the
IMU reads the specific force of a body at rest, +g up in the world, turned
into the IMU frame through the base orientation and the description's
mount. Nothing falls, nothing is unstable, and any description works.

It is not a model of any robot. It exists so that the whole pipeline, from
a description to a policy's commands, runs without a model file, a
simulator or hardware, and so that an adapter's conversions can be tested
against a robot whose true state is known.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts.conventions import (
    IDENTITY_QUATERNION,
    STANDARD_GRAVITY,
    as_vector,
    multiply_quaternions,
    normalize_quaternion,
    quaternion_from_rpy,
    rotate_vector_inverse,
)
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.types import ImuState, JointCommand, Mode, RobotState

__all__ = ["MasslessJoints"]

_GRAVITY = np.array([0.0, 0.0, STANDARD_GRAVITY])


class MasslessJoints:
    """Joints with viscous friction and no inertia, on a base that is held.

    `step` applies a law for one physics period: the friction torque
    balances the PD torque, `FRICTION * dq = kp * (q* - q) + kd * (dq* - dq)
    + tau_ff`, solved implicitly for the new position, so any gains are
    stable. `effort` reports the friction torque, which equals the torque
    the law applied. Time is counted in steps.
    """

    FRICTION = 0.1
    """Viscous friction of every joint, in N·m·s/rad.

    Small, so that a joint under a PD law approaches its target with the
    time constant `(FRICTION + kd) / kp`, close to the law's own `kd / kp`.
    """

    def __init__(self, description: RobotDescription) -> None:
        self._description = description
        self._dt = 1.0 / description.rates.physics_hz
        self._mount = quaternion_from_rpy(*description.imu.mount_rpy)
        self._default = np.array(
            [joint.default_position for joint in description.joints]
        )
        self.steps = 0
        """Physics steps taken since the start."""
        self.position: NDArray[np.float64] = self._default.copy()
        self.velocity: NDArray[np.float64] = np.zeros(description.num_joints)
        self.effort: NDArray[np.float64] = np.zeros(description.num_joints)
        self.base_orientation: NDArray[np.float64] = np.array(IDENTITY_QUATERNION)
        """Orientation of the base in the world, as an (x, y, z, w) quaternion."""

    @property
    def stamp_ns(self) -> int:
        """Simulated time, in ns: the steps taken so far."""
        return round(self.steps * 1e9 * self._dt)

    def step(self, law: JointCommand) -> None:
        """Apply `law` for one physics period."""
        dt = self._dt
        damping = self.FRICTION + law.kd
        position = (
            damping * self.position / dt
            + law.kp * law.position
            + law.kd * law.velocity
            + law.effort
        ) / (damping / dt + law.kp)
        self.velocity = (position - self.position) / dt
        self.effort = self.FRICTION * self.velocity
        self.position = position
        self.steps += 1

    def imu(self) -> ImuState:
        """The IMU reading of a body at rest: the orientation, no rotation, +g."""
        orientation = multiply_quaternions(self.base_orientation, self._mount)
        return ImuState(
            orientation=orientation,
            angular_velocity=np.zeros(3),
            linear_acceleration=rotate_vector_inverse(orientation, _GRAVITY),
        )

    def state(self, stamp_ns: int, *, temperature: bool = False) -> RobotState:
        """The state now, stamped `stamp_ns`, in PASSIVE; the adapter sets the mode.

        With `temperature`, every motor reports 35 °C.
        """
        return RobotState(
            stamp_ns=stamp_ns,
            position=self.position,
            velocity=self.velocity,
            effort=self.effort,
            imu=self.imu(),
            mode=Mode.PASSIVE,
            temperature=(
                np.full(self._description.num_joints, 35.0) if temperature else None
            ),
        )

    def set_base_pose(self, position: ArrayLike, orientation_xyzw: ArrayLike) -> None:
        """Put the base at `orientation_xyzw`; the position is checked and ignored."""
        as_vector(position, 3, "position")
        self.base_orientation = normalize_quaternion(orientation_xyzw)

    def set_joint_positions(self, positions: ArrayLike) -> None:
        """Set every joint's position, in description order, with zero velocity."""
        self.position = as_vector(positions, self._description.num_joints, "positions")
        self.velocity = np.zeros(self._description.num_joints)
        self.effort = np.zeros(self._description.num_joints)
