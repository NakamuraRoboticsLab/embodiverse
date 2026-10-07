"""Conventions shared by every Embodiverse module; see docs/architecture.md.

In short: SI units, ROS REP-103 axes, REP-105 frame names, quaternions in
(x, y, z, w) order, snake_case joint names, monotonic nanosecond timestamps.
"""

import re

import numpy as np
from numpy.typing import ArrayLike, NDArray

MAP_FRAME = "map"
"""World-fixed frame used for global localization (REP-105)."""

ODOM_FRAME = "odom"
"""World-fixed frame in which odometry is continuous but drifts (REP-105)."""

BASE_FRAME = "base_link"
"""Frame rigidly attached to the robot's base (REP-105)."""

STANDARD_GRAVITY = 9.80665
"""Standard gravity, in m/s²."""

IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
"""The rotation that does nothing, in (x, y, z, w) order."""

_JOINT_NAME = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_NORM_EPSILON = 1e-9


def is_valid_joint_name(name: str) -> bool:
    """Return whether `name` is lowercase snake_case, e.g. `left_hip_pitch`."""
    return _JOINT_NAME.fullmatch(name) is not None


def as_vector(values: ArrayLike, size: int, name: str = "value") -> NDArray[np.float64]:
    """Return `values` as a new float64 array of shape (size,).

    Raises ValueError if the shape is wrong or any element is not finite.
    """
    array = np.array(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite, got {array}")
    return array


def normalize_quaternion(q: ArrayLike) -> NDArray[np.float64]:
    """Return `q`, an (x, y, z, w) quaternion, scaled to unit length."""
    quaternion = as_vector(q, 4, "quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < _NORM_EPSILON:
        raise ValueError("quaternion has zero length")
    return quaternion / norm


def wxyz_to_xyzw(q: ArrayLike) -> NDArray[np.float64]:
    """Reorder a (w, x, y, z) quaternion, as used by MuJoCo, to (x, y, z, w)."""
    w, x, y, z = as_vector(q, 4, "quaternion")
    return np.array([x, y, z, w])


def xyzw_to_wxyz(q: ArrayLike) -> NDArray[np.float64]:
    """Reorder an (x, y, z, w) quaternion to (w, x, y, z), as used by MuJoCo."""
    x, y, z, w = as_vector(q, 4, "quaternion")
    return np.array([w, x, y, z])


def quaternion_from_axis_angle(axis: ArrayLike, angle: float) -> NDArray[np.float64]:
    """Return the (x, y, z, w) quaternion rotating by `angle` radians about `axis`."""
    direction = as_vector(axis, 3, "axis")
    norm = float(np.linalg.norm(direction))
    if norm < _NORM_EPSILON:
        raise ValueError("axis has zero length")
    half = 0.5 * angle
    x, y, z = direction / norm * np.sin(half)
    return np.array([x, y, z, np.cos(half)])


def multiply_quaternions(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return the Hamilton product `a ⊗ b` of two (x, y, z, w) quaternions.

    The result is the rotation `b` followed by the rotation `a`:
    `rotate_vector(multiply_quaternions(a, b), v)` equals
    `rotate_vector(a, rotate_vector(b, v))`. If `a` is the orientation of
    frame B in frame A and `b` that of frame C in B, the product is the
    orientation of C in A. Both inputs are normalized, so the result has
    unit length.
    """
    ax, ay, az, aw = normalize_quaternion(a)
    bx, by, bz, bw = normalize_quaternion(b)
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    """Return the (x, y, z, w) quaternion of fixed-axis roll, pitch and yaw, in rad.

    The rotations are about the parent frame's fixed x, y and z axes, applied
    in that order, as in a URDF `<origin rpy>` and a description's
    `mount_rpy`. The result equals `yaw ⊗ pitch ⊗ roll`, each the rotation
    about its axis.
    """
    about_x = quaternion_from_axis_angle((1.0, 0.0, 0.0), roll)
    about_y = quaternion_from_axis_angle((0.0, 1.0, 0.0), pitch)
    about_z = quaternion_from_axis_angle((0.0, 0.0, 1.0), yaw)
    return multiply_quaternions(about_z, multiply_quaternions(about_y, about_x))


def rotate_vector(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Rotate `v` by the (x, y, z, w) quaternion `q`.

    If `q` is the orientation of frame B in frame A, this maps a vector
    expressed in B to the same vector expressed in A.
    """
    x, y, z, w = normalize_quaternion(q)
    vector = as_vector(v, 3, "vector")
    axis = np.array([x, y, z])
    t = 2.0 * np.cross(axis, vector)
    return vector + w * t + np.cross(axis, t)


def rotate_vector_inverse(q: ArrayLike, v: ArrayLike) -> NDArray[np.float64]:
    """Apply the inverse rotation of `rotate_vector`: from frame A back to frame B."""
    x, y, z, w = normalize_quaternion(q)
    return rotate_vector([-x, -y, -z, w], v)
