import math

import numpy as np
import pytest

from embodiverse.contracts.conventions import (
    IDENTITY_QUATERNION,
    as_vector,
    is_valid_joint_name,
    multiply_quaternions,
    normalize_quaternion,
    quaternion_from_axis_angle,
    quaternion_from_rpy,
    rotate_vector,
    rotate_vector_inverse,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

X_AXIS = [1.0, 0.0, 0.0]
Y_AXIS = [0.0, 1.0, 0.0]
Z_AXIS = [0.0, 0.0, 1.0]


def random_quaternions(count: int) -> np.ndarray:
    rng = np.random.default_rng(seed=0)
    return rng.normal(size=(count, 4))


@pytest.mark.parametrize(
    "name", ["left_hip_pitch", "waist_yaw", "head_pitch", "joint2", "knee"]
)
def test_valid_joint_names(name: str) -> None:
    assert is_valid_joint_name(name)


@pytest.mark.parametrize(
    "name",
    ["", "Left_hip", "left-hip", "left__hip", "_left", "left_", "2left", "J00_HIP"],
)
def test_invalid_joint_names(name: str) -> None:
    assert not is_valid_joint_name(name)


def test_as_vector_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        as_vector([1.0, 2.0], 3)


def test_as_vector_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        as_vector([1.0, math.nan, 0.0], 3)


def test_as_vector_returns_a_copy() -> None:
    source = np.zeros(3)
    result = as_vector(source, 3)
    result[0] = 1.0
    assert source[0] == 0.0


def test_quaternion_order_conversions_are_inverses() -> None:
    q = [0.1, 0.2, 0.3, 0.9]
    assert np.allclose(wxyz_to_xyzw(xyzw_to_wxyz(q)), q)


def test_identity_quaternion_in_mujoco_order() -> None:
    assert np.allclose(wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0]), IDENTITY_QUATERNION)


def test_normalize_gives_unit_length() -> None:
    for q in random_quaternions(100):
        assert np.isclose(np.linalg.norm(normalize_quaternion(q)), 1.0)


def test_normalize_rejects_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="zero length"):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


def test_quarter_turn_about_z_maps_x_to_y() -> None:
    q = quaternion_from_axis_angle(Z_AXIS, math.pi / 2)
    assert np.allclose(rotate_vector(q, X_AXIS), Y_AXIS)


def test_half_turn_about_x_flips_z() -> None:
    # An upside-down sensor: roll of pi about x.
    q = quaternion_from_axis_angle(X_AXIS, math.pi)
    assert np.allclose(rotate_vector(q, Z_AXIS), [0.0, 0.0, -1.0])


def test_identity_leaves_vectors_unchanged() -> None:
    v = [0.3, -1.2, 2.5]
    assert np.allclose(rotate_vector(IDENTITY_QUATERNION, v), v)


def test_inverse_rotation_undoes_rotation() -> None:
    v = np.array([0.3, -1.2, 2.5])
    for q in random_quaternions(100):
        assert np.allclose(rotate_vector_inverse(q, rotate_vector(q, v)), v)


def test_rotation_preserves_length() -> None:
    v = np.array([0.3, -1.2, 2.5])
    for q in random_quaternions(100):
        assert np.isclose(np.linalg.norm(rotate_vector(q, v)), np.linalg.norm(v))


def test_rotation_ignores_quaternion_scale() -> None:
    q = quaternion_from_axis_angle(Y_AXIS, 0.7)
    assert np.allclose(rotate_vector(3.0 * q, X_AXIS), rotate_vector(q, X_AXIS))


def rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Return the matrix rotating by `angle` about the x, y or z axis."""
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def same_rotation(a: np.ndarray, b: np.ndarray) -> bool:
    """Return whether two unit quaternions are the same rotation (q or -q)."""
    return np.allclose(a, b) or np.allclose(a, -b)


def test_product_applies_the_second_rotation_first() -> None:
    v = np.array([0.3, -1.2, 2.5])
    for a, b in random_quaternions(200).reshape(100, 2, 4):
        product = multiply_quaternions(a, b)
        assert np.allclose(
            rotate_vector(product, v), rotate_vector(a, rotate_vector(b, v))
        )


def test_product_of_turns_about_one_axis_adds_the_angles() -> None:
    product = multiply_quaternions(
        quaternion_from_axis_angle(Z_AXIS, 0.4), quaternion_from_axis_angle(Z_AXIS, 0.3)
    )
    assert np.allclose(product, quaternion_from_axis_angle(Z_AXIS, 0.7))


def test_product_depends_on_the_order() -> None:
    about_x = quaternion_from_axis_angle(X_AXIS, math.pi / 2)
    about_z = quaternion_from_axis_angle(Z_AXIS, math.pi / 2)
    # Turning x onto y about z, then y onto z about x, ends on z.
    assert np.allclose(
        rotate_vector(multiply_quaternions(about_x, about_z), X_AXIS), Z_AXIS
    )
    assert not same_rotation(
        multiply_quaternions(about_x, about_z), multiply_quaternions(about_z, about_x)
    )


def test_product_with_the_identity_is_the_same_rotation() -> None:
    q = normalize_quaternion([0.1, 0.2, 0.3, 0.9])
    assert np.allclose(multiply_quaternions(IDENTITY_QUATERNION, q), q)
    assert np.allclose(multiply_quaternions(q, IDENTITY_QUATERNION), q)


def test_product_has_unit_length() -> None:
    product = multiply_quaternions([0.0, 0.0, 0.0, 3.0], [0.2, 0.0, 0.0, 2.0])
    assert np.isclose(np.linalg.norm(product), 1.0)


def test_product_rejects_a_zero_quaternion() -> None:
    with pytest.raises(ValueError, match="zero length"):
        multiply_quaternions([0.0, 0.0, 0.0, 0.0], IDENTITY_QUATERNION)


@pytest.mark.parametrize(
    ("rpy", "v", "expected"),
    [
        pytest.param((math.pi / 2, 0.0, 0.0), Y_AXIS, Z_AXIS, id="roll turns y onto z"),
        pytest.param(
            (0.0, math.pi / 2, 0.0), Z_AXIS, X_AXIS, id="pitch turns z onto x"
        ),
        pytest.param((0.0, 0.0, math.pi / 2), X_AXIS, Y_AXIS, id="yaw turns x onto y"),
    ],
)
def test_rpy_turns_about_the_named_axis(
    rpy: tuple[float, float, float], v: list[float], expected: list[float]
) -> None:
    assert np.allclose(rotate_vector(quaternion_from_rpy(*rpy), v), expected)


def test_rpy_of_zero_is_the_identity() -> None:
    assert np.allclose(quaternion_from_rpy(0.0, 0.0, 0.0), IDENTITY_QUATERNION)


def test_rpy_rotates_about_fixed_axes_roll_first() -> None:
    # A URDF <origin rpy="r p y"> is the matrix Rz(y) @ Ry(p) @ Rx(r).
    rng = np.random.default_rng(seed=0)
    v = np.array([0.3, -1.2, 2.5])
    for roll, pitch, yaw in rng.uniform(-math.pi, math.pi, size=(100, 3)):
        matrix = (
            rotation_matrix("z", yaw)
            @ rotation_matrix("y", pitch)
            @ rotation_matrix("x", roll)
        )
        assert np.allclose(
            rotate_vector(quaternion_from_rpy(roll, pitch, yaw), v), matrix @ v
        )


def test_rpy_is_yaw_times_pitch_times_roll() -> None:
    roll, pitch, yaw = 0.3, -0.5, 1.2
    expected = multiply_quaternions(
        quaternion_from_axis_angle(Z_AXIS, yaw),
        multiply_quaternions(
            quaternion_from_axis_angle(Y_AXIS, pitch),
            quaternion_from_axis_angle(X_AXIS, roll),
        ),
    )
    assert same_rotation(quaternion_from_rpy(roll, pitch, yaw), expected)
