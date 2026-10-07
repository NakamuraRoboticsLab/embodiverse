"""Tests of the built-in observation terms and the observation builder.

Every term is checked against values computed by hand, in the comments
next to them, because a term with the wrong frame, sign, order or scale
still produces plausible numbers.
"""

import math
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint
from importlib.resources import files
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml
from numpy.typing import ArrayLike, NDArray

from embodiverse.contracts import registry
from embodiverse.contracts.conventions import (
    multiply_quaternions,
    quaternion_from_axis_angle,
    quaternion_from_rpy,
    rotate_vector,
    rotate_vector_inverse,
)
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    ObservationTerm,
    PolicyDescriptor,
    load_descriptor,
    resolve_joints,
)
from embodiverse.contracts.observation import Term, TermInputs
from embodiverse.contracts.registry import (
    OBSERVATIONS,
    RegistryError,
    entry_points_in,
)
from embodiverse.contracts.types import ImuState, Mode, RobotState
from embodiverse.runtime import observations
from embodiverse.runtime.observations import ObservationBuilder, term_factory

POLICIES = Path(__file__).parents[1] / "fixtures/policies"
TESTBOT_ZERO = POLICIES / "testbot_zero/descriptor.yaml"
G1_VELOCITY = POLICIES / "g1_velocity/descriptor.yaml"

BUILT_IN = (
    "base_angular_velocity",
    "gait_phase",
    "joint_position_relative",
    "joint_velocity",
    "last_action",
    "projected_gravity",
    "velocity_command",
)

# The default positions of testbot's joints, in description order:
# left_hip_pitch, left_knee, left_ankle_pitch, then the same on the right.
TESTBOT_DEFAULTS = np.array([-0.3, 0.6, -0.3, -0.3, 0.6, -0.3])

LEVEL = (0.0, 0.0, 0.0, 1.0)
X_AXIS = (1.0, 0.0, 0.0)
Y_AXIS = (0.0, 1.0, 0.0)
Z_AXIS = (0.0, 0.0, 1.0)
SIN_30 = 0.5
COS_30 = math.sqrt(3.0) / 2.0


def robot(**imu: Any) -> RobotDescription:
    """Return testbot's description, with `imu` replacing fields of its IMU."""
    file = files("embodiverse.robots.testbot").joinpath("description.yaml")
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    data["imu"].update(imu)
    return RobotDescription.model_validate(data)


def descriptor_data(path: Path = TESTBOT_ZERO) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def descriptor(
    observations: list[dict[str, Any]], observation_size: int, **changes: Any
) -> PolicyDescriptor:
    """Return the testbot fixture with other observations and `changes`."""
    data = descriptor_data()
    data["observations"] = observations
    data["observation_size"] = observation_size
    data.update(changes)
    return PolicyDescriptor.model_validate(data)


def state(
    position: ArrayLike = TESTBOT_DEFAULTS,
    velocity: ArrayLike = (0.0,) * 6,
    orientation: ArrayLike = LEVEL,
    angular_velocity: ArrayLike = (0.0, 0.0, 0.0),
    effort: ArrayLike | None = None,
) -> RobotState:
    imu = ImuState(
        orientation=np.array(orientation, dtype=np.float64),
        angular_velocity=np.array(angular_velocity, dtype=np.float64),
        linear_acceleration=np.array([0.0, 0.0, 9.81]),
    )
    position = np.array(position, dtype=np.float64)
    return RobotState(
        stamp_ns=0,
        position=position,
        velocity=np.array(velocity, dtype=np.float64),
        effort=np.zeros(position.size) if effort is None else np.array(effort),
        imu=imu,
        mode=Mode.POLICY,
    )


def inputs(
    robot_state: RobotState | None = None,
    command: ArrayLike = (0.0, 0.0, 0.0),
    last_action: ArrayLike = (0.0,) * 6,
    elapsed_s: float = 0.0,
) -> TermInputs:
    return TermInputs(
        state=state() if robot_state is None else robot_state,
        command=np.array(command, dtype=np.float64),
        last_action=np.array(last_action, dtype=np.float64),
        elapsed_s=elapsed_s,
    )


def term(
    name: str,
    description: RobotDescription | None = None,
    policy: PolicyDescriptor | None = None,
    **fields: Any,
) -> Term:
    """Build the term `name` with its registered factory, for testbot by default."""
    entry = ObservationTerm.model_validate({"term": name, **fields})
    return term_factory(name)(
        entry,
        robot() if description is None else description,
        load_descriptor(TESTBOT_ZERO) if policy is None else policy,
    )


def values(built: Term, tick: TermInputs) -> list[float]:
    result = built(tick)
    assert result.shape == (built.size,)
    return result.tolist()


# The built-in terms.


def test_the_built_in_terms_are_registered() -> None:
    # nox installs the package from a built wheel, so this checks its metadata.
    registered = entry_points_in(OBSERVATIONS)
    for name in BUILT_IN:
        assert registered[name].value == f"embodiverse.runtime.observations:{name}"
        distribution = registered[name].dist
        assert distribution is not None
        assert distribution.name == "embodiverse"
        assert term_factory(name) is getattr(observations, name)


def test_base_angular_velocity_with_the_imu_aligned_to_the_base() -> None:
    built = term("base_angular_velocity")
    assert built.size == 3
    tick = inputs(state(angular_velocity=(0.1, -0.2, 0.3)))
    assert values(built, tick) == pytest.approx([0.1, -0.2, 0.3])


def test_base_angular_velocity_of_a_tilted_base_is_still_in_the_base_frame() -> None:
    built = term("base_angular_velocity")
    tilted = quaternion_from_rpy(0.3, -0.2, 1.0)
    tick = inputs(state(orientation=tilted, angular_velocity=(0.1, -0.2, 0.3)))
    assert values(built, tick) == pytest.approx([0.1, -0.2, 0.3])


def test_base_angular_velocity_with_the_imu_yawed_on_the_base() -> None:
    # The IMU is turned 90° left on the base: its x axis is the base's y,
    # and its y axis the base's -x. A reading of (1, 2, 3) in the IMU frame
    # is 1 about the base's y and 2 about the base's -x: (-2, 1, 3).
    built = term("base_angular_velocity", robot(mount_rpy=[0.0, 0.0, math.pi / 2]))
    tick = inputs(state(angular_velocity=(1.0, 2.0, 3.0)))
    assert values(built, tick) == pytest.approx([-2.0, 1.0, 3.0], abs=1e-12)


def test_base_angular_velocity_with_the_imu_mounted_upside_down() -> None:
    # Rolled 180° on the base: the IMU's x is the base's x, its y and z the
    # base's -y and -z.
    built = term("base_angular_velocity", robot(mount_rpy=[math.pi, 0.0, 0.0]))
    tick = inputs(state(angular_velocity=(1.0, 2.0, 3.0)))
    assert values(built, tick) == pytest.approx([1.0, -2.0, -3.0], abs=1e-12)


@pytest.mark.parametrize("heading", [0.0, 1.0, -2.5])
def test_projected_gravity_of_a_level_base_points_down(heading: float) -> None:
    built = term("projected_gravity")
    assert built.size == 3
    level = quaternion_from_axis_angle(Z_AXIS, heading)
    assert values(built, inputs(state(orientation=level))) == pytest.approx(
        [0.0, 0.0, -1.0], abs=1e-12
    )


def test_projected_gravity_of_a_base_rolled_left_side_up() -> None:
    # Rolled +30° about x: the base's y axis (left) tilts up, so gravity
    # has -sin 30° along y (towards the right side, which is now lower) and
    # -cos 30° along z.
    built = term("projected_gravity")
    rolled = quaternion_from_axis_angle(X_AXIS, math.radians(30.0))
    assert values(built, inputs(state(orientation=rolled))) == pytest.approx(
        [0.0, -SIN_30, -COS_30]
    )


def test_projected_gravity_of_a_base_pitched_nose_down() -> None:
    # Pitched +30° about y: the base's x axis (forward) tilts down, so
    # gravity has +sin 30° along x.
    built = term("projected_gravity")
    pitched = quaternion_from_axis_angle(Y_AXIS, math.radians(30.0))
    assert values(built, inputs(state(orientation=pitched))) == pytest.approx(
        [SIN_30, 0.0, -COS_30]
    )


def test_projected_gravity_with_the_imu_yawed_on_the_base() -> None:
    # The base is pitched nose down by 30° and the IMU is turned 90° left
    # on it, so the IMU's orientation is the base's followed by the mount
    # (docs/architecture.md). In the IMU's own frame gravity is
    # (0, -sin 30°, -cos 30°); in the base frame it is (sin 30°, 0, -cos 30°),
    # as for the base alone.
    mount = quaternion_from_rpy(0.0, 0.0, math.pi / 2)
    base = quaternion_from_axis_angle(Y_AXIS, math.radians(30.0))
    imu = multiply_quaternions(base, mount)
    in_imu = rotate_vector_inverse(imu, (0.0, 0.0, -1.0))
    assert in_imu.tolist() == pytest.approx([0.0, -SIN_30, -COS_30], abs=1e-12)
    built = term("projected_gravity", robot(mount_rpy=[0.0, 0.0, math.pi / 2]))
    assert values(built, inputs(state(orientation=imu))) == pytest.approx(
        [SIN_30, 0.0, -COS_30], abs=1e-12
    )


def test_projected_gravity_follows_its_definition_for_any_orientation() -> None:
    # The term computes rotate_vector_inverse in closed form; this checks it
    # against the definition built from the conventions' helpers.
    rng = np.random.default_rng(seed=0)
    for _ in range(50):
        mount_rpy = rng.uniform(-math.pi, math.pi, size=3)
        orientation = rng.normal(size=4)
        orientation /= np.linalg.norm(orientation)
        mount = quaternion_from_rpy(*mount_rpy)
        expected = rotate_vector(
            mount, rotate_vector_inverse(orientation, (0.0, 0.0, -1.0))
        )
        built = term("projected_gravity", robot(mount_rpy=mount_rpy.tolist()))
        tick = inputs(state(orientation=orientation))
        assert values(built, tick) == pytest.approx(expected.tolist(), abs=1e-12)


def test_imu_terms_pass_non_finite_readings_through() -> None:
    nan = math.nan
    tick = inputs(state(orientation=(nan, 0.0, 0.0, 1.0), angular_velocity=(nan,) * 3))
    assert np.isnan(term("projected_gravity")(tick)).all()
    assert np.isnan(term("base_angular_velocity")(tick)).all()


@pytest.mark.parametrize("name", ["base_angular_velocity", "projected_gravity"])
def test_imu_terms_need_the_imu_on_the_base_link(name: str) -> None:
    with pytest.raises(ValueError) as error:
        term(name, robot(parent="torso"))
    assert str(error.value) == (
        f"observation term {name!r} reads the IMU, which is mounted on 'torso',"
        " not on the base link 'pelvis': its readings can be expressed in the base"
        " frame through its mount only if it is fixed to the base"
    )


def test_velocity_command() -> None:
    built = term("velocity_command")
    assert built.size == 3
    tick = inputs(command=(0.5, -0.25, 1.0))
    assert values(built, tick) == [0.5, -0.25, 1.0]


def test_joint_position_relative_on_a_subset_in_another_order() -> None:
    # Positions in description order are 0.1 to 0.6; the term reads
    # right_knee (0.5, default 0.6), left_hip_pitch (0.1, default -0.3) and
    # left_ankle_pitch (0.3, default -0.3), in that order.
    built = term(
        "joint_position_relative",
        joints=["right_knee", "left_hip_pitch", "left_ankle_pitch"],
    )
    assert built.size == 3
    tick = inputs(state(position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    assert values(built, tick) == pytest.approx([-0.1, 0.4, 0.6])


def test_joint_position_relative_on_a_group_is_zero_at_the_default_pose() -> None:
    built = term("joint_position_relative", joints="right_leg")
    assert values(built, inputs(state(position=TESTBOT_DEFAULTS))) == [0.0] * 3


def test_joint_velocity_on_a_subset_in_another_order() -> None:
    built = term(
        "joint_velocity", joints=["right_knee", "left_hip_pitch", "left_ankle_pitch"]
    )
    assert built.size == 3
    tick = inputs(state(velocity=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert values(built, tick) == [5.0, 1.0, 3.0]


def test_joint_velocity_on_a_group_follows_the_group_s_order() -> None:
    built = term("joint_velocity", joints="legs")
    tick = inputs(state(velocity=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert values(built, tick) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_last_action_has_one_value_per_driven_joint() -> None:
    built = term("last_action")
    assert built.size == 6
    tick = inputs(last_action=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    assert values(built, tick) == [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

    data = descriptor_data()
    data["action"]["joints"] = ["left_knee", "right_knee"]
    policy = PolicyDescriptor.model_validate(data)
    assert term("last_action", policy=policy).size == 2


@pytest.mark.parametrize(
    ("elapsed_s", "expected"),
    [
        pytest.param(0.0, [0.0, 1.0], id="start"),
        pytest.param(0.2, [1.0, 0.0], id="a quarter period"),
        pytest.param(0.4, [0.0, -1.0], id="half a period"),
        pytest.param(0.8, [0.0, 1.0], id="one period"),
        pytest.param(1.0, [1.0, 0.0], id="a quarter past one period"),
        pytest.param(2.0, [0.0, -1.0], id="two and a half periods"),
    ],
)
def test_gait_phase(elapsed_s: float, expected: list[float]) -> None:
    built = term("gait_phase", params={"period": 0.8})
    assert built.size == 2
    tick = inputs(elapsed_s=elapsed_s)
    assert values(built, tick) == pytest.approx(expected, abs=1e-12)


def test_gait_phase_takes_a_whole_number_of_seconds() -> None:
    built = term("gait_phase", params={"period": 2})
    assert values(built, inputs(elapsed_s=0.5)) == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize(
    ("name", "fields", "message"),
    [
        pytest.param(
            "gait_phase",
            {},
            "observation term 'gait_phase' needs parameter 'period': the gait"
            " period, in s",
            id="missing period",
        ),
        pytest.param(
            "gait_phase",
            {"params": {"period": 0.8, "offset": 0.5}},
            "observation term 'gait_phase' has unknown parameter 'offset'; it takes"
            " 'period'",
            id="unknown parameter",
        ),
        pytest.param(
            "base_angular_velocity",
            {"params": {"frame": "base", "filter": True}},
            "observation term 'base_angular_velocity' has unknown parameters"
            " 'frame', 'filter'; it takes no parameters",
            id="parameters for a term that takes none",
        ),
        pytest.param(
            "gait_phase",
            {"params": {"period": 0}},
            "observation term 'gait_phase': parameter 'period' must be positive, got 0",
            id="zero period",
        ),
        pytest.param(
            "gait_phase",
            {"params": {"period": -0.8}},
            "observation term 'gait_phase': parameter 'period' must be positive,"
            " got -0.8",
            id="negative period",
        ),
        pytest.param(
            "gait_phase",
            {"params": {"period": "0.8"}},
            "observation term 'gait_phase': parameter 'period' must be a number,"
            " got '0.8'",
            id="period as text",
        ),
        pytest.param(
            "gait_phase",
            {"params": {"period": True}},
            "observation term 'gait_phase': parameter 'period' must be a number,"
            " got True",
            id="period as a boolean",
        ),
        pytest.param(
            "base_angular_velocity",
            {"joints": "legs"},
            "observation term 'base_angular_velocity' takes no joints, but joints is"
            " 'legs'",
            id="joints for a term that takes none",
        ),
        pytest.param(
            "last_action",
            {"joints": ["left_knee", "right_knee"]},
            "observation term 'last_action' takes no joints, but joints is"
            " [left_knee, right_knee]",
            id="joints for last_action",
        ),
        pytest.param(
            "joint_velocity",
            {},
            "observation term 'joint_velocity' needs joints: a joint group of the"
            " description, or a list of joint names",
            id="missing joints",
        ),
        pytest.param(
            "joint_position_relative",
            {"joints": ["left_knee", "left_foot"]},
            "observation term 'joint_position_relative': no joint 'left_foot' in"
            " robot 'testbot'",
            id="unknown joint",
        ),
        pytest.param(
            "joint_velocity",
            {"joints": "arms"},
            "observation term 'joint_velocity': no joint group 'arms' in robot"
            " 'testbot'",
            id="unknown group",
        ),
        pytest.param(
            "joint_velocity",
            {"joints": "legs", "params": {"relative": True}},
            "observation term 'joint_velocity' has unknown parameter 'relative'; it"
            " takes no parameters",
            id="parameters for a joint term",
        ),
    ],
)
def test_a_wrong_entry_is_rejected_naming_the_term(
    name: str, fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        term(name, **fields)
    assert str(error.value) == message


def test_last_action_needs_the_action_s_joints() -> None:
    data = descriptor_data()
    data["action"]["joints"] = "arms"
    policy = PolicyDescriptor.model_validate(data)
    with pytest.raises(ValueError) as error:
        term("last_action", policy=policy)
    assert str(error.value) == (
        "observation term 'last_action': action.joints: no joint group 'arms' in"
        " robot 'testbot'"
    )


def test_the_builder_rejects_a_wrong_entry() -> None:
    policy = descriptor([{"term": "gait_phase"}], 2)
    with pytest.raises(ValueError, match="^observation term 'gait_phase' needs"):
        ObservationBuilder(policy, robot())


# The builder.


def test_the_testbot_fixture_builds_with_its_declared_size() -> None:
    builder = ObservationBuilder(load_descriptor(TESTBOT_ZERO), robot())
    assert builder.frame_size == 27
    assert builder.size == 27
    assert builder.term_sizes == (
        ("base_angular_velocity", 3),
        ("projected_gravity", 3),
        ("velocity_command", 3),
        ("joint_position_relative", 6),
        ("joint_velocity", 6),
        ("last_action", 6),
    )
    tick = inputs(
        state(
            position=TESTBOT_DEFAULTS + [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            velocity=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            angular_velocity=(0.4, -0.8, 1.2),
        ),
        command=(0.5, 0.0, 0.0),
        last_action=[-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
    )
    built = builder.build(tick)
    assert built.dtype == np.float32
    expected = [
        *(0.1, -0.2, 0.3),  # base_angular_velocity × 0.25
        *(0.0, 0.0, -1.0),  # projected_gravity
        *(0.5, 0.0, 0.0),  # velocity_command
        *(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),  # joint_position_relative
        *(0.05, 0.1, 0.15, 0.2, 0.25, 0.3),  # joint_velocity × 0.05
        *(-0.1, -0.2, -0.3, -0.4, -0.5, -0.6),  # last_action
    ]
    np.testing.assert_allclose(built, expected, rtol=0, atol=1e-7)


def test_a_wrong_observation_size_lists_the_terms_sizes() -> None:
    data = descriptor_data()
    data["observation_size"] = 26
    with pytest.raises(ValueError) as error:
        ObservationBuilder(PolicyDescriptor.model_validate(data), robot())
    assert str(error.value) == (
        "observation_size is 26, but the terms give 27 values per frame:"
        "\n- observations[0] (base_angular_velocity): 3"
        "\n- observations[1] (projected_gravity): 3"
        "\n- observations[2] (velocity_command): 3"
        "\n- observations[3] (joint_position_relative): 6"
        "\n- observations[4] (joint_velocity): 6"
        "\n- observations[5] (last_action): 6"
    )


def g1_description() -> RobotDescription:
    """Return a 29-joint description shaped like the G1's, for the G1 fixture.

    The joints are the fixture's driven legs, then its held waist and arms,
    as in the G1's own order; `legs` is the group of the 12 leg joints.
    """
    data = descriptor_data(G1_VELOCITY)
    legs = data["action"]["joints"]
    defaults = {"hip_pitch": -0.1, "knee": 0.3, "ankle_pitch": -0.2}
    joints = [
        {
            "name": name,
            "motor_index": index,
            "default_position": defaults.get(name.split("_", 1)[1], 0.0),
            "limits": {"position": [-3.0, 3.0], "velocity": 30.0, "effort": 100.0},
            "gains": {"kp": 50.0, "kd": 1.0},
        }
        for index, name in enumerate([*legs, *data["held_joints"]])
    ]
    return RobotDescription.model_validate(
        {
            "schema_version": 1,
            "name": "g1_29dof",
            "status": "draft",
            "base_link": "pelvis",
            "joints": joints,
            "joint_groups": {"legs": legs},
            "imu": {"frame": "imu_link", "parent": "pelvis"},
            "rates": {"physics_hz": 1000.0, "control_hz_max": 500.0},
        }
    )


def test_the_g1_reference_descriptor_gives_47_inputs() -> None:
    # The fixture's description hash is a placeholder: the builder does not
    # check it.
    description = g1_description()
    assert description.num_joints == 29
    builder = ObservationBuilder(load_descriptor(G1_VELOCITY), description)
    assert builder.frame_size == 47
    assert builder.size == 47
    assert builder.term_sizes == (
        ("base_angular_velocity", 3),
        ("projected_gravity", 3),
        ("velocity_command", 3),
        ("joint_position_relative", 12),
        ("joint_velocity", 12),
        ("last_action", 12),
        ("gait_phase", 2),
    )
    defaults = np.array([joint.default_position for joint in description.joints])
    legs = np.arange(1.0, 13.0)  # 1 to 12, one value per leg joint
    upper_body = np.full(17, 9.0)  # waist and arms: not observed
    tick = inputs(
        state(
            position=defaults + np.concatenate([0.01 * legs, upper_body]),
            velocity=np.concatenate([legs, upper_body]),
            angular_velocity=(0.4, 0.8, -1.2),
        ),
        command=(0.5, -0.5, 1.0),
        last_action=-0.1 * legs,
        elapsed_s=0.2,  # a quarter of the 0.8 s gait period
    )
    expected = [
        *(0.1, 0.2, -0.3),  # base_angular_velocity × 0.25
        *(0.0, 0.0, -1.0),  # projected_gravity
        *(1.0, -1.0, 0.25),  # velocity_command × (2, 2, 0.25)
        *(0.01 * legs),  # joint_position_relative on legs
        *(0.05 * legs),  # joint_velocity on legs × 0.05
        *(-0.1 * legs),  # last_action
        *(1.0, 0.0),  # gait_phase: sin and cos of a quarter turn
    ]
    np.testing.assert_allclose(builder.build(tick), expected, rtol=0, atol=1e-6)


def test_scalar_and_element_wise_scales() -> None:
    policy = descriptor(
        [
            {"term": "base_angular_velocity", "scale": 0.25},
            {"term": "velocity_command", "scale": [2.0, -1.0, 0.5]},
            {"term": "joint_velocity", "joints": "left_leg", "scale": [1.0, 0.5, 0.25]},
        ],
        9,
    )
    builder = ObservationBuilder(policy, robot())
    tick = inputs(
        state(velocity=[4.0, 4.0, 4.0, 0.0, 0.0, 0.0], angular_velocity=(1, 2, 3)),
        command=(0.5, 0.5, 0.5),
    )
    assert builder.build(tick).tolist() == [
        *(0.25, 0.5, 0.75),
        *(1.0, -0.5, 0.25),
        *(4.0, 2.0, 1.0),
    ]


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        pytest.param(
            {"term": "velocity_command", "scale": [2.0, 2.0]},
            "observations[0] (velocity_command): scale has 2 elements, but the term"
            " has 3 values",
            id="a term without joints",
        ),
        pytest.param(
            # The descriptor cannot check this one: its joints are a group.
            {"term": "joint_velocity", "joints": "legs", "scale": [0.05] * 5},
            "observations[0] (joint_velocity): scale has 5 elements, but the term"
            " has 6 values",
            id="a term on a joint group",
        ),
    ],
)
def test_a_scale_of_the_wrong_length_is_rejected(
    entry: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        ObservationBuilder(descriptor([entry], 3), robot())
    assert str(error.value) == message


def test_build_returns_a_new_float32_array_each_time() -> None:
    builder = ObservationBuilder(descriptor([{"term": "velocity_command"}], 3), robot())
    first = builder.build(inputs(command=(1.0, 2.0, 3.0)))
    first[:] = 0.0
    second = builder.build(inputs(command=(1.0, 2.0, 3.0)))
    assert second.dtype == np.float32
    assert second.tolist() == [1.0, 2.0, 3.0]
    assert second is not first


def test_non_finite_values_reach_the_input_as_data() -> None:
    builder = ObservationBuilder(load_descriptor(TESTBOT_ZERO), robot())
    nan = math.nan
    tick = inputs(state(velocity=[nan, 0, 0, 0, 0, 0], orientation=(nan, 0, 0, 1)))
    built = builder.build(tick)
    assert np.isnan(built[3:6]).all()  # projected_gravity
    assert np.isnan(built[15])  # joint_velocity of left_hip_pitch
    assert np.isfinite(np.delete(built, [3, 4, 5, 15])).all()


# The history. Two terms with values that say which term, tick and element
# they are: velocity_command (3 values) is 10t + j and last_action (6
# values) is 100t + j at tick t, for element j from 1.


def history_builder(layout: str) -> ObservationBuilder:
    terms = [{"term": "velocity_command"}, {"term": "last_action"}]
    history = {"length": 3, "layout": layout}
    return ObservationBuilder(descriptor(terms, 9, history=history), robot())


def command_at(t: int) -> list[float]:
    return [10.0 * t + j for j in range(1, 4)]


def action_at(t: int) -> list[float]:
    return [100.0 * t + j for j in range(1, 7)]


def tick_at(t: int) -> TermInputs:
    return inputs(command=command_at(t), last_action=action_at(t))


def expected_input(ticks: list[int], layout: str) -> list[float]:
    """Return the input for the frames of `ticks`, oldest first."""
    if layout == "per_term":
        commands = [value for t in ticks for value in command_at(t)]
        return commands + [value for t in ticks for value in action_at(t)]
    return [value for t in ticks for value in command_at(t) + action_at(t)]


def test_history_sizes() -> None:
    builder = history_builder("per_term")
    assert builder.frame_size == 9
    assert builder.size == 27
    assert builder.term_sizes == (("velocity_command", 3), ("last_action", 6))


def test_per_term_layout_keeps_each_term_s_frames_together() -> None:
    builder = history_builder("per_term")
    for t in (1, 2):
        builder.build(tick_at(t))
    assert builder.build(tick_at(3)).tolist() == [
        *(11.0, 12.0, 13.0, 21.0, 22.0, 23.0, 31.0, 32.0, 33.0),
        *(101.0, 102.0, 103.0, 104.0, 105.0, 106.0),
        *(201.0, 202.0, 203.0, 204.0, 205.0, 206.0),
        *(301.0, 302.0, 303.0, 304.0, 305.0, 306.0),
    ]


def test_per_frame_layout_stacks_whole_frames() -> None:
    builder = history_builder("per_frame")
    for t in (1, 2):
        builder.build(tick_at(t))
    assert builder.build(tick_at(3)).tolist() == [
        *(11.0, 12.0, 13.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0),
        *(21.0, 22.0, 23.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0),
        *(31.0, 32.0, 33.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0),
    ]


@pytest.mark.parametrize("layout", ["per_term", "per_frame"])
def test_history_drops_the_oldest_frame(layout: str) -> None:
    builder = history_builder(layout)
    built = [builder.build(tick_at(t)).tolist() for t in (1, 2, 3, 4, 5)]
    assert built[2] == expected_input([1, 2, 3], layout)
    assert built[3] == expected_input([2, 3, 4], layout)
    assert built[4] == expected_input([3, 4, 5], layout)


@pytest.mark.parametrize("layout", ["per_term", "per_frame"])
def test_history_starts_filled_with_the_first_frame(layout: str) -> None:
    builder = history_builder(layout)
    assert builder.build(tick_at(1)).tolist() == expected_input([1, 1, 1], layout)
    assert builder.build(tick_at(2)).tolist() == expected_input([1, 1, 2], layout)


@pytest.mark.parametrize("layout", ["per_term", "per_frame"])
def test_reset_fills_the_history_with_the_next_frame(layout: str) -> None:
    builder = history_builder(layout)
    for t in (1, 2, 3):
        builder.build(tick_at(t))
    builder.reset()
    assert builder.build(tick_at(7)).tolist() == expected_input([7, 7, 7], layout)
    assert builder.build(tick_at(8)).tolist() == expected_input([7, 7, 8], layout)


def test_a_term_of_the_wrong_size_fails_the_build_and_keeps_the_history() -> None:
    builder = history_builder("per_term")
    builder.build(tick_at(1))
    wrong = inputs(command=command_at(2), last_action=action_at(2)[:5])
    with pytest.raises(ValueError) as error:
        builder.build(wrong)
    assert str(error.value) == (
        "observations[1] (last_action) returned an array of shape (5,), but the"
        " term's size is 6"
    )
    assert builder.build(tick_at(3)).tolist() == expected_input([1, 1, 3], "per_term")


# Terms from other packages.


class JointEffort:
    """A term the platform does not provide: joint torques, in N·m."""

    def __init__(self, indices: NDArray[np.intp]) -> None:
        self.indices = indices

    @property
    def size(self) -> int:
        return self.indices.size

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.state.effort[self.indices]


def joint_effort(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    if entry.joints is None or entry.params:
        raise ValueError(f"observation term {entry.term!r} needs joints, no params")
    names = resolve_joints(entry.joints, description)
    return JointEffort(np.array([description.index_of(name) for name in names]))


def not_a_term(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> object:
    return 3.0


@pytest.fixture
def lab_terms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a package `lab_terms` that registers terms, beside the built-in ones."""
    module = ModuleType("lab_terms")
    module.__dict__.update(
        joint_effort=joint_effort, not_a_term=not_a_term, not_callable=0.8
    )
    monkeypatch.setitem(sys.modules, "lab_terms", module)
    plugins = [
        EntryPoint(name=name, value=f"lab_terms:{name}", group=OBSERVATIONS)
        for name in ("joint_effort", "not_a_term", "not_callable")
    ]
    installed: Callable[..., Any] = registry.entry_points

    def select(*, group: str) -> list[EntryPoint]:
        found = list(installed(group=group))
        return [*found, *plugins] if group == OBSERVATIONS else found

    monkeypatch.setattr(registry, "entry_points", select)


@pytest.mark.usefixtures("lab_terms")
def test_a_term_from_another_package() -> None:
    assert term_factory("joint_effort") is joint_effort
    policy = descriptor(
        [
            {"term": "velocity_command"},
            {"term": "joint_effort", "joints": ["right_knee", "left_knee"]},
        ],
        5,
    )
    builder = ObservationBuilder(policy, robot())
    assert builder.term_sizes == (("velocity_command", 3), ("joint_effort", 2))
    tick = inputs(state(effort=[0.0, 10.0, 0.0, 0.0, 20.0, 0.0]), command=(1, 0, 0))
    assert builder.build(tick).tolist() == [1.0, 0.0, 0.0, 20.0, 10.0]


@pytest.mark.usefixtures("lab_terms")
def test_a_factory_must_build_a_term() -> None:
    with pytest.raises(TypeError) as error:
        ObservationBuilder(descriptor([{"term": "not_a_term"}], 3), robot())
    assert str(error.value) == (
        "observations[0] (not_a_term): the factory returned 3.0, which is not a"
        " Term: a term has a size and is called with the tick's inputs"
    )


@pytest.mark.usefixtures("lab_terms")
def test_a_registered_term_must_be_callable() -> None:
    with pytest.raises(TypeError) as error:
        term_factory("not_callable")
    assert str(error.value) == (
        "observation term 'not_callable' is registered as 0.8, which is not"
        " callable: an entry point in group 'embodiverse.observations' must name a"
        " TermFactory"
    )


def test_an_unknown_term_lists_the_registered_ones() -> None:
    policy = descriptor([{"term": "base_linear_velocity"}], 3)
    with pytest.raises(RegistryError) as error:
        ObservationBuilder(policy, robot())
    assert str(error.value) == (
        "no 'base_linear_velocity' in group 'embodiverse.observations'; available: "
        + ", ".join(BUILT_IN)
    )
