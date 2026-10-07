from importlib.resources import as_file, files

import pytest

from embodiverse.contracts.description import load_description
from embodiverse.contracts.registry import robot_description

TESTBOT = "embodiverse.robots.testbot"
TESTBOT_JOINTS = (
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
)


@pytest.mark.parametrize("name", ["description.yaml", "testbot.xml"])
def test_testbot_files_are_installed(name: str) -> None:
    assert files(TESTBOT).joinpath(name).is_file()


def test_testbot_description_is_valid() -> None:
    with as_file(files(TESTBOT).joinpath("description.yaml")) as path:
        description = load_description(path)
        mjcf = description.model_path("mjcf")
        assert mjcf is not None
        assert mjcf.is_file()
    assert description.name == "testbot"
    assert description.status == "active"
    assert description.joint_names == TESTBOT_JOINTS


@pytest.mark.parametrize(
    ("name", "joints"), [("g1_29dof", 29), ("pm01", 24), ("k1_22dof", 22)]
)
def test_draft_descriptions_load_through_the_registry(name: str, joints: int) -> None:
    description = robot_description(name)
    assert description.name == name
    assert description.status == "draft"
    assert description.num_joints == joints
    assert description.model_path("mjcf") is None
    assert description.model_path("urdf") is None


@pytest.mark.parametrize("name", ["g1_29dof", "pm01", "k1_22dof"])
def test_draft_descriptions_list_joints_in_motor_order(name: str) -> None:
    description = robot_description(name)
    indices = [joint.motor_index for joint in description.joints]
    assert indices == list(range(description.num_joints))
    assert all(joint.vendor_name for joint in description.joints)


def test_draft_descriptions_share_canonical_leg_names() -> None:
    legs = {
        f"{side}_{part}"
        for side in ("left", "right")
        for part in (
            "hip_pitch",
            "hip_roll",
            "hip_yaw",
            "knee",
            "ankle_pitch",
            "ankle_roll",
        )
    }
    for name in ("g1_29dof", "pm01", "k1_22dof"):
        assert set(robot_description(name).group("legs")) == legs
