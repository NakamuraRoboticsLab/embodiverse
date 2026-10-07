import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from embodiverse.contracts.description import (
    DescriptionError,
    RobotDescription,
    load_description,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/robots/minimal/description.yaml"


def fixture_data() -> dict[str, Any]:
    """Return the minimal fixture as plain data, for building variants."""
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def starts_with(path: Path) -> str:
    return f"^{re.escape(str(path))}: "


@pytest.fixture
def description() -> RobotDescription:
    return load_description(FIXTURE)


def test_fixture_loads(description: RobotDescription) -> None:
    assert description.name == "minimal"
    assert description.vendor is not None
    assert description.vendor.model == "EX-2"
    assert description.joints[0].gains.side == "joint"
    assert description.sensors["front_lidar"].mount_rpy == (0.0, 0.2, 0.0)


def test_joint_names_follow_description_order(description: RobotDescription) -> None:
    assert description.joint_names == ("left_knee", "right_knee")
    assert description.num_joints == 2


def test_joints_are_found_by_name(description: RobotDescription) -> None:
    assert description.index_of("right_knee") == 1
    assert description.joint("right_knee").vendor_name == "KNEE_R"


@pytest.mark.parametrize("lookup", [RobotDescription.index_of, RobotDescription.joint])
def test_unknown_joint_raises_key_error(
    description: RobotDescription, lookup: Callable[[RobotDescription, str], object]
) -> None:
    with pytest.raises(KeyError) as error:
        lookup(description, "left_ankle")
    assert error.value.args == ("no joint 'left_ankle' in robot 'minimal'",)


def test_group_returns_its_joint_names(description: RobotDescription) -> None:
    assert description.group("legs") == ("left_knee", "right_knee")


def test_unknown_group_raises_key_error(description: RobotDescription) -> None:
    with pytest.raises(KeyError) as error:
        description.group("arms")
    assert error.value.args == ("no joint group 'arms' in robot 'minimal'",)


def test_rejects_joint_name_that_is_not_snake_case() -> None:
    data = fixture_data()
    data["joints"][0]["name"] = "LeftKnee"
    with pytest.raises(ValidationError, match="joint name 'LeftKnee' is not lowercase"):
        RobotDescription.model_validate(data)


def test_rejects_duplicate_joint_names() -> None:
    data = fixture_data()
    data["joints"][1]["name"] = "left_knee"
    with pytest.raises(ValidationError, match="joint 'left_knee' is listed more than"):
        RobotDescription.model_validate(data)


def test_rejects_shared_motor_index() -> None:
    data = fixture_data()
    data["joints"][1]["motor_index"] = 0
    match = "joints 'left_knee' and 'right_knee' both use motor index 0"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_default_position_outside_limits() -> None:
    data = fixture_data()
    data["joints"][1]["default_position"] = 3.0
    match = r"joint 'right_knee': default position 3\.0 rad is outside"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_inverted_position_limits() -> None:
    data = fixture_data()
    data["joints"][0]["limits"]["position"] = [1.0, -1.0]
    with pytest.raises(ValidationError, match="lower limit 1.0 rad must be below"):
        RobotDescription.model_validate(data)


@pytest.mark.parametrize("limit", ["velocity", "effort"])
def test_rejects_limit_that_is_not_positive(limit: str) -> None:
    data = fixture_data()
    data["joints"][0]["limits"][limit] = 0.0
    match = rf"limits\.{limit}\s+Input should be greater than 0"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


@pytest.mark.parametrize("gain", ["kp", "kd"])
def test_rejects_negative_gain(gain: str) -> None:
    data = fixture_data()
    data["joints"][0]["gains"][gain] = -1.0
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        RobotDescription.model_validate(data)


def test_rejects_unknown_gain_side() -> None:
    data = fixture_data()
    data["joints"][0]["gains"]["side"] = "actuator"
    with pytest.raises(ValidationError, match="'joint' or 'motor'"):
        RobotDescription.model_validate(data)


def test_rejects_non_finite_number() -> None:
    data = fixture_data()
    data["joints"][0]["limits"]["velocity"] = float("inf")
    with pytest.raises(ValidationError, match="finite number"):
        RobotDescription.model_validate(data)


def test_rejects_group_with_unknown_joint() -> None:
    data = fixture_data()
    data["joint_groups"]["legs"].append("left_ankle")
    match = "joint group 'legs' names unknown joint 'left_ankle'"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_group_name_that_is_not_snake_case() -> None:
    data = fixture_data()
    data["joint_groups"] = {"Legs": ["left_knee"]}
    match = r"joint_groups\.Legs\.\[key\]\s+String should match pattern"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_imu_frame_that_is_the_base_link() -> None:
    data = fixture_data()
    data["imu"]["frame"] = "pelvis"
    match = "the base link and the IMU both use frame 'pelvis'"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_sensor_frame_that_is_the_imu_frame() -> None:
    data = fixture_data()
    data["sensors"]["front_lidar"]["frame"] = "imu_link"
    match = "the IMU and sensor 'front_lidar' both use frame 'imu_link'"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_sensors_that_share_a_frame() -> None:
    data = fixture_data()
    data["sensors"]["rear_lidar"] = dict(data["sensors"]["front_lidar"])
    match = "sensor 'front_lidar' and sensor 'rear_lidar' both use frame 'lidar_link'"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_sensor_name_that_is_not_snake_case() -> None:
    data = fixture_data()
    data["sensors"] = {"Front Lidar": data["sensors"]["front_lidar"]}
    match = r"sensors\.Front Lidar\.\[key\]\s+String should match pattern"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_sensor_type_that_is_not_snake_case() -> None:
    data = fixture_data()
    data["sensors"]["front_lidar"]["type"] = "LiDAR"
    match = r"sensors\.front_lidar\.type\s+String should match pattern"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_control_rate_above_physics_rate() -> None:
    data = fixture_data()
    data["rates"]["control_hz_max"] = 2000.0
    match = r"control_hz_max \(2000\.0 Hz\) must not exceed physics_hz \(1000\.0 Hz\)"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_misspelled_key() -> None:
    data = fixture_data()
    data["joints"][0]["defualt_position"] = 0.3
    match = r"joints\.0\.defualt_position\s+Extra inputs are not permitted"
    with pytest.raises(ValidationError, match=match):
        RobotDescription.model_validate(data)


def test_rejects_unknown_schema_version() -> None:
    data = fixture_data()
    data["schema_version"] = 2
    with pytest.raises(ValidationError, match=r"schema_version\s+Input should be 1"):
        RobotDescription.model_validate(data)


def test_rejects_robot_without_joints() -> None:
    data = fixture_data()
    data["joints"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        RobotDescription.model_validate(data)


def test_rejects_robot_name_that_is_not_snake_case() -> None:
    data = fixture_data()
    data["name"] = "Minimal"
    with pytest.raises(ValidationError, match=r"name\s+String should match pattern"):
        RobotDescription.model_validate(data)


def test_descriptions_are_immutable(description: RobotDescription) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        description.name = "other"
    with pytest.raises(ValidationError, match="frozen"):
        description.joints[0].default_position = 0.0


def test_load_error_for_missing_file_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "description.yaml"
    with pytest.raises(DescriptionError, match=starts_with(path) + "cannot read"):
        load_description(path)


def test_load_error_for_invalid_yaml_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "description.yaml"
    path.write_text("joints: [left_knee\n", encoding="utf-8")
    with pytest.raises(DescriptionError, match=starts_with(path) + "invalid YAML"):
        load_description(path)


def test_load_error_for_non_mapping_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "description.yaml"
    path.write_text("- left_knee\n", encoding="utf-8")
    match = starts_with(path) + "expected a mapping at the top level, got list"
    with pytest.raises(DescriptionError, match=match):
        load_description(path)


def test_load_error_for_invalid_description_keeps_field_details(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["joints"][0]["limits"]["velocity"] = -1.0
    path = tmp_path / "description.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptionError, match=starts_with(path)) as error:
        load_description(path)
    assert "joints.0.limits.velocity" in str(error.value)
    assert isinstance(error.value.__cause__, ValidationError)


def test_model_path_resolves_relative_to_the_file(
    description: RobotDescription,
) -> None:
    assert description.model_path("mjcf") == FIXTURE.parent.absolute() / "model.xml"


def test_model_path_is_none_when_unset(description: RobotDescription) -> None:
    assert description.model_path("urdf") is None


def test_model_path_keeps_absolute_paths(tmp_path: Path) -> None:
    data = fixture_data()
    data["models"]["urdf"] = str(tmp_path / "robot.urdf")
    description = RobotDescription.model_validate(data)
    assert description.model_path("urdf") == tmp_path / "robot.urdf"


def test_relative_model_path_needs_a_description_file() -> None:
    description = RobotDescription.model_validate(fixture_data())
    with pytest.raises(ValueError, match="'minimal' was not loaded from a file"):
        description.model_path("mjcf")


def test_dump_does_not_depend_on_file_location(tmp_path: Path) -> None:
    paths = [tmp_path / "a/description.yaml", tmp_path / "b/c/description.yaml"]
    for path in paths:
        path.parent.mkdir(parents=True)
        shutil.copy(FIXTURE, path)
    first, second = (load_description(path) for path in paths)
    assert first.model_path("mjcf") != second.model_path("mjcf")
    assert first.model_dump() == second.model_dump()
