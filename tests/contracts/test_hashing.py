import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel

from embodiverse.contracts.description import (
    Gains,
    RobotDescription,
    Vendor,
    load_description,
)
from embodiverse.contracts.hashing import canonical_json, content_hash

FIXTURE = Path(__file__).parents[1] / "fixtures/robots/minimal/description.yaml"

# The fixture's hash. It may change only with a deliberate change to the
# hashing scheme or to the fixture: anything else that changes it would also
# change the hash of every real description and invalidate recorded hashes.
GOLDEN_HASH = "sha256:4b65f4061f0e797fa34454b79fbbc48ab8b274fe1270bc4d11b3ccb9cab7bfa4"

# The fixture written by hand in another style: other indentation, flow
# mappings, other quoting, keys in another order and other comments.
HAND_WRITTEN = """\
# A two-joint robot.
models: {mjcf: model.xml}
rates: {control_hz_max: 500.0, physics_hz: 1000.0}
sensors:
    front_lidar: {type: lidar, parent: pelvis, frame: lidar_link,
                  mount_rpy: [0.0, 0.2, 0.0], mount_xyz: [0.1, 0.0, 0.15]}
imu: {parent: pelvis, frame: imu_link, mount_xyz: [-0.04, 0.0, -0.08]}
joint_groups:
    legs:
        - left_knee
        - right_knee
joints:
    -   name: left_knee      # knees only
        motor_index: 0
        vendor_name: "KNEE_L"
        gains: {kd: 4.0, kp: 150.0}
        limits: {effort: 139.0, velocity: 20.0, position: [-0.087, 2.88]}
        default_position: 0.3
    -   name: right_knee
        motor_index: 1
        vendor_name: 'KNEE_R'
        gains: {kd: 4.0, kp: 150.0}
        limits: {effort: 139.0, velocity: 20.0, position: [-0.087, 2.88]}
        default_position: 0.3
base_link: pelvis
vendor: {model: EX-2, name: "Example Robotics"}
status: draft
name: minimal
schema_version: 1
"""


def fixture_data() -> dict[str, Any]:
    """Return the minimal fixture as plain data, for building variants."""
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def hash_of(data: dict[str, Any]) -> str:
    return content_hash(RobotDescription.model_validate(data))


def hash_of_file(path: Path, text: str) -> str:
    """Write `text` to `path` and return the hash of the description it holds."""
    path.write_text(text, encoding="utf-8")
    return content_hash(load_description(path))


def reversed_keys(value: Any) -> Any:
    """Return `value` with the keys of every mapping in reverse order."""
    if isinstance(value, dict):
        return {key: reversed_keys(value[key]) for key in reversed(value)}
    if isinstance(value, list):
        return [reversed_keys(item) for item in value]
    return value


def whole_floats_as_ints(value: Any) -> Any:
    """Return `value` with every whole-number float, such as 100.0, as an int."""
    if isinstance(value, dict):
        return {key: whole_floats_as_ints(item) for key, item in value.items()}
    if isinstance(value, list):
        return [whole_floats_as_ints(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@pytest.fixture
def original_hash() -> str:
    return content_hash(load_description(FIXTURE))


def test_hash_names_its_algorithm_and_has_64_hex_digits(original_hash: str) -> None:
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", original_hash)


def test_hash_of_the_fixture_matches_the_golden_value(original_hash: str) -> None:
    assert original_hash == GOLDEN_HASH


def test_hash_is_the_sha256_of_the_canonical_json() -> None:
    description = load_description(FIXTURE)
    digest = hashlib.sha256(canonical_json(description).encode("utf-8")).hexdigest()
    assert content_hash(description) == f"sha256:{digest}"


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda data: yaml.safe_dump(data), id="sorted keys"),
        pytest.param(
            lambda data: yaml.safe_dump(reversed_keys(data), sort_keys=False),
            id="reversed keys",
        ),
        pytest.param(
            lambda data: yaml.safe_dump(data, default_flow_style=True), id="flow style"
        ),
        pytest.param(
            lambda data: yaml.safe_dump(data, sort_keys=False, indent=4),
            id="indented by four",
        ),
        pytest.param(lambda data: HAND_WRITTEN, id="by hand"),
    ],
)
def test_hash_ignores_formatting_comments_and_key_order(
    tmp_path: Path, original_hash: str, write: Callable[[dict[str, Any]], str]
) -> None:
    text = write(fixture_data())
    assert hash_of_file(tmp_path / "description.yaml", text) == original_hash


def test_hash_ignores_how_whole_numbers_are_written(
    tmp_path: Path, original_hash: str
) -> None:
    text = yaml.safe_dump(whole_floats_as_ints(fixture_data()))
    assert "physics_hz: 1000\n" in text
    assert hash_of_file(tmp_path / "description.yaml", text) == original_hash


@pytest.mark.parametrize(
    "write_default",
    [
        pytest.param(
            lambda data: data["joints"][0]["gains"].update(side="joint"),
            id="gains side",
        ),
        pytest.param(
            lambda data: data["imu"].update(mount_rpy=[0, 0, 0]), id="imu mount"
        ),
        pytest.param(lambda data: data["models"].update(urdf=None), id="model path"),
    ],
)
def test_hash_ignores_defaults_written_out(
    original_hash: str, write_default: Callable[[dict[str, Any]], None]
) -> None:
    data = fixture_data()
    write_default(data)
    assert hash_of(data) == original_hash


class WithOptionalField(RobotDescription):
    """The schema as it would be after gaining an optional field."""

    total_mass: float | None = None


def test_hash_survives_a_new_optional_field_in_the_schema(original_hash: str) -> None:
    description = WithOptionalField.model_validate(fixture_data())
    assert content_hash(description) == original_hash


def test_hash_does_not_depend_on_file_location(
    tmp_path: Path, original_hash: str
) -> None:
    paths = [tmp_path / "a/description.yaml", tmp_path / "b/c/robot.yaml"]
    for path in paths:
        path.parent.mkdir(parents=True)
        shutil.copy(FIXTURE, path)
    hashes = {content_hash(load_description(path)) for path in paths}
    assert hashes == {original_hash}


def test_hash_ignores_the_order_of_named_groups_and_sensors() -> None:
    # The dump keeps mappings in the order they were read; only sorting the
    # keys makes these two equal.
    data = fixture_data()
    data["joint_groups"]["knees"] = ["right_knee", "left_knee"]
    lidar = data["sensors"]["front_lidar"]
    data["sensors"]["rear_lidar"] = dict(lidar, frame="rear_lidar_link")
    assert hash_of(data) == hash_of(reversed_keys(data))


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(
            lambda data: data["joints"][0]["limits"].update(effort=140.0),
            id="limit",
        ),
        pytest.param(
            lambda data: data["joints"][1]["gains"].update(kp=151.0), id="gain"
        ),
        pytest.param(
            lambda data: data["joints"][0]["gains"].update(side="motor"),
            id="gains side",
        ),
        pytest.param(lambda data: data["joints"].reverse(), id="joint order"),
        pytest.param(
            lambda data: data["joints"][0].update(default_position=0.3 + 1e-9),
            id="default position",
        ),
        pytest.param(
            lambda data: data["joint_groups"]["legs"].reverse(), id="group order"
        ),
        pytest.param(lambda data: data.update(name="minimal_two"), id="name"),
        pytest.param(lambda data: data.update(status="active"), id="status"),
        pytest.param(
            lambda data: data["sensors"]["front_lidar"].update(mount_rpy=[0, 0.3, 0]),
            id="sensor mount",
        ),
        pytest.param(
            lambda data: data["imu"].update(mount_xyz=[-0.04, 0.0, -0.07]),
            id="imu mount",
        ),
        pytest.param(lambda data: data["rates"].update(physics_hz=500.0), id="rate"),
        pytest.param(
            lambda data: data["models"].update(mjcf="other.xml"), id="model path"
        ),
    ],
)
def test_hash_changes_with_any_value(
    original_hash: str, change: Callable[[dict[str, Any]], None]
) -> None:
    data = fixture_data()
    change(data)
    assert hash_of(data) != original_hash


def test_canonical_json_is_compact_sorted_and_leaves_out_defaults() -> None:
    assert canonical_json(Gains(kp=150, kd=4)) == '{"kd":4.0,"kp":150.0}'
    motor = Gains(kp=150, kd=4, side="motor")
    assert canonical_json(motor) == '{"kd":4.0,"kp":150.0,"side":"motor"}'


def test_canonical_json_of_the_fixture_has_no_whitespace_and_sorted_keys() -> None:
    text = canonical_json(load_description(FIXTURE))
    compact = json.dumps(
        json.loads(text), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    assert text == compact


def test_canonical_json_keeps_non_ascii_characters() -> None:
    vendor = Vendor(name="Müller", model="X1")
    assert canonical_json(vendor) == '{"model":"X1","name":"Müller"}'


class Reading(BaseModel):
    value: float


def test_hashing_rejects_non_finite_numbers() -> None:
    with pytest.raises(ValueError, match="not JSON compliant"):
        content_hash(Reading(value=math.inf))
