import hashlib
import re
import shutil
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    DescriptorError,
    PolicyDescriptor,
    load_descriptor,
    resolve_joints,
    validate_against,
    verify_policy_file,
)
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.registry import robot_description

POLICIES = Path(__file__).parents[1] / "fixtures/policies"
TESTBOT_ZERO = POLICIES / "testbot_zero/descriptor.yaml"
G1_VELOCITY = POLICIES / "g1_velocity/descriptor.yaml"

TESTBOT_JOINTS = (
    "left_hip_pitch",
    "left_knee",
    "left_ankle_pitch",
    "right_hip_pitch",
    "right_knee",
    "right_ankle_pitch",
)
LEFT_LEG = TESTBOT_JOINTS[:3]
RIGHT_LEG = TESTBOT_JOINTS[3:]


def fixture_data(path: Path = TESTBOT_ZERO) -> dict[str, Any]:
    """Return a descriptor fixture as plain data, for building variants."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def changed_testbot(change: Callable[[dict[str, Any]], None]) -> RobotDescription:
    """Return testbot's description with `change` applied to its data."""
    file = files("embodiverse.robots.testbot").joinpath("description.yaml")
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    change(data)
    return RobotDescription.model_validate(data)


def starts_with(path: Path) -> str:
    return f"^{re.escape(str(path))}: "


def explicit_gains(*joints: str) -> dict[str, Any]:
    """Return the data of explicit gains for `joints`."""
    values = {name: {"kp": 50.0, "kd": 1.0} for name in joints}
    return {"source": "explicit", "values": values}


def problems(data: dict[str, Any], description: RobotDescription) -> list[str]:
    """Return the problems validate_against reports for `data` on `description`."""
    try:
        validate_against(PolicyDescriptor.model_validate(data), description)
    except DescriptorError as error:
        return str(error).split("\n- ")[1:]
    return []


@pytest.fixture
def descriptor() -> PolicyDescriptor:
    return load_descriptor(TESTBOT_ZERO)


@pytest.fixture
def testbot() -> RobotDescription:
    return robot_description("testbot")


# The fixtures.


def test_testbot_fixture_loads(descriptor: PolicyDescriptor) -> None:
    assert descriptor.policy is not None
    assert descriptor.policy.format == "onnx"
    assert descriptor.robot.name == "testbot"
    assert descriptor.rate_hz == 50.0
    assert descriptor.command.ranges.lin_vel_y == (0.0, 0.0)
    assert descriptor.observations[4].term == "joint_velocity"
    assert descriptor.observations[4].joints == "legs"
    assert descriptor.observations[4].scale == 0.05
    assert descriptor.observations[0].params == {}
    assert descriptor.action.joints == "legs"
    assert descriptor.gains.source == "description"


def test_defaults(descriptor: PolicyDescriptor) -> None:
    assert descriptor.history.length == 1
    assert descriptor.history.order == "oldest_first"
    assert descriptor.history.layout == "per_term"
    assert descriptor.observations[1].scale == 1.0
    assert descriptor.observations[1].joints is None
    assert descriptor.gains.values == {}
    assert descriptor.held_joints == ()
    assert descriptor.provenance == "external"


def test_testbot_fixture_records_testbot_s_description_hash(
    descriptor: PolicyDescriptor, testbot: RobotDescription
) -> None:
    assert descriptor.robot.description_hash == content_hash(testbot), (
        f"testbot's description has changed: write its new hash into {TESTBOT_ZERO}"
    )


def test_testbot_fixture_fits_testbot(
    descriptor: PolicyDescriptor, testbot: RobotDescription
) -> None:
    validate_against(descriptor, testbot)


def test_testbot_fixture_names_its_policy_file_s_hash(
    descriptor: PolicyDescriptor,
) -> None:
    verify_policy_file(descriptor)


def test_g1_fixture_loads_and_validates_on_its_own() -> None:
    descriptor = load_descriptor(G1_VELOCITY)
    assert descriptor.robot.name == "g1_29dof"
    assert descriptor.rate_hz == 50.0
    assert descriptor.observations[2].scale == (2.0, 2.0, 0.25)
    assert descriptor.observations[6].params == {"period": 0.8}
    assert isinstance(descriptor.action.joints, tuple)
    assert isinstance(descriptor.held_joints, tuple)
    assert len(descriptor.action.joints) == 12
    assert len(set(descriptor.action.joints) | set(descriptor.held_joints)) == 29
    assert descriptor.gains.values["left_knee"].kp == 150.0


def test_g1_fixture_declares_the_reference_observation_size() -> None:
    # Angular velocity, gravity and command; leg positions, velocities and
    # last actions; the sine and cosine of the gait phase.
    assert load_descriptor(G1_VELOCITY).observation_size == 3 + 3 + 3 + 12 + 12 + 12 + 2


# The policy file.


def test_policy_path_resolves_next_to_the_descriptor(
    descriptor: PolicyDescriptor,
) -> None:
    assert descriptor.policy_path() == TESTBOT_ZERO.parent.absolute() / "policy.onnx"


def test_policy_path_keeps_absolute_paths(tmp_path: Path) -> None:
    data = fixture_data()
    data["policy"]["file"] = str(tmp_path / "policy.onnx")
    descriptor = PolicyDescriptor.model_validate(data)
    assert descriptor.policy_path() == tmp_path / "policy.onnx"


def test_relative_policy_path_needs_a_descriptor_file() -> None:
    descriptor = PolicyDescriptor.model_validate(fixture_data())
    with pytest.raises(ValueError, match="descriptor was not loaded from a file"):
        descriptor.policy_path()


def copy_fixture(tmp_path: Path) -> Path:
    """Copy the testbot fixture's directory and return the copied descriptor."""
    directory = tmp_path / "testbot_zero"
    shutil.copytree(TESTBOT_ZERO.parent, directory)
    return directory / TESTBOT_ZERO.name


def test_verify_policy_file_rejects_a_changed_file(tmp_path: Path) -> None:
    descriptor = load_descriptor(copy_fixture(tmp_path))
    policy = descriptor.policy_path()
    policy.write_bytes(policy.read_bytes() + b"\n")
    actual = f"sha256:{hashlib.sha256(policy.read_bytes()).hexdigest()}"
    with pytest.raises(DescriptorError, match=starts_with(policy)) as error:
        verify_policy_file(descriptor)
    assert f"the policy file's hash is {actual}" in str(error.value)
    assert descriptor.policy is not None
    assert f"the descriptor expects {descriptor.policy.sha256}" in str(error.value)


def test_verify_policy_file_hashes_large_files(tmp_path: Path) -> None:
    # Larger than one chunk read, so the hash covers several reads.
    content = bytes(range(256)) * 10_000
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(content)
    data = fixture_data()
    data["policy"]["sha256"] = f"sha256:{hashlib.sha256(content).hexdigest()}"
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    verify_policy_file(load_descriptor(path))


def test_verify_policy_file_reports_a_missing_file(tmp_path: Path) -> None:
    descriptor = load_descriptor(copy_fixture(tmp_path))
    descriptor.policy_path().unlink()
    match = starts_with(descriptor.policy_path()) + "cannot read the policy file"
    with pytest.raises(DescriptorError, match=match):
        verify_policy_file(descriptor)


def test_a_built_in_policy_has_no_policy_file(tmp_path: Path) -> None:
    data = fixture_data()
    del data["policy"]
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    descriptor = load_descriptor(path)
    assert descriptor.policy is None
    validate_against(descriptor, robot_description("testbot"))
    message = (
        "^the policy descriptor for robot 'testbot' names no policy file: it"
        " describes a policy built into the platform$"
    )
    with pytest.raises(DescriptorError, match=message):
        descriptor.policy_path()
    with pytest.raises(DescriptorError, match=message):
        verify_policy_file(descriptor)


# Resolving joints.


def test_resolve_joints_expands_a_group(testbot: RobotDescription) -> None:
    assert resolve_joints("left_leg", testbot) == LEFT_LEG


def test_resolve_joints_keeps_a_list_of_names(testbot: RobotDescription) -> None:
    names = ("right_knee", "left_knee")
    assert resolve_joints(names, testbot) == names


def test_resolve_joints_rejects_an_unknown_group(testbot: RobotDescription) -> None:
    with pytest.raises(ValueError, match="^no joint group 'arms' in robot 'testbot'$"):
        resolve_joints("arms", testbot)


def test_resolve_joints_says_a_single_joint_is_written_as_a_list(
    testbot: RobotDescription,
) -> None:
    match = (
        r"^no joint group 'left_knee' in robot 'testbot'; to select the joint,"
        r" write it as a list: \[left_knee\]$"
    )
    with pytest.raises(ValueError, match=match):
        resolve_joints("left_knee", testbot)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("left_knee", "tail"), "no joint 'tail' in robot 'testbot'"),
        (("wing", "left_knee", "tail"), "no joints 'wing', 'tail' in robot 'testbot'"),
    ],
)
def test_resolve_joints_names_every_unknown_joint(
    testbot: RobotDescription, names: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=f"^{message}$"):
        resolve_joints(names, testbot)


# Validation that needs no robot description.


def test_rejects_action_scale_tuple_of_the_wrong_length() -> None:
    data = fixture_data()
    data["action"]["joints"] = list(TESTBOT_JOINTS)
    data["action"]["scale"] = [0.25] * 5
    with pytest.raises(ValidationError, match=r"scale has 5 elements for 6 joints"):
        PolicyDescriptor.model_validate(data)


def test_accepts_action_scale_tuple_with_one_factor_per_joint() -> None:
    data = fixture_data()
    data["action"]["joints"] = list(TESTBOT_JOINTS)
    data["action"]["scale"] = [0.25, 0.5, 0.25, 0.25, 0.5, 0.25]
    descriptor = PolicyDescriptor.model_validate(data)
    assert descriptor.action.scale == (0.25, 0.5, 0.25, 0.25, 0.5, 0.25)


def test_rejects_observation_scale_tuple_of_the_wrong_length() -> None:
    data = fixture_data()
    data["observations"][4]["joints"] = ["left_knee"]
    data["observations"][4]["scale"] = [0.05, 0.05]
    match = r"observations\.4\s+Value error, scale has 2 elements for 1 joint"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize("component", ["lin_vel_x", "lin_vel_y", "ang_vel_z"])
def test_rejects_command_range_in_the_wrong_order(component: str) -> None:
    data = fixture_data()
    data["command"]["ranges"][component] = [0.5, -0.5]
    match = rf"{component}\s+Value error, lower bound 0.5 must not exceed upper bound"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize("field", ["policy.sha256", "robot.description_hash"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("sha256:" + "A" * 64, id="uppercase"),
        pytest.param("a" * 64, id="no prefix"),
        pytest.param("sha256:" + "a" * 63, id="too short"),
        pytest.param("md5:" + "a" * 32, id="other algorithm"),
        pytest.param("sha256:" + "a" * 64 + "\n", id="trailing newline"),
    ],
)
def test_rejects_malformed_hash(field: str, value: str) -> None:
    data = fixture_data()
    section, key = field.split(".")
    data[section][key] = value
    match = rf"{field}\s+Value error, expected 'sha256:' followed by 64 lowercase"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize("name", ["BaseAngularVelocity", "joint-velocity", "1st", ""])
def test_rejects_observation_term_name_that_is_not_snake_case(name: str) -> None:
    data = fixture_data()
    data["observations"][0]["term"] = name
    match = f"observation term name {name!r} is not snake_case"
    with pytest.raises(ValidationError, match=re.escape(match)):
        PolicyDescriptor.model_validate(data)


def test_rejects_unknown_schema_version() -> None:
    data = fixture_data()
    data["schema_version"] = 2
    with pytest.raises(ValidationError, match=r"schema_version\s+Input should be 1"):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [("rate_hz", 0.0), ("rate_hz", -50.0), ("observation_size", 0)],
)
def test_rejects_value_that_is_not_positive(field: str, value: float) -> None:
    data = fixture_data()
    data[field] = value
    with pytest.raises(ValidationError, match=rf"{field}\s+Input should be greater"):
        PolicyDescriptor.model_validate(data)


def test_rejects_descriptor_without_observations() -> None:
    data = fixture_data()
    data["observations"] = []
    with pytest.raises(ValidationError, match="at least 1 item"):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize(
    ("history", "match"),
    [
        ({"length": 0}, r"history\.length\s+Input should be greater than or equal"),
        ({"order": "newest_first"}, r"history\.order\s+Input should be 'oldest_first'"),
        ({"layout": "interleaved"}, r"history\.layout\s+Input should be 'per_term'"),
    ],
)
def test_rejects_invalid_history(history: dict[str, Any], match: str) -> None:
    data = fixture_data()
    data["history"] = history
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


def test_accepts_history_in_either_layout() -> None:
    data = fixture_data()
    data["history"] = {"length": 5, "layout": "per_frame"}
    history = PolicyDescriptor.model_validate(data).history
    assert (history.length, history.layout) == (5, "per_frame")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (("policy", "format"), "torchscript"),
        (("command", "type"), "position"),
        (("action", "type"), "torque"),
    ],
)
def test_rejects_unknown_kind(field: tuple[str, str], value: str) -> None:
    data = fixture_data()
    section, key = field
    data[section][key] = value
    with pytest.raises(ValidationError, match=rf"{section}\.{key}\s+Input should be"):
        PolicyDescriptor.model_validate(data)


def test_requires_gains() -> None:
    data = fixture_data()
    del data["gains"]
    with pytest.raises(ValidationError, match=r"gains\s+Field required"):
        PolicyDescriptor.model_validate(data)


def test_rejects_gain_values_when_gains_come_from_the_description() -> None:
    data = fixture_data()
    data["gains"] = explicit_gains(*TESTBOT_JOINTS) | {"source": "description"}
    match = r"gains\s+Value error, values must be empty when source is 'description'"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


def test_rejects_explicit_gains_without_values() -> None:
    data = fixture_data()
    data["gains"] = {"source": "explicit"}
    match = r"gains\s+Value error, values must give gains when source is 'explicit'"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


def test_rejects_explicit_gains_missing_for_a_listed_action_joint() -> None:
    data = fixture_data()
    data["action"]["joints"] = list(TESTBOT_JOINTS)
    data["gains"] = explicit_gains(*TESTBOT_JOINTS[:4])
    match = "gains.values has no gains for action joints 'right_knee', 'right_ankle_"
    with pytest.raises(ValidationError, match=re.escape(match)):
        PolicyDescriptor.model_validate(data)


def test_rejects_explicit_gains_for_a_joint_the_listed_action_does_not_drive() -> None:
    data = fixture_data()
    data["action"]["joints"] = list(LEFT_LEG)
    data["gains"] = explicit_gains(*LEFT_LEG, "right_knee")
    match = "gains.values names joint 'right_knee', which the action does not drive"
    with pytest.raises(ValidationError, match=re.escape(match)):
        PolicyDescriptor.model_validate(data)


def test_accepts_explicit_gains_for_every_listed_action_joint() -> None:
    data = fixture_data()
    data["action"]["joints"] = list(TESTBOT_JOINTS)
    data["gains"] = explicit_gains(*TESTBOT_JOINTS)
    descriptor = PolicyDescriptor.model_validate(data)
    assert descriptor.gains.values["right_knee"].kd == 1.0


@pytest.mark.parametrize("gain", ["kp", "kd"])
def test_rejects_negative_gain(gain: str) -> None:
    data = fixture_data()
    data["action"]["joints"] = ["left_knee"]
    data["action"]["scale"] = [0.25]
    data["gains"] = {"source": "explicit", "values": {"left_knee": {"kp": 1, "kd": 1}}}
    data["gains"]["values"]["left_knee"][gain] = -1.0
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        PolicyDescriptor.model_validate(data)


def test_accepts_term_parameters_of_each_kind() -> None:
    data = fixture_data()
    params = {
        "period": 0.8,
        "count": 3,
        "frame": "base",
        "wrap": True,
        "offsets": [0, 0.5],
    }
    data["observations"][0]["params"] = params
    term = PolicyDescriptor.model_validate(data).observations[0]
    assert term.params == params | {"offsets": (0.0, 0.5)}
    assert isinstance(term.params["count"], int)
    assert isinstance(term.params["wrap"], bool)


def test_rejects_term_parameter_that_is_a_mapping() -> None:
    data = fixture_data()
    data["observations"][0]["params"] = {"gait": {"period": 0.8}}
    with pytest.raises(ValidationError, match=r"observations\.0\.params\.gait"):
        PolicyDescriptor.model_validate(data)


def test_accepts_provenance_as_a_mapping() -> None:
    data = fixture_data()
    data["provenance"] = {"commit": "0123abc", "seed": 7}
    descriptor = PolicyDescriptor.model_validate(data)
    assert descriptor.provenance == {"commit": "0123abc", "seed": 7}


def test_rejects_other_provenance_text() -> None:
    data = fixture_data()
    data["provenance"] = "internal"
    with pytest.raises(ValidationError, match=r"provenance\.literal\['external'\]"):
        PolicyDescriptor.model_validate(data)


def test_rejects_non_finite_number() -> None:
    data = fixture_data()
    data["observations"][2]["scale"] = [2.0, float("inf"), 0.25]
    with pytest.raises(ValidationError, match="finite number"):
        PolicyDescriptor.model_validate(data)


@pytest.mark.parametrize(
    ("misspell", "location"),
    [
        (lambda data: data.update(observation_sise=27), "observation_sise"),
        (lambda data: data["history"].update(lenght=2), r"history\.lenght"),
        (lambda data: data["action"].update(scael=0.25), r"action\.scael"),
        (
            lambda data: data["observations"][0].update(name="x"),
            r"observations\.0\.name",
        ),
    ],
)
def test_rejects_misspelled_key(
    misspell: Callable[[dict[str, Any]], None], location: str
) -> None:
    data = fixture_data()
    data.setdefault("history", {})
    misspell(data)
    match = rf"{location}\s+Extra inputs are not permitted"
    with pytest.raises(ValidationError, match=match):
        PolicyDescriptor.model_validate(data)


def test_descriptors_are_immutable(descriptor: PolicyDescriptor) -> None:
    with pytest.raises(ValidationError, match="frozen"):
        descriptor.rate_hz = 100.0
    with pytest.raises(ValidationError, match="frozen"):
        descriptor.action.scale = 0.5
    with pytest.raises(ValidationError, match="frozen"):
        descriptor.observations[0].term = "other"


# Loading.


def test_load_error_for_missing_file_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.yaml"
    with pytest.raises(DescriptorError, match=starts_with(path) + "cannot read"):
        load_descriptor(path)


def test_load_error_for_invalid_yaml_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.yaml"
    path.write_text("observations: [base\n", encoding="utf-8")
    with pytest.raises(DescriptorError, match=starts_with(path) + "invalid YAML"):
        load_descriptor(path)


def test_load_error_for_non_mapping_starts_with_path(tmp_path: Path) -> None:
    path = tmp_path / "descriptor.yaml"
    path.write_text("", encoding="utf-8")
    match = starts_with(path) + "expected a mapping at the top level, got an empty"
    with pytest.raises(DescriptorError, match=match):
        load_descriptor(path)


def test_load_error_for_invalid_descriptor_keeps_field_details(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    data["rate_hz"] = -1.0
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match=starts_with(path)) as error:
        load_descriptor(path)
    assert "rate_hz" in str(error.value)
    assert isinstance(error.value.__cause__, ValidationError)


def test_dump_does_not_depend_on_file_location(tmp_path: Path) -> None:
    first = load_descriptor(TESTBOT_ZERO)
    second = load_descriptor(copy_fixture(tmp_path))
    assert first.policy_path() != second.policy_path()
    assert first.model_dump() == second.model_dump()


# Validation against the robot description.


def test_validate_against_lists_problems_under_one_heading(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["rate_hz"] = 60.0
    with pytest.raises(DescriptorError) as error:
        validate_against(PolicyDescriptor.model_validate(data), testbot)
    assert str(error.value).startswith(
        "policy descriptor for robot 'testbot' does not fit robot description"
        " 'testbot':\n- physics_hz 500.0 Hz is not a whole multiple of rate_hz"
    )


def test_reports_another_robot_s_name(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["robot"]["name"] = "g1_29dof"
    assert problems(data, testbot) == [
        "robot.name is 'g1_29dof', but the description is of robot 'testbot'"
    ]


def test_reports_another_description_hash(testbot: RobotDescription) -> None:
    data = fixture_data()
    recorded = "sha256:" + "0" * 64
    data["robot"]["description_hash"] = recorded
    assert problems(data, testbot) == [
        "the policy was trained for a different description of the robot:"
        f" robot.description_hash is {recorded}, but this description's hash is"
        f" {content_hash(testbot)}"
    ]


def test_reports_a_changed_description(testbot: RobotDescription) -> None:
    changed = changed_testbot(lambda data: data["joints"][1]["gains"].update(kp=81.0))
    (problem,) = problems(fixture_data(), changed)
    assert problem.startswith("the policy was trained for a different description")
    assert content_hash(testbot) in problem
    assert content_hash(changed) in problem


def test_reports_a_rate_above_the_robot_s_control_rate() -> None:
    slow = changed_testbot(lambda data: data["rates"].update(control_hz_max=25.0))
    data = fixture_data()
    data["robot"]["description_hash"] = content_hash(slow)
    assert problems(data, slow) == [
        "rate_hz 50.0 Hz is above the robot's control_hz_max 25.0 Hz"
    ]


@pytest.mark.parametrize("rate_hz", [60.0, 1000.0 / 3])
def test_reports_a_rate_that_is_not_a_whole_number_of_physics_steps(
    testbot: RobotDescription, rate_hz: float
) -> None:
    data = fixture_data()
    data["rate_hz"] = rate_hz
    assert problems(data, testbot) == [
        f"physics_hz 500.0 Hz is not a whole multiple of rate_hz {rate_hz} Hz:"
        " adapters step whole physics steps, so each policy step must last a whole"
        " number of them"
    ]


@pytest.mark.parametrize("rate_hz", [500.0, 250.0, 100.0, 50.0, 1.0])
def test_accepts_a_rate_that_is_a_whole_number_of_physics_steps(
    testbot: RobotDescription, rate_hz: float
) -> None:
    data = fixture_data()
    data["rate_hz"] = rate_hz
    assert problems(data, testbot) == []


def test_reports_an_unknown_group(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["action"]["joints"] = "all_joints"
    assert problems(data, testbot) == [
        "action.joints: no joint group 'all_joints' in robot 'testbot'"
    ]


def test_reports_unknown_joint_names(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = [*RIGHT_LEG, "tail"]
    assert problems(data, testbot) == [
        "held_joints: no joint 'tail' in robot 'testbot'"
    ]


def test_reports_an_unknown_group_in_an_observation_term(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["observations"][3]["joints"] = "arms"
    assert problems(data, testbot) == [
        "observations[3] (joint_position_relative).joints: no joint group 'arms' in"
        " robot 'testbot'"
    ]


def test_accepts_driven_and_held_joints_that_cover_the_robot(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = list(RIGHT_LEG)
    assert problems(data, testbot) == []


def test_reports_joints_both_driven_and_held(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["held_joints"] = ["right_knee", "left_knee"]
    assert problems(data, testbot) == [
        "action.joints and held_joints both name joints 'left_knee', 'right_knee'"
    ]


@pytest.mark.parametrize("field", ["action", "held"])
def test_reports_a_joint_named_twice(testbot: RobotDescription, field: str) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = list(RIGHT_LEG)
    if field == "action":
        data["action"]["joints"] = [*LEFT_LEG, "left_knee"]
        expected = "action.joints names joint 'left_knee' more than once"
    else:
        data["held_joints"] = [*RIGHT_LEG, "right_knee"]
        expected = "held_joints names joint 'right_knee' more than once"
    assert problems(data, testbot) == [expected]


def test_reports_every_joint_left_uncommanded(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = ["right_knee"]
    assert problems(data, testbot) == [
        "action.joints and held_joints together leave out joints 'right_hip_pitch',"
        " 'right_ankle_pitch': every joint must be driven by the action or held, so"
        " that every joint is commanded on every tick"
    ]


def test_reports_an_action_scale_tuple_of_the_wrong_length(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["action"]["scale"] = [0.25, 0.25, 0.25]
    assert problems(data, testbot) == ["action.scale has 3 elements for 6 joints"]


def test_reports_an_observation_scale_tuple_of_the_wrong_length(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["observations"][4]["scale"] = [0.05]
    assert problems(data, testbot) == [
        "observations[4] (joint_velocity).scale has 1 element for 6 joints"
    ]


def test_accepts_scale_tuples_with_one_factor_per_joint(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["action"]["scale"] = [0.25, 0.5, 0.25, 0.25, 0.5, 0.25]
    data["observations"][4]["scale"] = [0.05] * 6
    assert problems(data, testbot) == []


def test_accepts_explicit_gains_for_every_action_joint(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["gains"] = explicit_gains(*TESTBOT_JOINTS)
    assert problems(data, testbot) == []


def test_reports_explicit_gains_missing_for_an_action_joint(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["gains"] = explicit_gains(*TESTBOT_JOINTS[1:])
    assert problems(data, testbot) == [
        "gains.values has no gains for action joint 'left_hip_pitch'"
    ]


def test_reports_explicit_gains_for_an_unknown_joint(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["gains"] = explicit_gains(*TESTBOT_JOINTS, "tail")
    assert problems(data, testbot) == ["gains.values names unknown joint 'tail'"]


def test_reports_explicit_gains_for_a_held_joint(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = list(RIGHT_LEG)
    data["gains"] = explicit_gains(*LEFT_LEG, "right_knee")
    assert problems(data, testbot) == [
        "gains.values names joint 'right_knee', which the action does not drive:"
        " held joints use the description's gains"
    ]


def test_reports_every_problem_together(testbot: RobotDescription) -> None:
    data = fixture_data()
    data["robot"]["name"] = "other"
    data["rate_hz"] = 60.0
    data["observations"][3]["joints"] = "arms"
    data["action"]["joints"] = "left_leg"
    data["action"]["scale"] = [0.25, 0.25]
    data["held_joints"] = ["left_knee", "tail"]
    data["gains"] = explicit_gains(*LEFT_LEG[1:], "wing")
    assert problems(data, testbot) == [
        "robot.name is 'other', but the description is of robot 'testbot'",
        "physics_hz 500.0 Hz is not a whole multiple of rate_hz 60.0 Hz: adapters"
        " step whole physics steps, so each policy step must last a whole number of"
        " them",
        "observations[3] (joint_position_relative).joints: no joint group 'arms' in"
        " robot 'testbot'",
        "action.scale has 2 elements for 3 joints",
        "gains.values has no gains for action joint 'left_hip_pitch'",
        "gains.values names unknown joint 'wing'",
        "held_joints: no joint 'tail' in robot 'testbot'",
    ]


def test_reports_coverage_once_both_selections_resolve(
    testbot: RobotDescription,
) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = ["left_knee"]
    data["action"]["scale"] = [0.25]
    assert problems(data, testbot) == [
        "action.scale has 1 element for 3 joints",
        "action.joints and held_joints both name joint 'left_knee'",
        "action.joints and held_joints together leave out joints 'right_hip_pitch',"
        " 'right_knee', 'right_ankle_pitch': every joint must be driven by the"
        " action or held, so that every joint is commanded on every tick",
    ]
