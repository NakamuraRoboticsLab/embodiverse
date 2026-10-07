"""Tests of run recordings: the manifest, the MCAP recorder and the reader.

The tests that need git make their own repositories in a temporary
directory, with git's environment and configuration cleared, so they do
not depend on the checkout they run in, nor touch it. The runs are of
testbot in MuJoCo, in lockstep, so every tick's time is known exactly.
"""

import hashlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcap.reader import make_reader
from mcap.writer import Writer
from numpy.typing import NDArray

import embodiverse
from embodiverse.adapters.mujoco import MujocoRobot
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import load_descriptor
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.manifest import RunManifest
from embodiverse.contracts.registry import robot_description
from embodiverse.contracts.robot import Capabilities
from embodiverse.contracts.types import Fault, ImuState, JointCommand, Mode, RobotState
from embodiverse.runtime import recording
from embodiverse.runtime.policy import OnnxPolicy, ZeroPolicy
from embodiverse.runtime.recording import (
    COMMAND_TOPIC,
    FAULTS_TOPIC,
    MANIFEST_KEY,
    MANIFEST_METADATA,
    STATE_TOPIC,
    McapRecorder,
    RecordingError,
    collect_manifest,
    read_run,
)
from embodiverse.runtime.runner import (
    Runner,
    RunnerConfig,
    StartRefused,
    VelocityCommandSample,
    zero_policy_descriptor,
)

AffinePolicy = Callable[..., Path]
DescriptorWriter = Callable[[dict[str, Any], Path], Path]

TESTBOT = robot_description("testbot")
ZERO = zero_policy_descriptor(TESTBOT)
TESTBOT_ZERO = Path(__file__).parents[1] / "fixtures/policies/testbot_zero"
TOPICS = (STATE_TOPIC, COMMAND_TOPIC, FAULTS_TOPIC)
PERIOD_NS = 20_000_000  # the zero policy's 50 Hz
NO_RAMP = RunnerConfig(start_ramp_s=0.0)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


@pytest.fixture
def clean_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run git as on a machine with no git configuration, inside `tmp_path` only.

    Git hooks set variables such as GIT_DIR and GIT_INDEX_FILE, which would
    point git at the checkout the tests run in; they are removed. The
    user's and the system's configuration are ignored, and git does not
    look for a repository above `tmp_path`.
    """
    for name in list(os.environ):
        if name.startswith("GIT_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))


def git(directory: Path, *args: str) -> str:
    """Run git in `directory`, with an author for commits; return its output."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Embodiverse Tests",
            "-c",
            "user.email=tests@example.com",
            *args,
        ],
        cwd=directory,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout


def repository(tmp_path: Path) -> Path:
    """Make a git repository with one commit of one file and a .gitignore."""
    directory = tmp_path / "project"
    directory.mkdir()
    git(directory, "init", "--quiet")
    (directory / "policy.py").write_text("GAIN = 1.0\n", encoding="utf-8")
    (directory / ".gitignore").write_text("*.mcap\n", encoding="utf-8")
    git(directory, "add", "policy.py", ".gitignore")
    git(directory, "commit", "--quiet", "-m", "Add the policy")
    return directory


def manifest(**changes: Any) -> RunManifest:
    data = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco"
    ).model_dump()
    data.update(changes)
    return RunManifest.model_validate(data)


# Collecting the manifest.


def test_the_manifest_names_the_platform_robot_policy_and_machine(
    tmp_path: Path,
) -> None:
    before = datetime.now(timezone.utc)
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=tmp_path
    )
    after = datetime.now(timezone.utc)
    assert collected.schema_version == 1
    assert collected.platform_version == embodiverse.__version__
    assert collected.robot == "testbot"
    assert collected.description_hash == content_hash(TESTBOT)
    assert collected.adapter == "mujoco"
    assert collected.policy_sha256 is None  # the zero policy has no file
    assert collected.descriptor_hash == content_hash(ZERO)
    assert collected.python_version == platform.python_version()
    assert collected.host == socket.gethostname()
    assert before <= collected.started_utc <= after


def test_the_policy_hash_is_the_one_the_descriptor_records() -> None:
    descriptor = load_descriptor(TESTBOT_ZERO / "descriptor.yaml")
    assert descriptor.policy is not None
    collected = collect_manifest(
        description=TESTBOT, descriptor=descriptor, adapter="mujoco"
    )
    assert collected.policy_sha256 == descriptor.policy.sha256
    assert collected.descriptor_hash == content_hash(descriptor)


def test_an_adapter_given_as_a_robot_is_named_by_its_class() -> None:
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter=MujocoRobot(TESTBOT)
    )
    assert collected.adapter == f"{MujocoRobot.__module__}:MujocoRobot"


@needs_git
@pytest.mark.usefixtures("clean_git")
def test_a_clean_repository_gives_its_commit(tmp_path: Path) -> None:
    directory = repository(tmp_path)
    head = git(directory, "rev-parse", "HEAD").strip()
    (directory / "run.mcap").write_bytes(b"")  # ignored, so still clean
    (directory / "scripts").mkdir()  # empty, so not a change git sees
    for cwd in (directory, directory / "scripts"):
        collected = collect_manifest(
            description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=cwd
        )
        assert (collected.git_sha, collected.git_dirty) == (head, False)


@needs_git
@pytest.mark.usefixtures("clean_git")
def test_the_current_directory_is_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = repository(tmp_path)
    (directory / "scripts").mkdir()
    monkeypatch.chdir(directory / "scripts")
    collected = collect_manifest(description=TESTBOT, descriptor=ZERO, adapter="x")
    assert collected.git_sha == git(directory, "rev-parse", "HEAD").strip()


@needs_git
@pytest.mark.usefixtures("clean_git")
@pytest.mark.parametrize(
    ("file", "text"),
    [
        pytest.param("policy.py", "GAIN = 2.0\n", id="a changed file"),
        pytest.param("notes.py", "", id="an untracked file"),
    ],
)
def test_a_repository_with_changes_is_dirty(
    tmp_path: Path, file: str, text: str
) -> None:
    directory = repository(tmp_path)
    (directory / file).write_text(text, encoding="utf-8")
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=directory
    )
    assert collected.git_sha == git(directory, "rev-parse", "HEAD").strip()
    assert collected.git_dirty is True


@needs_git
@pytest.mark.usefixtures("clean_git")
def test_without_a_repository_the_commit_is_unknown(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=plain
    )
    assert (collected.git_sha, collected.git_dirty) == (None, None)


@needs_git
@pytest.mark.usefixtures("clean_git")
def test_a_repository_without_a_commit_gives_no_commit(tmp_path: Path) -> None:
    git(tmp_path, "init", "--quiet")
    (tmp_path / "policy.py").write_text("", encoding="utf-8")
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=tmp_path
    )
    assert (collected.git_sha, collected.git_dirty) == (None, None)


def test_without_git_the_commit_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))  # a directory without git
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=tmp_path
    )
    assert (collected.git_sha, collected.git_dirty) == (None, None)


def test_a_git_that_does_not_answer_is_given_up_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slow = tmp_path / "git"
    slow.write_text("#!/bin/sh\nsleep 10\n", encoding="utf-8")
    slow.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(recording, "_GIT_TIMEOUT_S", 0.2)
    collected = collect_manifest(
        description=TESTBOT, descriptor=ZERO, adapter="mujoco", cwd=tmp_path
    )
    assert (collected.git_sha, collected.git_dirty) == (None, None)


# Recording and reading back.


def array(values: list[float]) -> NDArray[np.float64]:
    return np.array(values, dtype=np.float64)


def state(
    stamp_ns: int,
    position: list[float],
    *,
    temperature: list[float] | None = None,
    faults: tuple[Fault, ...] = (),
) -> RobotState:
    return RobotState(
        stamp_ns=stamp_ns,
        position=array(position),
        velocity=array([0.5, -0.25, 1e-9]),
        effort=array([1.0, -2.0, 3.0]),
        temperature=None if temperature is None else array(temperature),
        imu=ImuState(
            orientation=array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity=array([0.01, -0.02, 0.03]),
            linear_acceleration=array([0.1, 0.2, 9.81]),
        ),
        mode=Mode.POLICY,
        faults=faults,
    )


def command(position: list[float]) -> JointCommand:
    return JointCommand(
        position=array(position),
        velocity=array([0.0, 0.0, 0.0]),
        effort=array([0.0, 0.5, -0.5]),
        kp=array([60.0, 80.0, 40.0]),
        kd=array([2.0, 2.0, 2.0]),
    )


def assert_state_equal(recorded: dict[str, Any], expected: RobotState) -> None:
    assert recorded["stamp_ns"] == expected.stamp_ns
    assert recorded["mode"] is expected.mode
    for field in ("position", "velocity", "effort"):
        np.testing.assert_array_equal(recorded[field], getattr(expected, field))
    if expected.temperature is None:
        assert recorded["temperature"] is None
    else:
        np.testing.assert_array_equal(recorded["temperature"], expected.temperature)
    for field in ("orientation", "angular_velocity", "linear_acceleration"):
        np.testing.assert_array_equal(
            recorded["imu"][field], getattr(expected.imu, field)
        )
    assert recorded["faults"] == expected.faults


def assert_command_equal(recorded: dict[str, Any], expected: JointCommand) -> None:
    assert set(recorded) == {"position", "velocity", "effort", "kp", "kd"}
    for field, values in recorded.items():
        np.testing.assert_array_equal(values, getattr(expected, field))
        assert values.dtype == np.float64


HOT = Fault(code="motor.hot", message="left_knee is at 75 °C", joint="left_knee")
STALE = Fault(code="command.stale", message="no velocity command for 0.6 s")


def test_a_recording_reads_back_exactly(tmp_path: Path) -> None:
    states = [
        state(0, [0.1, 0.2, 0.3]),
        state(20_000_000, [0.1, np.nan, -np.inf], temperature=[40.0, 41.5, np.inf]),
        state(40_000_000, [1 / 3, 2 / 3, 1e-300], faults=(STALE,)),
    ]
    commands = [command([0.1, 0.2, 0.3 + 1e-12]) for _ in states]
    faults: list[tuple[Fault, ...]] = [(), (HOT, STALE), (STALE,)]
    collected = manifest()
    path = tmp_path / "run.mcap"
    with McapRecorder(path, collected) as recorder:
        for tick in range(3):
            recorder.record_tick(states[tick], commands[tick], faults[tick])
        assert recorder.ticks == 3
        assert recorder.path == path
    run = read_run(path)
    assert run.manifest == collected
    assert run.ticks == 3
    for tick in range(3):
        assert_state_equal(run.states[tick], states[tick])
        assert_command_equal(run.commands[tick], commands[tick])
    assert run.faults == faults


def test_a_value_that_is_not_finite_is_written_as_a_string(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    with McapRecorder(path, manifest()) as recorder:
        recorder.record_tick(
            state(0, [np.nan, np.inf, -np.inf]), command([0.0] * 3), ()
        )
    messages = messages_of(path)
    assert messages[STATE_TOPIC][0]["position"] == ["NaN", "Infinity", "-Infinity"]
    assert messages[STATE_TOPIC][0]["velocity"] == [0.5, -0.25, 1e-9]


def messages_of(path: Path) -> dict[str, list[Any]]:
    """Return the decoded JSON of every message in `path`, by topic."""
    with path.open("rb") as stream:
        found: dict[str, list[Any]] = {topic: [] for topic in TOPICS}
        for _, channel, message in make_reader(stream).iter_messages():
            found[channel.topic].append(json.loads(message.data))
    return found


def test_the_file_is_mcap_with_a_schema_for_each_channel(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    collected = manifest()
    with McapRecorder(path, collected) as recorder:
        for tick in range(4):
            stamp = 1_000_000_000 + tick * 20_000_000
            recorder.record_tick(
                state(stamp, [0.0, 0.1, 0.2], temperature=[30.0] * 3, faults=(HOT,)),
                command([0.0] * 3),
                (HOT,) if tick % 2 else (),
            )
    with path.open("rb") as stream:
        reader = make_reader(stream, validate_crcs=True)
        assert reader.get_header().profile == ""
        summary = reader.get_summary()
        assert summary is not None and summary.statistics is not None
        channels = {channel.topic: channel for channel in summary.channels.values()}
        assert set(channels) == set(TOPICS)
        names = {}
        for topic, channel in channels.items():
            assert channel.message_encoding == "json"
            schema = summary.schemas[channel.schema_id]
            assert schema.encoding == "jsonschema"
            names[topic] = schema.name
            check_schema(json.loads(schema.data))
            assert summary.statistics.channel_message_counts[channel.id] == 4
        assert names == {
            STATE_TOPIC: "embodiverse.RobotState",
            COMMAND_TOPIC: "embodiverse.JointCommand",
            FAULTS_TOPIC: "embodiverse.Faults",
        }
        assert summary.statistics.message_count == 12
        (metadata,) = reader.iter_metadata()
        assert metadata.name == MANIFEST_METADATA
        assert set(metadata.metadata) == {MANIFEST_KEY}
        assert RunManifest.model_validate_json(metadata.metadata[MANIFEST_KEY]) == (
            collected
        )
        # Each tick's three messages carry its stamp and its number.
        for _, channel, message in reader.iter_messages():
            tick = message.sequence
            assert message.log_time == message.publish_time
            assert message.log_time == 1_000_000_000 + tick * 20_000_000
            data = json.loads(message.data)
            check_message(data, json.loads(summary.schemas[channel.schema_id].data))


def check_schema(schema: dict[str, Any], where: str = "") -> None:
    """Check that every field of `schema` has a single type, as Foxglove requires.

    Foxglove Studio reads a channel's JSON Schema to show its fields, and
    refuses unions, `null` and `oneOf` or `anyOf` alternatives.
    """
    assert schema["type"] == "object", where
    for name, field in schema["properties"].items():
        assert not {"oneOf", "anyOf", "allOf", "$ref"} & set(field), f"{where}.{name}"
        assert field["type"] in {"object", "array", "string", "number", "integer"}
        if field["type"] == "object":
            check_schema(field, f"{where}.{name}")
        if field["type"] == "array":
            items = field["items"]
            assert items["type"] in {"object", "number"}, f"{where}.{name}"
            if items["type"] == "object":
                check_schema(items, f"{where}.{name}[]")


def check_message(message: Any, schema: dict[str, Any]) -> None:
    """Check that `message` has the fields and the types of `schema`."""
    if schema["type"] == "object":
        assert isinstance(message, dict)
        assert set(schema["required"]) <= set(message) <= set(schema["properties"])
        for name, value in message.items():
            check_message(value, schema["properties"][name])
    elif schema["type"] == "array":
        assert isinstance(message, list)
        for item in message:
            check_message(item, schema["items"])
    elif schema["type"] == "number":
        assert isinstance(message, float)
    elif schema["type"] == "integer":
        assert isinstance(message, int)
    else:
        assert isinstance(message, str)
        assert message in schema.get("enum", [message])


def test_a_recording_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    path.write_bytes(b"an earlier run")
    message = f"^{path}: the file exists, and a recording is never overwritten"
    with pytest.raises(RecordingError, match=message):
        McapRecorder(path, manifest())
    assert path.read_bytes() == b"an earlier run"


def test_a_recording_that_cannot_be_created_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "run.mcap"
    message = f"^{path}: cannot create the recording: No such file or directory$"
    with pytest.raises(RecordingError, match=message):
        McapRecorder(path, manifest())


def test_a_closed_recording_records_nothing_more(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    recorder = McapRecorder(path, manifest())
    recorder.close()
    recorder.close()  # idempotent
    with pytest.raises(RuntimeError, match="the recording is closed"):
        recorder.record_tick(state(0, [0.0] * 3), command([0.0] * 3), ())
    assert read_run(path).ticks == 0


def test_recording_needs_the_mcap_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("mcap", "mcap.exceptions", "mcap.reader", "mcap.writer"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "mcap.well_known", None)
    with pytest.raises(ImportError, match=r"install embodiverse\[record\]"):
        McapRecorder(tmp_path / "run.mcap", manifest())
    with pytest.raises(ImportError, match=r"install embodiverse\[record\]"):
        read_run(tmp_path / "run.mcap")
    assert not (tmp_path / "run.mcap").exists()


def test_the_command_line_and_the_runtime_import_without_mcap() -> None:
    # In a new interpreter, because these tests import mcap.
    script = """\
import sys
import embodiverse.app.cli
import embodiverse.runtime.recording
print("mcap" in {name.partition(".")[0] for name in sys.modules})
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.splitlines()[-1] == "False"


# Reading what is not a recording.


def test_read_run_refuses_a_file_that_is_not_mcap(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    path.write_text("not a recording\n", encoding="utf-8")
    with pytest.raises(RecordingError, match=f"^{path}: not a valid recording"):
        read_run(path)


def test_read_run_refuses_a_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    message = f"^{path}: cannot read: No such file or directory$"
    with pytest.raises(RecordingError, match=message):
        read_run(path)


def write_mcap(path: Path, metadata: dict[str, str] | None) -> None:
    """Write an MCAP file with one state and no command, and `metadata`."""
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start()
        if metadata is not None:
            writer.add_metadata(MANIFEST_METADATA, metadata)
        channel = writer.register_channel(STATE_TOPIC, "json", 0)
        writer.add_message(channel, 0, b"{}", 0)
        writer.finish()


@pytest.mark.parametrize(
    ("metadata", "problem"),
    [
        pytest.param(
            None,
            "expected one 'embodiverse.manifest' metadata record, found 0",
            id="no manifest",
        ),
        pytest.param(
            {"run": "{}"},
            "the 'embodiverse.manifest' metadata record has no 'manifest' key",
            id="no manifest key",
        ),
        pytest.param(
            {MANIFEST_KEY: '{"schema_version": 1}'},
            "the manifest is not valid: ",
            id="invalid manifest",
        ),
        pytest.param(
            {MANIFEST_KEY: manifest().model_dump_json()},
            "expected one message per tick on each channel: 1 on /embodiverse/state,"
            " 0 on /embodiverse/command, 0 on /embodiverse/faults",
            id="a state without its command",
        ),
    ],
)
def test_read_run_refuses_an_mcap_file_that_is_not_a_recording_of_a_run(
    tmp_path: Path, metadata: dict[str, str] | None, problem: str
) -> None:
    path = tmp_path / "run.mcap"
    write_mcap(path, metadata)
    with pytest.raises(RecordingError, match=f"^{path}: ") as error:
        read_run(path)
    assert problem in str(error.value)


# Recording a run.


@pytest.fixture
def testbot() -> Iterator[MujocoRobot]:
    """testbot in lockstep, connected, standing in its default pose."""
    robot = MujocoRobot(TESTBOT)
    robot.connect()
    yield robot
    robot.close()


class WatchedRobot:
    """A robot in lockstep that keeps every state it returns and command it gets.

    It passes every call on to `robot`. After its first `healthy_reads`
    reads, it adds `fault` to every state it returns, as an adapter that
    detects a problem reports it.
    """

    def __init__(
        self,
        robot: MujocoRobot,
        fault: Fault | None = None,
        healthy_reads: int = 0,
    ) -> None:
        self.robot = robot
        self.fault = fault
        self.healthy_reads = healthy_reads
        self.reads = 0
        self.states: dict[int, RobotState] = {}
        self.commands: list[JointCommand] = []

    @property
    def description(self) -> RobotDescription:
        return self.robot.description

    def capabilities(self) -> Capabilities:
        return self.robot.capabilities()

    def connect(self) -> None:
        self.robot.connect()

    def read(self) -> RobotState:
        state = self.robot.read()
        self.reads += 1
        if self.fault is not None and self.reads > self.healthy_reads:
            state = replace(state, faults=(*state.faults, self.fault))
        self.states[state.stamp_ns] = state
        return state

    def write(self, command: JointCommand) -> None:
        self.commands.append(command)
        self.robot.write(command)

    def set_mode(self, mode: Mode) -> None:
        self.robot.set_mode(mode)

    def operator_stop(self) -> None:
        self.robot.operator_stop()

    def close(self) -> None:
        self.robot.close()

    def advance(self, seconds: float) -> None:
        self.robot.advance(seconds)


def codes(faults: Sequence[tuple[Fault, ...]]) -> list[list[str]]:
    return [[fault.code for fault in tick] for tick in faults]


def test_a_run_is_recorded_tick_by_tick(testbot: MujocoRobot, tmp_path: Path) -> None:
    watched = WatchedRobot(testbot)
    collected = collect_manifest(description=TESTBOT, descriptor=ZERO, adapter=testbot)
    path = tmp_path / "run.mcap"
    with McapRecorder(path, collected) as recorder:
        runner = Runner(watched, ZeroPolicy(3, 6), ZERO, recorder=recorder)
        stats = runner.run(1.0)
        runner.stop()
    run = read_run(path)
    assert run.manifest == collected
    # Every tick that wrote a command: the 2 s start ramp, then 1 s of policy.
    assert stats.ticks == 50
    assert run.ticks == 100 + 50 == len(watched.commands)
    stamps = [state["stamp_ns"] for state in run.states]
    assert stamps == [tick * PERIOD_NS for tick in range(run.ticks)]
    for recorded, sent in zip(run.commands, watched.commands, strict=True):
        assert_command_equal(recorded, sent)
    for recorded in run.states:
        assert_state_equal(recorded, watched.states[recorded["stamp_ns"]])
    assert run.faults == [()] * run.ticks
    with path.open("rb") as stream:
        summary = make_reader(stream, validate_crcs=True).get_summary()
    assert summary is not None and summary.statistics is not None
    assert set(summary.statistics.channel_message_counts.values()) == {150}


class FrozenCommand:
    """A command source that stops updating its command at `until_ns`."""

    def __init__(self, until_ns: int) -> None:
        self.until_ns = until_ns

    def read(self, now_ns: int) -> VelocityCommandSample:
        return VelocityCommandSample(
            command=np.zeros(3), stamp_ns=min(now_ns, self.until_ns)
        )


def test_faults_are_recorded_on_the_ticks_they_occur(
    testbot: MujocoRobot, tmp_path: Path
) -> None:
    path = tmp_path / "run.mcap"
    with McapRecorder(path, manifest()) as recorder:
        runner = Runner(
            testbot,
            ZeroPolicy(3, 6),
            ZERO,
            commands=FrozenCommand(500_000_000),
            config=NO_RAMP,
            recorder=recorder,
        )
        stats = runner.run(1.5)
    # The command is stale once more than 0.5 s old: from the tick at 1.02 s,
    # the 52nd of 75, on.
    assert dict(stats.faults) == {"command.stale": 24}
    assert codes(read_run(path).faults) == [[]] * 51 + [["command.stale"]] * 24


def test_a_held_tick_records_the_hold_command(
    testbot: MujocoRobot,
    tmp_path: Path,
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
) -> None:
    file = affine_policy(np.zeros((3, 6)), [math.nan] * 6)
    nan = load_descriptor(policy_descriptor(ZERO.model_dump(mode="json"), file))
    watched = WatchedRobot(testbot)
    collected = collect_manifest(description=TESTBOT, descriptor=nan, adapter="mujoco")
    assert collected.policy_sha256 == (
        f"sha256:{hashlib.sha256(file.read_bytes()).hexdigest()}"
    )
    path = tmp_path / "run.mcap"
    with McapRecorder(path, collected) as recorder:
        policy = OnnxPolicy.from_descriptor(nan, TESTBOT)
        stats = Runner(watched, policy, nan, config=NO_RAMP, recorder=recorder).run(0.1)
    assert stats.latched
    run = read_run(path)
    # The policy's NaN is never written: the hold command is, from then on.
    assert codes(run.faults) == [["policy.non_finite"]] + [[]] * 4
    for recorded, sent in zip(run.commands, watched.commands, strict=True):
        assert_command_equal(recorded, sent)
        assert np.isfinite(recorded["position"]).all()


def test_a_start_refused_during_the_ramp_is_recorded_up_to_the_refusal(
    testbot: MujocoRobot, tmp_path: Path
) -> None:
    error = Fault(code="imu.error", message="the IMU reports an error")
    # The start's check and ten ramp ticks are healthy; the eleventh is not.
    watched = WatchedRobot(testbot, fault=error, healthy_reads=11)
    path = tmp_path / "run.mcap"
    with McapRecorder(path, manifest()) as recorder:
        runner = Runner(watched, ZeroPolicy(3, 6), ZERO, recorder=recorder)
        with pytest.raises(StartRefused, match="a fault held the robot during the"):
            runner.start()
    assert testbot.read().mode is Mode.HOLD
    run = read_run(path)
    assert codes(run.faults) == [[]] * 10 + [["imu.error"]]
    assert run.states[-1]["faults"] == (error,)
    for recorded, sent in zip(run.commands, watched.commands, strict=True):
        assert_command_equal(recorded, sent)


class FullDisk:
    """A recorder that fails on its third tick, as on a full disk."""

    def __init__(self) -> None:
        self.ticks = 0

    def record_tick(
        self, state: RobotState, command: JointCommand, faults: tuple[Fault, ...]
    ) -> None:
        self.ticks += 1
        if self.ticks == 3:
            raise OSError(28, "No space left on device")


def test_a_recorder_that_fails_ends_the_run_and_leaves_the_robot_held(
    testbot: MujocoRobot,
) -> None:
    runner = Runner(
        testbot, ZeroPolicy(3, 6), ZERO, config=NO_RAMP, recorder=FullDisk()
    )
    with pytest.raises(OSError, match="No space left on device"):
        runner.run(1.0)
    assert testbot.read().mode is Mode.HOLD
    assert not runner.started
