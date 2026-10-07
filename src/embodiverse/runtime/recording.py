"""Recording runs: every tick's state, command and faults, in one MCAP file.

A failure on a robot is only debuggable if the run was recorded, and a
result is only useful if it can be traced to what produced it. A recording
holds both: a manifest of everything that went into the run, and, for every
tick on which the runner wrote a command, the state it read, the command it
wrote and the faults the safety layer reported.

The file is MCAP, the log format of ROS 2 and Foxglove, which needs no ROS
to read or write:

- the manifest is the metadata record `embodiverse.manifest`, its JSON text
  under the key `manifest`;
- each tick writes one message on each of three channels,
  `/embodiverse/state`, `/embodiverse/command` and `/embodiverse/faults`,
  with the state's stamp as the message's log and publish time and the
  tick's number, from 0, as its sequence;
- messages are JSON, each channel with a JSON Schema, so that Foxglove
  Studio can show and plot them. Arrays of joint values are lists in
  description joint order. A value that is not finite, which JSON has no
  number for, is the string `NaN`, `Infinity` or `-Infinity`, as in
  protobuf's JSON mapping.

`collect_manifest` fills in a run's manifest. `McapRecorder` writes a
recording; the runner calls its `record_tick` through the `TickRecorder`
protocol, so the runner does not depend on MCAP. `read_run` reads a
recording back. Writing and reading need the mcap package, the `record`
extra (`embodiverse[record]`); `collect_manifest` does not, and this module
imports without it.

docs/policies.md describes the format and how to read a recording.
"""

import json
import math
import os
import platform
import re
import socket
import struct
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import PolicyDescriptor
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.manifest import RunManifest
from embodiverse.contracts.robot import Robot
from embodiverse.contracts.types import Fault, JointCommand, Mode, RobotState

if TYPE_CHECKING:
    from mcap.reader import McapReader

__all__ = [
    "COMMAND_TOPIC",
    "FAULTS_TOPIC",
    "MANIFEST_KEY",
    "MANIFEST_METADATA",
    "STATE_TOPIC",
    "McapRecorder",
    "RecordedRun",
    "RecordingError",
    "collect_manifest",
    "read_run",
]

STATE_TOPIC = "/embodiverse/state"
"""The channel of the states the runner read, one per tick."""

COMMAND_TOPIC = "/embodiverse/command"
"""The channel of the joint commands the runner wrote, one per tick."""

FAULTS_TOPIC = "/embodiverse/faults"
"""The channel of the faults the safety layer reported, one message per tick."""

MANIFEST_METADATA = "embodiverse.manifest"
"""The name of the metadata record that holds the run's manifest."""

MANIFEST_KEY = "manifest"
"""The key of the manifest's JSON text in its metadata record."""

_TOPICS = (STATE_TOPIC, COMMAND_TOPIC, FAULTS_TOPIC)
_IMU_FIELDS = ("orientation", "angular_velocity", "linear_acceleration")
_COMMAND_FIELDS = ("position", "velocity", "effort", "kp", "kd")

_GIT_TIMEOUT_S = 5.0
"""How long each git command may take before the commit is taken as unknown."""

_GIT_SHA = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")

_CHUNK_SIZE = 64 * 1024
"""Bytes of messages collected before they are compressed and written.

Each chunk is compressed and written on the control thread, in the tick
that fills it, and a chunk not yet written is lost if the process dies
without closing the recording. 64 KiB holds about 70 ticks of testbot,
1.4 s at 50 Hz, and compresses in about 0.3 ms.
"""

_NON_FINITE = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


class RecordingError(ValueError):
    """A recording cannot be written or read, or is not a recording of a run."""


def collect_manifest(
    *,
    description: RobotDescription,
    descriptor: PolicyDescriptor,
    adapter: str | Robot,
    cwd: str | os.PathLike[str] | None = None,
) -> RunManifest:
    """Return the manifest of a run of the policy `descriptor` describes.

    `adapter` is the adapter's registered name, such as `mujoco`, or the
    robot itself, for an adapter not found through the registry: its class
    is then recorded as `module:QualifiedName`. The policy file's hash is
    the one the descriptor records, which `verify_policy_file` checks
    before a policy is loaded; it is None for a descriptor without a file.

    The commit is that of the git repository containing `cwd`, the current
    directory by default: `git rev-parse HEAD`, and whether
    `git status --porcelain` lists any change, untracked files included.
    Both are None if git is not installed, `cwd` is not in a repository,
    the repository has no commit, or git takes longer than a few seconds.
    `started_utc` is the time now.

    Raises PackageNotFoundError if Embodiverse is not installed.
    """
    git_sha, git_dirty = _git_state(Path.cwd() if cwd is None else Path(cwd))
    if isinstance(adapter, str):
        adapter_name = adapter
    else:
        kind = type(adapter)
        adapter_name = f"{kind.__module__}:{kind.__qualname__}"
    return RunManifest(
        schema_version=1,
        platform_version=version("embodiverse"),
        robot=description.name,
        description_hash=content_hash(description),
        adapter=adapter_name,
        policy_sha256=None if descriptor.policy is None else descriptor.policy.sha256,
        descriptor_hash=content_hash(descriptor),
        python_version=platform.python_version(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        started_utc=datetime.now(timezone.utc),
        host=socket.gethostname(),
    )


class McapRecorder:
    """Records a run to an MCAP file: its manifest, then every tick.

    Creating the recorder creates the file, writes the header and the
    manifest, and registers the three channels and their schemas.
    `record_tick` writes one tick's state, command and faults; `close`
    writes the summary and the indexes and closes the file. A recorder is a
    context manager that closes it on exit. The runner calls `record_tick`
    from the control thread; the caller creates and closes the recorder.

    A recording is never overwritten. Raises RecordingError, with the path
    at the start of its message, if `path` exists or cannot be created, and
    ImportError if the mcap package is not installed.
    """

    def __init__(self, path: str | os.PathLike[str], manifest: RunManifest) -> None:
        try:
            from mcap.well_known import MessageEncoding, SchemaEncoding
            from mcap.writer import CompressionType, Writer
        except ImportError as error:
            raise ImportError(
                "recording a run needs the mcap package: install embodiverse[record]"
            ) from error
        self._path = Path(path)
        try:
            stream = self._path.open("xb")
        except FileExistsError as error:
            raise RecordingError(
                f"{self._path}: the file exists, and a recording is never"
                " overwritten: choose another path"
            ) from error
        except OSError as error:
            reason = error.strerror or error
            raise RecordingError(
                f"{self._path}: cannot create the recording: {reason}"
            ) from error
        try:
            writer = Writer(
                stream, chunk_size=_CHUNK_SIZE, compression=CompressionType.ZSTD
            )
            writer.start()
            writer.add_metadata(
                MANIFEST_METADATA, {MANIFEST_KEY: manifest.model_dump_json()}
            )
            channels: list[int] = []
            for topic, schema in zip(_TOPICS, _SCHEMAS, strict=True):
                schema_id = writer.register_schema(
                    name=schema["title"],
                    encoding=SchemaEncoding.JSONSchema,
                    data=json.dumps(schema).encode("utf-8"),
                )
                channels.append(
                    writer.register_channel(
                        topic=topic,
                        message_encoding=MessageEncoding.JSON,
                        schema_id=schema_id,
                    )
                )
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        self._writer: Writer = writer
        self._state_channel, self._command_channel, self._faults_channel = channels
        self._ticks = 0
        self._closed = False

    @property
    def path(self) -> Path:
        """The file the run is recorded to."""
        return self._path

    @property
    def ticks(self) -> int:
        """The number of ticks recorded so far."""
        return self._ticks

    def record_tick(
        self, state: RobotState, command: JointCommand, faults: Sequence[Fault]
    ) -> None:
        """Record one tick: the state read, the command written, the faults found.

        The three messages are stamped with `state.stamp_ns`, and carry the
        tick's number as their sequence. Raises RuntimeError if the
        recording is closed, and OSError if the file cannot be written.
        """
        if self._closed:
            raise RuntimeError(f"{self._path}: the recording is closed")
        writer, stamp, tick = self._writer, state.stamp_ns, self._ticks
        for channel, message in (
            (self._state_channel, _state_message(state)),
            (self._command_channel, _command_message(command)),
            (self._faults_channel, {"faults": [_fault_message(f) for f in faults]}),
        ):
            writer.add_message(
                channel_id=channel,
                log_time=stamp,
                data=_encode(message),
                publish_time=stamp,
                sequence=tick,
            )
        self._ticks += 1

    def close(self) -> None:
        """Write the summary and the indexes, and close the file. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.finish()
        finally:
            self._stream.close()

    def __enter__(self) -> "McapRecorder":
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class RecordedRun:
    """A recording read back: its manifest, and what happened on each tick.

    The lists have one entry per tick, in order. Each state and command is
    its message as a dict, with every array of numbers as a float64 numpy
    array, non-finite values restored. A state's `mode` is a `Mode`, its
    `faults` a tuple of `Fault`, and its `temperature` None if the robot
    does not report temperatures.
    """

    manifest: RunManifest
    """The run's manifest."""

    states: list[dict[str, Any]]
    """The state read on each tick: `stamp_ns`, `mode`, `position`,
    `velocity`, `effort`, `temperature`, `imu` (a dict of `orientation`,
    `angular_velocity` and `linear_acceleration`) and `faults`."""

    commands: list[dict[str, NDArray[np.float64]]]
    """The command written on each tick: `position`, `velocity`, `effort`,
    `kp` and `kd`."""

    faults: list[tuple[Fault, ...]]
    """The faults the safety layer reported on each tick."""

    @property
    def ticks(self) -> int:
        """The number of ticks recorded."""
        return len(self.states)


def read_run(path: str | os.PathLike[str]) -> RecordedRun:
    """Read the recording at `path`.

    Reads the manifest and the messages of the three channels; messages on
    other channels are ignored. Raises RecordingError, with the path at the
    start of its message, if the file cannot be read, is not a valid MCAP
    file, or is not a complete recording of a run: no manifest, or one that
    is not valid, or channels that do not hold one message per tick. Raises
    ImportError if the mcap package is not installed.
    """
    try:
        from mcap.exceptions import McapError
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError(
            "reading a recording needs the mcap package: install embodiverse[record]"
        ) from error
    file = Path(path)
    try:
        with file.open("rb") as stream:
            reader = make_reader(stream)
            manifest = _read_manifest(reader)
            by_topic: dict[str, list[tuple[int, bytes]]] = {t: [] for t in _TOPICS}
            for _, channel, message in reader.iter_messages(
                topics=_TOPICS, log_time_order=False
            ):
                by_topic[channel.topic].append((message.sequence, message.data))
            ticks = _ticks(by_topic)
            states = [_decode_state(data) for data in ticks[STATE_TOPIC]]
            commands = [_decode_command(data) for data in ticks[COMMAND_TOPIC]]
            faults = [_decode_faults(data) for data in ticks[FAULTS_TOPIC]]
    except OSError as error:
        reason = error.strerror or error
        raise RecordingError(f"{file}: cannot read: {reason}") from error
    except RecordingError as error:
        raise RecordingError(f"{file}: {error}") from error
    except (McapError, ValueError, KeyError, TypeError, struct.error) as error:
        # mcap raises these, and struct.error, for a file that is not
        # MCAP or is cut short; json and the decoders for bad messages.
        raise RecordingError(
            f"{file}: not a valid recording of a run: {type(error).__name__}: {error}"
        ) from error
    return RecordedRun(
        manifest=manifest, states=states, commands=commands, faults=faults
    )


# --- The commit --------------------------------------------------------------------


def _git_state(cwd: Path) -> tuple[str | None, bool | None]:
    """Return the commit checked out in `cwd`'s repository, and whether it is dirty."""
    head = _git(cwd, "rev-parse", "HEAD")
    sha = head.decode("ascii", "replace").strip() if head is not None else ""
    if _GIT_SHA.fullmatch(sha) is None:
        return None, None
    # --no-optional-locks: status must not write the index, in case another
    # git command is running in the repository.
    status = _git(
        cwd, "--no-optional-locks", "status", "--porcelain", "--untracked-files=normal"
    )
    return sha, None if status is None else bool(status.strip())


def _git(cwd: Path, *args: str) -> bytes | None:
    """Return what `git <args>` prints in `cwd`, or None if it fails."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # No git, no such directory, or git took too long.
        return None
    return result.stdout if result.returncode == 0 else None


# --- Messages ----------------------------------------------------------------------


def _encode(message: dict[str, Any]) -> bytes:
    return json.dumps(message, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _values(array: NDArray[np.float64]) -> Sequence[float | str]:
    """Return `array` as a JSON list, with non-finite values as strings."""
    values: list[float] = array.tolist()
    if np.isfinite(array).all():
        return values
    return [_finite_or_name(value) for value in values]


def _finite_or_name(value: float) -> float | str:
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _fault_message(fault: Fault) -> dict[str, str]:
    message = {"code": fault.code, "message": fault.message}
    if fault.joint is not None:
        message["joint"] = fault.joint
    return message


def _state_message(state: RobotState) -> dict[str, Any]:
    message: dict[str, Any] = {
        "stamp_ns": state.stamp_ns,
        "mode": state.mode.value,
        "position": _values(state.position),
        "velocity": _values(state.velocity),
        "effort": _values(state.effort),
    }
    if state.temperature is not None:
        message["temperature"] = _values(state.temperature)
    imu = state.imu
    message["imu"] = {
        "orientation": _values(imu.orientation),
        "angular_velocity": _values(imu.angular_velocity),
        "linear_acceleration": _values(imu.linear_acceleration),
    }
    message["faults"] = [_fault_message(fault) for fault in state.faults]
    return message


def _command_message(command: JointCommand) -> dict[str, Any]:
    return {
        "position": _values(command.position),
        "velocity": _values(command.velocity),
        "effort": _values(command.effort),
        "kp": _values(command.kp),
        "kd": _values(command.kd),
    }


def _read_manifest(reader: "McapReader") -> RunManifest:
    records = [m for m in reader.iter_metadata() if m.name == MANIFEST_METADATA]
    if len(records) != 1:
        raise RecordingError(
            f"expected one {MANIFEST_METADATA!r} metadata record, found {len(records)}:"
            " the file is not a recording of a run"
        )
    if MANIFEST_KEY not in records[0].metadata:
        raise RecordingError(
            f"the {MANIFEST_METADATA!r} metadata record has no {MANIFEST_KEY!r} key"
        )
    try:
        return RunManifest.model_validate_json(records[0].metadata[MANIFEST_KEY])
    except ValueError as error:
        raise RecordingError(f"the manifest is not valid: {error}") from error


def _ticks(by_topic: dict[str, list[tuple[int, bytes]]]) -> dict[str, list[bytes]]:
    """Check that each channel holds one message per tick, in order; return them."""
    counts = {topic: len(messages) for topic, messages in by_topic.items()}
    if len(set(counts.values())) != 1:
        found = ", ".join(f"{count} on {topic}" for topic, count in counts.items())
        raise RecordingError(f"expected one message per tick on each channel: {found}")
    ticks: dict[str, list[bytes]] = {}
    for topic, messages in by_topic.items():
        messages.sort(key=lambda item: item[0])
        if [sequence for sequence, _ in messages] != list(range(len(messages))):
            raise RecordingError(
                f"the messages on {topic} are not numbered by tick from 0"
            )
        ticks[topic] = [data for _, data in messages]
    return ticks


def _array(values: Sequence[float | str]) -> NDArray[np.float64]:
    if any(isinstance(value, str) for value in values):
        return np.array([_number(value) for value in values], dtype=np.float64)
    return np.array(values, dtype=np.float64)


def _number(value: float | str) -> float:
    if isinstance(value, str):
        if value not in _NON_FINITE:
            raise RecordingError(f"{value!r} is not a number")
        return _NON_FINITE[value]
    return value


def _decode_fault(data: dict[str, str]) -> Fault:
    return Fault(code=data["code"], message=data["message"], joint=data.get("joint"))


def _decode_state(data: bytes) -> dict[str, Any]:
    message = json.loads(data)
    temperature = message.get("temperature")
    imu = message["imu"]
    return {
        "stamp_ns": int(message["stamp_ns"]),
        "mode": Mode(message["mode"]),
        "position": _array(message["position"]),
        "velocity": _array(message["velocity"]),
        "effort": _array(message["effort"]),
        "temperature": None if temperature is None else _array(temperature),
        "imu": {key: _array(imu[key]) for key in _IMU_FIELDS},
        "faults": tuple(_decode_fault(fault) for fault in message["faults"]),
    }


def _decode_command(data: bytes) -> dict[str, NDArray[np.float64]]:
    message = json.loads(data)
    return {key: _array(message[key]) for key in _COMMAND_FIELDS}


def _decode_faults(data: bytes) -> tuple[Fault, ...]:
    return tuple(_decode_fault(fault) for fault in json.loads(data)["faults"])


# --- Schemas -----------------------------------------------------------------------
#
# JSON Schema, draft 2020-12. Each field has one type, because Foxglove Studio
# reads only single types: no unions, no null. An optional field is left out
# of a message rather than null. Numbers are declared as numbers, although a
# value that is not finite is written as a string.


def _numbers(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "number"}, "description": description}


_JOINT_ORDER = ", one per joint, in the robot description's joint order."

_FAULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {
            "type": "string",
            "description": "The kind of fault, e.g. `state.stale`.",
        },
        "message": {"type": "string", "description": "An explanation for people."},
        "joint": {
            "type": "string",
            "description": "The joint concerned; left out if not about one joint.",
        },
    },
    "required": ["code", "message"],
}

_STATE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "embodiverse.RobotState",
    "description": "The robot's state as the runner read it on one tick"
    " (docs/architecture.md). A value that is not finite is the string `NaN`,"
    " `Infinity` or `-Infinity`.",
    "type": "object",
    "properties": {
        "stamp_ns": {
            "type": "integer",
            "minimum": 0,
            "description": "When the state was measured, in ns, on the monotonic"
            " clock, or in simulated time.",
        },
        "mode": {
            "type": "string",
            "enum": [mode.value for mode in Mode],
            "description": "The mode the adapter was in.",
        },
        "position": _numbers("Joint positions, in rad" + _JOINT_ORDER),
        "velocity": _numbers("Joint velocities, in rad/s" + _JOINT_ORDER),
        "effort": _numbers("Joint torques, in N·m" + _JOINT_ORDER),
        "temperature": _numbers(
            "Motor temperatures, in °C" + _JOINT_ORDER[:-1] + "; left out if the"
            " robot does not report them."
        ),
        "imu": {
            "type": "object",
            "properties": {
                "orientation": _numbers(
                    "The IMU frame's orientation in a world-fixed frame with z up,"
                    " as an (x, y, z, w) quaternion."
                ),
                "angular_velocity": _numbers(
                    "Angular velocity, in rad/s, in the IMU frame."
                ),
                "linear_acceleration": _numbers(
                    "Specific force, in m/s², in the IMU frame."
                ),
            },
            "required": list(_IMU_FIELDS),
        },
        "faults": {
            "type": "array",
            "items": _FAULT_SCHEMA,
            "description": "Faults the adapter reported in the state.",
        },
    },
    "required": ["stamp_ns", "mode", "position", "velocity", "effort", "imu", "faults"],
}

_COMMAND_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "embodiverse.JointCommand",
    "description": "The joint command the runner wrote on one tick, after the safety"
    " layer: tau = kp * (position - q) + kd * (velocity - dq) + effort.",
    "type": "object",
    "properties": {
        "position": _numbers("Target joint positions, in rad" + _JOINT_ORDER),
        "velocity": _numbers("Target joint velocities, in rad/s" + _JOINT_ORDER),
        "effort": _numbers("Feed-forward joint torques, in N·m" + _JOINT_ORDER),
        "kp": _numbers("Position stiffness, in N·m/rad" + _JOINT_ORDER),
        "kd": _numbers("Velocity damping, in N·m·s/rad" + _JOINT_ORDER),
    },
    "required": list(_COMMAND_FIELDS),
}

_FAULTS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "embodiverse.Faults",
    "description": "The faults the safety layer reported on one tick, by its check"
    " of the tick's inputs and its filter of the command; empty if none.",
    "type": "object",
    "properties": {"faults": {"type": "array", "items": _FAULT_SCHEMA}},
    "required": ["faults"],
}

_SCHEMAS = (_STATE_SCHEMA, _COMMAND_SCHEMA, _FAULTS_SCHEMA)
