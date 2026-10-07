"""Tests of the `embodiverse` command line."""

import hashlib
import re
import shutil
import subprocess
import sys
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
import yaml

from embodiverse import __version__
from embodiverse.adapters.mujoco import MujocoRobot
from embodiverse.app.cli import main
from embodiverse.contracts import registry
from embodiverse.contracts.descriptor import load_descriptor
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.registry import (
    ADAPTERS,
    GROUPS,
    ROBOT_DESCRIPTIONS,
    robot_description,
)
from embodiverse.contracts.types import Fault, Mode, RobotState
from embodiverse.runtime.recording import read_run
from embodiverse.runtime.runner import zero_policy_descriptor
from embodiverse.runtime.safety import SafetyLayer
from embodiverse.tools.generators import (
    begin_marker,
    end_marker,
    render_block,
    robot_joint_table,
)

ROOT = Path(__file__).parents[2]
TESTBOT_ZERO = Path(__file__).parents[1] / "fixtures/policies/testbot_zero"
RUN_ZERO = ["run", "--robot", "testbot", "--adapter", "mujoco", "--policy", "zero"]


def install(monkeypatch: pytest.MonkeyPatch, *installed: EntryPoint) -> None:
    """Make `installed` the only installed entry points."""

    def select(*, group: str) -> list[EntryPoint]:
        return [item for item in installed if item.group == group]

    monkeypatch.setattr(registry, "entry_points", select)


def test_list_shows_every_group_and_the_built_in_plugins(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["list"]) == 0
    lines = capsys.readouterr().out.splitlines()
    for group in GROUPS:
        assert group in lines
    entries = [line.split() for line in lines if line.startswith("  ")]
    assert ["testbot", "embodiverse.robots.testbot"] in entries
    assert ["g1_29dof", "embodiverse.robots.g1_29dof"] in entries
    assert ["loopback", "embodiverse.adapters.loopback:LoopbackRobot"] in entries
    assert ["mujoco", "embodiverse.adapters.mujoco:MujocoRobot"] in entries
    assert [
        "mujoco_realtime",
        "embodiverse.adapters.mujoco:RealtimeMujocoRobot",
    ] in entries


def test_list_aligns_the_values_and_marks_empty_groups(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(
        monkeypatch,
        EntryPoint(name="g1", value="lab_robots.g1", group=ROBOT_DESCRIPTIONS),
        EntryPoint(name="unitree_g1", value="lab_robots:G1Robot", group=ADAPTERS),
        EntryPoint(name="sim", value="lab_sim:SimRobot", group=ADAPTERS),
    )
    assert main(["list"]) == 0
    assert capsys.readouterr().out == (
        "embodiverse.robot_descriptions\n"
        "  g1          lab_robots.g1\n"
        "embodiverse.adapters\n"
        "  sim         lab_sim:SimRobot\n"
        "  unitree_g1  lab_robots:G1Robot\n"
        "embodiverse.observations\n"
        "  (none)\n"
        "embodiverse.components\n"
        "  (none)\n"
    )


def test_list_does_not_load_plugins(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(
        monkeypatch,
        EntryPoint(name="broken", value="lab_no_such_module:Robot", group=ADAPTERS),
    )
    assert main(["list"]) == 0
    assert "  broken  lab_no_such_module:Robot\n" in capsys.readouterr().out


def test_list_imports_no_adapter_or_simulator() -> None:
    # In a new interpreter, because other tests import the adapters.
    script = """\
import sys
from embodiverse.app.cli import main
main(["list"])
top_level = {name.partition(".")[0] for name in sys.modules}
print("mujoco" in top_level, "embodiverse.adapters" in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.splitlines()[-1] == "False False"


def test_a_registry_error_is_printed_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install(
        monkeypatch,
        EntryPoint(name="g1", value="lab_robots:G1Robot", group=ADAPTERS),
        EntryPoint(name="g1", value="g1_tools:G1", group=ADAPTERS),
    )
    assert main(["list"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "embodiverse: error: 'g1' in group 'embodiverse.adapters' is registered by"
    )
    assert "Traceback" not in captured.err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert capsys.readouterr().out == f"embodiverse {__version__}\n"


def test_help_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--help"]) == 0
    help_text = capsys.readouterr().out
    assert help_text.startswith("usage: embodiverse ")
    assert "list the installed plugins" in help_text
    assert "run a policy on a robot" in help_text
    assert "write the files generated from their sources" in help_text


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param(
            ["frobnicate"],
            # Python versions differ in how they quote the choices that follow.
            "argument <command>: invalid choice: 'frobnicate' (choose from ",
            id="unknown command",
        ),
        pytest.param(
            [], "the following arguments are required: <command>", id="no command"
        ),
        pytest.param(
            ["list", "--all"], "unrecognized arguments: --all", id="unknown option"
        ),
    ],
)
def test_a_usage_error_returns_2_with_a_message(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert error.startswith("usage: embodiverse ")
    assert f"embodiverse: error: {message}" in error


def test_the_installed_console_script_runs() -> None:
    script = Path(sys.executable).parent / "embodiverse"
    if not script.exists():
        pytest.skip(f"the console script is not installed at {script}")
    result = subprocess.run(
        [script, "list"], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "  testbot " in result.stdout


# embodiverse run.


def summary(output: str) -> dict[str, str]:
    """Return the lines of a run's summary, by label."""
    lines = [re.split(r"\s{2,}", line, maxsplit=1) for line in output.splitlines()]
    return {label: value for label, value in lines}


@pytest.fixture
def closed(monkeypatch: pytest.MonkeyPatch) -> list[Mode]:
    """Record the mode each MuJoCo robot is in when it is closed."""
    modes: list[Mode] = []
    close = MujocoRobot.close

    def record(robot: MujocoRobot) -> None:
        modes.append(robot.read().mode)
        close(robot)

    monkeypatch.setattr(MujocoRobot, "close", record)
    return modes


def test_run_runs_the_zero_policy_and_prints_a_summary(
    capsys: pytest.CaptureFixture[str], closed: list[Mode]
) -> None:
    assert main([*RUN_ZERO, "--seconds", "1"]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = summary(captured.out)
    assert re.fullmatch(r"\d+\.\d{3} ms", lines.pop("longest tick"))
    assert lines == {
        "robot": "testbot through mujoco, in lockstep",
        "policy": "zero, at 50 Hz",
        "ticks": "50",
        "simulated time": "1.000 s",
        "rate": "50.00 Hz",
        "jitter": "0.000 ms",
        "overruns": "0",
        "faults": "none",
    }
    assert closed == [Mode.HOLD]  # stopped, then closed


def test_run_shapes_the_zero_policy_by_a_descriptor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor = TESTBOT_ZERO / "descriptor.yaml"
    argv = [*RUN_ZERO, "--descriptor", str(descriptor), "--seconds", "0.5"]
    assert main([*argv, "--command", "0.3", "0", "0"]) == 0
    lines = summary(capsys.readouterr().out)
    assert lines["policy"] == f"zero, shaped by {descriptor}, at 50 Hz"
    assert lines["ticks"] == "25"


def test_run_records_the_run_with_its_manifest(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, closed: list[Mode]
) -> None:
    path = tmp_path / "run.mcap"
    assert main([*RUN_ZERO, "--seconds", "1", "--record", str(path)]) == 0
    lines = summary(capsys.readouterr().out)
    assert lines["recording"] == str(path)
    assert lines["ticks"] == "50"
    run = read_run(path)
    assert run.ticks == 100 + 50  # the start ramp, then the policy
    manifest = run.manifest
    assert (manifest.robot, manifest.adapter) == ("testbot", "mujoco")
    assert manifest.policy_sha256 is None
    testbot = robot_description("testbot")
    assert manifest.description_hash == content_hash(testbot)
    assert manifest.descriptor_hash == content_hash(zero_policy_descriptor(testbot))
    assert closed == [Mode.HOLD]


def test_run_records_no_policy_file_for_the_zero_policy_shaped_by_a_descriptor(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    # The descriptor names a policy file, but the zero policy runs instead.
    descriptor = TESTBOT_ZERO / "descriptor.yaml"
    path = tmp_path / "run.mcap"
    argv = [*RUN_ZERO, "--descriptor", str(descriptor), "--record", str(path)]
    assert main([*argv, "--seconds", "0.5"]) == 0
    assert summary(capsys.readouterr().out)["recording"] == str(path)
    manifest = read_run(path).manifest
    assert manifest.policy_sha256 is None
    assert manifest.descriptor_hash == content_hash(load_descriptor(descriptor))


def test_run_does_not_overwrite_a_recording_and_exits_1(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    path = tmp_path / "run.mcap"
    path.write_bytes(b"an earlier run")
    assert main([*RUN_ZERO, "--record", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"embodiverse: error: {path}: the file exists, and a recording is never"
        " overwritten: choose another path\n"
    )
    assert path.read_bytes() == b"an earlier run"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param(
            ["run", "--robot", "nobot", "--adapter", "mujoco", "--policy", "zero"],
            "no 'nobot' in group 'embodiverse.robot_descriptions'; available: ",
            id="unknown robot",
        ),
        pytest.param(
            ["run", "--robot", "testbot", "--adapter", "nosim", "--policy", "zero"],
            "no 'nosim' in group 'embodiverse.adapters'; available: ",
            id="unknown adapter",
        ),
        pytest.param(
            [
                "run",
                "--robot",
                "testbot",
                "--adapter",
                "mujoco",
                "--descriptor",
                "missing.yaml",
            ],
            "missing.yaml: cannot read: No such file or directory",
            id="missing descriptor",
        ),
        pytest.param(
            ["run", "--robot", "g1_29dof", "--adapter", "mujoco", "--policy", "zero"],
            "adapter 'mujoco' cannot drive robot 'g1_29dof': robot 'g1_29dof' has"
            " no MJCF model: set models.mjcf in its description",
            id="adapter refuses the robot",
        ),
    ],
)
def test_run_reports_what_it_cannot_load_and_exits_1(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"embodiverse: error: {message}")
    assert "Traceback" not in captured.err


def test_run_reports_a_policy_file_that_is_not_onnx_and_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policy = tmp_path / "policy.onnx"
    policy.write_text("not an ONNX model\n", encoding="utf-8")
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    data = yaml.safe_load((TESTBOT_ZERO / "descriptor.yaml").read_text())
    data["policy"]["sha256"] = f"sha256:{digest}"
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text(yaml.safe_dump(data), encoding="utf-8")
    argv = ["run", "--robot", "testbot", "--adapter", "mujoco"]
    assert main([*argv, "--descriptor", str(descriptor)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith(
        f"embodiverse: error: {policy}: cannot load the ONNX policy: "
    )
    assert "Traceback" not in captured.err


def test_run_reports_a_refused_start_and_closes_the_robot(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    closed: list[Mode],
) -> None:
    def preflight(layer: SafetyLayer, state: RobotState) -> tuple[Fault, ...]:
        return (Fault(code="imu.error", message="the IMU reports an error"),)

    monkeypatch.setattr(SafetyLayer, "preflight", preflight)
    assert main(RUN_ZERO) == 1
    assert capsys.readouterr().err == (
        "embodiverse: error: cannot start the policy on robot 'testbot':\n"
        "- imu.error: the IMU reports an error\n"
    )
    assert closed == [Mode.PASSIVE]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        pytest.param(
            ["run", "--policy", "zero"],
            "the following arguments are required: --robot, --adapter",
            id="no robot or adapter",
        ),
        pytest.param(
            ["run", "--robot", "testbot", "--adapter", "mujoco"],
            "give --descriptor PATH, which names the policy file and how to run it,"
            " or --policy zero",
            id="no policy",
        ),
        pytest.param(
            [*RUN_ZERO, "--descriptor", "descriptor.yaml", "--rate", "100"],
            "--rate is for the zero policy without a descriptor",
            id="rate with a descriptor",
        ),
        pytest.param(
            [*RUN_ZERO, "--seconds", "0"],
            "--seconds must be a positive number, got 0",
            id="no time",
        ),
        pytest.param(
            [*RUN_ZERO, "--rate", "-50"],
            "--rate must be a positive number, got -50",
            id="negative rate",
        ),
        pytest.param(
            [*RUN_ZERO, "--command", "0.3", "0"],
            "argument --command: expected 3 arguments",
            id="two command values",
        ),
    ],
)
def test_run_reports_a_usage_error_and_exits_2(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    error = capsys.readouterr().err
    assert error.startswith("usage: embodiverse run ")
    assert f"embodiverse run: error: {message}" in error


# embodiverse generate.

TESTBOT_PAGE = "docs/robots/testbot.md"
BEGIN = begin_marker("robot-joint-tables")
END = end_marker("robot-joint-tables")


@pytest.fixture
def repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a copy of the repository's pyproject.toml and testbot's page.

    The page holds hand-written notes and an empty block, so it is out of date.
    testbot is made the only installed robot description, so the expected
    output stays the same as robots are added to the package.
    """
    testbot = EntryPoint("testbot", "embodiverse.robots.testbot", ROBOT_DESCRIPTIONS)
    monkeypatch.setattr(
        registry,
        "entry_points",
        lambda *, group: [testbot] if group == ROBOT_DESCRIPTIONS else [],
    )
    shutil.copy(ROOT / "pyproject.toml", tmp_path)
    page = tmp_path / TESTBOT_PAGE
    page.parent.mkdir(parents=True)
    page.write_text(f"# testbot\n\nNotes.\n\n{BEGIN}\n{END}\n", encoding="utf-8")
    return tmp_path


def up_to_date_page() -> str:
    table = robot_joint_table(robot_description("testbot"))
    return "# testbot\n\nNotes.\n\n" + render_block("robot-joint-tables", table)


def test_generate_writes_the_files_out_of_date(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate", "--root", str(repository)]) == 0
    assert capsys.readouterr().out == f"wrote {repository / TESTBOT_PAGE}\n"
    assert (repository / TESTBOT_PAGE).read_text(encoding="utf-8") == (
        up_to_date_page()
    )

    assert main(["generate", "--root", str(repository)]) == 0
    assert capsys.readouterr().out == "every generated file is up to date\n"


def test_generate_check_passes_when_every_file_is_up_to_date(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repository / TESTBOT_PAGE).write_text(up_to_date_page(), encoding="utf-8")
    assert main(["generate", "--check", "--root", str(repository)]) == 0
    captured = capsys.readouterr()
    assert captured.out == "every generated file is up to date\n"
    assert captured.err == ""


def test_generate_check_lists_the_files_out_of_date_and_exits_1(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = repository / TESTBOT_PAGE
    before = page.read_text(encoding="utf-8")
    assert main(["generate", "--check", "--root", str(repository)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "embodiverse: error: generated files are out of date:\n"
        f"- {page}\n"
        f"Run 'embodiverse generate --root {repository}' and commit the result.\n"
    )
    assert page.read_text(encoding="utf-8") == before


def test_generate_works_in_the_current_directory_by_default(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)
    assert main(["generate", "--check"]) == 1
    assert capsys.readouterr().err == (
        "embodiverse: error: generated files are out of date:\n"
        f"- {TESTBOT_PAGE}\n"
        "Run 'embodiverse generate' and commit the result.\n"
    )
    assert main(["generate"]) == 0
    assert capsys.readouterr().out == f"wrote {TESTBOT_PAGE}\n"
    assert main(["generate", "--check"]) == 0


def test_generate_refuses_a_directory_that_is_not_the_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "docs").mkdir()
    assert main(["generate", "--check", "--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"embodiverse: error: {tmp_path} is not the root of the Embodiverse"
        " repository, which holds pyproject.toml and docs/: run the command"
        " there, or give --root PATH\n"
    )
    assert list(tmp_path.iterdir()) == [tmp_path / "docs"]


def test_generate_reports_a_page_without_markers_and_exits_1(
    repository: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = repository / TESTBOT_PAGE
    page.write_text("# testbot\n\nNotes.\n", encoding="utf-8")
    assert main(["generate", "--root", str(repository)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        f"embodiverse: error: {page}: no generated block for"
        f" 'robot-joint-tables': add the lines {BEGIN!r} and {END!r} where the"
        " block belongs\n"
    )
    assert page.read_text(encoding="utf-8") == "# testbot\n\nNotes.\n"
