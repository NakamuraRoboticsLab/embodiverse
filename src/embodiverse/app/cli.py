"""The `embodiverse` command line.

`main` parses the arguments and runs one subcommand. Each subcommand is a
function that takes the parsed arguments and returns the exit status. It is
registered on the parser in `build_parser`, with its own arguments, and the
parser stores the function as `run`. A subcommand whose arguments depend
on each other also stores a `check`, which reports a bad combination as a
usage error.

Expected errors, such as a plugin that is not installed, a descriptor that
does not fit the robot or a start the safety layer refuses, are printed as
`embodiverse: error: <message>` on stderr, without a traceback, and exit
with status 1. Usage errors exit with status 2, as argparse does.
"""

import argparse
import contextlib
import math
import shlex
import sys
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

from embodiverse import __version__
from embodiverse.contracts.description import DescriptionError, RobotDescription
from embodiverse.contracts.descriptor import (
    DescriptorError,
    PolicyDescriptor,
    load_descriptor,
    resolve_joints,
    validate_against,
)
from embodiverse.contracts.registry import (
    ADAPTERS,
    GROUPS,
    RegistryError,
    entry_points_in,
    load,
    robot_description,
)
from embodiverse.contracts.robot import Robot
from embodiverse.runtime.policy import OnnxPolicy, Policy, PolicyError, ZeroPolicy
from embodiverse.runtime.recording import (
    McapRecorder,
    RecordingError,
    collect_manifest,
)
from embodiverse.runtime.runner import (
    ConstantCommand,
    Runner,
    RunStats,
    StartRefused,
    zero_policy_descriptor,
)
from embodiverse.tools.generators import GENERATORS, GeneratorError, generate

__all__ = [
    "AdapterRefused",
    "build_parser",
    "generate_command",
    "list_command",
    "main",
    "run_command",
]

_PROG = "embodiverse"

_Command = Callable[[argparse.Namespace], int]
"""A subcommand: takes the parsed arguments and returns the exit status."""

_Check = Callable[[argparse.Namespace], None]
"""Checks a subcommand's arguments together; exits with status 2 if they clash."""


class AdapterRefused(Exception):
    """An adapter refused to drive the robot it was given."""


_EXPECTED_ERRORS = (
    RegistryError,
    DescriptionError,
    DescriptorError,
    PolicyError,
    RecordingError,
    StartRefused,
    GeneratorError,
    AdapterRefused,
    ConnectionError,
    TimeoutError,
)
"""Errors that mean the command cannot do what was asked, rather than a bug:
printed without a traceback."""

_ZERO_RATE_HZ = 50.0
"""The zero policy's rate when no descriptor gives one."""


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line on `argv`, or on `sys.argv[1:]`; return the exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        check: _Check | None = getattr(args, "check", None)
        if check is not None:
            check(args)
    except SystemExit as stop:
        # argparse exits after printing the help, the version or a usage error.
        return int(stop.code or 0)
    run: _Command = args.run
    try:
        return run(args)
    except _EXPECTED_ERRORS as error:
        print(f"{_PROG}: error: {error}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Return the parser for the command line and every subcommand."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Embodiverse: shared contracts, adapters and runtime for"
        " humanoid robots. Robot descriptions, adapters and other plugins are"
        " found in every installed package that registers them.",
        epilog=f"Run '{_PROG} <command> --help' for help on a command.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    commands = parser.add_subparsers(
        title="commands", dest="command", metavar="<command>", required=True
    )

    list_parser = commands.add_parser(
        "list",
        help="list the installed plugins",
        description="List the plugins installed in this environment, by group:"
        " each one's name and the object it names. Nothing is imported, so a"
        " plugin is listed even if loading it would fail.",
    )
    list_parser.set_defaults(run=list_command)

    run_parser = commands.add_parser(
        "run",
        help="run a policy on a robot",
        description="Run a policy on a robot, through an adapter, at the"
        " policy's rate; then leave the robot in HOLD, close it and print a"
        " summary. The robot's description and the adapter are found by name"
        f" among the installed plugins ('{_PROG} list'). An adapter that"
        " simulates in lockstep, such as mujoco, runs as fast as the machine"
        " allows; any other runs in real time.",
        epilog="Every command passes through the safety layer before it reaches"
        " the robot (docs/policies.md).",
    )
    run_parser.add_argument(
        "--robot",
        required=True,
        metavar="NAME",
        help="the robot description, e.g. testbot",
    )
    run_parser.add_argument(
        "--adapter",
        required=True,
        metavar="NAME",
        help="the adapter, e.g. mujoco (lockstep) or mujoco_realtime",
    )
    run_parser.add_argument(
        "--descriptor",
        type=Path,
        metavar="PATH",
        help="the policy descriptor: the policy file it names, and how to run it",
    )
    run_parser.add_argument(
        "--policy",
        choices=("onnx", "zero"),
        default="onnx",
        help="onnx (the default): the ONNX file the descriptor names; zero: the"
        " zero policy, which holds every joint it drives at its default"
        " position, shaped by the descriptor if one is given",
    )
    run_parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        metavar="S",
        help="how long to run the policy, after the start ramp (default: 10)",
    )
    run_parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help="a constant velocity command: forward and leftward velocity, in"
        " m/s, and yaw rate, in rad/s; clipped to the descriptor's ranges"
        " (default: 0 0 0)",
    )
    run_parser.add_argument(
        "--rate",
        type=float,
        metavar="HZ",
        help="the zero policy's rate without a descriptor"
        f" (default: {_ZERO_RATE_HZ:g})",
    )
    run_parser.add_argument(
        "--record",
        type=Path,
        metavar="PATH",
        help="record the run to a new MCAP file at PATH: every tick's state,"
        " command and faults, with a manifest of the platform, robot, policy and"
        " code that ran (needs embodiverse[record])",
    )
    run_parser.set_defaults(run=run_command, check=partial(_check_run, run_parser))

    generators = ", ".join(f"{item.name} ({item.description})" for item in GENERATORS)
    generate_parser = commands.add_parser(
        "generate",
        help="write the files generated from their sources",
        description="Write the files generated from a source of truth, such as"
        " the joint tables of the robot pages in docs/robots, generated from"
        " the robot descriptions, and print each file written. A generator"
        " owns a whole file, or a block between marker lines in a"
        " hand-written one, and leaves the rest of the file as it is. With"
        " --check, write nothing, print the files that are out of date, and"
        " exit with status 1 if there are any.",
        epilog=f"The generators: {generators}.",
    )
    generate_parser.add_argument(
        "--check",
        action="store_true",
        # `check` is the attribute that main calls to check the arguments.
        dest="check_only",
        help="write nothing; exit with status 1 if a generated file is out of date",
    )
    generate_parser.add_argument(
        "--root",
        type=Path,
        default=Path(),
        metavar="PATH",
        help="the root of the Embodiverse repository (default: the current directory)",
    )
    generate_parser.set_defaults(run=generate_command)

    return parser


def list_command(args: argparse.Namespace) -> int:
    """Print each group and the entry points registered in it, without loading any."""
    groups = {group: entry_points_in(group) for group in GROUPS}
    width = max((len(name) for found in groups.values() for name in found), default=0)
    for group, found in groups.items():
        print(group)
        if not found:
            print("  (none)")
        for name, entry_point in found.items():
            print(f"  {name:<{width}}  {entry_point.value}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Run a policy on a robot, leave the robot in HOLD, close it, print a summary.

    Everything is loaded and checked before the robot is connected: the
    description, the descriptor against it, the policy against the
    descriptor, and the runner. Once the adapter has built the robot, the
    robot is closed however the command ends. With --record, the recording
    is created before the robot is connected, and closed however the run
    ends, so a run that fails is recorded up to its failure.
    """
    description = robot_description(args.robot)
    if args.descriptor is None:
        rate_hz = _ZERO_RATE_HZ if args.rate is None else args.rate
        descriptor = zero_policy_descriptor(description, rate_hz=rate_hz)
    else:
        descriptor = load_descriptor(args.descriptor)
        validate_against(descriptor, description)
    policy: Policy
    if args.policy == "zero":
        driven = resolve_joints(descriptor.action.joints, description)
        size = descriptor.observation_size * descriptor.history.length
        policy = ZeroPolicy(size, len(driven))
    else:
        policy = OnnxPolicy.from_descriptor(descriptor, description)
    robot = _adapter(args.adapter, description)
    try:
        commands = ConstantCommand(*args.command)
        with _recorder(args, description, descriptor) as recorder:
            runner = Runner(
                robot, policy, descriptor, commands=commands, recorder=recorder
            )
            robot.connect()
            try:
                stats = runner.run(args.seconds)
            finally:
                runner.stop()
    finally:
        robot.close()
    if args.policy == "onnx":
        name = str(descriptor.policy_path())
    elif args.descriptor is not None:
        name = f"zero, shaped by {args.descriptor}"
    else:
        name = "zero"
    robot_name = f"{args.robot} through {args.adapter}"
    policy_name = f"{name}, at {descriptor.rate_hz:g} Hz"
    print(_summary(stats, robot_name, policy_name, args.record))
    return 0


def generate_command(args: argparse.Namespace) -> int:
    """Write the generated files that are out of date; with --check, list them.

    Every generator runs before anything is written. Prints each file
    written; with --check, prints each file out of date on stderr and
    returns 1 if there is one. Raises GeneratorError if the root does not
    look like the repository, with pyproject.toml and docs/ in it.
    """
    root: Path = args.root
    if not ((root / "pyproject.toml").is_file() and (root / "docs").is_dir()):
        raise GeneratorError(
            f"{root.absolute()} is not the root of the Embodiverse repository,"
            " which holds pyproject.toml and docs/: run the command there, or"
            " give --root PATH"
        )
    changed = generate(root, check=args.check_only)
    if args.check_only and changed:
        again = f"{_PROG} generate"
        if root != Path():
            again += f" --root {shlex.quote(str(root))}"
        print(f"{_PROG}: error: generated files are out of date:", file=sys.stderr)
        for path in changed:
            print(f"- {root / path}", file=sys.stderr)
        print(f"Run '{again}' and commit the result.", file=sys.stderr)
        return 1
    for path in changed:
        print(f"wrote {root / path}")
    if not changed:
        print("every generated file is up to date")
    return 0


def _check_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Report arguments of `run` that do not go together as a usage error."""
    if args.descriptor is None and args.policy == "onnx":
        parser.error(
            "give --descriptor PATH, which names the policy file and how to run"
            " it, or --policy zero"
        )
    if args.descriptor is not None and args.rate is not None:
        parser.error(
            "--rate is for the zero policy without a descriptor: a"
            " descriptor gives its policy's rate"
        )
    if not (math.isfinite(args.seconds) and args.seconds > 0):
        parser.error(f"--seconds must be a positive number, got {args.seconds:g}")
    if args.rate is not None and not (math.isfinite(args.rate) and args.rate > 0):
        parser.error(f"--rate must be a positive number, got {args.rate:g}")


def _recorder(
    args: argparse.Namespace,
    description: RobotDescription,
    descriptor: PolicyDescriptor,
) -> contextlib.AbstractContextManager[McapRecorder | None]:
    """Return the recorder --record asks for, with the run's manifest, or none."""
    if args.record is None:
        return contextlib.nullcontext()
    manifest = collect_manifest(
        description=description, descriptor=descriptor, adapter=args.adapter
    )
    if args.policy == "zero":
        # The zero policy runs, not the policy file the descriptor may name.
        manifest = manifest.model_copy(update={"policy_sha256": None})
    return McapRecorder(args.record, manifest)


def _adapter(name: str, description: RobotDescription) -> Robot:
    """Build the robot with the adapter registered as `name`, unconnected.

    Raises AdapterRefused if the adapter raises ValueError for the
    description, for example because the robot has no model file the
    adapter can simulate.
    """
    factory = load(ADAPTERS, name)
    if not callable(factory):
        raise RegistryError(
            f"adapter {name!r} is registered as {factory!r}, which is not callable:"
            f" an entry point in group {ADAPTERS!r} must name a callable that takes"
            " a RobotDescription and returns a Robot"
        )
    try:
        robot = factory(description)
    except ValueError as error:
        raise AdapterRefused(
            f"adapter {name!r} cannot drive robot {description.name!r}: {error}"
        ) from error
    if not isinstance(robot, Robot):
        raise RegistryError(
            f"adapter {name!r} returned {robot!r}, which is not a Robot"
        )
    return robot


def _summary(stats: RunStats, robot: str, policy: str, recording: Path | None) -> str:
    """Describe a run in a few aligned lines."""
    if stats.faults:
        faults = ", ".join(f"{code} × {count}" for code, count in stats.faults.items())
    else:
        faults = "none"
    lines = [
        ("robot", f"{robot}, {'in lockstep' if stats.lockstep else 'in real time'}"),
        ("policy", policy),
        ("ticks", str(stats.ticks)),
        (
            "simulated time" if stats.lockstep else "wall time",
            f"{stats.duration_s:.3f} s",
        ),
        ("rate", f"{stats.rate_hz:.2f} Hz"),
        ("jitter", f"{stats.jitter_s * 1e3:.3f} ms"),
        ("longest tick", f"{stats.max_compute_s * 1e3:.3f} ms"),
        ("overruns", str(stats.overruns)),
        ("faults", faults),
    ]
    if stats.latched:
        codes = ", ".join(fault.code for fault in stats.latched_by)
        lines.append(("hold", f"latched by {codes}"))
    if stats.left_policy is not None:
        lines.append(
            ("ended early", f"the robot left POLICY for {stats.left_policy.name}")
        )
    if recording is not None:
        lines.append(("recording", str(recording)))
    width = max(len(label) for label, _ in lines)
    return "\n".join(f"{label:<{width}}  {value}" for label, value in lines)
