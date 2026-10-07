"""Generators: files derived from a source of truth, and checked for drift.

A fact is written once, in its source of truth, such as a robot
description, but people also need it in a form they can read, such as a
joint table in the documentation. Copied by hand, the copy drifts from the
source. A generator writes the derived form from the source instead, and a
test regenerates every generated file and fails on any difference, so a
source changed without regenerating fails the checks
(docs/adding-a-robot.md).

A generator may own a whole file, or only a block of a hand-written one,
between two marker lines. The line after the first tells editors that the
block is generated and how to regenerate it:

    <!-- BEGIN GENERATED: robot-joint-tables -->
    <!-- This block is generated: do not edit it by hand. ... -->
    ...
    <!-- END GENERATED: robot-joint-tables -->

Everything outside the block belongs to the page's authors and is kept
exactly as it is. `replace_block` replaces a block in a file's content, and
`generate` runs every generator in GENERATORS and writes, or in check mode
only lists, the files whose content differs from what was generated.
"""

import math
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.registry import (
    ROBOT_DESCRIPTIONS,
    entry_points_in,
    robot_description,
)

__all__ = [
    "GENERATORS",
    "ROBOT_JOINT_TABLES",
    "Generator",
    "GeneratorError",
    "begin_marker",
    "end_marker",
    "generate",
    "render_block",
    "replace_block",
    "robot_joint_table",
]

_NOTICE = (
    "<!-- This block is generated: do not edit it by hand. Change its source,"
    " then run: embodiverse generate -->"
)


class GeneratorError(ValueError):
    """A generator cannot produce its files, or a file cannot be read or written."""


@dataclass(frozen=True)
class Generator:
    """Produces files, or blocks in files, from their source of truth.

    `produce` takes the repository's root directory and returns every file
    the generator produces, as {path: full new content}. Each path is
    relative to the root, in POSIX form, such as `docs/robots/testbot.md`.
    A generator that owns a block of a file reads the file and returns all
    of it, with the block replaced. `produce` writes nothing, and returns
    the same content for the same sources on every machine and Python
    version. It raises GeneratorError for a file it cannot produce.
    """

    name: str
    """Lowercase words joined by hyphens, e.g. `robot-joint-tables`; it names
    the generator's blocks."""

    description: str
    """What the generator produces, in one line."""

    produce: Callable[[Path], Mapping[str, str]] = field(repr=False)

    def __post_init__(self) -> None:
        # The name goes inside HTML comments, which must not contain `--`.
        if not re.fullmatch(r"[a-z][a-z0-9]*(-[a-z0-9]+)*", self.name):
            raise ValueError(
                f"generator name {self.name!r} is not lowercase words joined by"
                " single hyphens, e.g. 'robot-joint-tables'"
            )


# --- Generated blocks ------------------------------------------------------


def begin_marker(name: str) -> str:
    """Return the line that starts generator `name`'s block."""
    return f"<!-- BEGIN GENERATED: {name} -->"


def end_marker(name: str) -> str:
    """Return the line that ends generator `name`'s block."""
    return f"<!-- END GENERATED: {name} -->"


def render_block(name: str, content: str) -> str:
    """Return generator `name`'s block holding `content`, ending in a newline.

    The block is the begin marker, the line telling editors how to
    regenerate it, a blank line, `content` without its leading and trailing
    newlines, a blank line and the end marker. The blank lines keep the
    Markdown in `content` apart from the markers.
    """
    body = content.strip("\n")
    return f"{begin_marker(name)}\n{_NOTICE}\n\n{body}\n\n{end_marker(name)}\n"


def replace_block(text: str, name: str, content: str) -> str:
    """Return `text` with generator `name`'s block replaced by one holding `content`.

    The block runs from the begin marker's line to the end marker's line,
    both included, and becomes `render_block(name, content)`. A marker is a
    line of its own, with nothing before it and only whitespace after it.
    Every character before and after the block is kept as it is. Markers
    of other generators are ordinary text.

    Raises ValueError if a marker is missing or appears more than once, or
    if the end marker comes before the begin marker.
    """
    begin, end = begin_marker(name), end_marker(name)
    lines = text.splitlines(keepends=True)
    found: dict[str, list[int]] = {begin: [], end: []}
    for index, line in enumerate(lines):
        if line.rstrip() in found:
            found[line.rstrip()].append(index)
    if not found[begin] and not found[end]:
        raise ValueError(
            f"no generated block for {name!r}: add the lines {begin!r} and"
            f" {end!r} where the block belongs"
        )
    for marker, other in ((begin, end), (end, begin)):
        if not found[marker]:
            line = found[other][0] + 1
            raise ValueError(
                f"{other!r} is on line {line}, but {marker!r} is missing: a"
                f" generated block starts with {begin!r} and ends with {end!r}"
            )
        if len(found[marker]) > 1:
            where = _join([str(index + 1) for index in found[marker]])
            raise ValueError(
                f"{marker!r} is on lines {where}: each generator has one block"
                " in a file"
            )
    first, last = found[begin][0], found[end][0]
    if last < first:
        raise ValueError(
            f"{end!r}, on line {last + 1}, comes before {begin!r}, on line {first + 1}"
        )
    before, after = "".join(lines[:first]), "".join(lines[last + 1 :])
    return before + render_block(name, content) + after


# --- The robot joint tables ------------------------------------------------

_ROBOT_JOINT_TABLES = "robot-joint-tables"
_ROBOT_PAGES = "docs/robots"
_BUILT_IN_ROBOTS = "embodiverse.robots."
"""Prefix of the packages that hold the built-in robot descriptions."""

_JOINT_COLUMNS = (
    "#",
    "Joint",
    "Vendor name",
    "Motor",
    "Position limits (rad)",
    "Velocity limit (rad/s)",
    "Effort limit (N·m)",
    "Default (rad)",
    "kp (N·m/rad)",
    "kd (N·m·s/rad)",
    "Gains side",
)


def robot_joint_table(description: RobotDescription) -> str:
    """Return Markdown documenting `description`'s joints, joint groups and IMU.

    It is the content of the `robot-joint-tables` block: the description's
    name, status, joint count and content hash, then a table of its joints
    in description order, then its joint groups, then its IMU's frame,
    parent and mount. Numbers are written with `:g`, at most six
    significant digits, and degrees are rounded to 0.1° first, so the
    same description always gives the same text.
    """
    lines = [
        f"- Robot description: {_code(description.name)}",
        f"- Status: {_code(description.status)}",
        f"- Joints: {description.num_joints}",
        f"- Content hash: {_code(content_hash(description))}",
        "",
        *_table(
            _JOINT_COLUMNS,
            [
                (
                    str(index),
                    _code(joint.name),
                    "—" if joint.vendor_name is None else _code(joint.vendor_name),
                    str(joint.motor_index),
                    _limits(*joint.limits.position),
                    _number(joint.limits.velocity),
                    _number(joint.limits.effort),
                    _number(joint.default_position),
                    _number(joint.gains.kp),
                    _number(joint.gains.kd),
                    joint.gains.side,
                )
                for index, joint in enumerate(description.joints)
            ],
        ),
        "",
    ]
    if description.joint_groups:
        groups = description.joint_groups.items()
        rows = [
            (_code(group), ", ".join(map(_code, joints))) for group, joints in groups
        ]
        lines += [*_table(("Joint group", "Joints"), rows), ""]
    else:
        lines += ["No joint groups.", ""]
    imu = description.imu
    lines += _table(
        ("IMU frame", "Parent", "`mount_xyz` (m)", "`mount_rpy` (rad)"),
        [
            (
                _code(imu.frame),
                _code(imu.parent),
                _vector(imu.mount_xyz),
                _vector(imu.mount_rpy),
            )
        ],
    )
    return "\n".join(lines) + "\n"


def _robot_joint_tables(root: Path) -> dict[str, str]:
    """Produce the page of every built-in robot, with its tables in a block."""
    pages: dict[str, str] = {}
    for name, entry_point in entry_points_in(ROBOT_DESCRIPTIONS).items():
        if not entry_point.module.startswith(_BUILT_IN_ROBOTS):
            # A description from another package is documented with it.
            continue
        path = f"{_ROBOT_PAGES}/{name}.md"
        content = robot_joint_table(robot_description(name))
        text = _read(root, path)
        if text is None:
            pages[path] = (
                f"# {name}\n\nThis page is for hand-written notes about {name},"
                " above and below the block generated from its robot"
                f" description.\n\n{render_block(_ROBOT_JOINT_TABLES, content)}"
            )
            continue
        try:
            pages[path] = replace_block(text, _ROBOT_JOINT_TABLES, content)
        except ValueError as error:
            raise GeneratorError(f"{root / path}: {error}") from error
    return pages


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Return the lines of a Markdown table, laid out as the hand-written ones are."""
    lines = [_row(header), "|" + "---|" * len(header)]
    return lines + [_row(row) for row in rows]


def _row(cells: Sequence[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _code(text: str) -> str:
    """Return `text` as inline code, with `|` escaped so it stays in its cell."""
    return "`" + text.replace("|", "\\|") + "`"


def _number(value: float) -> str:
    # Adding 0.0 turns -0.0, which rounding can produce, into 0.0.
    return f"{value + 0.0:g}"


def _limits(lower: float, upper: float) -> str:
    """Return a position range in rad, with degrees in parentheses."""
    degrees = [_number(round(math.degrees(value), 1)) for value in (lower, upper)]
    return f"{_number(lower)} to {_number(upper)} ({degrees[0]}° to {degrees[1]}°)"


def _vector(values: Sequence[float]) -> str:
    return f"[{', '.join(map(_number, values))}]"


# --- The generators and running them --------------------------------------

ROBOT_JOINT_TABLES = Generator(
    name=_ROBOT_JOINT_TABLES,
    description="the joints, joint groups and IMU of each built-in robot,"
    f" in a block of {_ROBOT_PAGES}/<name>.md",
    produce=_robot_joint_tables,
)
"""Writes `robot_joint_table` of every robot description registered by a
package under `embodiverse.robots` into a block of `docs/robots/<name>.md`,
and creates the page if it does not exist. A page that exists without the
block's markers is an error. Descriptions registered by other packages are
documented in those packages."""

# The built-in generators. They are listed here, not found through an
# entry-point group like other plugins, because every generator so far is
# Embodiverse's own; a group can be added once generators exist in other
# packages.
GENERATORS: tuple[Generator, ...] = (ROBOT_JOINT_TABLES,)


def generate(
    root: str | os.PathLike[str],
    *,
    check: bool,
    generators: Sequence[Generator] = GENERATORS,
) -> list[str]:
    """Run `generators` on the repository at `root`; return the files that differ.

    A file differs if it does not exist or its content is not what a
    generator produced. In write mode (`check=False`), those files are
    written, creating directories as needed; in check mode, nothing is
    written. Either way their paths are returned, relative to `root`, in
    POSIX form and sorted. Every generator runs before anything is written,
    so a generator that fails leaves every file as it was.

    Raises GeneratorError if a generator produces a path that is not
    relative to the root, if two generators produce the same file, or if a
    file cannot be read or written. Errors raised by a generator, such as
    GeneratorError or the registry's RegistryError, pass through.
    """
    base = Path(root)
    produced: dict[str, tuple[str, str]] = {}
    for generator in generators:
        for path, content in generator.produce(base).items():
            pure = PurePosixPath(path)
            if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path:
                raise GeneratorError(
                    f"generator {generator.name!r} produced {path!r}: a path must"
                    " be relative to the repository's root, in POSIX form,"
                    " without '.' or '..'"
                )
            if path in produced:
                raise GeneratorError(
                    f"{path} is produced by both {produced[path][0]!r} and"
                    f" {generator.name!r}: a file has one generator, which may"
                    " own the whole file or one block in it"
                )
            produced[path] = (generator.name, content)
    changed = sorted(
        path for path, (_, content) in produced.items() if _read(base, path) != content
    )
    if not check:
        for path in changed:
            _write(base, path, produced[path][1])
    return changed


def _read(root: Path, path: str) -> str | None:
    """Return the content of the file at `path` under `root`, or None if there is none.

    The bytes are decoded as they are, so line endings are kept.
    """
    file = root / path
    try:
        return file.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    except OSError as error:
        raise GeneratorError(
            f"{file}: cannot read: {error.strerror or error}"
        ) from error
    except UnicodeDecodeError as error:
        raise GeneratorError(f"{file}: not valid UTF-8: {error}") from error


def _write(root: Path, path: str, content: str) -> None:
    """Write `content` to the file at `path` under `root`, as UTF-8 bytes."""
    file = root / path
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(content.encode("utf-8"))
    except OSError as error:
        raise GeneratorError(
            f"{file}: cannot write: {error.strerror or error}"
        ) from error


def _join(items: list[str]) -> str:
    """Return two or more `items` joined as in a sentence: `a, b and c`."""
    return f"{', '.join(items[:-1])} and {items[-1]}"
