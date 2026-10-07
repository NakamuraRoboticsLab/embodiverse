"""Tests of the generators, of generated blocks and of `generate`.

The robot joint tables are generated into a temporary directory standing in
for the repository, from the robot descriptions installed with the package.
"""

from collections.abc import Mapping
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from embodiverse.contracts import registry
from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.hashing import content_hash
from embodiverse.contracts.registry import ROBOT_DESCRIPTIONS, robot_description
from embodiverse.tools.generators import (
    GENERATORS,
    ROBOT_JOINT_TABLES,
    Generator,
    GeneratorError,
    begin_marker,
    end_marker,
    generate,
    render_block,
    replace_block,
    robot_joint_table,
)

MINIMAL = Path(__file__).parents[1] / "fixtures/robots/minimal/description.yaml"
BEGIN = "<!-- BEGIN GENERATED: tables -->"
END = "<!-- END GENERATED: tables -->"
NOTICE = (
    "<!-- This block is generated: do not edit it by hand. Change its source,"
    " then run: embodiverse generate -->"
)
TESTBOT_PAGE = "docs/robots/testbot.md"


def fake(name: str, files: Mapping[str, str]) -> Generator:
    """Return a generator called `name` that produces `files`."""
    return Generator(name=name, description=f"{name} files", produce=lambda _: files)


def install(monkeypatch: pytest.MonkeyPatch, *installed: EntryPoint) -> None:
    """Make `installed` the only installed entry points."""

    def select(*, group: str) -> list[EntryPoint]:
        return [item for item in installed if item.group == group]

    monkeypatch.setattr(registry, "entry_points", select)


@pytest.fixture(autouse=True)
def only_testbot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make testbot the only installed robot description.

    The files these tests expect then stay the same as robots are added to
    the package; the drift test checks the real ones.
    """
    install(
        monkeypatch,
        EntryPoint("testbot", "embodiverse.robots.testbot", ROBOT_DESCRIPTIONS),
    )


# Generated blocks.


def test_a_block_is_the_markers_the_notice_and_the_content() -> None:
    assert begin_marker("tables") == BEGIN
    assert end_marker("tables") == END
    expected = f"{BEGIN}\n{NOTICE}\n\n| a |\n| b |\n\n{END}\n"
    assert render_block("tables", "| a |\n| b |\n") == expected
    assert render_block("tables", "\n\n| a |\n| b |") == expected


def test_replace_block_replaces_the_block_and_nothing_else() -> None:
    before = "# Title\n\nHand-written notes.\n\n"
    after = "\nMore notes.\n"
    text = f"{before}{BEGIN}\nold table\n{END}\n{after}"
    assert replace_block(text, "tables", "new table") == (
        before + render_block("tables", "new table") + after
    )


def test_replace_block_keeps_the_text_around_the_block_exactly() -> None:
    # Line endings, trailing spaces, other generators' blocks and a last line
    # without a newline are the authors', and stay as they are.
    before = f"Notes\r\nwith  trailing spaces  \n{begin_marker('other')}\nkept\n"
    after = f"{end_marker('other')}\n  indented\r\nno final newline"
    text = f"{before}{BEGIN}  \r\nold\n{END}\n{after}"
    assert replace_block(text, "tables", "new") == (
        before + render_block("tables", "new") + after
    )


def test_replace_block_is_idempotent() -> None:
    text = f"Notes.\n\n{BEGIN}\n{END}\n"
    once = replace_block(text, "tables", "| a |")
    assert replace_block(once, "tables", "| a |") == once


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(
            "Notes only.\n",
            f"no generated block for 'tables': add the lines {BEGIN!r} and {END!r}"
            " where the block belongs",
            id="no markers",
        ),
        pytest.param(
            f"Notes.\n{BEGIN}\ntable\n",
            f"{BEGIN!r} is on line 2, but {END!r} is missing: a generated block"
            f" starts with {BEGIN!r} and ends with {END!r}",
            id="no end",
        ),
        pytest.param(
            f"table\n{END}\n",
            f"{END!r} is on line 2, but {BEGIN!r} is missing: a generated block"
            f" starts with {BEGIN!r} and ends with {END!r}",
            id="no begin",
        ),
        pytest.param(
            f"  {BEGIN}\n{END}\n",
            f"{END!r} is on line 2, but {BEGIN!r} is missing: a generated block"
            f" starts with {BEGIN!r} and ends with {END!r}",
            id="indented begin",
        ),
        pytest.param(
            f"See {BEGIN} and {END}.\n",
            f"no generated block for 'tables': add the lines {BEGIN!r} and {END!r}"
            " where the block belongs",
            id="markers inside a line",
        ),
    ],
)
def test_replace_block_refuses_a_missing_marker(text: str, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        replace_block(text, "tables", "new")
    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param(
            f"{BEGIN}\n{BEGIN}\n{END}\n",
            f"{BEGIN!r} is on lines 1 and 2: each generator has one block in a file",
            id="two begins",
        ),
        pytest.param(
            f"{BEGIN}\n{END}\nnotes\n{END}\n",
            f"{END!r} is on lines 2 and 4: each generator has one block in a file",
            id="two ends",
        ),
        pytest.param(
            f"{BEGIN}\n{END}\n{BEGIN}\n{END}\n{BEGIN}\n{END}\n",
            f"{BEGIN!r} is on lines 1, 3 and 5: each generator has one block in a file",
            id="three blocks",
        ),
    ],
)
def test_replace_block_refuses_a_duplicated_marker(text: str, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        replace_block(text, "tables", "new")
    assert str(caught.value) == message


def test_replace_block_refuses_markers_out_of_order() -> None:
    with pytest.raises(ValueError) as caught:
        replace_block(f"Notes.\n{END}\ntable\n{BEGIN}\n", "tables", "new")
    assert str(caught.value) == (
        f"{END!r}, on line 2, comes before {BEGIN!r}, on line 4"
    )


# Generators.


@pytest.mark.parametrize(
    "name", ["Tables", "robot_tables", "robot--tables", "-tables", "tables-", ""]
)
def test_a_generator_name_is_lowercase_words_joined_by_hyphens(name: str) -> None:
    with pytest.raises(ValueError, match="is not lowercase words joined by"):
        fake(name, {})


def test_the_built_in_generators() -> None:
    assert ROBOT_JOINT_TABLES in GENERATORS
    names = [generator.name for generator in GENERATORS]
    assert len(set(names)) == len(names)
    for generator in GENERATORS:
        assert generator.description
        assert "\n" not in generator.description


# The robot joint tables.


def test_the_testbot_tables(tmp_path: Path) -> None:
    assert generate(tmp_path, check=False) == [TESTBOT_PAGE]
    page = (tmp_path / TESTBOT_PAGE).read_text(encoding="utf-8")
    lines = page.splitlines()
    start = lines.index(begin_marker("robot-joint-tables"))
    testbot_hash = content_hash(robot_description("testbot"))
    assert lines[start : start + 11] == [
        "<!-- BEGIN GENERATED: robot-joint-tables -->",
        NOTICE,
        "",
        "- Robot description: `testbot`",
        "- Status: `active`",
        "- Joints: 6",
        f"- Content hash: `{testbot_hash}`",
        "",
        "| # | Joint | Vendor name | Motor | Position limits (rad) | Velocity limit"
        " (rad/s) | Effort limit (N·m) | Default (rad) | kp (N·m/rad) | kd"
        " (N·m·s/rad) | Gains side |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        "| 0 | `left_hip_pitch` | — | 0 | -1.5 to 0.5 (-85.9° to 28.6°) | 20 | 60"
        " | -0.3 | 60 | 2 | joint |",
    ]
    assert (
        "| 4 | `right_knee` | — | 4 | 0 to 2.2 (0° to 126.1°) | 20 | 80 | 0.6 | 80"
        " | 2 | joint |"
    ) in lines
    assert "| `left_leg` | `left_hip_pitch`, `left_knee`, `left_ankle_pitch` |" in (
        lines
    )
    assert "| `imu_link` | `pelvis` | [-0.02, 0, 0.04] | [0, 0, 0] |" in lines
    assert lines[-1] == "<!-- END GENERATED: robot-joint-tables -->"


def minimal_variant() -> RobotDescription:
    """Return the minimal fixture with a motor-side gain, a `|` and no groups."""
    data = load_description(MINIMAL).model_dump()
    data["joints"][0]["limits"]["position"] = (-0.0001, 2.88)
    data["joints"][1]["vendor_name"] = "KNEE|R"
    data["joints"][1]["gains"]["side"] = "motor"
    data["joint_groups"] = {}
    data["imu"]["mount_rpy"] = (0.0, 0.25, -1.5)
    return RobotDescription.model_validate(data)


def test_the_joint_table_of_a_vendor_robot() -> None:
    description = minimal_variant()
    lines = robot_joint_table(description).splitlines()
    assert lines[:4] == [
        "- Robot description: `minimal`",
        "- Status: `draft`",
        "- Joints: 2",
        f"- Content hash: `{content_hash(description)}`",
    ]
    assert lines[7:] == [
        # -0.0001 rad is -0.0057°, which rounds to 0°, not -0°.
        "| 0 | `left_knee` | `KNEE_L` | 0 | -0.0001 to 2.88 (0° to 165°) | 20 | 139"
        " | 0.3 | 150 | 4 | joint |",
        "| 1 | `right_knee` | `KNEE\\|R` | 1 | -0.087 to 2.88 (-5° to 165°) | 20"
        " | 139 | 0.3 | 150 | 4 | motor |",
        "",
        "No joint groups.",
        "",
        "| IMU frame | Parent | `mount_xyz` (m) | `mount_rpy` (rad) |",
        "|---|---|---|---|",
        "| `imu_link` | `pelvis` | [-0.04, 0, -0.08] | [0, 0.25, -1.5] |",
    ]


def test_a_missing_robot_page_is_created(tmp_path: Path) -> None:
    assert generate(tmp_path, check=False) == [TESTBOT_PAGE]
    table = robot_joint_table(robot_description("testbot"))
    assert (tmp_path / TESTBOT_PAGE).read_text(encoding="utf-8") == (
        "# testbot\n\nThis page is for hand-written notes about testbot, above and"
        " below the block generated from its robot description.\n\n"
        + render_block("robot-joint-tables", table)
    )


def test_a_robot_page_keeps_its_notes(tmp_path: Path) -> None:
    page = tmp_path / TESTBOT_PAGE
    page.parent.mkdir(parents=True)
    begin, end = begin_marker("robot-joint-tables"), end_marker("robot-joint-tables")
    page.write_text(f"# testbot\n\nAbove.\n\n{begin}\n{end}\n\nBelow.\n", "utf-8")
    assert generate(tmp_path, check=False) == [TESTBOT_PAGE]
    table = robot_joint_table(robot_description("testbot"))
    assert page.read_text(encoding="utf-8") == (
        "# testbot\n\nAbove.\n\n"
        + render_block("robot-joint-tables", table)
        + "\nBelow.\n"
    )


def test_a_robot_page_without_markers_is_an_error(tmp_path: Path) -> None:
    page = tmp_path / TESTBOT_PAGE
    page.parent.mkdir(parents=True)
    page.write_text("# testbot\n\nNotes only.\n", encoding="utf-8")
    with pytest.raises(GeneratorError) as caught:
        generate(tmp_path, check=False)
    begin, end = begin_marker("robot-joint-tables"), end_marker("robot-joint-tables")
    assert str(caught.value) == (
        f"{page}: no generated block for 'robot-joint-tables': add the lines"
        f" {begin!r} and {end!r} where the block belongs"
    )
    assert page.read_text(encoding="utf-8") == "# testbot\n\nNotes only.\n"


def test_robot_descriptions_from_other_packages_are_left_to_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Loading lab_robots.g1 would fail: it does not exist.
    install(
        monkeypatch,
        EntryPoint("testbot", "embodiverse.robots.testbot", ROBOT_DESCRIPTIONS),
        EntryPoint("g1", "lab_robots.g1", ROBOT_DESCRIPTIONS),
    )
    assert generate(tmp_path, check=False) == [TESTBOT_PAGE]
    assert [path.name for path in (tmp_path / "docs/robots").iterdir()] == [
        "testbot.md"
    ]


# Running the generators.


def test_check_mode_writes_nothing(tmp_path: Path) -> None:
    page = tmp_path / TESTBOT_PAGE
    page.parent.mkdir(parents=True)
    begin, end = begin_marker("robot-joint-tables"), end_marker("robot-joint-tables")
    page.write_text(f"Notes.\n{begin}\nstale\n{end}\n", encoding="utf-8")
    other = fake("other", {"docs/new/file.md": "new\n"})
    generators = (ROBOT_JOINT_TABLES, other)
    assert generate(tmp_path, check=True, generators=generators) == [
        "docs/new/file.md",
        TESTBOT_PAGE,
    ]
    assert page.read_text(encoding="utf-8") == f"Notes.\n{begin}\nstale\n{end}\n"
    assert not (tmp_path / "docs/new").exists()


def test_write_mode_writes_what_differs_and_is_idempotent(tmp_path: Path) -> None:
    (tmp_path / "same.md").write_text("same\n", encoding="utf-8")
    (tmp_path / "old.md").write_text("old\n", encoding="utf-8")
    generator = fake(
        "files", {"same.md": "same\n", "old.md": "new\n", "a/b/made.md": "made\n"}
    )
    generators = (generator, ROBOT_JOINT_TABLES)
    assert generate(tmp_path, check=False, generators=generators) == [
        "a/b/made.md",
        TESTBOT_PAGE,
        "old.md",
    ]
    assert (tmp_path / "old.md").read_text(encoding="utf-8") == "new\n"
    assert (tmp_path / "a/b/made.md").read_text(encoding="utf-8") == "made\n"
    page = (tmp_path / TESTBOT_PAGE).read_bytes()
    assert generate(tmp_path, check=False, generators=generators) == []
    assert generate(tmp_path, check=True, generators=generators) == []
    assert (tmp_path / TESTBOT_PAGE).read_bytes() == page


def test_nothing_is_written_if_a_generator_fails(tmp_path: Path) -> None:
    def fail(root: Path) -> dict[str, str]:
        raise GeneratorError("the source is broken")

    generators = (fake("first", {"first.md": "first\n"}), Generator("fails", "", fail))
    with pytest.raises(GeneratorError, match="the source is broken"):
        generate(tmp_path, check=False, generators=generators)
    assert list(tmp_path.iterdir()) == []


def test_two_generators_cannot_produce_one_file(tmp_path: Path) -> None:
    generators = (fake("first", {"a.md": "a\n"}), fake("second", {"a.md": "b\n"}))
    with pytest.raises(GeneratorError) as caught:
        generate(tmp_path, check=True, generators=generators)
    assert str(caught.value) == (
        "a.md is produced by both 'first' and 'second': a file has one generator,"
        " which may own the whole file or one block in it"
    )


@pytest.mark.parametrize(
    "path", ["/etc/motd", "../outside.md", "docs/../../outside.md", "./a.md", "a//b"]
)
def test_a_generator_writes_only_inside_the_root(tmp_path: Path, path: str) -> None:
    with pytest.raises(GeneratorError) as caught:
        generate(tmp_path, check=False, generators=(fake("files", {path: "x\n"}),))
    assert str(caught.value) == (
        f"generator 'files' produced {path!r}: a path must be relative to the"
        " repository's root, in POSIX form, without '.' or '..'"
    )
    assert list(tmp_path.iterdir()) == []


def test_a_file_that_is_not_utf8_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_bytes(b"\xff\xfe")
    with pytest.raises(GeneratorError, match=r"a\.md: not valid UTF-8: "):
        generate(tmp_path, check=True, generators=(fake("files", {"a.md": "a\n"}),))
