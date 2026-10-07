"""Tests of the plugin registry.

Most tests replace the installed entry points with fakes, so they do not
depend on what else is installed. Modules that fake plugins import are
written to a temporary directory on sys.path.
"""

import shutil
import sys
import zipfile
from collections.abc import Callable, Iterator
from importlib.metadata import Distribution, EntryPoint
from importlib.resources import as_file, files
from pathlib import Path

import pytest

from embodiverse.adapters.mujoco import MujocoRobot, RealtimeMujocoRobot
from embodiverse.contracts import registry
from embodiverse.contracts.registry import (
    ADAPTERS,
    COMPONENTS,
    GROUPS,
    OBSERVATIONS,
    ROBOT_DESCRIPTIONS,
    RegistryError,
    entry_points_in,
    load,
    robot_description,
)
from embodiverse.contracts.robot import Robot

Install = Callable[..., None]


def entry_point(name: str, value: str, group: str = ADAPTERS) -> EntryPoint:
    return EntryPoint(name=name, value=value, group=group)


def distribution(directory: Path, name: str, entry_points: str) -> Distribution:
    """Return a distribution in `directory` declaring `entry_points`.

    `entry_points` is the text of its entry_points.txt, which is what
    installers write from a pyproject.toml's entry-point tables.
    """
    info = directory / f"{name.replace('-', '_')}-1.0.dist-info"
    info.mkdir()
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    (info / "METADATA").write_text(metadata, encoding="utf-8")
    (info / "entry_points.txt").write_text(entry_points, encoding="utf-8")
    return Distribution.at(info)


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Return a function that makes its arguments the only installed entry points."""

    def install(*installed: EntryPoint) -> None:
        def select(*, group: str) -> list[EntryPoint]:
            return [item for item in installed if item.group == group]

        monkeypatch.setattr(registry, "entry_points", select)

    return install


@pytest.fixture
def plugin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Return a directory on sys.path; modules imported from it are unloaded after."""
    monkeypatch.syspath_prepend(str(tmp_path))
    yield tmp_path
    for name, module in list(sys.modules.items()):
        if str(getattr(module, "__file__", None)).startswith(str(tmp_path)):
            del sys.modules[name]


def test_group_names() -> None:
    # Plugins in other repositories name these groups, so they never change.
    assert GROUPS == (
        "embodiverse.robot_descriptions",
        "embodiverse.adapters",
        "embodiverse.observations",
        "embodiverse.components",
    )
    assert GROUPS == (ROBOT_DESCRIPTIONS, ADAPTERS, OBSERVATIONS, COMPONENTS)


def test_the_built_in_plugins_are_registered() -> None:
    # nox installs the package from a built wheel, so this checks its metadata.
    testbot = entry_points_in(ROBOT_DESCRIPTIONS)["testbot"]
    adapters = entry_points_in(ADAPTERS)
    assert testbot.value == "embodiverse.robots.testbot"
    assert adapters["mujoco"].value == "embodiverse.adapters.mujoco:MujocoRobot"
    assert (
        adapters["mujoco_realtime"].value
        == "embodiverse.adapters.mujoco:RealtimeMujocoRobot"
    )
    for item in (testbot, adapters["mujoco"], adapters["mujoco_realtime"]):
        assert item.dist is not None
        assert item.dist.name == "embodiverse"


def test_the_built_in_adapters_load_and_make_robots() -> None:
    assert load(ADAPTERS, "mujoco") is MujocoRobot
    assert load(ADAPTERS, "mujoco_realtime") is RealtimeMujocoRobot
    testbot = robot_description("testbot")
    for name in ("mujoco", "mujoco_realtime"):
        make_robot = load(ADAPTERS, name)
        assert callable(make_robot)
        assert isinstance(make_robot(testbot), Robot)


def test_robot_description_loads_testbot() -> None:
    testbot = robot_description("testbot")
    assert testbot.name == "testbot"
    mjcf = testbot.model_path("mjcf")
    assert mjcf is not None
    assert mjcf.is_file()


def test_entry_points_are_listed_by_name_in_their_group(install: Install) -> None:
    install(
        entry_point("sim", "lab_sim:SimRobot"),
        entry_point("g1", "lab_robots.adapter:G1Robot"),
        entry_point("g1", "lab_robots.g1", group=ROBOT_DESCRIPTIONS),
    )
    adapters = entry_points_in(ADAPTERS)
    assert list(adapters) == ["g1", "sim"]
    assert adapters["g1"].value == "lab_robots.adapter:G1Robot"
    assert list(entry_points_in(ROBOT_DESCRIPTIONS)) == ["g1"]
    assert entry_points_in(COMPONENTS) == {}


def test_listing_imports_nothing(plugin_dir: Path, install: Install) -> None:
    (plugin_dir / "explodes.py").write_text(
        "raise RuntimeError('imported')\n", encoding="utf-8"
    )
    install(entry_point("explodes", "explodes:Robot"))
    assert list(entry_points_in(ADAPTERS)) == ["explodes"]
    assert "explodes" not in sys.modules


def test_load_returns_the_named_object(plugin_dir: Path, install: Install) -> None:
    (plugin_dir / "lab_sim.py").write_text(
        "class SimRobot:\n    pass\n", encoding="utf-8"
    )
    install(entry_point("sim", "lab_sim:SimRobot"))
    loaded = load(ADAPTERS, "sim")
    assert loaded is sys.modules["lab_sim"].SimRobot


def test_an_unknown_name_lists_the_available_ones(install: Install) -> None:
    install(entry_point("sim", "lab_sim:SimRobot"), entry_point("g1", "lab:G1"))
    with pytest.raises(RegistryError) as error:
        load(ADAPTERS, "pm01")
    assert str(error.value) == (
        "no 'pm01' in group 'embodiverse.adapters'; available: g1, sim"
    )


def test_an_unknown_name_in_an_empty_group(install: Install) -> None:
    install()
    with pytest.raises(RegistryError) as error:
        load(COMPONENTS, "lidar_localizer")
    assert str(error.value) == (
        "no 'lidar_localizer' in group 'embodiverse.components': the group is"
        " empty; install the package that provides it"
    )


def test_a_duplicate_name_names_both_distributions(
    tmp_path: Path, install: Install
) -> None:
    table = "[embodiverse.adapters]\ng1 = {}\n"
    first = distribution(tmp_path, "lab-robots", table.format("lab_robots:G1Robot"))
    second = distribution(tmp_path, "g1-tools", table.format("g1_tools:G1"))
    install(*first.entry_points, *second.entry_points)
    with pytest.raises(RegistryError) as error:
        entry_points_in(ADAPTERS)
    assert str(error.value) == (
        "'g1' in group 'embodiverse.adapters' is registered by lab-robots 1.0 and"
        " g1-tools 1.0; names must be unique across installed packages: uninstall"
        " one of them, or rename the entry point in one of them and reinstall it"
    )
    with pytest.raises(RegistryError, match="registered by lab-robots 1.0 and"):
        load(ADAPTERS, "g1")


def test_duplicates_from_unknown_distributions(install: Install) -> None:
    # Entry points built in code, not read from metadata, have no distribution.
    install(*(entry_point("g1", f"lab_{index}:G1") for index in range(3)))
    unknown = "unknown distribution"
    with pytest.raises(
        RegistryError, match=f"is registered by {unknown}, {unknown} and {unknown};"
    ):
        entry_points_in(ADAPTERS)


def test_a_duplicate_in_another_group_is_not_an_error(install: Install) -> None:
    install(
        entry_point("g1", "lab:G1"), entry_point("g1", "lab.g1", ROBOT_DESCRIPTIONS)
    )
    assert list(entry_points_in(ADAPTERS)) == ["g1"]


@pytest.mark.parametrize(
    ("value", "cause"),
    [
        pytest.param("lab_no_such_module:Robot", ModuleNotFoundError, id="no module"),
        pytest.param("json:NoSuchRobot", AttributeError, id="no attribute"),
    ],
)
def test_a_failed_import_is_a_registry_error(
    install: Install, value: str, cause: type[Exception]
) -> None:
    install(entry_point("broken", value))
    with pytest.raises(RegistryError) as error:
        load(ADAPTERS, "broken")
    assert str(error.value).startswith(
        f"cannot load 'broken' from group 'embodiverse.adapters': importing {value!r}"
        f" failed with {cause.__name__}: "
    )
    assert isinstance(error.value.__cause__, cause)


def test_an_error_while_importing_is_chained(
    plugin_dir: Path, install: Install
) -> None:
    (plugin_dir / "explodes.py").write_text(
        "raise RuntimeError('no SDK here')\n", encoding="utf-8"
    )
    install(entry_point("explodes", "explodes:Robot"))
    with pytest.raises(RegistryError, match="failed with RuntimeError: no SDK here$"):
        load(ADAPTERS, "explodes")


def write_robot_package(directory: Path, name: str) -> Path:
    """Write a package holding a copy of testbot's description and model."""
    package = directory / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    testbot = files("embodiverse.robots.testbot")
    for file in ("description.yaml", "testbot.xml"):
        with as_file(testbot / file) as source:
            shutil.copyfile(source, package / file)
    return package


def test_robot_description_reads_the_package_s_file(
    plugin_dir: Path, install: Install
) -> None:
    package = write_robot_package(plugin_dir, "lab_testbot")
    install(entry_point("lab_testbot", "lab_testbot", ROBOT_DESCRIPTIONS))
    description = robot_description("lab_testbot")
    assert description.name == "testbot"
    assert description.model_path("mjcf") == package / "testbot.xml"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("embodiverse.contracts.hashing", id="a module"),
        pytest.param("embodiverse.contracts.registry:load", id="an object"),
    ],
)
def test_robot_description_needs_a_package(install: Install, value: str) -> None:
    install(entry_point("lab_robot", value, ROBOT_DESCRIPTIONS))
    with pytest.raises(RegistryError) as error:
        robot_description("lab_robot")
    assert str(error.value) == (
        f"robot description 'lab_robot' names {value!r}, which is not a package:"
        " it must name the package that contains description.yaml"
    )


def test_robot_description_needs_a_description_file(install: Install) -> None:
    install(entry_point("robots", "embodiverse.robots", ROBOT_DESCRIPTIONS))
    with pytest.raises(
        RegistryError,
        match=r"^robot description 'robots': package 'embodiverse\.robots' has no"
        r" description\.yaml \(looked for .*description\.yaml\)$",
    ):
        robot_description("robots")


def test_robot_description_refuses_a_zipped_package(
    plugin_dir: Path, install: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = write_robot_package(plugin_dir / "build", "zipped_robot")
    archive = plugin_dir / "robots.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for file in sorted(package.iterdir()):
            zipped.write(file, f"zipped_robot/{file.name}")
    monkeypatch.syspath_prepend(str(archive))
    install(entry_point("zipped_robot", "zipped_robot", ROBOT_DESCRIPTIONS))
    with pytest.raises(
        RegistryError,
        match=r"^robot description 'zipped_robot': package 'zipped_robot' is not"
        r" installed as a directory \(.*\); install it as a directory, not as a"
        r" zip file, so that its model files can be read$",
    ):
        robot_description("zipped_robot")
