"""The plugin registry: robots, adapters and other plugins, found by name.

Plugins are found through entry points. An entry point is a name that an
installed distribution declares in its metadata, in a group, together with
the object it refers to, written `module` or `module:attribute`. A package
in any repository registers a plugin by declaring one in its
`pyproject.toml`; this adapter is then found as `my_robot` in the
`embodiverse.adapters` group once the package is installed:

    [project.entry-points."embodiverse.adapters"]
    my_robot = "my_package.adapter:MyRobot"

Nothing in Embodiverse lists the plugins, so adding one never means editing
the platform. The groups are the constants below, and GROUPS lists them.

Finding plugins reads metadata only. `entry_points_in` lists a group without
importing anything, and `load` imports a single plugin when it is used, so
listing what is installed never imports an adapter or its SDK. A name must
be unique within its group across every installed distribution: two
distributions registering the same name is an error, rather than one
silently shadowing the other.

docs/adding-a-robot.md shows how to register plugins, and
docs/architecture.md why they are found this way.
"""

from importlib.metadata import EntryPoint, entry_points
from importlib.resources import files
from pathlib import Path
from types import ModuleType

from embodiverse.contracts.description import RobotDescription, load_description

__all__ = [
    "ADAPTERS",
    "COMPONENTS",
    "GROUPS",
    "OBSERVATIONS",
    "ROBOT_DESCRIPTIONS",
    "RegistryError",
    "entry_points_in",
    "load",
    "robot_description",
]

ROBOT_DESCRIPTIONS = "embodiverse.robot_descriptions"
"""Robot descriptions, loaded with `robot_description`.

An entry point names a package, not an object in it. The package contains a
`description.yaml` and the model files it names, for example
`testbot = "embodiverse.robots.testbot"`.
"""

ADAPTERS = "embodiverse.adapters"
"""Adapters for robots and simulators.

An entry point names a callable that takes a `RobotDescription` and returns
a `Robot`, usually the adapter class, for example
`mujoco = "embodiverse.adapters.mujoco:MujocoRobot"`.
"""

OBSERVATIONS = "embodiverse.observations"
"""Observation terms, which compute a policy's inputs from the robot's state.

An entry point names a `TermFactory` (see `embodiverse.contracts.observation`):
a callable that takes a term's entry in a policy descriptor, the robot
description and the descriptor, and returns the `Term` that computes it.
The entry point's name is the term's name in descriptors, for example
`projected_gravity = "embodiverse.runtime.observations:projected_gravity"`.
"""

COMPONENTS = "embodiverse.components"
"""Components: policies, estimators, localizers and other capabilities.

What an entry point in this group names is not defined yet.
"""

GROUPS = (ROBOT_DESCRIPTIONS, ADAPTERS, OBSERVATIONS, COMPONENTS)
"""Every group, in the order `embodiverse list` shows them."""

_DESCRIPTION_FILE = "description.yaml"


class RegistryError(LookupError):
    """A plugin is not installed, cannot be loaded, or shares its name."""


def entry_points_in(group: str) -> dict[str, EntryPoint]:
    """Return the entry points registered in `group`, by name, sorted by name.

    Reads the installed distributions' metadata and imports nothing. Raises
    RegistryError if two entry points in the group have the same name.
    """
    found: dict[str, list[EntryPoint]] = {}
    for entry_point in entry_points(group=group):
        found.setdefault(entry_point.name, []).append(entry_point)
    for name, candidates in sorted(found.items()):
        if len(candidates) > 1:
            providers = _join([_distribution_of(item) for item in candidates])
            raise RegistryError(
                f"{name!r} in group {group!r} is registered by {providers}; names"
                " must be unique across installed packages: uninstall one of them,"
                " or rename the entry point in one of them and reinstall it"
            )
    return {name: candidates[0] for name, candidates in sorted(found.items())}


def load(group: str, name: str) -> object:
    """Import and return the object registered as `name` in `group`.

    Raises RegistryError if no entry point in the group has that name, naming
    the ones that exist, or if importing it fails, with the original error
    chained as its cause.
    """
    return _load(_find(group, name))


def robot_description(name: str) -> RobotDescription:
    """Load the robot description registered as `name`.

    The entry point names a package. Its `description.yaml` is read with
    `load_description`, so relative model paths resolve next to the file.
    The package must be installed as a directory, not inside a zip file,
    because the model files have to exist as files.

    Raises RegistryError if the entry point cannot be found or loaded, does
    not name a package, or names one that is not a directory or has no
    `description.yaml`. Raises DescriptionError, from `load_description`,
    if the file is invalid.
    """
    entry_point = _find(ROBOT_DESCRIPTIONS, name)
    package = _load(entry_point)
    if not (isinstance(package, ModuleType) and hasattr(package, "__path__")):
        raise RegistryError(
            f"robot description {name!r} names {entry_point.value!r}, which is not"
            f" a package: it must name the package that contains {_DESCRIPTION_FILE}"
        )
    file = files(package).joinpath(_DESCRIPTION_FILE)
    if not isinstance(file, Path):
        raise RegistryError(
            f"robot description {name!r}: package {package.__name__!r} is not"
            f" installed as a directory ({file}); install it as a directory, not as"
            " a zip file, so that its model files can be read"
        )
    if not file.is_file():
        raise RegistryError(
            f"robot description {name!r}: package {package.__name__!r} has no"
            f" {_DESCRIPTION_FILE} (looked for {file})"
        )
    return load_description(file)


def _find(group: str, name: str) -> EntryPoint:
    """Return the entry point registered as `name` in `group`."""
    available = entry_points_in(group)
    if name in available:
        return available[name]
    if not available:
        raise RegistryError(
            f"no {name!r} in group {group!r}: the group is empty; install the"
            " package that provides it"
        )
    raise RegistryError(
        f"no {name!r} in group {group!r}; available: {', '.join(available)}"
    )


def _load(entry_point: EntryPoint) -> object:
    """Import and return the object `entry_point` names."""
    try:
        return entry_point.load()
    except Exception as error:
        # Importing a plugin runs its code, which can raise anything.
        raise RegistryError(
            f"cannot load {entry_point.name!r} from group {entry_point.group!r}:"
            f" importing {entry_point.value!r} failed with"
            f" {type(error).__name__}: {error}"
        ) from error


def _distribution_of(entry_point: EntryPoint) -> str:
    """Return the name and version of the distribution that declares `entry_point`."""
    distribution = entry_point.dist
    if distribution is None:
        return "unknown distribution"
    return f"{distribution.name} {distribution.version}"


def _join(items: list[str]) -> str:
    """Return two or more `items` joined as in a sentence: `a, b and c`."""
    return f"{', '.join(items[:-1])} and {items[-1]}"
