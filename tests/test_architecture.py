"""The layering rules of docs/architecture.md, checked on every module.

Each module belongs to a layer: the subpackage of `embodiverse` it is in, or
the top-level package for modules directly in it. RULES says what each layer
may import. The checker reads the source with `ast` and imports nothing, so
it sees every import statement, including those inside functions and under
`if TYPE_CHECKING:`. It does not see modules loaded at run time through
`importlib` or entry points, which is how the runtime finds adapters and
components without importing them.
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest

PACKAGE = "embodiverse"
ROOT = Path(__file__).parents[1]
THIS_FILE = Path(__file__).relative_to(ROOT).as_posix()

# What every layer between the contracts and the application may import.
SHARED = frozenset({"numpy", "pydantic", "yaml", "embodiverse.contracts"})

# The layering rules, and the only place to change them. Each layer may
# import the standard library, its own modules, and the names listed here:
# third-party packages by their top-level name, other layers as
# `embodiverse.<layer>`. "anything" lifts every restriction. "data only"
# allows data files and `__init__.py` files without imports, and nothing else.
RULES: dict[str, frozenset[str] | Literal["anything", "data only"]] = {
    # Modules directly in `embodiverse`: the standard library only.
    PACKAGE: frozenset(),
    # Kept small, so the contracts install and import everywhere.
    "contracts": frozenset({"numpy", "pydantic", "yaml"}),
    # The simulator and the vendor SDKs of the lab's robots: the G1's
    # unitree_sdk2py, the K1's booster_robotics_sdk_python, and the PM01's
    # ROS 2 interface, rclpy with the vendor's interface_protocol messages
    # (docs/adding-a-robot.md). Only an adapter imports its own SDK.
    "adapters": SHARED
    | {
        "mujoco",
        "unitree_sdk2py",
        "booster_robotics_sdk_python",
        "rclpy",
        "interface_protocol",
    },
    "runtime": SHARED | {"onnxruntime", "mcap"},
    "tools": SHARED | {"pytest"},
    "components": SHARED,
    "app": "anything",
    "robots": "data only",
}


def layer_of(module: str, is_package: bool) -> str:
    """Return the layer of `module`: its subpackage of `embodiverse`, or PACKAGE."""
    parts = module.split(".")
    in_subpackage = len(parts) > 2 or (is_package and len(parts) == 2)
    return parts[1] if in_subpackage else PACKAGE


def resolve(module: str | None, level: int, package: str) -> str:
    """Return the absolute name of `from <dots><module> import ...` in `package`."""
    if level == 0:
        return module or ""
    parts = package.split(".")
    if level > len(parts):
        # Beyond the top-level package: kept as written, so it is never allowed.
        return "." * level + (module or "")
    base = parts[: len(parts) - level + 1]
    return ".".join([*base, module] if module else base)


def imported_modules(
    tree: ast.Module, module: str, is_package: bool
) -> Iterator[tuple[int, str]]:
    """Yield the line and absolute name of every module that `tree` imports."""
    package = module if is_package else module.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            name = resolve(node.module, node.level, package)
            if name == PACKAGE:
                # `from embodiverse import runtime` imports a layer.
                for alias in node.names:
                    yield node.lineno, f"{PACKAGE}.{alias.name}"
            else:
                yield node.lineno, name


def is_allowed(layer: str, name: str) -> bool:
    """Return whether a module in `layer` may import the module `name`."""
    rule = RULES[layer]
    if rule == "anything":
        return True
    if rule == "data only":
        return False
    parts = name.split(".")
    if parts[0] == PACKAGE:
        key = ".".join(parts[:2])
        return key in rule or (layer != PACKAGE and key == f"{PACKAGE}.{layer}")
    return parts[0] in sys.stdlib_module_names or parts[0] in rule


def check_module(
    module: str, source: str, *, is_package: bool = False, path: str = "<string>"
) -> list[str]:
    """Return the layering violations in `source`, the code of `module`.

    `is_package` says whether the source is the module's `__init__.py`, and
    `path` is used in the messages: `path:line: <layer> may not import <name>`.
    """
    layer = layer_of(module, is_package)
    if layer not in RULES:
        return [f"{path}:1: {layer} is not a layer; add it to RULES in {THIS_FILE}"]
    problems: list[tuple[int, str]] = []
    if RULES[layer] == "data only" and not is_package:
        problems.append((1, f"{layer} may hold only data and __init__.py files"))
    tree = ast.parse(source, filename=path)
    for line, name in imported_modules(tree, module, is_package):
        if not is_allowed(layer, name):
            problems.append((line, f"{layer} may not import {name}"))
    return [f"{path}:{line}: {problem}" for line, problem in sorted(problems)]


def check_package(directory: Path, relative_to: Path) -> list[str]:
    """Return the layering violations in every module under `directory`.

    Paths in the messages are relative to `relative_to`.
    """
    problems: list[str] = []
    for file in sorted(directory.rglob("*.py")):
        parts = file.relative_to(directory.parent).with_suffix("").parts
        is_package = parts[-1] == "__init__"
        problems += check_module(
            ".".join(parts[:-1] if is_package else parts),
            file.read_text(encoding="utf-8"),
            is_package=is_package,
            path=file.relative_to(relative_to).as_posix(),
        )
    return problems


def test_the_package_follows_the_layering_rules() -> None:
    directory = ROOT / "src" / PACKAGE
    assert (directory / "__init__.py").is_file(), f"no package at {directory}"
    problems = check_package(directory, ROOT)
    assert not problems, "\n".join(["layering violations:", *problems])


def test_allows_the_standard_library_shared_packages_and_the_own_layer() -> None:
    source = """\
from __future__ import annotations
import json
from collections.abc import Mapping
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
import yaml
import embodiverse.contracts.types
from embodiverse.contracts.types import Mode
from embodiverse import contracts
from . import types
from .types import Mode
from ..contracts.types import Mode
"""
    assert check_module(f"{PACKAGE}.contracts.robot", source) == []
    assert check_module(f"{PACKAGE}.contracts", source, is_package=True) == []


def test_allows_a_layer_its_own_third_party_packages() -> None:
    source = "import mujoco\nfrom ..common import units\nfrom . import sim\n"
    assert check_module(f"{PACKAGE}.adapters.mujoco.robot", source) == []


@pytest.mark.parametrize(
    ("module", "source", "problem"),
    [
        pytest.param(
            "contracts.robot",
            "import torch",
            "1: contracts may not import torch",
            id="contracts imports torch",
        ),
        pytest.param(
            "contracts.robot",
            "from embodiverse.runtime.loop import run",
            "1: contracts may not import embodiverse.runtime.loop",
            id="contracts imports the runtime",
        ),
        pytest.param(
            "runtime.loop",
            "import json\nimport embodiverse.adapters.mujoco",
            "2: runtime may not import embodiverse.adapters.mujoco",
            id="runtime imports an adapter",
        ),
        pytest.param(
            "adapters.sim",
            "from ..runtime import loop",
            "1: adapters may not import embodiverse.runtime",
            id="relative import out of the layer",
        ),
        pytest.param(
            "adapters.sim",
            "from .. import runtime",
            "1: adapters may not import embodiverse.runtime",
            id="relative import of another layer",
        ),
        pytest.param(
            "adapters.sim",
            "from embodiverse import __version__",
            "1: adapters may not import embodiverse.__version__",
            id="import from the top-level package",
        ),
        pytest.param(
            "adapters.sim",
            "from ... import x",
            "1: adapters may not import ...",
            id="relative import beyond the top-level package",
        ),
        pytest.param(
            "components.estimator",
            "def estimate():\n    import mujoco\n",
            "2: components may not import mujoco",
            id="import inside a function",
        ),
        pytest.param(
            "contracts.robot",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import torch\n",
            "3: contracts may not import torch",
            id="import for type checking only",
        ),
        pytest.param(
            "runtime.loop",
            "import mujoco",
            "1: runtime may not import mujoco",
            id="another layer's third-party package",
        ),
        pytest.param(
            "tools.kit",
            "from embodiverse.components import estimator",
            "1: tools may not import embodiverse.components",
            id="tools import a component",
        ),
    ],
)
def test_catches_a_violation(module: str, source: str, problem: str) -> None:
    assert check_module(f"{PACKAGE}.{module}", source) == [f"<string>:{problem}"]


def test_reports_every_violation_in_line_order() -> None:
    source = "import torch\nimport json\ndef f():\n    import mujoco\nimport scipy\n"
    assert check_module(f"{PACKAGE}.contracts.x", source, path="x.py") == [
        "x.py:1: contracts may not import torch",
        "x.py:4: contracts may not import mujoco",
        "x.py:5: contracts may not import scipy",
    ]


def test_the_top_level_package_may_import_only_the_standard_library() -> None:
    source = "from importlib.metadata import version\nimport numpy\nfrom . import app\n"
    assert check_module(PACKAGE, source, is_package=True) == [
        "<string>:2: embodiverse may not import numpy",
        "<string>:3: embodiverse may not import embodiverse.app",
    ]


def test_the_application_may_import_anything() -> None:
    source = """\
import torch
import embodiverse
from embodiverse.runtime import loop
from embodiverse.adapters.mujoco import MujocoRobot
from ..components import estimator
"""
    assert check_module(f"{PACKAGE}.app.main", source) == []


def test_robots_hold_no_imports() -> None:
    assert (
        check_module(f"{PACKAGE}.robots.g1", '"""The G1."""\n', is_package=True) == []
    )
    assert check_module(f"{PACKAGE}.robots", "import json\n", is_package=True) == [
        "<string>:1: robots may not import json"
    ]


def test_robots_hold_no_modules_but_init() -> None:
    assert check_module(f"{PACKAGE}.robots.loader", "") == [
        "<string>:1: robots may hold only data and __init__.py files"
    ]


def test_an_unknown_subpackage_is_reported() -> None:
    message = "utils is not a layer; add it to RULES in tests/test_architecture.py"
    assert check_module(f"{PACKAGE}.utils.text", "") == [f"<string>:1: {message}"]


def test_checks_a_tree_with_only_some_layers(tmp_path: Path) -> None:
    package = tmp_path / "src" / PACKAGE
    files = {
        "__init__.py": "import importlib.metadata\n",
        "contracts/__init__.py": "",
        "contracts/types.py": "import numpy\n",
        "robots/__init__.py": "",
        "robots/g1/__init__.py": "",
        "robots/g1/description.yaml": "name: g1\n",
    }
    for name, text in files.items():
        (package / name).parent.mkdir(parents=True, exist_ok=True)
        (package / name).write_text(text, encoding="utf-8")
    assert check_package(package, tmp_path) == []

    (package / "contracts/bad.py").write_text("import torch\n", encoding="utf-8")
    (package / "robots/g1/load.py").write_text("import yaml\n", encoding="utf-8")
    assert check_package(package, tmp_path) == [
        "src/embodiverse/contracts/bad.py:1: contracts may not import torch",
        "src/embodiverse/robots/g1/load.py:1: robots may hold only data and"
        " __init__.py files",
        "src/embodiverse/robots/g1/load.py:1: robots may not import yaml",
    ]
