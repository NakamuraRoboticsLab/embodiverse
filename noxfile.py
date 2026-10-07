"""Local CI. `uv run nox` runs every check; `uv run nox -s <name>` runs one."""

import os

import nox

PYTHONS = ["3.10", "3.11", "3.12"]

nox.options.default_venv_backend = "uv"
nox.options.reuse_venv = "yes"
nox.options.error_on_missing_interpreters = True

# Check as a clean machine would: a sourced ROS 2 puts its packages on
# PYTHONPATH, which would let undeclared imports (e.g. rclpy) pass here.
os.environ.pop("PYTHONPATH", None)


def install(session: nox.Session) -> None:
    """Install the locked dependencies and the package itself, non-editable."""
    session.run_install(
        "uv",
        "sync",
        "--locked",
        "--no-editable",
        # uv reuses a cached build of the package unless pyproject.toml or the
        # top level of src/ changed, so edits deeper in src/ would be missed.
        "--reinstall-package=embodiverse",
        f"--python={session.virtualenv.location}",
        env={"UV_PROJECT_ENVIRONMENT": session.virtualenv.location},
    )


@nox.session(python=PYTHONS[0])
def lint(session: nox.Session) -> None:
    """Check formatting and lint rules without changing files."""
    install(session)
    session.run("ruff", "format", "--check")
    session.run("ruff", "check")


@nox.session(python=PYTHONS[0])
def typecheck(session: nox.Session) -> None:
    """Type-check the package and tests."""
    install(session)
    # Name the configuration explicitly: without -p, pyright uses the first
    # pyrightconfig.json it finds in a parent directory, and then checks
    # that directory's files with that directory's settings.
    session.run("pyright", "-p", "pyproject.toml")


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Run the tests against the installed package on every supported Python."""
    install(session)
    session.run("pytest", *session.posargs)
