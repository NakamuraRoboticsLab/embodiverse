# Contributing to Embodiverse

Thank you for building on Embodiverse. This page collects the habits that
keep a repository shared by many people pleasant to work in: how to set up,
how to shape a change, and where things go. The
[README](README.md) has the quick start, and [`docs/`](docs/) the
architecture and the guides.

## Setting up

You need git and [uv](https://docs.astral.sh/uv/). In your clone:

```bash
uv sync                                   # Python 3.10 and every dependency, as locked
uv run embodiverse list                   # the robots, adapters and terms installed
uv run pytest                             # the tests, in seconds
uv run nox                                # every check; a few minutes the first time
```

`uv sync` installs Embodiverse in editable mode with the development tools,
MuJoCo, onnxruntime and mcap. The vendor SDKs are optional extras
([Adding a robot](docs/adding-a-robot.md#vendor-sdks)).

**If your shell sources ROS 2** (for example `/opt/ros/humble/setup.bash`),
ROS 2's Python packages are on `PYTHONPATH`. They can hide a missing
dependency: an import of `rclpy` works on your machine and fails on every
machine without ROS 2. `uv run nox` removes `PYTHONPATH` for every check;
when you run a tool yourself, you can do the same for one command:

```bash
env -u PYTHONPATH uv run pytest
env -u PYTHONPATH uv run pyright -p pyproject.toml
```

The PM01's SDK tests are the exception: they need ROS 2 on `PYTHONPATH`.

## Branches and commits

Name a branch `<type>/<topic>`, such as `feat/unitree-g1-adapter` or
`docs/k1-notes`, and keep it to one coherent change:

```bash
git switch -c feat/my-topic main
```

Small commits, each making one change with its tests and docs, are easier
to review and to revert. We recommend
[Conventional Commits](https://www.conventionalcommits.org/) for the
messages: a first line `type: summary` in the imperative, at most 72
characters, with a type such as `feat`, `fix`, `docs`, `test`, `build`,
`ci`, `refactor` or `chore`; then, after a blank line, a body that says
*why*. The diff already shows what changed; the reason is what a reader
months later will need.

```text
test: allow for shared machines in the conformance timing checks

The kit's 1 ms budget for read and write is meant for a robot's own
computer. On shared development machines the scheduler can pause a
thread for a few milliseconds, so these tests use 5 ms, which still
shows that read and write never wait for the physics.
```

## Checks

Run `uv run nox` before opening a pull request: formatting and lint
(ruff), types (pyright, strict for the contracts) and the tests on Python
3.10, 3.11 and 3.12. GitHub Actions runs the same command on every pull
request, so a green run locally means a green run there.

```bash
uv run nox -l                 # list the checks
uv run nox -s lint            # one of them
uv run ruff format            # fix the formatting
uv run ruff check --fix       # fix what lint can fix
```

The optional git hooks do this for you: `git config core.hooksPath
.githooks` runs formatting and lint before each commit, and `uv run nox`
before each push and merge commit.

Three tests check rules that span the repository, and each failure says
what to do: the layering test (`tests/test_architecture.py`, see
[architecture](docs/architecture.md#layers)), the generated files test
(run `uv run embodiverse generate` after changing a robot description),
and the conformance kit that every adapter's tests run.

## Pull requests and reviews

Push your branch and open a pull request against `main`. The template asks
what changes and why, and how you tested it. Draft pull requests are
welcome early, when you would like a second pair of eyes on a direction.

A review is a conversation about the code, not a verdict on its author.
When you review, say what you noticed and why it matters, and tell small
suggestions apart from things that need to change. When you are reviewed,
it is fine to disagree and explain.

## Keeping a branch up to date

Until a branch is pushed, rebase it on `main` as you like. Once others
may have it, merge `main` into it instead, so their copies stay valid:

```bash
git fetch origin
git merge origin/main
```

## Where things go

| What | Where |
|---|---|
| Conventions, runtime types, schemas, the robot port, the registry | `src/embodiverse/contracts/` |
| An adapter for one of the lab's robots | `src/embodiverse/adapters/<vendor>_<model>/` |
| A robot description and its model files | `src/embodiverse/robots/<name>/` |
| Observation terms, the safety layer, the runner, recording | `src/embodiverse/runtime/` |
| The conformance kit, the generators | `src/embodiverse/tools/` |
| Tests, mirroring the package | `tests/` |
| Vendor interface notes; each robot's page | `docs/adapters/`; `docs/robots/` |
| Policies, experiments, project-specific code | your project's own repository, as plugins ([Adding a robot](docs/adding-a-robot.md#from-your-own-package)) |

Code that only one project uses is happiest in that project; when a second
project needs it, it can move here under the same plugin name.

## Changing the shared contracts

The contracts in `src/embodiverse/contracts/` are what everyone's code is
written against: the robot port, the runtime types, the description and
descriptor schemas, the observation term interface and the plugin groups.
A change there can ripple into every adapter, policy and project. Before
changing one, have a chat with the maintainer, or open an issue or a draft
pull request, so we can find a way that keeps everyone's code working.
A new robot, adapter, observation term or other plugin that follows the
existing contracts can go straight to a pull request.

## Issues

The issue forms cover a bug report, a feature request and a new robot.
For a bug, the smallest command that shows it, ideally on testbot in
simulation, makes it quickest to fix; a recording made with `--record`
helps for problems during a run.

## Licence

Embodiverse is licensed under the Apache License 2.0 ([LICENSE](LICENSE)).
Unless you say otherwise, what you contribute is licensed under the same
terms (section 5 of the licence). Third-party files, such as a vendor's
robot model, can be added where their licence allows it, with their
source, version and licence listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
