# Embodiverse

[![checks](https://github.com/NakamuraRoboticsLab/embodiverse/actions/workflows/checks.yml/badge.svg)](https://github.com/NakamuraRoboticsLab/embodiverse/actions/workflows/checks.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10 | 3.11 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

Embodiverse is a shared software platform for humanoid robots, built
around the Unitree G1, the EngineAI PM01 and the Booster K1. It gives
everyone the same conventions, contracts and APIs, so code written against
it runs on any of the robots and in simulation without changes, and work
done in one project, such as a policy, an observation term or an adapter,
can be reused in another.

It is a working skeleton to build on: the contracts that every robot,
simulator and policy agree on; one robot port that every robot and
simulator is driven through; an adapter base, a loopback adapter (a
stand-in robot that is also the template for hardware adapters) and a
MuJoCo adapter; a conformance kit that checks adapters; a policy runner
with a safety layer; recording of runs; and draft descriptions of the three
robots, with notes on each vendor's interface.

## Quick start

You need [uv](https://docs.astral.sh/uv/). It installs the right Python and
every dependency at the versions pinned in `uv.lock`:

```bash
uv sync
```

`embodiverse list` shows the robots, adapters and other plugins installed,
from Embodiverse itself and from any other package that registers them:

```console
$ uv run embodiverse list
embodiverse.robot_descriptions
  g1_29dof                 embodiverse.robots.g1_29dof
  k1_22dof                 embodiverse.robots.k1_22dof
  pm01                     embodiverse.robots.pm01
  testbot                  embodiverse.robots.testbot
embodiverse.adapters
  loopback                 embodiverse.adapters.loopback:LoopbackRobot
  loopback_realtime        embodiverse.adapters.loopback:RealtimeLoopbackRobot
  mujoco                   embodiverse.adapters.mujoco:MujocoRobot
  mujoco_realtime          embodiverse.adapters.mujoco:RealtimeMujocoRobot
embodiverse.observations
  base_angular_velocity    embodiverse.runtime.observations:base_angular_velocity
  gait_phase               embodiverse.runtime.observations:gait_phase
  joint_position_relative  embodiverse.runtime.observations:joint_position_relative
  joint_velocity           embodiverse.runtime.observations:joint_velocity
  last_action              embodiverse.runtime.observations:last_action
  projected_gravity        embodiverse.runtime.observations:projected_gravity
  velocity_command         embodiverse.runtime.observations:velocity_command
embodiverse.components
  (none)
```

`embodiverse run` runs a policy on a robot through an adapter. The zero
policy holds every joint at its default position, so this runs testbot, a
small biped made for testing, in MuJoCo through the whole pipeline
(observations, policy, safety layer and adapter), in lockstep, as fast as
the machine allows:

```console
$ uv run embodiverse run --robot testbot --adapter mujoco --policy zero --seconds 5
robot           testbot through mujoco, in lockstep
policy          zero, at 50 Hz
ticks           250
simulated time  5.000 s
rate            50.00 Hz
jitter          0.000 ms
longest tick    0.107 ms
overruns        0
faults          none
```

The loopback adapter runs any robot description, the G1, PM01 and K1
drafts included, on a stand-in robot whose joints follow their targets,
with no model file or simulator:

```bash
uv run embodiverse run --robot g1_29dof --adapter loopback --policy zero --seconds 2
```

`uv run embodiverse --help` lists every command.

## Running the checks

```bash
uv run nox            # formatting, lint, type checks and tests, on Python 3.10–3.12
uv run nox -s lint    # one check; `uv run nox -l` lists them
uv run pytest         # a quick test run while you work
```

`uv run nox` installs the package into a fresh environment for each check,
so it sees what a clean machine would. The first run builds the
environments and takes a few minutes; later runs reuse them.
GitHub Actions runs the same command on every pull request.

Optional git hooks run the checks for you:

```bash
git config core.hooksPath .githooks
```

They run formatting and lint before each commit, and `uv run nox` before
each push and merge commit.

## Repository layout

| Path | What it is |
|---|---|
| `src/embodiverse/contracts/` | Conventions, runtime types, schemas, the robot port and the plugin registry |
| `src/embodiverse/adapters/` | The adapter base, the loopback and MuJoCo adapters |
| `src/embodiverse/runtime/` | Observations, the safety layer, the runner and recording |
| `src/embodiverse/tools/` | The conformance kit and the generators |
| `src/embodiverse/app/` | The `embodiverse` command line |
| `src/embodiverse/robots/` | Built-in robot descriptions and model files |
| `tests/` | The tests, with fixtures in `tests/fixtures/` |
| `docs/` | Documentation |
| `typings/` | Type stubs for libraries that ship none, such as MuJoCo |
| `pyproject.toml` | Package definition, dependencies, plugins and tool settings |
| `noxfile.py` | The checks |
| `.githooks/` | Optional git hooks that run the checks |
| `uv.lock` | Exact versions of every dependency (generated by uv) |
| `AGENTS.md`, `CLAUDE.md` | Guidance for AI coding agents working in the repository |

## Where to go next

- [`docs/architecture.md`](docs/architecture.md): how the platform fits
  together: the layers, the control loop, the conventions, the robot port
  and its modes.
- [`docs/adding-a-robot.md`](docs/adding-a-robot.md): the skeleton guide:
  a robot description, an adapter on the base, the vendor SDKs, a MuJoCo
  model, and the first sessions on the robot.
- [`docs/policies.md`](docs/policies.md): for training and running
  policies: the descriptor, observation terms, the runner, the safety
  layer and recording.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): setting up, branches, checks and
  pull requests.
- The vendor interface notes, [Unitree G1](docs/adapters/unitree-g1.md),
  [EngineAI PM01](docs/adapters/engineai-pm01.md) and
  [Booster K1](docs/adapters/booster-k1.md), and each robot's joint tables
  in [`docs/robots/`](docs/robots/).

## Licence

Apache-2.0; see [`LICENSE`](LICENSE). Facts taken from vendors' public
sources are listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).
