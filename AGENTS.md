# AGENTS.md

Guidance for AI coding agents working in this repository. People
contributing by hand: see [CONTRIBUTING.md](CONTRIBUTING.md).

## The project

- Embodiverse is a shared platform for the lab's humanoid robots (Unitree G1,
  EngineAI PM01, Booster K1): contracts every robot, simulator and policy
  agree on, one robot port, adapters, and a policy runner with a safety layer.
- Read [docs/architecture.md](docs/architecture.md) before changing code.
  Robots and adapters: [docs/adding-a-robot.md](docs/adding-a-robot.md).
  Policies: [docs/policies.md](docs/policies.md).
- Code: `src/embodiverse/{contracts,adapters,runtime,tools,app,robots}`.
  Tests mirror it under `tests/`.

## Commands

- Set up: `uv sync`. Vendor SDKs: `uv sync --extra g1`, `uv sync --extra k1`.
- Fast check after every change: `uv run pytest -q`, or just the file you
  touched: `uv run pytest tests/<path> -q`.
- Full check before finishing a task: `uv run nox` (format, lint, types,
  tests on Python 3.10–3.12).
- Fix formatting and lint: `uv run ruff format && uv run ruff check --fix`.
- After changing a robot description: `uv run embodiverse generate`.
- A shell that sources ROS 2 puts ROS packages on `PYTHONPATH`, which can hide
  a missing dependency. For ad-hoc tools use `env -u PYTHONPATH uv run ...`;
  `uv run nox` already does this.

## How to work

- Read the relevant docs, the module you are changing, its docstrings and
  its tests before writing code. Follow the patterns already in the
  codebase before introducing new ones.
- Keep each change small and focused on one concern. No unrelated
  refactors, renames or reformatting.
- Prefer the simplest solution that works.
- If the task is ambiguous, ask.

## Tests

- Add or update tests for every change in behaviour.
- When fixing a bug, add a test that reproduces it.
- Test behaviour through public interfaces rather than implementation
  details.
- Do not disable, skip or weaken tests to get a passing run, and do not
  loosen the safety layer's checks or the conformance kit's tolerances. If
  a test is wrong, fix it and explain why in the commit.
- Keep tests deterministic. Fix a flaky test rather than rerunning it.
- Run `uv run nox` before finishing, and report what you ran and the result.

## Rules the code depends on

- **Layers.** `contracts` imports only the standard library, numpy, pydantic
  and pyyaml. `adapters`, `runtime` and `tools` import `contracts`, themselves
  and their own allowed libraries, never each other. `app` may import
  anything. `robots` holds data only. `tests/test_architecture.py` checks it.
- **Conventions at every interface.** SI units; REP-103 axes (x forward,
  y left, z up); quaternions in (x, y, z, w) order; snake_case canonical
  joint names; arrays in the robot description's joint order. Vendor formats
  are converted only inside adapters.
- **Adapters** build on `src/embodiverse/adapters/base.py`: copy
  `adapters/loopback/`, implement the transport, test the codec on hand-made
  messages, and run the conformance kit (`embodiverse.tools.conformance`).
- **Contracts are shared.** The robot port, runtime types, description and
  descriptor schemas and plugin groups are what everyone's code is written
  against. Propose a change to them; do not make it on the side.
- **Vendor SDKs** are imported only inside their adapter, behind an optional
  extra. Tests that need an SDK use `pytest.importorskip`.
- **Dependencies.** Do not add one unless the task needs it. Never edit
  `uv.lock` by hand.
- **Robot facts.** Never invent limits, gains, rates, mounts or vendor APIs.
  Take each from a source you cite, or mark it `# verify on unit`.

## No slop

- Comments say why, only where it is not obvious. No comments that restate
  the code, narrate what you did, or talk to the reader ("This function...",
  "Now we...").
- Docstrings: a one-line summary, then what a caller needs (units, frames,
  errors). No filler.
- No placeholder code, TODO stubs, commented-out code, dead code or unused
  helpers.
- No files nobody asked for: no summaries, reports, plans, notes or extra
  READMEs. Documentation lives in `README.md`, `CONTRIBUTING.md` and `docs/`;
  update the existing page when behaviour changes.
- No emoji. No marketing words ("robust", "powerful", "seamless",
  "comprehensive").
- No speculative generality: no abstraction, option or hook for a use that
  does not exist yet.
- Error messages say what was wrong and what was expected. Never swallow an
  exception.

## Commits and branches

- Branch names: `<type>/<topic>`, for example `feat/unitree-g1-adapter`.
- Conventional Commits: `type: summary`, imperative, at most 72 characters,
  and a body that says why.
- One logical change per commit, with its tests and docs. `uv run nox`
  passes before you commit.
- Never use `--no-verify`. Do not push, merge, rewrite pushed history or
  delete branches unless you are asked to.
- Do not commit generated or local files: recordings, environments, caches.

## Robots

- Agents never command a real robot and never change a robot's network.
  Code that drives hardware is run by a person.
- Test with the loopback adapter, MuJoCo, or a vendor's simulator.

## When unsure

- Ask rather than guess, above all about the contracts, safety behaviour and
  vendor facts.
- When you finish, say what you verified and what you did not.
