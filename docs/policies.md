# Policies

A trained policy is a file of weights. To run it, the platform also needs
to know exactly how its input is built, how its output becomes joint
commands, at what rate it runs and which robot it was trained for. Those
facts usually live in a training repository's code, and get lost or
copied wrongly on the way to the robot. Nothing fails when they do: the
robot just behaves differently from the simulation.

So every policy ships with a **policy descriptor**, one YAML file next to
the ONNX file that states those facts. The runner reads the descriptor,
checks it against the robot, builds the policy's observations from it,
runs the policy at its rate, and passes every command through the safety
layer before it reaches the robot. This page covers each of those, how to
run a policy from the command line, and how to record a run and read it
back.

## Quick start

The zero policy outputs zeros, which holds every joint at its default
position. It needs no file, so it is the way to try a robot, an adapter or
a descriptor through the whole pipeline:

```bash
uv run embodiverse run --robot testbot --adapter mujoco --policy zero --seconds 5
```

The same, shaped by a descriptor, building every observation term it names
(this one is a test fixture for testbot), and recorded:

```bash
uv run embodiverse run --robot testbot --adapter mujoco --policy zero \
    --descriptor tests/fixtures/policies/testbot_zero/descriptor.yaml --seconds 5 \
    --record testbot-zero.mcap
```

A trained policy runs from its descriptor, here walking forward at 0.3 m/s:

```bash
uv run embodiverse run --robot testbot --adapter mujoco \
    --descriptor path/to/descriptor.yaml --command 0.3 0 0 --seconds 20
```

## The policy descriptor

The descriptor of a policy that drives testbot's six leg joints at 50 Hz
(`tests/fixtures/policies/testbot_zero/descriptor.yaml`):

```yaml
schema_version: 1

policy:
  file: policy.onnx               # relative to this file
  sha256: "sha256:f86928da6a2a0c014bea833abf208a2bad3598b1213f702d13b53957b3f93512"

robot:
  name: testbot
  description_hash: "sha256:a336b65eb10734d1c906313c8961046bbf4856da43ed8721fb9a366e1eefcaf0"

rate_hz: 50.0                     # 10 physics steps of 2 ms

command:
  type: velocity
  ranges:                         # what the policy was trained to follow
    lin_vel_x: [-0.5, 0.5]        # m/s, forward
    lin_vel_y: [0.0, 0.0]         # testbot cannot step sideways
    ang_vel_z: [0.0, 0.0]         # rad/s; nor turn

# One frame: 3 + 3 + 3 + 6 + 6 + 6 = 27 values, in this order.
observations:
  - term: base_angular_velocity
    scale: 0.25
  - term: projected_gravity       # scale defaults to 1
  - term: velocity_command
  - term: joint_position_relative
    joints: legs                  # a joint group of the description
  - term: joint_velocity
    joints: legs
    scale: 0.05
  - term: last_action
observation_size: 27

action:
  type: position_offset           # target = default position + 0.25 × output
  joints: legs
  scale: 0.25

gains:
  source: description             # or explicit, with the training gains
```

`tests/fixtures/policies/g1_velocity/descriptor.yaml` is a larger example:
Unitree's reference walking policy on the 29-joint G1, with explicit gains,
a gait phase, and the waist and arms held.

### Fields

A **joint selection** is either the name of a joint group of the
description, standing for its members in order, or a list of joint names,
in the order given, such as `[left_knee]`.

| Field | Type and unit | Default | Meaning |
|---|---|---|---|
| `schema_version` | `1` | required | The schema version. |
| `policy.file` | path | required | The policy file, relative to the descriptor. |
| `policy.sha256` | `sha256:` and 64 hex digits | required | SHA-256 of the file's bytes. |
| `policy.format` | `onnx` | `onnx` | The file format. |
| `robot.name` | string | required | The robot description's `name`. |
| `robot.description_hash` | `sha256:` and 64 hex digits | required | The content hash of the description the policy was trained against. |
| `rate_hz` | > 0, Hz | required | One observation, one output and one command per tick. |
| `command.type` | `velocity` | required | The policy follows a base velocity. |
| `command.ranges.lin_vel_x`, `lin_vel_y`, `ang_vel_z` | `[lower, upper]`, m/s and rad/s | required | The commands it was trained on, in the base frame. |
| `observations[].term` | snake_case | required | An observation term, by name. |
| `observations[].scale` | number, or one per value | `1.0` | Multiplies the term's output. |
| `observations[].joints` | joint selection | none | For a per-joint term, the joints in the policy's order. |
| `observations[].params` | mapping | `{}` | The term's parameters, such as `{period: 0.8}`. |
| `observation_size` | integer > 0 | required | Values in one frame; checked against the terms. |
| `history.length` | integer ≥ 1 | `1` | Frames the policy sees, the current one included. |
| `history.order` | `oldest_first` | `oldest_first` | Frames run from oldest to current. |
| `history.layout` | `per_term` or `per_frame` | `per_term` | How the frames are laid out (below). |
| `action.type` | `position_offset` | required | Each output is an offset from a joint's default position. |
| `action.joints` | joint selection | required | The joints the policy drives; output i drives the i-th. |
| `action.scale` | number, or one per joint; rad per unit | required | Target = default position + scale × output. |
| `gains` | mapping | required | Where the driven joints' gains come from. |
| `gains.source` | `description` or `explicit` | `description` | The description's gains, or the training gains in `values`. |
| `gains.values` | joint → `{kp, kd}` | `{}` | With `explicit`: gains for every driven joint, and no other. |
| `held_joints` | joint selection | none | Joints held at their default position with the description's gains. |
| `provenance` | `external`, or a run manifest | `external` | Where the policy comes from. |

`gains` has no default, so every descriptor says where its gains come
from: gains that differ between training and the robot are a common cause
of policies that only work in simulation. Every joint of the description
is either driven or held, so every joint is commanded on every tick.

### Observation history

One frame is the terms, each scaled, concatenated in order. With
`history.length` above 1, the policy's input holds the last frames, oldest
first, laid out per term or per frame. For a term `a` of two values, a term
`b` of one, and `length: 3`, at tick t:

| Index | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| `per_term` | a₁(t−2) | a₂(t−2) | a₁(t−1) | a₂(t−1) | a₁(t) | a₂(t) | b(t−2) | b(t−1) | b(t) |
| `per_frame` | a₁(t−2) | a₂(t−2) | b(t−2) | a₁(t−1) | a₂(t−1) | b(t−1) | a₁(t) | a₂(t) | b(t) |

`per_term` is how Isaac Lab lays out terms that have a history. On the
first tick, and after a reset, the whole history is filled with copies of
the first frame, as Isaac Lab's history buffers do.

### From output to joint command

On every tick the runner builds one command for every joint of the
description:

| Joint | `position` | `velocity`, `effort` | `kp`, `kd` |
|---|---|---|---|
| The i-th of `action.joints` | default position + scale(i) × output(i) | 0 | its entry in `gains.values` if explicit, otherwise the description's |
| A held joint | default position | 0 | the description's |

### Checks

A policy runs only after three checks pass, each raising
`DescriptorError` with a message that names what is wrong:

- **On loading,** `load_descriptor` checks the schema: unknown or missing
  keys, non-finite numbers, malformed hashes, ranges out of order, scale
  lists of the wrong length and inconsistent gains.
- **Against the robot,** `validate_against(descriptor, description)` lists
  every problem at once: another robot's name or content hash, a rate
  above the robot's `control_hz_max` or that does not divide its
  `physics_hz` evenly, unknown joints or groups, and joints driven twice, both
  driven and held, or neither.
- **Against the file,** `verify_policy_file(descriptor)` hashes the policy
  file and compares it with `policy.sha256`, so a different file is never
  loaded.

The two hashes, for a policy trained on testbot:

```bash
echo "sha256:$(sha256sum tests/fixtures/policies/testbot_zero/policy.onnx | cut -d ' ' -f 1)"
uv run python -c 'from embodiverse.contracts.hashing import content_hash; from embodiverse.contracts.registry import robot_description; print(content_hash(robot_description("testbot")))'
```

A policy built into the platform, such as the zero policy, has no file:
its descriptor leaves `policy` out and is otherwise checked the same way.

## Observations

The `ObservationBuilder` (`embodiverse.runtime.observations`) turns the
descriptor's terms into the policy's input on every tick, as a new float32
array. Each term has one exact definition, checked in the tests against
values computed by hand, because a term with the wrong frame, sign or
joint order still gives plausible numbers.

| Term | Size | `joints` | Definition |
|---|---|---|---|
| `base_angular_velocity` | 3 | no | The IMU's angular velocity, rotated into the base frame through the IMU's mount, in rad/s. |
| `projected_gravity` | 3 | no | The unit vector of gravity in the base frame: (0, 0, −1) when level. |
| `velocity_command` | 3 | no | The velocity command (lin_vel_x, lin_vel_y, ang_vel_z), in m/s and rad/s. |
| `joint_position_relative` | one per joint | required | Each joint's position minus its default position, in rad. |
| `joint_velocity` | one per joint | required | Each joint's velocity, in rad/s. |
| `last_action` | one per driven joint | no | The policy's previous output, before `action.scale`; zeros at first. |
| `gait_phase` | 2 | no | (sin 2πp, cos 2πp), with phase p = (elapsed time mod `period`) / `period`; needs `params: {period: S}`. |

The two IMU terms need the IMU mounted on the description's `base_link`.
Some values to check a robot or a descriptor by hand:

| Situation | Term | Values |
|---|---|---|
| Base pitched nose down by a | `projected_gravity` | (sin a, 0, −cos a) |
| Base rolled left side up by a | `projected_gravity` | (0, −sin a, −cos a) |
| IMU turned 90° left on the base, reading (1, 2, 3) | `base_angular_velocity` | (−2, 1, 3) |
| `period: 0.8`, at 0, 0.2 and 0.4 s | `gait_phase` | (0, 1), (1, 0), (0, −1) |

### Matching the training framework

Each term must match what the training code computed, value for value:

| Term | Isaac Lab (`isaaclab.envs.mdp`) | legged_gym and unitree_rl_gym |
|---|---|---|
| `base_angular_velocity` | `base_ang_vel` | `base_ang_vel` |
| `projected_gravity` | `projected_gravity` | `projected_gravity` |
| `velocity_command` | `generated_commands` of a base velocity command | `commands[:, :3]` |
| `joint_position_relative` | `joint_pos_rel` | `dof_pos - default_dof_pos` |
| `joint_velocity` | `joint_vel` (or `joint_vel_rel`, with zero default velocities) | `dof_vel` |
| `last_action` | `last_action` | `actions` |
| `gait_phase` | not provided | the sine and cosine of `phase` in unitree_rl_gym's G1 task |

Things that are easy to get wrong:

- **Joint order.** List the joints in the order the policy was trained
  with. Isaac Lab orders a term's joints by the articulation's order, not
  by the order in the term's configuration, unless `preserve_order` is
  set.
- **Scales.** Copy each term's scale and the action's from the training
  configuration. A command scaled 2, 2 and 0.25 takes a list.
- **Noise and clipping** are not applied: noise is for training only, and
  a clip that training applied needs a term of its own.
- **Default positions** come from the description, so they must be the
  ones the policy was trained with; the description hash in the
  descriptor makes sure they have not changed since.

### Adding a term

A term the platform does not provide is a plugin, in this repository or in
your own package. It has two parts: a **factory**, called once when the
builder is created, which checks the descriptor's entry and computes
whatever does not change between ticks; and the **term**, an object with a
`size` that returns `size` unscaled float64 values on each tick. For
example, a package `lab-terms` with `joint_effort`, the torques of the
selected joints, in `lab_terms/effort.py`:

```python
"""Observation terms that read the joints' torques."""

import numpy as np
from numpy.typing import NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    ObservationTerm,
    PolicyDescriptor,
    resolve_joints,
)
from embodiverse.contracts.observation import Term, TermInputs


class JointEffort:
    """Torques of the selected joints, in N·m, in the order selected."""

    def __init__(self, indices: NDArray[np.intp]) -> None:
        self.indices = indices
        self.size = indices.size

    def __call__(self, inputs: TermInputs) -> NDArray[np.float64]:
        return inputs.state.effort[self.indices]


def joint_effort(
    entry: ObservationTerm, description: RobotDescription, descriptor: PolicyDescriptor
) -> Term:
    """Build `joint_effort`: one torque per joint of `joints`, in N·m."""
    if entry.joints is None:
        raise ValueError(f"observation term {entry.term!r} needs joints")
    if entry.params:
        raise ValueError(f"observation term {entry.term!r} takes no parameters")
    try:
        names = resolve_joints(entry.joints, description)
    except ValueError as error:
        raise ValueError(f"observation term {entry.term!r}: {error}") from error
    indices = np.array([description.index_of(name) for name in names], dtype=np.intp)
    return JointEffort(indices)
```

Register the factory under the term's name, install the package, and
`uv run embodiverse list` shows it:

```toml
[project.entry-points."embodiverse.observations"]
joint_effort = "lab_terms.effort:joint_effort"
```

A descriptor then uses it like a built-in term:
`{term: joint_effort, joints: legs, scale: 0.01}`. Test a term against
values computed by hand, with joints in an order other than the
description's. Policies are trained against a term's definition, so a
different quantity, frame or order is best given a new name rather than a
changed definition.

## The runner

`Runner`, in `embodiverse.runtime.runner`, is what `embodiverse run` uses.
In Python, with the zero policy on testbot:

```python
from embodiverse.adapters.mujoco import MujocoRobot
from embodiverse.contracts.registry import robot_description
from embodiverse.runtime.policy import ZeroPolicy
from embodiverse.runtime.runner import ConstantCommand, Runner, zero_policy_descriptor

description = robot_description("testbot")
descriptor = zero_policy_descriptor(description)
policy = ZeroPolicy(3, description.num_joints)
robot = MujocoRobot(description)
runner = Runner(robot, policy, descriptor, commands=ConstantCommand(0.0))
robot.connect()
try:
    stats = runner.run(5.0)  # starts the policy first
finally:
    runner.stop()  # leaves the robot in HOLD
    robot.close()
print(stats.ticks, stats.rate_hz, dict(stats.faults))
```

For a trained policy, load the descriptor with `load_descriptor(path)` and
the policy with `OnnxPolicy.from_descriptor(descriptor, description)`,
from `embodiverse.runtime.policy`, which checks the file's hash before
loading it. The model takes one float32 input of shape `[1, N]` and
returns one float32 output of shape `[1, M]`; it runs on the CPU, on one
thread, which keeps each call's time steady.

**Before anything moves,** `Runner(...)` checks the descriptor against
the robot's description, that the policy's sizes match the descriptor,
and that every observation term can be built. The caller connects the
robot before the run and closes it after.

**Starting.** `start()`, which `run` calls, needs the robot in PASSIVE or
HOLD and a state with no fault: no non-finite value, no joint outside its
limits, no hot motor, nothing the adapter reports. Otherwise it raises
`StartRefused` and changes nothing. It then switches the robot to HOLD and
POLICY and **ramps** every joint's target in a straight line to its
default position over `RunnerConfig.start_ramp_s` (2 s), since a robot
rarely starts in its default pose and a policy's first command at full
gains would yank it there. A `start_ramp_s` of 0 skips the ramp.

**Each tick** reads the state, has the safety layer check it, builds the
observation, runs the policy, builds the joint command, has the safety
layer filter it, and writes it. The action, before its scale, is fed back
as the next tick's `last_action`, as training frameworks do.

**Lockstep and real time.** A `Steppable` robot, such as `MujocoRobot`,
runs in lockstep: the runner advances it by one period after each write,
so the run is reproducible and as fast as the machine allows. Anything
else, hardware or `RealtimeMujocoRobot`, runs in real time: the runner
sleeps until each tick's deadline on the monotonic clock. The loop runs in
plain Python, so measure its timing on the robot's computer.

**Stopping.** `stop()` switches the robot from POLICY to HOLD, as does any
error during a run, Ctrl-C included. The runner never damps the robot.
When a person's stop control damps it, the next tick sees that the robot
left POLICY and ends the run; `RunStats.left_policy` says so.

**Velocity commands** come from a `CommandSource`, whose `read(now_ns)`
returns the latest command and when it was updated, without blocking.
`ConstantCommand(vx, vy, wz)` is the one provided. Commands are clipped
to the descriptor's ranges.

`run` returns a `RunStats`: `ticks`, `duration_s`, `rate_hz`,
`jitter_s`, `max_compute_s`, `overruns`, `faults` (how often each fault
code was reported), `latched`, `latched_by`, `lockstep` and
`left_policy`.

## The safety layer

`SafetyLayer`, in `embodiverse.runtime.safety`, sits between the policy
and the robot. `check` looks at every tick's inputs before the policy
runs; `filter` looks at every command after it; `preflight` decides
whether a policy can start. It returns verdicts and safe commands, and the
runner acts on them.

| Action | What the runner does |
|---|---|
| `NONE` | Runs the policy and writes its filtered command. |
| `ZERO_VELOCITY` | Runs the policy with a zero velocity command, so a walking robot comes to a stand. Clears when its cause clears. |
| `HOLD` | Skips the policy and writes the hold command: the last safe targets and gains, with zero velocity and effort. Stays until `reset()`, which the runner's start sequence calls. |

| Fault code | When | Action |
|---|---|---|
| `state.stale` | The state is older than `state_timeout_s`. | HOLD |
| `state.non_finite` | A value of the state is NaN or infinite. | HOLD |
| `command.stale` | The velocity command is older than `command_timeout_s`. | ZERO_VELOCITY |
| `motor.hot` | A motor is within `temperature_margin_c` of `temperature_limit_c`. | ZERO_VELOCITY |
| `loop.overrun` | The previous tick took longer than its period. | HOLD |
| `policy.non_finite` | The command built from the policy's output has a NaN or infinity. | HOLD |
| any other code | The adapter reports it in the state. | `unknown_fault_action`: HOLD by default |

A stale command and a hot motor are recoverable, so the robot stands and
carries on once they clear. The others mean the policy's inputs or
outputs can no longer be trusted, so the robot holds until the policy is
started again. Automatic faults never damp the robot: a humanoid without
stiffness falls unless something is holding it.

On every command, `filter` keeps each position target inside the joint's
limits less `position_margin_rad`, each velocity target and effort within
the joint's limits, and `kp` and `kd` at most `gain_ratio_max` times the
description's. Clamping is not a fault; `stats` counts it per joint, so a
policy that leans on a limit shows up there.

`SafetyConfig` holds the thresholds:

| Field | Default | Why this default |
|---|---|---|
| `state_timeout_s` | 0.05 s | 2.5 periods at 50 Hz: tolerant of jitter, and 5 cm at 1 m/s. |
| `command_timeout_s` | 0.5 s | Several lost messages from a 10 Hz joystick or planner. |
| `position_margin_rad` | 0.05 rad | About 3°, so an overshooting PD loop stays off the mechanical stop. |
| `temperature_limit_c` | 80 °C | A general value; set it from each robot's motor rating. |
| `temperature_margin_c` | 10 °C | Time to stand and cool before the vendor's own protection cuts torque. |
| `gain_ratio_max` | 2 | Allows stiffer tuning, and catches gains copied with the wrong units. |
| `unknown_fault_action` | `hold` | A fault nobody has classified gets the strongest response. |

```python
from embodiverse.runtime.safety import SafetyConfig

config = SafetyConfig(temperature_limit_c=75.0, command_timeout_s=0.3)
```

Pass it as `SafetyLayer(description, config)` to `Runner(..., safety=...)`.
The layer works on commands: it does not detect falls or self-collision,
and it runs inside the same process as the runner, so the robot's own
protections and the vendor's stop control are what act if that process
stalls.

## The command line

`embodiverse run` loads the robot and the adapter by name, checks
everything before connecting, runs, stops the policy, closes the robot in
all cases, and prints a summary:

| Option | Meaning |
|---|---|
| `--robot NAME` | The robot description. Required. |
| `--adapter NAME` | The adapter, such as `mujoco` (lockstep), `mujoco_realtime`, `loopback` or a robot's own. Required. |
| `--descriptor PATH` | The policy descriptor. |
| `--policy onnx\|zero` | `onnx`, the default, runs the file the descriptor names. `zero` runs the zero policy, shaped by the descriptor if one is given. |
| `--seconds S` | How long to run the policy, after the start ramp; 10 by default. |
| `--command VX VY WZ` | A constant velocity command, clipped to the descriptor's ranges; zeros by default. |
| `--rate HZ` | The zero policy's rate without a descriptor; 50 by default. |
| `--record PATH` | Record the run to a new MCAP file. |

Errors print one line starting `embodiverse: error:` and exit with
status 1; arguments that do not go together exit with status 2.

## Recording

`--record PATH` records the run to an [MCAP](https://mcap.dev) file, the
log format ROS 2 and Foxglove use, and adds `recording PATH` to the
summary. A recording is never overwritten. In Python, give the runner an
`McapRecorder`; with the names of the runner's example above:

```python
from embodiverse.runtime.recording import McapRecorder, collect_manifest

manifest = collect_manifest(
    description=description, descriptor=descriptor, adapter="mujoco"
)
robot = MujocoRobot(description)
with McapRecorder("run.mcap", manifest) as recorder:
    runner = Runner(robot, policy, descriptor, recorder=recorder)
    robot.connect()
    try:
        runner.run(5.0)
    finally:
        runner.stop()
        robot.close()
```

Every tick that writes a command is recorded, the start ramp included, on
three channels:

| Channel | What it holds |
|---|---|
| `/embodiverse/state` | The state the runner read: every value, the mode, and the adapter's faults. |
| `/embodiverse/command` | The command written to the robot, after the safety layer. |
| `/embodiverse/faults` | The faults the safety layer reported on the tick. |

Each message is a JSON object with a JSON Schema, stamped with the
state's `stamp_ns`; arrays are in the description's joint order and SI
units. A value that is not finite is written as the string `"NaN"`,
`"Infinity"` or `"-Infinity"`, so a broken state is kept as it was read.
The policy's observation and action are not recorded yet.

Each file also holds a **manifest** of what ran (`RunManifest`, in
`embodiverse.contracts.manifest`):

| Field | What it records |
|---|---|
| `schema_version` | The manifest's schema version, `1`. |
| `platform_version` | The Embodiverse version. |
| `robot`, `description_hash` | The robot description's name and content hash. |
| `adapter` | The adapter's registered name. |
| `policy_sha256` | The policy file's hash; `null` for the zero policy. |
| `descriptor_hash` | The descriptor's content hash. |
| `python_version` | The Python that ran. |
| `git_sha`, `git_dirty` | The commit of the directory the run started in, and whether it had uncommitted changes; `null` outside a git repository. |
| `started_utc` | When the run started, in UTC. |
| `host` | The machine's name. |

### Reading a run

`read_run` gives back the manifest and one entry per tick:

```python
import numpy as np

from embodiverse.contracts.registry import robot_description
from embodiverse.runtime.recording import read_run

run = read_run("testbot-zero.mcap")
print(run.manifest.robot, run.manifest.git_sha, run.ticks)

position = np.stack([state["position"] for state in run.states])  # ticks × joints
target = np.stack([command["position"] for command in run.commands])
knee = robot_description(run.manifest.robot).index_of("left_knee")
print(np.abs(target[:, knee] - position[:, knee]).max())  # largest tracking error
```

In [Foxglove Studio](https://foxglove.dev), open the file and plot
`/embodiverse/state.position[1]` against `/embodiverse/command.position[1]`
(testbot's `left_knee`, as measured and as commanded), show
`/embodiverse/faults` in a Raw Messages panel, or `/embodiverse/state.mode`
in a State Transitions panel. Times are the states' stamps, so a simulated
run starts at 0 s; `started_utc` in the manifest says when it took place.

A testbot run takes about 300 bytes per tick, about 55 MB per hour at
50 Hz, and recording adds well under a millisecond to a tick.

## Adding a policy

1. **Export it to ONNX** with one float32 input of shape `[1, N]` and one
   float32 output of shape `[1, M]`. Policies with recurrent state, such
   as LSTMs, are not supported yet.
2. **Write its descriptor** next to the file, checking every term, scale,
   joint order and gain against the training code.
3. **Fill in its two hashes**: `policy.sha256` and
   `robot.description_hash` ([Checks](#checks)).
4. **Try the descriptor with the zero policy**, which builds every
   observation term and holds the default pose:
   `embodiverse run ... --policy zero --descriptor path/to/descriptor.yaml`.
5. **Run it in simulation**, in lockstep, with `--record`, and compare the
   recording with training.
6. **Keep the policy with its project.** This repository holds policies
   only as test fixtures.
