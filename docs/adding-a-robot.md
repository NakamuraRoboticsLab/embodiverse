# Adding a robot

A robot in Embodiverse is two things:

- a **robot description**: one YAML file with every fact about the robot
  model, such as its joints, limits, default pose, gains, IMU and rates;
- one or more **adapters**: code that connects the robot, or a simulation
  of it, to the robot port, the one interface the rest of the platform
  uses (see [architecture](architecture.md#the-robot-port)).

This guide goes from an empty branch to a robot that runs a policy. The
first steps need no hardware at all: a description runs on the loopback
adapter the day it is written, and an adapter is tested against a fake
transport and hand-made messages long before it meets the robot. The
Unitree G1 is the running example; the EngineAI PM01 and the Booster K1
differ only in the vendor's wire.

Useful to have open while you work: your robot's interface notes
([G1](adapters/unitree-g1.md), [PM01](adapters/engineai-pm01.md),
[K1](adapters/booster-k1.md)), which describe the vendor's transport,
messages, rates, modes, joint order and IMU from public sources, with a
list of what is still to be checked on a real unit; and the docstrings of
`embodiverse/contracts/robot.py` and `types.py`, which state exactly what
the port promises.

## The robot description

A built-in robot is a small package of data under `src/embodiverse/robots/`:

```text
src/embodiverse/robots/<name>/
├── __init__.py        a docstring and nothing else: this layer holds data only
├── description.yaml   the description
└── <name>.xml         model files it names, if any
```

A short, complete description of a robot with one knee and one
parallel-linkage ankle:

```yaml
schema_version: 1
name: example_leg
status: draft             # until every value is checked on the robot
vendor: {name: Example Robotics, model: EX-1}
base_link: pelvis         # root link in the model files

# Canonical joint order: every array of joint values follows this list.
joints:
  - name: left_knee
    vendor_name: J03_KNEE_L   # the vendor's name; only the adapter uses it
    motor_index: 0            # slot in the vendor's motor arrays
    default_position: 0.3     # rad
    limits: {position: [-0.087, 2.88], velocity: 20.0, effort: 139.0}
    gains: {kp: 150.0, kd: 4.0}            # side defaults to joint
  - name: left_ankle_pitch
    vendor_name: J04_ANKLE_P_L
    motor_index: 1
    default_position: -0.2
    limits: {position: [-0.87, 0.52], velocity: 37.0, effort: 50.0}
    gains: {kp: 40.0, kd: 2.0, side: motor}  # applied at the ankle's motors

joint_groups:
  left_leg: [left_knee, left_ankle_pitch]

imu:
  frame: imu_link
  parent: pelvis
  mount_xyz: [-0.04, 0.0, -0.08]   # m, in the pelvis frame
  mount_rpy: [0.0, 0.0, 0.0]       # rad

sensors:
  head_camera:
    type: depth_camera
    frame: head_camera_link
    parent: pelvis
    mount_xyz: [0.1, 0.0, 0.4]
    mount_rpy: [0.0, 0.5, 0.0]     # pitched 0.5 rad down

rates:
  physics_hz: 1000.0        # the motor-control loop on the robot
  control_hz_max: 500.0     # the fastest joint commands it accepts

models:                     # relative to this file
  mjcf: example_leg.xml
```

### Fields

| Field | Type and unit | Default | Meaning |
|---|---|---|---|
| `schema_version` | `1` | required | The schema version. |
| `name` | snake_case | required | The robot model, e.g. `g1_29dof`. |
| `status` | `draft` or `active` | required | `active` once every value has been checked on the robot. |
| `vendor.name`, `vendor.model` | string | no vendor | Who makes the robot, and their model name. |
| `base_link` | string | required | The root link of the kinematic tree in the model files. |
| `joints` | list, at least one | required | The actuated joints, in canonical order. |
| `joints[].name` | canonical joint name | required | See [joint names](#joint-names). |
| `joints[].vendor_name` | string | none | The vendor's name for the joint. |
| `joints[].motor_index` | integer ≥ 0, unique | required | The joint's slot in the vendor's motor arrays. |
| `joints[].default_position` | rad, inside the limits | required | The joint's position in the default pose. |
| `joints[].limits.position` | `[lower, upper]`, rad | required | The position range. |
| `joints[].limits.velocity` | > 0, rad/s | required | The highest speed. |
| `joints[].limits.effort` | > 0, N·m | required | The highest torque. |
| `joints[].gains.kp`, `.kd` | ≥ 0, N·m/rad and N·m·s/rad | required | Default PD gains. |
| `joints[].gains.side` | `joint` or `motor` | `joint` | Where the gains apply. |
| `joint_groups` | name → list of joint names | `{}` | Named subsets of the joints, each in the order given. |
| `imu.frame`, `imu.parent` | string | required | The IMU's frame, and the link it is mounted on. |
| `imu.mount_xyz` | `[x, y, z]`, m | `[0, 0, 0]` | The frame's position in the parent link. |
| `imu.mount_rpy` | `[roll, pitch, yaw]`, rad | `[0, 0, 0]` | The frame's orientation in the parent link. |
| `sensors` | name → `type`, `frame`, `parent`, `mount_xyz`, `mount_rpy` | `{}` | Every other sensor; `type` is snake_case, e.g. `lidar`. |
| `rates.physics_hz` | > 0, Hz | required | The low-level loop: the physics step, or the robot's motor-control loop. |
| `rates.control_hz_max` | > 0, Hz, at most `physics_hz` | required | The highest rate at which the robot accepts joint commands. |
| `models.mjcf`, `models.urdf` | path | none | Model files, relative to the description file. |

- **`side: motor`** is for joints driven together through a linkage, such
  as parallel-linkage ankles, where the vendor's controller closes the loop
  on the motors. The adapter maps joint targets to motor targets.
- **`mount_xyz` and `mount_rpy`** mean what a URDF `<origin>` means: roll,
  pitch and yaw are rotations about the parent's fixed x, y and z axes, in
  that order.

### Joint names

Joints have canonical names, lowercase snake_case in the pattern
`side_part_axis`, so the same joint has the same name on every robot and
code written for one robot's legs works on another's:

| Part | Canonical names |
|---|---|
| Legs | `{left,right}_hip_pitch`, `_hip_roll`, `_hip_yaw`, `_knee`, `_ankle_pitch`, `_ankle_roll` |
| Waist | `waist_yaw`, `waist_roll`, `waist_pitch` |
| Arms | `{left,right}_shoulder_pitch`, `_shoulder_roll`, `_shoulder_yaw`, `_elbow`, `_wrist_roll`, `_wrist_pitch`, `_wrist_yaw` |
| Head | `head_yaw`, `head_pitch` |

A joint is named by what it does, not by the vendor's label: the K1 joint
its vendor calls "elbow pitch" rotates the upper arm, so it is
`left_shoulder_yaw`. A robot with two elbow axes uses `_elbow_pitch` and
`_elbow_yaw`, and a joint the table does not have gets a name in the same
pattern.

### Checking it

Loading checks everything and names the field, joint or frame at fault:
unknown or missing keys, non-finite numbers, names that are not snake_case,
duplicate joint names or motor indices, limits out of order, negative
gains, a default position outside its limits, a group naming a joint that
does not exist, the IMU, a sensor and `base_link` sharing a frame name, and
`control_hz_max` above `physics_hz`. To check a file:

```bash
uv run python -c 'from embodiverse.contracts.description import load_description; print(load_description("src/embodiverse/robots/testbot/description.yaml").joint_names)'
```

In Python, `robot.index_of("left_knee")` gives a joint's place in every
joint array, `robot.joint("left_knee").limits.effort` its limits,
`robot.group("left_leg")` a group, and `robot.model_path("mjcf")` the
resolved path of a model file.

### Where the values come from

Vendors publish what a description needs in pieces: URDF and MJCF files
hold kinematics and limits, SDK headers hold motor indices and vendor
names, and reference deployments hold gains and default poses. The
built-in descriptions say in their header which file and commit each value
comes from, mark the values that still need checking with a
`# verify on unit` comment, and list every third-party source and its
licence in [`THIRD_PARTY.md`](../THIRD_PARTY.md). Doing the same for a new
robot makes it easy to check later. A vendor's files can be copied into
the repository only where their licence allows it.

`status` stays `draft` until every value has been checked against the
robot. Changing it to `active` changes the description's content hash, so
regenerate its page, and update the `description_hash` of any policy
descriptor written against the draft.

## Registering it

Entry points are read from the installed package's metadata, so add the
robot to `pyproject.toml` and install again:

```toml
[project.entry-points."embodiverse.robot_descriptions"]
<name> = "embodiverse.robots.<name>"
```

```bash
uv sync
uv run embodiverse list
uv run embodiverse run --robot <name> --adapter loopback --policy zero --seconds 2
```

The last command runs the whole pipeline (observations, the zero policy,
the safety layer and the runner) on the loopback adapter, a stand-in robot
whose joints follow their targets. `faults none` in its summary means the
description and the pipeline fit together. For a robot that is already
here:

```bash
uv run embodiverse run --robot g1_29dof --adapter loopback --policy zero --seconds 2
```

A test that loads the description and checks its joints, as
`tests/test_robots.py` does for the built-in robots, keeps it that way.

### From your own package

A robot, an adapter or an observation term can also live in a package of
your own, in its own repository. It registers the same way:

```text
lab-robots/
├── pyproject.toml
└── src/lab_robots/
    ├── __init__.py
    ├── adapter.py           # G1Robot, an adapter
    └── g1/                  # a robot description, as a package
        ├── __init__.py
        ├── description.yaml
        └── g1.xml
```

```toml
[project]
name = "lab-robots"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["embodiverse"]

[project.entry-points."embodiverse.robot_descriptions"]
g1 = "lab_robots.g1"

[project.entry-points."embodiverse.adapters"]
unitree_g1 = "lab_robots.adapter:G1Robot"

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.uv.sources]
embodiverse = { path = "../embodiverse", editable = true }  # a clone next to it
```

`uv sync` in that project installs it together with Embodiverse, and
`uv run embodiverse list` there shows `g1` and `unitree_g1`. Good to know:

- **Names are unique within a group** across every installed package; two
  packages registering the same name raise `RegistryError`, naming both.
- **Reinstall after changing entry points.** An editable install sees code
  changes at once, but not new entry points: run `uv sync` again.
- **Listing imports nothing**, so a plugin is listed even if loading it
  would fail, for example because its SDK is missing. `load(ADAPTERS,
  name)` imports it, and an import error becomes a `RegistryError` naming
  the entry point.
- **A description package holds data only**: loading the description
  imports the package, and the model files must be installed as files.
- **A plugin keeps its name when it moves** into this repository, so the
  projects that use it keep working.

## Writing an adapter

An adapter for a real robot is built on the adapter base, in
`embodiverse.adapters.base`, which implements every promise of the robot
port once: the lifecycle, the mode table, each mode's control law, command
checks, and a stop that enters DAMP from any thread without waiting. What
is left to write is what differs between robots: how state arrives, and
how the current command is sent.

`TransportRobot` is the base with the shape of hardware built in. A real
robot keeps applying the last command it received, so stopping sending is
not a stop: `TransportRobot` runs a sender thread that sends the current
command every physics period, and at once when it changes, whether or not
the runner wrote anything. On `close` it enters DAMP and keeps sending it
for a moment before releasing the robot. You give it a `Transport`, one
vendor's wire in the port's terms:

| Method | Thread | What it does |
|---|---|---|
| `open()` | control | Opens the connection and starts receiving. Raises `ConnectionError` if the robot cannot be reached. |
| `receive()` | control | Returns the newest state received, or `None` before the first. Never blocks. |
| `send(law)` | sender | Sends `law`, the `JointCommand` the motors should apply now, without waiting. Raises if the send fails. |
| `close()` | control | Closes the connection, after the last `send`. |

What crosses the transport is already converted: description joint order,
SI units, (x, y, z, w) quaternions, `stamp_ns` from `time.monotonic_ns()`
at receipt, and the robot's own problems as `Fault`s in `faults`. A send
that raises is reported as a `transport.send_failed` fault in every later
state.

The loopback adapter has exactly this shape, so it is the template:

```text
src/embodiverse/adapters/loopback/        src/embodiverse/adapters/unitree_g1/
├── __init__.py                           ├── __init__.py
├── model.py       (the fake robot)       ├── codec.py      (vendor messages <-> RobotState, JointCommand)
├── transport.py   (LoopbackTransport)    ├── transport.py  (a Transport over the vendor's SDK)
└── robot.py       (the two classes)      └── robot.py      (a TransportRobot with that transport)
```

### Copy the loopback

```bash
git switch -c feat/unitree-g1-adapter main
cp -r src/embodiverse/adapters/loopback src/embodiverse/adapters/unitree_g1
```

In the copy, delete `model.py`; in `robot.py`, remove the lockstep class
and rename `RealtimeLoopbackRobot` to `UnitreeG1Robot`; empty
`transport.py` down to a `UnitreeG1Transport` class whose four methods
raise `NotImplementedError`; and make `__init__.py` export
`UnitreeG1Robot`. Then make the test file,
`tests/adapters/test_unitree_g1.py`, run the conformance kit on the real
adapter class over the loopback's transport:

```python
import pytest

from embodiverse.adapters.loopback import LoopbackTransport
from embodiverse.adapters.unitree_g1 import UnitreeG1Robot
from embodiverse.contracts.registry import robot_description
from embodiverse.tools.conformance.robot import RobotConformance

G1 = robot_description("g1_29dof")


class TestUnitreeG1OverLoopback(RobotConformance):
    timing_budget_s = 0.005  # the kit's 1 ms is for the robot's own computer

    @pytest.fixture
    def make_robot(self):
        return lambda: UnitreeG1Robot(G1, transport=LoopbackTransport(G1))
```

`uv run pytest tests/adapters/test_unitree_g1.py` passes straight away:
the base implements the port, and the loopback transport behaves like a
robot. Keep it passing from here on.

### The codec

`codec.py` holds the two conversions as pure functions, so tests can feed
them messages written by hand:

```python
def decode_state(message, stamp_ns: int) -> RobotState: ...
def encode_command(law: JointCommand, message) -> None: ...
```

They read and fill the vendor's message objects through their attributes
only, so a test can pass a plain object with the same attribute names and
run without the SDK. Everything the notes list under "Mapping onto the
Robot port" happens here:

| Conversion | Where the notes cover it | Helper |
|---|---|---|
| Vendor motor slots to description joint order, and back | section 6, and each joint's `motor_index` | `MotorMap(description, slots=...)` |
| Quaternion order, if the vendor uses (w, x, y, z) | section 7 | `wxyz_to_xyzw`, `xyzw_to_wxyz` |
| Units to SI: rad, rad/s, N·m, m/s², °C | sections 3 and 7 | |
| One temperature per joint from per-motor values | section 9 | |
| The vendor's error fields to `Fault` codes | section 9 | |
| Fields the vendor needs on every command, such as mode bytes and checksums | section 3 | |

`MotorMap` converts between description order and a vendor's motor arrays,
which often have more slots than the robot has joints:

```python
from embodiverse.adapters.base import MotorMap

motors = MotorMap(description, slots=35)  # the G1's LowCmd_ has 35 slots
q_motors = motors.to_motors(command.position, fill=0.0)  # description order -> slots
q_joints = motors.from_motors(state_q)  # slots -> description order
```

Test each conversion with a message written by hand: a state with known
motor values comes out in description order and SI units; a command lands
in the right slots with the unused ones zero; the quaternion round-trips;
and each pitfall in the notes gets a test of its own, such as the
(w, x, y, z) order, 35 slots with 29 used, or a checksum compared with a
value from the vendor's reference code. These tests need no SDK, so they
run on every machine, every time.

### The transport

`transport.py` implements the four methods over the vendor's SDK:

- `open()` initialises the SDK, checks that no other process is already
  commanding the robot where the notes say how to detect one, subscribes
  to the state topic with a callback that decodes each message with
  `decode_state` and keeps the newest in one attribute, and creates the
  command publisher. Anything the vendor's mode machine needs before it
  accepts commands, such as releasing its own controller, goes here too,
  as the notes describe. `connect` waits for the first state, with a
  timeout, so `open` does not.
- `receive()` returns that newest state, or `None`.
- `send(law)` encodes into a preallocated message and publishes it.
- `close()` stops the subscription and releases the SDK.

`send` runs on the sender thread while `receive` runs on the control
thread, so swap whole objects rather than filling one in place. Report the
robot's problems, stale or inconsistent messages included, as faults in
the state; the adapter never changes mode on its own.

### The robot class

`robot.py` is the loopback's real-time class with your transport and the
robot's capabilities, from section 9 of the notes:

```python
from embodiverse.adapters.base import TransportRobot
from embodiverse.adapters.unitree_g1.transport import UnitreeG1Transport
from embodiverse.contracts.robot import Capabilities


class UnitreeG1Robot(TransportRobot):
    def __init__(self, description, transport=None):
        super().__init__(
            description,
            Capabilities(effort_command=True, temperature=True, max_rate_hz=500.0),
            transport or UnitreeG1Transport(description),
        )
```

Keep the `transport` parameter: it is what lets the tests build the real
class over the loopback transport. `TransportRobot` also takes
`connect_timeout_s` (5 s by default) and `damp_s`, how long DAMP is sent on
`close` (0.1 s).

The loopback transport reports no motor temperatures, and the kit checks
that a robot declaring `temperature=True` reports them. So once the class
declares the G1's capabilities, give the test a loopback transport that
adds temperatures:

```python
import dataclasses

import numpy as np


class LoopbackWithTemperatures(LoopbackTransport):
    """The loopback transport, with every motor at 35 °C, as the G1 reports."""

    def receive(self):
        state = super().receive()
        if state is None:
            return None
        return dataclasses.replace(state, temperature=np.full(G1.num_joints, 35.0))
```

and build the robot in `make_robot` with
`UnitreeG1Robot(G1, transport=LoopbackWithTemperatures(G1))`.

A simulator that steps its own physics, such as the MuJoCo adapter, is
built on `RobotBase` instead and fills in its hooks: `_open`, `_latest`
(the newest state, without waiting), `_apply(law)` (called whenever the law
changes, possibly from the stop's thread, so it must return at once),
`_close` and, optionally, `_faults`. The smallest such adapter is about
fifteen lines; `StillRobot` in `tests/adapters/test_base.py` is one.

### The conformance kit

The kit, `RobotConformance` in `embodiverse.tools.conformance.robot`,
turns the port's docstrings into ten checks, each a test method of your
test class. The first seven need only the port, so they run on every
adapter, over the loopback transport on every commit. The last three need
a simulation probe that can hold the base at a known pose, so they run on
simulators.

| Check | What must hold |
|---|---|
| `check_description` | `description` and `capabilities()` are the same on every call. |
| `check_lifecycle` | Nothing works before `connect` or after `close`; `connect` starts in PASSIVE; `close` is idempotent. |
| `check_state` | `read()` gives finite float64 arrays of the right sizes, a unit quaternion, and temperatures exactly when declared. |
| `check_timestamps` | `stamp_ns` never decreases, and advances with time. |
| `check_timing` | `read` and `write` take at most `timing_budget_s` (1 ms) at the 95th percentile. |
| `check_modes` | Every transition in the mode table works; every other one raises `ValueError`; `operator_stop` from another thread enters DAMP from every mode. |
| `check_rejects_malformed_commands` | `write` refuses wrong lengths, NaN, infinity, negative gains, and effort the robot cannot apply. |
| `check_joint_order` | Commanding one joint moves that joint, and only that joint. |
| `check_units` | Each joint settles at its target, and its velocity agrees with its motion. |
| `check_imu` | With the base held level, rolled, pitched and turned, the IMU reads the expected orientation, gravity and zero rotation. |

Failures say what was done, what was expected and what was seen, and name
the likely mistake:

```text
AssertionError: joint 'left_hip_pitch' did not move when commanded: expected
at least 0.100 rad toward a target -0.200 rad away, observed +0.000 rad;
joint 'left_knee' moved -0.199 rad instead: joint order may be wrong; map
description order to the motors by motor_index, in read and in write
```

Tolerances are class attributes, such as `timing_budget_s`,
`joint_settle_s` or `imu_angle_tolerance_rad`, and the checks are also plain
functions that take the same tolerances as keyword arguments:

```python
from embodiverse.tools.conformance.robot import check_modes, check_state

check_state(robot)
check_modes(robot)
```

What the kit cannot see: a mistake made the same way in `read` and `write`
(two joints swapped in both directions still behave consistently), the
gyro's frame and scale (the probe holds the base still), and timing on the
robot's own computer. The codec tests, on hand-made messages, are the place
to catch the first two.

### Registering the adapter

One line in the adapters group of `pyproject.toml`, then reinstall:

```toml
[project.entry-points."embodiverse.adapters"]
unitree_g1 = "embodiverse.adapters.unitree_g1:UnitreeG1Robot"
```

```bash
uv sync
uv run embodiverse list
```

`uv.lock` does not change: the vendor SDKs are already locked as extras.

## Vendor SDKs

Each robot's SDK is an optional extra, installed only where you need it.
`uv sync` and `uv run nox` install none of them, so the checks pass on any
machine without an SDK.

| Robot | Interface | Install | Import names |
|---|---|---|---|
| Unitree G1 | `unitree_sdk2_python`, DDS over CycloneDDS 0.10.2 | `uv sync --extra g1` | `unitree_sdk2py` |
| EngineAI PM01 | ROS 2 Humble topics of the stock controller's `joint_bridge` | ROS 2 Humble and the vendor's `interface_protocol` package (below) | `rclpy`, `interface_protocol` |
| Booster K1 | `booster_robotics_sdk_python`, DDS over Fast DDS | `uv sync --extra k1` | `booster_robotics_sdk_python` |

`uv sync` on its own removes an extra again. Tests that need the SDK's
real message types start with `pytest.importorskip("<import name>")`, so
they skip where the SDK is absent and run where it is installed:

```bash
uv sync --extra g1
uv run pytest tests/adapters
```

**Unitree G1.** The extra installs `unitree_sdk2py` from GitHub at the
commit the G1 notes cite, with `cyclonedds==0.10.2`, the version the robot
runs. That `cyclonedds` has wheels only for Python 3.10 on x86_64, the
development environment. Elsewhere, including the robot's aarch64
computer, uv builds it from source, which needs the CycloneDDS C library
of the same release, built from the `releases/0.10.x` branch and named in
`CYCLONEDDS_HOME`; the SDK's README gives the steps. To move to a newer
SDK commit, change its `rev` under `[tool.uv.sources]` in
`pyproject.toml` and run `uv lock`.

**EngineAI PM01.** The adapter goes through the ROS 2 topic bridge: the
stock controller stays in charge, the robot is put in `joint_bridge` with
the remote, and the adapter publishes `/hardware/joint_command` and
subscribes to `/hardware/joint_state`, `/hardware/imu_info` and
`/motion/motion_state` (route A in the [PM01 notes](adapters/engineai-pm01.md),
sections 2 and 9). This route is reachable from Python, is what the
vendor's own examples use, and needs nothing compiled beyond one message
package. The vendor's native C++ SDK is the fallback if the bridge's
latency or its motion-state switching proves unusable on the unit. The
bridge's Python side comes from ROS 2 rather than PyPI, so there is no
extra:

1. Install ROS 2 Humble on Ubuntu 22.04 with `ros-humble-rmw-cyclonedds-cpp`,
   and source it: `source /opt/ros/humble/setup.bash`.
2. Build the vendor's message package at the commit the notes cite:

   ```bash
   git clone https://github.com/engineai-robotics/engineai_ros2_workspace
   cd engineai_ros2_workspace
   git checkout c41043f64ca1d308be187ebdcc2eaae40884f59c
   colcon build --packages-select interface_protocol
   source install/setup.bash
   ```

3. Set the robot's ROS 2 settings in every shell that talks to it:

   ```bash
   export ROS_DOMAIN_ID=69 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_LOCALHOST_ONLY=0
   ```

The uv environment then sees both packages through `PYTHONPATH`. ROS 2
Humble's `rclpy` is built for Python 3.10, the project's Python. Because
`uv run nox` removes `PYTHONPATH`, the PM01's SDK tests always skip there;
run them with `uv run pytest tests/adapters` in a shell that has sourced
both.

**Booster K1.** The extra installs `booster_robotics_sdk_python` 1.6.3 from
PyPI: one wheel that bundles Fast DDS, for Python 3.10 to 3.14 on x86_64
and aarch64 (Ubuntu 22.04 or later). The robot's firmware needs to be at
the version the [K1 notes](adapters/booster-k1.md) give. The wheel also
installs the vendor's examples as a top-level `python_example` package.

The layering test lets the adapters layer import these SDKs and MuJoCo.
Only an adapter imports its own SDK; the runtime and the tools never see
it. An adapter that needs another package adds it to the adapters row of
`RULES` in `tests/test_architecture.py`.

## A MuJoCo model (optional)

With an MJCF model, the MuJoCo adapter simulates the robot, and the three
conformance checks that need ground truth run on it. They catch mistakes
in the description itself, such as a wrong joint order or IMU mount. Put
the model next to the description and name it in `models.mjcf`. It works
with the adapter if:

1. **Its timestep is one physics period**, `1 / rates.physics_hz`:
   `<option timestep="0.002"/>` for 500 Hz.
2. **Every joint in the description is a hinge or slide joint with the
   same canonical name.** The model may have other joints, which the
   adapter leaves alone.
3. **Each of those joints is driven by exactly one `<motor>`** with a joint
   transmission. The adapter computes the PD torque itself, so `<position>`
   and `<velocity>` actuators are refused. MuJoCo clamps `ctrl` to the
   motor's `ctrlrange`, so make the range cover the effort limit divided
   by the gear.
4. **The description's `base_link` is a body with a `<freejoint/>`.**
5. **A `<site>` named after the IMU's frame** carries a `framequat` sensor
   (with `objtype="site"`), a `gyro` and an `accelerometer`, placed where
   `mount_xyz` and `mount_rpy` say.
6. **Optionally, a keyframe named `default`** gives the initial state, such
   as the default pose standing on the floor.
7. **The names `embodiverse_gantry` and `embodiverse_gantry_weld` are
   free:** the adapter adds a body and a weld with those names, to hold the
   base still for the conformance checks, as a gantry does.

A model that does not fit raises `ValueError` when the robot is created,
naming the file and the problem. testbot's model,
`src/embodiverse/robots/testbot/testbot.xml`, is a complete example:

```xml
<option timestep="0.002" integrator="implicitfast"/>
...
<body name="pelvis" pos="0 0 0.578">
  <freejoint name="floating_base"/>
  <site name="imu_link" pos="-0.02 0 0.04"/>
  ...
</body>
...
<actuator>
  <motor name="left_knee" joint="left_knee" gear="1" ctrlrange="-80 80"/>
  ...
</actuator>
<sensor>
  <framequat name="imu_orientation" objtype="site" objname="imu_link"/>
  <gyro name="imu_angular_velocity" site="imu_link"/>
  <accelerometer name="imu_linear_acceleration" site="imu_link"/>
</sensor>
<keyframe>
  <key name="default" qpos="0 0 0.578 1 0 0 0 -0.3 0.6 -0.3 -0.3 0.6 -0.3"/>
</keyframe>
```

testbot is a six-joint biped written from scratch for testing: small,
fast, deterministic, and free of any vendor's licence terms. Its model
repeats the description's limits and default pose, and a test fails if the
two differ. With the default gains it stands in HOLD; with zero torque it
falls.

The adapter's PD loop is explicit: each step's torque is computed from the
state at the start of the step. With stiff vendor gains, give the joints a
realistic `armature`, or use a shorter timestep. To run the kit on your
model, subclass `RobotConformance` with the adapter as its own probe, as
`tests/adapters/test_mujoco.py` does for testbot:

```python
import pytest

from embodiverse.adapters.mujoco import MujocoRobot
from embodiverse.contracts.registry import robot_description
from embodiverse.tools.conformance.robot import RobotConformance


class TestMyRobotInMujoco(RobotConformance):
    @pytest.fixture
    def make_robot(self):
        return lambda: MujocoRobot(robot_description("my_robot"))

    @pytest.fixture
    def probe(self, robot):
        return robot  # the adapter is its own simulation probe
```

`mujoco` runs in lockstep, deterministically and as fast as the machine
allows; `mujoco_realtime` runs on its own physics thread at the physics
rate, as hardware does. `robot.model` and `robot.data` are MuJoCo's
`MjModel` and `MjData`, for viewers and diagnostics.

## Running it

`embodiverse run` loads the robot and the adapter by name and runs a
policy through the whole pipeline. The zero policy holds every joint at
its default position, which makes it the first thing to run on any new
robot or adapter:

```bash
uv run embodiverse run --robot testbot --adapter mujoco --policy zero --seconds 5
uv run embodiverse run --robot pm01 --adapter loopback_realtime --policy zero --seconds 2
```

The summary shows the rate achieved, the jitter, the longest tick and the
faults. `faults none` means the description, the adapter and the runner
fit together. Where a vendor ships a simulator that speaks the robot's own
wire, such as the G1's `unitree_mujoco`, the same command with your
adapter exercises the real transport without a robot. [Policies](policies.md#the-command-line)
lists every option.

## The robot's page

Each built-in robot has a page in `docs/robots/`, whose tables of joints,
joint groups and IMU are generated from its description, so they cannot
drift from it. After adding or changing a description:

```bash
uv run embodiverse generate          # writes the pages that changed
uv run embodiverse generate --check  # writes nothing; exits 1 if a page is out of date
```

A new robot's page is created with a heading, a line of text and the
generated block. Write your notes above and below the block, such as
where the values come from and what someone bringing up the robot needs to
know. The block itself sits between two marker lines and is rewritten on
every run:

```markdown
<!-- BEGIN GENERATED: robot-joint-tables -->
<!-- This block is generated: do not edit it by hand. Change its source, then run: embodiverse generate -->
...
<!-- END GENERATED: robot-joint-tables -->
```

`tests/test_generated_files.py` runs the check, so `uv run nox` fails
while a page is out of date. The block shows the content hash, so any
change to a description, even one no table shows, changes its page.

Generators live in `embodiverse.tools.generators`. A new one is a function
that takes the repository's root and returns `{path: full new text}`,
wrapped in a `Generator` with a name and a one-line description and added
to `GENERATORS`; `replace_block` rewrites one block of a hand-written file.
It writes nothing itself, and gives the same text on every machine, so the
check passes everywhere.

## Testing without the robot

Most of an adapter can be tested before it meets the robot:

- **The codec tests**, on messages written by hand, cover every
  conversion: joint order, units, quaternion order, faults and checksums.
- **The conformance kit over the loopback transport** runs the real
  adapter class through the seven checks that need no simulator.
- **SDK tests** with `pytest.importorskip` check the vendor's real message
  types where the SDK is installed.
- **The vendor's simulator**, where there is one, exercises the real
  transport: `embodiverse run --robot g1_29dof --adapter <yours> --policy
  zero`.
- **A MuJoCo model**, if the robot has one, checks the description's joint
  order and IMU mount against ground truth.

`uv run nox` on a machine without the SDK, and `uv run pytest
tests/adapters` with it installed, together cover everything that can be
covered off the robot.

## Trying it on the robot

Some practical tips for the first sessions with the real robot:

- **Start in simulation and on the loopback.** When the zero policy runs
  there with `faults none`, the description, the adapter and the runner fit
  together, and what is left to learn is the robot itself.
- **Hang the robot on a gantry or another support,** feet clear of the
  ground, with the vendor's remote or emergency stop in your hand. PASSIVE
  is zero torque and DAMP is damping only, so a humanoid in either falls
  unless something holds it. The vendor's own stop also works when your
  program has stalled. Two people make it easier: one watching the robot
  with the stop, one at the keyboard.
- **Begin read-only.** `connect()` and `read()` only: move each joint by
  hand and compare its position and sign with the vendor's tools, tilt the
  robot and compare the IMU, and check that `stamp_ns` advances at the
  rate the notes give.
- **Then HOLD.** From PASSIVE, `set_mode(HOLD)` should hold every joint
  where it is, with the description's gains. Try the stop from HOLD too:
  `operator_stop()` damps every joint at once.
- **Run the seven checks** that need no simulator through the real
  transport, with `make_robot` returning the real adapter, or call the
  check functions from a script.
- **Settle the "To verify on the real unit" list** in the robot's notes,
  such as rates, what the robot does when commands stop, IMU units and
  signs, and the damping value for DAMP. Put each answer in the
  description and the notes.
- **Check the safety layer's thresholds** in `SafetyConfig`, such as the
  motor temperature limit, against the robot's own ratings
  ([Policies](policies.md#the-safety-layer)).
- **Start a policy from HOLD,** with the zero policy first, and record
  every run with `--record run.mcap`: when something surprises you, the
  recording shows each tick's state, command and faults.
- **Stop at the first surprise,** such as a joint moving the wrong way or
  an IMU axis with the wrong sign. Fix it in the code or the description,
  check it again in simulation, and come back.
