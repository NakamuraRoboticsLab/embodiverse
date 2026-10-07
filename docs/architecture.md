# Architecture

How Embodiverse fits together: the layers, the control loop, the contracts
that every robot, simulator and policy agree on, and the pieces built on
them. The module docstrings are the full reference; this page is the map.

## The goal

The lab's humanoids come from three vendors, and each SDK has its own
transport, motor order, joint names, units and motor modes. Policies also
run in simulators before they reach a robot. Embodiverse puts all of them
behind one interface and one set of conventions, so code written against
it runs on any of the robots and in simulation without changes, and work
done in one project can be picked up by another. Each fact about a robot
is written once, in its robot description, and everything else reads it
from there.

Moving from simulation to hardware, or from one simulator to another, means
swapping the **adapter**: the piece of code that connects one robot or
simulator to the platform. Everything above it stays the same.

## Layers

Embodiverse is split into layers, each a subpackage of `embodiverse`:

| Layer | Package | May import, besides the standard library |
|---|---|---|
| Contracts | `embodiverse.contracts` | numpy, pydantic, PyYAML |
| Adapters | `embodiverse.adapters` | the contracts, MuJoCo and the vendor SDKs |
| Runtime | `embodiverse.runtime` | the contracts, onnxruntime, mcap |
| Tools | `embodiverse.tools` | the contracts, pytest |
| App | `embodiverse.app` | anything |
| Robots | `embodiverse.robots` | nothing: data files and `__init__.py` files without imports |

**The dependency rule:** adapters, the runtime and the tools depend only on
the contracts, never on each other. They meet in the app layer, which finds
robots and adapters by name through the plugin registry. Because the runner
never imports an adapter, the same runner drives every robot, and a new
adapter cannot break it. The contracts are imported by everything, so they
stay small enough to install everywhere: robot computers (aarch64, Python
3.10), training environments and laptops without ROS 2 or a vendor SDK.

`tests/test_architecture.py` checks the rule on every test run. It reads
each module's source and checks every import, even inside functions,
against the table `RULES` at the top of the test, and names the file, line
and import of any that breaks it. Such an import works on a machine that
has everything installed and fails later where a layer is used on its own,
so it is worth catching early. When a layer needs a new package, add it to
`RULES` in the same change. A `components` layer, for estimators and
localizers, is reserved there.

## The control loop

The runner runs a policy on a robot, one tick at the policy's rate:

```text
  Robot (an adapter)       hardware, MuJoCo or the loopback
     │ read() → RobotState
     ▼
  SafetyLayer.check        stale or broken state, stale velocity command,
     │                     hot motors, a late tick → NONE, ZERO_VELOCITY or HOLD
     ▼
  ObservationBuilder       the terms the policy descriptor names, scaled
     │ float32 vector
     ▼
  Policy                   an ONNX file, or the zero policy
     │ action
     ▼
  joint command            default pose + scale × action, with gains;
     │                     held joints at their default pose
     ▼
  SafetyLayer.filter       targets inside limits, gains capped, effort clipped
     │ JointCommand
     ▼
  Robot.write()            then the recorder, if the run is recorded
```

A simulator that can be stepped runs in **lockstep**: simulated time
advances by one tick after each write, so runs are reproducible and as
fast as the machine allows. Hardware, and simulators that run on their own
clock, run in **real time**: the runner sleeps until each tick's deadline.

## Contracts

### Conventions

SI units throughout; adapters convert at the boundary. Mixing conventions
gives plausible but wrong numbers, such as gravity read upside down, so
they are fixed once and checked by the conformance kit.

| Quantity | Unit |
|---|---|
| Joint position, angle | rad |
| Joint velocity, angular velocity | rad/s |
| Joint effort (torque) | N·m |
| Position stiffness (kp) | N·m/rad |
| Velocity damping (kd) | N·m·s/rad |
| Linear acceleration | m/s² |
| Length, position | m |
| Time | s, or ns for timestamps |
| Rate | Hz |
| Temperature | °C |

- **Axes** follow ROS [REP-103](https://www.ros.org/reps/rep-0103.html):
  x forward, y left, z up, right-handed. **Frame names** follow
  [REP-105](https://www.ros.org/reps/rep-0105.html): `map`, `odom` and
  `base_link`. Sensor frames are named in the robot description.
- **Quaternions** are (x, y, z, w) at every interface, as in ROS messages.
  MuJoCo and Isaac Lab 2.x use (w, x, y, z); their adapters convert with
  `wxyz_to_xyzw` and `xyzw_to_wxyz`. A quaternion giving frame B's
  orientation in frame A maps vectors from B to A with `rotate_vector(q, v)`;
  `multiply_quaternions(a, b)` is `b` followed by `a`. An IMU's orientation
  is the IMU frame in a world-fixed frame with z up.
- **Joint names** are lowercase snake_case, `side_part_axis` where it
  applies, such as `left_hip_pitch` or `waist_yaw`, and name what the joint
  does, so the same joint has the same name on every robot. Vendor names
  are kept in the description as `vendor_name` and never used above the
  adapter. [Adding a robot](adding-a-robot.md#joint-names) lists them.
- **Arrays** of joint values follow the joint order of the robot's
  description. Motor indices exist only inside the adapter.
- **Time** is a monotonic clock in nanoseconds: `time.monotonic_ns()` on
  hardware, simulation time in a simulator.

The helpers are in `embodiverse.contracts.conventions`.

### The robot description

One YAML file per robot model states its joints in canonical order, with
limits, default pose and gains; its IMU and other sensors with their
mounts; its loop rates; and its model files. `load_description` validates
it, and unknown keys are rejected, so a misspelled field is an error
rather than a line silently ignored. [Adding a robot](adding-a-robot.md#the-robot-description)
lists every field.

### The content hash

A description is identified by its **content hash**, `sha256:` and 64 hex
digits, computed from its validated data in a canonical form. It changes
with any value, the joint order and `status` included, and ignores
formatting, comments, key order and where the file is stored. A policy
descriptor records the hash of the description the policy was trained
against, and the runner refuses to run it on any other; recordings record
it too. A hand-kept version number drifts when someone forgets to bump it,
and a hash of the file's bytes changes when it is reformatted; the content
hash has neither problem. `content_hash` and `canonical_json` are in
`embodiverse.contracts.hashing`.

### Runtime types

Adapters produce a `RobotState` and consume a `JointCommand` on every tick
(`embodiverse.contracts.types`). Every array is a read-only float64 copy in
description joint order, so no component can change what another one is
reading, and a simulator's buffers cannot change under a state that was
built from them.

| Type | Fields | Units |
|---|---|---|
| `RobotState` | `stamp_ns` | ns, monotonic clock |
| | `position`, `velocity`, `effort` | rad, rad/s, N·m |
| | `temperature` (None if not reported) | °C |
| | `imu` (an `ImuState`), `mode`, `faults` | |
| `ImuState` | `orientation` | (x, y, z, w) quaternion |
| | `angular_velocity`, `linear_acceleration` | rad/s, m/s², in the IMU frame |
| `JointCommand` | `position`, `velocity`, `effort` | rad, rad/s, N·m |
| | `kp`, `kd` | N·m/rad, N·m·s/rad |

The robot applies `tau = kp * (position - q) + kd * (velocity - dq) + effort`
to each joint, where `q` and `dq` are its measured position and velocity.
`linear_acceleration` is specific force: a level IMU at rest reads about
+9.81 m/s² on z. Construction checks only shapes, so a non-finite value
travels as data to the safety layer, which handles it in one place.

### The robot port

`Robot`, in `embodiverse.contracts.robot`, is the one interface every robot
and simulator is driven through. It is a `Protocol`, so an adapter
satisfies it by having the right methods, without inheriting from a
platform class. Its docstrings state what blocks, which thread calls what
and which errors are raised.

| Method | What it does |
|---|---|
| `description`, `capabilities()` | The robot's description, and what it supports. |
| `connect()` | Opens the connection and waits for the first state; the robot starts in PASSIVE. |
| `read()` | Returns the latest state, without blocking. |
| `write(command)` | Checks the command; in POLICY, it becomes the command applied. Never blocks. |
| `set_mode(mode)` | Changes the mode, following the mode table. |
| `operator_stop()` | Enters DAMP from any mode, from any thread, without waiting for the control thread. |
| `close()` | Releases the robot; hardware adapters enter DAMP first. |

One control thread calls every method except `operator_stop`, which a
person's stop control may call at any time.

| Mode | What the motors do |
|---|---|
| PASSIVE | Zero torque: the joints move freely. |
| HOLD | Hold the positions measured on entering HOLD, with the description's gains. |
| POLICY | Apply the latest command written in POLICY; hold until the first one. |
| DAMP | Damping only: kp = 0, with the description's kd. |

The mode table, `MODE_TRANSITIONS`:

| From | `set_mode` may go to | `operator_stop` goes to |
|---|---|---|
| PASSIVE | HOLD | DAMP |
| HOLD | PASSIVE, POLICY | DAMP |
| POLICY | HOLD | DAMP |
| DAMP | PASSIVE | DAMP |

POLICY is entered and left only through HOLD, so the robot is always held
with known gains when a policy takes over or lets go. DAMP is entered only
by `operator_stop` (and by hardware adapters on `close`): a humanoid
without stiffness falls unless something is holding it, and only a person
can see whether something is. An adapter reports what it detects in
`RobotState.faults` and never changes mode on its own.

`capabilities()` declares where robots differ: whether they take
feed-forward torque (`effort_command`), whether they report temperatures,
and the highest useful command rate (`max_rate_hz`). A simulator that also
implements `Steppable` has `advance(seconds)` and runs in lockstep.

## Adapters

- **The adapter base** (`embodiverse.adapters.base`) implements the port's
  promises once: the lifecycle, the mode table, the control law of each
  mode, command checks and a stop that never waits. `RobotBase` suits a
  simulator that steps its own physics. `TransportRobot` adds the shape of
  hardware: a `Transport` for the vendor's wire, and a sender thread that
  sends the current command every physics period.
- **The loopback** (`loopback`, `loopback_realtime`) is a stand-in robot
  for any description: joints without inertia that follow their targets,
  on a base that stays where it is put. It needs no model file and nothing
  installed, so the whole pipeline runs on a robot the day its description
  exists. It is also the template for hardware adapters, and its transport
  is the fake transport their tests run on
  ([Adding a robot](adding-a-robot.md#writing-an-adapter)).
- **MuJoCo** (`mujoco`, `mujoco_realtime`) simulates any robot whose
  description names an MJCF model, in lockstep or in real time. It is two
  classes rather than one with a switch, so that `isinstance(robot,
  Steppable)` and the type checker agree on which robots can be advanced.

## Runtime

- **Observations.** A policy descriptor names its input terms in order,
  such as `projected_gravity` or `joint_velocity`, and the
  `ObservationBuilder` turns them into the exact input vector on every
  tick. Each term has one written definition, checked against values
  computed by hand, because a term with the wrong sign or joint order still
  gives plausible numbers.
- **The safety layer.** Every tick's inputs pass through
  `SafetyLayer.check` before the policy runs, and every command passes
  through `SafetyLayer.filter` before it reaches the robot. It keeps joint
  targets inside the description's limits, caps gains at a multiple of the
  description's and clips effort. On stale or non-finite state, a
  non-finite policy output or a late tick, it holds the last safe targets;
  on a stale velocity command or a hot motor, it zeroes the velocity
  command so a walking robot comes to a stand.

  Automatic faults never damp the robot, for the reason above: the layer
  has no DAMP action and holds no reference to the robot. Its thresholds
  are in `SafetyConfig`. [Policies](policies.md#the-safety-layer) has the
  fault table and the defaults.
- **The runner** joins the pieces in the loop above. It checks everything
  it can before the robot is touched, refuses to start from a faulty
  state, ramps every joint to its default pose before the policy takes
  over, and leaves the robot in HOLD when it stops.
- **Recording.** `--record` writes every tick's state, command and faults
  to an MCAP file, with a manifest of the platform version, robot
  description, policy, descriptor and commit that ran. MCAP is
  self-describing, needs no ROS to read or write, and opens in Foxglove
  Studio.

[Policies](policies.md) covers all four in detail.

## Tools

- **The conformance kit** (`embodiverse.tools.conformance.robot`) turns the
  robot port's docstrings into ten checks. The most dangerous adapter bugs
  are silent: joints in the wrong order, degrees for radians, a quaternion
  in the wrong order. A simulator can hold the robot at known poses, so
  three of the checks compare what the adapter reports with ground truth;
  the other seven need only the port, and run on every adapter.
- **The generators** (`embodiverse.tools.generators`) write the files
  derived from a source of truth, such as the joint tables on each robot's
  page, generated from its description. A table copied by hand drifts
  without anyone noticing; `embodiverse generate --check`, run by the
  tests, fails while a generated file is out of date.

## Plugins

Robot descriptions, adapters and observation terms can live in any
installed package. Embodiverse finds them through Python entry points, so
adding one never means editing a central list, and listing them imports
nothing, so a missing SDK only matters where it is used. A plugin keeps
its name when it moves into this repository, so the projects that use it
keep working.

| Group | An entry point names |
|---|---|
| `embodiverse.robot_descriptions` | A package that holds a `description.yaml` and its model files |
| `embodiverse.adapters` | A callable that takes a `RobotDescription` and returns a `Robot` |
| `embodiverse.observations` | A factory for an observation term, named as descriptors name it |
| `embodiverse.components` | Reserved for estimators, localizers and other capabilities |

`uv run embodiverse list` shows everything installed. The registry is
`embodiverse.contracts.registry`; [Adding a robot](adding-a-robot.md#registering-it)
shows how to register a robot and an adapter.
