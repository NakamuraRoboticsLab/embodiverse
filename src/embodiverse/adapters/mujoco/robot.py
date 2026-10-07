"""The MuJoCo adapter: a robot's MJCF model, simulated behind the robot port.

Two classes implement `Robot` for any robot whose description names an MJCF
model, so code above the port drives the simulated robot exactly as it will
drive the hardware. They differ only in how simulated time passes:

- `MujocoRobot` runs in **lockstep**. Simulated time advances only in
  `advance`, so runs are deterministic and do not depend on the speed of the
  machine. This is for tests and evaluation. It is `Steppable`.
- `RealtimeMujocoRobot` runs in **real time**. `connect` starts a physics
  thread that steps the simulation at the description's physics rate
  against the monotonic clock, as hardware runs whether anyone reads it or
  not. This is for interactive use. `read` returns the latest state, and
  `write` and `set_mode` hand data to the thread. It has no `advance`, so it
  is not `Steppable`: a simulator that runs in real time implements only
  `Robot`. Being two classes, `isinstance(robot, Steppable)` and static
  type checks agree on which is which.

Both are built on the adapter base (`embodiverse.adapters.base`), which
implements the lifecycle, the mode table and each mode's control law, and
they fill in the physics: at every physics step they apply the current
law, as a motor driver does, driving each joint's motor with
`kp * (q_target - q) + kd * (dq_target - dq) + tau_ff`, clipped to the
joint's effort limit:

- PASSIVE applies zero torque. HOLD holds the positions measured on
  entering it, with zero target velocity, no feed-forward torque and the
  description's gains. POLICY applies the latest command written in
  POLICY, and keeps the HOLD targets until the first one. DAMP applies
  `-kd * dq` with the description's kd.
- State is stamped with the simulation time. `effort` is the torque the
  motors applied in the last step. MuJoCo computes the IMU readings and the
  motor torques at the start of each step, so they are one physics step
  older than the positions and velocities. There are no motor temperatures.
- `connect` resets the simulation to the keyframe named `default`, if the
  model has one, and otherwise to the model's reference pose with the
  joints at the description's default positions.

The model is loaded and checked against the description when the robot is
created. ValueError names the file and the first mismatch, unless:

- `description.model_path("mjcf")` names an MJCF file that compiles;
- the model's timestep is `1 / description.rates.physics_hz`, within 1e-9 s;
- each of the description's joints is a hinge or slide joint of the same
  name, driven by exactly one actuator, a motor: no activation dynamics, a
  fixed gain and no bias, so the joint torque is `gear * gain * ctrl`;
- the description's `base_link` is a body with a free joint;
- a site named after the description's IMU frame carries a `framequat`
  sensor with no reference frame, a `gyro` and an `accelerometer`.

Both also implement the conformance kit's `SimulationProbe`, except that
only `MujocoRobot` has its `advance`. To hold the base still and supported,
as a gantry does, the adapter adds a mocap body and an inactive weld between
that body and the base to the model before compiling it. `set_base_pose`
moves the mocap body to the pose and activates the weld, whose constraint
force then carries the base's weight, so the IMU reads specific force +g.
Resetting the base's position at every step instead would leave it in free
fall between resets, and the IMU would read zero.

docs/adding-a-robot.md explains what a model needs, for model authors.
"""

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from embodiverse.adapters.base import RobotBase
from embodiverse.contracts.conventions import (
    as_vector,
    normalize_quaternion,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.robot import Capabilities
from embodiverse.contracts.types import Fault, ImuState, RobotState

try:
    import mujoco
except ImportError as error:
    raise ImportError(
        "the MuJoCo adapter needs the mujoco package: install embodiverse[sim]"
    ) from error

__all__ = ["MujocoRobot", "RealtimeMujocoRobot"]

DEFAULT_KEYFRAME = "default"
"""Name of the keyframe that `connect` resets to, if the model has one."""

GANTRY_BODY = "embodiverse_gantry"
"""Name of the mocap body the adapter adds to hold the base."""

GANTRY_WELD = "embodiverse_gantry_weld"
"""Name of the weld the adapter adds between the gantry body and the base."""

SIMULATION_STOPPED = "simulation.stopped"
"""Code of the fault a real-time robot reports once a physics step failed."""

# How the weld behaves. Its time constant is twice the timestep, the smallest
# MuJoCo allows (a stiffer weld would be unstable), and its impedance is near
# one, so testbot's held base sags by about 0.01 mm and turns by under 0.01°.
_GANTRY_SOLREF_STEPS = 2.0
_GANTRY_DAMPING_RATIO = 1.0
_GANTRY_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)

_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_JOINT_TRANSMISSION = int(mujoco.mjtTrn.mjTRN_JOINT)
_NO_DYNAMICS = int(mujoco.mjtDyn.mjDYN_NONE)
_FIXED_GAIN = int(mujoco.mjtGain.mjGAIN_FIXED)
_NO_BIAS = int(mujoco.mjtBias.mjBIAS_NONE)
_SITE = int(mujoco.mjtObj.mjOBJ_SITE)
_IMU_SENSORS = (
    ("framequat", int(mujoco.mjtSensor.mjSENS_FRAMEQUAT)),
    ("gyro", int(mujoco.mjtSensor.mjSENS_GYRO)),
    ("accelerometer", int(mujoco.mjtSensor.mjSENS_ACCELEROMETER)),
)


class _MujocoRobotBase(RobotBase):
    """Everything the two ways of running share: the model, its checks, the
    physics step that applies the base's current law, and the probe's
    methods other than `advance`. See the module docstring.
    """

    def __init__(self, description: RobotDescription) -> None:
        model, layout = _load(description)
        super().__init__(
            description,
            Capabilities(
                effort_command=True,
                temperature=False,
                max_rate_hz=description.rates.physics_hz,
            ),
            damp_on_close=False,
        )
        self._model, self._layout = model, layout
        self._data = mujoco.MjData(self._model)
        joints = description.joints
        self._default = np.array([joint.default_position for joint in joints])
        self._effort_limit = np.array([joint.limits.effort for joint in joints])
        # Guards the simulation data, which a physics thread writes in real
        # time. Never taken under the base's mode lock.
        self._sim_lock = threading.Lock()

    # --- Hooks ---------------------------------------------------------------

    def _open(self) -> None:
        """Reset the simulation to its initial state."""
        with self._sim_lock:
            self._reset()

    def _latest(self) -> RobotState:
        """The state at the current simulation time."""
        layout = self._layout
        with self._sim_lock:
            data = self._data
            stamp_ns = round(data.time * 1e9)
            position = data.qpos[layout.qpos]
            velocity = data.qvel[layout.dof]
            effort = data.qfrc_actuator[layout.dof]
            orientation = data.sensordata[layout.orientation].copy()
            angular_velocity = data.sensordata[layout.angular_velocity].copy()
            linear_acceleration = data.sensordata[layout.linear_acceleration].copy()
        imu = ImuState(
            orientation=wxyz_to_xyzw(orientation),
            angular_velocity=angular_velocity,
            linear_acceleration=linear_acceleration,
        )
        return RobotState(
            stamp_ns=stamp_ns,
            position=position,
            velocity=velocity,
            effort=effort,
            imu=imu,
            mode=self._control[0],
        )

    # --- SimulationProbe -----------------------------------------------------

    def set_base_pose(self, position: ArrayLike, orientation_xyzw: ArrayLike) -> None:
        """Hold the base at this pose, at rest and supported, until `release_base`.

        Moves the gantry's mocap body to the pose, puts the base there with
        zero velocity, and activates the weld between them.
        """
        self._check_connected()
        position = as_vector(position, 3, "position")
        orientation = xyzw_to_wxyz(normalize_quaternion(orientation_xyzw))
        layout = self._layout
        base = layout.base_qpos
        with self._sim_lock:
            data = self._data
            data.mocap_pos[layout.gantry_mocap] = position
            data.mocap_quat[layout.gantry_mocap] = orientation
            data.qpos[base : base + 3] = position
            data.qpos[base + 3 : base + 7] = orientation
            data.qvel[layout.base_dof : layout.base_dof + 6] = 0.0
            data.eq_active[layout.gantry_weld] = 1
            mujoco.mj_forward(self._model, data)

    def release_base(self) -> None:
        """Deactivate the gantry's weld, so the base moves freely again."""
        self._check_connected()
        with self._sim_lock:
            self._data.eq_active[self._layout.gantry_weld] = 0

    def set_joint_positions(self, positions: ArrayLike) -> None:
        """Set every joint's position, in description order, with zero velocity.

        The next `read` reports the new positions. The mode and its targets
        do not change.
        """
        self._check_connected()
        positions = as_vector(positions, self._description.num_joints, "positions")
        with self._sim_lock:
            self._data.qpos[self._layout.qpos] = positions
            self._data.qvel[self._layout.dof] = 0.0
            mujoco.mj_forward(self._model, self._data)

    # --- MuJoCo --------------------------------------------------------------

    @property
    def model(self) -> mujoco.MjModel:
        """The compiled model, with the gantry's mocap body and weld added."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """The simulation's data, for viewers and diagnostics.

        Changing it bypasses the adapter. In real time the physics thread
        writes it while it is read, so a reader may see a step half done.
        """
        return self._data

    # --- Internals -----------------------------------------------------------

    def _reset(self) -> None:
        """Reset to the initial state: the `default` keyframe, or the default pose."""
        model, data = self._model, self._data
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, DEFAULT_KEYFRAME)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(model, data, key)
        else:
            mujoco.mj_resetData(model, data)
            data.qpos[self._layout.qpos] = self._default
        mujoco.mj_forward(model, data)

    def _step(self) -> None:
        """Apply the current mode's control law, then advance one physics step."""
        _, law = self._control
        layout, data = self._layout, self._data
        q = data.qpos[layout.qpos]
        dq = data.qvel[layout.dof]
        tau = law.kp * (law.position - q) + law.kd * (law.velocity - dq) + law.effort
        np.clip(tau, -self._effort_limit, self._effort_limit, out=tau)
        data.ctrl[layout.actuator] = tau / layout.scale
        mujoco.mj_step(self._model, data)


class MujocoRobot(_MujocoRobotBase):
    """A robot simulated in MuJoCo in lockstep, for tests and evaluation.

    Implements `Robot`, `Steppable` and the conformance kit's
    `SimulationProbe`. Simulated time advances only in `advance`, so the
    same calls give bit-identical states. The module docstring describes
    the modes, the state and the checks on the model, which raise ValueError
    when the robot is created.
    """

    def advance(self, seconds: float) -> None:
        """Simulate the next `seconds` of time, as `Steppable.advance` says.

        Applies the current mode's control at every physics step. Raises
        ValueError unless `seconds` is a positive whole number of physics
        steps, and RuntimeError if not connected.
        """
        self._check_connected()
        hz = self._description.rates.physics_hz
        steps = round(seconds * hz) if math.isfinite(seconds) else 0
        if steps < 1 or not math.isclose(steps / hz, seconds, rel_tol=1e-9):
            raise ValueError(
                f"advance({seconds!r}): expected a positive whole number of"
                f" physics steps of 1/{hz:g} s"
            )
        # Only the control thread steps the simulation, and operator_stop
        # takes only the base's mode lock, so one acquisition suffices.
        with self._sim_lock:
            for _ in range(steps):
                self._step()


class RealtimeMujocoRobot(_MujocoRobotBase):
    """A robot simulated in MuJoCo in real time, for interactive use.

    Implements `Robot`. `connect` starts a physics thread that steps the
    simulation at the description's physics rate against the monotonic
    clock, and `close` stops it. It has no `advance`, so it is not
    `Steppable`, and of the conformance kit's `SimulationProbe` it has every
    method but `advance`. The module docstring describes the modes, the
    state and the checks on the model, which raise ValueError when the robot
    is created.

    If a physics step raises, the thread stops and simulated time stands
    still. Every later state is the last one, and reports a fault with code
    `simulation.stopped` and the exception as its message.
    """

    def __init__(self, description: RobotDescription) -> None:
        super().__init__(description)
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._overruns = 0
        self._failure: Fault | None = None

    @property
    def overruns(self) -> int:
        """How many physics steps started more than a step late."""
        return self._overruns

    def _open(self) -> None:
        """Reset the simulation and start the physics thread."""
        super()._open()
        self._thread = threading.Thread(
            target=self._run, name=f"mujoco-{self._description.name}", daemon=True
        )
        self._thread.start()

    def _close(self) -> None:
        """Stop the physics thread and wait for it."""
        thread = self._thread
        if thread is not None:
            self._stopping.set()
            thread.join()
            self._thread = None

    def _faults(self) -> tuple[Fault, ...]:
        failure = self._failure
        return () if failure is None else (failure,)

    def _run(self) -> None:
        """Step the physics at the physics rate until `close` or a failure.

        Each step has a deadline on the monotonic clock, one period after the
        previous one. The thread sleeps until the deadline; if it is already
        past, it steps at once to catch up, and counts the step as an overrun
        if it is more than a period late. If a step raises, the thread keeps
        the exception as a fault and stops.
        """
        period = 1.0 / self._description.rates.physics_hz
        deadline = time.monotonic()
        while True:
            lateness = time.monotonic() - deadline
            if lateness < 0.0:
                if self._stopping.wait(-lateness):
                    return
            elif self._stopping.is_set():
                return
            elif lateness > period:
                self._overruns += 1
            try:
                with self._sim_lock:
                    self._step()
            except Exception as error:
                message = " ".join(f"{type(error).__name__}: {error}".split())
                self._failure = Fault(code=SIMULATION_STOPPED, message=message)
                return
            deadline += period


# --- Loading and checking the model ---------------------------------------------


@dataclass(frozen=True)
class _Layout:
    """Where a description's joints, base and IMU are in a compiled model.

    The joint arrays are in description order, so per-step code indexes the
    model's arrays with them, vectorised.
    """

    qpos: NDArray[np.intp]
    """Each joint's address in qpos."""

    dof: NDArray[np.intp]
    """Each joint's address in qvel and the other per-dof arrays."""

    actuator: NDArray[np.intp]
    """The id of each joint's motor."""

    scale: NDArray[np.float64]
    """Each motor's joint torque per unit of ctrl: its gear times its gain."""

    base_qpos: int
    """Address in qpos of the base's free joint: position, then (w, x, y, z)."""

    base_dof: int
    """Address in qvel of the base's free joint: linear, then angular velocity."""

    gantry_mocap: int
    """Mocap id of the gantry's body."""

    gantry_weld: int
    """Id of the weld between the gantry's body and the base."""

    orientation: slice
    """The IMU's framequat, (w, x, y, z), in sensordata."""

    angular_velocity: slice
    """The IMU's gyro in sensordata."""

    linear_acceleration: slice
    """The IMU's accelerometer in sensordata."""


def _load(description: RobotDescription) -> tuple[mujoco.MjModel, _Layout]:
    """Load the description's MJCF, add the gantry, compile it and check it."""
    path = description.model_path("mjcf")
    if path is None:
        raise ValueError(
            f"robot {description.name!r} has no MJCF model: set models.mjcf in its"
            " description"
        )
    try:
        spec = mujoco.MjSpec.from_file(str(path))
    except ValueError as error:
        raise ValueError(f"{path}: cannot load the MJCF model: {error}") from error
    if spec.body(description.base_link) is None:
        raise _mismatch(
            path,
            description,
            f"it has no body named {description.base_link!r}, the description's"
            " base_link",
        )
    _add_gantry(spec, description.base_link)
    try:
        model = spec.compile()
    except ValueError as error:
        raise ValueError(f"{path}: cannot compile the MJCF model: {error}") from error
    return model, _check(model, description, path)


def _add_gantry(spec: mujoco.MjSpec, base_link: str) -> None:
    """Add a mocap body and an inactive weld that holds the base to it."""
    spec.worldbody.add_body(name=GANTRY_BODY, mocap=True)
    weld = spec.add_equality(
        name=GANTRY_WELD,
        type=mujoco.mjtEq.mjEQ_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1=GANTRY_BODY,
        name2=base_link,
        active=False,
    )
    # The weld's anchor (3), the base's pose relative to the gantry body, as
    # a position (3) and a (w, x, y, z) quaternion (4), and its torque scale:
    # the base is held at the gantry body's pose, against forces and torques.
    weld.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    weld.solref = [_GANTRY_SOLREF_STEPS * spec.option.timestep, _GANTRY_DAMPING_RATIO]
    weld.solimp = list(_GANTRY_SOLIMP)


def _check(model: mujoco.MjModel, description: RobotDescription, path: Path) -> _Layout:
    """Check that `model` fits `description`, and return where everything is."""
    expected = 1.0 / description.rates.physics_hz
    timestep = float(model.opt.timestep)
    if abs(timestep - expected) > 1e-9:
        raise _mismatch(
            path,
            description,
            f"its timestep is {timestep:g} s, but physics_hz"
            f" {description.rates.physics_hz:g} needs {expected:g} s",
        )

    motors: dict[int, list[int]] = {}
    for actuator in range(model.nu):
        if int(model.actuator_trntype[actuator]) == _JOINT_TRANSMISSION:
            joint = int(model.actuator_trnid[actuator, 0])
            motors.setdefault(joint, []).append(actuator)
    qpos: list[int] = []
    dof: list[int] = []
    actuators: list[int] = []
    scale: list[float] = []
    for name in description.joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise _mismatch(path, description, f"it has no joint named {name!r}")
        if int(model.jnt_type[joint]) not in (_HINGE, _SLIDE):
            kind = mujoco.mjtJoint(int(model.jnt_type[joint])).name
            raise _mismatch(
                path,
                description,
                f"joint {name!r} is of type {kind}, expected a hinge or slide joint",
            )
        driving = motors.get(joint, [])
        if len(driving) != 1:
            names = ", ".join(repr(_actuator_name(model, a)) for a in driving)
            raise _mismatch(
                path,
                description,
                f"joint {name!r} is driven by {len(driving)} actuators"
                f"{f' ({names})' if names else ''}, expected exactly one motor",
            )
        actuator = driving[0]
        if (
            int(model.actuator_dyntype[actuator]) != _NO_DYNAMICS
            or int(model.actuator_gaintype[actuator]) != _FIXED_GAIN
            or int(model.actuator_biastype[actuator]) != _NO_BIAS
        ):
            raise _mismatch(
                path,
                description,
                f"actuator {_actuator_name(model, actuator)!r} of joint {name!r}"
                " is not a motor: the adapter computes each joint's torque, so it"
                " needs a <motor>, with no activation dynamics, a fixed gain and"
                " no bias",
            )
        torque_per_ctrl = float(
            model.actuator_gear[actuator, 0] * model.actuator_gainprm[actuator, 0]
        )
        if torque_per_ctrl == 0.0:
            raise _mismatch(
                path,
                description,
                f"actuator {_actuator_name(model, actuator)!r} of joint {name!r}"
                " has a zero gear or gain, so it applies no torque",
            )
        qpos.append(int(model.jnt_qposadr[joint]))
        dof.append(int(model.jnt_dofadr[joint]))
        actuators.append(actuator)
        scale.append(torque_per_ctrl)

    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, description.base_link)
    first = int(model.body_jntadr[base])
    if int(model.body_jntnum[base]) < 1 or int(model.jnt_type[first]) != _FREE:
        raise _mismatch(
            path,
            description,
            f"body {description.base_link!r}, the description's base_link, has no"
            " free joint",
        )

    frame = description.imu.frame
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, frame)
    if site < 0:
        raise _mismatch(
            path,
            description,
            f"it has no site named {frame!r}, the description's IMU frame",
        )
    readings: list[slice] = []
    for kind, sensor_type in _IMU_SENSORS:
        reading = _site_sensor(model, site, sensor_type)
        if reading is None:
            raise _mismatch(
                path,
                description,
                f"IMU site {frame!r} has no {kind} sensor"
                + (" without a reference frame" if kind == "framequat" else ""),
            )
        readings.append(reading)

    gantry = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GANTRY_BODY)
    return _Layout(
        qpos=np.array(qpos, dtype=np.intp),
        dof=np.array(dof, dtype=np.intp),
        actuator=np.array(actuators, dtype=np.intp),
        scale=np.array(scale),
        base_qpos=int(model.jnt_qposadr[first]),
        base_dof=int(model.jnt_dofadr[first]),
        gantry_mocap=int(model.body_mocapid[gantry]),
        gantry_weld=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, GANTRY_WELD),
        orientation=readings[0],
        angular_velocity=readings[1],
        linear_acceleration=readings[2],
    )


def _site_sensor(model: mujoco.MjModel, site: int, sensor_type: int) -> slice | None:
    """Return where the first sensor of `sensor_type` on `site` is in sensordata."""
    for sensor in range(model.nsensor):
        if (
            int(model.sensor_type[sensor]) == sensor_type
            and int(model.sensor_objtype[sensor]) == _SITE
            and int(model.sensor_objid[sensor]) == site
            and int(model.sensor_refid[sensor]) == -1
        ):
            address = int(model.sensor_adr[sensor])
            return slice(address, address + int(model.sensor_dim[sensor]))
    return None


def _actuator_name(model: mujoco.MjModel, actuator: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    return name or f"#{actuator}"


def _mismatch(path: Path, description: RobotDescription, problem: str) -> ValueError:
    return ValueError(
        f"{path}: the MJCF model does not fit robot {description.name!r}: {problem}"
    )
