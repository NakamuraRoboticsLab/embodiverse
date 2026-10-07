"""Tests of the MuJoCo adapter, on testbot.

testbot passes the robot conformance kit in lockstep, with the adapter as
its own probe, and every check that needs no probe in real time. The other
tests check that testbot's MJCF repeats its description, that models which
do not fit a description are refused, how testbot behaves in each mode, and
that the kit catches broken variants of the adapter.
"""

import math
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import pytest

from embodiverse.adapters.mujoco import MujocoRobot, RealtimeMujocoRobot
from embodiverse.contracts.conventions import (
    IDENTITY_QUATERNION,
    rotate_vector,
    xyzw_to_wxyz,
)
from embodiverse.contracts.description import RobotDescription, load_description
from embodiverse.contracts.robot import Robot, Steppable
from embodiverse.contracts.types import Fault, JointCommand, Mode, RobotState
from embodiverse.tools.conformance.robot import (
    RobotConformance,
    SimulationProbe,
    check_imu,
    check_joint_order,
)


def load_testbot() -> RobotDescription:
    with as_file(files("embodiverse.robots.testbot") / "description.yaml") as path:
        return load_description(path)


def mjcf_path(description: RobotDescription) -> Path:
    path = description.model_path("mjcf")
    assert path is not None
    return path


TESTBOT = load_testbot()
TESTBOT_MJCF = mjcf_path(TESTBOT)
DEFAULT_POSE = np.array([joint.default_position for joint in TESTBOT.joints])
EFFORT_LIMITS = np.array([joint.limits.effort for joint in TESTBOT.joints])
KP = np.array([joint.gains.kp for joint in TESTBOT.joints])
KD = np.array([joint.gains.kd for joint in TESTBOT.joints])
ZEROS = np.zeros(TESTBOT.num_joints)


# The kit.


# The kit's default timing budget of 1 ms is meant for a robot's own
# computer. These tests also run on shared development machines, where the
# scheduler can pause a thread for milliseconds; 5 ms still shows that read
# and write never wait for the physics.
SHARED_MACHINE_BUDGET_S = 0.005


class TestMujocoTestbot(RobotConformance):
    """In lockstep, testbot passes the whole kit, with the adapter as its probe."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: MujocoRobot(TESTBOT)

    @pytest.fixture
    def probe(self, robot: Robot) -> SimulationProbe | None:
        assert isinstance(robot, MujocoRobot)
        return robot


class TestMujocoTestbotRealtime(RobotConformance):
    """In real time, testbot passes every check that needs no probe."""

    timing_budget_s = SHARED_MACHINE_BUDGET_S

    @pytest.fixture
    def make_robot(self) -> Callable[[], Robot]:
        return lambda: RealtimeMujocoRobot(TESTBOT)


def test_only_lockstep_is_steppable() -> None:
    lockstep = MujocoRobot(TESTBOT)
    assert isinstance(lockstep, Robot)
    assert isinstance(lockstep, Steppable)
    assert isinstance(lockstep, SimulationProbe)
    real_time = RealtimeMujocoRobot(TESTBOT)
    assert isinstance(real_time, Robot)
    assert not isinstance(real_time, Steppable)
    assert not hasattr(real_time, "advance")


# testbot's MJCF repeats its description.


def test_testbot_mjcf_repeats_its_description() -> None:
    model = mujoco.MjModel.from_xml_path(str(TESTBOT_MJCF))
    hinges = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        for joint in range(model.njnt)
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    assert tuple(hinges) == TESTBOT.joint_names
    assert model.opt.timestep == pytest.approx(1.0 / TESTBOT.rates.physics_hz)
    default = model.key("default").qpos
    actuators: list[int] = []
    for joint in TESTBOT.joints:
        mj_joint = model.joint(joint.name)
        assert tuple(mj_joint.range) == joint.limits.position
        assert default[mj_joint.qposadr[0]] == joint.default_position
        (actuator,) = np.flatnonzero(model.actuator_trnid[:, 0] == mj_joint.id)
        actuators.append(int(actuator))
        effort = joint.limits.effort
        assert tuple(model.actuator_ctrlrange[actuator]) == (-effort, effort)
        assert model.actuator_gear[actuator, 0] == 1.0
    assert actuators == sorted(actuators), "actuators are not in description order"

    site = model.site(TESTBOT.imu.frame)
    assert model.body(site.bodyid[0]).name == TESTBOT.imu.parent
    assert tuple(site.pos) == TESTBOT.imu.mount_xyz
    assert tuple(site.quat) == (1.0, 0.0, 0.0, 0.0)
    on_site = {
        int(model.sensor_type[sensor])
        for sensor in range(model.nsensor)
        if model.sensor_objtype[sensor] == mujoco.mjtObj.mjOBJ_SITE
        and model.sensor_objid[sensor] == site.id
    }
    assert on_site == {
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT),
        int(mujoco.mjtSensor.mjSENS_GYRO),
        int(mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    }


# Models that do not fit the description are refused.


Replacements = tuple[tuple[str, str], ...]


def variant(
    tmp_path: Path,
    *,
    mjcf: Replacements = (),
    edit: Callable[[dict[str, Any]], None] = lambda data: None,
) -> RobotDescription:
    """Return testbot's description changed by `edit`, its MJCF by `mjcf`.

    Each pair in `mjcf` replaces every occurrence of its first string, which
    must occur, with its second, in a copy of testbot.xml in `tmp_path`.
    """
    text = TESTBOT_MJCF.read_text(encoding="utf-8")
    for old, new in mjcf:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "testbot.xml"
    path.write_text(text, encoding="utf-8")
    data = TESTBOT.model_dump()
    data["models"]["mjcf"] = str(path)
    edit(data)
    return RobotDescription.model_validate(data)


def rename_first_joint(name: str) -> Callable[[dict[str, Any]], None]:
    def edit(data: dict[str, Any]) -> None:
        data["joints"][0]["name"] = name
        data["joint_groups"] = {}

    return edit


def set_rate(data: dict[str, Any]) -> None:
    data["rates"]["physics_hz"] = 1000.0


def set_base_link(data: dict[str, Any]) -> None:
    data["base_link"] = "torso"


def drop_models(data: dict[str, Any]) -> None:
    data["models"] = {}


KNEE_MOTOR = '<motor name="left_knee" joint="left_knee" gear="1" ctrlrange="-80 80"/>'


@pytest.mark.parametrize(
    ("mjcf", "edit", "problem"),
    [
        pytest.param(
            (),
            rename_first_joint("left_elbow"),
            "it has no joint named 'left_elbow'",
            id="joint missing",
        ),
        pytest.param(
            (),
            rename_first_joint("floating_base"),
            "joint 'floating_base' is of type mjJNT_FREE, expected a hinge or slide"
            " joint",
            id="free joint as a description joint",
        ),
        pytest.param(
            (),
            set_rate,
            "its timestep is 0.002 s, but physics_hz 1000 needs 0.001 s",
            id="wrong timestep",
        ),
        pytest.param(
            ((KNEE_MOTOR, ""),),
            lambda data: None,
            "joint 'left_knee' is driven by 0 actuators, expected exactly one motor",
            id="actuator missing",
        ),
        pytest.param(
            (
                (
                    KNEE_MOTOR,
                    KNEE_MOTOR + '<motor name="left_knee_2" joint="left_knee"/>',
                ),
            ),
            lambda data: None,
            "joint 'left_knee' is driven by 2 actuators \\('left_knee',"
            " 'left_knee_2'\\), expected exactly one motor",
            id="two actuators",
        ),
        pytest.param(
            (
                (
                    KNEE_MOTOR,
                    '<position name="left_knee" joint="left_knee" kp="80"'
                    ' ctrlrange="0 2.2"/>',
                ),
            ),
            lambda data: None,
            "actuator 'left_knee' of joint 'left_knee' is not a motor",
            id="position actuator",
        ),
        pytest.param(
            (),
            set_base_link,
            "it has no body named 'torso', the description's base_link",
            id="base missing",
        ),
        pytest.param(
            (
                ('<freejoint name="floating_base"/>', ""),
                ('qpos="0 0 0.578 1 0 0 0 ', 'qpos="'),
            ),
            lambda data: None,
            "body 'pelvis', the description's base_link, has no free joint",
            id="no free joint",
        ),
        pytest.param(
            (('"imu_link"', '"chest_imu"'),),
            lambda data: None,
            "it has no site named 'imu_link', the description's IMU frame",
            id="IMU site missing",
        ),
        pytest.param(
            (('<gyro name="imu_angular_velocity" site="imu_link"/>', ""),),
            lambda data: None,
            "IMU site 'imu_link' has no gyro sensor",
            id="gyro missing",
        ),
    ],
)
def test_refuses_a_model_that_does_not_fit_the_description(
    tmp_path: Path,
    mjcf: Replacements,
    edit: Callable[[dict[str, Any]], None],
    problem: str,
) -> None:
    description = variant(tmp_path, mjcf=mjcf, edit=edit)
    match = f"^{tmp_path / 'testbot.xml'}: the MJCF model does not fit robot"
    with pytest.raises(ValueError, match=f"{match} 'testbot': {problem}"):
        MujocoRobot(description)


def test_refuses_a_description_without_an_mjcf_model(tmp_path: Path) -> None:
    description = variant(tmp_path, edit=drop_models)
    with pytest.raises(ValueError, match="^robot 'testbot' has no MJCF model"):
        MujocoRobot(description)


def test_maps_actuators_by_joint_not_by_order(tmp_path: Path) -> None:
    text = TESTBOT_MJCF.read_text(encoding="utf-8")
    start, end = text.index("<actuator>"), text.index("</actuator>")
    motors = [line for line in text[start:end].splitlines() if "<motor" in line]
    reversed_actuators = "<actuator>\n" + "\n".join(reversed(motors)) + "\n"
    description = variant(tmp_path, mjcf=((text[start:end], reversed_actuators),))
    robot = MujocoRobot(description)
    robot.connect()
    try:
        check_joint_order(robot, robot)
    finally:
        robot.close()


# Behaviour.


@pytest.fixture
def testbot() -> Iterator[MujocoRobot]:
    robot = MujocoRobot(TESTBOT)
    robot.connect()
    yield robot
    robot.close()


def pelvis_height(robot: MujocoRobot | RealtimeMujocoRobot) -> float:
    return float(robot.data.joint("floating_base").qpos[2])


def tilt(state: RobotState) -> float:
    """Return the angle between the IMU's z axis and the world's, in rad."""
    up = rotate_vector(state.imu.orientation, (0.0, 0.0, 1.0))
    return math.acos(min(1.0, float(up[2])))


def command(position: np.ndarray, kp: np.ndarray = KP) -> JointCommand:
    return JointCommand(position=position, velocity=ZEROS, effort=ZEROS, kp=kp, kd=KD)


def test_starts_in_the_default_pose(testbot: MujocoRobot) -> None:
    state = testbot.read()
    assert state.stamp_ns == 0
    assert state.mode is Mode.PASSIVE
    np.testing.assert_array_equal(state.position, DEFAULT_POSE)
    np.testing.assert_array_equal(state.velocity, ZEROS)
    assert state.temperature is None


def test_hold_keeps_testbot_standing(testbot: MujocoRobot) -> None:
    start = pelvis_height(testbot)
    testbot.set_mode(Mode.HOLD)
    testbot.advance(5.0)
    assert abs(pelvis_height(testbot) - start) < 0.02
    assert tilt(testbot.read()) < math.radians(3.0)


def test_passive_lets_testbot_collapse(testbot: MujocoRobot) -> None:
    start = pelvis_height(testbot)
    testbot.advance(2.0)
    assert pelvis_height(testbot) < start - 0.2


def test_a_stop_in_policy_damps_and_settles(testbot: MujocoRobot) -> None:
    testbot.set_mode(Mode.HOLD)
    testbot.set_mode(Mode.POLICY)
    crouch = DEFAULT_POSE + np.array([-0.3, 0.6, -0.3, -0.3, 0.6, -0.3])
    testbot.write(command(crouch))
    testbot.advance(0.5)
    testbot.operator_stop()
    testbot.write(command(DEFAULT_POSE))  # discarded: a stop cannot be overridden
    # With damping only, testbot folds slowly under its weight: it comes to
    # rest on its joint limits after about 5 s.
    testbot.advance(6.0)
    state = testbot.read()
    assert state.mode is Mode.DAMP
    assert np.all(np.abs(state.velocity) < 0.01)
    assert np.linalg.norm(testbot.data.joint("floating_base").qvel) < 0.01
    assert testbot.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number == 0


def test_effort_is_clipped_to_the_limits(testbot: MujocoRobot) -> None:
    testbot.set_mode(Mode.HOLD)
    testbot.set_mode(Mode.POLICY)
    testbot.write(command(DEFAULT_POSE + 0.1, kp=np.full(TESTBOT.num_joints, 1e6)))
    testbot.advance(1.0 / TESTBOT.rates.physics_hz)
    np.testing.assert_allclose(testbot.read().effort, EFFORT_LIMITS)
    assert np.all(np.abs(testbot.data.ctrl) <= EFFORT_LIMITS)


def test_setting_joint_positions_keeps_the_mode_and_its_targets(
    testbot: MujocoRobot,
) -> None:
    testbot.set_mode(Mode.HOLD)
    testbot.set_base_pose((0.0, 0.0, 1.0), IDENTITY_QUATERNION)
    moved = DEFAULT_POSE + 0.1
    testbot.set_joint_positions(moved)
    state = testbot.read()
    assert state.mode is Mode.HOLD
    np.testing.assert_array_equal(state.position, moved)
    testbot.advance(0.5)
    np.testing.assert_allclose(testbot.read().position, DEFAULT_POSE, atol=0.02)


def test_advance_takes_whole_physics_steps_only(testbot: MujocoRobot) -> None:
    for seconds in (0.0, -0.002, 0.001, 0.003, math.nan, math.inf):
        with pytest.raises(ValueError, match="positive whole number of physics steps"):
            testbot.advance(seconds)
    assert testbot.read().stamp_ns == 0


def test_the_probe_refuses_a_robot_that_is_not_connected() -> None:
    robot = MujocoRobot(TESTBOT)
    calls: tuple[Callable[[], object], ...] = (
        lambda: robot.advance(0.002),
        lambda: robot.set_base_pose((0.0, 0.0, 1.0), IDENTITY_QUATERNION),
        robot.release_base,
        lambda: robot.set_joint_positions(DEFAULT_POSE),
    )
    for call in calls:
        with pytest.raises(RuntimeError, match="is not connected"):
            call()


def lockstep_run() -> tuple[RobotState, bytes]:
    """Run testbot through a short policy, and return its last state and qpos."""
    robot = MujocoRobot(TESTBOT)
    robot.connect()
    robot.set_mode(Mode.HOLD)
    robot.set_mode(Mode.POLICY)
    for tick in range(50):
        offset = 0.2 * math.sin(2.0 * math.pi * tick / 25)
        robot.write(command(DEFAULT_POSE + np.array([offset, -2 * offset, offset] * 2)))
        robot.advance(0.02)
    state = robot.read()
    qpos = robot.data.qpos.tobytes()
    robot.close()
    return state, qpos


def test_lockstep_runs_are_deterministic() -> None:
    first, first_qpos = lockstep_run()
    second, second_qpos = lockstep_run()
    assert first_qpos == second_qpos
    assert first.stamp_ns == second.stamp_ns
    for field in ("position", "velocity", "effort"):
        assert getattr(first, field).tobytes() == getattr(second, field).tobytes()
    for field in ("orientation", "angular_velocity", "linear_acceleration"):
        assert (
            getattr(first.imu, field).tobytes() == getattr(second.imu, field).tobytes()
        )


def test_real_time_follows_the_clock_and_stops_on_close() -> None:
    robot = RealtimeMujocoRobot(TESTBOT)
    robot.connect()
    try:
        robot.set_mode(Mode.HOLD)
        robot.set_base_pose((0.0, 0.0, 1.0), IDENTITY_QUATERNION)
        start_s, start = time.monotonic(), robot.read().stamp_ns
        time.sleep(0.2)
        end, end_s = robot.read().stamp_ns, time.monotonic()
        simulated, elapsed = (end - start) / 1e9, end_s - start_s
        # Never ahead of the clock, and not far behind it.
        assert 0.5 * elapsed < simulated <= elapsed + 2.0 / TESTBOT.rates.physics_hz
        assert pelvis_height(robot) == pytest.approx(1.0, abs=0.001)
    finally:
        robot.close()
    assert not any(t.name == "mujoco-testbot" for t in threading.enumerate())


def test_a_failing_step_stops_real_time_with_a_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot = RealtimeMujocoRobot(TESTBOT)
    robot.connect()
    try:
        assert robot.read().faults == ()

        def fail() -> None:
            raise RuntimeError("the solver\ndiverged")

        monkeypatch.setattr(robot, "_step", fail)
        deadline = time.monotonic() + 1.0
        while not robot.read().faults and time.monotonic() < deadline:
            time.sleep(0.005)
        state = robot.read()
        assert state.faults == (
            Fault(
                code="simulation.stopped", message="RuntimeError: the solver diverged"
            ),
        )
        time.sleep(0.05)
        assert robot.read().stamp_ns == state.stamp_ns  # time stands still
        assert not any(t.name == "mujoco-testbot" for t in threading.enumerate())
    finally:
        robot.close()


# The kit catches broken variants of the adapter.


class ImuInWxyzOrder(MujocoRobot):
    """Skips the quaternion conversion, reporting MuJoCo's (w, x, y, z) order."""

    def read(self) -> RobotState:
        state = super().read()
        imu = replace(state.imu, orientation=xyzw_to_wxyz(state.imu.orientation))
        return replace(state, imu=imu)


class ReversedJointMap(MujocoRobot):
    """Reads the joints through a reversed index map."""

    def read(self) -> RobotState:
        state = super().read()
        return replace(
            state,
            position=state.position[::-1],
            velocity=state.velocity[::-1],
            effort=state.effort[::-1],
        )


def test_skipping_the_quaternion_conversion_fails_the_imu_check() -> None:
    robot = ImuInWxyzOrder(TESTBOT)
    robot.connect()
    match = (
        r"^IMU orientation with the base level: expected \(0\.000, 0\.000, 0\.000,"
        r" 1\.000\), .* 180\.0° apart .*: it is in \(w, x, y, z\) order, as MuJoCo"
        r" reports it"
    )
    try:
        with pytest.raises(AssertionError, match=match):
            check_imu(robot, robot)
    finally:
        robot.close()


def test_a_reversed_joint_map_fails_the_joint_order_check() -> None:
    robot = ReversedJointMap(TESTBOT)
    robot.connect()
    match = (
        r"^joint 'left_hip_pitch' did not move when commanded: .*; joint"
        r" 'right_ankle_pitch' moved -0\.\d+ rad instead: joint order may be wrong"
    )
    try:
        with pytest.raises(AssertionError, match=match):
            check_joint_order(robot, robot)
    finally:
        robot.close()
