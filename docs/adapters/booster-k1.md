# Booster K1 interface notes

Scope: Booster Robotics K1, 22-DoF joint layout (the only K1 layout; description name `k1_22dof`). Public sources only, pinned to the commits and doc versions in section 13. "Verify" marks anything the public sources do not settle.

## 1. Summary

| Item | Value |
|---|---|
| Transport | Gigabit Ethernet or Wi-Fi 6 to the robot; DDS over UDPv4; loopback when running on the robot [22][24][14] |
| Middleware | eProsima Fast DDS (bundled in the SDK as `booster_fastdds`), wire-compatible with ROS 2 Humble DDS; high-level calls are JSON RPC over DDS [15][16][28] |
| Low-level channels | `rt/joint_ctrl` (LowCmd, publish) and `rt/low_state` (LowState, subscribe); ROS 2 names `/joint_ctrl`, `/low_state` [1][17][6] |
| Low-level rate | Not published. Vendor examples publish `rt/joint_ctrl` at up to 500 Hz; shipped K1 RL policies run at 50 Hz [6][13] |
| DoF variants | One joint layout: 22 DoF = head 2 + arms 4x2 + legs 6x2, no waist [22][1]. Editions Geek / Education / Professional differ in compute and battery only [22]. Model variants: `K1_22dof` (serial ankles), `K1_22dof_parallel` (closed-loop ankles), `K1_locomotion` (head and arms fixed); a ZED-head URDF is documented but not in the repo [11][27] |
| SDK languages | C++17 (prebuilt static library, x86_64 and aarch64, Ubuntu 22.04/24.04) [21][28]; Python >= 3.10 bindings `booster_robotics_sdk_python` 1.6.3 (pybind11) [31]; ROS 2 message package `booster_interface` [5]. A separate high-level Python client `boosteros` exists (proprietary licence) [32] |
| Onboard computer | Geek: ARM SoC, 48 TOPS, 8 GB; Education: Jetson Orin NX 8 GB; Professional: Jetson AGX Orin 32 GB [22] |

## 2. Transport and middleware

- DDS init: `ChannelFactory::Instance()->Init(domain_id=0, "<network_ip>")`. The second argument is the IP of the local interface that faces the robot (`127.0.0.1` on the robot, e.g. `192.168.10.10` on an external PC); an empty string selects the default network configuration [16][21]. Python: `ChannelFactory.Instance().Init(0, ip)` [13][31].
- Topic naming: SDK topics carry the ROS 2 `rt/` prefix, so SDK `rt/low_state` is ROS 2 `/low_state` and SDK `rt/joint_ctrl` is ROS 2 `/joint_ctrl` [1][6][15].
- QoS: SDK `ChannelPublisher`/`ChannelSubscriber` default to best effort (`reliable=false`) [16]. The ROS 2 reference deployment subscribes `/low_state` BEST_EFFORT, KEEP_LAST depth 1, and publishes `joint_ctrl` RELIABLE, KEEP_LAST depth 1 [6].
- Low-level topics relevant to an adapter (minimum firmware in parentheses) [17][1]:
  - `rt/joint_ctrl` LowCmd (>= v1.0.0.0); accepted only in CUSTOM mode.
  - `rt/low_state` LowState: motors and IMU (>= v1.0.0.0).
  - `rt/robot_states` RobotStatesMsg: current mode, body control, actions (>= v1.3.1.1).
  - `rt/fall_down` FallDownState (>= v1.2.0.2).
  - `rt/odometer_state` Odometer {x, y, theta} (>= v1.3.1.1).
  - `rt/remote_controller_state`, `rt/button_event` (joystick and back-panel buttons).
  - ROS-shaped topics, which need the robot's ROS bridge (>= v1.7.1.0): `rt/imu/data` (sensor_msgs/Imu), `rt/joint_states`, `rt/odom`, `rt/tf` (ROS 2 subscription only).
- High-level RPC: `B1LocoClient` on channel `rt/LocoApiTopic` (`Init(robot_name)` appends `/<robot_name>`), integer `LocoApiId` plus JSON body, `int32_t` status code (0 success; 100 timeout; 400 bad request; 409 conflict; 429 too frequent; 500 internal; 501 refused; 502 state transition failed) [2][18][28]. ROS 2 equivalent: service `booster_rpc_service` (`booster_interface/srv/RpcService`, request `BoosterApiReqMsg{api_id, body}`, response `BoosterApiRespMsg{status, body}`) [5][6].
  - `ChangeMode` = 2000, body `{"mode": <int>}`; `GetMode` = 2017; `GetStatus` = 2018 returns `current_mode`, `current_body_control`, `current_actions` [2][6].
  - `GetSensors` = 2044 (IMU catalogue), `GetRobotModel` = 2046 (URDF/RBDL model catalogue), `GetRobotInfo` = 2022 (firmware version) [2].
- Network: wired, set the PC to static `192.168.10.10/24`, gateway `192.168.10.1`; robot at `192.168.10.102`; SSH `booster@192.168.10.102` [24][30]. The vendor's competition demo pins Fast DDS to UDPv4 on `127.0.0.1`, `192.168.10.101` and `192.168.10.102` with built-in transports (including shared memory) disabled [14]; the role of `.101` on K1 is not documented.
- Onboard vs external PC: the reference policy deployment runs on the robot, using the robot's ROS 2 Humble and `booster_interface` (`source /opt/booster/BoosterRos2Interface/install/setup.bash`, Python 3.10) [10][28]. SDK clients also run on an external PC given the robot-facing IP (`--net`, default `127.0.0.1`) [13]. Wired or onboard is the safe default for joint-level loops; Wi-Fi latency is unmeasured.
- Entering joint-level control (reference sequence [6], rules [23][25]):
  1. Robot in PREP (standing) or DAMP (on a hoist). CUSTOM can be entered only from PREP or DAMP.
  2. Wait for the first `rt/low_state`; refuse if no state arrives.
  3. Publish one SERIAL LowCmd that holds the measured joint positions with hold gains, while the vendor controller still owns the robot. The low-level controller keeps this command across the switch.
  4. `ChangeMode(kCustom)` (mode 3), then poll `GetStatus` until `current_mode == 3` (the deployment retries 20 times, 0.5 s apart).
  5. Stream `rt/joint_ctrl`. Firmware >= v1.8.0.9 holds a standing pose after entering CUSTOM until the first valid `rt/joint_ctrl` arrives [20][18].
  6. Leave with `ChangeMode(kDamping)` or `ChangeMode(kPrepare)`.
- Partial alternative: `UpperBodyCustomControl(true)` (2030) lets `rt/joint_ctrl` drive joints 2..9 while the vendor gait keeps balancing the legs. The SDK `ArmController` writes kp = kd = 0 in all other slots [4][19]. This is outside the port's mode set but useful for arm-only work.

## 3. Messages

IDL in `booster_interface::msg` (SDK headers [3]; identical ROS 2 `.msg` definitions [5]). No field carries units in the IDL; the units below come from how the vendor's code and models use the values. LowState has no header or timestamp.

LowCmd

| Field | Type | Meaning |
|---|---|---|
| `cmd_type` | enum CmdType `PARALLEL = 0`, `SERIAL = 1` (IDL default PARALLEL) | Representation of `motor_cmd`: SERIAL = joint space (ankle pitch/roll), PARALLEL = actuator space (ankle crank motors) |
| `motor_cmd` | MotorCmd[22] for K1 (`kJointCntK1`) | Indexed by `JointIndexK1`; a wrong count or layout can address the wrong actuator or be rejected [1] |

MotorCmd

| Field | Type | Unit | Notes |
|---|---|---|---|
| `mode` | uint8 (ROS msg: int8) | none | Undocumented; all reference code sends 0 [6][13][31] |
| `q` | float32 | rad | Position target |
| `dq` | float32 | rad/s | Velocity target |
| `tau` | float32 | N·m | Feed-forward torque |
| `kp` | float32 | N·m/rad | |
| `kd` | float32 | N·m·s/rad | |
| `weight` | float32 | 0..1 | Not documented. The T1 deployment comments "weight is not effective in custom mode" and sends 0.0 [13]; booster_deploy sends 0.0 [6]; the SDK Python example sends 1.0 [31] |

LowState: `imu_state` (ImuState), `motor_state_parallel` (MotorState[], actuator space), `motor_state_serial` (MotorState[], joint space) [3][5].

MotorState

| Field | Type | Unit / meaning |
|---|---|---|
| `mode` | uint8 (ROS msg: int8) | Undocumented |
| `q`, `dq`, `ddq` | float32 | rad, rad/s, rad/s² |
| `tau_est` | float32 | Estimated torque, N·m |
| `temperature` | uint8 (ROS msg: int8) | Motor temperature; ROS comment gives range -100..150, unit not stated (verify °C) |
| `lost` | uint32 | Motor packet-loss count |
| `reserve[2]` | uint32[2] | `[0]` motor error flag bits (0..255), `[1]` current motor communication frequency (0..800) (ROS msg comment, translated) |

ImuState: `rpy[3]` (roll, pitch, yaw, rad), `gyro[3]` (x, y, z), `acc[3]` (x, y, z) [3][5]. See section 7.

Serial vs parallel on K1: all slots except the ankles are directly driven, so 0..13 and 16..19 hold the same values in both arrays. Slots 14/15 (left) and 20/21 (right) hold ankle pitch/roll in SERIAL and the upper/lower crank motors in PARALLEL [1][22][11]. The firmware converts between the two. Two vendor notes bear on gains:
- booster_deploy README: for parallel-actuated joints the `kd` sent is applied directly at the motors, so the simulator `Kd` must not be reused; motor-side `Kd = 2·ζ·J_eq·(2π·f_n)` [10].
- booster_gym (T1, same ankle scheme): in SERIAL mode it sends `kp = 0`, `q = measured q` and `tau = clip(kp·(q* − q))` for the ankle slots, with the comment "Use series-parallel conversion for torque to avoid non-linearity" [13].

Other messages: `RobotStatesMsg {int32 current_mode, int32 current_body_control, int32[] current_actions}`; `FallDownState {fall_down_state: IS_READY 0, IS_FALLING 1, HAS_FALLEN 2, IS_GETTING_UP 3; bool is_recovery_available}`; `Odometer {x, y, theta}` [3][5].

## 4. Rates and timing

- The firmware joint-loop rate and the `rt/low_state` publish rate are not published. Measure both on the unit (booster_deploy prints `low_state_handler` frequency at exit [6]).
- K1 reference deployment: policy period 0.02 s (50 Hz), one `joint_ctrl` publish per policy step; the "standing" preparation interpolates for 1 s in 2 ms steps (500 Hz) [6][8].
- T1 reference deployment: publish thread at 2 ms (500 Hz) with a 0.8/0.2 first-order filter on targets, policy every 10 publish ticks (50 Hz) [13].
- SDK `ArmController` publishes about 100 Hz [4][19].
- Training: K1 tracking tasks use sim dt 0.005 s with decimation 4 (50 Hz) and a random actuator delay of 2..8 physics steps (10..40 ms) [12]. The MJCF uses a 1 ms timestep [11].
- `MotorState.reserve[1]` reports each motor's communication frequency (range up to 800) [5].
- LowState has no timestamp: stamp on receipt with a monotonic clock.
- RPC calls are blocking request/response (examples wait up to 5 s for the service; the deployment retries mode changes for up to about 10 s) [6][21]. Keep them off the control thread.

## 5. Modes and state machines

Vendor `RobotMode` [2][18]: `kUnknown -1`, `kDamping 0`, `kPrepare 1`, `kWalking 2`, `kCustom 3`, `kSoccer 4` (K1 and T1 only). The product manual also defines PROTECT [23].

| Vendor mode | Behaviour [23] | Allowed transitions [23][25] |
|---|---|---|
| DAMP | All joints damped: they resist motion but neither hold nor seek a position; the robot cannot stand | to PREP (not directly to WALK); to CUSTOM |
| PREP | Assumes and holds a standing posture; strong position hold, no active balance | to any mode |
| WALK | Vendor locomotion controller (walking, turning, head, actions); actively balances | to any mode; enter only from PREP while standing |
| CUSTOM | All joints, legs included, follow `rt/joint_ctrl` | enter only from PREP or DAMP; leave only to DAMP or PREP |
| PROTECT | Entered automatically on limit violations or a fall; joints as in DAMP | not in `RobotMode`; how it is reported is unverified |

`BodyControl` (reported by `GetStatus` and `rt/robot_states`) gives the active controller within a mode (damping 1, prepare 2, humanlike gait 3, ..., custom 6, get-up 7, WBC gait 12, ...) [2].

Mapping onto the port:

| Port mode | K1 implementation |
|---|---|
| PASSIVE (zero torque) | CUSTOM with kp = kd = tau = 0 in every slot. There is no whole-body zero-torque vendor mode (`ZeroTorqueDrag`, 2026, is an arm action). The robot collapses: hoist only. Verify that zero gains give zero torque, ankles included. |
| HOLD | CUSTOM, q* = latched measured q, dq* = 0, hold gains. booster_deploy uses `K1_CFG.prepare_state` gains for this hand-off: head 40/1.5; arms kp 40, 50, 20, 20 with kd 0.5, 1.5, 0.2, 0.2; legs kp 350, 350, 180, 350, 250, 250 with kd 7.5, 7.5, 3, 5.5, 5, 5 [7]. Vendor PREP also holds, but it drives to its own standing pose rather than the current one. |
| POLICY | CUSTOM, commands from `write()` streamed to `rt/joint_ctrl`. |
| DAMP (via `operator_stop()`) | `ChangeMode(kDamping)`. Until the RPC completes, publish kp = 0 with damping-only kd in CUSTOM. Physical stops: joystick L2+Back (Xbox LT+Back), back-panel F1 (DAMP by default, configurable), app soft e-stop [26][33]. |

PREP, WALK and SOCCER are vendor-owned states outside the port's mode set. Report them as not-under-adapter-control, and require PREP or DAMP before any port mode change.

## 6. Joints

Order = SDK `JointIndexK1` = description order (identity index map) [1]. Canonical names follow the joint function, using the product-manual names [22] and the axes in the URDF [11]. Limits, velocity and effort come from `K1_22dof.urdf` (the MJCF has the same ranges and force ranges) [11]. Actuator models are from booster_train [12]. The spec-page column lists the manual's limits in degrees for comparison [22].

| Idx | Canonical | URDF / vendor name | SDK enum | Position [rad] | Vel [rad/s] | Effort [N·m] | Actuator | Spec page [deg] |
|---|---|---|---|---|---|---|---|---|
| 0 | head_yaw | aahead_yaw_joint | kHeadYaw | [-1.012, 1.012] | 7.85 | 6.0 | HT4438 | -59..59 |
| 1 | head_pitch | aahead_pitch_joint | kHeadPitch | [-0.314, 0.794] | 7.85 | 6.0 | HT4438 | -19..49 |
| 2 | left_shoulder_pitch | aaleft_shoulder_pitch_joint | kLeftShoulderPitch | [-2.932, 1.196] | 33.51 | 14.0 | R14 | -169..69 |
| 3 | left_shoulder_roll | left_shoulder_roll_joint | kLeftShoulderRoll | [-1.642, 1.629] | 33.51 | 14.0 | R14 | -94..94 |
| 4 | left_shoulder_yaw | left_elbow_pitch_joint | kLeftElbowPitch | [-1.885, 1.885] | 33.51 | 14.0 | R14 | -109..109 |
| 5 | left_elbow | left_elbow_yaw_joint | kLeftElbowYaw | [-2.242, 0.816] | 33.51 | 14.0 | R14 | -129..39 |
| 6 | right_shoulder_pitch | aaright_shoulder_pitch_joint | kRightShoulderPitch | [-2.932, 1.196] | 33.51 | 14.0 | R14 | -169..69 |
| 7 | right_shoulder_roll | right_shoulder_roll_joint | kRightShoulderRoll | [-1.642, 1.629] | 33.51 | 14.0 | R14 | -94..94 |
| 8 | right_shoulder_yaw | right_elbow_pitch_joint | kRightElbowPitch | [-1.885, 1.885] | 33.51 | 14.0 | R14 | -109..109 |
| 9 | right_elbow | right_elbow_yaw_joint | kRightElbowYaw | [-0.816, 2.242] | 33.51 | 14.0 | R14 | -39..129 |
| 10 | left_hip_pitch | left_hip_pitch_joint | kLeftHipPitch | [-2.958, 2.226] | 14.66 | 68.0 | E6408 | -170..128 |
| 11 | left_hip_roll | left_hip_roll_joint | kLeftHipRoll | [-0.375, 1.536] | 12.57 | 43.0 | E4315 | -22..89 |
| 12 | left_hip_yaw | left_hip_yaw_joint | kLeftHipYaw | [-1.012, 1.012] | 17.59 | 38.3 | E4310 | -59..59 |
| 13 | left_knee | left_knee_pitch_joint | kLeftKneePitch | [0.0, 2.321] | 12.57 | 112.0 | E6416 | 0..133 |
| 14 | left_ankle_pitch | left_ankle_pitch_joint | kCrankUpLeft (PARALLEL) | [-0.87, 0.345] | 17.59 | 38.3 | 2x E4310 linkage | Ankle Up motor -17..38 |
| 15 | left_ankle_roll | left_ankle_roll_joint | kCrankDownLeft (PARALLEL) | [-0.345, 0.345] | 17.59 | 38.3 | 2x E4310 linkage | Ankle Down motor -16..41 |
| 16 | right_hip_pitch | right_hip_pitch_joint | kRightHipPitch | [-2.958, 2.226] | 14.66 | 68.0 | E6408 | -170..128 |
| 17 | right_hip_roll | right_hip_roll_joint | kRightHipRoll | [-1.536, 0.375] | 12.57 | 43.0 | E4315 | -89..22 |
| 18 | right_hip_yaw | right_hip_yaw_joint | kRightHipYaw | [-1.012, 1.012] | 17.59 | 38.3 | E4310 | -59..59 |
| 19 | right_knee | right_knee_pitch_joint | kRightKneePitch | [0.0, 2.321] | 12.57 | 112.0 | E6416 | 0..133 |
| 20 | right_ankle_pitch | right_ankle_pitch_joint | kCrankUpRight (PARALLEL) | [-0.87, 0.345] | 17.59 | 38.3 | 2x E4310 linkage | Ankle Up motor -17..38 |
| 21 | right_ankle_roll | right_ankle_roll_joint | kCrankDownRight (PARALLEL) | [-0.345, 0.345] | 17.59 | 38.3 | 2x E4310 linkage | Ankle Down motor -16..41 |

Notes:
- Arm naming: the vendor's "elbow pitch" slot (4/8) rotates about the upper-arm axis, so it is a humeral rotation, and the "elbow yaw" slot (5/9) is elbow flexion. URDF forward kinematics at the standing pose puts the slot-4 axis near vertical and the slot-5 axis lateral. The spec page names them "Shoulder Yaw" and "Elbow", and `ArmController` names them `kLeftYaw` and `kLeftElbow` [22][19][11]. Canonical names follow the function: `*_shoulder_yaw`, `*_elbow`.
- Parallel linkage: each ankle is two E4310 crank motors driving the foot through two rods ("Ankle Up / Ankle Down"); the closed loop is modelled in `K1_22dof_parallel.xml` (`*_ankle_drive_a/b_joint` hinges, ball-jointed rods, `equality/connect`) [11][22]. In SERIAL mode the adapter sees ankle pitch/roll; in PARALLEL mode it sees the cranks. Only the ankles are parallel; K1 has no waist [1].
- URDF effort values are actuator peak ratings. booster_deploy clips lower in its K1 configs (walk: hips 30/20/15, knee 35, ankle 24/15; tracking: 30/35/20/40/20/20) [7][8][9]. Which limit the firmware enforces is unverified.
- Isaac Lab and booster_deploy also use a breadth-first "sim" order (`K1_CFG.sim_joint_names`); policies trained in Isaac Lab expect that order [7].

## 7. IMU

- Source: `LowState.imu_state` [3][5].
  - `rpy`: roll, pitch, yaw in rad.
  - `gyro`: x, y, z body rates. Reference deployments feed it to the policy unchanged as body-frame angular velocity in rad/s [6][13].
  - `acc`: x, y, z. Units and whether gravity is included are not documented (expected m/s² specific force; verify).
- RPY to quaternion: both reference deployments use the Z-Y-X composition `R = Rz(yaw)·Ry(pitch)·Rx(roll)`, i.e. `q = q_z(yaw) ⊗ q_y(pitch) ⊗ q_x(roll)`. booster_gym does this through `rotate_vector_inverse_rpy` [13]; booster_deploy uses Isaac Lab `quat_from_euler_xyz`, which returns (w, x, y, z) [6]. Convert, then reorder to (x, y, z, w) for the port.
- Yaw reference and drift are not documented. The shipped policies use only projected gravity and the gyro [8].
- Mounting: the MJCF places site `imu` at the `trunk` origin with identity orientation and attaches `framequat` and `gyro` sensors to it; the URDF has no IMU link [11]. The IDL enumerates IMU model `IMU_MODEL_HI13R4N` [3]. `GetSensors` (2044) returns the configured IMU catalogue (`index`, `name`, `mount_position`, `model`) [2][3]; query it on the unit.
- The robot must stay still for about 1 min after power-on while the IMU initialises [33].
- Alternative source: `rt/imu/data` (sensor_msgs/Imu, quaternion) through the ROS bridge, firmware >= v1.7.1.0 [17]. Its frame id, rate and quaternion convention are unverified.

## 8. Other sensors

- Head depth camera on the head pitch link [22]. The fitted camera depends on edition and production date:
  - URDF frame `head_booster_stereo_rgb_link`: parent `aahead_pitch_link`, xyz (0.060138, 0.0351, 0.092774) m, rpy (-π/2, 0, -π/2) (optical convention) [11].
  - URDF frame `head_realsense_rgb_link`: xyz (0.053617, -0.0115, 0.10231) m, rpy (-1.835388, 0, -π/2), i.e. optical and pitched about 15° down [11].
  - The V1.0 manual (2025) states that K1 uses a ZED, with the ZED ROS 2 wrapper started at boot [30]. The docs list a `K1_22dof-ZED.urdf` that the repo does not contain; the repo does contain `Head_2_ZED.STL` [27][11].
  - The vendor RoboCup demo defaults to D-Robotics StereoNet topics (`/StereoNetNode/rectified_image`, `/StereoNetNode/stereonet_depth`, `.../camera_info`), with alternative topic sets for RealSense, Orbbec and ZED. It ships one camera-to-head extrinsic and intrinsics (fx = fy = 206.45, cx = 224.96, cy = 242.72; resolution not stated) [14].
  - `CameraClient::GetCameras` returns the robot's camera catalogue as JSON. `X5CameraClient` switches resolution modes on compatible K1 camera editions (firmware >= v1.5.0.9) [29].
- Also present: microphone array, speaker, 1 RGB LED, power button and three interaction buttons (WALK, STAND, F1), battery state topic, planar odometry `rt/odometer_state` [22][26][17].
- No foot contact or force sensors appear in any public source.

## 9. Mapping onto the Robot port

- `description`: `description.yaml` (22 joints, SDK order, SERIAL joint space).
- `capabilities()`:
  - `effort_command = True`: MotorCmd `tau`.
  - `temperature = True`: MotorState `temperature`.
  - `max_rate_hz = 500.0`: highest rate in vendor examples; verify.
- `connect()`:
  - `ChannelFactory.Init(0, ip)`, then `B1LocoClient.Init()` and `WaitForService(5000)`.
  - Subscribe `rt/low_state`, `rt/robot_states` and `rt/fall_down`. Each callback copies into a double buffer stamped with a monotonic clock.
  - Create the `rt/joint_ctrl` publisher and wait for the first LowState.
  - Do not change the robot mode on connect.
  - ROS 2 alternative: `/low_state`, `joint_ctrl`, service `booster_rpc_service` [6].
- `close()`: if CUSTOM is active, switch to DAMP (or PREP if the robot is standing and stable) before closing the channels. The reference deployments switch to Damping or Walking on exit [6][13].
- `read()` (non-blocking, latest buffer):
  - `position`, `velocity`, `effort` from `motor_state_serial[i].q`, `.dq`, `.tau_est` for i = 0..21.
  - `imu.orientation`: RPY to quaternion (section 7), reordered to xyzw.
  - `imu.angular_velocity`: `gyro`.
  - `imu.linear_acceleration`: `acc` (verify units and gravity).
  - `temperature` from `motor_state_parallel[i].temperature` (per physical motor; verify what the serial array reports for ankles).
  - `faults`: `reserve[0] != 0`, rising `lost`, communication frequency `reserve[1]` dropping, `fall_down_state >= IS_FALLING`, PROTECT.
  - `mode`: from `rt/robot_states.current_mode` plus the adapter's own sub-state inside CUSTOM.
- `write(JointCommand)`:
  - Fill a preallocated LowCmd: `cmd_type = SERIAL`, 22 MotorCmd with `q`, `dq`, `tau`, `kp`, `kd`, `mode = 0`, `weight = 0.0` (matching the reference deployments).
  - Clamp `q` to the position limits and `tau` to the effort limits.
  - Publish from a fixed-rate sender thread that re-sends the latest command, so `write()` only swaps a buffer.
  - The firmware ignores the command unless the robot is in CUSTOM.
- `set_mode(mode)`: PASSIVE, HOLD and POLICY all live inside CUSTOM; they differ only in what the sender publishes (section 5). From PREP or DAMP, publish the HOLD command, then `ChangeMode(kCustom)` on a worker thread and confirm with `GetStatus`. From WALK, go to PREP first. Reject the change if no fresh LowState is available.
- `operator_stop()`: immediately publish kp = 0 with damping kd in every slot, then `ChangeMode(kDamping)` on a worker thread.
- Conversions:
  - RPY (rad) to quaternion xyzw.
  - Identity index map, but canonical names differ from vendor names for slots 4/5/8/9 (shoulder yaw, elbow) and 14/15/20/21 (crank names in PARALLEL).
  - Units already SI as used by the vendor stack (verify `acc` and `temperature`).
  - Stamp state on receipt.
- Ankle gains: the published K1 gains (ankle kp 65, kd 1) are SERIAL-slot values from the walking deployment [8]. Per the vendor README, kd on parallel-actuated joints acts motor-side [10]. Treat ankle gain semantics as unverified.

## 10. Policy inference formats

- Runtime (booster_deploy) [10]: the checkpoint suffix picks the backend. `.pt`, `.jit` and `.torchscript` load with `torch.jit.load`; the K1 walk policy uses the `actor` submodule. `.onnx` runs on onnxruntime's CPU execution provider and must have exactly one input and one output.
- K1 walking (`tasks/locomotion/robots/k1/models/k1_walk.pt`, task `k1_walk`) [8]:
  - Network: TorchScript MLP 690 -> 512 -> 256 -> 128 -> 20, ELU activations.
  - Rate: 50 Hz (`policy_dt = 0.02`).
  - Per-frame observation (69): base angular velocity (3, body, rad/s), projected gravity (3), command (vx, vy, vyaw) (3), q − q_default (20), 0.1·dq (20), last action (20). The 20 policy joints exclude the head and use the order below.
  - History: 10 frames, oldest first, seeded by repeating the first frame, flattened to 690; observations clipped to ±100.
  - Action (20, clipped to ±100): target = q_default + 0.25·action, mapped to SDK order. The action for `right_elbow_pitch_joint` (canonical `right_shoulder_yaw`) has 0.2 subtracted before scaling. Head targets stay at default.
  - Filter: the target is low-pass filtered as `filtered = filtered + 0.8·(target − filtered)`.
  - Command range: vx in [-0.3, 1.6] m/s, |vy| <= 0.3 m/s, |vyaw| <= 1.8 rad/s.
  - Safety: the policy stops if projected gravity z > -0.5.
  - Default pose and gains: see `description.yaml`.
  - Policy joint order: aaleft_shoulder_pitch, aaright_shoulder_pitch, left_hip_pitch, right_hip_pitch, left_shoulder_roll, right_shoulder_roll, left_hip_roll, right_hip_roll, left_elbow_pitch, right_elbow_pitch, left_hip_yaw, right_hip_yaw, left_elbow_yaw, right_elbow_yaw, left_knee_pitch, right_knee_pitch, left_ankle_pitch, right_ankle_pitch, left_ankle_roll, right_ankle_roll (each with the `_joint` suffix).
- K1 motion tracking (BeyondMimic; tasks `k1_mj2`, `k1_fight`; TorchScript `.pt` plus `.npz` motions) [9][12]:
  - Observation (119, sim joint order): reference joint pos (22), reference joint vel (22), anchor orientation error (6, first two columns of the rotation matrix), base angular velocity (3), q − q_default (22), dq (22), last action (22).
  - Action: target = default + action·(0.25·effort_limit/kp) per joint.
  - Rate: 50 Hz; trained at sim dt 0.005 s with decimation 4.
  - The anchor body is `trunk`, with yaw zeroed at start.
- Vendor-side execution: `LoadCustomTrainedTraj` / `ActivateCustomTrainedTraj` (K1 and T2, firmware >= v1.5.0.9) run an ONNX trajectory and an ONNX control model inside the vendor controller, with `joint_order` set to `kMuJoCo` or `kIsaacLab` and per-joint `action_scale`, `kp`, `kd` [18].
- booster_gym is T1-only (TorchScript, 47 observations, 12 actions) [13].

## 11. Public model files and licences

| Source | Files | Contents | Licence |
|---|---|---|---|
| booster_assets [11] | `robots/K1/K1_22dof.urdf` | SolidWorks-exported URDF: base `trunk`, 22 revolute joints (serial ankles), limits, inertias (total 19.67 kg), two head camera frames | BSD-3-Clause (LICENSE file; `setup.cfg` says MIT) |
| | `robots/K1/K1_22dof.xml` | MJCF: serial ankles, armature, `motor` actuators with force ranges, `imu` site with `framequat` and `gyro`, 1 ms timestep, floor | BSD-3-Clause |
| | `robots/K1/K1_22dof_parallel.xml` | MJCF with the closed-loop ankle linkage; ankle actuators on the crank hinges | BSD-3-Clause |
| | `robots/K1/K1_locomotion.urdf` | Head and arm joints fixed (12 actuated leg joints) | BSD-3-Clause |
| | `robots/K1/meshes/*.STL` (52 files, 21 MB) | Visual and collision meshes, including crank, rod and ZED-head meshes | BSD-3-Clause |
| | `motions/K1/*.csv` | Retargeted motions, 50 Hz, root pose (xyz + quaternion xyzw) and 22 joint positions | BSD-3-Clause |
| booster_deploy [7][8][9] | `robots/k1.py`, `tasks/*/robots/k1/` | Deployment configs, `k1_walk.pt`, BeyondMimic `.pt` and `.npz` | Apache-2.0 (short header; GitHub shows NOASSERTION) |
| booster_train [12] | `assets/robots/booster_k1.py`, `actuator.py` | Isaac Lab articulation and actuator models (uses the booster_assets URDF) | Apache-2.0 (short header); includes MIT code from whole_body_tracking |
| Vendor wiki (V1.0 manual) [30] | `K1.zip` (URDF/MJCF), `k1.zip` (USD), Webots simulation archive | Download links behind a Feishu login | Not stated |

## 12. To verify on the real unit

- [ ] Firmware version (`GetRobotInfo`) and SDK compatibility. Docs and the deploy README disagree on the minimum (v1.4 vs v1.7.2).
- [ ] `rt/low_state` publish rate, jitter and latency, onboard and over Ethernet.
- [ ] Highest `rt/joint_ctrl` rate the firmware accepts, and its internal joint-loop rate.
- [ ] Behaviour when `rt/joint_ctrl` stops in CUSTOM: last command latched, timeout, or fallback.
- [ ] SERIAL mode on ankles: whether kp, kd and tau act in joint space or motor space. Compare a joint-space step response with the README statement; pick `side` accordingly.
- [ ] kp = kd = tau = 0 in CUSTOM gives zero torque on every joint (PASSIVE).
- [ ] Semantics of MotorCmd `mode` and `weight` in CUSTOM.
- [ ] `motor_state_serial` vs `motor_state_parallel` for ankle `tau_est`, `temperature` and error flags.
- [ ] `acc`: units, frame and whether gravity is included. Gyro units and frame.
- [ ] RPY convention on hardware (Z-Y-X) and yaw drift/reference. Physical IMU mount (`GetSensors`).
- [ ] `temperature` unit and limits; meaning of `reserve[0]` error bits.
- [ ] How PROTECT mode is reported (`current_mode`, `body_control`, `fall_down`).
- [ ] Whether CUSTOM can exit directly to WALK (the deployment does this; the manual forbids it).
- [ ] Which head camera is fitted, its topics, intrinsics and extrinsics (`GetCameras`).
- [ ] Joint limits enforced by firmware vs URDF (elbow upper limit 0.816 vs 0.681 rad; head pitch; ankle motor-space limits).
- [ ] Effort limits enforced by firmware (URDF peak vs the deployment's lower clips); hip roll velocity limit (12.57 vs 12.15 rad/s).
- [ ] Role of `192.168.10.101` in the DDS whitelist; DDS behaviour over Wi-Fi.
- [ ] `rt/robot_states` rate and whether `current_mode` updates immediately after `ChangeMode`.

## 13. Sources

GitHub repositories at the commits below, as of 2026-10-04; docs.booster.tech doc-centre release 0.0.20, C++ SDK v1.6.3 (2026-09-15).

1. booster_robotics_sdk @ 87a9a26, `include/booster/robot/b1/b1_api_const.hpp`: https://github.com/BoosterRobotics/booster_robotics_sdk/blob/87a9a26d06a94ddd50968e7576f1170f6fa5a80f/include/booster/robot/b1/b1_api_const.hpp
2. booster_robotics_sdk @ 87a9a26, `include/booster/robot/common/robot_shared.hpp`, `include/booster/robot/b1/b1_loco_api.hpp`: https://github.com/BoosterRobotics/booster_robotics_sdk/tree/87a9a26d06a94ddd50968e7576f1170f6fa5a80f/include/booster/robot
3. booster_robotics_sdk @ 87a9a26, IDL headers `include/booster/idl/b1/{LowCmd,MotorCmd,LowState,MotorState,ImuState,ImuInfo,FallDownState,RobotStates}.h`: https://github.com/BoosterRobotics/booster_robotics_sdk/tree/87a9a26d06a94ddd50968e7576f1170f6fa5a80f/include/booster/idl/b1
4. booster_robotics_sdk @ 87a9a26, `include/booster/robot/b1/arm_controller.hpp`, `example/low_level/`: https://github.com/BoosterRobotics/booster_robotics_sdk/tree/87a9a26d06a94ddd50968e7576f1170f6fa5a80f/example/low_level
5. booster_robotics_sdk_ros2 @ 5f9182e, `booster_ros2_interface/msg/`, `srv/`: https://github.com/BoosterRobotics/booster_robotics_sdk_ros2/tree/5f9182e1a1b704f65e0065c83a3b6be2d066f3b3/booster_ros2_interface
6. booster_deploy @ 7bb1462, `booster_deploy/controllers/booster_robot_controller.py`: https://github.com/BoosterRobotics/booster_deploy/blob/7bb1462e2ac742eba5c01659b815e2e99fbe617f/booster_deploy/controllers/booster_robot_controller.py
7. booster_deploy @ 7bb1462, `booster_deploy/robots/k1.py`: https://github.com/BoosterRobotics/booster_deploy/blob/7bb1462e2ac742eba5c01659b815e2e99fbe617f/booster_deploy/robots/k1.py
8. booster_deploy @ 7bb1462, `tasks/locomotion/locomotion.py`, `tasks/locomotion/robots/k1/__init__.py`, `tasks/locomotion/robots/k1/models/k1_walk.pt`: https://github.com/BoosterRobotics/booster_deploy/tree/7bb1462e2ac742eba5c01659b815e2e99fbe617f/tasks/locomotion
9. booster_deploy @ 7bb1462, `tasks/beyond_mimic/beyond_mimic.py`, `tasks/beyond_mimic/robots/k1/`: https://github.com/BoosterRobotics/booster_deploy/tree/7bb1462e2ac742eba5c01659b815e2e99fbe617f/tasks/beyond_mimic
10. booster_deploy @ 7bb1462, `README.md`, `booster_deploy/utils/policy_runner.py`: https://github.com/BoosterRobotics/booster_deploy/blob/7bb1462e2ac742eba5c01659b815e2e99fbe617f/README.md
11. booster_assets @ 12a9751, `robots/K1/` (URDF, MJCF, meshes), `README.md`, `LICENSE`, `setup.cfg`: https://github.com/BoosterRobotics/booster_assets/tree/12a97516f57469078707afb587a5f84c1cb110d1/robots/K1
12. booster_train @ 8bb5b53, `source/booster_train/booster_train/assets/robots/{booster_k1,actuator}.py`, `tasks/manager_based/beyond_mimic/robots/k1/`: https://github.com/BoosterRobotics/booster_train/tree/8bb5b53824e0fbcfe69a5cb78a81f4bddb3e2a6e/source/booster_train/booster_train
13. booster_gym @ da396a0, `deploy/deploy.py`, `deploy/utils/{command,rotate}.py`, `deploy/configs/T1.yaml` (T1 reference): https://github.com/BoosterRobotics/booster_gym/tree/da396a06d6eed99e2de72d7749c48ee8748950f9/deploy
14. robocup_demo @ fa4f31c, `configs/fastdds.xml`, `src/vision/config/vision.yaml`: https://github.com/BoosterRobotics/robocup_demo/tree/fa4f31ccb4428a9e60e90b4f54eb80499caeb0ca
15. Booster docs, C++ SDK Overview: https://docs.booster.tech/docs/developer-guide/cpp/sdk-overview/
16. Booster docs, SDK Architecture and Communication Model: https://docs.booster.tech/docs/developer-guide/cpp/architecture/
17. Booster docs, Low-Level Topics: https://docs.booster.tech/docs/developer-guide/cpp/low-level-topics/
18. Booster docs, Motion-Control Interfaces (RPC): https://docs.booster.tech/docs/developer-guide/cpp/rpc/motion/
19. Booster docs, C++ Controller Wrappers: https://docs.booster.tech/docs/developer-guide/cpp/controllers/
20. Booster docs, C++ SDK Changelog (v1.6.3, 2026-09-15): https://docs.booster.tech/docs/developer-guide/cpp/changelog/
21. Booster docs, C++ Quick Start: https://docs.booster.tech/docs/developer-guide/cpp/quick-start/
22. Booster docs, K1 Specifications: https://docs.booster.tech/docs/product-manual/k1/getting-started/specifications/
23. Booster docs, K1 Modes: https://docs.booster.tech/docs/product-manual/k1/basic-operations/modes/
24. Booster docs, K1 Connect to Robot: https://docs.booster.tech/docs/product-manual/k1/basic-operations/connect-robot/
25. Booster docs, K1 Safety and Warnings: https://docs.booster.tech/docs/product-manual/k1/getting-started/safety-warnings/
26. Booster docs, K1 Body Operations and Joystick Control: https://docs.booster.tech/docs/product-manual/k1/basic-operations/body-operations/ , https://docs.booster.tech/docs/product-manual/k1/basic-operations/joystick-control/
27. Booster docs, booster_assets User Guide: https://docs.booster.tech/docs/developer-guide/open-source/booster-assets/
28. Booster docs, C++ API Reference and booster_deploy User Guide: https://docs.booster.tech/docs/developer-guide/cpp/cpp-api/ , https://docs.booster.tech/docs/developer-guide/open-source/booster-deploy/
29. Booster docs, Camera Interfaces: https://docs.booster.tech/docs/developer-guide/cpp/rpc/camera/
30. K1 Instruction Manual V1.0 (vendor Feishu wiki `AHctwLAYjiyThnkZctIc6iYunmf`, login-gated: https://booster.feishu.cn/wiki/AHctwLAYjiyThnkZctIc6iYunmf). Public PDF export (2025-07-25) hosted by a distributor: https://cdn-reichelt.de/documents/datenblatt/A300/BOOSTER%20K1%20-%20INSTRUCTION%20MANUAL-V1.0%20-%20FEISHU%20DOCS.pdf
31. PyPI `booster_robotics_sdk_python` 1.6.3 (wheel `python_example/sdk_pybind_low_level.py`, `sdk_pybind_ros_rt_topics_example.py`): https://pypi.org/project/booster_robotics_sdk_python/1.6.3/
32. Booster docs, Python SDK (boosteros) Overview: https://docs.booster.tech/docs/developer-guide/booster-os-python-sdk/sdk-overview/ ; PyPI `boosteros` 1.2.0: https://pypi.org/project/boosteros/
33. Booster docs, K1 Quick Start: https://docs.booster.tech/docs/product-manual/k1/getting-started/quick-start/
