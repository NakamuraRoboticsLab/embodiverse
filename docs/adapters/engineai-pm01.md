# EngineAI PM01 interface notes

Target: PM01 2026 Education Edition (vendor model ids `pm01_edu` in the native SDK, `pm_v2` in the ROS 2 workspace and simulation runtime). The Navigation Edition has the same joints and adds a Livox Mid-360 LiDAR [17][20]. All facts below come from public vendor repositories at the commits listed in Sources, and from the vendor product page.

## 1. Summary

| Item | Value |
|---|---|
| Transport | Ethernet, robot internal network 192.168.0.0/24 via the rear port; robot Wi-Fi hotspot also documented [7] |
| Middleware | ROS 2 Humble, Cyclone DDS (`rmw_cyclonedds_cpp`), `ROS_DOMAIN_ID=69`, custom `interface_protocol` messages [1][2]. Native SDK: in-process C++ runner plugins; LCM only links the controller to MuJoCo in simulation [7][8] |
| Low-level rate | 500 Hz (motor task period 0.002 s; `/hardware/joint_state` published every 0.002 s) [8] |
| DoF variants | 24 joint motors: 2 × 6 leg, 1 waist yaw, 2 × 5 arm, 1 head yaw [12][20]. Older 23-DoF model (head fixed) in `engineai_amp` [16]. Navigation Edition: same 24 joints + LiDAR [14][17] |
| SDK languages | ROS 2 interface: C++ (rclcpp) and Python (rclpy) examples [1]. Native SDK: C++ (C++17 runner plugins on precompiled core libraries; Humble x86_64/aarch64, Jazzy aarch64) [7] |
| Low-level entry | Stock controller: `joint_bridge` motion state, entered from `passive` or `pd_stand` with the remote [1]. Native SDK: custom motion runner replacing the stock service [7] |
| Policy format on robot | MNN (Alibaba), converted from ONNX [7][15] |

## 2. Transport and middleware

Two public routes reach joint-level control. **The adapter takes route A,
the ROS 2 topic bridge** ([vendor SDKs](../adding-a-robot.md#vendor-sdks)):
it is reachable from Python, it is what the vendor's examples use, and it
needs nothing compiled beyond the message package. Route B is the fallback
if route A's latency or `set_motion_state` behaviour (section 12) turns out
to be unusable.

**Route A: stock controller + ROS 2 joint bridge** (`engineai_ros2_workspace`) [1][2]
- The stock motion-control service runs on the robot. A client on an external PC (rear Ethernet port) or on the onboard computers publishes joint commands over ROS 2.
- Host setup: Ubuntu 22.04, ROS 2 Humble, `export ROS_DOMAIN_ID=69 ROS_LOCALHOST_ONLY=0 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` [1]. The `interface_protocol` package from the workspace must be built on the client.
- Vendor warning: do not use USB network adapters; use the built-in Ethernet port [1].
- Hosts on the internal network [5][13]: motion-control computer "Nezha" at 192.168.0.163 (user `user`; runs the controller and the native SDK), Jetson Orin NX at 192.168.0.162 (user `ubuntu`), router 192.168.0.1.
- Entering joint-level control: put the robot in `passive` (hanging) or `pd_stand` (standing), press the Joint Bridge button on the remote, then confirm `/motion/motion_state.current_motion_task == "joint_bridge"` before publishing [1][3]. The vendor examples exit if the state leaves `joint_bridge` [3][4].
- Topics [2]:

| Topic | Dir (client) | Type | Use |
|---|---|---|---|
| `/hardware/joint_state` | sub | `JointState` | joint position, velocity, torque (24, SDK order) |
| `/hardware/imu_info` | sub | `ImuInfo` | orientation, rpy, accel, gyro |
| `/hardware/joint_command` | pub | `JointCommand` | joint targets and gains, used in `joint_bridge` |
| `/motion/motion_state` | sub | `MotionState` | current state and allowed transitions |
| `/motion/set_motion_state` | pub | `MotionStateRequest` | request a state by name |
| `/hardware/gamepad_keys` | sub | `GamepadKeys` | remote input |
| `/motion/body_vel_cmd` | pub | `BodyVelCmd` | velocity command to vendor walking modes |
| `/motion/joint_override_command` | pub | `JointOverrideCommand` | override selected joints during vendor modes |
| `/motion/joint_motion_plan/request`, `/state` | pub/sub | `JointMotionPlanRequest/State` | vendor joint trajectory planner |
| `/hardware/led_control` | pub | `LedControl` | LEDs |

- QoS in vendor examples: state and command topics best-effort, volatile, depth 3; `MotionStateRequest` reliable depth 1 [3][4].

**Route B: native SDK** (`engineai_robotics_native_sdk`) [7]
- Replaces the stock controller: `ssh user@192.168.0.163`, `sudo systemctl stop robotics.service`, `sudo ./run_robot.sh pm01_edu` [7]. Built with `./build.sh` (x86_64 default) and deployed with `./install.sh pm01_edu robot` [7][13].
- Control code is a C++ `MotionRunner` (`Enter/Run/TryExit/Exit`) registered in `assets/config/pm01_edu/task_motion/default.yaml` with a period and a remote key combination [8][10]. Runners read `data_store_->joint_info` (state) and `imu_info`, and write `GetMutableOutput().SetCommand(q, qd, kp, kd, tau_ff)` [10][11].
- The SDK's ROS 2 bridge publishes state only (`/hardware/joint_state`, `/hardware/imu_info`, `/hardware/motor_state`, `/hardware/motor_command`, `/hardware/joint_command_feedback`, `/hardware/motor_debug`, `/hardware/power_info`, `/hardware/gamepad_keys`) and subscribes only to `/hardware/led_control` [8][9]. At commit 335c60e there is no public ROS 2 joint-command subscriber in the native SDK; an external adapter on this route needs a custom runner that accepts commands.
- IMU device on Nezha: serial `/dev/ttyACM0` [13]. Motor, IMU, estimator and parallel-ankle code ship as precompiled libraries (`core/lib_1.0.0_*`) [7].

**Simulation without hardware**: `engineai_robot_simulation_runtime` runs the vendor motion-control image for `pm_v2` with MuJoCo and a virtual gamepad in Docker (ROS 2 domain 69, `ROS_LOCALHOST_ONLY=1` by default) [18]. Whether that image exposes `joint_bridge` is not documented. The ROS 2 workspace also ships a MuJoCo node that accepts `JointCommand` directly [1][6].

## 3. Messages

Package `interface_protocol` [2]. Field units are not written in the `.msg` files. Units below follow the URDF (rad), the vendor PD code and the RL deployment code [6][10].

`JointCommand` (`/hardware/joint_command`), arrays of length 24 in SDK order:

| Field | Type | Unit | Notes |
|---|---|---|---|
| `header` | `std_msgs/Header` | | stamped by vendor examples |
| `position` | `float64[]` | rad | target q* |
| `velocity` | `float64[]` | rad/s | target dq* |
| `feed_forward_torque` | `float64[]` | N·m | tau_ff |
| `torque` | `float64[]` | N·m | full-torque field; vendor examples send 0; ignored by the ROS 2 MuJoCo node; meaning on hardware not documented |
| `stiffness` | `float64[]` | N·m/rad | kp |
| `damping` | `float64[]` | N·m·s/rad | kd |
| `parallel_parser_type` | `uint8` | | `CLASSIC_PARSER=0`, `RL_PARSER=1` (ankle joint-to-motor conversion mode); vendor RL example sends `RL_PARSER` [4] |

PD law: tau = kp·(q* − q) + kd·(dq* − dq) + tau_ff, as implemented in the ROS 2 MuJoCo node [6] and documented in the vendor passive runner [10]. On hardware the law runs inside closed-source motor code; the same form is assumed.

`JointState` (`/hardware/joint_state`, also `/hardware/motor_state` in motor space): `header`, `position[]` rad, `velocity[]` rad/s, `torque[]` N·m (estimated; source on hardware not documented) [2][9].

`ImuInfo` (`/hardware/imu_info`): `header`; `quaternion` (`geometry_msgs/Quaternion`, fields x, y, z, w); `rpy` (`Vector3`, rad, roll-pitch-yaw); `linear_acceleration` (`Vector3`, m/s²); `angular_velocity` (`Vector3`, rad/s) [2][9].

`MotionState`: `current_motion_task` (string), `available_transition_motions` (string[]). `MotionStateRequest`: `target_motion_name` (string) [2].

`MotorDebug` (`/hardware/motor_debug`, per motor): `mos_temperature[]`, `motor_temperature[]` (float32, unit not stated, presumably °C), `voltage[]`, `current[]`, `error_code[]` (int32), `offline[]`, `enable[]` (uint8) [2][9][11].

`PowerInfo`: `enable`, `percentage`, `voltage`, `current`, `current_limit`, `error_code` [2]. `GamepadKeys`: `hardware_connected`, `digital_states[12]`, `analog_states[6]` (−1..1) [2]. `BodyVelCmd`: `linear_velocity[]`, `yaw_velocity` [2]. `JointOverrideCommand`: `weight`, `joint_indices[]` plus the `JointCommand` arrays; native SDK clamps override commands (velocity ±10 rad/s, damping 1–10, feed-forward 0–30 N·m) [8]. `EnableMotor.srv` (`bool enable` → `success`, `message`) exists, but no public code shows its service name [2].

## 4. Rates and timing

| Loop | Rate | Source |
|---|---|---|
| Motor task (joint↔motor transform + motor I/O), priority 90 | 500 Hz | `task_resident/robot.yaml` [8] |
| IMU task, state estimator, input arbiter | 500 Hz | [8] |
| ROS 2 bridge: joint_state, imu_info, motor_state, motor_command, joint_command_feedback | 500 Hz | `ros2_bridge/default.yaml` [8] |
| ROS 2 bridge: motor_debug / gamepad_keys / power_info | 100 / 50 / 20 Hz | [8] |
| Vendor states idle, passive, pd_stand | 500 Hz | `task_motion/default.yaml` [8] |
| Vendor RL states walk, rl_lab / dance | 100 Hz / 50 Hz | [8] |
| Joint-bridge example command publisher | 500 Hz | [3] |
| ROS 2 RL example, body-velocity example | 100 Hz | [4][3] |
| MuJoCo timestep (both vendor sims) | 0.002 s | [6][12] |
| Isaac Lab AMP training: sim dt / decimation | 0.002 s / 5 (100 Hz), actuator delay 2–8 physics steps | [15] |
| Isaac Lab tracking training | 0.005 s / 4 (50 Hz) | [15] |

Timestamps: the native SDK bridge does not set `header.stamp` on `/hardware/joint_state` or `/hardware/imu_info` (only on motor_state/motor_command/feedback) [9]. Stamp on receipt with a monotonic clock unless the unit proves otherwise. Command timeout behaviour in `joint_bridge` is not documented.

## 5. Modes and state machines

Stock controller states named in the ROS 2 docs: `passive`, `pd_stand` ("pd-stand"), `joint_bridge`, walking modes (e.g. "Basic Walk", `rl_terrain`) [1]. The authoritative list is `/motion/motion_state.available_transition_motions` at runtime.

Native SDK FSM for `pm01_edu` [7][8][10]:

| State | Behaviour | Key | Transitions |
|---|---|---|---|
| `idle` | motors disabled, zero output; initial state | LB+START | → passive (auto) |
| `passive` | kp = 0, kd only (legs 4,4,4,4,3,3; waist 2; arms 3; head 1 N·m·s/rad) | LB+RB (from any state) | → idle, pd_stand |
| `pd_stand` | quintic move to fixed stand pose in 3 s, kp 200–450; refuses if start is > 1.2 rad from target | LB+A | → passive, walk, rl_lab, dance |
| `walk` | vendor RL walking (100 Hz) | LB+B | → passive, pd_stand, rl_lab, dance |
| `rl_lab` | AMP policy from `engineai_rl_lab` (100 Hz) | LB+X | → passive, pd_stand, walk, dance |
| `dance` | tracking policy (50 Hz) | RB+B | → passive, pd_stand, walk |

Mapping to platform modes:

| Platform | Vendor equivalent |
|---|---|
| PASSIVE (zero torque) | native `idle` (motors disabled). In `joint_bridge`: kp = kd = tau_ff = 0 from the adapter |
| HOLD | in `joint_bridge`: q* = measured q at entry, description gains. `pd_stand` is not a hold: it moves to a fixed pose |
| POLICY | `joint_bridge` (Route A) or a custom runner (Route B) |
| DAMP | vendor `passive` (kp = 0, kd > 0); LB+RB on the remote reaches it from any state |

The physical emergency-stop remote enables and cuts motor power [7]. LB+RB is a software stop [7].

## 6. Joints

SDK order = URDF order = `model/default.yaml` limb order = `JointState`/`JointCommand` array index [8][12]. Limits from the URDF [12] (identical in [6] and [14]). Effort limits used by RL configs differ: see the last column [15][16].

| Idx | Canonical | Vendor | Position (rad) | Vel (rad/s) | Effort (N·m) | RL-config effort |
|---|---|---|---|---|---|---|
| 0 | left_hip_pitch | J00_HIP_PITCH_L | −3.141 … 2.443 | 26.3 | 164 | 164 |
| 1 | left_hip_roll | J01_HIP_ROLL_L | −0.436 … 2.094 | 26.3 | 164 | 164 |
| 2 | left_hip_yaw | J02_HIP_YAW_L | −1.57 … 4.014 | 35.2 | 61 | 61 (AMP) / 52 (tracking) |
| 3 | left_knee | J03_KNEE_PITCH_L | −0.3491 … 2.3911 | 26.3 | 164 | 164 |
| 4 | left_ankle_pitch | J04_ANKLE_PITCH_L | −0.6807 … 0.7243 | 35.2 | 61 | 54.9 (AMP) / 52 |
| 5 | left_ankle_roll | J05_ANKLE_ROLL_L | −0.2618 … 0.2618 | 35.2 | 61 | 54.9 / 52 |
| 6 | right_hip_pitch | J06_HIP_PITCH_R | −3.141 … 2.443 | 26.3 | 164 | 164 |
| 7 | right_hip_roll | J07_HIP_ROLL_R | −2.094 … 0.436 | 26.3 | 164 | 164 |
| 8 | right_hip_yaw | J08_HIP_YAW_R | −4.014 … 1.57 | 35.2 | 61 | 61 / 52 |
| 9 | right_knee | J09_KNEE_PITCH_R | −0.3491 … 2.3911 | 26.3 | 164 | 164 |
| 10 | right_ankle_pitch | J10_ANKLE_PITCH_R | −0.6807 … 0.7243 | 35.2 | 61 | 54.9 / 52 |
| 11 | right_ankle_roll | J11_ANKLE_ROLL_R | −0.2618 … 0.2618 | 35.2 | 61 | 54.9 / 52 |
| 12 | waist_yaw | J12_WAIST_YAW | −4.014 … 1.57 | 35.2 | 61 | 61 / 52 |
| 13 | left_shoulder_pitch | J13_SHOULDER_PITCH_L | −2.9671 … 2.7925 | 35.2 | 61 | 61 / 52 |
| 14 | left_shoulder_roll | J14_SHOULDER_ROLL_L | −0.6108 … 2.3562 | 35.2 | 61 | 61 / 52 |
| 15 | left_shoulder_yaw | J15_SHOULDER_YAW_L | −2.618 … 2.618 | 35.2 | 61 | 61 / 52 |
| 16 | left_elbow_pitch | J16_ELBOW_PITCH_L | −2.1948 … 0.7374 | 35.2 | 61 | 61 / 52 |
| 17 | left_elbow_yaw | J17_ELBOW_YAW_L | −2.618 … 2.618 | 35.2 | 61 | 61 / 52 |
| 18 | right_shoulder_pitch | J18_SHOULDER_PITCH_R | −2.9671 … 2.7925 | 35.2 | 61 | 61 / 52 |
| 19 | right_shoulder_roll | J19_SHOULDER_ROLL_R | −2.3562 … 0.6108 | 35.2 | 61 | 61 / 52 |
| 20 | right_shoulder_yaw | J20_SHOULDER_YAW_R | −2.618 … 2.618 | 35.2 | 61 | 61 / 52 |
| 21 | right_elbow_pitch | J21_ELBOW_PITCH_R | −2.1948 … 0.7374 | 35.2 | 61 | 61 / 52 |
| 22 | right_elbow_yaw | J22_ELBOW_YAW_R | −2.618 … 2.618 | 35.2 | 61 | 61 / 52 |
| 23 | head_yaw | J23_HEAD_YAW | −0.6109 … 0.6109 | 35.2 | 61 | – / 52 |

Notes:
- Two motor classes per the tracking config: "Q90" (hip pitch, hip roll, knee: 164 N·m, 26.3 rad/s) and "Q25" (all others: 52 N·m, 35.2 rad/s) [15].
- Parallel linkage: both ankles (pitch/roll pairs J04/J05, J10/J11) [8]. The SDK converts joint commands to the two ankle motors (`joint_motor_transform_runner: enable_parallel_converter: true`, geometry and a fitted network in `parallel_ankle/default.yaml`). Joint topics are in joint space; `/hardware/motor_state` and `/hardware/motor_command` show motor space. The waist is a single serial yaw joint.
- `parallel_parser_type` selects the ankle conversion: vendor RL runners and the ROS 2 RL example use `RL_PARSER`; `passive` and `pd_stand` use the classic parser [4][10]. The difference between the two parsers is not documented.
- Hip pitch axes are tilted 15° about x: (0, 0.966, ∓0.259) [12]. The elbow yaw axis lies close to the forearm, so J17/J22 act as forearm roll. PM01 has no wrist joints.
- Waist range is asymmetric (−230° … +90°); the product page quotes ">300°" [20].
- Per-motor direction signs and offsets live in `motor/default.yaml` and are applied inside the SDK [8]. Joint topics are assumed to follow URDF sign conventions, since vendor policies trained on the URDF run on them.
- Hip and knee ranges on the product page differ slightly from the URDF (hip roll −29°…+134° vs −25°…+120°; knee −30°…+137° vs −20°…+137°) [20].

## 7. IMU

- Mount: the URDF has no IMU link. The MJCF puts site `imu` in `LINK_BASE` at (0.02311, 0, −0.09631) m with no rotation [12]. The IMU model is not public; it connects over USB serial (`/dev/ttyACM0`) [13].
- Quaternion: `geometry_msgs/Quaternion` fields x, y, z, w; read by field name. The native SDK fills it from an `Eigen::Quaterniond` [9]. Vendor RL code treats it as body-to-world rotation: projected gravity = −Rᵀ·ẑ [10]. The world-frame yaw reference (power-on heading or magnetic) is not documented.
- Gyro: `angular_velocity` in rad/s, IMU/body frame. Vendor RL code uses it directly as base angular velocity, after an optional install-bias rotation (zero in the native SDK configs; [0, 0.015, 0] rad roll/pitch/yaw in the ROS 2 RL example) [4][8][10].
- Accelerometer: `linear_acceleration` in m/s²; whether it is specific force (includes +g at rest) is not documented. The MuJoCo accelerometer in the native SDK sim reports specific force [12].
- `rpy`: rad, roll-pitch-yaw. The ROS 2 RL example recomputes Euler angles from the quaternion instead [4].
- The ROS 2 workspace MuJoCo node fills `ImuInfo` from the first 13 sensor values, which in its MJCF are the base site's framequat, framepos, framelinvel and frameangvel. In that simulator `rpy` carries base position, `linear_acceleration` carries world-frame linear velocity, and `angular_velocity` is world-frame [6]. The native SDK MuJoCo link uses the real accelerometer and gyro sensors [7].

## 8. Other sensors

- Two "ultra-wide field-of-view binocular cameras", head and waist [20]. Camera model, depth output, resolution and driver topics are not public.
- URDF frames in `engineai_robotics_description` (no licence) [14]: `camera_link_head` on `LINK_HEAD_YAW` at (0.07426, 0.0175, 0.09677) m, rpy (0, 0.2642, 0) rad; `camera_link_torso` on `LINK_TORSO_YAW` at (0.07112, 0.0175, 0.01109) m, rpy (0, 0.8695, 0) rad. These are body-convention frames, not optical frames.
- Navigation Edition: Livox Mid-360 [17], `LIVOX_FRAME` on `LINK_TORSO_YAW` at (0.12438, 0, 0.21289) m, rpy (0.8727, 0, 1.5708) rad [14]. Driver: `livox_ros_driver2` [17].
- MJCF foot force/torque sites (`force_sensor_left_foot`, etc.) are simulation-only; no public evidence of foot F/T hardware [12].
- Also published: battery (`/hardware/power_info`), remote (`/hardware/gamepad_keys`). Chest touchscreen and voice are on the product page; their interfaces are not public [20].

## 9. Mapping onto the Robot port

Route A, the chosen route (ROS 2 client, `rclpy` with a background
executor thread):

| Port call | Vendor mapping |
|---|---|
| `connect()` | Set `ROS_DOMAIN_ID=69`, `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`. Subscribe to `/hardware/joint_state`, `/hardware/imu_info`, `/motion/motion_state` and `/hardware/motor_debug` (best-effort, depth 3). Create the `/hardware/joint_command` publisher (best-effort) and the `/motion/set_motion_state` publisher (reliable, depth 1). Wait for first messages; check array length 24 |
| `read()` | Copy the latest cached messages (no blocking). Index mapping is identity: SDK order = description order. `stamp_ns` = monotonic receipt time of the joint_state message. `imu.orientation` = (q.x, q.y, q.z, q.w) by field name. Gyro and accelerometer pass through, pending checks. `mode` from `current_motion_task`. `temperature` from `motor_debug.motor_temperature` (per motor; slots 4/5 and 10/11 are ankle linkage motors, not pitch/roll). `faults` from `error_code`/`offline`/`enable` |
| `write(cmd)` | If `current_motion_task == "joint_bridge"`, publish one `JointCommand` with all 24 entries: position, velocity, `feed_forward_torque` = effort, `stiffness` = kp, `damping` = kd, `torque` = 0, `parallel_parser_type = RL_PARSER` (vendor RL default; verify). Otherwise drop the command and raise a fault. Clip q* to the description limits |
| `set_mode(PASSIVE)` | In `joint_bridge`: stream kp = kd = tau_ff = 0. Vendor `idle` (motors off) is reachable only through the vendor FSM |
| `set_mode(HOLD)` | In `joint_bridge`: latch measured q, stream q* = latched q with description gains |
| `set_mode(POLICY)` | Pass `write()` commands through. Entering `joint_bridge` needs the remote unless `/motion/set_motion_state` accepts `joint_bridge` (verify) |
| `operator_stop()` | Publish `MotionStateRequest(target_motion_name="passive")` and stop streaming. The remote's LB+RB and the hardware e-stop are the primary stops |
| `capabilities()` | `effort_command: true` (`feed_forward_torque` field; verify honoured on hardware), `temperature: true` (via `/hardware/motor_debug`; verify the stock controller publishes it), `max_rate_hz: 500` |

Conversions: none for units (rad, rad/s, N·m, m/s² throughout). Quaternion handled by field name (x, y, z, w). Joint order needs no permutation for Route A. Policies from `engineai_rl_lab` tracking use Isaac Lab breadth-first order and need the `joint_names` permutation in `rl_dance_example/default.yaml` [8]. Route B: Eigen quaternion `coeffs()` are (x, y, z, w), but the constructor takes (w, x, y, z). LCM sim messages carry (w, x, y, z) arrays [7][11].

## 10. Policy inference formats

On-robot runtime is MNN: Isaac Lab → ONNX → `MNN.tools.mnnconvert` → `.mnn`, loaded by `math::MNNModel` [7][15]. Navigation uses ONNX Runtime [17]. Public PM01 deployments:

| Deployment | Policy file | Rate | Observation (per step) | History | Action |
|---|---|---|---|---|---|
| Native SDK `rl_lab` (AMP walk) [8][10][15] | `rl_lab/policies/pm01_amp_walk.mnn` | 100 Hz | term-major: q−q0 (23), dq (23), last action (23), gyro (3), projected gravity (3); then command (vx, vy, wz) | 15 per term, oldest first; total 75·15+3 = 1128 | q* = q0 + a·scale (legs 0.5/0.2/0.2/0.5/0.5/0.2, waist and arms 0.2); 23 joints, head held |
| Native SDK `walk` [8][10] | `rl_walking_example/policies/251009_221907_41000.mnn` | 100 Hz | frame-major: q−q0 (22, no waist or head), dq·0.05 (22), last action (22), gyro (3), projected gravity (3) = 72 | 15 frames + 3 command = 1083 | 22 joints |
| ROS 2 `rl_basic_example` [4] | `pm01_v2_rough_ppo_42obs.mnn` | 100 Hz | q−q0 (12 legs), dq·0.05, last action, gyro, Euler rpy = 42 | 15 frames + clock (sin, cos; 0.8 s cycle) + command = 635 | legs only, `RL_PARSER`, 0.5 s blend-in |
| Native SDK `dance` (tracking) [8][15] | `rl_dance_example/policies/dance.mnn` | 50 Hz | command, motion_anchor_ori_b, base_ang_vel, joint_pos, joint_vel, actions | 1 | 24 joints, Isaac Lab order |

All actions are position targets around the default pose, with zero velocity and feed-forward targets [4][10]. The 2024 `engineai_legged_gym`/`engineai_humanoid` repositories (ONNX) target the SA01 robot, not PM01 [22].

## 11. Public model files and licences

| Repo @ commit | Licence (SPDX) | PM01 model content |
|---|---|---|
| engineai_robotics_native_sdk @ 335c60e [12] | BSD-3-Clause | `assets/resource/robot/pm01_edu/`: URDF, MJCF (`serial_pm01_edu.xml` + links/actuators/sensors), meshes (.dae, .obj), textures. IMU site, armature, foot F/T sites. No camera links |
| engineai_ros2_workspace @ c41043f [6] | BSD-3-Clause (MuJoCo `simulate/` Apache-2.0) | `src/simulation/mujoco/assets/resource/robot/pm_v2/`: URDF, command URDF, MJCF, STL meshes. Same joint data |
| engineai_robotics_description @ 1d2d760 [14] | none (no licence file) | `pm01_edu/` URDF + .dae/.obj meshes + textures, with camera links; `pm01_lidar/` adds `LIVOX_FRAME`. Not redistributable without permission |
| engineai_rl_lab @ c53f7ec [15] | BSD-3-Clause | `assets/pm01/serial_pm01_edu.usd` + configuration USDs (Git LFS), Isaac Lab actuator configs |
| engineai_amp @ 83ba64b [16] | BSD-3-Clause | older `assets/pm01/urdf/serial_pm01.urdf` (23 DoF, head fixed, 52 N·m Q25 joints) + meshes |
| engineai_navigation @ 7ee29a3 [17] | Apache-2.0 for EngineAI code; vendored parts keep own licences (GPL-3.0 Super-LIO, MIT Livox) | `pm01_lidar_v1` URDF/xacro (Gazebo), `pm01_edu_v2` URDF, navigation USDs |
| GMR @ 71e02c0 [19] | MIT | `assets/engineai_pm01/`: copy of the `pm_v2` MJCF + meshes |

## 12. To verify on the real unit

- [ ] `/motion/set_motion_state` with `joint_bridge` and `passive`: accepted from software, or remote only?
- [ ] `joint_bridge` command-timeout behaviour (hold, damp, or last command) and minimum command rate.
- [ ] Hardware honours `feed_forward_torque`; meaning of the `torque` field.
- [ ] `RL_PARSER` vs `CLASSIC_PARSER`: ankle tracking difference at the same joint gains; which suits a serial-ankle sim policy.
- [ ] `/hardware/joint_state` and `/hardware/imu_info` header stamps (zero in the native SDK bridge); measure latency and jitter at 500 Hz over Ethernet.
- [ ] `/hardware/joint_state.torque`: joint-space estimate or motor current scaled; units.
- [ ] Joint sign conventions match the URDF for all 24 joints (move each joint by hand in `passive`).
- [ ] IMU frame: axes aligned with `LINK_BASE`; gyro in body frame; accelerometer reads +9.81 m/s² on z at rest (specific force); yaw reference and drift; mounting offset (vendor ROS 2 example uses 0.015 rad pitch bias).
- [ ] Is `/hardware/motor_debug` published by the stock controller? Temperature unit, and motor-slot to joint mapping for the ankle motors.
- [ ] Effective ankle and Q25 torque limits (61 vs 54.9 vs 52 N·m) and whether the SDK clamps commands.
- [ ] Motor-level loop rate inside the closed-source driver (500 Hz motor task assumed as `physics_hz`).
- [ ] Description gains (vendor AMP config) are stable for HOLD on the stand, especially ankles (kp 30, kd 0.3) and the head.
- [ ] Camera models, depth availability, ROS topics, intrinsics and extrinsics; whether the URDF camera mounts match the unit.
- [ ] Hip roll and knee ranges (URDF vs product page).
- [ ] Which computer runs the adapter best (external PC, Orin 192.168.0.162, or Nezha 192.168.0.163); CPU headroom on Nezha.
- [ ] Whether the stock service and the native SDK can coexist (the native SDK requires stopping `robotics.service`).

## 13. Sources

Repositories are cited at the commits shown, as of 2026-10-04.

1. engineai_ros2_workspace README (branch `community`, commit c41043f64ca1d308be187ebdcc2eaae40884f59c): https://github.com/engineai-robotics/engineai_ros2_workspace/blob/c41043f64ca1d308be187ebdcc2eaae40884f59c/README.md
2. engineai_ros2_workspace `src/interface_protocol` (README, `msg/`, `srv/`): https://github.com/engineai-robotics/engineai_ros2_workspace/tree/c41043f64ca1d308be187ebdcc2eaae40884f59c/src/interface_protocol
3. engineai_ros2_workspace `src/interface_example/scripts/` (`joint_bridge_example.py`, `body_velocity_control_example.py`, `switch_to_target_motion_example.py`) and `src/interface_example/config/pm01/*.yaml`: https://github.com/engineai-robotics/engineai_ros2_workspace/tree/c41043f64ca1d308be187ebdcc2eaae40884f59c/src/interface_example
4. engineai_ros2_workspace `rl_basic_example.cc`, `components/message_handler.cc`, `config/pm01/rl_basic/basic/rl_basic_param.yaml`: https://github.com/engineai-robotics/engineai_ros2_workspace/blob/c41043f64ca1d308be187ebdcc2eaae40884f59c/src/interface_example/src/rl_basic_example.cc
5. engineai_ros2_workspace `scripts/sync_src.sh`: https://github.com/engineai-robotics/engineai_ros2_workspace/blob/c41043f64ca1d308be187ebdcc2eaae40884f59c/scripts/sync_src.sh
6. engineai_ros2_workspace `src/simulation/mujoco` (`src/sim_manager.cc`, `src/ros_interface.cc`, `assets/resource/robot/pm_v2/`): https://github.com/engineai-robotics/engineai_ros2_workspace/tree/c41043f64ca1d308be187ebdcc2eaae40884f59c/src/simulation/mujoco
7. engineai_robotics_native_sdk README (commit 335c60e88772c26c7852d0abd6b3c7439037dd8f): https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/335c60e88772c26c7852d0abd6b3c7439037dd8f/README.md
8. engineai_robotics_native_sdk `assets/config/pm01_edu/` (`model`, `motor`, `parallel_ankle`, `joint_motor_transform_runner`, `task_resident`, `task_motion`, `passive`, `pd_stand`, `rl_lab`, `rl_walking_example`, `rl_dance_example`, `ros2_bridge`, `lcm`): https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/335c60e88772c26c7852d0abd6b3c7439037dd8f/assets/config/pm01_edu
9. engineai_robotics_native_sdk `src/ros2_node/src/hardware_interface_node.cc`, `ros2_message_convertor.cc`: https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/335c60e88772c26c7852d0abd6b3c7439037dd8f/src/ros2_node/src
10. engineai_robotics_native_sdk `src/runner/` (`idle`, `passive`, `pd_stand`, `rl_lab`, `rl_walking_example`): https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/335c60e88772c26c7852d0abd6b3c7439037dd8f/src/runner
11. engineai_robotics_native_sdk `core/include/data/` (`joint_info.h`, `imu_info.h`, `control_output.h`, `motor_debug.h`, `data_store.h`): https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/335c60e88772c26c7852d0abd6b3c7439037dd8f/core/include/data
12. engineai_robotics_native_sdk `assets/resource/robot/pm01_edu/` (URDF, MJCF): https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/335c60e88772c26c7852d0abd6b3c7439037dd8f/assets/resource/robot/pm01_edu
13. engineai_robotics_native_sdk `install.sh`, `scripts/run_robot.sh`, `scripts/set_imu_tty.sh`, `LICENSE.txt`: https://github.com/engineai-robotics/engineai_robotics_native_sdk/blob/335c60e88772c26c7852d0abd6b3c7439037dd8f/install.sh
14. engineai_robotics_description (commit 1d2d7601f520f716863c32a83abe9c77b26ee06c) `README_EN.md`, `pm01_edu/urdf/serial_pm01_edu.urdf`, `pm01_lidar/urdf/serial_pm01_edu.urdf`: https://github.com/engineai-robotics/engineai_robotics_description/tree/1d2d7601f520f716863c32a83abe9c77b26ee06c
15. engineai_rl_lab (commit c53f7ec614e34f1ea514cefb935894082f3d4c2a) `tasks/locomotion/amp/{robots/pm01.py, amp_env_cfg.py, config/pm01/flat_amp_env_cfg.py}`, `tasks/tracking/{robots/pm01.py, tracking_env_cfg.py}`, `scripts/amp/play.py`, `README_EN.md`: https://github.com/engineai-robotics/engineai_rl_lab/tree/c53f7ec614e34f1ea514cefb935894082f3d4c2a
16. engineai_amp (commit 83ba64bbb58a02e14483e52adce5f893f3f31cdf) `source/engineai_lab/robots/pm01.py`, `source/engineai_lab/assets/pm01/urdf/serial_pm01.urdf`: https://github.com/engineai-robotics/engineai_amp/tree/83ba64bbb58a02e14483e52adce5f893f3f31cdf
17. engineai_navigation (commit 7ee29a3ede5ecd7d884a1a53b258cd5a0147bd81) `README.en.md`, `NOTICE`, `ros/src/simulation/simulator/`: https://github.com/engineai-robotics/engineai_navigation/tree/7ee29a3ede5ecd7d884a1a53b258cd5a0147bd81
18. engineai_robot_simulation_runtime (commit 8c0ece7448a2299cee284f0905a3b84d8114fa5b) `README.md`, `docker-compose.yml`: https://github.com/engineai-robotics/engineai_robot_simulation_runtime/tree/8c0ece7448a2299cee284f0905a3b84d8114fa5b
19. GMR (commit 71e02c0598f6d6c6fe1514bf1d69bc24e82f801e) `assets/engineai_pm01/`: https://github.com/engineai-robotics/GMR/tree/71e02c0598f6d6c6fe1514bf1d69bc24e82f801e/assets/engineai_pm01
20. EngineAI PM01 product page (2026 Education / Navigation Edition table), accessed 2026-10-04: https://en.engineai.com.cn/product-pm01.html
21. EngineAI knowledge base on Feishu, linked from [1] and [7]; login required; not used in these notes: https://dx3a2bminsq.feishu.cn/wiki/P6izwRvaDi9Zo3kR0okc2kg7n2c, https://dx3a2bminsq.feishu.cn/wiki/KyD9wDc4mi03uXkTVuAc5LQan4C
22. engineai_legged_gym (commit 1a659aa0aa77445766ecea6d95708a5812e9f290) and engineai_humanoid (commit 5c0cb92a99819744abe0511291533fc37e556903), 2024, SA01 robot, no licence file: https://github.com/engineai-robotics/engineai_legged_gym, https://github.com/engineai-robotics/engineai_humanoid
