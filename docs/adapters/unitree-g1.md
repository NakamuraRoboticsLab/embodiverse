# Unitree G1 interface notes

Scope: G1 EDU, 29 actuated body joints (`g1_29dof`), low-level (per-motor) control over DDS. Hands (Dex3-1, Inspire, BrainCo), the G1-D wheeled base and the G1+ 31-DoF head are out of scope except where they share the message types. All citations are to the vendor's public repositories at the commits listed in section 13.

## 1. Summary

| Item | Value |
|---|---|
| Transport | Ethernet, robot-internal switch, subnet `192.168.123.0/24`; motion-control computer PC1 `192.168.123.161`, development computer PC2 `192.168.123.164` [17] |
| Middleware | DDS, Eclipse CycloneDDS **0.10.2** (C++ SDK bundles it; Python SDK pins `cyclonedds==0.10.2`; ROS 2 needs `rmw_cyclonedds_cpp` against 0.10.x) [9][10][11] |
| IDL family | `unitree_hg` (G1, H1-2), not `unitree_go` [10] |
| Low-level topics | `rt/lowcmd` (`unitree_hg::msg::dds_::LowCmd_`), `rt/lowstate` (`LowState_`), `rt/secondary_imu` (`IMUState_`, torso IMU) [3] |
| Low-level rate | `rt/lowstate` 500 Hz [11]; vendor examples publish `rt/lowcmd` every 2 ms (500 Hz) [3][10][11] |
| Motor slots | 35 per message; 29 used (0–28), 29–34 unused [2][13] |
| DoF variants | 23-DoF, 29-DoF, 29-DoF waist-locked, dual-arm (14), with/without hands; identified by `mode_machine` [12] |
| SDK languages | C++ (`unitree_sdk2`, headers + prebuilt static library, x86_64 and aarch64), Python (`unitree_sdk2_python`), ROS 2 Foxy/Humble messages (`unitree_ros2`) [1][10][11] |
| Command law (per motor) | `tau = tau_ff + kp·(q* − q) + kd·(dq* − dq)` (same law in the vendor MuJoCo bridge) [13] |

## 2. Transport and middleware

- **DDS implementation.** CycloneDDS 0.10.2 (`thirdparty/include/dds/version.h` in the C++ SDK) [9]. The Python SDK requires exactly `cyclonedds==0.10.2` and building CycloneDDS from `releases/0.10.x` if no wheel matches [10]. `unitree_ros2` states "The cyclonedds version of Unitree robot is 0.10.2" and uses `rmw_cyclonedds_cpp`; with Humble the bundled rmw is used directly [11].
- **Domain.** Domain ID 0 on the robot; every vendor example calls `ChannelFactory::Instance()->Init(0, <nic>)` / `ChannelFactoryInitialize(0, nic)` [3][10]. The vendor simulator defaults to domain 1 on `lo` so simulation traffic does not reach a robot on the same L2 segment [13][17].
- **Network interface.** The NIC is passed explicitly (second argument of `Init`); for ROS 2 via `CYCLONEDDS_URI` `<NetworkInterface name="enp3s0"/>` [11][17]. Discovery uses UDP multicast, so host and robot must share the L2 domain and the `192.168.123.0/24` subnet; the vendor guide recommends a static host address such as `192.168.123.99` or `.222` [17].
- **Topic naming.** DDS topic `rt/<name>`; ROS 2 sees it as `<name>` (e.g. `lowstate`, `lowcmd`). `rt/lf/lowstate` (ROS 2 `lf/lowstate`) is a lower-frequency copy of the state [11].
- **Topics relevant to an adapter** [3][7][8][13]:

  | Topic | Type | Direction | Use |
  |---|---|---|---|
  | `rt/lowcmd` | `LowCmd_` | host → robot | per-motor commands (debug / released mode) |
  | `rt/lowstate` | `LowState_` | robot → host | motor state, pelvis IMU, remote-controller bytes, `mode_machine` |
  | `rt/lf/lowstate` | `LowState_` | robot → host | low-frequency state |
  | `rt/secondary_imu` | `IMUState_` | robot → host | torso IMU |
  | `rt/arm_sdk` | `LowCmd_` | host → robot | upper-body override while the vendor locomotion controller runs; blend weight in `motor_cmd[29].q` ∈ [0, 1] |
  | `rt/user_lowcmd` | `LowCmd_` | host → robot | "user control" path through the loco service (section 5) |
  | `rt/lf/bmsstate` | `BmsState_` | robot → host | battery (topic name taken from the simulator bridge) |
  | `rt/dex3/{left,right}/{cmd,state}`, `rt/inspire/state` | `HandCmd_`/`HandState_` etc. | | hands, out of scope |

- **Entering low-level mode.** The on-board locomotion controller also publishes `rt/lowcmd`; it must be stopped first. Three public ways:
  1. **Motion switcher (programmatic).** Service `motion_switcher` (API 1001 `CheckMode`, 1002 `SelectMode`, 1003 `ReleaseMode`). Loop `CheckMode` → `ReleaseMode` until the returned mode name is empty, then publish `rt/lowcmd` [3][5][10]. The G1 uses the `unitree::robot::b2::MotionSwitcherClient` class in C++ [3].
  2. **Remote controller "debug mode".** From the power-on zero-torque state press `L2 + R2`; the joints go to damping and the robot accepts `rt/lowcmd` [14].
  3. **User control via the loco service** (added to the SDK 2026-03-02 [20]). With the loco FSM in id 1 (damping), publish an initial `rt/user_lowcmd` (kp = 0, kd > 0), call `LocoClient::SwitchToUserCtrl()` (API 7110), stream `rt/user_lowcmd`, then hand back with `SwitchToInternalCtrl(LAST | PASSIVE | WALKRUN)` (API 7111) [8]. Firmware support for this path is not documented in the repositories.
- **Exclusive writer.** `unitree_rl_lab` refuses to start if another process is already publishing `rt/lowcmd` (subscribes for 0.2 s first) [15]. An adapter should do the same.
- **Where to run.** PC1 (`.161`) runs the vendor motion control and is not a user target. PC2 (`.164`) is the user development computer on the internal switch (runs the camera image server and hand services in vendor tooling) [16][17]. An external PC on the RJ45 port is equivalent at the DDS level; on-board PC2 avoids the cable-loss failure the SDK warns about ("a loose network cable … will send a step signal to the motors, causing violent movement") [6].

## 3. Messages

Field lists from the C++ IDL headers [2], identical in the ROS 2 `.msg` files [11] and the Python IDL [10].

**`LowCmd_`** (`rt/lowcmd`, `rt/arm_sdk`, `rt/user_lowcmd`)

| Field | Type | Meaning |
|---|---|---|
| `mode_pr` | uint8 | 0 = PR (ankle and waist roll/pitch commanded in joint space, pitch/roll), 1 = AB (parallel-linkage motors A/B commanded directly) [3][13] |
| `mode_machine` | uint8 | robot type id; copy the value read from `LowState_.mode_machine` [3][14] |
| `motor_cmd` | `MotorCmd_[35]` | index = SDK motor index (section 6) |
| `reserve` | uint32[4] | |
| `crc` | uint32 | CRC-32 of the message (below) |

**`MotorCmd_`**: `mode` uint8 (1 = enable, 0 = disable [3]); `q` float32 rad; `dq` float32 rad/s; `tau` float32 N·m (feed-forward); `kp` float32 N·m/rad; `kd` float32 N·m·s/rad; `reserve` uint32.

**`LowState_`** (`rt/lowstate`)

| Field | Type | Meaning |
|---|---|---|
| `version` | uint32[2] | |
| `mode_pr` | uint8 | current PR/AB mode |
| `mode_machine` | uint8 | robot type id (section 6, variant table); vendor simulator reports 5 for the 29-DoF scene, 4 for 23-DoF [13] |
| `tick` | uint32 | counter; units not documented in the repositories |
| `imu_state` | `IMUState_` | pelvis IMU (section 7) |
| `motor_state` | `MotorState_[35]` | |
| `wireless_remote` | uint8[40] | remote-controller bytes: `[2:4]` uint16 button bits (R1, L1, start, select, R2, L2, F1, F2, A, B, X, Y, up, right, down, left), float32 `lx`@4, `rx`@8, `ry`@12, `ly`@20 [14] |
| `reserve` | uint32[4] | |
| `crc` | uint32 | |

**`MotorState_`**: `mode` uint8; `q` rad; `dq` rad/s; `ddq` rad/s²; `tau_est` N·m (estimated); `temperature` int16[2] = [casing, winding] (the SDK's protection helpers use 85 for casing and 120 for winding) [6]; `vol` float32 (supply voltage); `sensor` uint32[2]; `motorstate` uint32 (non-zero = error code; codes not published) [3]; `reserve` uint32[4].

**`IMUState_`**: `quaternion` float32[4] **(w, x, y, z)**; `gyroscope` float32[3] rad/s; `accelerometer` float32[3]; `rpy` float32[3] rad; `temperature` int16 [2][6][14].

**CRC.** Every vendor publisher fills `crc` before writing [3][7][10][14]. Algorithm (`Crc32Core` [3]): polynomial `0x04C11DB7`, initial value `0xFFFFFFFF`, no reflection, no final XOR, over the packed message read as little-endian uint32 words, each word fed MSB-first, **excluding the final `crc` word** (`(sizeof(msg) >> 2) − 1` words). This equals CRC-32/MPEG-2 computed over the words re-serialised big-endian (checked numerically against the reference loop). Python packs `LowCmd_` as `'<2B2x' + 'B3x5fI'*35 + '5I'` (1004 bytes, CRC over 250 words) and `LowState_` as `'<2I2B2xI' + '13fh2x' + 'B3x4f2hf7I'*35 + '40B5I'`; `CRC().Crc(msg)` handles both and falls back to pure Python when the bundled `crc_*.so` cannot load [10]. The C++ example also verifies the CRC of incoming `LowState_` and drops mismatches [3].

Note: `unitree_rl_gym` deploy code assigns `motor_cmd[i].qd`, which is not a field of the hg `MotorCmd_` (the field is `dq`); it has no effect there because `dq` stays 0 [14].

## 4. Rates and timing

| Path | Rate | Source |
|---|---|---|
| `rt/lowstate` publish | 500 Hz ("high frequencies (500Hz)") | [11] |
| `rt/lf/lowstate` | lower, value not stated | [11] |
| Vendor low-level examples, `rt/lowcmd` writer thread | 2 ms (500 Hz) | [3][10][11] |
| `unitree_rl_lab` deploy FSM thread (writes `rt/lowcmd`) | 1 ms (1 kHz); policy thread 20 ms | [15] |
| `unitree_rl_gym` Python deploy | 20 ms (50 Hz), command written once per policy step | [14] |
| `xr_teleoperate` arm controller | 4 ms (250 Hz) | [17] |
| Vendor MuJoCo bridge | 1 kHz bridge thread; G1 MJCF has no `<option>`, so MuJoCo's default 2 ms step | [13] |

- Between messages the robot appears to keep applying the last received command: the 50 Hz `unitree_rl_gym` deployment depends on it. Not documented; see section 12. The motor-side PD loop rate is not published.
- Timestamps: `LowState_.tick` units are undocumented. Stamp `RobotState.stamp_ns` with the host's `CLOCK_MONOTONIC` at callback time; keep `tick` only for drop detection.
- Loss of state: `g1::lost_connection(subscriber, 1000 ms)` uses the subscriber's last-data time [6]. The adapter stamps each state on receipt, so the safety layer sees a stale state and holds the robot.

## 5. Modes and state machines

**Vendor loco service** (`sport`, API 7001–7111) FSM ids [4][10]:

| fsm_id | C++ helper | Python helper | Meaning |
|---|---|---|---|
| 0 | `ZeroTorque()` | `ZeroTorque()` | all joints zero torque |
| 1 | `Damp()` | `Damp()` | damping; the user-control example calls this state "PASSIVE" [8] |
| 2 | `Squat()` | | |
| 3 | `Sit()` | `Sit()` | |
| 4 | `StandUp()` | | |
| 500 | `Start()` | `Start()` | main locomotion controller |
| 702 | | `Lie2StandUp()` | |
| 706 | | `Squat2StandUp()` / `StandUp2Squat()` | |
| 550xxx | `GetMimicMotion()` + `SetFsmId(550<id>)` | | stored mimic motions |

The C++ and Python clients expose different stand-up ids (4 vs 702/706) [4][10].

**Remote-controller sequences** [14][17]:
- Power on (suspended) → zero-torque state.
- `L2 + R2` from zero torque → **debug mode**: joints damped, on-board controller off, `rt/lowcmd` accepted.
- `L2 + B` → damping; `L2 + UP` → locked standing; `R1 + X` → main motion control (current firmware). Older firmware: `L1 + A`, `L1 + UP`, `R1 + X`.
- `unitree_rl_lab`'s own controller maps the same buttons in software: `LT + up` → FixStand, `RB + X` → policy, `LT + B` → Passive (damping, kd = 3) [15].

**Mapping to the port**

| Port mode | Implementation on G1 (low-level mode) | Vendor equivalent |
|---|---|---|
| PASSIVE (zero torque) | all 29 slots `mode = 1`, `kp = kd = tau = 0` (`create_zero_cmd`) [14] | fsm 0 / power-on state |
| HOLD | `q*` latched from `LowState_` at entry (or ramp to `default_position` over ~2–3 s as both reference deployments do), `dq* = 0`, `tau = 0`, description gains [14][15] | locked standing (`L2 + UP`) when the vendor controller is in charge |
| POLICY | pass `JointCommand` through after limit clamping | |
| DAMP (via `operator_stop()` only) | `kp = 0`, `kd > 0`, `q* = dq* = tau = 0`: `unitree_rl_gym` uses kd = 8 on every motor, `unitree_rl_lab` kd = 3 [14][15] | fsm 1 `Damp()`, `L2 + B` |

Naming hazard: the vendor and `unitree_rl_lab` call the damping state "Passive"; the port's PASSIVE is zero torque. Map by behaviour, not by name.

The SDK ships protection predicates that "recommend setting the motors to passive mode": orientation error > 1.0 rad from upright, any `|dq| > 10 rad/s`, any gyro axis > 6 rad/s, winding > 120, casing > 85, battery SOC < 20 %, no state for 1000 ms [6]. `unitree_rl_lab` switches to its damping state on bad orientation (1.0 rad) [15].

## 6. Joints

Motor index and SDK enum from [3][7]; vendor joint name and limits from `g1_29dof_rev_1_0.urdf` [12] (MJCF `g1_29dof_rev_1_0.xml` has identical ranges; its `actuatorfrcrange` is in the last column). Velocity limits exist only in the URDF.

| idx | canonical | vendor (URDF) | SDK enum | position [rad] | vel [rad/s] | effort URDF [N·m] | effort MJCF |
|---:|---|---|---|---|---:|---:|---:|
| 0 | left_hip_pitch | left_hip_pitch_joint | LeftHipPitch | −2.5307 … 2.8798 | 32 | 88 | 88 |
| 1 | left_hip_roll | left_hip_roll_joint | LeftHipRoll | −0.5236 … 2.9671 | 20 | 139 | 139 |
| 2 | left_hip_yaw | left_hip_yaw_joint | LeftHipYaw | −2.7576 … 2.7576 | 32 | 88 | 88 |
| 3 | left_knee | left_knee_joint | LeftKnee | −0.087267 … 2.8798 | 20 | 139 | 139 |
| 4 | left_ankle_pitch | left_ankle_pitch_joint | LeftAnklePitch / LeftAnkleB | −0.87267 … 0.5236 | 30 | 35 | 50 |
| 5 | left_ankle_roll | left_ankle_roll_joint | LeftAnkleRoll / LeftAnkleA | −0.2618 … 0.2618 | 30 | 35 | 50 |
| 6 | right_hip_pitch | right_hip_pitch_joint | RightHipPitch | −2.5307 … 2.8798 | 32 | 88 | 88 |
| 7 | right_hip_roll | right_hip_roll_joint | RightHipRoll | −2.9671 … 0.5236 | 20 | 139 | 139 |
| 8 | right_hip_yaw | right_hip_yaw_joint | RightHipYaw | −2.7576 … 2.7576 | 32 | 88 | 88 |
| 9 | right_knee | right_knee_joint | RightKnee | −0.087267 … 2.8798 | 20 | 139 | 139 |
| 10 | right_ankle_pitch | right_ankle_pitch_joint | RightAnklePitch / RightAnkleB | −0.87267 … 0.5236 | 30 | 35 | 50 |
| 11 | right_ankle_roll | right_ankle_roll_joint | RightAnkleRoll / RightAnkleA | −0.2618 … 0.2618 | 30 | 35 | 50 |
| 12 | waist_yaw | waist_yaw_joint | WaistYaw | −2.618 … 2.618 | 32 | 88 | 88 |
| 13 | waist_roll | waist_roll_joint | WaistRoll / WaistA | −0.52 … 0.52 | 30 | 35 | 50 |
| 14 | waist_pitch | waist_pitch_joint | WaistPitch / WaistB | −0.52 … 0.52 | 30 | 35 | 50 |
| 15 | left_shoulder_pitch | left_shoulder_pitch_joint | LeftShoulderPitch | −3.0892 … 2.6704 | 37 | 25 | 25 |
| 16 | left_shoulder_roll | left_shoulder_roll_joint | LeftShoulderRoll | −1.5882 … 2.2515 | 37 | 25 | 25 |
| 17 | left_shoulder_yaw | left_shoulder_yaw_joint | LeftShoulderYaw | −2.618 … 2.618 | 37 | 25 | 25 |
| 18 | left_elbow | left_elbow_joint | LeftElbow | −1.0472 … 2.0944 | 37 | 25 | 25 |
| 19 | left_wrist_roll | left_wrist_roll_joint | LeftWristRoll | −1.972222054 … 1.972222054 | 37 | 25 | 25 |
| 20 | left_wrist_pitch | left_wrist_pitch_joint | LeftWristPitch | −1.614429558 … 1.614429558 | 22 | 5 | 5 |
| 21 | left_wrist_yaw | left_wrist_yaw_joint | LeftWristYaw | −1.614429558 … 1.614429558 | 22 | 5 | 5 |
| 22 | right_shoulder_pitch | right_shoulder_pitch_joint | RightShoulderPitch | −3.0892 … 2.6704 | 37 | 25 | 25 |
| 23 | right_shoulder_roll | right_shoulder_roll_joint | RightShoulderRoll | −2.2515 … 1.5882 | 37 | 25 | 25 |
| 24 | right_shoulder_yaw | right_shoulder_yaw_joint | RightShoulderYaw | −2.618 … 2.618 | 37 | 25 | 25 |
| 25 | right_elbow | right_elbow_joint | RightElbow | −1.0472 … 2.0944 | 37 | 25 | 25 |
| 26 | right_wrist_roll | right_wrist_roll_joint | RightWristRoll | −1.972222054 … 1.972222054 | 37 | 25 | 25 |
| 27 | right_wrist_pitch | right_wrist_pitch_joint | RightWristPitch | −1.614429558 … 1.614429558 | 22 | 5 | 5 |
| 28 | right_wrist_yaw | right_wrist_yaw_joint | RightWristYaw | −1.614429558 … 1.614429558 | 22 | 5 | 5 |

Axes (URDF): pitch joints about y, roll about x, yaw about z, all `+` by the right-hand rule in the parent link frame (REP-103 aligned, pelvis x forward, z up) [12].

**Ankle and waist: PR vs AB.** The ankle (both 23- and 29-DoF) and the 29-DoF waist roll/pitch are parallel linkages driven by two motors. With `mode_pr = 0` (PR) slots 4/5, 10/11 and 13/14 carry joint-space pitch/roll (and the firmware solves the linkage); with `mode_pr = 1` (AB) the same slots carry the two motor angles A/B [3][13]. Reference deployments all run PR mode [14][15]; the description uses `side: joint` throughout. The state follows the selected mode (the index table is headed "LowCmd_.mode or LowState_.mode") [13].

**Variants by `mode_machine`** (`g1_description/README.md` [12]; read it on the unit under App → Device → Data → Robot → Machine Type):

| mode_machine | model file | hip pitch/roll gear | wrist pitch/yaw motor | waist |
|---:|---|---|---|---|
| 5 | g1_29dof_rev_1_0 (target of these notes) | 14.3 / 22.5 | 4010 (5 N·m, 22 rad/s) | 3 DoF |
| 6 | g1_29dof_lock_waist_rev_1_0 | 14.3 / 22.5 | 4010 | roll/pitch locked: slots 13, 14 invalid [3] |
| 11 / 12 | g1_29dof_mode_11 / _12 (lock waist) | 22.5 / 22.5 → hip pitch 139 N·m, 20 rad/s | 4010 | |
| 13 / 14 | g1_29dof_mode_13 / _14 | 14.3 / 22.5 | 5010 (13.4 N·m, 27 rad/s) | |
| 15 / 16 | g1_29dof_mode_15 / _16 | 22.5 / 22.5 | 5010 | |
| 18 | g1_29dof_mode_18 | 22.5 / 22.5 | 5010 | roll/pitch 100 N·m, 26.9 rad/s |
| 2 / 3 | g1_29dof / g1_29dof_lock_waist (deprecated) | 14.3 / 14.5 | 4010 | hip roll 88 N·m, 32 rad/s |
| 4 / 10 | 23-DoF variants | | none | 1 DoF |

Position ranges are identical across all 29-DoF variants. `unitree_rl_lab` hard-codes `mode_machine = 5` and exits on mismatch [15].

**Motor data.** `unitree_rl_lab` torque–speed curves (Y1 peak torque, X2 no-load speed): N7520-14.3 71 / 83.3 N·m, 35.5 rad/s; N7520-22.5 111 / 131 N·m, 22.7 rad/s; N5020-16 24.8 / 31.9 N·m, 40.1 rad/s; W4010-25 4.8 / 8.6 N·m, 24.8 rad/s; armatures 0.0102, 0.0251, 0.00361, 0.00425 kg·m² [15]. `g1_description/motor_info.md` (G1+ era) lists different ratings: 5019-16 41 N·m, 6010-32 60 N·m (shoulder pitch, waist roll/pitch), 7519-22.5 121 N·m [12].

## 7. IMU

- **`LowState_.imu_state` is the pelvis IMU.** `unitree_rl_gym` configures the G1 with `imu_type: "pelvis"` (H1/H1-2 use torso and transform) [14]; the SDK ankle example prints it as `IMU.pelvis.rpy` and reads the torso IMU from `rt/secondary_imu` [3].
- **Mounting (URDF rev 1.0)** [12]: `imu_in_pelvis` fixed to `pelvis` at xyz (0.04525, 0, −0.08339) m, rpy (0, 0, 0); `imu_in_torso` fixed to `torso_link` at (−0.03959, −0.00224, 0.14792) m, rpy 0. Both frames are axis-aligned with their parent links in the model.
- **Quaternion order in the message: (w, x, y, z).** The deploy code comments "imu_state quaternion: w, x, y, z" [14]; the SDK and `unitree_rl_lab` construct `Eigen::Quaternionf(q[0], q[1], q[2], q[3])` (Eigen's constructor is w, x, y, z) [6][15]; the simulator copies MuJoCo `framequat` (w, x, y, z) straight in [13].
- **Gyroscope**: rad/s, IMU (≈ pelvis) body frame; fed unchanged to policies as base angular velocity [14][15].
- **Accelerometer**: body frame; the simulator writes a MuJoCo `accelerometer` (specific force, +9.81 m/s² on z at rest) [13]. Units on hardware are not stated in the repositories.
- **`rpy`**: rad, roll-pitch-yaw derived from the quaternion (the simulator computes it with the standard ZYX formulas) [13].
- **Gravity vector** used by policies: `projected_gravity_b = q.conjugate() * (0, 0, −1)` [6][15].
- **Torso IMU to pelvis** (when needed): `unitree_rl_gym` `transform_imu_data` removes the waist-yaw rotation, `R_pelvis = R_torso · Rz(waist_yaw)ᵀ`, `ω_pelvis = Rz(waist_yaw)·ω_torso − (0, 0, dq_waist_yaw)` (ignores waist roll/pitch) [14].

## 8. Other sensors

| Sensor | Model | Frame / mount (URDF rev 1.0, parent `torso_link`) | Interface |
|---|---|---|---|
| Head depth camera | Intel RealSense D435i ("built-in Realsense D435i") [16] | `d435_link` xyz (0.0576235, 0.01753, 0.42987), rpy (0, 0.8308, 0) (≈ 47.6° pitched down) [12] | USB on PC2; vendor image server runs on PC2 (`192.168.123.164`) [16] |
| Head 3D lidar | Livox MID-360 (from the link name; the product page says "3D LiDAR" only [18]) | `mid360_link` xyz (0.0002835, 0.00003, 0.428434), rpy (π, 0.0511, 0): mounted upside down, corrected in unitree_ros commit 7c40519 (2026-06-16) [12][20] | not on the SDK DDS topics; Livox network protocol |
| Torso IMU | — | `imu_in_torso` (section 7) | `rt/secondary_imu` |
| Wrist cameras (optional) | RealSense D405 ×2 [16] | — | USB on PC2 |
| Battery | — | — | `BmsState_` on `rt/lf/bmsstate` [13] |

The deprecated `g1_29dof.urdf` has different camera/lidar/torso-IMU z offsets and no lidar flip [12]; use the rev 1.0 or mode-specific file.

## 9. Mapping onto the Robot port

**`description`**: `description.yaml` (description joint order = SDK motor order, so `motor_index[i] == i`).

**`capabilities()`**: `effort_command = True` (`MotorCmd_.tau`); `temperature = True` (`MotorState_.temperature`); `max_rate_hz = 500`.

**`connect()`**
1. `ChannelFactory.Init(0, nic)`.
2. Check no other `rt/lowcmd` publisher (0.2 s listen, as `unitree_rl_lab`) [15].
3. Subscribe `rt/lowstate` (and `rt/secondary_imu` if the torso IMU is exposed); wait for the first message (`tick != 0`) [14]; verify CRC on receipt [3].
4. Read `mode_machine`; refuse to run if it is not in the set the description was made for (5 for this file) [15].
5. Release the vendor controller (motion switcher `ReleaseMode` loop) or require the operator to have entered debug mode; never do both silently [3][14].
6. Start the writer thread (500 Hz) in PASSIVE.

**`read()`** (latest cached message, no blocking)
- `position[i] = motor_state[i].q`, `velocity[i] = .dq`, `effort[i] = .tau_est` (joint space in PR mode).
- `imu.orientation = (q[1], q[2], q[3], q[0])` from `imu_state.quaternion` (w, x, y, z) → (x, y, z, w).
- `imu.angular_velocity = gyroscope`; `imu.linear_acceleration = accelerometer` (frame `imu_in_pelvis`).
- `temperature[i]`: `temperature[1]` (winding) or `max(temperature)`; the port has one value per joint, choose one and document it.
- `faults`: any `motorstate != 0` (code passed through), CRC failures, stale state (> N ms since last message), `mode_machine` change.
- `stamp_ns`: host monotonic time at receipt.

**`write(JointCommand)`** (store; the writer thread publishes)
- Clamp `position` to the limits (the reference deployments additionally use a 0.9 soft factor in training [15]) and `effort` to ±effort limit.
- For i in 0..28: `motor_cmd[i] = {mode: 1, q, dq, tau, kp, kd}`; slots 29–34 zero (slot 29 is the `rt/arm_sdk` weight; leave 0 on `rt/lowcmd`).
- `mode_pr = 0`, `mode_machine = last LowState_.mode_machine`, `crc = CRC32(...)`, publish.
- Only in POLICY; ignored (logged) in PASSIVE/HOLD/DAMP.

**`set_mode(mode)`**: PASSIVE / HOLD / POLICY as in section 5. Ramp from the current pose when entering HOLD (both references interpolate over 2–3 s [14][15]).

**`operator_stop()`**: switch the writer to DAMP immediately (kp = 0, kd = damping value, tau = 0) and latch it until a person clears it. Keep publishing in DAMP: assume the robot keeps applying the last received command (section 4), so stopping publication is not a stop.

**`close()`**: DAMP, keep publishing for a short period, then stop the writer. Optionally return control with `SwitchToInternalCtrl` (user-control path) or `MotionSwitcher.SelectMode(...)`; the robot must be supported (suspended or on the ground in a safe pose) first.

**Conversions**
- Quaternion (w, x, y, z) → (x, y, z, w).
- Index: identity for SDK order. Isaac Lab policies use breadth-first order; `deploy.yaml` `joint_ids_map` maps policy index → SDK index (`[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]`) [15].
- Units are already SI (rad, rad/s, N·m) for joints and gyro; accelerometer and temperature units need confirming on the unit.
- Frames: pelvis and URDF links follow REP-103; no axis remapping found in any vendor source.

## 10. Policy inference formats

**`unitree_rl_gym` (legged_gym, Isaac Gym)** [14]
- Format: TorchScript (`torch.jit.load`), exported by `PolicyExporterLSTM`: an LSTM (1 layer, 64 hidden) whose hidden/cell state are buffers inside the module, plus a 32-unit MLP actor. Runtimes: Python `deploy_real.py`; C++ `cpp_g1` with libtorch 2.7.1 CPU.
- Controls the 12 leg joints only; waist and arms held at 0 with kp `[300, 300, 300, 100, 100, 50, 50, 20, 20, 20, …]`, kd `[3, 3, 3, 2, 2, 2, 2, 1, 1, 1, …]` (`configs/g1.yaml`).
- Rate 50 Hz (`control_dt 0.02`; training sim dt 0.005, decimation 4).
- Observation (47): `ang_vel·0.25` (3), `gravity` (3), `cmd·[2, 2, 0.25]·max_cmd[0.8, 0.5, 1.57]` (3), `q − default` (12), `dq·0.05` (12), `last_action` (12), `sin, cos` of gait phase (period 0.8 s) (2).
- Action: `q* = default + 0.25·a`; legs kp `[100, 100, 100, 150, 40, 40]`, kd `[2, 2, 2, 4, 2, 2]`; leg default `[-0.1, 0, 0, 0.3, -0.2, 0]`.

**`unitree_rl_lab` (Isaac Lab 2.3 / Isaac Sim 5.1)** [15]
- Format: ONNX, opset 11, input `obs` [1, 480], output `actions` [1, 29] (`deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx`); C++ runtime onnxruntime 1.22.0. Each policy directory carries `params/deploy.yaml` written by `export_deploy_cfg.py`.
- `deploy.yaml` fields: `joint_ids_map`; `step_dt` 0.02; `stiffness`/`damping` **in SDK order**; `default_joint_pos` **in policy order**; action term (`scale` 0.25, `offset` = default pose); observation terms with scale and `history_length`.
- Observation (velocity task): `base_ang_vel·0.2` (3), `projected_gravity` (3), `velocity_commands` (3), `joint_pos_rel` (29), `joint_vel_rel·0.05` (29), `last_action` (29), each with history 5, concatenated term by term (each term's 5 frames oldest → newest, then the next term) → 96 × 5 = 480.
- Command ranges: `lin_vel_x` [−0.5, 1.0], `lin_vel_y` [−0.3, 0.3], `ang_vel_z` [−0.2, 0.2] m/s, rad/s.
- Training: sim dt 0.005, decimation 4 → 50 Hz; deployment FSM writes at 1 kHz, policy thread at 50 Hz.
- Also ships motion-tracking ("Mimic") ONNX policies with 60 Hz reference-motion CSVs and their own analytically derived gains (`stiffness = armature·(2π·10 Hz)²`, `damping = 2·2·armature·2π·10`).

## 11. Public model files and licences

| Repository | SPDX | G1 content |
|---|---|---|
| unitreerobotics/unitree_ros `robots/g1_description` [12] | BSD-3-Clause | URDF for every variant (23-DoF, 29-DoF rev 1.0, lock-waist, modes 10–18, with Dex3/Inspire/BrainCo hands, dual-arm, G1+ `g1_pl`), MJCF for the main variants (`g1_29dof_rev_1_0.xml` etc.), STL meshes, `motor_info.md`, IMU/camera/lidar frames |
| unitreerobotics/unitree_mujoco `unitree_robots/g1` [13] | BSD-3-Clause | `g1_29dof.xml`, `g1_23dof.xml`, scenes, STL meshes, `g1_joint_index_dds.md`; simulator bridge speaking the same DDS topics |
| unitreerobotics/unitree_rl_gym `resources/robots/g1_description` [14] | BSD-3-Clause | 12/23/29-DoF URDF + MJCF + meshes; pre-trained `deploy/pre_train/g1/motion.pt` |
| unitreerobotics/unitree_rl_lab [15] | Apache-2.0 (repo `LICENCE`); `assets/robots/unitree.py` header says BSD-3-Clause (Isaac Lab Project Developers) | articulation configs (gains, defaults, actuator models); ONNX policies; loads USD from Hugging Face `unitreerobotics/unitree_model` or URDF from unitree_ros |
| huggingface.co/datasets/unitreerobotics/unitree_model [19] | NOASSERTION (no licence declared) | USD conversions (`G1/29dof/usd/g1_29dof_rev_1_0`) |
| unitreerobotics/unitree_sdk2 [1] | BSD-3-Clause; bundles CycloneDDS (EPL-2.0 OR BSD-3-Clause), iceoryx (Apache-2.0), RapidJSON (MIT); core is a prebuilt static library | IDL headers, examples, no model files |

## 12. To verify on the real unit

- [ ] `LowState_.mode_machine` value (expected 5 for rev 1.0 29-DoF); pick the matching URDF and limits.
- [ ] Hip pitch, wrist and waist roll/pitch effort/velocity limits for that `mode_machine`; ankle and waist roll/pitch joint-space effort in PR mode (35 vs 50 vs 41/60 N·m).
- [ ] Actual `rt/lowstate` rate and jitter; whether `rt/lowcmd` above 500 Hz has any effect; motor-side PD rate.
- [ ] Behaviour when `rt/lowcmd` stops (hold last command, damp, or time-out to zero torque) and after it resumes.
- [ ] Behaviour on bad CRC and on `mode_machine` mismatch in `LowCmd_` (dropped silently?).
- [ ] Debug-mode entry (`L2 + R2`) and exit on the installed firmware; motion-switcher `CheckMode` names; whether `SwitchToUserCtrl`/`rt/user_lowcmd` exists on the firmware.
- [ ] Meaning of `MotorCmd_.mode = 0` (disable = free-wheel?) vs `mode = 1, kp = kd = 0`.
- [ ] `LowState_.tick` units and wrap; `MotorState_.mode` and `motorstate` code meanings.
- [ ] IMU: quaternion world frame (initial yaw, drift), accelerometer units (m/s² vs g) and sign at rest, gyro axes vs pelvis axes, actual IMU position (URDF vs MJCF disagree).
- [ ] `temperature` units (°C) and which index is winding vs casing.
- [ ] Joint zero positions and sign conventions match the URDF (command 0.1 rad on each joint in HOLD while suspended).
- [ ] `rt/lf/lowstate` and `rt/lf/bmsstate` topic presence and rates on hardware.
- [ ] Lidar model, mount (upside down), IP and data path; D435i serial/USB path on PC2 and its optical-frame transform.
- [ ] PC2 hardware, OS, CycloneDDS version installed on it; whether low-level control from PC2 meets 500 Hz.
- [ ] Damping value for DAMP (kd 3 vs 8) that brings the robot down safely on the gantry.

## 13. Sources

Repositories are cited at the commits below, as of 2026-10-04. Paths are relative to the repository root.

1. unitreerobotics/unitree_sdk2 @ `63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36` (2026-09-21): https://github.com/unitreerobotics/unitree_sdk2/tree/63096d0ac0c5d2dec9d6e0c22cd5233410ca2f36 — `README.md`, `LICENSE`, `licenses/`
2. unitree_sdk2 hg IDL: `include/unitree/idl/hg/{LowCmd_,LowState_,MotorCmd_,MotorState_,IMUState_,HandCmd_,BmsState_}.hpp`, `include/unitree/idl/hg_doubleimu/doubleIMUState_.hpp` (same commit as [1])
3. unitree_sdk2 `example/g1/low_level/g1_ankle_swing_example.cpp` (same commit)
4. unitree_sdk2 `include/unitree/robot/g1/loco/g1_loco_api.hpp`, `g1_loco_client.hpp` (same commit)
5. unitree_sdk2 `include/unitree/robot/b2/motion_switcher/motion_switcher_api.hpp`, `motion_switcher_client.hpp` (same commit)
6. unitree_sdk2 `include/unitree/robot/g1/common/terminations.hpp` (same commit)
7. unitree_sdk2 `include/unitree/dds_wrapper/robots/g1/{defines.h,g1_pub.h,g1_sub.h}` (same commit)
8. unitree_sdk2 `example/g1/high_level/g1_userctrl_dds_example.cpp` (same commit)
9. unitree_sdk2 `thirdparty/include/dds/version.h` (`DDS_VERSION "0.10.2"`) (same commit)
10. unitreerobotics/unitree_sdk2_python @ `814556d15970dd2ecf1c9984e845ca02ab07e206` (2026-09-21): https://github.com/unitreerobotics/unitree_sdk2_python/tree/814556d15970dd2ecf1c9984e845ca02ab07e206 — `README.md`, `setup.py`, `unitree_sdk2py/utils/crc.py`, `unitree_sdk2py/idl/unitree_hg/msg/dds_/`, `unitree_sdk2py/g1/loco/g1_loco_client.py`, `example/g1/low_level/g1_low_level_example.py`, `example/g1/readme.md`, `example/motionSwitcher/motion_switcher_example.py`
11. unitreerobotics/unitree_ros2 @ `668d1ec5a05d1c38d3306bdca7d59f2ba3581a88` (2026-07-02): https://github.com/unitreerobotics/unitree_ros2/tree/668d1ec5a05d1c38d3306bdca7d59f2ba3581a88 — `README.md`, `cyclonedds_ws/src/unitree/unitree_hg/msg/*.msg`, `example/src/src/g1/lowlevel/g1_low_level_example.cpp`
12. unitreerobotics/unitree_ros @ `5994d4faef0a9cadd3287f8de0199a67eeb2a259` (2026-09-30): https://github.com/unitreerobotics/unitree_ros/tree/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/g1_description — `README.md`, `motor_info.md`, `g1_29dof_rev_1_0.urdf`, `g1_29dof_rev_1_0.xml`, `g1_29dof_mode_{11,13,15,18}.urdf`, `g1_29dof.urdf`
13. unitreerobotics/unitree_mujoco @ `1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d` (2026-09-07): https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d — `unitree_robots/g1/g1_29dof.xml`, `unitree_robots/g1/g1_joint_index_dds.md`, `simulate/src/unitree_sdk2_bridge.h`, `simulate/config.yaml`, `simulate_python/config.py`
14. unitreerobotics/unitree_rl_gym @ `276801e46c5d433564f24658bac64f254b7d2d4b` (2025-07-25): https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b — `deploy/deploy_real/{README.md,deploy_real.py,configs/g1.yaml,common/command_helper.py,common/rotation_helper.py,common/remote_controller.py,cpp_g1/Controller.cpp}`, `deploy/deploy_mujoco/configs/g1.yaml`, `legged_gym/envs/g1/g1_config.py`, `legged_gym/utils/helpers.py`, `README.md`
15. unitreerobotics/unitree_rl_lab @ `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` (2025-11-19): https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3 — `README.md`, `LICENCE`, `source/unitree_rl_lab/unitree_rl_lab/assets/robots/{unitree.py,unitree_actuators.py}`, `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py`, `source/unitree_rl_lab/unitree_rl_lab/utils/export_deploy_cfg.py`, `deploy/robots/g1_29dof/{main.cpp,config/config.yaml,config/policy/velocity/v0/params/deploy.yaml,config/policy/velocity/v0/exported/policy.onnx}`, `deploy/include/{unitree_articulation.h,FSM/State_Passive.h,FSM/State_RLBase.h,FSM/CtrlFSM.h,isaaclab/manager/observation_manager.h,isaaclab/manager/manager_term_cfg.h}`
16. unitreerobotics/xr_teleoperate @ `817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b` (2026-09-07): https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b — `README.md`, `Device.md`
17. xr_teleoperate wiki @ `25e500693bdfbef8f796039a451d590a9e1abca1` (2026-06-16): https://github.com/unitreerobotics/xr_teleoperate/wiki/Motion, https://github.com/unitreerobotics/xr_teleoperate/wiki/CycloneDDS-and-UnitreeSDK-(en), https://github.com/unitreerobotics/xr_teleoperate/wiki/Network
18. Unitree G1 product page, https://www.unitree.com/g1 (as of 2026-10-04)
19. Hugging Face dataset unitreerobotics/unitree_model @ `323e350252b9c3aee9c40acbfdad84f6ce46a5ac`: https://huggingface.co/datasets/unitreerobotics/unitree_model
20. Commit history: unitree_ros `7c40519e02` "fix g1 mid360_joint transform" (2026-06-16), https://github.com/unitreerobotics/unitree_ros/commit/7c40519e02 ; unitree_sdk2 `30405b31d8` "add user control example" (2026-03-02), https://github.com/unitreerobotics/unitree_sdk2/commit/30405b31d8

The repositories also link to the vendor's developer documentation at support.unitree.com, which these notes do not cite; the checklist in section 12 covers what it would settle: https://support.unitree.com/home/en/G1_developer
