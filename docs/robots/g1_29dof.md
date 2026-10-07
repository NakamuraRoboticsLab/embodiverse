# g1_29dof

Draft description of the Unitree G1 with 29 actuated joints, the
`rev_1_0` variant that reports `mode_machine` 5. It is built from
Unitree's public model files and reference deployment configuration,
listed in the description's header and in
[`THIRD_PARTY.md`](../../THIRD_PARTY.md); the G1's
[interface notes](../adapters/unitree-g1.md) describe its low-level
interface and what is still to be checked on a real unit, notably the
ankle and waist effort limits, on which the vendor's files disagree. It
has no MJCF model yet. The tables below are generated from the
description by `embodiverse generate`.

<!-- BEGIN GENERATED: robot-joint-tables -->
<!-- This block is generated: do not edit it by hand. Change its source, then run: embodiverse generate -->

- Robot description: `g1_29dof`
- Status: `draft`
- Joints: 29
- Content hash: `sha256:24a9c425a7a1deb0530703676588d5f06be904ea39915c2ad90b7699d40c52d8`

| # | Joint | Vendor name | Motor | Position limits (rad) | Velocity limit (rad/s) | Effort limit (N·m) | Default (rad) | kp (N·m/rad) | kd (N·m·s/rad) | Gains side |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `left_hip_pitch` | `left_hip_pitch_joint` | 0 | -2.5307 to 2.8798 (-145° to 165°) | 32 | 88 | -0.1 | 100 | 2 | joint |
| 1 | `left_hip_roll` | `left_hip_roll_joint` | 1 | -0.5236 to 2.9671 (-30° to 170°) | 20 | 139 | 0 | 100 | 2 | joint |
| 2 | `left_hip_yaw` | `left_hip_yaw_joint` | 2 | -2.7576 to 2.7576 (-158° to 158°) | 32 | 88 | 0 | 100 | 2 | joint |
| 3 | `left_knee` | `left_knee_joint` | 3 | -0.087267 to 2.8798 (-5° to 165°) | 20 | 139 | 0.3 | 150 | 4 | joint |
| 4 | `left_ankle_pitch` | `left_ankle_pitch_joint` | 4 | -0.87267 to 0.5236 (-50° to 30°) | 30 | 35 | -0.2 | 40 | 2 | joint |
| 5 | `left_ankle_roll` | `left_ankle_roll_joint` | 5 | -0.2618 to 0.2618 (-15° to 15°) | 30 | 35 | 0 | 40 | 2 | joint |
| 6 | `right_hip_pitch` | `right_hip_pitch_joint` | 6 | -2.5307 to 2.8798 (-145° to 165°) | 32 | 88 | -0.1 | 100 | 2 | joint |
| 7 | `right_hip_roll` | `right_hip_roll_joint` | 7 | -2.9671 to 0.5236 (-170° to 30°) | 20 | 139 | 0 | 100 | 2 | joint |
| 8 | `right_hip_yaw` | `right_hip_yaw_joint` | 8 | -2.7576 to 2.7576 (-158° to 158°) | 32 | 88 | 0 | 100 | 2 | joint |
| 9 | `right_knee` | `right_knee_joint` | 9 | -0.087267 to 2.8798 (-5° to 165°) | 20 | 139 | 0.3 | 150 | 4 | joint |
| 10 | `right_ankle_pitch` | `right_ankle_pitch_joint` | 10 | -0.87267 to 0.5236 (-50° to 30°) | 30 | 35 | -0.2 | 40 | 2 | joint |
| 11 | `right_ankle_roll` | `right_ankle_roll_joint` | 11 | -0.2618 to 0.2618 (-15° to 15°) | 30 | 35 | 0 | 40 | 2 | joint |
| 12 | `waist_yaw` | `waist_yaw_joint` | 12 | -2.618 to 2.618 (-150° to 150°) | 32 | 88 | 0 | 200 | 5 | joint |
| 13 | `waist_roll` | `waist_roll_joint` | 13 | -0.52 to 0.52 (-29.8° to 29.8°) | 30 | 35 | 0 | 200 | 5 | joint |
| 14 | `waist_pitch` | `waist_pitch_joint` | 14 | -0.52 to 0.52 (-29.8° to 29.8°) | 30 | 35 | 0 | 200 | 5 | joint |
| 15 | `left_shoulder_pitch` | `left_shoulder_pitch_joint` | 15 | -3.0892 to 2.6704 (-177° to 153°) | 37 | 25 | 0.3 | 40 | 10 | joint |
| 16 | `left_shoulder_roll` | `left_shoulder_roll_joint` | 16 | -1.5882 to 2.2515 (-91° to 129°) | 37 | 25 | 0.25 | 40 | 10 | joint |
| 17 | `left_shoulder_yaw` | `left_shoulder_yaw_joint` | 17 | -2.618 to 2.618 (-150° to 150°) | 37 | 25 | 0 | 40 | 10 | joint |
| 18 | `left_elbow` | `left_elbow_joint` | 18 | -1.0472 to 2.0944 (-60° to 120°) | 37 | 25 | 0.97 | 40 | 10 | joint |
| 19 | `left_wrist_roll` | `left_wrist_roll_joint` | 19 | -1.97222 to 1.97222 (-113° to 113°) | 37 | 25 | 0.15 | 40 | 10 | joint |
| 20 | `left_wrist_pitch` | `left_wrist_pitch_joint` | 20 | -1.61443 to 1.61443 (-92.5° to 92.5°) | 22 | 5 | 0 | 40 | 10 | joint |
| 21 | `left_wrist_yaw` | `left_wrist_yaw_joint` | 21 | -1.61443 to 1.61443 (-92.5° to 92.5°) | 22 | 5 | 0 | 40 | 10 | joint |
| 22 | `right_shoulder_pitch` | `right_shoulder_pitch_joint` | 22 | -3.0892 to 2.6704 (-177° to 153°) | 37 | 25 | 0.3 | 40 | 10 | joint |
| 23 | `right_shoulder_roll` | `right_shoulder_roll_joint` | 23 | -2.2515 to 1.5882 (-129° to 91°) | 37 | 25 | -0.25 | 40 | 10 | joint |
| 24 | `right_shoulder_yaw` | `right_shoulder_yaw_joint` | 24 | -2.618 to 2.618 (-150° to 150°) | 37 | 25 | 0 | 40 | 10 | joint |
| 25 | `right_elbow` | `right_elbow_joint` | 25 | -1.0472 to 2.0944 (-60° to 120°) | 37 | 25 | 0.97 | 40 | 10 | joint |
| 26 | `right_wrist_roll` | `right_wrist_roll_joint` | 26 | -1.97222 to 1.97222 (-113° to 113°) | 37 | 25 | -0.15 | 40 | 10 | joint |
| 27 | `right_wrist_pitch` | `right_wrist_pitch_joint` | 27 | -1.61443 to 1.61443 (-92.5° to 92.5°) | 22 | 5 | 0 | 40 | 10 | joint |
| 28 | `right_wrist_yaw` | `right_wrist_yaw_joint` | 28 | -1.61443 to 1.61443 (-92.5° to 92.5°) | 22 | 5 | 0 | 40 | 10 | joint |

| Joint group | Joints |
|---|---|
| `legs` | `left_hip_pitch`, `left_hip_roll`, `left_hip_yaw`, `left_knee`, `left_ankle_pitch`, `left_ankle_roll`, `right_hip_pitch`, `right_hip_roll`, `right_hip_yaw`, `right_knee`, `right_ankle_pitch`, `right_ankle_roll` |
| `waist` | `waist_yaw`, `waist_roll`, `waist_pitch` |
| `arms` | `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow`, `left_wrist_roll`, `left_wrist_pitch`, `left_wrist_yaw`, `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow`, `right_wrist_roll`, `right_wrist_pitch`, `right_wrist_yaw` |
| `left_arm` | `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow`, `left_wrist_roll`, `left_wrist_pitch`, `left_wrist_yaw` |
| `right_arm` | `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow`, `right_wrist_roll`, `right_wrist_pitch`, `right_wrist_yaw` |

| IMU frame | Parent | `mount_xyz` (m) | `mount_rpy` (rad) |
|---|---|---|---|
| `imu_in_pelvis` | `pelvis` | [0.04525, 0, -0.08339] | [0, 0, 0] |

<!-- END GENERATED: robot-joint-tables -->
