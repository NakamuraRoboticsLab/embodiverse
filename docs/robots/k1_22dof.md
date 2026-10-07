# k1_22dof

Draft description of the Booster K1 with 22 actuated joints, built from
Booster's public model files and reference deployment configuration,
listed in the description's header and in
[`THIRD_PARTY.md`](../../THIRD_PARTY.md); the K1's
[interface notes](../adapters/booster-k1.md) describe its low-level
interface and what is still to be checked on a real unit. The joints the
vendor calls "elbow pitch" and "elbow yaw" are named by what they do,
`_shoulder_yaw` and `_elbow`. The parallel-linkage ankles are commanded in
joint space (the SDK's serial mode), so their gains are recorded as
joint-side; whether the damping acts at the motors is still to be
checked. It has no MJCF model yet. The tables below are generated from
the description by `embodiverse generate`.

<!-- BEGIN GENERATED: robot-joint-tables -->
<!-- This block is generated: do not edit it by hand. Change its source, then run: embodiverse generate -->

- Robot description: `k1_22dof`
- Status: `draft`
- Joints: 22
- Content hash: `sha256:6d2d269ca5639841556cd29afc13843e6c0c616030ce42c8f45266fa521420bf`

| # | Joint | Vendor name | Motor | Position limits (rad) | Velocity limit (rad/s) | Effort limit (N·m) | Default (rad) | kp (N·m/rad) | kd (N·m·s/rad) | Gains side |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `head_yaw` | `aahead_yaw_joint` | 0 | -1.012 to 1.012 (-58° to 58°) | 7.85 | 6 | 0 | 4 | 1 | joint |
| 1 | `head_pitch` | `aahead_pitch_joint` | 1 | -0.314 to 0.794 (-18° to 45.5°) | 7.85 | 6 | 0 | 4 | 1 | joint |
| 2 | `left_shoulder_pitch` | `aaleft_shoulder_pitch_joint` | 2 | -2.932 to 1.196 (-168° to 68.5°) | 33.51 | 14 | 0.2 | 20 | 2 | joint |
| 3 | `left_shoulder_roll` | `left_shoulder_roll_joint` | 3 | -1.642 to 1.629 (-94.1° to 93.3°) | 33.51 | 14 | -1.25 | 20 | 2 | joint |
| 4 | `left_shoulder_yaw` | `left_elbow_pitch_joint` | 4 | -1.885 to 1.885 (-108° to 108°) | 33.51 | 14 | 0 | 20 | 2 | joint |
| 5 | `left_elbow` | `left_elbow_yaw_joint` | 5 | -2.242 to 0.816 (-128.5° to 46.8°) | 33.51 | 14 | -0.5 | 20 | 2 | joint |
| 6 | `right_shoulder_pitch` | `aaright_shoulder_pitch_joint` | 6 | -2.932 to 1.196 (-168° to 68.5°) | 33.51 | 14 | 0.2 | 20 | 2 | joint |
| 7 | `right_shoulder_roll` | `right_shoulder_roll_joint` | 7 | -1.642 to 1.629 (-94.1° to 93.3°) | 33.51 | 14 | 1.25 | 20 | 2 | joint |
| 8 | `right_shoulder_yaw` | `right_elbow_pitch_joint` | 8 | -1.885 to 1.885 (-108° to 108°) | 33.51 | 14 | 0 | 20 | 2 | joint |
| 9 | `right_elbow` | `right_elbow_yaw_joint` | 9 | -0.816 to 2.242 (-46.8° to 128.5°) | 33.51 | 14 | 0.5 | 20 | 2 | joint |
| 10 | `left_hip_pitch` | `left_hip_pitch_joint` | 10 | -2.958 to 2.226 (-169.5° to 127.5°) | 14.66 | 68 | -0.15 | 100 | 2 | joint |
| 11 | `left_hip_roll` | `left_hip_roll_joint` | 11 | -0.375 to 1.536 (-21.5° to 88°) | 12.57 | 43 | 0 | 100 | 2 | joint |
| 12 | `left_hip_yaw` | `left_hip_yaw_joint` | 12 | -1.012 to 1.012 (-58° to 58°) | 17.59 | 38.3 | 0 | 100 | 2 | joint |
| 13 | `left_knee` | `left_knee_pitch_joint` | 13 | 0 to 2.321 (0° to 133°) | 12.57 | 112 | 0.3 | 100 | 2 | joint |
| 14 | `left_ankle_pitch` | `left_ankle_pitch_joint` | 14 | -0.87 to 0.345 (-49.8° to 19.8°) | 17.59 | 38.3 | -0.15 | 65 | 1 | joint |
| 15 | `left_ankle_roll` | `left_ankle_roll_joint` | 15 | -0.345 to 0.345 (-19.8° to 19.8°) | 17.59 | 38.3 | 0 | 65 | 1 | joint |
| 16 | `right_hip_pitch` | `right_hip_pitch_joint` | 16 | -2.958 to 2.226 (-169.5° to 127.5°) | 14.66 | 68 | -0.15 | 100 | 2 | joint |
| 17 | `right_hip_roll` | `right_hip_roll_joint` | 17 | -1.536 to 0.375 (-88° to 21.5°) | 12.57 | 43 | 0 | 100 | 2 | joint |
| 18 | `right_hip_yaw` | `right_hip_yaw_joint` | 18 | -1.012 to 1.012 (-58° to 58°) | 17.59 | 38.3 | 0 | 100 | 2 | joint |
| 19 | `right_knee` | `right_knee_pitch_joint` | 19 | 0 to 2.321 (0° to 133°) | 12.57 | 112 | 0.3 | 100 | 2 | joint |
| 20 | `right_ankle_pitch` | `right_ankle_pitch_joint` | 20 | -0.87 to 0.345 (-49.8° to 19.8°) | 17.59 | 38.3 | -0.15 | 65 | 1 | joint |
| 21 | `right_ankle_roll` | `right_ankle_roll_joint` | 21 | -0.345 to 0.345 (-19.8° to 19.8°) | 17.59 | 38.3 | 0 | 65 | 1 | joint |

| Joint group | Joints |
|---|---|
| `legs` | `left_hip_pitch`, `left_hip_roll`, `left_hip_yaw`, `left_knee`, `left_ankle_pitch`, `left_ankle_roll`, `right_hip_pitch`, `right_hip_roll`, `right_hip_yaw`, `right_knee`, `right_ankle_pitch`, `right_ankle_roll` |
| `arms` | `left_shoulder_pitch`, `left_shoulder_roll`, `left_shoulder_yaw`, `left_elbow`, `right_shoulder_pitch`, `right_shoulder_roll`, `right_shoulder_yaw`, `right_elbow` |
| `head` | `head_yaw`, `head_pitch` |

| IMU frame | Parent | `mount_xyz` (m) | `mount_rpy` (rad) |
|---|---|---|---|
| `imu_link` | `trunk` | [0, 0, 0] | [0, 0, 0] |

<!-- END GENERATED: robot-joint-tables -->
