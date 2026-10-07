# testbot

testbot is a small six-joint biped, written from scratch for testing
Embodiverse in simulation: the MuJoCo adapter, the conformance kit, the
runner and the safety layer all run against it end to end. It is not a
model of any real robot, so it has no vendor, no meshes and no licence
terms beyond the project's. Its description and its MuJoCo model,
`testbot.xml`, ship in `embodiverse.robots.testbot`, and the model is the
example in [Adding a robot](../adding-a-robot.md#a-mujoco-model-optional).
The physics step is 2 ms; with its default gains it stands in HOLD, and
with zero torque it falls. The tables below are generated from its
description by `embodiverse generate`.

<!-- BEGIN GENERATED: robot-joint-tables -->
<!-- This block is generated: do not edit it by hand. Change its source, then run: embodiverse generate -->

- Robot description: `testbot`
- Status: `active`
- Joints: 6
- Content hash: `sha256:a336b65eb10734d1c906313c8961046bbf4856da43ed8721fb9a366e1eefcaf0`

| # | Joint | Vendor name | Motor | Position limits (rad) | Velocity limit (rad/s) | Effort limit (N·m) | Default (rad) | kp (N·m/rad) | kd (N·m·s/rad) | Gains side |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | `left_hip_pitch` | — | 0 | -1.5 to 0.5 (-85.9° to 28.6°) | 20 | 60 | -0.3 | 60 | 2 | joint |
| 1 | `left_knee` | — | 1 | 0 to 2.2 (0° to 126.1°) | 20 | 80 | 0.6 | 80 | 2 | joint |
| 2 | `left_ankle_pitch` | — | 2 | -0.8 to 0.6 (-45.8° to 34.4°) | 20 | 30 | -0.3 | 40 | 2 | joint |
| 3 | `right_hip_pitch` | — | 3 | -1.5 to 0.5 (-85.9° to 28.6°) | 20 | 60 | -0.3 | 60 | 2 | joint |
| 4 | `right_knee` | — | 4 | 0 to 2.2 (0° to 126.1°) | 20 | 80 | 0.6 | 80 | 2 | joint |
| 5 | `right_ankle_pitch` | — | 5 | -0.8 to 0.6 (-45.8° to 34.4°) | 20 | 30 | -0.3 | 40 | 2 | joint |

| Joint group | Joints |
|---|---|
| `legs` | `left_hip_pitch`, `left_knee`, `left_ankle_pitch`, `right_hip_pitch`, `right_knee`, `right_ankle_pitch` |
| `left_leg` | `left_hip_pitch`, `left_knee`, `left_ankle_pitch` |
| `right_leg` | `right_hip_pitch`, `right_knee`, `right_ankle_pitch` |

| IMU frame | Parent | `mount_xyz` (m) | `mount_rpy` (rad) |
|---|---|---|---|
| `imu_link` | `pelvis` | [-0.02, 0, 0.04] | [0, 0, 0] |

<!-- END GENERATED: robot-joint-tables -->
