# Third-party sources

Embodiverse's own code and data are under the Apache License 2.0
([`LICENSE`](LICENSE)). The draft robot descriptions in
`src/embodiverse/robots/` (`g1_29dof`, `pm01`, `k1_22dof`) also contain
facts taken from the vendors' public repositories: joint names and order,
position, velocity and effort limits, default poses, reference gains and
sensor mounts. No vendor files (models, meshes, code or configuration) are
included. Each description's header lists the exact files and commits it
was built from.

## Unitree G1 (`g1_29dof`)

| Source | Commit | Licence | Used for |
|---|---|---|---|
| [unitreerobotics/unitree_ros](https://github.com/unitreerobotics/unitree_ros), `robots/g1_description/g1_29dof_rev_1_0.urdf` | `5994d4f` | BSD-3-Clause, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") | Vendor joint names; position, velocity and effort limits; IMU and sensor mounts |
| [unitreerobotics/unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2) | `63096d0` | BSD-3-Clause, Copyright (c) 2016-2023 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics") | Motor order |
| [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab), `deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml` | `4960b84` | Apache-2.0 (repository licence) | Default pose and reference gains |

The `unitree_rl_lab` repository is licensed Apache-2.0, but some of its
files carry other headers: `assets/robots/unitree.py` declares
BSD-3-Clause, and the deployment sources say "All rights reserved". Only
numeric values were taken from it.

## EngineAI PM01 (`pm01`)

| Source | Commit | Licence | Used for |
|---|---|---|---|
| [engineai-robotics/engineai_robotics_native_sdk](https://github.com/engineai-robotics/engineai_robotics_native_sdk) | `335c60e` | BSD-3-Clause, Copyright (c) 2025- Shenzhen Zhongqing Robot Technology Co., Ltd. ("EngineAI") | Vendor joint names, order and limits; IMU mount; default pose and reference gains; rates |
| [engineai-robotics/engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab) | `c53f7ec` | BSD-3-Clause | Cross-check of the default pose and gains |
| [engineai-robotics/engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace) | `c41043f` | BSD-3-Clause | Cross-check of joint data |
| [engineai-robotics/engineai_robotics_description](https://github.com/engineai-robotics/engineai_robotics_description) | `1d2d760` | None stated | Camera and lidar mount positions only |

`engineai_robotics_description` has no licence file, so nothing from it
may be copied; the description records only the mount positions it
states, as facts.

## Booster K1 (`k1_22dof`)

| Source | Commit | Licence | Used for |
|---|---|---|---|
| [BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets), `robots/K1/K1_22dof.urdf` and `.xml` | `12a9751` | BSD-3-Clause, Copyright (c) 2025, BoosterRobotics | Vendor joint names; position, velocity and effort limits; camera frames; IMU site |
| [BoosterRobotics/booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) | `87a9a26` | Apache-2.0 | Motor order |
| [BoosterRobotics/booster_deploy](https://github.com/BoosterRobotics/booster_deploy) | `7bb1462` | Apache-2.0 | Default pose and reference gains |

## Licence texts

The Apache License 2.0 is in [`LICENSE`](LICENSE). The BSD-3-Clause
sources above are distributed under these terms, with the copyright
notices given in the tables:

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are
> met:
>
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in the
>    documentation and/or other materials provided with the distribution.
> 3. Neither the name of the copyright holder nor the names of its
>    contributors may be used to endorse or promote products derived from
>    this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
> IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
> TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
> PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
> HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
> TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
> PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
> LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
> NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
