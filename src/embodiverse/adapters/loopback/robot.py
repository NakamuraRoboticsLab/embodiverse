"""The loopback adapter: a stand-in robot for any description, behind the port.

Two classes drive a `MasslessJoints` model, which any description can
have, so the whole pipeline runs on a robot that has no model file, no
simulator and no hardware yet. They differ in how time passes, as the
MuJoCo adapter's two classes do:

- `LoopbackRobot` runs in **lockstep**: time advances only in `advance`,
  so runs are deterministic. It is `Steppable`, and it is its own
  `SimulationProbe`, so the conformance kit runs every check on it,
  including the three that need ground truth.
- `RealtimeLoopbackRobot` runs in **real time**, and has the shape of a
  hardware adapter: it is a `TransportRobot` over a `LoopbackTransport`,
  with the base's sender thread re-sending the current law at the physics
  rate. It is the template for a vendor adapter: copy this package,
  replace the transport with one that speaks the vendor's wire, and keep
  the rest.

Both report feed-forward torque as supported, no motor temperatures, and
`control_hz_max` as the useful command rate. The joints follow their
targets exponentially (`embodiverse.adapters.loopback.model`); the base
rests wherever the probe puts it. docs/adding-a-robot.md shows how to copy
this package into a hardware adapter.
"""

import math

from numpy.typing import ArrayLike

from embodiverse.adapters.base import RobotBase, Transport, TransportRobot
from embodiverse.adapters.loopback.model import MasslessJoints
from embodiverse.adapters.loopback.transport import LoopbackTransport
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.robot import Capabilities
from embodiverse.contracts.types import RobotState

__all__ = ["LoopbackRobot", "RealtimeLoopbackRobot", "default_capabilities"]


def default_capabilities(description: RobotDescription) -> Capabilities:
    """What the loopback supports: effort, no temperatures, the description's rate."""
    return Capabilities(
        effort_command=True,
        temperature=False,
        max_rate_hz=description.rates.control_hz_max,
    )


class LoopbackRobot(RobotBase):
    """A massless-joint robot in lockstep: `Robot`, `Steppable` and `SimulationProbe`.

    `capabilities` defaults to `default_capabilities(description)`; with
    `temperature=True`, every motor reports 35 °C.
    """

    def __init__(
        self, description: RobotDescription, capabilities: Capabilities | None = None
    ) -> None:
        super().__init__(
            description,
            capabilities or default_capabilities(description),
            damp_on_close=False,
        )
        self._model = MasslessJoints(description)

    @property
    def model(self) -> MasslessJoints:
        """The robot's true state, for tests."""
        return self._model

    # --- Hooks ---------------------------------------------------------------

    def _open(self) -> None:
        pass

    def _latest(self) -> RobotState:
        model = self._model
        return model.state(model.stamp_ns, temperature=self._capabilities.temperature)

    # --- Steppable -----------------------------------------------------------

    def advance(self, seconds: float) -> None:
        """Apply the current law for `seconds`, a positive whole number of periods.

        Raises ValueError otherwise, and RuntimeError if not connected.
        """
        self._check_connected()
        hz = self._description.rates.physics_hz
        steps = round(seconds * hz) if math.isfinite(seconds) else 0
        if steps < 1 or not math.isclose(steps / hz, seconds, rel_tol=1e-9):
            raise ValueError(
                f"advance({seconds!r}): expected a positive whole number of"
                f" physics steps of 1/{hz:g} s"
            )
        model = self._model
        for _ in range(steps):
            model.step(self._control[1])

    # --- SimulationProbe -----------------------------------------------------

    def set_base_pose(self, position: ArrayLike, orientation_xyzw: ArrayLike) -> None:
        """Hold the base at this orientation, at rest; the position is ignored."""
        self._check_connected()
        self._model.set_base_pose(position, orientation_xyzw)

    def release_base(self) -> None:
        """Release the base, which rests where it is."""
        self._check_connected()

    def set_joint_positions(self, positions: ArrayLike) -> None:
        """Set every joint's position, in description order, with zero velocity."""
        self._check_connected()
        self._model.set_joint_positions(positions)


class RealtimeLoopbackRobot(TransportRobot):
    """A massless-joint robot in real time, shaped like a hardware adapter.

    A `TransportRobot` over a `LoopbackTransport`, or over the `transport`
    given, which a test may replace with a broken one.
    """

    def __init__(
        self,
        description: RobotDescription,
        transport: Transport | None = None,
    ) -> None:
        super().__init__(
            description,
            default_capabilities(description),
            transport or LoopbackTransport(description),
        )
