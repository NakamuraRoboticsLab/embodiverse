"""The loopback transport: a wire whose far end is a massless-joint robot.

`LoopbackTransport` implements `Transport` without a robot: the law it is
sent drives a `MasslessJoints` model, and the model's state comes back
from `receive`. It runs in real time against the monotonic clock, as
hardware does: each `receive` first steps the model up to the present, one
physics period at a time, applying the latest law, and stamps the state
with the clock time of its last step. A hardware transport differs only
in what `open`, `receive`, `send` and `close` talk to.
"""

import threading
import time

from embodiverse.adapters.loopback.model import MasslessJoints
from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.types import JointCommand, RobotState

__all__ = ["LoopbackTransport"]

_MAX_CATCH_UP_S = 1.0
"""How far behind the clock the model may be before it skips ahead."""


class LoopbackTransport:
    """A `Transport` whose robot is a `MasslessJoints` model run in real time.

    `send` may run on one thread while `receive` runs on another; a lock
    guards the model and the law. If no one received for longer than
    `_MAX_CATCH_UP_S`, the model skips the gap rather than stepping through
    it, so a paused process does not stall its next read.
    """

    def __init__(self, description: RobotDescription) -> None:
        self.model = MasslessJoints(description)
        """The robot at the far end, for tests that want its true state."""
        self._period_ns = round(1e9 / description.rates.physics_hz)
        self._max_catch_up = round(_MAX_CATCH_UP_S * description.rates.physics_hz)
        self._law: JointCommand | None = None
        self._start_ns: int | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        """Start the clock; the first state is available at once."""
        self._start_ns = time.monotonic_ns()

    def receive(self) -> RobotState | None:
        """Step the model up to now with the latest law, and return its state."""
        if self._start_ns is None:
            return None
        with self._lock:
            due = (time.monotonic_ns() - self._start_ns) // self._period_ns
            model = self.model
            if due - model.steps > self._max_catch_up:
                model.steps = due - self._max_catch_up
            if self._law is not None:
                while model.steps < due:
                    model.step(self._law)
            else:
                model.steps = max(model.steps, due)
            return model.state(self._start_ns + model.steps * self._period_ns)

    def send(self, law: JointCommand) -> None:
        """Make `law` the one the model applies from now on."""
        with self._lock:
            self._law = law

    def close(self) -> None:
        """Stop; the model keeps its last state."""
        self._start_ns = None
