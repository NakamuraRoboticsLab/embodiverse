"""Policies: what turns each tick's observation into an action.

A `Policy` takes one observation, the `input_size` float32 values the
observation builder made for a tick, and returns one action, `output_size`
float32 values: one per joint the descriptor's action drives, in output
order, before the action's scale. There is no batch dimension. The runner
calls it once per tick, from the control thread, and turns the action into
a joint command (docs/policies.md).

Two policies are provided:

- `ZeroPolicy` returns zeros. With the descriptor of `zero_policy_descriptor`
  in `embodiverse.runtime.runner`, it holds every joint at its default
  position through the whole pipeline, which tests the pipeline itself.
- `OnnxPolicy` runs a trained policy exported to ONNX, with onnxruntime on
  the CPU. It needs the `runtime` extra (`embodiverse[runtime]`).

A policy's values are not checked: a non-finite action reaches the safety
layer as data, and the safety layer holds the robot.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from embodiverse.contracts.description import RobotDescription
from embodiverse.contracts.descriptor import (
    PolicyDescriptor,
    resolve_joints,
    verify_policy_file,
)

if TYPE_CHECKING:
    import onnxruntime

__all__ = ["OnnxPolicy", "Policy", "PolicyError", "ZeroPolicy"]

_FLOAT = "tensor(float)"
"""How onnxruntime names the type of a float32 tensor."""


class PolicyError(ValueError):
    """A policy file cannot be loaded, or does not take and produce the
    sizes it is run with."""


@runtime_checkable
class Policy(Protocol):
    """Turns one observation into one action, once per control tick.

    The runner checks the sizes against the policy's descriptor before the
    policy runs, and calls the policy from the control thread only.
    """

    @property
    def input_size(self) -> int:
        """Number of values in an observation: the descriptor's
        `observation_size` times its history length."""
        ...

    @property
    def output_size(self) -> int:
        """Number of values in an action: one per joint the action drives."""
        ...

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return the action for `observation`.

        `observation` has shape `(input_size,)`; the result has shape
        `(output_size,)`, as the policy computed it, before the action's
        scale. Raises ValueError if `observation` has another shape.
        """
        ...

    def reset(self) -> None:
        """Forget anything kept from earlier calls, before the policy starts
        again. Does nothing for a policy whose action depends only on its
        observation."""
        ...


class ZeroPolicy:
    """A policy whose action is always zero.

    A position-offset action of zero puts every joint it drives at its
    default position, so with a descriptor that drives every joint, this
    policy makes a robot stand in its default pose, through the observation
    builder, the safety layer and the adapter like any trained policy.

    Raises ValueError if a size is not a positive integer.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        self._input_size = _positive("input_size", input_size)
        self._output_size = _positive("output_size", output_size)

    @property
    def input_size(self) -> int:
        """Number of values in an observation."""
        return self._input_size

    @property
    def output_size(self) -> int:
        """Number of values in an action."""
        return self._output_size

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return `output_size` zeros.

        Raises ValueError if `observation` does not have shape `(input_size,)`.
        """
        _check_observation(observation, self._input_size)
        return np.zeros(self._output_size, dtype=np.float32)

    def reset(self) -> None:
        """Do nothing: the zero policy keeps no state."""


class OnnxPolicy:
    """A trained policy in an ONNX file, run with onnxruntime on the CPU.

    The model must have exactly one input, the observation, and one output,
    the action, both float32 tensors of rank 2. The input must accept shape
    `[1, input_size]` and the output produce `[1, output_size]`: the first
    dimension, the batch, is 1 or symbolic, and the second is the size
    itself, so the sizes can be checked before the policy runs. A recurrent
    policy exported with its state as extra inputs and outputs is refused;
    the descriptor cannot describe one yet (docs/policies.md).

    The session runs on one thread. The networks of locomotion policies are
    small, so more threads gain little, and one thread keeps the time each
    call takes steady and its results the same from run to run.

    Raises PolicyError, a ValueError, naming the file, if it cannot be
    loaded or the model's inputs or outputs do not fit; ValueError if a size
    is not a positive integer; and ImportError if onnxruntime is not
    installed.
    """

    def __init__(
        self, path: str | os.PathLike[str], *, input_size: int, output_size: int
    ) -> None:
        self._path = Path(path)
        self._input_size = _positive("input_size", input_size)
        self._output_size = _positive("output_size", output_size)
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "running an ONNX policy needs the onnxruntime package: install"
                " embodiverse[runtime]"
            ) from error
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        try:
            session = ort.InferenceSession(
                str(self._path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as error:
            # onnxruntime raises its own exception types, derived from Exception.
            raise PolicyError(
                f"{self._path}: cannot load the ONNX policy: {error}"
            ) from error
        inputs, outputs = session.get_inputs(), session.get_outputs()
        _check_count(self._path, "input", "the observation", [i.name for i in inputs])
        _check_count(self._path, "output", "the action", [o.name for o in outputs])
        _check_tensor(self._path, "input", inputs[0], input_size, "accept")
        _check_tensor(self._path, "output", outputs[0], output_size, "produce")
        self._session: onnxruntime.InferenceSession = session
        self._input_name: str = inputs[0].name
        self._output_names: list[str] = [outputs[0].name]

    @classmethod
    def from_descriptor(
        cls, descriptor: PolicyDescriptor, description: RobotDescription
    ) -> "OnnxPolicy":
        """Load the policy file `descriptor` names, with the sizes it gives.

        Checks the file's hash with `verify_policy_file` before loading it,
        so a file other than the one the descriptor describes never runs.
        The input size is `observation_size` times the history length; the
        output size is the number of joints the action drives in
        `description`.

        Raises DescriptorError if the descriptor names no policy file, or
        the file cannot be read or has another hash; ValueError if the
        action names a joint or group the description does not have; and
        what the constructor raises.
        """
        verify_policy_file(descriptor)
        driven = resolve_joints(descriptor.action.joints, description)
        return cls(
            descriptor.policy_path(),
            input_size=descriptor.observation_size * descriptor.history.length,
            output_size=len(driven),
        )

    @property
    def path(self) -> Path:
        """The ONNX file the policy was loaded from."""
        return self._path

    @property
    def input_size(self) -> int:
        """Number of values in an observation."""
        return self._input_size

    @property
    def output_size(self) -> int:
        """Number of values in an action."""
        return self._output_size

    def __call__(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        """Run the model on `observation` and return its action.

        Raises ValueError if `observation` does not have shape
        `(input_size,)`, and PolicyError if the model returns an action of
        another size than it declared.
        """
        _check_observation(observation, self._input_size)
        feed = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        (result,) = self._session.run(self._output_names, {self._input_name: feed})
        action = np.asarray(result, dtype=np.float32)
        if action.shape != (1, self._output_size):
            raise PolicyError(
                f"{self._path}: the policy returned an action of shape"
                f" {action.shape}, but declared [1, {self._output_size}]"
            )
        return action.reshape(self._output_size)

    def reset(self) -> None:
        """Do nothing: the policy's action depends only on its observation."""


def _positive(name: str, value: int) -> int:
    """Return `value` if it is a positive integer; raise ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _check_observation(observation: NDArray[np.float32], size: int) -> None:
    if observation.shape != (size,):
        raise ValueError(
            f"expected an observation of shape ({size},), got shape {observation.shape}"
        )


def _check_count(path: Path, kind: str, meaning: str, names: list[str]) -> None:
    """Check that the model has exactly one `kind` (input or output)."""
    if len(names) == 1:
        return
    listed = ", ".join(repr(name) for name in names) or "none"
    raise PolicyError(
        f"{path}: the policy has {len(names)} {kind}s ({listed}), but a policy has"
        f" exactly one {kind}, {meaning}; a recurrent policy that passes its state"
        " as extra inputs and outputs cannot be run yet"
    )


def _check_tensor(
    path: Path, kind: str, tensor: "onnxruntime.NodeArg", size: int, verb: str
) -> None:
    """Check that an input or output is a float32 tensor of shape [1, size]."""
    shape: list[int | str | None] = list(tensor.shape)
    name = tensor.name
    if tensor.type != _FLOAT:
        raise PolicyError(
            f"{path}: the policy's {kind} {name!r} is a {tensor.type}, but it must"
            f" be a {_FLOAT} (float32)"
        )
    batch_fits = len(shape) == 2 and (shape[0] == 1 or not isinstance(shape[0], int))
    if not (batch_fits and shape[1] == size):
        raise PolicyError(
            f"{path}: the policy's {kind} {name!r} has shape {_shape(shape)}, but it"
            f" must {verb} [1, {size}]: a batch of 1, or a symbolic batch, of"
            f" {size} values"
        )


def _shape(shape: list[int | str | None]) -> str:
    """Return a shape as onnxruntime reports it, symbolic dimensions by name."""
    return "[" + ", ".join("?" if size is None else str(size) for size in shape) + "]"
