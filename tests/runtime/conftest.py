"""Fixtures for the runtime tests: tiny ONNX policies, written to files.

Each policy is an affine map, `actions = observation @ weights + bias`, built
with onnx.helper, so a test knows exactly what it outputs. A descriptor
written next to it records the file's real SHA-256.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnx
import pytest
import yaml
from numpy.typing import ArrayLike
from onnx import TensorProto, helper, numpy_helper

# onnxruntime 1.23, the newest release for Python 3.10, reads models up to IR
# version 11; onnx writes newer ones by default.
IR_VERSION = 8
OPSET = 17

Dimension = int | str | None


class AffinePolicy(Protocol):
    """Writes an ONNX file computing `observation @ weights + bias`; returns its path.

    `input_shape` and `output_shape` default to a symbolic batch, `batch`,
    and the sizes the weights give. `element_type` is the type of the input,
    output and weights. `state_size` adds a second input, `state`, and a
    second output, `next_state`, as a recurrent policy exported with its
    state has; it is passed through unchanged.
    """

    def __call__(
        self,
        weights: ArrayLike,
        bias: ArrayLike,
        *,
        name: str = "policy.onnx",
        input_shape: Sequence[Dimension] | None = None,
        output_shape: Sequence[Dimension] | None = None,
        element_type: int = TensorProto.FLOAT,
        state_size: int = 0,
    ) -> Path: ...


class DescriptorWriter(Protocol):
    """Writes descriptor `data` next to `policy`, naming it with its real hash."""

    def __call__(self, data: dict[str, Any], policy: Path) -> Path: ...


@pytest.fixture
def affine_policy(tmp_path: Path) -> AffinePolicy:
    def write(
        weights: ArrayLike,
        bias: ArrayLike,
        *,
        name: str = "policy.onnx",
        input_shape: Sequence[Dimension] | None = None,
        output_shape: Sequence[Dimension] | None = None,
        element_type: int = TensorProto.FLOAT,
        state_size: int = 0,
    ) -> Path:
        dtype = helper.tensor_dtype_to_np_dtype(element_type)
        w = np.asarray(weights, dtype=dtype)
        b = np.asarray(bias, dtype=dtype)
        inputs = [
            helper.make_tensor_value_info(
                "observation", element_type, input_shape or ["batch", w.shape[0]]
            )
        ]
        outputs = [
            helper.make_tensor_value_info(
                "actions", element_type, output_shape or ["batch", w.shape[1]]
            )
        ]
        nodes = [
            helper.make_node("MatMul", ["observation", "weights"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["actions"]),
        ]
        if state_size:
            shape = ["batch", state_size]
            inputs.append(helper.make_tensor_value_info("state", element_type, shape))
            outputs.append(
                helper.make_tensor_value_info("next_state", element_type, shape)
            )
            nodes.append(helper.make_node("Identity", ["state"], ["next_state"]))
        graph = helper.make_graph(
            nodes,
            "affine",
            inputs,
            outputs,
            initializer=[
                numpy_helper.from_array(w, "weights"),
                numpy_helper.from_array(b, "bias"),
            ],
        )
        model = helper.make_model(
            graph,
            ir_version=IR_VERSION,
            opset_imports=[helper.make_opsetid("", OPSET)],
        )
        onnx.checker.check_model(model)
        path = tmp_path / name
        onnx.save(model, path)
        return path

    return write


@pytest.fixture
def policy_descriptor() -> DescriptorWriter:
    def write(data: dict[str, Any], policy: Path) -> Path:
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        data = {**data, "policy": {"file": policy.name, "sha256": f"sha256:{digest}"}}
        path = policy.parent / "descriptor.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return write
