"""Tests of the policies: the zero policy, and ONNX policies built in the tests.

The ONNX policies are affine maps, `actions = observation @ weights + bias`,
written to temporary files by the `affine_policy` fixture (conftest.py),
so every action can be computed here by hand.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
import pytest
import yaml
from onnx import TensorProto

from embodiverse.contracts.descriptor import DescriptorError, load_descriptor
from embodiverse.contracts.registry import robot_description
from embodiverse.runtime.policy import OnnxPolicy, Policy, PolicyError, ZeroPolicy

TESTBOT_ZERO = Path(__file__).parents[1] / "fixtures/policies/testbot_zero"

AffinePolicy = Callable[..., Path]
DescriptorWriter = Callable[[dict[str, Any], Path], Path]


def fixture_data() -> dict[str, Any]:
    """Return the testbot descriptor fixture as data: 27 inputs, 6 outputs."""
    text = (TESTBOT_ZERO / "descriptor.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def observation(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# The zero policy.


def test_the_zero_policy_returns_zeros() -> None:
    policy = ZeroPolicy(input_size=3, output_size=6)
    assert isinstance(policy, Policy)
    assert (policy.input_size, policy.output_size) == (3, 6)
    action = policy(observation(0.1, -0.2, -1.0))
    assert action.dtype == np.float32
    assert action.tolist() == [0.0] * 6
    policy.reset()
    assert policy(observation(0.0, 0.0, -1.0)).tolist() == [0.0] * 6


@pytest.mark.parametrize("shape", [(4,), (1, 3), ()])
def test_the_zero_policy_refuses_an_observation_of_another_shape(
    shape: tuple[int, ...],
) -> None:
    policy = ZeroPolicy(input_size=3, output_size=6)
    match = re.escape(f"expected an observation of shape (3,), got shape {shape}")
    with pytest.raises(ValueError, match=match):
        policy(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_sizes_are_positive_integers(size: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^input_size must be a positive integer"):
        ZeroPolicy(size, 6)
    with pytest.raises(ValueError, match="^output_size must be a positive integer"):
        ZeroPolicy(3, size)
    path = tmp_path / "policy.onnx"
    with pytest.raises(ValueError, match="^input_size must be a positive integer"):
        OnnxPolicy(path, input_size=size, output_size=6)


# ONNX policies.


def test_an_onnx_policy_computes_its_model(affine_policy: AffinePolicy) -> None:
    rng = np.random.default_rng(seed=0)
    weights = rng.normal(size=(5, 3)).astype(np.float32)
    bias = rng.normal(size=3).astype(np.float32)
    path = affine_policy(weights, bias)
    policy = OnnxPolicy(path, input_size=5, output_size=3)
    assert isinstance(policy, Policy)
    assert (policy.path, policy.input_size, policy.output_size) == (path, 5, 3)
    for _ in range(3):
        inputs = rng.normal(size=5).astype(np.float32)
        action = policy(inputs)
        assert action.dtype == np.float32
        assert action.shape == (3,)
        np.testing.assert_allclose(action, inputs @ weights + bias, rtol=1e-6)
    policy.reset()


@pytest.mark.parametrize(
    ("input_shape", "output_shape"),
    [
        pytest.param(["batch", 5], ["batch", 3], id="symbolic batch"),
        pytest.param([1, 5], [1, 3], id="batch of 1"),
        pytest.param([None, 5], [None, 3], id="unnamed batch"),
    ],
)
def test_an_onnx_policy_takes_a_batch_of_1_or_a_symbolic_batch(
    affine_policy: AffinePolicy,
    input_shape: list[int | str | None],
    output_shape: list[int | str | None],
) -> None:
    path = affine_policy(
        np.ones((5, 3)),
        np.zeros(3),
        input_shape=input_shape,
        output_shape=output_shape,
    )
    policy = OnnxPolicy(path, input_size=5, output_size=3)
    assert policy(observation(1, 2, 3, 4, 5)).tolist() == [15.0, 15.0, 15.0]


@pytest.mark.parametrize(
    ("model", "sizes", "message"),
    [
        pytest.param(
            {},
            (4, 3),
            "the policy's input 'observation' has shape [batch, 5], but it must"
            " accept [1, 4]: a batch of 1, or a symbolic batch, of 4 values",
            id="another input size",
        ),
        pytest.param(
            {},
            (5, 2),
            "the policy's output 'actions' has shape [batch, 3], but it must"
            " produce [1, 2]: a batch of 1, or a symbolic batch, of 2 values",
            id="another output size",
        ),
        pytest.param(
            {"input_shape": [2, 5], "output_shape": [2, 3]},
            (5, 3),
            "the policy's input 'observation' has shape [2, 5], but it must accept"
            " [1, 5]",
            id="a batch of 2",
        ),
        pytest.param(
            {"input_shape": [5], "output_shape": [3]},
            (5, 3),
            "the policy's input 'observation' has shape [5], but it must accept [1, 5]",
            id="no batch",
        ),
        pytest.param(
            {"input_shape": ["batch", "features"]},
            (5, 3),
            "the policy's input 'observation' has shape [batch, features], but it"
            " must accept [1, 5]",
            id="a symbolic size",
        ),
        pytest.param(
            {"element_type": TensorProto.DOUBLE},
            (5, 3),
            "the policy's input 'observation' is a tensor(double), but it must be a"
            " tensor(float) (float32)",
            id="float64",
        ),
        pytest.param(
            {"state_size": 4},
            (5, 3),
            "the policy has 2 inputs ('observation', 'state'), but a policy has"
            " exactly one input, the observation; a recurrent policy that passes"
            " its state as extra inputs and outputs cannot be run yet",
            id="recurrent state",
        ),
    ],
)
def test_refuses_a_model_that_does_not_fit(
    affine_policy: AffinePolicy,
    model: dict[str, Any],
    sizes: tuple[int, int],
    message: str,
) -> None:
    path = affine_policy(np.ones((5, 3)), np.zeros(3), **model)
    input_size, output_size = sizes
    with pytest.raises(PolicyError, match=f"^{re.escape(f'{path}: {message}')}"):
        OnnxPolicy(path, input_size=input_size, output_size=output_size)


def test_refuses_a_file_that_is_not_an_onnx_model(tmp_path: Path) -> None:
    path = tmp_path / "policy.onnx"
    path.write_text("not an ONNX model\n", encoding="utf-8")
    match = f"^{re.escape(str(path))}: cannot load the ONNX policy: "
    with pytest.raises(PolicyError, match=match):
        OnnxPolicy(path, input_size=27, output_size=6)


def test_an_onnx_policy_refuses_an_observation_of_another_shape(
    affine_policy: AffinePolicy,
) -> None:
    policy = OnnxPolicy(
        affine_policy(np.ones((5, 3)), np.zeros(3)), input_size=5, output_size=3
    )
    with pytest.raises(ValueError, match=re.escape("of shape (5,), got shape (6,)")):
        policy(np.zeros(6, dtype=np.float32))


# Loading from a descriptor.


def test_from_descriptor_takes_the_sizes_from_the_descriptor(
    affine_policy: AffinePolicy, policy_descriptor: DescriptorWriter
) -> None:
    # Two frames of 27 values in; one output per joint of the legs.
    data = fixture_data() | {"history": {"length": 2}}
    path = affine_policy(np.zeros((54, 6)), np.full(6, 0.5))
    descriptor = load_descriptor(policy_descriptor(data, path))
    policy = OnnxPolicy.from_descriptor(descriptor, robot_description("testbot"))
    assert (policy.path, policy.input_size, policy.output_size) == (path, 54, 6)
    assert policy(np.zeros(54, dtype=np.float32)).tolist() == [0.5] * 6


def test_the_testbot_fixture_is_a_zero_policy() -> None:
    descriptor = load_descriptor(TESTBOT_ZERO / "descriptor.yaml")
    policy = OnnxPolicy.from_descriptor(descriptor, robot_description("testbot"))
    action = policy(np.ones(policy.input_size, dtype=np.float32))
    np.testing.assert_array_equal(action, np.zeros(6, dtype=np.float32))


def test_from_descriptor_counts_the_driven_joints(
    affine_policy: AffinePolicy, policy_descriptor: DescriptorWriter
) -> None:
    data = fixture_data()
    data["action"]["joints"] = "left_leg"
    data["held_joints"] = "right_leg"
    path = affine_policy(np.zeros((27, 3)), np.zeros(3))
    descriptor = load_descriptor(policy_descriptor(data, path))
    policy = OnnxPolicy.from_descriptor(descriptor, robot_description("testbot"))
    assert (policy.input_size, policy.output_size) == (27, 3)


def test_from_descriptor_refuses_a_model_that_does_not_fit_the_descriptor(
    affine_policy: AffinePolicy, policy_descriptor: DescriptorWriter
) -> None:
    # The model takes one frame; the descriptor says it sees two.
    data = fixture_data() | {"history": {"length": 2}}
    path = affine_policy(np.zeros((27, 6)), np.zeros(6))
    descriptor = load_descriptor(policy_descriptor(data, path))
    with pytest.raises(PolicyError, match=re.escape("but it must accept [1, 54]")):
        OnnxPolicy.from_descriptor(descriptor, robot_description("testbot"))


def test_from_descriptor_refuses_a_file_with_another_hash_before_loading_it(
    affine_policy: AffinePolicy,
    policy_descriptor: DescriptorWriter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = affine_policy(np.zeros((27, 6)), np.zeros(6))
    descriptor = load_descriptor(policy_descriptor(fixture_data(), path))
    affine_policy(np.zeros((27, 6)), np.ones(6))  # replaces the file

    def load(*args: object, **kwargs: object) -> None:
        raise AssertionError("the policy file was loaded")

    monkeypatch.setattr(onnxruntime, "InferenceSession", load)
    match = f"^{re.escape(str(path))}: the policy file's hash is sha256:"
    with pytest.raises(DescriptorError, match=match):
        OnnxPolicy.from_descriptor(descriptor, robot_description("testbot"))


def test_from_descriptor_refuses_a_descriptor_without_a_policy_file(
    tmp_path: Path,
) -> None:
    data = fixture_data()
    del data["policy"]
    path = tmp_path / "descriptor.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(DescriptorError, match="names no policy file"):
        OnnxPolicy.from_descriptor(load_descriptor(path), robot_description("testbot"))
