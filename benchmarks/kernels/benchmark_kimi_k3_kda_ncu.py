# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-GPU Nsight Compute targets for production Kimi-K3 KDA kernels.

Setup, correctness checks, and warmup are outside the target NVTX range. NCU
results from this microbenchmark describe kernels, not end-to-end latency.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

HEAD_DIM = 128
NUM_HEADS = 96
CONV_WIDTH = 4
PREFILL_TOKENS = 16384
LOWER_BOUND = -5.0
NORM_EPS = 1e-5
DTYPE = torch.bfloat16
TARGETS = ("prefill-conv1d", "prefill-kda", "decode-fused")
NVTX_RANGES = {
    "prefill-conv1d": "kimi_k3_kda_ncu/prefill_conv1d",
    "prefill-kda": "kimi_k3_kda_ncu/prefill_kda",
    "decode-fused": "kimi_k3_kda_ncu/decode_fused",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-iters", type=int, default=1)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if args.profile_iters != 1:
        raise ValueError("profile_iters must be 1 because KDA state is mutable")
    return args


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return (error / scale).item()


def _check_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = _relative_rmse(actual, expected)
    if not torch.isfinite(actual).all() or error >= 0.02:
        raise AssertionError(f"{name} relative RMSE {error:.6f} >= 0.02")
    return error


@dataclass
class ConvInputs:
    x: torch.Tensor
    weight: torch.Tensor
    state: torch.Tensor
    starts: torch.Tensor
    cache_indices: torch.Tensor
    has_initial_state: torch.Tensor
    metadata: SimpleNamespace


def _conv_inputs(tokens: int, dim: int) -> ConvInputs:
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    x = torch.randn(tokens, dim, device="cuda", dtype=DTYPE).transpose(0, 1)
    weight = torch.randn(dim, CONV_WIDTH, device="cuda", dtype=torch.float32)
    state = torch.zeros(1, dim, CONV_WIDTH - 1, device="cuda", dtype=DTYPE)
    starts = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    nums, batch_ptr, offsets = compute_causal_conv1d_metadata(
        starts.cpu(), device=x.device
    )
    return ConvInputs(
        x=x,
        weight=weight,
        state=state,
        starts=starts,
        cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
        has_initial_state=torch.zeros(1, device="cuda", dtype=torch.bool),
        metadata=SimpleNamespace(
            nums_dict=nums,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=offsets,
        ),
    )


def _run_conv(inputs: ConvInputs) -> torch.Tensor:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    return causal_conv1d_fn(
        inputs.x,
        inputs.weight,
        None,
        inputs.state,
        inputs.starts,
        cache_indices=inputs.cache_indices,
        has_initial_state=inputs.has_initial_state,
        activation="silu",
        metadata=inputs.metadata,
    )


def _check_conv() -> float:
    inputs = _conv_inputs(32, 256)
    actual = _run_conv(inputs)
    expected = F.conv1d(
        inputs.x.float().unsqueeze(0),
        inputs.weight.float().unsqueeze(1),
        padding=CONV_WIDTH - 1,
        groups=inputs.x.shape[0],
    )[..., : inputs.x.shape[1]].squeeze(0)
    output_error = _check_close("prefill-conv1d", actual, F.silu(expected).to(DTYPE))
    state_error = _check_close(
        "prefill-conv1d-state",
        inputs.state[0],
        inputs.x[:, -(CONV_WIDTH - 1) :],
    )
    return max(output_error, state_error)


@dataclass
class KDAInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state: torch.Tensor
    starts: torch.Tensor
    out: torch.Tensor
    final_state: torch.Tensor
    workspace: torch.Tensor


def _kda_inputs(tokens: int, heads: int) -> KDAInputs:
    import vllm._flashkda_C  # noqa: F401

    shape = (1, tokens, heads, HEAD_DIM)
    q, k, v, g = [torch.randn(shape, device="cuda", dtype=DTYPE) for _ in range(4)]
    state = torch.zeros(1, heads, HEAD_DIM, HEAD_DIM, device="cuda")
    workspace_size = torch.ops._flashkda_C.get_workspace_size(tokens, heads, 1)
    return KDAInputs(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=torch.randn(1, tokens, heads, device="cuda", dtype=DTYPE),
        a_log=torch.randn(heads, device="cuda") * 0.5,
        dt_bias=torch.randn(heads, HEAD_DIM, device="cuda") * 0.1,
        state=state,
        starts=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
        out=torch.empty_like(v),
        final_state=torch.empty_like(state),
        workspace=torch.empty(workspace_size, device="cuda", dtype=torch.uint8),
    )


def _run_kda(x: KDAInputs) -> None:
    from vllm.models.kimi_k3.nvidia.kda import _flashkda_prefill

    _flashkda_prefill(
        x.q,
        x.k,
        x.v,
        x.g,
        x.beta,
        x.a_log,
        x.dt_bias,
        LOWER_BOUND,
        x.state,
        x.starts,
        x.out,
        x.final_state,
        x.workspace,
    )


def _check_kda() -> float:
    from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

    x = _kda_inputs(16, 2)
    gate = LOWER_BOUND * torch.sigmoid(
        x.a_log.exp()[None, None, :, None] * (x.g.float() + x.dt_bias[None, None, :, :])
    )
    beta = x.beta.float().sigmoid()
    q = l2norm_fwd(x.q.contiguous()).float() * HEAD_DIM**-0.5
    k = l2norm_fwd(x.k.contiguous()).float()
    state = x.state.transpose(-1, -2).float().clone()
    expected = torch.empty_like(x.v)
    for token in range(x.v.shape[1]):
        state *= gate[:, token, :, :, None].exp()
        key = k[:, token]
        value = x.v[:, token].float()
        residual = value - torch.einsum("bhk,bhkv->bhv", key, state)
        state += torch.einsum("bhk,bhv->bhkv", beta[:, token, :, None] * key, residual)
        expected[:, token] = torch.einsum("bhk,bhkv->bhv", q[:, token], state)
    _run_kda(x)
    output_error = _check_close("prefill-kda-output", x.out, expected)
    state_error = _check_close(
        "prefill-kda-state", x.final_state, state.transpose(-1, -2)
    )
    return max(output_error, state_error)


@dataclass
class DecodeInputs:
    x: torch.Tensor
    weight: torch.Tensor
    conv_state: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    indices: torch.Tensor
    state: torch.Tensor
    out: torch.Tensor
    output_gate: torch.Tensor
    norm_weight: torch.Tensor


def _decode_inputs() -> DecodeInputs:
    dim = NUM_HEADS * HEAD_DIM
    return DecodeInputs(
        x=torch.randn(1, 3 * dim, device="cuda", dtype=DTYPE),
        weight=torch.randn(3, CONV_WIDTH, dim, device="cuda"),
        conv_state=torch.randn(2, 3 * dim, CONV_WIDTH - 1, device="cuda", dtype=DTYPE),
        g=torch.randn(1, 1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        beta=torch.randn(1, 1, NUM_HEADS, device="cuda", dtype=DTYPE),
        a_log=torch.randn(NUM_HEADS, device="cuda") * 0.5,
        dt_bias=torch.randn(dim, device="cuda") * 0.1,
        indices=torch.tensor([1], device="cuda", dtype=torch.int32),
        state=torch.randn(2, NUM_HEADS, HEAD_DIM, HEAD_DIM, device="cuda"),
        out=torch.empty(1, 1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        output_gate=torch.randn(1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=DTYPE),
        norm_weight=torch.ones(HEAD_DIM, device="cuda"),
    )


def _run_decode(x: DecodeInputs) -> None:
    from vllm import _custom_ops as ops

    ops.fused_kda_decode(
        x=x.x,
        weight=x.weight,
        bias=None,
        conv_state=x.conv_state,
        raw_g=x.g,
        raw_beta=x.beta,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
        state_indices=x.indices,
        state=x.state,
        out=x.out,
        lower_bound=LOWER_BOUND,
        output_gate=x.output_gate,
        norm_weight=x.norm_weight,
        norm_eps=NORM_EPS,
    )


def _check_decode() -> float:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
        fused_recurrent_kda_packed_decode,
    )

    x = _decode_inputs()
    reference_conv_state = x.conv_state.clone()
    reference_state = x.state.clone()
    conv_out = torch.empty_like(x.x)
    conv_weight = x.weight.permute(0, 2, 1).reshape(-1, CONV_WIDTH)
    causal_conv1d_update(
        x.x,
        reference_conv_state,
        conv_weight,
        None,
        activation="silu",
        conv_state_indices=x.indices,
        validate_data=False,
        out=conv_out,
    )
    core, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=conv_out,
        raw_g=x.g,
        raw_beta=x.beta,
        A_log=x.a_log,
        dt_bias=x.dt_bias,
        lower_bound=LOWER_BOUND,
        initial_state=reference_state,
        state_indices=x.indices,
    )
    core_float = core.float()
    variance = core_float.square().mean(dim=-1, keepdim=True)
    expected = core_float * torch.rsqrt(variance + NORM_EPS)
    expected *= x.norm_weight
    expected *= torch.sigmoid(x.output_gate.float()).unsqueeze(0)
    _run_decode(x)
    output_error = _check_close("decode-fused-output", x.out, expected.to(DTYPE))
    conv_error = _check_close(
        "decode-fused-conv-state", x.conv_state, reference_conv_state
    )
    state_error = _check_close("decode-fused-state", x.state, reference_state)
    return max(output_error, conv_error, state_error)


def _prepare(
    target: str,
) -> tuple[Callable[[], object], Callable[[], None], float, dict[str, int]]:
    if target == "prefill-conv1d":
        error = _check_conv()
        inputs = _conv_inputs(PREFILL_TOKENS, NUM_HEADS * HEAD_DIM)
        pristine_state = inputs.state.clone()
        return (
            lambda: _run_conv(inputs),
            lambda: inputs.state.copy_(pristine_state),
            error,
            {"production_calls_per_layer": 3},
        )
    if target == "prefill-kda":
        error = _check_kda()
        inputs = _kda_inputs(PREFILL_TOKENS, NUM_HEADS)
        return lambda: _run_kda(inputs), lambda: None, error, {}
    error = _check_decode()
    inputs = _decode_inputs()
    pristine_conv_state = inputs.conv_state.clone()
    pristine_state = inputs.state.clone()

    def reset_decode_state() -> None:
        inputs.conv_state.copy_(pristine_conv_state)
        inputs.state.copy_(pristine_state)

    return lambda: _run_decode(inputs), reset_decode_state, error, {"decode_tokens": 1}


def _validate_platform(target: str) -> None:
    from vllm.models.kimi_k3.nvidia.kda import (
        is_flashkda_supported,
        is_fused_kda_decode_supported,
    )
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    device_name = current_platform.get_device_name()
    if capability is None or tuple(capability) != (9, 0) or "H20" not in device_name:
        raise SystemExit(
            f"This benchmark requires NVIDIA H20 (SM90), got {device_name} "
            f"with capability {capability}"
        )
    if target == "prefill-kda" and not is_flashkda_supported(
        HEAD_DIM, DTYPE, LOWER_BOUND
    ):
        raise SystemExit("FlashKDA prefill is not supported by this vLLM build")
    if target == "decode-fused" and not is_fused_kda_decode_supported(
        NUM_HEADS, HEAD_DIM, CONV_WIDTH, 0, DTYPE, DTYPE
    ):
        raise SystemExit("The production fused KDA decode kernel is unavailable")


@torch.inference_mode()
def main() -> None:
    from vllm.platforms import current_platform

    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.accelerator.set_device_index(0)
    _validate_platform(args.target)
    torch.manual_seed(0)
    target, reset_state, error, extra = _prepare(args.target)
    for _ in range(args.warmup_iters):
        target()
    reset_state()
    torch.accelerator.synchronize()
    metadata = {
        "target": args.target,
        "nvtx_range": NVTX_RANGES[args.target],
        "device": current_platform.get_device_name(),
        "dtype": str(DTYPE),
        "batch_size": 1,
        "prefill_tokens": PREFILL_TOKENS,
        "num_heads": NUM_HEADS,
        "head_dim": HEAD_DIM,
        "conv_width": CONV_WIDTH,
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "correctness_relative_rmse": error,
        **extra,
    }
    print(json.dumps({**metadata, "status": "READY"}, sort_keys=True), flush=True)
    with torch.cuda.nvtx.range(NVTX_RANGES[args.target]):
        target()
    torch.accelerator.synchronize()
    print(json.dumps({**metadata, "status": "PASS"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
