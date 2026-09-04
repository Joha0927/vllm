# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-H20 NCU targets for Kimi-K3 MLA and AttnRes production kernels."""

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DTYPE = torch.bfloat16
PREFILL_TOKENS = 16384
NUM_HEADS = 96
HIDDEN_SIZE = 7168
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
KV_CACHE_ENTRY = KV_LORA_RANK + QK_ROPE_HEAD_DIM
KV_BLOCK_SIZE = 768
EPS = 1e-5
TARGETS = (
    "mla-kv-insert",
    "attn-res-prefill",
    "attn-res-decode",
    "attn-res-block-write",
    "mla-fa-prefill",
)
NVTX_RANGES = {target: f"kimi_k3_operator_ncu/{target}" for target in TARGETS}


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
        raise ValueError("profile_iters must be 1 because cache state is mutable")
    return args


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return (error / scale).item()


def _check_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    limit: float = 0.03,
) -> float:
    error = _relative_rmse(actual, expected)
    if not torch.isfinite(actual).all() or error >= limit:
        raise AssertionError(f"{name} relative RMSE {error:.6f} >= {limit}")
    return error


@dataclass
class PreparedTarget:
    run: Callable[[], object]
    reset: Callable[[], None]
    correctness_error: float
    shape: dict[str, object]
    production_entry: str


@dataclass
class MLAKVInsertInputs:
    q: torch.Tensor
    k_nope: torch.Tensor
    k_pe: torch.Tensor
    kv_c: torch.Tensor
    cache: torch.Tensor
    slots: torch.Tensor


def _mla_kv_insert_inputs(tokens: int, heads: int) -> MLAKVInsertInputs:
    num_blocks = (tokens + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    kv_nope_v = torch.randn(
        tokens,
        heads,
        QK_NOPE_HEAD_DIM + V_HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    k_nope, _ = kv_nope_v.split((QK_NOPE_HEAD_DIM, V_HEAD_DIM), dim=-1)
    fused_qkv_lora = torch.randn(
        tokens,
        Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    k_pe = fused_qkv_lora[:, -QK_ROPE_HEAD_DIM:].unsqueeze(1)
    return MLAKVInsertInputs(
        q=torch.randn(tokens, heads, QK_HEAD_DIM, device="cuda", dtype=DTYPE),
        k_nope=k_nope,
        k_pe=k_pe,
        kv_c=torch.randn(tokens, KV_LORA_RANK, device="cuda", dtype=DTYPE),
        cache=torch.full(
            (num_blocks, KV_BLOCK_SIZE, KV_CACHE_ENTRY),
            -7,
            device="cuda",
            dtype=DTYPE,
        ),
        slots=torch.arange(tokens, device="cuda", dtype=torch.int64),
    )


def _run_mla_kv_insert(x: MLAKVInsertInputs) -> torch.Tensor:
    from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
        fused_mla_key_concat_kv_cache_insert,
    )

    return fused_mla_key_concat_kv_cache_insert(
        x.q, x.k_nope, x.k_pe, x.kv_c, x.cache, x.slots
    )


def _check_mla_kv_insert() -> float:
    x = _mla_kv_insert_inputs(8, 2)
    x.slots.copy_(torch.tensor([0, 1, -1, 3, 4, -1, 6, 7], device="cuda"))
    initial_cache = x.cache.clone()
    output = _run_mla_kv_insert(x)
    k_pe = x.k_pe.reshape(x.k_pe.shape[0], QK_ROPE_HEAD_DIM)
    expected_output = torch.cat((x.k_nope, k_pe[:, None, :].expand(-1, 2, -1)), dim=-1)
    expected_cache = initial_cache.clone().view(-1, KV_CACHE_ENTRY)
    latent = torch.cat((x.kv_c, k_pe), dim=-1)
    valid = x.slots >= 0
    expected_cache[x.slots[valid]] = latent[valid]
    if not torch.equal(output, expected_output):
        raise AssertionError("MLA KV insert produced an incorrect concatenated key")
    if not torch.equal(x.cache.view(-1, KV_CACHE_ENTRY), expected_cache):
        raise AssertionError("MLA KV insert updated incorrect cache slots")
    return 0.0


def _prepare_mla_kv_insert() -> PreparedTarget:
    error = _check_mla_kv_insert()
    x = _mla_kv_insert_inputs(PREFILL_TOKENS, NUM_HEADS)
    initial_cache = x.cache.clone()
    return PreparedTarget(
        run=lambda: _run_mla_kv_insert(x),
        reset=lambda: x.cache.copy_(initial_cache),
        correctness_error=error,
        shape={
            "tokens": PREFILL_TOKENS,
            "heads": NUM_HEADS,
            "qk_head_dim": QK_HEAD_DIM,
            "kv_lora_rank": KV_LORA_RANK,
            "kv_block_size": KV_BLOCK_SIZE,
            "rope": False,
            "k_nope_stride": list(x.k_nope.stride()),
            "k_pe_stride": list(x.k_pe.stride()),
        },
        production_entry="fused_mla_key_concat_kv_cache_insert",
    )


@dataclass
class AttnResInputs:
    prefix: torch.Tensor
    delta: torch.Tensor
    blocks: torch.Tensor
    norm_weight: torch.Tensor
    qk_weight: torch.Tensor
    output_norm_weight: torch.Tensor
    num_blocks: int
    block_write_idx: int


def _attn_res_inputs(tokens: int, *, write_block: bool) -> AttnResInputs:
    num_blocks = 0 if write_block else 1
    return AttnResInputs(
        prefix=torch.randn(tokens, HIDDEN_SIZE, device="cuda", dtype=DTYPE),
        delta=torch.randn(tokens, HIDDEN_SIZE, device="cuda", dtype=DTYPE),
        blocks=torch.randn(tokens, 1, HIDDEN_SIZE, device="cuda", dtype=DTYPE),
        norm_weight=1 + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda", dtype=DTYPE),
        qk_weight=torch.randn(HIDDEN_SIZE, device="cuda", dtype=DTYPE)
        / HIDDEN_SIZE**0.5,
        output_norm_weight=1
        + 0.1 * torch.randn(HIDDEN_SIZE, device="cuda", dtype=DTYPE),
        num_blocks=num_blocks,
        block_write_idx=0 if write_block else -1,
    )


def _run_attn_res(x: AttnResInputs) -> torch.Tensor:
    from vllm.models.kimi_k3.nvidia.ops.attn_res import attn_res

    return attn_res(
        x.prefix,
        x.delta,
        x.blocks,
        x.norm_weight,
        x.qk_weight,
        x.output_norm_weight,
        x.num_blocks,
        x.block_write_idx,
        EPS,
        EPS,
    )


def _attn_res_reference(x: AttnResInputs) -> tuple[torch.Tensor, torch.Tensor]:
    updated_prefix = (x.prefix + x.delta).to(DTYPE)
    values = torch.cat(
        (x.blocks[:, : x.num_blocks], updated_prefix.unsqueeze(1)), dim=1
    )
    keys = F.rms_norm(values, (HIDDEN_SIZE,), x.norm_weight, EPS)
    probabilities = (keys @ x.qk_weight).softmax(dim=-1)
    output = torch.matmul(probabilities.unsqueeze(1), values).squeeze(1)
    output = F.rms_norm(output, (HIDDEN_SIZE,), x.output_norm_weight, EPS).to(DTYPE)
    return output, updated_prefix


def _check_attn_res(tokens: int, *, write_block: bool) -> float:
    x = _attn_res_inputs(tokens, write_block=write_block)
    initial_blocks = x.blocks.clone()
    expected, expected_prefix = _attn_res_reference(x)
    actual = _run_attn_res(x)
    output_error = _check_close("attn-res-output", actual, expected)
    if not torch.equal(x.prefix, expected_prefix):
        raise AssertionError("AttnRes prefix update is incorrect")
    if write_block:
        initial_blocks[:, x.block_write_idx].copy_(expected_prefix)
    if not torch.equal(x.blocks, initial_blocks):
        raise AssertionError("AttnRes block update is incorrect")
    return output_error


def _prepare_attn_res(target: str) -> PreparedTarget:
    write_block = target == "attn-res-block-write"
    tokens = 1 if target == "attn-res-decode" else PREFILL_TOKENS
    correctness_tokens = 1 if target == "attn-res-decode" else 320
    error = _check_attn_res(correctness_tokens, write_block=write_block)
    x = _attn_res_inputs(tokens, write_block=write_block)
    initial_prefix = x.prefix.clone()
    initial_blocks = x.blocks.clone()

    def reset() -> None:
        x.prefix.copy_(initial_prefix)
        x.blocks.copy_(initial_blocks)

    return PreparedTarget(
        run=lambda: _run_attn_res(x),
        reset=reset,
        correctness_error=error,
        shape={
            "tokens": tokens,
            "hidden_size": HIDDEN_SIZE,
            "num_blocks": x.num_blocks,
            "block_write_idx": x.block_write_idx,
            "has_delta": True,
            "apply_output_norm": True,
            "backend": "triton",
        },
        production_entry="vllm.models.kimi_k3.nvidia.ops.attn_res.attn_res",
    )


@dataclass
class MLAFAPrefillInputs:
    backend: object
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor


def _mla_fa_inputs(tokens: int, heads: int) -> MLAFAPrefillInputs:
    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonPrefillMetadata,
    )
    from vllm.v1.attention.backends.mla.prefill.flash_attn import (
        FlashAttnPrefillBackend,
    )

    if not FlashAttnPrefillBackend.is_available():
        raise SystemExit("FlashAttention MLA prefill is unavailable")
    backend = FlashAttnPrefillBackend(
        num_heads=heads,
        scale=QK_HEAD_DIM**-0.5,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        vllm_config=VllmConfig(),
    )
    starts = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    backend.prepare_metadata(
        MLACommonPrefillMetadata(
            block_table=torch.empty(1, 0, device="cuda", dtype=torch.int32),
            query_start_loc=starts,
            max_query_len=tokens,
        )
    )
    kv_nope_v = torch.randn(
        tokens,
        heads,
        QK_NOPE_HEAD_DIM + V_HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    _, v = kv_nope_v.split((QK_NOPE_HEAD_DIM, V_HEAD_DIM), dim=-1)
    return MLAFAPrefillInputs(
        backend=backend,
        q=torch.randn(tokens, heads, QK_HEAD_DIM, device="cuda", dtype=DTYPE),
        k=torch.randn(tokens, heads, QK_HEAD_DIM, device="cuda", dtype=DTYPE),
        v=v,
        out=torch.empty(tokens, heads, V_HEAD_DIM, device="cuda", dtype=DTYPE),
    )


def _run_mla_fa(x: MLAFAPrefillInputs) -> torch.Tensor:
    return x.backend.run_prefill_new_tokens(  # type: ignore[attr-defined,no-any-return]
        q=x.q,
        k=x.k,
        v=x.v,
        return_softmax_lse=False,
        out=x.out,
    )


def _check_mla_fa() -> float:
    x = _mla_fa_inputs(64, 2)
    actual = _run_mla_fa(x)
    q = x.q.transpose(0, 1).float()
    k = x.k.transpose(0, 1).float()
    v = x.v.transpose(0, 1).float()
    expected = F.scaled_dot_product_attention(
        q, k, v, is_causal=True, scale=QK_HEAD_DIM**-0.5
    ).transpose(0, 1)
    return _check_close("mla-fa-prefill", actual, expected.to(DTYPE))


def _prepare_mla_fa() -> PreparedTarget:
    error = _check_mla_fa()
    x = _mla_fa_inputs(PREFILL_TOKENS, NUM_HEADS)
    backend = x.backend
    return PreparedTarget(
        run=lambda: _run_mla_fa(x),
        reset=lambda: None,
        correctness_error=error,
        shape={
            "batch_size": 1,
            "q_tokens": PREFILL_TOKENS,
            "kv_tokens": PREFILL_TOKENS,
            "heads": NUM_HEADS,
            "qk_head_dim": QK_HEAD_DIM,
            "v_head_dim": V_HEAD_DIM,
            "causal": True,
            "v_stride": list(x.v.stride()),
            "fa_version": backend.vllm_flash_attn_version,  # type: ignore[attr-defined]
        },
        production_entry="FlashAttnPrefillBackend.run_prefill_new_tokens",
    )


def _prepare(target: str) -> PreparedTarget:
    if target == "mla-kv-insert":
        return _prepare_mla_kv_insert()
    if target.startswith("attn-res-"):
        return _prepare_attn_res(target)
    return _prepare_mla_fa()


def _validate_platform() -> str:
    from vllm.platforms import current_platform

    capability = current_platform.get_device_capability()
    device_name = current_platform.get_device_name()
    if capability is None or tuple(capability) != (9, 0) or "H20" not in device_name:
        raise SystemExit(
            f"This benchmark requires NVIDIA H20 (SM90), got {device_name} "
            f"with capability {capability}"
        )
    return device_name


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.accelerator.set_device_index(0)
    device_name = _validate_platform()
    torch.manual_seed(0)
    prepared = _prepare(args.target)
    warmup_output = None
    for _ in range(args.warmup_iters):
        warmup_output = prepared.run()
    if (
        isinstance(warmup_output, torch.Tensor)
        and not torch.isfinite(warmup_output).all()
    ):
        raise AssertionError("production-shape warmup produced non-finite output")
    prepared.reset()
    torch.accelerator.synchronize()
    metadata = {
        "target": args.target,
        "nvtx_range": NVTX_RANGES[args.target],
        "production_entry": prepared.production_entry,
        "device": device_name,
        "dtype": str(DTYPE),
        "shape": prepared.shape,
        "warmup_iters": args.warmup_iters,
        "profile_iters": args.profile_iters,
        "correctness_relative_rmse": prepared.correctness_error,
    }
    print(json.dumps({**metadata, "status": "READY"}, sort_keys=True), flush=True)
    with torch.cuda.nvtx.range(NVTX_RANGES[args.target]):
        prepared.run()
    torch.accelerator.synchronize()
    print(json.dumps({**metadata, "status": "PASS"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
