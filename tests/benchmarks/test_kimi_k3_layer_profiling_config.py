# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

from benchmarks.kimi_k3_layer_profiling.benchmark import parse_args, run
from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)
from benchmarks.kimi_k3_layer_profiling.distributed import parallel_config_kwargs


ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"


def _smoke_data() -> dict:
    return load_yaml(SMOKE_CONFIG)


def test_smoke_dry_run_describes_first_block() -> None:
    result = dry_run(_smoke_data())

    assert result.config.model == (
        "benchmarks/kimi_k3_layer_profiling/model_config"
    )
    assert result.config.hidden_size == 7168
    assert result.config.dtype == "bfloat16"
    assert result.config.weight_format == "mxfp4-pack-quantized"
    assert result.config.logical_start_layer == 0
    assert result.config.input_shape == (1, 128, 7168)
    assert result.config.packed_shape == (128, 7168)
    assert result.config.expert_parallel_size == 8
    assert len(result.layers) == 12
    assert [layer.attention_type for layer in result.layers] == [
        "KDA",
        "KDA",
        "KDA",
        "MLA",
        "KDA",
        "KDA",
        "KDA",
        "MLA",
        "KDA",
        "KDA",
        "KDA",
        "MLA",
    ]
    assert result.layers[0].ffn_type == "dense"
    assert all(layer.ffn_type == "MoE" for layer in result.layers[1:])
    assert result.layers[0].attn_res_block_write
    assert not any(layer.attn_res_block_write for layer in result.layers[1:])


def test_dry_run_output_is_deterministic() -> None:
    data = _smoke_data()
    assert dry_run(data).to_json() == dry_run(data).to_json()


def test_cli_overrides_shape_without_mutating_yaml() -> None:
    data = _smoke_data()
    args = parse_args(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--dry-run",
            "--batch-size",
            "8",
            "--query-len",
            "256",
            "--context-len",
            "256",
        ]
    )
    overrides = {
        "batch_size": args.batch_size,
        "query_len": args.query_len,
        "context_len": args.context_len,
    }

    result = dry_run(apply_overrides(data, overrides))

    assert data["batch_size"] == 1
    assert result.config.input_shape == (8, 256, 7168)
    assert result.config.packed_shape == (2048, 7168)


def test_manifest_preview_has_stable_block_contract() -> None:
    manifest = dry_run(_smoke_data()).manifest()

    assert manifest["logical_layer_range"] == [0, 11]
    assert manifest["profiling_unit"] == "block"
    assert manifest["applies_model_output_attn_res"] is False
    assert manifest["block_output_contract"] == [
        "pending_hidden_states",
        "prefix_sum",
        "block_residual_bank",
    ]
    assert manifest["git_commit"] is None
    assert "latency_ms_by_rank" not in manifest


def test_cli_writes_deterministic_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    base_args = ["--config", str(SMOKE_CONFIG), "--dry-run"]

    assert run([*base_args, "--manifest-out", str(first)]) == 0
    assert run([*base_args, "--manifest-out", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()


def test_parallel_config_matches_benchmark_config() -> None:
    config = dry_run(_smoke_data()).config

    assert parallel_config_kwargs(config) == {
        "all2all_backend": "allgather_reducescatter",
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "tensor_parallel_size": 8,
    }


def test_distributed_smoke_cli_is_explicit() -> None:
    args = parse_args(
        ["--config", str(SMOKE_CONFIG), "--distributed-smoke"]
    )

    assert args.distributed_smoke


def test_distributed_smoke_rejects_manifest_output(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"

    with pytest.raises(SystemExit, match="not supported"):
        run(
            [
                "--config",
                str(SMOKE_CONFIG),
                "--distributed-smoke",
                "--manifest-out",
                str(manifest),
            ]
        )

    assert not manifest.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 0, "batch_size must be positive"),
        ("query_len", -1, "query_len must be positive"),
        ("context_len", 64, "context_len must be greater"),
        ("num_layers", 8, "requires exactly 12 layers"),
        ("decode_context_parallel_size", 3, "must divide"),
        ("tensor_parallel_size", 4, "must equal gpu_count"),
        ("profile_iters", 0, "profile_iters must be positive"),
        ("random_seed", -1, "random_seed must be non-negative"),
    ],
)
def test_invalid_config_fails_closed(field: str, value: object, message: str) -> None:
    data = _smoke_data()
    data[field] = value
    with pytest.raises(ValueError, match=message):
        dry_run(data)


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "hidden_size",
        "logical_start_layer",
        "num_attention_heads",
        "num_experts",
    ],
)
def test_model_structure_fields_cannot_be_overridden(field: str) -> None:
    data = _smoke_data()
    data[field] = 1

    with pytest.raises(ValueError, match="Unsupported config fields"):
        dry_run(data)


@pytest.mark.parametrize("num_layers", [1, 4, 8, 12])
def test_diagnostic_layer_counts(num_layers: int) -> None:
    data = _smoke_data()
    data["num_layers"] = num_layers
    data["diagnostic_partial_block"] = True

    result = dry_run(data)

    assert len(result.layers) == num_layers
