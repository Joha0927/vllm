# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.kimi_k3_layer_profiling.benchmark import parse_args, run
from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)
from benchmarks.kimi_k3_layer_profiling.production_profile import (
    production_engine_args_kwargs,
    production_profile_evidence,
    validate_production_profile_config,
)

ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"


def _config():
    return dry_run(load_yaml(SMOKE_CONFIG)).config


def test_smoke_config_describes_the_first_real_block() -> None:
    result = dry_run(load_yaml(SMOKE_CONFIG))

    assert result.config.hidden_size == 7168
    assert result.config.packed_shape == (128, 7168)
    assert result.config.expert_parallel_size == 8
    assert [layer.attention_type for layer in result.layers] == [
        "KDA",
        "KDA",
        "KDA",
        "MLA",
    ] * 3
    assert result.layers[0].ffn_type == "dense"
    assert all(layer.ffn_type == "MoE" for layer in result.layers[1:])
    assert [
        layer.logical_layer for layer in result.layers if layer.attn_res_block_write
    ] == [0]


def test_cli_shape_overrides_do_not_mutate_yaml() -> None:
    data = load_yaml(SMOKE_CONFIG)
    result = dry_run(
        apply_overrides(
            data,
            {"batch_size": 8, "query_len": 4096, "context_len": 4096},
        )
    )

    assert data["batch_size"] == 1
    assert result.config.num_scheduled_tokens == 32768
    assert result.config.packed_shape == (32768, 7168)


def test_production_profile_uses_the_original_model_and_engine_core() -> None:
    kwargs = production_engine_args_kwargs(_config())

    assert kwargs["hf_overrides"] == {"text_config": {"num_hidden_layers": 12}}
    assert kwargs["enable_layerwise_nvtx_tracing"] is False
    assert kwargs["kv_cache_memory_bytes"] == 4 * 1024**3
    assert kwargs["enforce_eager"] is True
    assert "model_class_overrides" not in kwargs
    assert "enable_prompt_embeds" not in kwargs
    assert "profiler_config" not in kwargs


def test_torch_profile_config_has_an_absolute_rank_output_dir(tmp_path: Path) -> None:
    config = replace(
        _config(), profile="torch", profile_output_dir=str(tmp_path / "traces")
    )

    kwargs = production_engine_args_kwargs(config)
    profiler_config = kwargs["profiler_config"]

    assert profiler_config.profiler == "torch"
    assert kwargs["enable_layerwise_nvtx_tracing"] is True
    assert Path(profiler_config.torch_profiler_dir).is_absolute()
    assert Path(profiler_config.torch_profiler_dir).is_dir()
    assert profiler_config.torch_profiler_record_shapes is True
    assert profiler_config.torch_profiler_with_memory is False
    assert profiler_config.torch_profiler_with_stack is False


def test_production_evidence_excludes_custom_execution_paths() -> None:
    evidence = production_profile_evidence(_config())

    assert evidence["execution_path"] == "LLM/EngineCore/production_model"
    assert evidence["expected_layer_range"] == [0, 11]
    assert evidence["uses_custom_block_wrapper"] is False
    assert evidence["uses_manual_kv_cache_init"] is False
    assert evidence["model_class_override"] is False


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (lambda c: replace(c, phase="decode"), "supports prefill only"),
        (lambda c: replace(c, context_len=256), "context_len=query_len"),
        (lambda c: replace(c, execution_mode="cudagraph"), "execution_mode=eager"),
        (lambda c: replace(c, num_layers=8), "formal 12-layer block"),
        (
            lambda c: replace(c, profile="torch", profile_output_dir=None),
            "profile_output_dir is required",
        ),
        (
            lambda c: replace(c, profile="none", profile_output_dir="traces"),
            "profile_output_dir requires",
        ),
    ],
)
def test_production_profile_rejects_unsupported_modes(config, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_production_profile_config(config(_config()))


def test_cli_exposes_production_profile() -> None:
    args = parse_args(["--config", str(SMOKE_CONFIG), "--production-profile"])

    assert args.production_profile


def test_cli_writes_deterministic_dry_run_manifest(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    base_args = ["--config", str(SMOKE_CONFIG), "--dry-run"]

    assert run([*base_args, "--manifest-out", str(first)]) == 0
    assert run([*base_args, "--manifest-out", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()


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
    ],
)
def test_invalid_config_fails_closed(field: str, value: object, message: str) -> None:
    data = load_yaml(SMOKE_CONFIG)
    data[field] = value
    with pytest.raises(ValueError, match=message):
        dry_run(data)


@pytest.mark.parametrize(
    "field", ["model", "hidden_size", "logical_start_layer", "num_experts"]
)
def test_model_structure_fields_cannot_be_overridden(field: str) -> None:
    data = load_yaml(SMOKE_CONFIG)
    data[field] = 1

    with pytest.raises(ValueError, match="Unsupported config fields"):
        dry_run(data)
