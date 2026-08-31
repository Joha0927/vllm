# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kimi_k3_layer_profiling.benchmark import parse_args, run
from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)
from benchmarks.kimi_k3_layer_profiling.distributed import parallel_config_kwargs
from benchmarks.kimi_k3_layer_profiling.forward_smoke import (
    align_cache_block_size,
    ensure_tracked_worktree_clean,
    expected_all2all_manager_class,
    forward_engine_args_kwargs,
    kv_cache_spec_evidence,
    validate_forward_smoke_config,
)
from benchmarks.kimi_k3_layer_profiling.model_construction import (
    MODEL_CLASS_OVERRIDES,
    engine_args_kwargs,
    validate_model_construction_config,
)
from benchmarks.kimi_k3_layer_profiling.production_profile import (
    production_engine_args_kwargs,
    production_profile_evidence,
    validate_production_profile_config,
)

ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"


def _smoke_data() -> dict:
    return load_yaml(SMOKE_CONFIG)


def test_smoke_dry_run_describes_first_block() -> None:
    result = dry_run(_smoke_data())

    assert result.config.model == ("benchmarks/kimi_k3_layer_profiling/model_config")
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
    args = parse_args(["--config", str(SMOKE_CONFIG), "--distributed-smoke"])

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


def test_model_construction_uses_production_engine_config() -> None:
    config = dry_run(_smoke_data()).config

    assert engine_args_kwargs(config) == {
        "all2all_backend": "allgather_reducescatter",
        "data_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "disable_log_stats": True,
        "dtype": "bfloat16",
        "enable_expert_parallel": True,
        "enforce_eager": True,
        "language_model_only": True,
        "load_format": "dummy",
        "max_model_len": 128,
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "model": "benchmarks/kimi_k3_layer_profiling/model_config",
        "model_class_overrides": MODEL_CLASS_OVERRIDES,
        "seed": 0,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": 8,
    }


def test_model_construction_smoke_cli_is_explicit() -> None:
    args = parse_args(["--config", str(SMOKE_CONFIG), "--model-construction-smoke"])

    assert args.model_construction_smoke


def test_model_construction_rejects_diagnostic_layer_count() -> None:
    data = _smoke_data()
    data["num_layers"] = 1
    data["diagnostic_partial_block"] = True

    with pytest.raises(ValueError, match="formal 12-layer block"):
        validate_model_construction_config(dry_run(data).config)


def test_forward_smoke_uses_prompt_embeds_with_production_engine_config() -> None:
    config = dry_run(_smoke_data()).config
    expected = engine_args_kwargs(config)
    expected["enable_prompt_embeds"] = True

    assert forward_engine_args_kwargs(config) == expected


def test_forward_smoke_cli_is_explicit() -> None:
    args = parse_args(["--config", str(SMOKE_CONFIG), "--forward-smoke"])

    assert args.forward_smoke


def test_production_profile_uses_engine_core_without_model_override() -> None:
    config = dry_run(_smoke_data()).config

    kwargs = production_engine_args_kwargs(config)

    assert kwargs["hf_overrides"] == {"text_config": {"num_hidden_layers": 12}}
    assert kwargs["enable_layerwise_nvtx_tracing"] is True
    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["kv_cache_memory_bytes"] == 4 * 1024**3
    assert "model_class_overrides" not in kwargs
    assert "enable_prompt_embeds" not in kwargs


def test_production_profile_evidence_rejects_custom_execution_paths() -> None:
    evidence = production_profile_evidence(dry_run(_smoke_data()).config)

    assert evidence["execution_path"] == "LLM/EngineCore/production_model"
    assert evidence["expected_layer_range"] == [0, 11]
    assert evidence["uses_custom_block_wrapper"] is False
    assert evidence["uses_manual_kv_cache_init"] is False
    assert evidence["model_class_override"] is False


def test_production_profile_cli_is_explicit() -> None:
    args = parse_args(["--config", str(SMOKE_CONFIG), "--production-profile"])

    assert args.production_profile


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"phase": "decode"}, "supports prefill only"),
        ({"context_len": 256}, "context_len=query_len"),
        ({"execution_mode": "cudagraph"}, "requires execution_mode=eager"),
        (
            {"num_layers": 8, "diagnostic_partial_block": True},
            "formal 12-layer block",
        ),
        ({"profile": "torch"}, "profile=none or profile=cuda"),
    ],
)
def test_production_profile_rejects_nonformal_modes(
    overrides: dict[str, object], message: str
) -> None:
    data = _smoke_data()
    data.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_production_profile_config(dry_run(data).config)


def test_forward_smoke_requires_clean_tracked_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results: list[subprocess.CompletedProcess[str]] = [
        subprocess.CompletedProcess(["git", "diff"], 0),
        subprocess.CompletedProcess(["git", "diff", "--cached"], 1),
    ]
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: results.pop(0))

    with pytest.raises(RuntimeError, match="Git index has uncommitted changes"):
        ensure_tracked_worktree_clean()


@pytest.mark.parametrize("backend", ("allgather_reducescatter", "naive"))
def test_forward_smoke_validates_ag_rs_backend(backend: str) -> None:
    assert expected_all2all_manager_class(backend) == "AgRsAll2AllManager"


def test_forward_smoke_rejects_unknown_backend_evidence() -> None:
    with pytest.raises(ValueError, match="cannot validate all2all backend"):
        expected_all2all_manager_class("unknown")


def test_forward_smoke_reports_raw_cache_specs_in_layer_order() -> None:
    mamba_spec = type("MambaSpec", (), {})()
    mamba_spec.block_size = 128
    mamba_spec.num_heads = 1
    mamba_spec.num_states = 1
    mamba_spec.page_size_bytes = 4096
    mamba_spec.page_size_padded = None
    mamba_spec.state_content_size_bytes = 4096
    mamba_spec.tokens_per_state = -1
    mla_spec = type("MLAAttentionSpec", (), {})()
    mla_spec.block_size = 128
    mla_spec.num_heads = 1
    mla_spec.num_states = 128
    mla_spec.page_size_bytes = 8192
    mla_spec.state_content_size_bytes = 64
    mla_spec.tokens_per_state = 1

    evidence = kv_cache_spec_evidence(
        {
            "model.layers.3.self_attn": mla_spec,
            "model.layers.0.self_attn": mamba_spec,
        }
    )

    assert [record["layer_index"] for record in evidence] == [0, 3]
    assert [record["cache_kind"] for record in evidence] == ["KDA", "MLA"]
    assert [record["page_size_bytes"] for record in evidence] == [4096, 8192]


def test_forward_smoke_uses_production_cache_alignment_api(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cache_config = SimpleNamespace(
        block_size=16,
        mamba_page_size_padded=None,
    )
    vllm_config = SimpleNamespace(cache_config=cache_config)

    class FakePlatform:
        @classmethod
        def update_block_size_for_backend(cls, config) -> None:
            assert config is vllm_config
            config.cache_config.block_size = 768
            config.cache_config.mamba_page_size_padded = 884736

    align_cache_block_size(vllm_config, 3, FakePlatform)
    evidence = json.loads(capsys.readouterr().out)

    assert evidence["rank"] == 3
    assert evidence["stage"] == "cache_backend_alignment"
    assert evidence["before"] == {
        "block_size": 16,
        "mamba_page_size_padded": None,
    }
    assert evidence["after"] == {
        "block_size": 768,
        "mamba_page_size_padded": 884736,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"execution_mode": "cudagraph"}, "execution_mode=eager"),
        ({"phase": "decode"}, "supports prefill only"),
        ({"context_len": 256}, "context_len=query_len"),
        ({"cache_mode": "prefix"}, "cache_mode=none"),
    ],
)
def test_forward_smoke_rejects_unsupported_modes(
    overrides: dict[str, object], message: str
) -> None:
    data = _smoke_data()
    data.update(overrides)

    with pytest.raises(ValueError, match=message):
        validate_forward_smoke_config(dry_run(data).config)


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
