# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.kimi_k3_layer_profiling.benchmark import parse_args, run
from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)
from benchmarks.kimi_k3_layer_profiling.production_profile import (
    _validate_output_token_counts,
    production_engine_args_kwargs,
    production_profile_evidence,
    validate_production_profile_config,
)

ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"
PREFILL_DECODE_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_with_stack.yaml"
)
PREFILL_DECODE_TP2_DP4_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_tp2_dp4_with_stack.yaml"
)
PREFILL_DECODE_TP2_DP4_FLASHINFER_ONE_SIDED_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_tp2_dp4_flashinfer_one_sided_with_stack.yaml"
)
NCU_TP2_DP4_AG_RS_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/ncu_tp2_dp4_ep8_ag_rs.yaml"
)
NCU_TP2_DP4_FLASHINFER_ONE_SIDED_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "ncu_tp2_dp4_ep8_flashinfer_one_sided.yaml"
)


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


def test_performance_path_defaults_are_explicit() -> None:
    data = {
        "workload": "full_prefill",
        "batch_size": 1,
        "history_len": 0,
        "query_len": 128,
    }

    config = dry_run(data).config

    assert config.moe_backend == "auto"
    assert config.linear_backend == "auto"
    assert config.attention_backend == "auto"
    assert config.kda_prefill_backend == "auto"
    assert config.mla_prefill_backend == "auto"
    assert config.kv_cache_dtype == "auto"
    assert config.kv_cache_memory_bytes == 4 * 1024**3
    assert config.expert_placement_strategy == "linear"
    assert config.enable_dbo is False
    assert config.shard_sp_shared_expert is False
    assert config.profiler_with_stack is False
    assert config.enable_layerwise_nvtx_tracing is False
    assert config.layerwise_nvtx_tracing_enabled is False
    assert config.local_batch_size == 1


def test_required_workload_fields_have_no_defaults() -> None:
    with pytest.raises(ValueError, match="Missing required fields: history_len"):
        dry_run({"workload": "full_prefill", "batch_size": 1, "query_len": 128})


def test_cli_shape_overrides_do_not_mutate_yaml() -> None:
    data = load_yaml(SMOKE_CONFIG)
    result = dry_run(
        apply_overrides(
            data,
            {"batch_size": 8, "query_len": 4096},
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
    assert kwargs["moe_backend"] == "auto"
    assert kwargs["linear_backend"] == "auto"
    assert kwargs["attention_backend"] is None
    assert kwargs["kda_prefill_backend"] == "auto"
    assert kwargs["kv_cache_dtype"] == "auto"
    assert kwargs["expert_placement_strategy"] == "linear"
    assert kwargs["enable_dbo"] is False
    assert kwargs["enforce_eager"] is True
    assert kwargs["long_prefill_token_threshold"] == 0
    assert kwargs["max_model_len"] == 129
    assert kwargs["max_num_batched_tokens"] == 129
    assert "model_class_overrides" not in kwargs
    assert "enable_prompt_embeds" not in kwargs
    assert "profiler_config" not in kwargs


def test_backend_overrides_reach_production_engine_args() -> None:
    config = replace(
        _config(),
        attention_backend="flash_attn",
        enable_dbo=True,
        expert_placement_strategy="round_robin",
        kda_prefill_backend="triton",
        kv_cache_dtype="bfloat16",
        kv_cache_memory_bytes=8 * 1024**3,
        linear_backend="torch",
        mla_prefill_backend="flash_attn",
        moe_backend="deep_gemm_mega_moe",
    )

    kwargs = production_engine_args_kwargs(config)

    assert kwargs["attention_backend"] == "flash_attn"
    assert kwargs["attention_config"] == {"mla_prefill_backend": "flash_attn"}
    assert kwargs["enable_dbo"] is True
    assert kwargs["expert_placement_strategy"] == "round_robin"
    assert kwargs["kda_prefill_backend"] == "triton"
    assert kwargs["kv_cache_dtype"] == "bfloat16"
    assert kwargs["kv_cache_memory_bytes"] == 8 * 1024**3
    assert kwargs["linear_backend"] == "torch"
    assert kwargs["moe_backend"] == "deep_gemm_mega_moe"


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


def test_torch_profile_with_stack_reaches_profiler_config(tmp_path: Path) -> None:
    config = replace(
        dry_run(load_yaml(PREFILL_DECODE_WITH_STACK_CONFIG)).config,
        profile="torch",
        profile_output_dir=str(tmp_path / "traces"),
    )

    kwargs = production_engine_args_kwargs(config)

    assert kwargs["profiler_config"].torch_profiler_with_stack is True
    assert production_profile_evidence(config)["profiler_with_stack"] is True
    assert (
        dry_run(load_yaml(PREFILL_DECODE_WITH_STACK_CONFIG)).manifest()[
            "profiler_with_stack"
        ]
        is True
    )


@pytest.mark.parametrize(
    ("path", "all2all_backend"),
    [
        (NCU_TP2_DP4_AG_RS_CONFIG, "allgather_reducescatter"),
        (
            NCU_TP2_DP4_FLASHINFER_ONE_SIDED_CONFIG,
            "flashinfer_nvlink_one_sided",
        ),
    ],
)
def test_ncu_configs_enable_nvtx_without_torch_profiler(
    path: Path, all2all_backend: str
) -> None:
    config = dry_run(load_yaml(path)).config

    kwargs = production_engine_args_kwargs(config)
    evidence = production_profile_evidence(config)

    assert config.tensor_parallel_size == 2
    assert config.data_parallel_size == 4
    assert config.expert_parallel_size == 8
    assert config.all2all_backend == all2all_backend
    assert config.profile == "none"
    assert config.enable_layerwise_nvtx_tracing is True
    assert kwargs["enable_layerwise_nvtx_tracing"] is True
    assert "profiler_config" not in kwargs
    assert evidence["layerwise_profiler_scopes"] is True


@pytest.mark.parametrize(
    ("path", "tp", "dp", "all2all_backend", "with_stack"),
    [
        (PREFILL_DECODE_WITH_STACK_CONFIG, 1, 8, "allgather_reducescatter", True),
        (
            PREFILL_DECODE_TP2_DP4_WITH_STACK_CONFIG,
            2,
            4,
            "allgather_reducescatter",
            True,
        ),
        (
            PREFILL_DECODE_TP2_DP4_FLASHINFER_ONE_SIDED_WITH_STACK_CONFIG,
            2,
            4,
            "flashinfer_nvlink_one_sided",
            True,
        ),
    ],
)
def test_formal_matrix_configs(
    path: Path,
    tp: int,
    dp: int,
    all2all_backend: str,
    with_stack: bool,
) -> None:
    config = dry_run(load_yaml(path)).config

    assert config.batch_size == 8
    assert config.prompt_len == 16384
    assert config.max_tokens == 2
    assert config.tensor_parallel_size == tp
    assert config.data_parallel_size == dp
    assert config.expert_parallel_size == 8
    assert config.all2all_backend == all2all_backend
    assert config.profiler_with_stack is with_stack


def test_production_evidence_records_current_execution_path() -> None:
    evidence = production_profile_evidence(_config())

    assert evidence["execution_path"] == "LLM/EngineCore/production_model"
    assert evidence["expected_layer_range"] == [0, 11]
    assert evidence["requested_moe_backend"] == "auto"
    assert evidence["requested_kda_prefill_backend"] == "auto"
    assert evidence["local_batch_size"] == 1


def test_prefill_decode_derives_one_decode_execution() -> None:
    config = dry_run(load_yaml(PREFILL_DECODE_WITH_STACK_CONFIG)).config

    assert config.workload == "prefill_decode"
    assert config.prompt_len == 16384
    assert config.query_len == 1
    assert config.max_tokens == 2
    assert config.max_model_len == 16386
    assert config.prefill_tokens == 8 * 16384
    assert config.num_scheduled_tokens == 8

    kwargs = production_engine_args_kwargs(config)
    assert kwargs["max_model_len"] == 16386
    assert kwargs["max_num_batched_tokens"] == 16386
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["data_parallel_size"] == 8
    assert kwargs["distributed_executor_backend"] == "external_launcher"

    evidence = production_profile_evidence(config)
    assert evidence["workload"] == "prefill_decode"
    assert evidence["max_tokens"] == 2


def test_prefill_decode_rejects_multiple_profile_iterations() -> None:
    config = dry_run(load_yaml(PREFILL_DECODE_WITH_STACK_CONFIG)).config

    with pytest.raises(ValueError, match="prefill_decode requires profile_iters=1"):
        validate_production_profile_config(replace(config, profile_iters=2))


def test_prefill_decode_tp2_dp4_uses_two_local_requests() -> None:
    config = dry_run(load_yaml(PREFILL_DECODE_TP2_DP4_WITH_STACK_CONFIG)).config

    kwargs = production_engine_args_kwargs(config)
    assert config.expert_parallel_size == 8
    assert config.local_batch_size == 2
    assert kwargs["max_num_seqs"] == 2
    assert kwargs["max_num_batched_tokens"] == 2 * 16384
    assert kwargs["distributed_executor_backend"] == "external_launcher"


def test_output_token_evidence_requires_exactly_two_tokens() -> None:
    outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[11, 12])]),
        SimpleNamespace(outputs=[SimpleNamespace(token_ids=[21, 22])]),
    ]

    assert _validate_output_token_counts(outputs, 2, 2) == [2, 2]

    with pytest.raises(RuntimeError, match="produced 2 output tokens, expected 1"):
        _validate_output_token_counts(outputs, 2, 1)


@pytest.mark.parametrize(
    ("config", "message"),
    [
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


def test_cli_exposes_performance_path_overrides() -> None:
    args = parse_args(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--dry-run",
            "--moe-backend",
            "deep_gemm_mega_moe",
            "--linear-backend",
            "torch",
            "--kda-prefill-backend",
            "flashkda",
            "--mla-prefill-backend",
            "flashinfer",
            "--shard-sp-shared-expert",
            "--profiler-with-stack",
        ]
    )

    assert args.moe_backend == "deep_gemm_mega_moe"
    assert args.linear_backend == "torch"
    assert args.kda_prefill_backend == "flashkda"
    assert args.mla_prefill_backend == "flashinfer"
    assert args.shard_sp_shared_expert is True
    assert args.profiler_with_stack is True

    disabled = parse_args(
        [
            "--config",
            str(PREFILL_DECODE_WITH_STACK_CONFIG),
            "--dry-run",
            "--no-profiler-with-stack",
        ]
    )
    assert disabled.profiler_with_stack is False


def test_cli_exposes_workload_overrides() -> None:
    args = parse_args(
        [
            "--config",
            str(SMOKE_CONFIG),
            "--dry-run",
            "--workload",
            "prefill_decode",
            "--history-len",
            "16384",
            "--query-len",
            "1",
        ]
    )

    assert args.workload == "prefill_decode"
    assert args.history_len == 16384


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
        ("history_len", -1, "history_len must be non-negative"),
        ("decode_context_parallel_size", 3, "must divide"),
        ("tensor_parallel_size", 4, "must equal gpu_count"),
        ("profile_iters", 0, "profile_iters must be positive"),
        ("kv_cache_memory_bytes", 0, "kv_cache_memory_bytes must be positive"),
        ("kda_prefill_backend", "invalid", "kda_prefill_backend must be one"),
        ("mla_prefill_backend", "invalid", "mla_prefill_backend must be one"),
    ],
)
def test_invalid_config_fails_closed(field: str, value: object, message: str) -> None:
    data = load_yaml(SMOKE_CONFIG)
    data[field] = value
    with pytest.raises(ValueError, match=message):
        dry_run(data)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (
            {
                "workload": "full_prefill",
                "batch_size": 1,
                "history_len": 1,
                "query_len": 128,
            },
            "full_prefill requires history_len=0",
        ),
        (
            {
                "workload": "prefill_decode",
                "batch_size": 1,
                "history_len": 128,
                "query_len": 2,
            },
            "prefill_decode requires history_len>0 and query_len=1",
        ),
    ],
)
def test_workload_shape_contracts_fail_closed(
    data: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        dry_run(data)


@pytest.mark.parametrize(
    "field",
    ["model", "hidden_size", "logical_start_layer", "num_experts", "num_layers"],
)
def test_model_structure_fields_cannot_be_overridden(field: str) -> None:
    data = load_yaml(SMOKE_CONFIG)
    data[field] = 1

    with pytest.raises(ValueError, match="Unsupported config fields"):
        dry_run(data)


def test_boolean_fields_reject_string_values() -> None:
    data = load_yaml(SMOKE_CONFIG)
    data["enable_dbo"] = "false"

    with pytest.raises(ValueError, match="enable_dbo must be a boolean"):
        dry_run(data)


def test_profiler_with_stack_rejects_string_values() -> None:
    data = load_yaml(SMOKE_CONFIG)
    data["profiler_with_stack"] = "false"

    with pytest.raises(ValueError, match="profiler_with_stack must be a boolean"):
        dry_run(data)


def test_layerwise_nvtx_rejects_string_values() -> None:
    data = load_yaml(SMOKE_CONFIG)
    data["enable_layerwise_nvtx_tracing"] = "false"

    with pytest.raises(
        ValueError, match="enable_layerwise_nvtx_tracing must be a boolean"
    ):
        dry_run(data)


def test_shared_expert_sharding_requires_sequence_parallel_execution() -> None:
    data = load_yaml(SMOKE_CONFIG)
    data["shard_sp_shared_expert"] = True

    with pytest.raises(ValueError, match="requires sequence-parallel"):
        dry_run(data)
