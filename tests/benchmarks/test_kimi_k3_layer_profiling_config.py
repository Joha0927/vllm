# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from benchmarks.kimi_k3_layer_profiling.benchmark import parse_args, run
from benchmarks.kimi_k3_layer_profiling.config import (
    apply_overrides,
    dry_run,
    load_yaml,
)
from benchmarks.kimi_k3_layer_profiling.production_profile import (
    _aggregate_routing_audit,
    _label_tp_all_gather_measurements,
    _routing_histograms,
    _validate_output_token_counts,
    production_engine_args_kwargs,
    production_profile_evidence,
    validate_production_profile_config,
)
from vllm.outputs import CompletionOutput, RequestOutput

ROOT = Path(__file__).parents[2]
SMOKE_CONFIG = ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/smoke.yaml"
PREFILL_DECODE_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_with_stack.yaml"
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


PREFILL_DECODE_TP2_DP4_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_tp2_dp4_with_stack.yaml"
)
PREFILL_DECODE_TP2_DP4_FLASHINFER_ONE_SIDED_WITH_STACK_CONFIG = (
    ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
    "prefill_decode_bs8_p16384_tp2_dp4_flashinfer_one_sided_with_stack.yaml"
)
BS32_CONFIGS = (
    (
        ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
        "prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml",
        1,
        8,
        "allgather_reducescatter",
    ),
    (
        ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
        "prefill_decode_bs32_p4096_tp2_dp4_ag_rs.yaml",
        2,
        4,
        "allgather_reducescatter",
    ),
    (
        ROOT / "benchmarks/kimi_k3_layer_profiling/shapes/"
        "prefill_decode_bs32_p4096_tp2_dp4_a2a.yaml",
        2,
        4,
        "flashinfer_nvlink_one_sided",
    ),
)


def _config():
    return dry_run(load_yaml(SMOKE_CONFIG)).config


def _request_output(*routed_experts: Any) -> RequestOutput:
    completions = [
        CompletionOutput(
            index=index,
            text="",
            token_ids=[1, 2],
            cumulative_logprob=None,
            logprobs=None,
            routed_experts=routed,
        )
        for index, routed in enumerate(routed_experts)
    ]
    return RequestOutput(
        request_id="request-0",
        prompt=None,
        prompt_token_ids=None,
        prompt_logprobs=None,
        outputs=completions,
        finished=True,
    )


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
    assert "worker_extension_cls" not in kwargs
    assert kwargs["enable_return_routed_experts"] is False


def test_tp_sync_ablation_is_opt_in_and_requires_tp() -> None:
    config = replace(_config(), tp_sync_before_all_gather=True)
    kwargs = production_engine_args_kwargs(config)

    assert kwargs["worker_extension_cls"].endswith("KimiK3ProfilingWorkerExtension")

    with pytest.raises(ValueError, match="requires tensor_parallel_size > 1"):
        dry_run(
            {
                **load_yaml(PREFILL_DECODE_WITH_STACK_CONFIG),
                "tp_sync_before_all_gather": True,
            }
        )


def test_routed_expert_capture_is_separate_from_torch_profile() -> None:
    config = replace(_config(), capture_routed_experts=True)
    assert production_engine_args_kwargs(config)["enable_return_routed_experts"]

    with pytest.raises(ValueError, match="audit mode and requires profile=none"):
        dry_run(
            {
                **load_yaml(SMOKE_CONFIG),
                "capture_routed_experts": True,
                "profile": "torch",
                "profile_output_dir": "traces",
            }
        )


def test_routing_audit_ignores_dense_layer_zero() -> None:
    import numpy as np

    config = replace(
        _config(),
        workload="prefill_decode",
        batch_size=1,
        history_len=2,
        query_len=1,
        data_parallel_size=1,
    )
    routed = np.zeros((3, 12, 16), dtype=np.int32)
    routed[:, 0, :] = -1
    routed[:, 1:, :] = np.arange(16, dtype=np.int32)
    outputs = [_request_output(routed)]

    local = _routing_histograms(outputs, config)
    assert [layer["layer"] for layer in local["phases"]["prefill"]] == list(
        range(1, 12)
    )
    global_audit = _aggregate_routing_audit(local, config, tp_rank=0)
    assert global_audit is not None
    assert global_audit["hbm_bytes_scope"] == "not_measured"
    assert global_audit["estimated_checkpoint_weight_bytes_per_expert"] > 0
    assert global_audit["phases"]["prefill"][0]["global_assignment_count"] == 32
    assert global_audit["phases"]["decode"][0]["global_assignment_count"] == 16


def test_routing_audit_requires_completion_routed_experts() -> None:
    import numpy as np

    config = replace(
        _config(),
        workload="prefill_decode",
        batch_size=1,
        history_len=2,
        query_len=1,
        data_parallel_size=1,
    )

    with pytest.raises(
        RuntimeError, match="request 0 completion has no routed_experts"
    ):
        _routing_histograms([_request_output(None)], config)

    wrong_shape = np.zeros((2, 12, 16), dtype=np.int32)
    with pytest.raises(RuntimeError, match=r"has shape \(2, 12, 16\)"):
        _routing_histograms([_request_output(wrong_shape)], config)


@pytest.mark.parametrize("completion_count", [0, 2])
def test_routing_audit_requires_exactly_one_completion(
    completion_count: int,
) -> None:
    import numpy as np

    routed = np.zeros((3, 12, 16), dtype=np.int32)
    output = _request_output(*([routed] * completion_count))

    with pytest.raises(
        RuntimeError,
        match=(rf"request 0 produced {completion_count} sequences, expected exactly 1"),
    ):
        _routing_histograms([output], replace(_config(), data_parallel_size=1))


@pytest.mark.parametrize("invalid_expert_id", [-1, 896])
def test_routing_audit_rejects_out_of_range_expert_ids(
    invalid_expert_id: int,
) -> None:
    import numpy as np

    config = replace(
        _config(),
        workload="prefill_decode",
        batch_size=1,
        history_len=2,
        query_len=1,
        data_parallel_size=1,
    )
    routed = np.zeros((3, 12, 16), dtype=np.int32)
    routed[0, 1, 0] = invalid_expert_id

    with pytest.raises(RuntimeError, match=r"expert IDs outside \[0, 896\)"):
        _routing_histograms([_request_output(routed)], config)


def test_tp_sync_measurements_are_attributed_to_preceding_work() -> None:
    config = dry_run(load_yaml(PREFILL_DECODE_TP2_DP4_WITH_STACK_CONFIG)).config
    measurements: list[dict[str, Any]] = [
        {
            "global_rank": 0,
            "calls": [
                {"call_index": index, "elapsed_ms": 1.0, "input_shape": [1, 1]}
                for index in range(26)
            ],
        }
    ]

    result = _label_tp_all_gather_measurements(measurements, config)[0]["calls"]
    assert result[0]["phase"] == "prefill"
    assert result[0]["arrival_skew_source"] == "input_pipeline"
    assert result[1]["arrival_skew_source"] == "layers.0.tail"
    assert result[12]["scope"] == "model.final_sp_all_gather"
    assert result[13]["phase"] == "decode"

    with pytest.raises(RuntimeError, match="recorded 25 TP all-gathers"):
        _label_tp_all_gather_measurements(
            [{"global_rank": 0, "calls": measurements[0]["calls"][:-1]}], config
        )


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


@pytest.mark.parametrize(("path", "tp", "dp", "backend"), BS32_CONFIGS)
def test_bs32_p4096_matrix(path: Path, tp: int, dp: int, backend: str) -> None:
    config = dry_run(load_yaml(path)).config

    assert config.batch_size == 32
    assert config.prompt_len == 4096
    assert config.max_tokens == 2
    assert config.tensor_parallel_size == tp
    assert config.data_parallel_size == dp
    assert config.all2all_backend == backend
    assert config.capture_routed_experts is False
    assert config.tp_sync_before_all_gather is False


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


def test_shared_expert_sharding_requires_sequence_parallel_execution() -> None:
    data = load_yaml(SMOKE_CONFIG)
    data["shard_sp_shared_expert"] = True

    with pytest.raises(ValueError, match="requires sequence-parallel"):
        dry_run(data)
