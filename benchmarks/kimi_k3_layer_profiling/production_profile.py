# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import (
    BenchmarkConfig,
    describe_layers,
    load_model_text_config,
)


def validate_production_profile_config(config: BenchmarkConfig) -> None:
    if config.workload not in {"full_prefill", "prefill_decode"}:
        raise ValueError("unsupported production workload")
    if config.execution_mode != "eager":
        raise ValueError("layerwise Torch profiling requires execution_mode=eager")
    if config.num_layers != 12:
        raise ValueError("production profile requires the formal 12-layer block")
    if config.profile not in {"none", "torch"}:
        raise ValueError("production profile supports profile=none or profile=torch")
    if config.profile == "torch" and not config.profile_output_dir:
        raise ValueError("profile_output_dir is required when profile=torch")
    if config.profile == "none" and config.profile_output_dir:
        raise ValueError("profile_output_dir requires profile=torch")
    if config.workload == "prefill_decode" and config.profile_iters != 1:
        raise ValueError("prefill_decode requires profile_iters=1")


def production_engine_args_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    validate_production_profile_config(config)
    kwargs: dict[str, Any] = {
        "all2all_backend": config.all2all_backend,
        "attention_backend": (
            None if config.attention_backend == "auto" else config.attention_backend
        ),
        "data_parallel_size": config.data_parallel_size,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "disable_log_stats": True,
        "dtype": config.dtype,
        "enable_dbo": config.enable_dbo,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_return_routed_experts": config.capture_routed_experts,
        "enable_layerwise_nvtx_tracing": config.profile == "torch",
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "hf_overrides": {"text_config": {"num_hidden_layers": config.num_layers}},
        "language_model_only": True,
        "load_format": "dummy",
        "kv_cache_memory_bytes": config.kv_cache_memory_bytes,
        "expert_placement_strategy": config.expert_placement_strategy,
        "kda_prefill_backend": config.kda_prefill_backend,
        "kv_cache_dtype": config.kv_cache_dtype,
        "linear_backend": config.linear_backend,
        "long_prefill_token_threshold": 0,
        "max_model_len": config.max_model_len,
        "max_num_batched_tokens": max(
            config.local_batch_size * config.prompt_len, config.max_model_len
        ),
        "max_num_seqs": config.local_batch_size,
        "model": config.model,
        "moe_backend": config.moe_backend,
        "seed": config.random_seed,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": config.tensor_parallel_size,
    }
    if config.tp_sync_before_all_gather:
        kwargs["worker_extension_cls"] = (
            "benchmarks.kimi_k3_layer_profiling.worker_extension."
            "KimiK3ProfilingWorkerExtension"
        )
    if config.mla_prefill_backend != "auto":
        kwargs["attention_config"] = {"mla_prefill_backend": config.mla_prefill_backend}
    if config.data_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "external_launcher"
    if config.profile == "torch":
        from vllm.config.profiler import ProfilerConfig

        assert config.profile_output_dir is not None
        profile_output_dir = Path(config.profile_output_dir).resolve()
        profile_output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(profile_output_dir),
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=False,
            torch_profiler_with_stack=config.profiler_with_stack,
        )
    return kwargs


def production_profile_evidence(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "data_parallel_size": config.data_parallel_size,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "enable_dbo": config.enable_dbo,
        "enable_expert_parallel": config.enable_expert_parallel,
        "capture_routed_experts": config.capture_routed_experts,
        "requested_all2all_backend": config.all2all_backend,
        "requested_attention_backend": config.attention_backend,
        "history_len": config.history_len,
        "execution_path": "LLM/EngineCore/production_model",
        "expected_layer_range": [0, config.num_layers - 1],
        "expert_parallel_size": config.expert_parallel_size,
        "expert_placement_strategy": config.expert_placement_strategy,
        "layerwise_profiler_scopes": config.profile == "torch",
        "local_batch_size": config.local_batch_size,
        "max_model_len": config.max_model_len,
        "max_tokens": config.max_tokens,
        "num_layers": config.num_layers,
        "requested_kv_cache_memory_bytes": config.kv_cache_memory_bytes,
        "requested_kda_prefill_backend": config.kda_prefill_backend,
        "requested_kv_cache_dtype": config.kv_cache_dtype,
        "requested_linear_backend": config.linear_backend,
        "requested_mla_prefill_backend": config.mla_prefill_backend,
        "requested_moe_backend": config.moe_backend,
        "routing_strategy": config.routing_strategy,
        "shard_sp_shared_expert": config.shard_sp_shared_expert,
        "tensor_parallel_size": config.tensor_parallel_size,
        "tp_sync_before_all_gather": config.tp_sync_before_all_gather,
        "profile": config.profile,
        "profile_output_dir": config.profile_output_dir,
        "profiler_with_stack": config.profiler_with_stack,
        "query_len": config.query_len,
        "warmup_iters": config.warmup_iters,
        "workload": config.workload,
        "profile_iters": config.profile_iters,
    }


def _validate_output_token_counts(
    outputs: list[Any], expected_requests: int, expected_tokens: int
) -> list[int]:
    if len(outputs) != expected_requests:
        raise RuntimeError(
            f"expected {expected_requests} request outputs, got {len(outputs)}"
        )
    token_counts: list[int] = []
    for request_index, output in enumerate(outputs):
        if len(output.outputs) != 1:
            raise RuntimeError(
                f"request {request_index} produced {len(output.outputs)} sequences, "
                "expected 1"
            )
        token_count = len(output.outputs[0].token_ids)
        if token_count != expected_tokens:
            raise RuntimeError(
                f"request {request_index} produced {token_count} output tokens, "
                f"expected {expected_tokens}"
            )
        token_counts.append(token_count)
    return token_counts


def _routing_histograms(outputs: list[Any], config: BenchmarkConfig) -> dict[str, Any]:
    import numpy as np

    text_config = load_model_text_config(config.model)
    num_experts = int(text_config["num_experts"])
    topk = int(text_config["num_experts_per_token"])
    expected_tokens = config.prompt_len + config.max_tokens - 1
    moe_layer_ids = {
        layer.logical_layer
        for layer in describe_layers(config)
        if layer.ffn_type == "MoE"
    }
    arrays = []
    for request_index, output in enumerate(outputs):
        routed = output.routed_experts
        if routed is None:
            raise RuntimeError(f"request {request_index} has no routed_experts")
        array = np.asarray(routed)
        expected_shape = (expected_tokens, config.num_layers, topk)
        if array.shape != expected_shape:
            raise RuntimeError(
                f"request {request_index} routed_experts has shape {array.shape}, "
                f"expected {expected_shape}"
            )
        arrays.append(array)

    phase_ranges = {
        "prefill": (0, config.prompt_len),
        "decode": (config.prompt_len, expected_tokens),
    }
    phases: dict[str, Any] = {}
    for phase, (start, end) in phase_ranges.items():
        phase_data = np.concatenate([array[start:end] for array in arrays], axis=0)
        layers = []
        for layer_id in range(config.num_layers):
            if layer_id not in moe_layer_ids:
                continue
            ids = phase_data[:, layer_id, :].reshape(-1)
            if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= num_experts):
                raise RuntimeError(
                    f"layer {layer_id} has expert IDs outside [0, {num_experts})"
                )
            counts = np.bincount(ids, minlength=num_experts)
            layers.append(
                {
                    "layer": layer_id,
                    "assignment_count": int(counts.sum()),
                    "expert_counts": {
                        str(i): int(count) for i, count in enumerate(counts) if count
                    },
                    "unique_expert_count": int(np.count_nonzero(counts)),
                }
            )
        phases[phase] = layers
    return {
        "num_experts": num_experts,
        "num_experts_per_token": topk,
        "request_count": len(outputs),
        "routing_shape_per_request": list(arrays[0].shape),
        "phases": phases,
    }


def _aggregate_routing_audit(
    local_audit: dict[str, Any] | None,
    config: BenchmarkConfig,
    tp_rank: int | None,
) -> dict[str, Any] | None:
    import torch.distributed as dist

    source = local_audit if tp_rank in (None, 0) else None
    gathered: list[dict[str, Any] | None]
    if dist.is_initialized():
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, source)
        if dist.get_rank() != 0:
            return None
    else:
        gathered = [source]
    sources = [item for item in gathered if item is not None]
    if len(sources) != config.data_parallel_size:
        raise RuntimeError(
            f"expected {config.data_parallel_size} routing sources, got {len(sources)}"
        )

    num_experts = int(sources[0]["num_experts"])
    if any(int(source["num_experts"]) != num_experts for source in sources):
        raise RuntimeError("routing sources disagree on num_experts")
    text_config = load_model_text_config(config.model)
    hidden_size = int(text_config["routed_expert_hidden_size"])
    intermediate_size = int(text_config["moe_intermediate_size"])
    weight_config = text_config["quantization_config"]["config_groups"]["group_0"][
        "weights"
    ]
    if int(weight_config["num_bits"]) != 4:
        raise RuntimeError("routing audit weight estimate requires 4-bit weights")
    group_size = int(weight_config["group_size"])
    w13_values = 2 * hidden_size * intermediate_size
    w2_values = hidden_size * intermediate_size
    expert_weight_bytes = sum(
        (values + 1) // 2 + (values + group_size - 1) // group_size
        for values in (w13_values, w2_values)
    )
    ep_size = config.expert_parallel_size
    if num_experts % ep_size:
        raise RuntimeError("num_experts must be divisible by expert_parallel_size")
    experts_per_rank = num_experts // ep_size
    phases: dict[str, Any] = {}
    for phase in ("prefill", "decode"):
        layer_summaries = []
        source_layers = {
            int(layer["layer"]): layer for layer in sources[0]["phases"][phase]
        }
        for layer_id in sorted(source_layers):
            counts: Counter[int] = Counter()
            for source_audit in sources:
                layers_by_id = {
                    int(layer["layer"]): layer
                    for layer in source_audit["phases"][phase]
                }
                expert_counts = layers_by_id[layer_id]["expert_counts"]
                counts.update(
                    {int(expert): int(count) for expert, count in expert_counts.items()}
                )
            rank_summaries = []
            for ep_rank in range(ep_size):
                if config.expert_placement_strategy == "linear":
                    expert_ids = range(
                        ep_rank * experts_per_rank,
                        (ep_rank + 1) * experts_per_rank,
                    )
                else:
                    expert_ids = range(ep_rank, num_experts, ep_size)
                local_counts = {i: counts[i] for i in expert_ids if counts[i]}
                rank_summaries.append(
                    {
                        "ep_rank": ep_rank,
                        "assignment_count": sum(local_counts.values()),
                        "unique_local_expert_count": len(local_counts),
                        "cold_unique_expert_weight_bytes_estimate": (
                            len(local_counts) * expert_weight_bytes
                        ),
                        "expert_counts": {
                            str(i): count for i, count in local_counts.items()
                        },
                    }
                )
            layer_summaries.append(
                {
                    "layer": layer_id,
                    "global_assignment_count": sum(counts.values()),
                    "ep_ranks": rank_summaries,
                }
            )
        phases[phase] = layer_summaries
    return {
        "audit_kind": "logical_expert_routing",
        "data_parallel_sources": len(sources),
        "expert_parallel_size": ep_size,
        "expert_placement_strategy": config.expert_placement_strategy,
        "experts_per_rank": experts_per_rank,
        "estimated_checkpoint_weight_bytes_per_expert": expert_weight_bytes,
        "weight_byte_estimate_assumption": (
            "MXFP4 packed values plus one uint8 scale per group; each unique "
            "expert weight hypothetically read once from cold memory"
        ),
        "hbm_bytes_scope": "not_measured",
        "phases": phases,
    }


def _label_tp_all_gather_measurements(
    measurements: list[dict[str, Any]], config: BenchmarkConfig
) -> list[dict[str, Any]]:
    calls_per_execution = config.num_layers + 1
    execution_count = 2 if config.workload == "prefill_decode" else 1
    expected_calls = calls_per_execution * execution_count * config.profile_iters
    for worker in measurements:
        calls = worker["calls"]
        if len(calls) != expected_calls:
            raise RuntimeError(
                f"rank {worker['global_rank']} recorded {len(calls)} TP all-gathers, "
                f"expected {expected_calls}"
            )
        for index, call in enumerate(calls):
            execution_index = index // calls_per_execution
            position = index % calls_per_execution
            call["phase"] = (
                "prefill"
                if config.workload == "full_prefill" or execution_index % 2 == 0
                else "decode"
            )
            if position < config.num_layers:
                call["scope"] = f"layers.{position}.entry_sp_all_gather"
                call["arrival_skew_source"] = (
                    "input_pipeline" if position == 0 else f"layers.{position - 1}.tail"
                )
            else:
                call["scope"] = "model.final_sp_all_gather"
                call["arrival_skew_source"] = f"layers.{config.num_layers - 1}.tail"
    return measurements


def run_production_profile(config: BenchmarkConfig) -> None:
    validate_production_profile_config(config)
    if config.data_parallel_size > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != config.gpu_count:
            raise ValueError(
                "data-parallel profiling must run under torchrun with "
                f"WORLD_SIZE={config.gpu_count}, got {world_size}"
            )
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = config.routing_strategy
    os.environ["VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT"] = (
        "1" if config.shard_sp_shared_expert else "0"
    )

    import numpy as np

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt

    text_config = load_model_text_config(config.model)
    vocab_size = min(int(text_config["vocab_size"]), 10_000)
    rng = np.random.default_rng(config.random_seed)
    token_ids = rng.integers(
        0,
        vocab_size,
        size=(config.batch_size, config.prompt_len),
    )
    sampling_params = SamplingParams(
        detokenize=False,
        ignore_eos=True,
        # The prefill samples output token 1. Requesting output token 2 forces
        # exactly one subsequent single-token decode model execution.
        max_tokens=config.max_tokens,
        routed_experts_prompt_start=0,
        temperature=0.0,
    )

    evidence = production_profile_evidence(config)
    print(json.dumps({**evidence, "stage": "initializing"}, sort_keys=True))
    llm = LLM.from_engine_args(EngineArgs(**production_engine_args_kwargs(config)))
    if config.tp_sync_before_all_gather:
        install_results = llm.llm_engine.collective_rpc(
            "install_tp_all_gather_sync_ablation"
        )
        if not all(result.get("installed") for result in install_results):
            raise RuntimeError("failed to install TP all-gather sync ablation")
        print(
            json.dumps(
                {
                    "stage": "tp_sync_ablation_installed",
                    "worker_count": len(install_results),
                },
                sort_keys=True,
            )
        )

    import torch.distributed as dist

    parallel_config = llm.llm_engine.vllm_config.parallel_config
    dp_rank = parallel_config.data_parallel_rank
    if dp_rank is None:
        raise RuntimeError("production engine did not resolve data_parallel_rank")
    local_start = dp_rank * config.local_batch_size
    local_end = local_start + config.local_batch_size
    prompts = [
        TokensPrompt(prompt_token_ids=row.tolist())
        for row in token_ids[local_start:local_end]
    ]
    distributed_evidence: dict[str, int | None]
    if dist.is_initialized():
        from vllm.distributed.parallel_state import (
            get_dcp_group,
            get_ep_group,
            get_tp_group,
        )

        distributed_evidence = {
            "decode_context_parallel_rank": get_dcp_group().rank_in_group,
            "expert_parallel_rank": get_ep_group().rank_in_group,
            "global_rank": dist.get_rank(),
            "tensor_parallel_rank": get_tp_group().rank_in_group,
            "world_size": dist.get_world_size(),
        }
    else:
        distributed_evidence = {
            "decode_context_parallel_rank": None,
            "expert_parallel_rank": None,
            "global_rank": None,
            "tensor_parallel_rank": None,
            "world_size": None,
        }
    rank_evidence = {
        **evidence,
        **distributed_evidence,
        "data_parallel_rank": dp_rank,
        "global_request_indices": list(range(local_start, local_end)),
        "stage": "ready",
    }
    print(json.dumps(rank_evidence, sort_keys=True))

    def barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def generate_once() -> tuple[list[int], list[Any]]:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return (
            _validate_output_token_counts(
                outputs,
                expected_requests=config.local_batch_size,
                expected_tokens=config.max_tokens,
            ),
            outputs,
        )

    for _ in range(config.warmup_iters):
        generate_once()

    if config.tp_sync_before_all_gather:
        llm.llm_engine.collective_rpc("reset_tp_all_gather_sync_measurements")

    barrier()
    if config.profile == "torch":
        llm.start_profile("kimi_k3_first_block")
        barrier()
    profiled_output_token_counts: list[list[int]] = []
    routing_audits: list[dict[str, Any]] = []
    try:
        for _ in range(config.profile_iters):
            token_counts, outputs = generate_once()
            profiled_output_token_counts.append(token_counts)
            if config.capture_routed_experts:
                routing_audits.append(_routing_histograms(outputs, config))
    finally:
        if config.profile == "torch":
            llm.stop_profile()

    if config.tp_sync_before_all_gather:
        tp_measurements = llm.llm_engine.collective_rpc(
            "collect_tp_all_gather_sync_measurements"
        )
        tp_measurements = _label_tp_all_gather_measurements(tp_measurements, config)
        print(
            json.dumps(
                {
                    "measurements": tp_measurements,
                    "stage": "tp_sync_measurements",
                    "timing_rule": "CUDA event elapsed time per synchronized call",
                },
                sort_keys=True,
            )
        )

    if config.capture_routed_experts:
        if len(routing_audits) != 1:
            raise RuntimeError("routing audit requires exactly one profiled iteration")
        global_routing_audit = _aggregate_routing_audit(
            routing_audits[0],
            config,
            distributed_evidence["tensor_parallel_rank"],
        )
        if global_routing_audit is not None:
            print(
                json.dumps(
                    {**global_routing_audit, "stage": "routing_audit"},
                    sort_keys=True,
                )
            )

    print(
        json.dumps(
            {
                **rank_evidence,
                "decode_executions_per_request": config.max_tokens - 1,
                "profiled_output_token_counts": profiled_output_token_counts,
                "stage": "complete",
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
