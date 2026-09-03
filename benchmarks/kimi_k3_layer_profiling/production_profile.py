# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import (
    BenchmarkConfig,
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
        "enable_layerwise_nvtx_tracing": config.layerwise_nvtx_tracing_enabled,
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
        "enable_layerwise_nvtx_tracing": config.enable_layerwise_nvtx_tracing,
        "requested_all2all_backend": config.all2all_backend,
        "requested_attention_backend": config.attention_backend,
        "history_len": config.history_len,
        "execution_path": "LLM/EngineCore/production_model",
        "expected_layer_range": [0, config.num_layers - 1],
        "expert_parallel_size": config.expert_parallel_size,
        "expert_placement_strategy": config.expert_placement_strategy,
        "layerwise_profiler_scopes": config.layerwise_nvtx_tracing_enabled,
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
        temperature=0.0,
    )

    evidence = production_profile_evidence(config)
    print(json.dumps({**evidence, "stage": "initializing"}, sort_keys=True))
    llm = LLM.from_engine_args(EngineArgs(**production_engine_args_kwargs(config)))

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

    def generate_once() -> list[int]:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        return _validate_output_token_counts(
            outputs,
            expected_requests=config.local_batch_size,
            expected_tokens=config.max_tokens,
        )

    for _ in range(config.warmup_iters):
        generate_once()

    barrier()
    if config.profile == "torch":
        llm.start_profile("kimi_k3_first_block")
        barrier()
    profiled_output_token_counts: list[list[int]] = []
    try:
        for _ in range(config.profile_iters):
            profiled_output_token_counts.append(generate_once())
    finally:
        if config.profile == "torch":
            llm.stop_profile()

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
