# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from pathlib import Path
from typing import Any

from benchmarks.minimax_m3_layer_profiling.config import BenchmarkConfig, manifest


def engine_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "all2all_backend": config.all2all_backend,
        "block_size": config.block_size,
        "data_parallel_size": config.data_parallel_size,
        "disable_log_stats": True,
        "dtype": "bfloat16",
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_layerwise_nvtx_tracing": config.profile == "torch",
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "hf_overrides": {"text_config": {"num_hidden_layers": config.num_layers}},
        "language_model_only": True,
        "load_format": "dummy",
        "kv_cache_memory_bytes": config.kv_cache_memory_bytes,
        "long_prefill_token_threshold": 0,
        "max_model_len": config.max_model_len,
        "max_num_batched_tokens": max(
            config.local_batch_size * config.prompt_len, config.max_model_len
        ),
        "max_num_seqs": config.local_batch_size,
        "model": config.model,
        "seed": config.random_seed,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": config.tensor_parallel_size,
    }
    if config.data_parallel_size > 1:
        kwargs["distributed_executor_backend"] = "external_launcher"
    if config.profile == "torch":
        from vllm.config.profiler import ProfilerConfig

        assert config.profile_output_dir is not None
        output_dir = Path(config.profile_output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=str(output_dir),
            torch_profiler_record_shapes=True,
            torch_profiler_with_memory=False,
            torch_profiler_with_stack=config.profiler_with_stack,
        )
    return kwargs


def _validate_outputs(outputs: list[Any], expected_requests: int) -> list[int]:
    if len(outputs) != expected_requests:
        raise RuntimeError(f"expected {expected_requests} outputs, got {len(outputs)}")
    counts = [len(output.outputs[0].token_ids) for output in outputs]
    if counts != [2] * expected_requests:
        raise RuntimeError(f"expected two output tokens per request, got {counts}")
    return counts


def _validate_resolved_block_size(llm: Any, expected: int) -> int:
    resolved = llm.llm_engine.vllm_config.cache_config.block_size
    if resolved != expected:
        raise RuntimeError(
            f"expected resolved KV manager block size {expected}, got {resolved}"
        )
    return resolved


def run(config: BenchmarkConfig) -> None:
    if config.data_parallel_size > 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != config.gpu_count:
            raise ValueError(
                f"expected WORLD_SIZE={config.gpu_count}, got {world_size}"
            )
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "uniform_random"

    import numpy as np
    import torch.distributed as dist

    from vllm import LLM, SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.inputs import TokensPrompt

    rng = np.random.default_rng(config.random_seed)
    token_ids = rng.integers(0, 10_000, size=(config.batch_size, config.prompt_len))
    evidence = manifest(config)
    print(json.dumps({**evidence, "stage": "initializing"}, sort_keys=True))
    llm = LLM.from_engine_args(EngineArgs(**engine_kwargs(config)))
    resolved_block_size = _validate_resolved_block_size(llm, config.block_size)
    dp_rank = llm.llm_engine.vllm_config.parallel_config.data_parallel_rank
    if dp_rank is None:
        raise RuntimeError("production engine did not resolve data_parallel_rank")
    start = dp_rank * config.local_batch_size
    end = start + config.local_batch_size
    prompts = [
        TokensPrompt(prompt_token_ids=row.tolist()) for row in token_ids[start:end]
    ]
    if dist.is_initialized():
        from vllm.distributed.parallel_state import get_ep_group, get_tp_group

        rank_evidence = {
            "expert_parallel_rank": get_ep_group().rank_in_group,
            "global_rank": dist.get_rank(),
            "tensor_parallel_rank": get_tp_group().rank_in_group,
            "world_size": dist.get_world_size(),
        }
    else:
        rank_evidence = {
            "expert_parallel_rank": None,
            "global_rank": None,
            "tensor_parallel_rank": None,
            "world_size": None,
        }
    runtime_evidence = {
        **rank_evidence,
        "data_parallel_rank": dp_rank,
        "global_request_indices": list(range(start, end)),
        "resolved_kv_manager_block_size": resolved_block_size,
    }
    print(
        json.dumps({**evidence, **runtime_evidence, "stage": "ready"}, sort_keys=True)
    )
    sampling = SamplingParams(
        detokenize=False,
        ignore_eos=True,
        max_tokens=2,
        temperature=0.0,
    )

    def barrier() -> None:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def generate_once() -> list[int]:
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=False)
        return _validate_outputs(outputs, config.local_batch_size)

    for _ in range(config.warmup_iters):
        generate_once()
    barrier()
    if config.profile == "torch":
        llm.start_profile("minimax_m3_first_four_layers")
        barrier()
    try:
        counts = generate_once()
    finally:
        if config.profile == "torch":
            llm.stop_profile()
    print(
        json.dumps(
            {
                **evidence,
                **runtime_evidence,
                "profiled_output_token_counts": counts,
                "decode_executions_per_request": 1,
                "stage": "complete",
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
