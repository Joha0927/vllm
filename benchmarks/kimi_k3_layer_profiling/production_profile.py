# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import (
    BenchmarkConfig,
    load_model_text_config,
)


def validate_production_profile_config(config: BenchmarkConfig) -> None:
    if config.phase != "prefill":
        raise ValueError("production profile currently supports prefill only")
    if config.context_len != config.query_len:
        raise ValueError("prefill requires context_len=query_len")
    if config.execution_mode != "eager":
        raise ValueError("layerwise NVTX profiling requires execution_mode=eager")
    if config.num_layers != 12 or config.diagnostic_partial_block:
        raise ValueError("production profile requires the formal 12-layer block")
    if config.profile not in {"none", "cuda"}:
        raise ValueError("production profile supports profile=none or profile=cuda")


def production_engine_args_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    validate_production_profile_config(config)
    kwargs: dict[str, Any] = {
        "all2all_backend": config.all2all_backend,
        "data_parallel_size": config.data_parallel_size,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "disable_log_stats": True,
        "dtype": config.dtype,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_layerwise_nvtx_tracing": True,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "hf_overrides": {"text_config": {"num_hidden_layers": config.num_layers}},
        "language_model_only": True,
        "load_format": "dummy",
        "kv_cache_memory_bytes": 4 * 1024**3,
        "max_model_len": config.context_len + 1,
        "max_num_batched_tokens": max(
            config.num_scheduled_tokens, config.context_len + 1
        ),
        "max_num_seqs": config.batch_size,
        "model": config.model,
        "seed": config.random_seed,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": config.tensor_parallel_size,
    }
    if config.profile == "cuda":
        from vllm.config.profiler import ProfilerConfig

        kwargs["profiler_config"] = ProfilerConfig(profiler="cuda")
    return kwargs


def production_profile_evidence(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "context_len": config.context_len,
        "execution_path": "LLM/EngineCore/production_model",
        "expected_layer_range": [0, config.num_layers - 1],
        "layerwise_nvtx_tracing": True,
        "model_class_override": False,
        "num_layers": config.num_layers,
        "profile": config.profile,
        "query_len": config.query_len,
        "uses_custom_block_wrapper": False,
        "uses_manual_kv_cache_init": False,
        "warmup_iters": config.warmup_iters,
        "profile_iters": config.profile_iters,
    }


def run_production_profile(config: BenchmarkConfig) -> None:
    validate_production_profile_config(config)
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = config.routing_strategy

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
        size=(config.batch_size, config.query_len),
    )
    prompts = [TokensPrompt(prompt_token_ids=row.tolist()) for row in token_ids]
    sampling_params = SamplingParams(
        detokenize=False,
        ignore_eos=True,
        max_tokens=1,
        temperature=0.0,
    )

    evidence = production_profile_evidence(config)
    print(json.dumps({**evidence, "stage": "initializing"}, sort_keys=True))
    llm = LLM.from_engine_args(EngineArgs(**production_engine_args_kwargs(config)))

    def generate_once() -> None:
        llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)

    for _ in range(config.warmup_iters):
        generate_once()

    if config.profile == "cuda":
        llm.start_profile("kimi_k3_first_block")
    try:
        for _ in range(config.profile_iters):
            generate_once()
    finally:
        if config.profile == "cuda":
            llm.stop_profile()

    print(
        json.dumps({**evidence, "stage": "complete", "status": "PASS"}, sort_keys=True)
    )
