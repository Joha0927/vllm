# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import os
from dataclasses import asdict
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import BenchmarkConfig

_TORCHRUN_ENV = {"LOCAL_RANK", "RANK", "WORLD_SIZE"}


def parallel_config_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "all2all_backend": config.all2all_backend,
        "data_parallel_size": config.data_parallel_size,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "enable_expert_parallel": config.enable_expert_parallel,
        "tensor_parallel_size": config.tensor_parallel_size,
    }


def benchmark_config_digest(config: BenchmarkConfig) -> bytes:
    payload = json.dumps(asdict(config), separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).digest()


def run_distributed_smoke(config: BenchmarkConfig) -> None:
    missing = sorted(_TORCHRUN_ENV - os.environ.keys())
    if missing:
        raise RuntimeError(
            "Distributed smoke must be launched with torchrun; missing "
            + ", ".join(missing)
        )

    import torch
    import torch.distributed as dist

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dcp_group,
        get_ep_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.platforms import current_platform

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != config.gpu_count:
        raise RuntimeError(
            f"torchrun world size {world_size} does not match gpu_count "
            f"{config.gpu_count}"
        )
    if local_rank >= torch.accelerator.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
            f"{torch.accelerator.device_count()}"
        )

    torch.accelerator.set_device_index(local_rank)
    parallel_config = ParallelConfig(**parallel_config_kwargs(config))
    vllm_config = VllmConfig(parallel_config=parallel_config)

    initialized = False
    try:
        init_distributed_environment()
        initialized = True

        digest = torch.tensor(
            list(benchmark_config_digest(config)), dtype=torch.uint8, device="cuda"
        )
        gathered = [torch.empty_like(digest) for _ in range(world_size)]
        dist.all_gather(gathered, digest)
        if any(not torch.equal(item, digest) for item in gathered):
            raise RuntimeError("Benchmark configuration differs across ranks")

        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(
                tensor_model_parallel_size=config.tensor_parallel_size,
                decode_context_model_parallel_size=(
                    config.decode_context_parallel_size
                ),
            )

        record = {
            "dcp_rank": get_dcp_group().rank_in_group,
            "dcp_size": get_dcp_group().world_size,
            "device": current_platform.get_device_name(local_rank),
            "ep_rank": get_ep_group().rank_in_group,
            "ep_size": get_ep_group().world_size,
            "local_rank": local_rank,
            "rank": rank,
            "requested_all2all_backend": parallel_config.all2all_backend,
            "smoke_scope": "distributed_groups",
            "status": "PASS",
            "tp_rank": get_tp_group().rank_in_group,
            "tp_size": get_tp_group().world_size,
            "world_size": world_size,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        dist.barrier()
    finally:
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
