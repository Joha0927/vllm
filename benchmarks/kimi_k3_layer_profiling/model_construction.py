# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import json
import os
import subprocess
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import BenchmarkConfig
from benchmarks.kimi_k3_layer_profiling.distributed import (
    benchmark_config_digest,
)

MODEL_CLASS_OVERRIDES = {
    "KimiLinearForCausalLM": (
        "benchmarks.kimi_k3_layer_profiling.block_model:KimiK3BlockProfiler"
    )
}
_TORCHRUN_ENV = {"LOCAL_RANK", "RANK", "WORLD_SIZE"}


def engine_args_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "all2all_backend": config.all2all_backend,
        "data_parallel_size": config.data_parallel_size,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "disable_log_stats": True,
        "dtype": config.dtype,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enforce_eager": True,
        "language_model_only": True,
        "load_format": "dummy",
        "max_model_len": config.context_len,
        "max_num_batched_tokens": config.num_scheduled_tokens,
        "max_num_seqs": config.batch_size,
        "model": config.model,
        "model_class_overrides": MODEL_CLASS_OVERRIDES,
        "seed": config.random_seed,
        "skip_tokenizer_init": True,
        "tensor_parallel_size": config.tensor_parallel_size,
    }


def validate_model_construction_config(config: BenchmarkConfig) -> None:
    if config.num_layers != 12 or config.diagnostic_partial_block:
        raise ValueError("Model construction smoke requires the formal 12-layer block")


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise RuntimeError(f"Unexpected Git commit: {commit!r}")
    return commit


def memory_mib(torch_module) -> dict[str, float]:
    return {
        "allocated_mib": torch_module.accelerator.memory_allocated() / 2**20,
        "reserved_mib": torch_module.accelerator.memory_reserved() / 2**20,
    }


def _validate_stage_missing(module, name: str) -> dict[str, int | str]:
    if type(module).__name__ != "StageMissingLayer":
        raise RuntimeError(
            f"language_model_only did not replace {name} with "
            f"StageMissingLayer: {type(module).__name__}"
        )
    registered_parameter_count = sum(
        parameter.numel() for parameter in module.parameters()
    )
    if registered_parameter_count != 0:
        raise RuntimeError(
            f"{name} has {registered_parameter_count} registered parameters"
        )

    backing_module = module.__dict__.get("module")
    if backing_module is None:
        raise RuntimeError(f"{name} StageMissingLayer has no backing module")
    backing_parameters = list(backing_module.parameters())
    backing_buffers = list(backing_module.buffers())
    non_meta_parameter_devices = sorted(
        {
            str(parameter.device)
            for parameter in backing_parameters
            if parameter.device.type != "meta"
        }
    )
    if non_meta_parameter_devices:
        raise RuntimeError(
            f"{name} retained non-meta parameters on {non_meta_parameter_devices}"
        )
    accelerator_buffer_devices = sorted(
        {
            str(buffer.device)
            for buffer in backing_buffers
            if buffer.device.type not in {"cpu", "meta"}
        }
    )
    if accelerator_buffer_devices:
        raise RuntimeError(
            f"{name} retained accelerator buffers on {accelerator_buffer_devices}"
        )
    return {
        "backing_buffer_count": sum(buffer.numel() for buffer in backing_buffers),
        "backing_parameter_count": sum(
            parameter.numel() for parameter in backing_parameters
        ),
        "class": type(module).__name__,
        "registered_parameter_count": registered_parameter_count,
    }


def run_model_construction_smoke(config: BenchmarkConfig) -> None:
    validate_model_construction_config(config)
    missing = sorted(_TORCHRUN_ENV - os.environ.keys())
    if missing:
        raise RuntimeError(
            "Model construction smoke must be launched with torchrun; missing "
            + ", ".join(missing)
        )

    import torch
    import torch.distributed as dist

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dcp_group,
        get_ep_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader import get_model
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
    initialized = False
    model = None
    block_model = None
    layers = None
    run_commit = None
    failure_stage = "engine_config"
    try:
        run_commit = git_commit()
        vllm_config = EngineArgs(**engine_args_kwargs(config)).create_engine_config()
        failure_stage = "distributed_init"
        init_distributed_environment()
        initialized = True

        failure_stage = "configuration_agreement"
        run_identity = hashlib.sha256(
            benchmark_config_digest(config) + run_commit.encode()
        ).digest()
        digest = torch.tensor(
            list(run_identity),
            dtype=torch.uint8,
            device="cuda",
        )
        gathered = [torch.empty_like(digest) for _ in range(world_size)]
        dist.all_gather(gathered, digest)
        if any(not torch.equal(item, digest) for item in gathered):
            raise RuntimeError("Benchmark configuration differs across ranks")

        failure_stage = "model_parallel_init"
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(
                tensor_model_parallel_size=config.tensor_parallel_size,
                decode_context_model_parallel_size=(
                    config.decode_context_parallel_size
                ),
            )

        torch.accelerator.reset_peak_memory_stats()
        memory_before = memory_mib(torch)
        failure_stage = "dummy_model_loading"
        model = get_model(vllm_config=vllm_config)
        torch.accelerator.synchronize()
        memory_after = memory_mib(torch)

        failure_stage = "model_validation"
        if type(model).__name__ != "KimiK3ForConditionalGeneration":
            raise RuntimeError(f"Unexpected outer model class: {type(model).__name__}")
        block_model = getattr(model, "language_model", None)
        if block_model is None or type(block_model).__name__ != "KimiK3BlockProfiler":
            raise RuntimeError(
                "Expected language_model to be KimiK3BlockProfiler, got "
                f"{type(block_model).__name__}"
            )

        vision_metadata = _validate_stage_missing(
            getattr(model, "vision_tower", None), "vision_tower"
        )
        projector_metadata = _validate_stage_missing(
            getattr(model, "mm_projector", None), "mm_projector"
        )

        layers = block_model.model.layers
        layer_indices = [layer.layer_idx for layer in layers]
        if layer_indices != list(range(12)):
            raise RuntimeError(f"Unexpected layer indices: {layer_indices}")

        module_names = [name for name, _ in block_model.named_modules()]
        parameter_names = [name for name, _ in block_model.named_parameters()]
        forbidden = (
            "embed_tokens",
            "lm_head",
            "vision_tower",
            "model.layers.12.",
            "model.norm",
            "output_attn_res_norm",
            "output_attn_res_proj",
        )
        unexpected = [
            name
            for name in [*module_names, *parameter_names]
            if any(item in name for item in forbidden)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected modules or parameters: {unexpected[:8]}")
        outer_parameter_names = [name for name, _ in model.named_parameters()]
        unexpected_outer_parameters = [
            name
            for name in outer_parameter_names
            if not name.startswith("language_model.")
        ]
        if unexpected_outer_parameters:
            raise RuntimeError(
                "Outer model retained parameters outside language_model: "
                f"{unexpected_outer_parameters[:8]}"
            )

        layer_types = [
            {
                "attention": type(layer.self_attn).__name__,
                "ffn": "MoE" if layer.is_moe_layer else "dense",
                "layer": layer.layer_idx,
            }
            for layer in layers
        ]
        expected_attention = [
            "KimiK3DeltaAttention",
            "KimiK3DeltaAttention",
            "KimiK3DeltaAttention",
            "MultiHeadLatentAttention",
        ] * 3
        if [item["attention"] for item in layer_types] != expected_attention:
            raise RuntimeError(f"Unexpected attention layers: {layer_types}")
        if [item["ffn"] for item in layer_types] != ["dense", *("MoE",) * 11]:
            raise RuntimeError(f"Unexpected FFN layers: {layer_types}")
        block_write_layers = [
            layer.layer_idx for layer in layers if layer.is_block_write_layer
        ]
        if block_write_layers != [0]:
            raise RuntimeError(
                f"Unexpected AttnRes block-write layers: {block_write_layers}"
            )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in block_model.parameters()
        )
        record = {
            "block_model_class": type(block_model).__name__,
            "block_write_layers": block_write_layers,
            "dcp_size": get_dcp_group().world_size,
            "device": current_platform.get_device_name(local_rank),
            "ep_size": get_ep_group().world_size,
            "git_commit": run_commit,
            "layer_indices": layer_indices,
            "layer_types": layer_types,
            "local_rank": local_rank,
            "memory_after": memory_after,
            "memory_before": memory_before,
            "mm_projector": projector_metadata,
            "model_class": type(model).__name__,
            "parameter_bytes": parameter_bytes,
            "parameter_count": sum(
                parameter.numel() for parameter in block_model.parameters()
            ),
            "peak_allocated_mib": (torch.accelerator.max_memory_allocated() / 2**20),
            "rank": rank,
            "requested_all2all_backend": config.all2all_backend,
            "smoke_scope": "real_model_construction",
            "status": "PASS",
            "tp_size": get_tp_group().world_size,
            "vision_tower": vision_metadata,
            "weight_source": "dummy",
            "world_size": world_size,
        }
        failure_stage = "final_barrier"
        dist.barrier()
        print(json.dumps(record, sort_keys=True), flush=True)
    except Exception as error:
        failure_record = {
            "error_message": str(error),
            "exception_type": type(error).__name__,
            "failure_stage": failure_stage,
            "git_commit": run_commit,
            "local_rank": local_rank,
            "rank": rank,
            "smoke_scope": "real_model_construction",
            "status": "FAIL",
            "world_size": world_size,
        }
        print(json.dumps(failure_record, sort_keys=True), flush=True)
        raise
    finally:
        layers = None
        block_model = None
        model = None
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
