# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import json
import os
import subprocess
from typing import Any

from benchmarks.kimi_k3_layer_profiling.config import BenchmarkConfig
from benchmarks.kimi_k3_layer_profiling.distributed import benchmark_config_digest
from benchmarks.kimi_k3_layer_profiling.model_construction import (
    engine_args_kwargs,
    git_commit,
    memory_mib,
    validate_model_construction_config,
)

_TORCHRUN_ENV = {"LOCAL_RANK", "RANK", "WORLD_SIZE"}
_ALL2ALL_MANAGER_CLASSES = {
    "allgather_reducescatter": "AgRsAll2AllManager",
    "naive": "AgRsAll2AllManager",
    "deepep_high_throughput": "DeepEPHTAll2AllManager",
    "deepep_low_latency": "DeepEPLLAll2AllManager",
    "mori_high_throughput": "MoriAll2AllManager",
    "mori_low_latency": "MoriAll2AllManager",
    "deepep_v2": "DeepEPV2All2AllManager",
    "nixl_ep": "NixlEPAll2AllManager",
    "flashinfer_all2allv": "FlashInferNVLinkTwoSidedManager",
    "flashinfer_nvlink_two_sided": "FlashInferNVLinkTwoSidedManager",
    "flashinfer_nvlink_one_sided": "FlashInferNVLinkOneSidedManager",
}


def ensure_tracked_worktree_clean() -> None:
    for command, description in (
        (["git", "diff", "--quiet"], "tracked worktree"),
        (["git", "diff", "--cached", "--quiet"], "Git index"),
    ):
        result = subprocess.run(command, check=False)
        if result.returncode == 1:
            raise RuntimeError(
                f"The {description} has uncommitted changes; commit them before "
                "running forward smoke"
            )
        if result.returncode != 0:
            raise RuntimeError(f"Could not inspect the {description}")


def expected_all2all_manager_class(backend: str) -> str:
    try:
        return _ALL2ALL_MANAGER_CLASSES[backend]
    except KeyError as error:
        raise ValueError(
            f"Forward smoke cannot validate all2all backend {backend!r}"
        ) from error


def forward_engine_args_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    kwargs = engine_args_kwargs(config)
    kwargs["enable_prompt_embeds"] = True
    return kwargs


def validate_forward_smoke_config(config: BenchmarkConfig) -> None:
    validate_model_construction_config(config)
    if config.execution_mode != "eager":
        raise ValueError("Forward smoke requires execution_mode=eager")
    if config.phase != "prefill":
        raise ValueError("The first forward smoke supports prefill only")
    if config.context_len != config.query_len:
        raise ValueError(
            "The first forward smoke requires a fresh prefill with "
            "context_len=query_len"
        )
    if config.cache_mode != "none":
        raise ValueError("Fresh-prefill forward smoke requires cache_mode=none")


def _layer_index(layer_name: str) -> int | None:
    parts = layer_name.split(".")
    try:
        return int(parts[parts.index("layers") + 1])
    except (ValueError, IndexError):
        return None


def _spec_value(spec, name: str):
    try:
        value = getattr(spec, name)
    except AttributeError:
        return None
    except Exception as error:
        return f"<{type(error).__name__}: {error}>"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return str(value)


def kv_cache_spec_evidence(kv_cache_spec: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for layer_name, spec in kv_cache_spec.items():
        spec_type = type(spec).__name__
        cache_kind = {
            "MambaSpec": "KDA",
            "MLAAttentionSpec": "MLA",
        }.get(spec_type, "unknown")
        records.append(
            {
                "block_size": _spec_value(spec, "block_size"),
                "cache_kind": cache_kind,
                "dtypes": _spec_value(spec, "dtypes"),
                "layer_index": _layer_index(layer_name),
                "layer_name": layer_name,
                "num_heads": _spec_value(spec, "num_heads"),
                "num_states": _spec_value(spec, "num_states"),
                "page_size_bytes": _spec_value(spec, "page_size_bytes"),
                "page_size_padded": _spec_value(spec, "page_size_padded"),
                "shapes": _spec_value(spec, "shapes"),
                "spec_type": spec_type,
                "state_content_size_bytes": _spec_value(
                    spec, "state_content_size_bytes"
                ),
                "tokens_per_state": _spec_value(spec, "tokens_per_state"),
            }
        )
    return sorted(
        records,
        key=lambda record: (
            record["layer_index"] is None,
            record["layer_index"] if record["layer_index"] is not None else 0,
            record["layer_name"],
        ),
    )


def _initialize_kv_cache(runner, config: BenchmarkConfig, rank: int):
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_config_from_groups,
        get_kv_cache_groups,
    )

    kv_cache_spec = runner.get_kv_cache_spec()
    print(
        json.dumps(
            {
                "cache_specs": kv_cache_spec_evidence(kv_cache_spec),
                "rank": rank,
                "smoke_scope": "real_block_forward",
                "stage": "raw_kv_cache_specs",
                "status": "DIAGNOSTIC",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    kv_cache_groups = get_kv_cache_groups(runner.vllm_config, kv_cache_spec)
    if not kv_cache_groups:
        raise RuntimeError("Kimi-K3 block did not register KV cache layers")
    widths = [
        group.kv_cache_spec.max_num_blocks_per_req(
            runner.vllm_config, config.context_len
        )
        for group in kv_cache_groups
    ]
    required_blocks = config.batch_size * max(widths)
    saved_override = runner.cache_config.num_gpu_blocks_override
    runner.cache_config.num_gpu_blocks_override = required_blocks
    try:
        kv_cache_config = get_kv_cache_config_from_groups(
            runner.vllm_config,
            kv_cache_groups,
            available_memory=0,
        )
    finally:
        runner.cache_config.num_gpu_blocks_override = saved_override
    if kv_cache_config.num_blocks != required_blocks:
        raise RuntimeError(
            f"Expected {required_blocks} KV blocks, got {kv_cache_config.num_blocks}"
        )
    runner.initialize_kv_cache(kv_cache_config, is_profiling=True)
    runner.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
    return kv_cache_config, widths


def _prepare_prefill(runner, config: BenchmarkConfig, kv_cache_config, widths):
    import numpy as np
    import torch

    from vllm.config import CUDAGraphMode
    from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
    from vllm.v1.worker.gpu.input_batch import InputBatch

    num_tokens = config.num_scheduled_tokens
    input_batch = InputBatch.make_dummy(
        config.batch_size,
        num_tokens,
        runner.input_buffers,
        max_query_len=config.query_len,
    )
    positions = torch.arange(
        config.query_len,
        dtype=input_batch.positions.dtype,
        device=runner.device,
    ).repeat(config.batch_size)
    input_batch.positions.copy_(positions)
    input_batch.seq_lens.fill_(config.context_len)
    input_batch.seq_lens_cpu_upper_bound.fill_(config.context_len)
    input_batch.num_computed_tokens_np.fill(0)
    input_batch.prefill_len_np.fill(config.context_len)
    input_batch.num_computed_prefill_tokens_np.fill(0)
    input_batch.is_prefilling_np.fill(True)
    input_batch.has_prefill = True
    input_batch.max_seq_len_np = np.full(
        config.batch_size, config.context_len, dtype=np.int32
    )
    input_batch.is_padding.fill_(False)

    for req_index in range(config.batch_size):
        block_ids = tuple(
            list(range(req_index * width, (req_index + 1) * width)) for width in widths
        )
        runner.block_tables.append_block_ids(
            req_index,
            block_ids,
            overwrite=True,
        )
    runner.block_tables.apply_staged_writes()
    block_tables, slot_mappings = runner.prepare_attn(input_batch)
    runner.model_state.preprocess_state(
        input_batch,
        block_tables,
        kv_cache_config,
        runner.req_states.num_computed_tokens.gpu,
    )
    attn_metadata = runner.model_state.prepare_attn(
        input_batch,
        CUDAGraphMode.NONE,
        block_tables,
        slot_mappings,
        runner.attn_groups,
        kv_cache_config,
    )
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )
    inputs_embeds = runner.model_state.dummy_inputs_embeds(num_tokens)
    if inputs_embeds is None:
        raise RuntimeError("Model runner did not allocate prompt-embedding inputs")
    generator = torch.Generator(device=runner.device)
    generator.manual_seed(config.random_seed)
    inputs_embeds.uniform_(-0.02, 0.02, generator=generator)
    return input_batch, attn_metadata, slot_mappings_by_layer, inputs_embeds


def _tensor_evidence(tensor) -> dict[str, Any]:
    import torch

    finite = bool(torch.isfinite(tensor).all().item())
    if not finite:
        raise RuntimeError("Block forward produced NaN or Inf")
    values = tensor.float()
    return {
        "abs_max": float(values.abs().max().item()),
        "dtype": str(tensor.dtype),
        "mean": float(values.mean().item()),
        "shape": list(tensor.shape),
    }


def _validate_forward_output(output, inputs_embeds, runner, config: BenchmarkConfig):
    from benchmarks.kimi_k3_layer_profiling.block_model import BlockForwardOutput
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        get_ep_all2all_manager,
    )

    if not isinstance(output, BlockForwardOutput):
        raise RuntimeError(f"Expected BlockForwardOutput, got {type(output).__name__}")
    block_model = runner.model.language_model.model
    local_tokens = config.num_scheduled_tokens
    if block_model.use_sequence_parallel:
        local_tokens = (
            config.num_scheduled_tokens + config.tensor_parallel_size - 1
        ) // config.tensor_parallel_size
    expected_shapes = {
        "block_residual_bank": [local_tokens, 1, config.hidden_size],
        "pending_hidden_states": [local_tokens, config.hidden_size],
        "prefix_sum": [local_tokens, config.hidden_size],
    }
    output_evidence = {
        name: _tensor_evidence(getattr(output, name)) for name in expected_shapes
    }
    input_evidence = _tensor_evidence(inputs_embeds)
    actual_shapes = {
        name: evidence["shape"] for name, evidence in output_evidence.items()
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            f"Unexpected block output shapes: {actual_shapes}, "
            f"expected {expected_shapes}"
        )
    if input_evidence["shape"] != list(config.packed_shape):
        raise RuntimeError(
            f"Unexpected input shape: {input_evidence['shape']}, "
            f"expected {list(config.packed_shape)}"
        )
    expected_dtype = str(runner.dtype)
    output_dtypes = {
        name: evidence["dtype"] for name, evidence in output_evidence.items()
    }
    if input_evidence["dtype"] != expected_dtype or any(
        dtype != expected_dtype for dtype in output_dtypes.values()
    ):
        raise RuntimeError(
            f"Unexpected input/output dtypes: input={input_evidence['dtype']}, "
            f"output={output_dtypes}, expected={expected_dtype}"
        )
    zero_outputs = [
        name for name, evidence in output_evidence.items() if evidence["abs_max"] == 0.0
    ]
    if zero_outputs:
        raise RuntimeError(f"Block forward produced all-zero outputs: {zero_outputs}")

    all2all_manager = get_ep_all2all_manager()
    actual_manager_class = type(all2all_manager).__name__
    expected_manager_class = expected_all2all_manager_class(config.all2all_backend)
    if actual_manager_class != expected_manager_class:
        raise RuntimeError(
            f"Requested all2all backend {config.all2all_backend!r} requires "
            f"{expected_manager_class}, got {actual_manager_class}"
        )
    return block_model, input_evidence, output_evidence, actual_manager_class


def _print_stage(rank: int, stage: str) -> None:
    print(
        json.dumps(
            {
                "rank": rank,
                "smoke_scope": "real_block_forward",
                "stage": stage,
                "status": "RUNNING",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def align_cache_block_size(vllm_config, rank: int, current_platform) -> None:
    cache_config = vllm_config.cache_config
    before = {
        "block_size": cache_config.block_size,
        "mamba_page_size_padded": cache_config.mamba_page_size_padded,
    }
    current_platform.update_block_size_for_backend(vllm_config)
    after = {
        "block_size": cache_config.block_size,
        "mamba_page_size_padded": cache_config.mamba_page_size_padded,
    }
    print(
        json.dumps(
            {
                "after": after,
                "before": before,
                "rank": rank,
                "smoke_scope": "real_block_forward",
                "stage": "cache_backend_alignment",
                "status": "DIAGNOSTIC",
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_forward_smoke(config: BenchmarkConfig) -> None:
    validate_forward_smoke_config(config)
    ensure_tracked_worktree_clean()
    missing = sorted(_TORCHRUN_ENV - os.environ.keys())
    if missing:
        raise RuntimeError(
            "Forward smoke must be launched with torchrun; missing "
            + ", ".join(missing)
        )

    import torch
    import torch.distributed as dist

    from vllm.config import CUDAGraphMode, set_current_vllm_config
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
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import set_random_seed
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.workspace import init_workspace_manager

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
            f"LOCAL_RANK={local_rank} exceeds visible accelerator count "
            f"{torch.accelerator.device_count()}"
        )

    torch.accelerator.set_device_index(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")
    initialized = False
    runner = None
    output = None
    input_batch = None
    inputs_embeds = None
    attn_metadata = None
    slot_mappings = None
    kv_cache_config = None
    block_model = None
    vllm_config = None
    run_commit = None
    failure_stage = "engine_config"
    try:
        run_commit = git_commit()
        vllm_config = EngineArgs(
            **forward_engine_args_kwargs(config)
        ).create_engine_config()
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
            device=device,
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
            set_random_seed(config.random_seed)
            init_workspace_manager(device)
            runner = GPUModelRunner(vllm_config, device)

        torch.accelerator.reset_peak_memory_stats()
        memory_before = memory_mib(torch)
        failure_stage = "dummy_model_loading"
        _print_stage(rank, failure_stage)
        with set_current_vllm_config(vllm_config):
            runner.load_model(load_dummy_weights=True)
        torch.accelerator.synchronize()
        memory_after_load = memory_mib(torch)

        failure_stage = "cache_backend_alignment"
        _print_stage(rank, failure_stage)
        align_cache_block_size(vllm_config, rank, current_platform)

        failure_stage = "kv_cache_init"
        _print_stage(rank, failure_stage)
        kv_cache_config, widths = _initialize_kv_cache(runner, config, rank)
        failure_stage = "input_and_metadata"
        _print_stage(rank, failure_stage)
        input_batch, attn_metadata, slot_mappings, inputs_embeds = _prepare_prefill(
            runner,
            config,
            kv_cache_config,
            widths,
        )
        runner.eplb.prepare_forward(
            runner.model_config,
            input_batch.num_tokens,
        )

        failure_stage = "block_forward"
        _print_stage(rank, failure_stage)
        batch_descriptor = BatchDescriptor(
            num_tokens=config.num_scheduled_tokens,
            num_reqs=config.batch_size,
            uniform=True,
        )
        with (
            torch.inference_mode(),
            set_current_vllm_config(vllm_config),
            set_forward_context(
                attn_metadata,
                vllm_config,
                num_tokens=config.num_scheduled_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=batch_descriptor,
                slot_mapping=slot_mappings,
                is_padding=input_batch.is_padding,
            ),
        ):
            output = runner.model(
                input_ids=None,
                positions=input_batch.positions,
                intermediate_tensors=None,
                inputs_embeds=inputs_embeds,
            )
        torch.accelerator.synchronize()
        _print_stage(rank, "block_forward_complete")
        memory_after_forward = memory_mib(torch)

        failure_stage = "output_validation"
        validation = None
        validation_error = None
        try:
            validation = _validate_forward_output(output, inputs_embeds, runner, config)
        except Exception as error:
            validation_error = f"{type(error).__name__}: {error}"
        validation_errors = [None] * world_size
        dist.all_gather_object(validation_errors, validation_error)
        if any(error is not None for error in validation_errors):
            failures = {
                failed_rank: error
                for failed_rank, error in enumerate(validation_errors)
                if error is not None
            }
            raise RuntimeError(f"Forward output validation failed: {failures}")
        assert validation is not None
        block_model, input_evidence, output_evidence, actual_manager_class = validation
        record = {
            "actual_all2all_manager": actual_manager_class,
            "batch_size": config.batch_size,
            "cache_group_count": len(kv_cache_config.kv_cache_groups),
            "context_len": config.context_len,
            "dcp_size": get_dcp_group().world_size,
            "device": current_platform.get_device_name(local_rank),
            "ep_size": get_ep_group().world_size,
            "execution_mode": "eager",
            "git_commit": run_commit,
            "input": input_evidence,
            "kv_cache_blocks": kv_cache_config.num_blocks,
            "local_rank": local_rank,
            "memory_after_forward": memory_after_forward,
            "memory_after_load": memory_after_load,
            "memory_before": memory_before,
            "output": output_evidence,
            "peak_allocated_mib": (torch.accelerator.max_memory_allocated() / 2**20),
            "phase": config.phase,
            "query_len": config.query_len,
            "rank": rank,
            "requested_all2all_backend": config.all2all_backend,
            "smoke_scope": "real_block_forward",
            "status": "PASS",
            "tp_size": get_tp_group().world_size,
            "use_sequence_parallel": block_model.use_sequence_parallel,
            "weight_source": "dummy",
            "world_size": world_size,
        }
        print(json.dumps(record, sort_keys=True), flush=True)
    except Exception as error:
        failure_record = {
            "error_message": str(error),
            "exception_type": type(error).__name__,
            "failure_stage": failure_stage,
            "git_commit": run_commit,
            "local_rank": local_rank,
            "rank": rank,
            "smoke_scope": "real_block_forward",
            "status": "FAIL",
            "world_size": world_size,
        }
        print(json.dumps(failure_record, sort_keys=True), flush=True)
        raise
    finally:
        output = None
        input_batch = None
        inputs_embeds = None
        attn_metadata = None
        slot_mappings = None
        kv_cache_config = None
        block_model = None
        if runner is not None:
            if hasattr(runner.model_state, "_mamba_ctx"):
                runner.model_state._mamba_ctx = None
            if hasattr(runner, "kv_caches"):
                runner.kv_caches.clear()
            if hasattr(runner, "attn_groups"):
                runner.attn_groups.clear()
            runner.cudagraph_manager = None
        if vllm_config is not None:
            vllm_config.compilation_config.static_forward_context.clear()
        runner = None
        gc.collect()
        if torch.accelerator.is_available():
            torch.accelerator.empty_cache()
        if initialized:
            destroy_model_parallel()
            destroy_distributed_environment()
