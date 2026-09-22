# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

MODEL_DIR = "benchmarks/glm52_layer_profiling/model_config"
NUM_LAYERS = 7
DSA_BLOCK_SIZE = 64


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str
    batch_size: int
    prompt_len: int
    tensor_parallel_size: int
    data_parallel_size: int
    enable_expert_parallel: bool
    all2all_backend: str
    block_size: int
    kv_cache_memory_bytes: int
    warmup_iters: int
    profile_iters: int
    profile: str
    profile_output_dir: str | None
    profiler_with_stack: bool
    gpu_count: int
    random_seed: int
    num_layers: int = NUM_LAYERS

    @property
    def local_batch_size(self) -> int:
        return self.batch_size // self.data_parallel_size

    @property
    def max_tokens(self) -> int:
        return 2

    @property
    def max_model_len(self) -> int:
        return self.prompt_len + self.max_tokens

    @property
    def expert_parallel_size(self) -> int:
        if not self.enable_expert_parallel:
            return 1
        return self.tensor_parallel_size * self.data_parallel_size


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("expected a YAML mapping")
    return data


def load_model_config() -> dict[str, Any]:
    with (Path(MODEL_DIR) / "config.json").open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("expected a model config mapping")
    return data


def parse_config(data: dict[str, Any]) -> BenchmarkConfig:
    required = {"batch_size", "prompt_len"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    defaults = {
        "all2all_backend": "allgather_reducescatter",
        "block_size": DSA_BLOCK_SIZE,
        "data_parallel_size": 8,
        "enable_expert_parallel": True,
        "gpu_count": 8,
        "kv_cache_memory_bytes": 4 * 1024**3,
        "profile": "none",
        "profile_iters": 1,
        "profile_output_dir": None,
        "profiler_with_stack": True,
        "random_seed": 0,
        "tensor_parallel_size": 1,
        "warmup_iters": 3,
    }
    unsupported = sorted(data.keys() - (required | defaults.keys()))
    if unsupported:
        raise ValueError(f"unsupported config fields: {', '.join(unsupported)}")
    values = {**defaults, **data}
    config = BenchmarkConfig(
        model=MODEL_DIR,
        batch_size=int(values["batch_size"]),
        prompt_len=int(values["prompt_len"]),
        tensor_parallel_size=int(values["tensor_parallel_size"]),
        data_parallel_size=int(values["data_parallel_size"]),
        enable_expert_parallel=_require_bool(
            values["enable_expert_parallel"], "enable_expert_parallel"
        ),
        all2all_backend=str(values["all2all_backend"]),
        block_size=int(values["block_size"]),
        kv_cache_memory_bytes=int(values["kv_cache_memory_bytes"]),
        warmup_iters=int(values["warmup_iters"]),
        profile_iters=int(values["profile_iters"]),
        profile=str(values["profile"]),
        profile_output_dir=(
            None
            if values["profile_output_dir"] is None
            else str(values["profile_output_dir"])
        ),
        profiler_with_stack=_require_bool(
            values["profiler_with_stack"], "profiler_with_stack"
        ),
        gpu_count=int(values["gpu_count"]),
        random_seed=int(values["random_seed"]),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    for name in (
        "batch_size",
        "prompt_len",
        "tensor_parallel_size",
        "data_parallel_size",
        "gpu_count",
        "block_size",
        "kv_cache_memory_bytes",
        "profile_iters",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if config.batch_size % config.data_parallel_size:
        raise ValueError("batch_size must be divisible by data_parallel_size")
    if config.tensor_parallel_size * config.data_parallel_size != config.gpu_count:
        raise ValueError(
            "tensor_parallel_size * data_parallel_size must equal gpu_count"
        )
    if config.profile not in {"none", "torch"}:
        raise ValueError("profile must be none or torch")
    if config.profile == "torch" and not config.profile_output_dir:
        raise ValueError("profile_output_dir is required when profile=torch")
    if config.profile == "none" and config.profile_output_dir:
        raise ValueError("profile_output_dir requires profile=torch")
    if config.profile_iters != 1:
        raise ValueError("prefill_decode requires profile_iters=1")
    if not config.all2all_backend:
        raise ValueError("all2all_backend must be non-empty")
    if config.block_size != DSA_BLOCK_SIZE:
        raise ValueError(
            f"block_size must equal the CUDA DSA indexer block size "
            f"({DSA_BLOCK_SIZE}), got {config.block_size}"
        )

    model_config = load_model_config()
    if model_config["architectures"] != ["GlmMoeDsaForCausalLM"]:
        raise ValueError("expected GlmMoeDsaForCausalLM model architecture")
    quant_config = model_config.get("quantization_config")
    if not isinstance(quant_config, dict) or quant_config.get("quant_method") != "fp8":
        raise ValueError("expected the official FP8 quantization config")
    num_experts = int(model_config["n_routed_experts"])
    if config.enable_expert_parallel and num_experts % config.expert_parallel_size:
        raise ValueError("n_routed_experts must be divisible by expert_parallel_size")
    if int(model_config["num_attention_heads"]) % config.tensor_parallel_size:
        raise ValueError(
            "num_attention_heads must be divisible by tensor_parallel_size"
        )
    for field in ("indexer_types", "mlp_layer_types"):
        if len(model_config[field]) < config.num_layers:
            raise ValueError(f"{field} does not cover all profiled layers")
    for layer in range(config.num_layers):
        if model_config["indexer_types"][layer] != _runtime_indexer_type(
            model_config, layer
        ):
            raise ValueError(
                f"indexer_types disagrees with runtime logic at layer {layer}"
            )
        runtime_ffn = _runtime_ffn_type(model_config, layer)
        expected_mlp = "sparse" if runtime_ffn == "MoE" else "dense"
        if model_config["mlp_layer_types"][layer] != expected_mlp:
            raise ValueError(
                f"mlp_layer_types disagrees with runtime logic at layer {layer}"
            )


def manifest(config: BenchmarkConfig) -> dict[str, Any]:
    model_config = load_model_config()
    return {
        **asdict(config),
        "architecture": "GlmMoeDsaForCausalLM",
        "execution_path": "LLM/EngineCore/production_model",
        "expert_parallel_size": config.expert_parallel_size,
        "expected_layer_range": [0, config.num_layers - 1],
        "experts_per_rank": (
            int(model_config["n_routed_experts"]) // config.expert_parallel_size
        ),
        "layer_types": [
            {
                "layer": layer,
                "indexer": _runtime_indexer_type(model_config, layer),
                "ffn": _runtime_ffn_type(model_config, layer),
            }
            for layer in range(config.num_layers)
        ],
        "max_model_len": config.max_model_len,
        "max_tokens": config.max_tokens,
        "prefill_tokens": config.batch_size * config.prompt_len,
        "quantization": model_config["quantization_config"]["quant_method"],
        "routing_strategy": "uniform_random",
        "weight_source": "dummy",
    }


def _runtime_indexer_type(model_config: dict[str, Any], layer: int) -> str:
    pattern = model_config.get("index_topk_pattern")
    if pattern is not None and 0 <= layer < len(pattern):
        skip_topk = pattern[layer] == "S"
    else:
        offset = int(model_config.get("index_skip_topk_offset", 2))
        frequency = int(model_config.get("index_topk_freq", 1))
        skip_topk = max(layer - offset + 1, 0) % frequency != 0
    return "shared" if skip_topk else "full"


def _runtime_ffn_type(model_config: dict[str, Any], layer: int) -> str:
    has_experts = model_config.get("n_routed_experts") is not None
    first_moe_layer = int(model_config["first_k_dense_replace"])
    moe_frequency = int(model_config.get("moe_layer_freq", 1))
    is_moe = has_experts and layer >= first_moe_layer and layer % moe_frequency == 0
    return "MoE" if is_moe else "dense"


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
