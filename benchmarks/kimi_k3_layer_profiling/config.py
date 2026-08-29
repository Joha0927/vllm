# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


_PHASES = {"prefill", "decode"}
_PROFILES = {"none", "torch", "cuda", "proton"}
_EXECUTION_MODES = {"eager", "cudagraph"}
_DIAGNOSTIC_LAYER_COUNTS = {1, 4, 8, 12}
_MODEL_CONFIG_DIR = "benchmarks/kimi_k3_layer_profiling/model_config"
_LOGICAL_START_LAYER = 0
_CONFIG_FIELDS = {
    "all2all_backend",
    "batch_size",
    "cache_mode",
    "context_len",
    "data_parallel_size",
    "decode_context_parallel_size",
    "diagnostic_partial_block",
    "enable_expert_parallel",
    "execution_mode",
    "gpu_count",
    "num_layers",
    "phase",
    "profile",
    "profile_iters",
    "query_len",
    "random_seed",
    "repeat_iters",
    "routing_strategy",
    "tensor_parallel_size",
    "warmup_iters",
}


@dataclass(frozen=True)
class LayerDescription:
    logical_layer: int
    attention_type: str
    ffn_type: str
    attn_res_depth: int
    attn_res_block_write: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    model: str
    phase: str
    batch_size: int
    query_len: int
    context_len: int
    hidden_size: int
    dtype: str
    weight_format: str
    logical_start_layer: int
    num_layers: int
    tensor_parallel_size: int
    data_parallel_size: int
    decode_context_parallel_size: int
    enable_expert_parallel: bool
    all2all_backend: str
    execution_mode: str
    routing_strategy: str
    cache_mode: str
    warmup_iters: int
    profile_iters: int
    repeat_iters: int
    profile: str
    gpu_count: int
    random_seed: int
    diagnostic_partial_block: bool = False

    @property
    def num_scheduled_tokens(self) -> int:
        return self.batch_size * self.query_len

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.query_len, self.hidden_size)

    @property
    def packed_shape(self) -> tuple[int, int]:
        return (self.num_scheduled_tokens, self.hidden_size)

    @property
    def expert_parallel_size(self) -> int:
        if not self.enable_expert_parallel:
            return 1
        return self.tensor_parallel_size * self.data_parallel_size


@dataclass(frozen=True)
class DryRunResult:
    config: BenchmarkConfig
    layers: tuple[LayerDescription, ...]

    def to_dict(self) -> dict[str, Any]:
        config = asdict(self.config)
        config.update(
            {
                "input_shape": list(self.config.input_shape),
                "packed_shape": list(self.config.packed_shape),
                "num_scheduled_tokens": self.config.num_scheduled_tokens,
                "expert_parallel_size": self.config.expert_parallel_size,
            }
        )
        return {
            "config": config,
            "layers": [asdict(layer) for layer in self.layers],
            "manifest": self.manifest(),
            "validation": "PASS",
        }

    def manifest(self) -> dict[str, Any]:
        config = self.config
        return {
            "all2all_backend": config.all2all_backend,
            "applies_model_output_attn_res": False,
            "batch_size": config.batch_size,
            "block_output_contract": [
                "pending_hidden_states",
                "prefix_sum",
                "block_residual_bank",
            ],
            "cache_mode": config.cache_mode,
            "context_lengths": [config.context_len] * config.batch_size,
            "data_parallel_size": config.data_parallel_size,
            "decode_context_parallel_size": config.decode_context_parallel_size,
            "dtype": config.dtype,
            "execution_mode": config.execution_mode,
            "expert_parallel_size": config.expert_parallel_size,
            "git_commit": None,
            "gpu_count": config.gpu_count,
            "input_shape": list(config.input_shape),
            "logical_layer_range": [
                config.logical_start_layer,
                config.logical_start_layer + config.num_layers - 1,
            ],
            "logical_start_layer": config.logical_start_layer,
            "measurement_fidelity": "shape-faithful/backend-faithful",
            "model": "moonshotai/Kimi-K3",
            "model_config_source": str(Path(config.model) / "config.json"),
            "num_profiled_layers": config.num_layers,
            "num_scheduled_tokens": config.num_scheduled_tokens,
            "packed_shape": list(config.packed_shape),
            "phase": config.phase,
            "physical_layer_index": config.logical_start_layer,
            "physical_start_layer": config.logical_start_layer,
            "profile": config.profile,
            "profile_iters": config.profile_iters,
            "profiling_unit": "block",
            "query_lengths": [config.query_len] * config.batch_size,
            "random_seed": config.random_seed,
            "repeat_iters": config.repeat_iters,
            "routing_strategy": config.routing_strategy,
            "tensor_parallel_size": config.tensor_parallel_size,
            "warmup_iters": config.warmup_iters,
            "weight_format": config.weight_format,
            "weight_source": "dummy",
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    return data


def load_model_config(model: str | Path) -> dict[str, Any]:
    config_path = Path(model) / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Local model config does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON mapping in {config_path}")
    return config


def load_model_text_config(model: str | Path) -> dict[str, Any]:
    config_path = Path(model) / "config.json"
    config = load_model_config(model)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        raise ValueError(f"Missing text_config mapping in {config_path}")
    return text_config


def parse_config(data: dict[str, Any]) -> BenchmarkConfig:
    unsupported = sorted(data.keys() - _CONFIG_FIELDS)
    if unsupported:
        raise ValueError(f"Unsupported config fields: {', '.join(unsupported)}")
    required = {"phase", "batch_size", "query_len", "context_len"}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    model_config = load_model_config(_MODEL_CONFIG_DIR)
    text_config = load_model_text_config(_MODEL_CONFIG_DIR)
    quantization_config = text_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("The model config does not define quantization_config")

    config = BenchmarkConfig(
        model=_MODEL_CONFIG_DIR,
        phase=str(data["phase"]),
        batch_size=int(data["batch_size"]),
        query_len=int(data["query_len"]),
        context_len=int(data["context_len"]),
        hidden_size=int(text_config["hidden_size"]),
        dtype=str(model_config["dtype"]),
        weight_format=str(quantization_config["format"]),
        logical_start_layer=_LOGICAL_START_LAYER,
        num_layers=int(data.get("num_layers", 12)),
        tensor_parallel_size=int(data.get("tensor_parallel_size", 8)),
        data_parallel_size=int(data.get("data_parallel_size", 1)),
        decode_context_parallel_size=int(
            data.get("decode_context_parallel_size", 1)
        ),
        enable_expert_parallel=bool(data.get("enable_expert_parallel", True)),
        all2all_backend=str(
            data.get("all2all_backend", "allgather_reducescatter")
        ),
        execution_mode=str(data.get("execution_mode", "eager")),
        routing_strategy=str(data.get("routing_strategy", "uniform_random")),
        cache_mode=str(data.get("cache_mode", "none")),
        warmup_iters=int(data.get("warmup_iters", 1)),
        profile_iters=int(data.get("profile_iters", 1)),
        repeat_iters=int(data.get("repeat_iters", 1)),
        profile=str(data.get("profile", "none")),
        gpu_count=int(data.get("gpu_count", 8)),
        random_seed=int(data.get("random_seed", 0)),
        diagnostic_partial_block=bool(data.get("diagnostic_partial_block", False)),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    if config.phase not in _PHASES:
        raise ValueError(f"phase must be one of {sorted(_PHASES)}")
    if config.profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
    if config.execution_mode not in _EXECUTION_MODES:
        raise ValueError(
            f"execution_mode must be one of {sorted(_EXECUTION_MODES)}"
        )
    for field_name in (
        "batch_size",
        "query_len",
        "context_len",
        "hidden_size",
        "num_layers",
        "tensor_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "gpu_count",
    ):
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.profile_iters <= 0:
        raise ValueError("profile_iters must be positive")
    if config.repeat_iters <= 0:
        raise ValueError("repeat_iters must be positive")
    if config.random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if config.context_len < config.query_len:
        raise ValueError("context_len must be greater than or equal to query_len")
    if config.diagnostic_partial_block:
        if config.num_layers not in _DIAGNOSTIC_LAYER_COUNTS:
            raise ValueError("Diagnostic num_layers must be one of 1, 4, 8, or 12")
    elif config.num_layers != 12:
        raise ValueError("Formal block profiling requires exactly 12 layers")
    if config.tensor_parallel_size % config.decode_context_parallel_size != 0:
        raise ValueError(
            "decode_context_parallel_size must divide tensor_parallel_size"
        )
    if (
        config.tensor_parallel_size * config.data_parallel_size
        != config.gpu_count
    ):
        raise ValueError(
            "tensor_parallel_size * data_parallel_size must equal gpu_count"
        )

    text_config = load_model_text_config(config.model)
    if config.num_layers > int(text_config["num_hidden_layers"]):
        raise ValueError("num_layers exceeds the model configuration")


def describe_layers(config: BenchmarkConfig) -> tuple[LayerDescription, ...]:
    text_config = load_model_text_config(config.model)
    linear_attn_config = text_config.get("linear_attn_config")
    if not isinstance(linear_attn_config, dict):
        raise ValueError("The model config does not define linear_attn_config")
    kda_layers = set(linear_attn_config["kda_layers"])
    first_k_dense_replace = int(text_config["first_k_dense_replace"])
    moe_layer_freq = int(text_config["moe_layer_freq"])
    block_size = int(text_config["attn_res_block_size"])

    layers = []
    for layer_idx in range(
        config.logical_start_layer,
        config.logical_start_layer + config.num_layers,
    ):
        is_moe = (
            text_config.get("num_experts") is not None
            and layer_idx >= first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        )
        layers.append(
            LayerDescription(
                logical_layer=layer_idx,
                attention_type="KDA" if layer_idx + 1 in kda_layers else "MLA",
                ffn_type="MoE" if is_moe else "dense",
                attn_res_depth=(layer_idx + block_size - 1) // block_size,
                attn_res_block_write=layer_idx % block_size == 0,
            )
        )
    return tuple(layers)


def dry_run(data: dict[str, Any]) -> DryRunResult:
    config = parse_config(data)
    return DryRunResult(config=config, layers=describe_layers(config))


def apply_overrides(
    data: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(data)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged
