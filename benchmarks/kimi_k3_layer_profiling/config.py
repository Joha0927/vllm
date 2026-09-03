# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

_WORKLOADS = {"full_prefill", "prefill_decode"}
_PROFILES = {"none", "torch"}
_EXECUTION_MODES = {"eager", "cudagraph"}
_MODEL_CONFIG_DIR = "benchmarks/kimi_k3_layer_profiling/model_config"
_LOGICAL_START_LAYER = 0
_NUM_LAYERS = 12
_REQUIRED_CONFIG_FIELDS = {
    "batch_size",
    "history_len",
    "query_len",
    "workload",
}
_CONFIG_DEFAULTS: dict[str, Any] = {
    "all2all_backend": "allgather_reducescatter",
    "attention_backend": "auto",
    "data_parallel_size": 1,
    "decode_context_parallel_size": 1,
    "enable_dbo": False,
    "enable_expert_parallel": True,
    "execution_mode": "eager",
    "expert_placement_strategy": "linear",
    "gpu_count": 8,
    "kda_prefill_backend": "auto",
    "kv_cache_dtype": "auto",
    "kv_cache_memory_bytes": 4 * 1024**3,
    "linear_backend": "auto",
    "mla_prefill_backend": "auto",
    "moe_backend": "auto",
    "profile": "none",
    "profile_iters": 1,
    "profile_output_dir": None,
    "profiler_with_stack": False,
    "random_seed": 0,
    "routing_strategy": "uniform_random",
    "shard_sp_shared_expert": False,
    "tensor_parallel_size": 8,
    "warmup_iters": 1,
}
_CONFIG_FIELDS = _REQUIRED_CONFIG_FIELDS | _CONFIG_DEFAULTS.keys()
_KDA_PREFILL_BACKENDS = {"auto", "flashkda", "triton"}
_MLA_PREFILL_BACKENDS = {
    "auto",
    "flash_attn",
    "flashinfer",
    "tokenspeed_mla",
    "trtllm_ragged",
}
_EXPERT_PLACEMENT_STRATEGIES = {"linear", "round_robin"}


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
    workload: str
    batch_size: int
    history_len: int
    query_len: int
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
    expert_placement_strategy: str
    enable_dbo: bool
    moe_backend: str
    linear_backend: str
    attention_backend: str
    kda_prefill_backend: str
    mla_prefill_backend: str
    kv_cache_dtype: str
    kv_cache_memory_bytes: int
    shard_sp_shared_expert: bool
    execution_mode: str
    routing_strategy: str
    warmup_iters: int
    profile_iters: int
    profile: str
    profile_output_dir: str | None
    profiler_with_stack: bool
    gpu_count: int
    random_seed: int

    @property
    def num_scheduled_tokens(self) -> int:
        return self.batch_size * self.query_len

    @property
    def prompt_len(self) -> int:
        if self.workload == "prefill_decode":
            return self.history_len
        return self.query_len

    @property
    def max_tokens(self) -> int:
        return 2 if self.workload == "prefill_decode" else 1

    @property
    def max_model_len(self) -> int:
        return self.prompt_len + self.max_tokens

    @property
    def prefill_tokens(self) -> int:
        return self.batch_size * self.prompt_len

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

    @property
    def local_batch_size(self) -> int:
        return self.batch_size // self.data_parallel_size


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
            "attention_backend": config.attention_backend,
            "batch_size": config.batch_size,
            "context_lengths": [config.prompt_len] * config.batch_size,
            "data_parallel_size": config.data_parallel_size,
            "decode_context_parallel_size": config.decode_context_parallel_size,
            "dtype": config.dtype,
            "enable_dbo": config.enable_dbo,
            "enable_expert_parallel": config.enable_expert_parallel,
            "expert_parallel_size": config.expert_parallel_size,
            "expert_placement_strategy": config.expert_placement_strategy,
            "git_commit": None,
            "gpu_count": config.gpu_count,
            "input_shape": list(config.input_shape),
            "logical_layer_range": [
                config.logical_start_layer,
                config.logical_start_layer + config.num_layers - 1,
            ],
            "logical_start_layer": config.logical_start_layer,
            "local_batch_size": config.local_batch_size,
            "kda_prefill_backend": config.kda_prefill_backend,
            "kv_cache_dtype": config.kv_cache_dtype,
            "kv_cache_memory_bytes": config.kv_cache_memory_bytes,
            "linear_backend": config.linear_backend,
            "execution_path": "LLM/EngineCore/production_model",
            "measurement_fidelity": "production-path/shape/backend-faithful",
            "model": "moonshotai/Kimi-K3",
            "model_config_source": str(Path(config.model) / "config.json"),
            "mla_prefill_backend": config.mla_prefill_backend,
            "moe_backend": config.moe_backend,
            "num_profiled_layers": config.num_layers,
            "num_scheduled_tokens": config.num_scheduled_tokens,
            "prefill_tokens": config.prefill_tokens,
            "packed_shape": list(config.packed_shape),
            "history_len": config.history_len,
            "max_model_len": config.max_model_len,
            "max_tokens": config.max_tokens,
            "profile": config.profile,
            "profile_output_dir": config.profile_output_dir,
            "profile_iters": config.profile_iters,
            "profiler_with_stack": config.profiler_with_stack,
            "profiling_unit": "block",
            "query_lengths": [config.query_len] * config.batch_size,
            "random_seed": config.random_seed,
            "routing_strategy": config.routing_strategy,
            "shard_sp_shared_expert": config.shard_sp_shared_expert,
            "tensor_parallel_size": config.tensor_parallel_size,
            "warmup_iters": config.warmup_iters,
            "weight_format": config.weight_format,
            "weight_source": "dummy",
            "workload": config.workload,
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
    missing = sorted(_REQUIRED_CONFIG_FIELDS - data.keys())
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    model_config = load_model_config(_MODEL_CONFIG_DIR)
    text_config = load_model_text_config(_MODEL_CONFIG_DIR)
    quantization_config = text_config.get("quantization_config")
    if not isinstance(quantization_config, dict):
        raise ValueError("The model config does not define quantization_config")

    values = {**_CONFIG_DEFAULTS, **data}
    config = BenchmarkConfig(
        model=_MODEL_CONFIG_DIR,
        workload=str(values["workload"]),
        batch_size=int(values["batch_size"]),
        history_len=int(values["history_len"]),
        query_len=int(values["query_len"]),
        hidden_size=int(text_config["hidden_size"]),
        dtype=str(model_config["dtype"]),
        weight_format=str(quantization_config["format"]),
        logical_start_layer=_LOGICAL_START_LAYER,
        num_layers=_NUM_LAYERS,
        tensor_parallel_size=int(values["tensor_parallel_size"]),
        data_parallel_size=int(values["data_parallel_size"]),
        decode_context_parallel_size=int(values["decode_context_parallel_size"]),
        enable_expert_parallel=_require_bool(
            values["enable_expert_parallel"], "enable_expert_parallel"
        ),
        all2all_backend=str(values["all2all_backend"]),
        expert_placement_strategy=str(values["expert_placement_strategy"]),
        enable_dbo=_require_bool(values["enable_dbo"], "enable_dbo"),
        moe_backend=str(values["moe_backend"]),
        linear_backend=str(values["linear_backend"]),
        attention_backend=str(values["attention_backend"]),
        kda_prefill_backend=str(values["kda_prefill_backend"]),
        mla_prefill_backend=str(values["mla_prefill_backend"]),
        kv_cache_dtype=str(values["kv_cache_dtype"]),
        kv_cache_memory_bytes=int(values["kv_cache_memory_bytes"]),
        shard_sp_shared_expert=_require_bool(
            values["shard_sp_shared_expert"], "shard_sp_shared_expert"
        ),
        execution_mode=str(values["execution_mode"]),
        routing_strategy=str(values["routing_strategy"]),
        warmup_iters=int(values["warmup_iters"]),
        profile_iters=int(values["profile_iters"]),
        profile=str(values["profile"]),
        profile_output_dir=(
            str(values["profile_output_dir"])
            if values["profile_output_dir"] is not None
            else None
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
    if config.workload not in _WORKLOADS:
        raise ValueError(f"workload must be one of {sorted(_WORKLOADS)}")
    if config.profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}")
    if config.execution_mode not in _EXECUTION_MODES:
        raise ValueError(f"execution_mode must be one of {sorted(_EXECUTION_MODES)}")
    for field_name in (
        "batch_size",
        "query_len",
        "hidden_size",
        "num_layers",
        "tensor_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "kv_cache_memory_bytes",
        "gpu_count",
    ):
        if getattr(config, field_name) <= 0:
            raise ValueError(f"{field_name} must be positive")
    if config.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if config.profile_iters <= 0:
        raise ValueError("profile_iters must be positive")
    if config.random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    if config.history_len < 0:
        raise ValueError("history_len must be non-negative")
    if config.workload == "full_prefill" and config.history_len != 0:
        raise ValueError("full_prefill requires history_len=0")
    if config.workload == "prefill_decode" and (
        config.history_len <= 0 or config.query_len != 1
    ):
        raise ValueError("prefill_decode requires history_len>0 and query_len=1")
    if config.batch_size % config.data_parallel_size != 0:
        raise ValueError("batch_size must be divisible by data_parallel_size")
    if config.num_layers != 12:
        raise ValueError("Formal block profiling requires exactly 12 layers")
    if config.tensor_parallel_size % config.decode_context_parallel_size != 0:
        raise ValueError(
            "decode_context_parallel_size must divide tensor_parallel_size"
        )
    if config.tensor_parallel_size * config.data_parallel_size != config.gpu_count:
        raise ValueError(
            "tensor_parallel_size * data_parallel_size must equal gpu_count"
        )
    if config.kda_prefill_backend not in _KDA_PREFILL_BACKENDS:
        raise ValueError(
            f"kda_prefill_backend must be one of {sorted(_KDA_PREFILL_BACKENDS)}"
        )
    if config.mla_prefill_backend not in _MLA_PREFILL_BACKENDS:
        raise ValueError(
            f"mla_prefill_backend must be one of {sorted(_MLA_PREFILL_BACKENDS)}"
        )
    if config.expert_placement_strategy not in _EXPERT_PLACEMENT_STRATEGIES:
        raise ValueError(
            "expert_placement_strategy must be one of "
            f"{sorted(_EXPERT_PLACEMENT_STRATEGIES)}"
        )
    for field_name in (
        "all2all_backend",
        "attention_backend",
        "kv_cache_dtype",
        "linear_backend",
        "moe_backend",
        "routing_strategy",
    ):
        if not getattr(config, field_name):
            raise ValueError(f"{field_name} must be non-empty")
    if config.moe_backend == "deep_gemm_mega_moe":
        if not config.enable_expert_parallel:
            raise ValueError("deep_gemm_mega_moe requires expert parallel")
        text_config = load_model_text_config(config.model)
        if int(text_config["num_experts"]) % config.expert_parallel_size != 0:
            raise ValueError("num_experts must be divisible by expert_parallel_size")
    if config.shard_sp_shared_expert and not (
        config.enable_expert_parallel
        and config.tensor_parallel_size > 1
        and (
            config.data_parallel_size > 1 or config.moe_backend == "deep_gemm_mega_moe"
        )
    ):
        raise ValueError(
            "shard_sp_shared_expert requires sequence-parallel expert execution"
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


def _require_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(data)
    merged.update({key: value for key, value in overrides.items() if value is not None})
    return merged
