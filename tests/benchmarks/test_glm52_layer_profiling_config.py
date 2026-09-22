# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.glm52_layer_profiling.config import (
    _runtime_ffn_type,
    _runtime_indexer_type,
    load_model_config,
    load_yaml,
    manifest,
    parse_config,
)
from benchmarks.glm52_layer_profiling.production_profile import (
    _model_parameter_evidence,
    _validate_resolved_block_size,
    engine_kwargs,
)
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.worker.utils import select_common_block_size

ROOT = Path(__file__).parents[2]
TP1_CONFIG = (
    ROOT / "benchmarks/glm52_layer_profiling/shapes/"
    "prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml"
)
TP2_CONFIG = (
    ROOT / "benchmarks/glm52_layer_profiling/shapes/"
    "prefill_decode_bs32_p4096_tp2_dp4_ag_rs.yaml"
)
CONFIGS = ((TP1_CONFIG, 1, 8), (TP2_CONFIG, 2, 4))
MODEL_CONFIG = ROOT / "benchmarks/glm52_layer_profiling/model_config/config.json"


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize(("path", "tp", "dp"), CONFIGS)
def test_first_seven_layers_cover_all_production_layer_combinations(
    path: Path, tp: int, dp: int
) -> None:
    config = parse_config(load_yaml(path))
    result = manifest(config)

    assert config.tensor_parallel_size == tp
    assert config.data_parallel_size == dp
    assert config.expert_parallel_size == 8
    assert config.local_batch_size == 32 // dp
    assert config.block_size == 64
    assert result["expected_layer_range"] == [0, 6]
    assert result["experts_per_rank"] == 32
    assert result["layer_types"] == [
        {"layer": 0, "indexer": "full", "ffn": "dense"},
        {"layer": 1, "indexer": "full", "ffn": "dense"},
        {"layer": 2, "indexer": "full", "ffn": "dense"},
        {"layer": 3, "indexer": "shared", "ffn": "MoE"},
        {"layer": 4, "indexer": "shared", "ffn": "MoE"},
        {"layer": 5, "indexer": "shared", "ffn": "MoE"},
        {"layer": 6, "indexer": "full", "ffn": "MoE"},
    ]


def test_local_model_config_is_official_fp8_asset() -> None:
    with MODEL_CONFIG.open(encoding="utf-8") as file:
        raw = json.load(file)
    model_config = load_model_config()

    assert raw == model_config
    assert model_config["architectures"] == ["GlmMoeDsaForCausalLM"]
    assert model_config["model_type"] == "glm_moe_dsa"
    assert model_config["num_hidden_layers"] == 78
    assert model_config["hidden_size"] == 6144
    assert model_config["n_routed_experts"] == 256
    assert model_config["num_experts_per_tok"] == 8
    assert model_config["quantization_config"]["quant_method"] == "fp8"
    assert model_config["quantization_config"]["weight_block_size"] == [128, 128]
    assert "GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs()

    hf_config = get_config(str(MODEL_CONFIG.parent), False)
    assert type(hf_config).__name__ == "GlmMoeDsaConfig"
    assert hf_config.architectures == ["GlmMoeDsaForCausalLM"]
    assert hf_config.quantization_config["quant_method"] == "fp8"


def test_layer_manifest_matches_production_selection_logic() -> None:
    model_config = load_model_config()

    assert [_runtime_indexer_type(model_config, layer) for layer in range(7)] == [
        "full",
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    ]
    assert [_runtime_ffn_type(model_config, layer) for layer in range(7)] == [
        "dense",
        "dense",
        "dense",
        "MoE",
        "MoE",
        "MoE",
        "MoE",
    ]


@pytest.mark.parametrize(("path", "tp", "dp"), CONFIGS)
def test_production_args_keep_fp8_config_and_cut_only_layer_count(
    path: Path, tp: int, dp: int
) -> None:
    config = parse_config(load_yaml(path))
    kwargs = engine_kwargs(config)

    assert kwargs["hf_overrides"] == {"num_hidden_layers": 7}
    assert kwargs["load_format"] == "dummy"
    assert kwargs["block_size"] == 64
    assert kwargs["tensor_parallel_size"] == tp
    assert kwargs["data_parallel_size"] == dp
    assert kwargs["max_num_seqs"] == 32 // dp
    assert kwargs["max_num_batched_tokens"] == (32 // dp) * 4096
    assert kwargs["max_model_len"] == 4098
    assert "quantization" not in kwargs
    assert "speculative_config" not in kwargs
    assert "profiler_config" not in kwargs


def test_cuda_dsa_manager_block_size_is_fixed() -> None:
    data = load_yaml(TP1_CONFIG)
    data["block_size"] = 16

    with pytest.raises(
        ValueError,
        match=r"block_size must equal the CUDA DSA indexer block size \(64\)",
    ):
        parse_config(data)


def test_block_size_64_is_common_to_indexer_and_mla() -> None:
    class GenericMlaBackend:
        @staticmethod
        def get_supported_kernel_block_sizes() -> list[MultipleOf]:
            return [MultipleOf(16)]

    backends = [DeepseekV32IndexerBackend, GenericMlaBackend]
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes() == [64]
    assert select_common_block_size(64, backends) == 64
    with pytest.raises(ValueError, match="No common block size for 16"):
        select_common_block_size(16, backends)


def test_resolved_manager_block_size_is_checked_after_engine_init() -> None:
    def fake_llm(block_size: int) -> SimpleNamespace:
        cache_config = SimpleNamespace(block_size=block_size)
        vllm_config = SimpleNamespace(cache_config=cache_config)
        return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vllm_config))

    assert _validate_resolved_block_size(fake_llm(64), 64) == 64
    with pytest.raises(
        RuntimeError,
        match="expected resolved KV manager block size 64, got 16",
    ):
        _validate_resolved_block_size(fake_llm(16), 64)


def test_parameter_evidence_counts_runtime_storage() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(3, 2, bias=False, dtype=torch.float32),
    )

    result = _model_parameter_evidence(model)

    assert result["parameter_count"] == 18
    assert result["parameter_bytes"] == 48
    assert result["buffer_count"] == 0
    assert result["buffer_bytes"] == 0
    assert result["persistent_tensor_bytes"] == 48
    assert result["model_class"] == "Sequential"
    assert result["model_module"] == "torch.nn.modules.container"
    assert result["parameter_bytes_by_dtype"] == {
        "torch.bfloat16": 24,
        "torch.float32": 24,
    }


@pytest.mark.parametrize(("path", "_tp", "_dp"), CONFIGS)
def test_torch_profile_is_opt_in(
    tmp_path: Path, path: Path, _tp: int, _dp: int
) -> None:
    config = parse_config(load_yaml(path))
    profiled = replace(
        config,
        profile="torch",
        profile_output_dir=str(tmp_path / "traces"),
    )
    kwargs = engine_kwargs(profiled)

    assert kwargs["enable_layerwise_nvtx_tracing"] is True
    assert kwargs["profiler_config"].torch_profiler_with_stack is True
    assert kwargs["profiler_config"].torch_profiler_record_shapes is True
