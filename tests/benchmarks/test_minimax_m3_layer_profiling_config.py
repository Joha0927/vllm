# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.minimax_m3_layer_profiling.config import (
    load_text_config,
    load_yaml,
    manifest,
    parse_config,
)
from benchmarks.minimax_m3_layer_profiling.production_profile import (
    _validate_resolved_block_size,
    engine_kwargs,
)
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

ROOT = Path(__file__).parents[2]
TP1_CONFIG = (
    ROOT / "benchmarks/minimax_m3_layer_profiling/shapes/"
    "prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml"
)
TP2_CONFIG = (
    ROOT / "benchmarks/minimax_m3_layer_profiling/shapes/"
    "prefill_decode_bs32_p4096_tp2_dp4_ag_rs.yaml"
)
CONFIGS = ((TP1_CONFIG, 1, 8), (TP2_CONFIG, 2, 4))
MODEL_CONFIG = ROOT / "benchmarks/minimax_m3_layer_profiling/model_config/config.json"


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize(("path", "tp", "dp"), CONFIGS)
def test_first_four_layers_cover_dense_and_sparse_moe(
    path: Path, tp: int, dp: int
) -> None:
    config = parse_config(load_yaml(path))
    result = manifest(config)

    assert config.tensor_parallel_size == tp
    assert config.data_parallel_size == dp
    assert config.expert_parallel_size == 8
    assert config.block_size == 128
    assert result["expected_layer_range"] == [0, 3]
    assert result["layer_types"] == [
        {"layer": 0, "attention": "full", "ffn": "dense"},
        {"layer": 1, "attention": "full", "ffn": "dense"},
        {"layer": 2, "attention": "full", "ffn": "dense"},
        {"layer": 3, "attention": "sparse", "ffn": "MoE"},
    ]


def test_local_model_config_is_a_complete_official_asset() -> None:
    with MODEL_CONFIG.open(encoding="utf-8") as file:
        model_config = json.load(file)
    text_config = load_text_config()
    defaults = MiniMaxM3TextConfig()

    assert model_config["architectures"] == ["MiniMaxM3SparseForConditionalGeneration"]
    assert model_config["model_type"] == "minimax_m3_vl"
    assert model_config["vision_config"]["model_type"] == "clip_vision_model"
    assert model_config["torch_dtype"] == "bfloat16"
    assert text_config["architectures"] == ["MiniMaxM3SparseForCausalLM"]
    assert text_config["hidden_size"] == defaults.hidden_size == 6144
    assert text_config["max_position_embeddings"] == 1048576
    assert text_config["num_local_experts"] == defaults.num_local_experts == 128
    assert text_config["num_experts_per_tok"] == defaults.num_experts_per_tok == 4
    assert text_config["num_mtp_modules"] == 7
    assert text_config["moe_layer_freq"] == defaults.moe_layer_freq
    assert (
        text_config["sparse_attention_config"]["sparse_attention_freq"]
        == defaults.sparse_attention_config["sparse_attention_freq"]
    )


@pytest.mark.parametrize(("path", "tp", "dp"), CONFIGS)
def test_production_args_keep_real_model_path_and_cut_only_layer_count(
    path: Path, tp: int, dp: int
) -> None:
    config = parse_config(load_yaml(path))
    kwargs = engine_kwargs(config)

    assert kwargs["hf_overrides"] == {"text_config": {"num_hidden_layers": 4}}
    assert kwargs["language_model_only"] is True
    assert kwargs["load_format"] == "dummy"
    assert kwargs["block_size"] == 128
    assert kwargs["tensor_parallel_size"] == tp
    assert kwargs["data_parallel_size"] == dp
    assert kwargs["max_num_seqs"] == 32 // dp
    assert kwargs["max_num_batched_tokens"] == (32 // dp) * 4096
    assert kwargs["max_model_len"] == 4098
    assert "profiler_config" not in kwargs


def test_manager_block_size_matches_sparse_kernel_block_size() -> None:
    data = load_yaml(TP1_CONFIG)
    data["block_size"] = 16

    with pytest.raises(
        ValueError,
        match="block_size must equal MiniMax M3 sparse_block_size \\(128\\)",
    ):
        parse_config(data)


def test_resolved_manager_block_size_is_checked_after_engine_init() -> None:
    def fake_llm(block_size: int) -> SimpleNamespace:
        cache_config = SimpleNamespace(block_size=block_size)
        vllm_config = SimpleNamespace(cache_config=cache_config)
        return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=vllm_config))

    assert _validate_resolved_block_size(fake_llm(128), 128) == 128
    with pytest.raises(
        RuntimeError,
        match="expected resolved KV manager block size 128, got 16",
    ):
        _validate_resolved_block_size(fake_llm(16), 128)


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
