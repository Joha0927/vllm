# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.minimax_m3_layer_profiling.config import (
    load_text_config,
    load_yaml,
    manifest,
    parse_config,
)
from benchmarks.minimax_m3_layer_profiling.production_profile import engine_kwargs
from vllm.transformers_utils.configs.minimax_m3 import MiniMaxM3TextConfig

ROOT = Path(__file__).parents[2]
CONFIG = (
    ROOT / "benchmarks/minimax_m3_layer_profiling/shapes/"
    "prefill_decode_bs32_p4096_tp1_dp8_ag_rs.yaml"
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def test_first_four_layers_cover_dense_and_sparse_moe() -> None:
    config = parse_config(load_yaml(CONFIG))
    result = manifest(config)

    assert result["expected_layer_range"] == [0, 3]
    assert result["layer_types"] == [
        {"layer": 0, "attention": "full", "ffn": "dense"},
        {"layer": 1, "attention": "full", "ffn": "dense"},
        {"layer": 2, "attention": "full", "ffn": "dense"},
        {"layer": 3, "attention": "sparse", "ffn": "MoE"},
    ]


def test_local_model_config_matches_registered_minimax_defaults() -> None:
    text_config = load_text_config()
    defaults = MiniMaxM3TextConfig()

    assert text_config["hidden_size"] == defaults.hidden_size == 6144
    assert text_config["num_local_experts"] == defaults.num_local_experts == 128
    assert text_config["num_experts_per_tok"] == defaults.num_experts_per_tok == 4
    assert text_config["moe_layer_freq"] == defaults.moe_layer_freq
    assert (
        text_config["sparse_attention_config"]["sparse_attention_freq"]
        == defaults.sparse_attention_config["sparse_attention_freq"]
    )


def test_production_args_keep_real_model_path_and_cut_only_layer_count() -> None:
    config = parse_config(load_yaml(CONFIG))
    kwargs = engine_kwargs(config)

    assert kwargs["hf_overrides"] == {"text_config": {"num_hidden_layers": 4}}
    assert kwargs["language_model_only"] is True
    assert kwargs["load_format"] == "dummy"
    assert kwargs["max_num_seqs"] == 4
    assert kwargs["max_num_batched_tokens"] == 4 * 4096
    assert kwargs["max_model_len"] == 4098
    assert "profiler_config" not in kwargs


def test_torch_profile_is_opt_in(tmp_path: Path) -> None:
    config = parse_config(load_yaml(CONFIG))
    profiled = replace(
        config,
        profile="torch",
        profile_output_dir=str(tmp_path / "traces"),
    )
    kwargs = engine_kwargs(profiled)

    assert kwargs["enable_layerwise_nvtx_tracing"] is True
    assert kwargs["profiler_config"].torch_profiler_with_stack is True
    assert kwargs["profiler_config"].torch_profiler_record_shapes is True
