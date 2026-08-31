# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.utils.nvtx_pytorch_hooks import PytLayerProfilerHooks
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp_size),
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3


def test_layerwise_profiler_hooks_register_once_after_v2_warmup(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    runner = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=model_runner_module.CUDAGraphMode.NONE
        ),
        layerwise_profiler_hooks_registered=False,
        model=model,
        observability_config=SimpleNamespace(enable_layerwise_nvtx_tracing=True),
        vllm_config=SimpleNamespace(profiler_config=SimpleNamespace(profiler="torch")),
    )
    registered: list[tuple[torch.nn.Module, str]] = []

    def register(_self, module, prefix):
        registered.append((module, prefix))
        return 1

    monkeypatch.setattr(PytLayerProfilerHooks, "register_hooks", register)

    GPUModelRunner.register_layerwise_profiler_hooks(runner)
    GPUModelRunner.register_layerwise_profiler_hooks(runner)

    assert registered == [(model, "Sequential")]
    assert runner.layerwise_profiler_hooks_registered is True


def test_layerwise_torch_profiler_scopes_wrap_only_decoder_layers():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
            )
            self.head = torch.nn.Linear(2, 2)

        def forward(self, inputs):
            for layer in self.layers:
                inputs = layer(inputs)
            return self.head(inputs)

    model = Model()
    hook_count = PytLayerProfilerHooks().register_hooks(model, "Model")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profiler:
        model(torch.ones(1, 2))

    event_names = {event.key for event in profiler.key_averages()}
    assert hook_count == 2
    assert "Model.layers.0" in event_names
    assert "Model.layers.1" in event_names
    assert "Model.head" not in event_names
